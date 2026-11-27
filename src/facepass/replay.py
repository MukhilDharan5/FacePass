"""Conservative multi-region print/display artifact diagnostics for RGB frames."""

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from .illumination import face_patch_centers
from .landmarks import FaceObservation
from .quality import validate_frame
from .replay_evidence import replay_signal
from .results import MotionDiagnostics, ReplayDiagnostics, SignalResult


FACE_REGION_INDICES = (0, 1, 2, 3, 6)  # forehead, nose, cheeks, chin
FAMILY_NAMES = ("periodic", "chroma_grid", "banding", "sharpening",
                "resampling", "glare", "flat_edge")
FAMILY_THRESHOLDS = np.array((.45, .48, .52, .58, .52, .55, .58), dtype=float)
WINDOW_SECONDS = 2.0
MAX_SAMPLES = 64
MAX_SAMPLE_GAP = 0.35
MIN_FRAMES = 6
MIN_DURATION_MS = 450
MIN_AFFECTED_REGIONS = 3
MIN_FAMILIES = 2
MIN_PERSISTENCE = 0.75
MAX_BACKGROUND_ARTIFACT = 0.45


@dataclass(frozen=True)
class ArtifactFrame:
    face_scores: np.ndarray  # 5 × 7 private scalar features.
    background_scores: np.ndarray  # 4 × 7 private scalar features.
    minimum_patch_width: int
    texture_contrast: float


def _spectral_concentration(values):
    centered = values - float(values.mean())
    window = np.hanning(values.shape[0])[:, None] * np.hanning(values.shape[1])[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(values.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(values.shape[1]))[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    band = spectrum[(radius >= .08) & (radius <= .45)]
    if band.size < 8 or float(band.sum()) <= 1e-12:
        return 0.0
    top = np.partition(band, -min(4, band.size))[-min(4, band.size):].sum() / band.sum()
    return float(np.clip((top - .04) / .46, 0, 1))


def _line_spectrum(values):
    scores = []
    for profile in (values.mean(axis=0), values.mean(axis=1)):
        profile = (profile - profile.mean()) * np.hanning(len(profile))
        power = np.abs(np.fft.rfft(profile)) ** 2
        frequencies = np.fft.rfftfreq(len(profile))
        selected = power[frequencies >= .06]
        ratio = 0.0 if float(selected.sum()) <= 1e-12 else float(selected.max() / selected.sum())
        scores.append(float(np.clip((ratio - .14) / .66, 0, 1)))
    return max(scores)


def _lag_periodicity(values):
    centered = values - values.mean()
    variance = float(np.mean(centered * centered))
    if variance <= 1e-8:
        return 0.0
    correlations = []
    for lag in range(2, 7):
        correlations.extend((abs(float(np.mean(centered[:, :-lag] * centered[:, lag:]) / variance)),
                             abs(float(np.mean(centered[:-lag, :] * centered[lag:, :]) / variance))))
    return float(np.clip((max(correlations) - .18) / .72, 0, 1))


def patch_features(patch):
    """Seven bounded diagnostics; they are camera/content dependent, not labels."""
    normalized = patch.astype(np.float64) / 255
    rgb = normalized[:, :, ::-1]
    gray = rgb @ np.array((.2126, .7152, .0722))
    contrast = float(gray.std())
    periodic = _spectral_concentration(gray)
    opponent = np.maximum.reduce((np.abs(rgb[:, :, 0] - rgb[:, :, 1]),
                                  np.abs(rgb[:, :, 2] - rgb[:, :, 1]),
                                  np.abs(rgb[:, :, 0] - rgb[:, :, 2])))
    chroma_strength = float(np.clip(opponent.std() / max(contrast, .02), 0, 1))
    chroma_grid = _spectral_concentration(opponent) * chroma_strength
    banding = _line_spectrum(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    sharp_ratio = float(np.mean(np.abs(laplacian)) / max(np.mean(gradient), 1e-5))
    sharpening = float(np.clip((sharp_ratio - .75) / 1.5, 0, 1))
    resampling = _lag_periodicity(gray)
    low_chroma = rgb.max(axis=2) - rgb.min(axis=2) < .035
    glare = float(np.clip(np.mean((gray > .93) & low_chroma) / .15, 0, 1))
    strong = gradient >= max(float(np.quantile(gradient, .75)), .03)
    if np.count_nonzero(strong) < 16:
        flat_edge = 0.0
    else:
        axis_aligned = (np.abs(gx) > 4 * np.abs(gy)) | (np.abs(gy) > 4 * np.abs(gx))
        fraction = float(np.mean(axis_aligned[strong]))
        flat_edge = float(np.clip((fraction - .42) / .48, 0, 1))
    return np.array((periodic, chroma_grid, banding, sharpening, resampling, glare, flat_edge)), contrast


def artifact_regions(frame, points):
    """Return temporary equal-size face/background patches; no patch is retained."""
    validate_frame(frame)
    FaceObservation((points,)).validate()
    located = face_patch_centers(points, frame.shape)
    if located is None:
        return None
    centers, eye_width = located
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / width, 480 / height)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    scaled_points = points[:468, :2] * [size[0], size[1]]
    hull = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(scaled_points.astype(np.int32)), 255)
    inner = cv2.erode(hull, np.ones((5, 5), np.uint8)) != 0
    outside = cv2.dilate(hull, np.ones((31, 31), np.uint8)) == 0
    radius = max(8, int(round(eye_width * scale * .11)))

    def crop(center, allowed):
        x, y = np.rint(center).astype(int)
        if x - radius < 0 or y - radius < 0 or x + radius >= size[0] or y + radius >= size[1]:
            return None
        support = allowed[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if support.shape != (2 * radius + 1, 2 * radius + 1) or np.mean(support) < .90:
            return None
        patch = small[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)

    face = [crop(centers[index] * scale, inner) for index in FACE_REGION_INDICES]
    background_centers = [(size[0] * x, size[1] * y) for y in (.13, .87) for x in (.13, .87)]
    background = [crop(center, outside) for center in background_centers]
    if any(patch is None for patch in (*face, *background)):
        return None
    return tuple(face), tuple(background), 2 * radius + 1


def extract_artifacts(frame, points):
    regions = artifact_regions(frame, points)
    if regions is None:
        return None
    face, background, width = regions
    face_values = [patch_features(patch) for patch in face]
    background_values = [patch_features(patch) for patch in background]
    face_scores = np.array([value[0] for value in face_values])
    background_scores = np.array([value[0] for value in background_values])
    contrasts = np.array([value[1] for value in face_values])
    if np.count_nonzero(contrasts >= .012) < 3:
        return None
    return ArtifactFrame(face_scores, background_scores, width, float(np.median(contrasts)))


class ReplayArtifactAnalyzer:
    """Keeps bounded scalar artifact summaries; never retains frames or patches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._track_id = self._last_timestamp = None
        self._samples = deque(maxlen=MAX_SAMPLES)
        self._diagnostics = ReplayDiagnostics()
        self._signal = SignalResult(diagnostic_information="replay_not_observed")

    def _missing(self, event, track_id=None, timestamp=None):
        self.reset()
        self._track_id, self._last_timestamp = track_id, timestamp
        self._diagnostics = ReplayDiagnostics(status="observing", event=event)
        self._signal = SignalResult(diagnostic_information=event)
        return self._diagnostics, self._signal

    def current_result(self, now):
        try:
            if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("invalid replay clock")
            if self._last_timestamp is not None and now - self._last_timestamp > MAX_SAMPLE_GAP:
                return self._missing("replay_sampling_gap")
            return self._summarize(now) if self._samples else (self._diagnostics, self._signal)
        except Exception:
            self.reset()
            raise

    def update(self, frame, history, *, stable, motion):
        try:
            return self._update(frame, history, stable=stable, motion=motion)
        except Exception:
            self.reset()
            raise

    def _update(self, frame, history, *, stable, motion):
        validate_frame(frame)
        if not history or not stable:
            return self._missing("replay_tracking_not_stable")
        current = history[-1]
        timestamp = current.timestamp
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid replay timestamp")
        if tuple(frame.shape[:2]) != current.frame_shape or not isinstance(motion, MotionDiagnostics):
            raise ValueError("invalid replay context")
        if current.track_id != self._track_id:
            self.reset()
            self._track_id = current.track_id
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("non-monotonic replay timestamp")
            if timestamp - self._last_timestamp > MAX_SAMPLE_GAP:
                self.reset()
                self._track_id = current.track_id
        FaceObservation((current.landmarks,)).validate()
        extracted = extract_artifacts(frame, current.landmarks)
        if extracted is None:
            return self._missing("replay_regions_unreliable", current.track_id, timestamp)
        center = current.landmarks[[33, 263, 1], :2].mean(axis=0) * [frame.shape[1], frame.shape[0]]
        self._samples.append((timestamp, extracted.face_scores, extracted.background_scores,
                              extracted.minimum_patch_width, extracted.texture_contrast, center,
                              motion.background_available, motion.common_motion_px))
        self._last_timestamp = timestamp
        self._diagnostics = ReplayDiagnostics(
            status="observing", regions_available=True, background_available=True,
            face_region_count=5, background_region_count=4,
            minimum_patch_width_px=extracted.minimum_patch_width,
            texture_contrast=extracted.texture_contrast,
        )
        return self._summarize(timestamp)

    def _summarize(self, now):
        while self._samples and now - self._samples[0][0] > WINDOW_SECONDS:
            self._samples.popleft()
        if not self._samples:
            return self._missing("replay_window_expired")
        samples = list(self._samples)
        face = np.array([sample[1] for sample in samples])
        background = np.array([sample[2] for sample in samples])
        face_medians = np.median(face, axis=(0, 1))
        background_medians = np.median(background, axis=(0, 1))
        adjusted = np.clip(face_medians - .75 * background_medians, 0, 1)
        per_frame_face = np.median(face, axis=1)
        per_frame_background = np.median(background, axis=1)
        per_frame_adjusted = np.clip(per_frame_face - .75 * per_frame_background, 0, 1)
        family_present = per_frame_adjusted >= FAMILY_THRESHOLDS
        persistence_by_family = family_present.mean(axis=0)
        persistent = persistence_by_family >= MIN_PERSISTENCE
        family_count = int(np.count_nonzero(persistent))
        affected_each = np.count_nonzero(
            np.max(np.clip(face - .75 * np.median(background, axis=1)[:, None, :], 0, 1)
                   - FAMILY_THRESHOLDS, axis=2) >= 0, axis=1)
        affected = int(round(float(np.median(affected_each))))
        persistence = float(np.max(persistence_by_family))
        background_artifact = float(np.max(background_medians))
        temporal_variation = float(np.median(np.std(per_frame_adjusted, axis=0)))
        centers = np.array([sample[5] for sample in samples])
        face_motion = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
        motion_context = all(sample[6] and sample[7] is not None for sample in samples)
        background_motion = max(float(sample[7]) for sample in samples) if motion_context else None
        suppressed = face_motion > 3 or background_motion is not None and background_motion > 3
        duration = int(round((samples[-1][0] - samples[0][0]) * 1000))
        enough = len(samples) >= MIN_FRAMES and duration >= MIN_DURATION_MS
        observed = bool(enough and motion_context and not suppressed and family_count >= MIN_FAMILIES
                        and affected >= MIN_AFFECTED_REGIONS and persistence >= MIN_PERSISTENCE
                        and background_artifact <= MAX_BACKGROUND_ARTIFACT)
        values = {name: float(value) for name, value in zip(FAMILY_NAMES, adjusted)}
        self._diagnostics = replace(
            self._diagnostics, window_frame_count=len(samples), window_duration_ms=duration,
            minimum_patch_width_px=min(sample[3] for sample in samples),
            texture_contrast=float(np.median([sample[4] for sample in samples])),
            periodic_peak_score=values["periodic"], chroma_grid_score=values["chroma_grid"],
            banding_score=values["banding"], sharpening_score=values["sharpening"],
            resampling_score=values["resampling"], glare_score=values["glare"],
            flat_edge_score=values["flat_edge"], background_artifact_score=background_artifact,
            affected_face_regions=affected, artifact_family_count=family_count,
            artifact_persistence=persistence, temporal_score_variation=temporal_variation,
            face_motion_px=face_motion, background_motion_px=background_motion,
            motion_context_available=bool(motion_context), motion_suppressed=bool(suppressed),
            replay_artifact_observed=observed,
            event="persistent_multifamily_artifact" if observed else "observing_replay_artifacts",
        )
        self._signal = replay_signal(self._diagnostics)
        return self._diagnostics, self._signal
