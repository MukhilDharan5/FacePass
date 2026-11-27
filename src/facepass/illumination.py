"""Multi-region temporal photometry; no calibrated light, depth or PAD claim."""

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from .illumination_evidence import illumination_signal
from .landmarks import FaceObservation
from .quality import validate_frame
from .results import IlluminationDiagnostics, MotionDiagnostics, SignalResult


FACE_REGION_NAMES = ("forehead", "nose", "image_left_cheek", "image_right_cheek",
                     "image_left_socket", "image_right_socket", "chin")
# Fixed image-plane response coordinates, not measured surface normals/depth.
FACE_COORDINATES = np.array([[0, -1], [0, 0], [-1, .3], [1, .3],
                             [-.45, -.4], [.45, -.4], [0, 1]], dtype=float)
AMBIENT_COORDINATES = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
FACE_DESIGN = np.column_stack((FACE_COORDINATES, np.ones(7)))
AMBIENT_DESIGN = np.column_stack((AMBIENT_COORDINATES, np.ones(4)))
WINDOW_SECONDS = 2.0
MAX_SAMPLES = 64
MAX_SAMPLE_GAP = 0.35
MIN_REGION_PIXELS = 24
MAX_REGION_SAMPLES = 512
MAX_CLIPPED_FRACTION = 0.10
MIN_AMBIENT_CHANGE = 0.05
MIN_FACE_CHANGE = 0.03
MIN_REGIONAL_RESIDUAL = 0.012
MIN_RESPONSE_COHERENCE = 0.80
MIN_DIRECTION_AGREEMENT = 0.80
MAX_COLOR_RESPONSE_ERROR = 0.35


def linear_rgb(frame):
    """Approximate inverse sRGB; webcam transfer/tone mapping are uncalibrated."""
    rgb = frame[..., ::-1].astype(np.float64) / 255
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def face_patch_centers(points, shape):
    pixels = points[:, :2] * [shape[1], shape[0]]
    eye_vector = pixels[263] - pixels[33]
    eye_width = float(np.linalg.norm(eye_vector))
    if eye_width < 40:
        return None
    normal = np.array([-eye_vector[1], eye_vector[0]]) / eye_width
    centers = np.array([
        (pixels[10] + pixels[168]) / 2, (pixels[1] + pixels[4]) / 2,
        pixels[[93, 102, 132]].mean(axis=0), pixels[[323, 331, 361]].mean(axis=0),
        (pixels[33] + pixels[133]) / 2 + normal * eye_width * .14,
        (pixels[362] + pixels[263]) / 2 + normal * eye_width * .14,
        pixels[152] * .7 + pixels[4] * .3,
    ])
    return centers, eye_width


def region_masks(points, shape):
    """Ephemeral masks at ≤320×240. Ambient masks exclude a dilated face hull."""
    patch = face_patch_centers(points, shape)
    if patch is None:
        return None
    centers, eye_width = patch
    height, width = shape[:2]
    scale = min(1.0, 320 / width, 240 / height)
    size = (int(round(width * scale)), int(round(height * scale)))
    hull = np.zeros((size[1], size[0]), np.uint8)
    xy = points[:468, :2] * [width * scale, height * scale]
    cv2.fillConvexPoly(hull, cv2.convexHull(xy.astype(np.int32)), 255)
    inner = cv2.erode(hull, np.ones((5, 5), np.uint8))
    outside = cv2.dilate(hull, np.ones((25, 25), np.uint8)) == 0
    faces = []
    for index, center in enumerate(centers):
        radius = max(3, int(round(eye_width * scale * (.045 if index in (1, 4, 5) else .065))))
        mask = np.zeros_like(hull)
        cx, cy = np.rint(center * scale).astype(int)
        if cx - radius < 0 or cy - radius < 0 or cx + radius >= size[0] or cy + radius >= size[1]:
            return None
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        if np.count_nonzero((mask != 0) & (inner == 0)) > 0.1 * np.count_nonzero(mask):
            return None
        faces.append((mask != 0) & (inner != 0))
    if any(np.count_nonzero(a & b) > 0.1 * min(a.sum(), b.sum())
           for index, a in enumerate(faces) for b in faces[index + 1:]):
        return None
    ambient = []
    for y0, y1 in ((.05, .45), (.55, .95)):
        for x0, x1 in ((.05, .45), (.55, .95)):
            mask = np.zeros_like(outside)
            mask[int(y0 * size[1]):int(y1 * size[1]), int(x0 * size[0]):int(x1 * size[0])] = True
            ambient.append(mask & outside)
    if min(int(mask.sum()) for mask in faces) < MIN_REGION_PIXELS or min(int(mask.sum()) for mask in ambient) < 128:
        return None
    return size, tuple(faces), tuple(ambient)


@dataclass(frozen=True)
class RegionPhotometry:
    face_rgb: np.ndarray  # Private medians only, no pixels or masks retained.
    ambient_rgb: np.ndarray
    minimum_pixels: int
    clipped_fraction: float


def extract_photometry(frame, points):
    validate_frame(frame)
    FaceObservation((points,)).validate()
    masks = region_masks(points, frame.shape)
    if masks is None:
        return None
    size, faces, ambient = masks
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    medians, clipped = [], 0.0
    for mask in (*faces, *ambient):
        encoded = small[mask]
        fraction = float(np.mean(np.any((encoded <= 5) | (encoded >= 250), axis=1)))
        clipped = max(clipped, fraction)
        stride = max(1, int(math.ceil(len(encoded) / MAX_REGION_SAMPLES)))
        pixels = linear_rgb(encoded[::stride])
        value = np.median(pixels, axis=0)
        if fraction > MAX_CLIPPED_FRACTION or np.any(value < 0.005):
            return None
        medians.append(value)
    return RegionPhotometry(np.array(medians[:7]), np.array(medians[7:]),
                            min(int(mask.sum()) for mask in (*faces, *ambient)), clipped)


class IlluminationAnalyzer:
    """Local bounded scalar-region history; never retains an image or ROI mask."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._track_id = self._last_timestamp = None
        self._samples = deque(maxlen=MAX_SAMPLES)
        self._diagnostics = IlluminationDiagnostics()
        self._signal = SignalResult(diagnostic_information="illumination_not_observed")

    def _missing(self, event, track_id=None, timestamp=None):
        self.reset()
        self._track_id, self._last_timestamp = track_id, timestamp
        self._diagnostics = IlluminationDiagnostics(status="observing", event=event)
        self._signal = SignalResult(diagnostic_information=event)
        return self._diagnostics, self._signal

    def current_result(self, now):
        try:
            if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("invalid illumination clock")
            if self._last_timestamp is not None and now - self._last_timestamp > MAX_SAMPLE_GAP:
                return self._missing("illumination_sampling_gap")
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
            return self._missing("illumination_tracking_not_stable")
        current = history[-1]
        timestamp = current.timestamp
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid illumination timestamp")
        if tuple(frame.shape[:2]) != current.frame_shape or not isinstance(motion, MotionDiagnostics):
            raise ValueError("invalid illumination context")
        if current.track_id != self._track_id:
            self.reset()
            self._track_id = current.track_id
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("non-monotonic illumination timestamp")
            if timestamp - self._last_timestamp > MAX_SAMPLE_GAP:
                self.reset()
                self._track_id = current.track_id
        FaceObservation((current.landmarks,)).validate()
        photometry = extract_photometry(frame, current.landmarks)
        if photometry is None:
            return self._missing("illumination_regions_unreliable", current.track_id, timestamp)
        angles = (motion.pitch_degrees, motion.yaw_degrees, motion.roll_degrees)
        if not motion.pose_available or any(v is None for v in angles):
            return self._missing("illumination_pose_unreliable", current.track_id, timestamp)
        if not all(math.isfinite(v) for v in angles):
            raise ValueError("non-finite illumination pose")
        if abs(angles[0]) > 20 or abs(angles[1]) > 25:
            return self._missing("illumination_pose_unreliable", current.track_id, timestamp)
        face, ambient = np.log(photometry.face_rgb), np.log(photometry.ambient_rgb)
        if self._samples and np.max(np.abs(np.median(ambient - self._samples[-1][2], axis=0))) > 0.35:
            return self._missing("illumination_radiometric_jump", current.track_id, timestamp)
        center = current.landmarks[[33, 263, 1], :2].mean(axis=0) * [frame.shape[1], frame.shape[0]]
        self._samples.append((timestamp, face, ambient, angles, center,
                              motion.background_available, motion.common_motion_px))
        self._last_timestamp = timestamp
        weights = np.array([.2126, .7152, .0722])
        face_luma, ambient_luma = photometry.face_rgb @ weights, photometry.ambient_rgb @ weights
        balance = photometry.ambient_rgb.mean(axis=0)
        balance /= balance.sum()
        logs = np.log(face_luma)
        self._diagnostics = IlluminationDiagnostics(
            status="observing", face_regions_available=True, ambient_available=True, pose_available=True,
            face_region_count=7, ambient_region_count=4, minimum_region_pixels=photometry.minimum_pixels,
            clipped_fraction=photometry.clipped_fraction, ambient_luminance=float(ambient_luma.mean()),
            ambient_red_balance=float(balance[0]), ambient_green_balance=float(balance[1]), ambient_blue_balance=float(balance[2]),
            cheek_log_contrast=float(logs[3] - logs[2]), forehead_chin_log_contrast=float(logs[0] - logs[6]),
            nose_cheek_log_contrast=float(logs[1] - logs[[2, 3]].mean()), socket_log_contrast=float(logs[5] - logs[4]),
        )
        return self._summarize(timestamp)

    def _summarize(self, now):
        while self._samples and now - self._samples[0][0] > WINDOW_SECONDS:
            self._samples.popleft()
        if not self._samples:
            return self._missing("illumination_window_expired")
        samples = list(self._samples)
        count = len(samples)
        faces, ambient = np.array([s[1] for s in samples]), np.array([s[2] for s in samples])
        gains = np.median(ambient - ambient[0], axis=1)
        face_response = faces - faces[0] - gains[:, None, :]
        ambient_response = ambient - ambient[0] - gains[:, None, :]
        face_coeff = np.linalg.lstsq(FACE_DESIGN, face_response.transpose(1, 0, 2).reshape(7, -1), rcond=None)[0]
        face_coeff = face_coeff.reshape(3, count, 3).transpose(1, 0, 2)
        ambient_coeff = np.linalg.lstsq(AMBIENT_DESIGN, ambient_response.transpose(1, 0, 2).reshape(4, -1), rcond=None)[0]
        ambient_coeff = ambient_coeff.reshape(3, count, 3).transpose(1, 0, 2)
        fg, ag = face_coeff[:, :2].mean(axis=2), ambient_coeff[:, :2].mean(axis=2)
        df = np.median(fg[-3:], axis=0) - np.median(fg[:3], axis=0) if count >= 6 else np.zeros(2)
        da = np.median(ag[-3:], axis=0) - np.median(ag[:3], axis=0) if count >= 6 else np.zeros(2)
        fs, ars = float(np.linalg.norm(df)), float(np.linalg.norm(da))
        direction = float(np.clip(np.dot(df, da) / (fs * ars), 0, 1)) if min(fs, ars) > 1e-8 else 0.0
        centered_face, centered_ambient = fg - fg.mean(axis=0), ag - ag.mean(axis=0)
        norm = float(np.linalg.norm(centered_face) * np.linalg.norm(centered_ambient))
        coherence = float(np.clip(np.sum(centered_face * centered_ambient) / norm, 0, 1)) if norm > 1e-10 else 0.0
        delta = np.median(face_response[-3:], axis=0) - np.median(face_response[:3], axis=0) if count >= 6 else np.zeros((7, 3))
        fit = FACE_DESIGN @ np.linalg.lstsq(FACE_DESIGN, delta, rcond=None)[0]
        residual = float(np.sqrt(np.mean((delta - fit) ** 2)))
        changing = int(np.count_nonzero(np.sqrt(np.mean(delta ** 2, axis=1)) >= MIN_FACE_CHANGE))
        cf = np.median(face_coeff[-3:, :2], axis=0) - np.median(face_coeff[:3, :2], axis=0)
        ca = np.median(ambient_coeff[-3:, :2], axis=0) - np.median(ambient_coeff[:3, :2], axis=0)
        total = float(np.sum(cf ** 2))
        error = None
        if total > 1e-10 and np.sum(ca ** 2) > 1e-10:
            scale = max(0.0, float(np.sum(cf * ca) / np.sum(ca ** 2)))
            error = float(np.clip(np.sqrt(np.sum((cf - scale * ca) ** 2) / total), 0, 1))
        head_span = float(np.max(np.ptp(np.array([s[3] for s in samples]), axis=0)))
        centers = np.array([s[4] for s in samples])
        face_motion = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
        motion_context = all(s[5] and s[6] is not None for s in samples)
        background_motion = max(float(s[6]) for s in samples) if motion_context else None
        suppressed = head_span > 1.5 or face_motion > 2 or background_motion is not None and background_motion > 2
        duration = int(round((samples[-1][0] - samples[0][0]) * 1000))
        enough = count >= 8 and duration >= 600
        consistent = bool(enough and motion_context and not suppressed and ars >= MIN_AMBIENT_CHANGE
                          and fs >= MIN_FACE_CHANGE and residual >= MIN_REGIONAL_RESIDUAL and changing >= 4
                          and coherence >= MIN_RESPONSE_COHERENCE and direction >= MIN_DIRECTION_AGREEMENT
                          and error is not None and error <= MAX_COLOR_RESPONSE_ERROR)
        mismatch = bool(enough and motion_context and not suppressed and ars >= MIN_AMBIENT_CHANGE
                        and (fs < MIN_FACE_CHANGE or direction < MIN_DIRECTION_AGREEMENT or coherence < MIN_RESPONSE_COHERENCE
                             or error is not None and error > MAX_COLOR_RESPONSE_ERROR))
        self._diagnostics = replace(
            self._diagnostics, window_frame_count=count, window_duration_ms=duration,
            global_gain_ev=float(gains[-1].mean() / math.log(2)), common_color_shift=float(np.ptp(gains[-1])),
            ambient_spatial_change=ars, face_spatial_change=fs, regional_response_residual=residual,
            changing_face_regions=changing, response_coherence=coherence, direction_agreement=direction,
            color_response_error=error,
            ambient_response_direction_degrees=float(np.rad2deg(np.arctan2(da[1], da[0]))) if ars >= MIN_AMBIENT_CHANGE else None,
            face_response_direction_degrees=float(np.rad2deg(np.arctan2(df[1], df[0]))) if fs >= MIN_FACE_CHANGE else None,
            head_rotation_span_degrees=head_span, face_motion_px=face_motion, background_motion_px=background_motion,
            motion_context_available=bool(motion_context), motion_suppressed=bool(suppressed),
            consistent_response_observed=consistent, response_mismatch=mismatch,
            event="consistent_spatial_response" if consistent else "response_mismatch" if mismatch else "observing_illumination",
        )
        self._signal = illumination_signal(self._diagnostics)
        return self._diagnostics, self._signal
