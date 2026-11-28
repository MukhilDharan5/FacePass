"""Bounded bilateral relative-iris motion, not calibrated gaze direction."""

from collections import deque
from dataclasses import replace
import math

import cv2
import numpy as np

from .blink import LEFT_EYE, RIGHT_EYE, OPEN_EAR, eye_measurement
from .gaze_evidence import gaze_signal
from .head_pose import estimate_pose
from .landmarks import FaceObservation
from .results import GazeDiagnostics, SignalResult


# Four ordered iris-rim landmarks per eye, as defined by MediaPipe's connections.
# Use their centroid; predicted iris/pupil landmarks do not attest visibility.
LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)
MIN_EYE_WIDTH_PX = 28.0
MIN_IRIS_DIAMETER_PX = 6.0
MAX_SAMPLE_GAP = 0.22
WINDOW_SECONDS = 2.0
MAX_SAMPLES = 64
MAX_HEAD_SPAN_DEGREES = 1.5
MIN_MOTION_PIXELS = 2.0
MIN_MOTION_RATIO = 0.035
MIN_COHERENCE = 0.70
MIN_BILATERAL_AGREEMENT = 0.70


def iris_measurement(points, shape, eye, iris):
    """Return x/y in eye-width units and geometry scalars, or unavailable.

    x starts at the image-left eye corner; y is signed perpendicular distance
    from the corner line, positive toward image-down before roll normalization.
    """
    measurement = eye_measurement(points, shape, eye)
    if measurement is None:
        return None
    ear, width = measurement
    if width < MIN_EYE_WIDTH_PX or ear < OPEN_EAR:
        return None
    rim = points[list(iris), :2]
    if not np.all((rim >= 0) & (rim <= 1)):
        return None
    rim = rim * [shape[1], shape[0]]
    corners = points[[eye[0], eye[3]], :2] * [shape[1], shape[0]]
    axis = (corners[1] - corners[0]) / width
    normal = np.array([-axis[1], axis[0]])
    center = rim.mean(axis=0)
    relative = rim - corners[0]
    xs, ys = relative @ axis / width, relative @ normal / width
    diameters = np.linalg.norm(rim[[0, 1]] - rim[[2, 3]], axis=1)
    diameter = float(diameters.mean())
    if (min(diameters) < MIN_IRIS_DIAMETER_PX or not 0.15 <= diameter / width <= 0.5
            or max(diameters) / min(diameters) > 2.0
            or np.linalg.norm((rim[0] + rim[2] - rim[1] - rim[3]) / 2) > width * 0.03
            or not np.all((xs > 0.05) & (xs < 0.95)) or np.any(np.abs(ys) > 0.25)):
        return None
    diagonals = rim[[2, 3]] - rim[[0, 1]]
    if abs(float(np.dot(*diagonals))) / float(np.prod(diameters)) > 0.35:
        return None
    # A plausible ring must be convex and its center must lie in the eye aperture.
    # Some rim points can be behind a lid; this is not an occlusion classifier.
    contour = np.ascontiguousarray(points[list(eye), :2] * [shape[1], shape[0]], dtype=np.float32)
    if not cv2.isContourConvex(rim.astype(np.float32)) or cv2.pointPolygonTest(contour, tuple(center), False) < 0:
        return None
    x, y = (center - corners[0]) @ axis / width, (center - corners[0]) @ normal / width
    return float(x), float(y), width, diameter, ear


class GazeAnalyzer:
    """Resettable two-second scalar history on one uninterrupted stable track."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._track_id = self._last_timestamp = None
        self._samples = deque(maxlen=MAX_SAMPLES)
        self._previous_geometry = self._previous_positions = None
        self._diagnostics = GazeDiagnostics()
        self._signal = SignalResult(diagnostic_information="gaze_not_observed")

    def _missing(self, event, track_id=None, timestamp=None, left=None, right=None):
        self.reset()
        self._track_id, self._last_timestamp = track_id, timestamp
        self._diagnostics = GazeDiagnostics(status="observing", event=event, **self._eye_fields(left, right))
        self._signal = SignalResult(diagnostic_information=event)
        return self._diagnostics, self._signal

    @staticmethod
    def _eye_fields(left, right):
        values = {}
        for side, measurement in (("left", left), ("right", right)):
            values[side + "_eye_available"] = measurement is not None
            for name, index in (("horizontal_ratio", 0), ("vertical_ratio", 1),
                                ("eye_width_px", 2), ("iris_diameter_px", 3)):
                values[side + "_" + name] = measurement[index] if measurement else None
        return values

    def _summarize(self, now):
        while self._samples and now - self._samples[0][0] > WINDOW_SECONDS:
            self._samples.popleft()
        count = len(self._samples)
        if not count:
            return self._missing("gaze_window_expired")
        samples = np.asarray(self._samples)
        positions = samples[:, 1:5]
        times = samples[:, 0] - samples[0, 0]
        span = np.percentile(positions.reshape(count, 2, 2).mean(axis=1), [5, 95], axis=0)
        head_span = float(np.max(np.ptp(samples[:, 5:8], axis=0)))
        displacement, coherence, agreement = 0.0, 0.0, 0.0
        if count >= 6:
            deltas = (np.median(positions[-3:], axis=0) - np.median(positions[:3], axis=0)).reshape(2, 2)
            lengths = np.linalg.norm(deltas, axis=1)
            displacement = float(min(lengths))
            if min(lengths) > 1e-8:
                agreement = float(max(0.0, np.dot(*deltas) / np.prod(lengths)) * min(lengths) / max(lengths))
            centered = positions - positions.mean(axis=0)
            variance = float(np.sum(centered ** 2))
            if variance > 1e-10:
                design = np.column_stack((times, np.ones(count)))
                fitted = design @ np.linalg.lstsq(design, positions, rcond=None)[0]
                coherence = float(np.clip(1 - np.sum((positions - fitted) ** 2) / variance, 0, 1))
        minimum_width = min(self._diagnostics.left_eye_width_px, self._diagnostics.right_eye_width_px)
        enough_motion = displacement >= max(MIN_MOTION_RATIO, MIN_MOTION_PIXELS / minimum_width)
        suppressed = head_span > MAX_HEAD_SPAN_DEGREES
        independent = bool(count >= 6 and times[-1] >= 0.45 and not suppressed and enough_motion
                           and coherence >= MIN_COHERENCE and agreement >= MIN_BILATERAL_AGREEMENT)
        self._diagnostics = replace(
            self._diagnostics, window_frame_count=count, window_duration_ms=int(round(times[-1] * 1000)),
            horizontal_span=float(span[1, 0] - span[0, 0]), vertical_span=float(span[1, 1] - span[0, 1]),
            robust_displacement=displacement, motion_coherence=coherence, bilateral_agreement=agreement,
            head_rotation_span_degrees=head_span, head_motion_suppressed=suppressed,
            independent_motion_observed=independent,
            event="relative_iris_motion" if independent else "head_motion_confounded" if suppressed else "observing_iris",
        )
        self._signal = gaze_signal(self._diagnostics)
        return self._diagnostics, self._signal

    def current_result(self, now):
        try:
            return self._current_result(now)
        except Exception:
            self.reset()
            raise

    def _current_result(self, now):
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            self.reset()
            raise ValueError("invalid gaze clock")
        if self._last_timestamp is not None and now - self._last_timestamp > MAX_SAMPLE_GAP:
            return self._missing("gaze_sampling_gap")
        return self._summarize(now) if self._samples else (self._diagnostics, self._signal)

    def update(self, history, *, stable):
        try:
            return self._update(history, stable=stable)
        except Exception:
            self.reset()
            raise

    def _update(self, history, *, stable):
        if not history or not stable:
            return self._missing("gaze_tracking_not_stable")
        current = history[-1]
        timestamp = current.timestamp
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid gaze timestamp")
        if current.track_id != self._track_id:
            self.reset()
            self._track_id = current.track_id
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("non-monotonic gaze timestamp")
            if timestamp - self._last_timestamp > MAX_SAMPLE_GAP:
                self.reset()
                self._track_id = current.track_id
        FaceObservation((current.landmarks,)).validate()
        left = iris_measurement(current.landmarks, current.frame_shape, LEFT_EYE, LEFT_IRIS)
        right = iris_measurement(current.landmarks, current.frame_shape, RIGHT_EYE, RIGHT_IRIS)
        if left is None or right is None:
            return self._missing("gaze_eye_geometry_unreliable", current.track_id, timestamp, left, right)
        if abs(left[4] - right[4]) > 0.10 or max(left[2], right[2]) / min(left[2], right[2]) > 1.8:
            return self._missing("gaze_eye_bilateral_disagreement", current.track_id, timestamp)
        geometry = np.array([left[2], right[2], left[3], right[3]])
        positions = np.array([*left[:2], *right[:2]])
        if (self._previous_geometry is not None and np.max(np.abs(np.log(geometry / self._previous_geometry))) > 0.2
                or self._previous_positions is not None and np.max(np.abs(positions - self._previous_positions)) > 0.12):
            return self._missing("gaze_geometry_jump", current.track_id, timestamp)
        pose = estimate_pose(current.landmarks, current.frame_shape)
        if pose is None or abs(pose.yaw) > 25 or abs(pose.pitch) > 20:
            return self._missing("gaze_head_pose_unreliable", current.track_id, timestamp, left, right)
        if not all(math.isfinite(v) for v in (pose.pitch, pose.yaw, pose.roll, pose.reprojection_error_px)):
            raise ValueError("non-finite gaze pose")
        self._previous_geometry, self._previous_positions = geometry, positions
        self._last_timestamp = timestamp
        self._samples.append((timestamp, *positions, pose.yaw, pose.pitch, pose.roll))
        self._diagnostics = GazeDiagnostics(status="observing", pose_available=True, **self._eye_fields(left, right))
        return self._summarize(timestamp)
