"""Conservative short-term continuity, not recognition or presentation-attack detection."""

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from .landmarks import FaceObservation
from .quality import validate_frame
from .results import TrackingResult, TrackingState


# Avoid eyelids, irises and mouth so natural blinks/gaze/expressions do not by
# themselves break continuity. Raw history still retains all 478 landmarks.
ANCHORS = (10, 152, 234, 454, 33, 263, 168, 6, 1, 98, 327,
           127, 356, 93, 323, 172, 397, 132, 361, 4, 102, 331)


@dataclass(frozen=True)
class TrackingPolicy:
    stable_frames: int = 3
    stable_seconds: float = 0.15
    max_gap_seconds: float = 0.35
    history_seconds: float = 2.0
    history_frames: int = 64
    max_shape_residual: float = 0.06
    max_anchor_shape_residual: float = 0.10
    max_appearance_change: float = 0.40

    def __post_init__(self):
        if (isinstance(self.stable_frames, bool) or not isinstance(self.stable_frames, int)
                or not 3 <= self.stable_frames <= 30
                or isinstance(self.history_frames, bool) or not isinstance(self.history_frames, int)
                or not self.stable_frames <= self.history_frames <= 128):
            raise ValueError("invalid tracking frame limits")
        for value in (self.stable_seconds, self.max_gap_seconds, self.history_seconds,
                      self.max_shape_residual, self.max_anchor_shape_residual, self.max_appearance_change):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid tracking policy")
        if not self.stable_seconds <= self.history_seconds <= 5 or self.max_gap_seconds > 1:
            raise ValueError("invalid tracking durations")
        if self.max_appearance_change > 1 or self.max_anchor_shape_residual > 1 or self.max_shape_residual > 1:
            raise ValueError("invalid matching thresholds")


@dataclass(frozen=True)
class TrackedSample:
    track_id: int
    timestamp: float
    landmarks: np.ndarray  # Immutable owned geometry, never public diagnostics.
    frame_shape: tuple[int, int]


def _owned_points(points):
    # Bytes-backed arrays cannot be made writeable by an API consumer. Avoid
    # aliasing a backend which reuses a mutable result buffer between frames.
    return np.frombuffer(points.astype(np.float64).tobytes(), dtype=np.float64).reshape(478, 3)


def _geometry(points, shape):
    xy = points[:468, :2] * [shape[1], shape[0]]
    bounds = np.concatenate((xy.min(axis=0), xy.max(axis=0)))
    size = bounds[2:] - bounds[:2]
    if min(size) < 4:
        raise ValueError("degenerate face geometry")
    return (bounds[:2] + bounds[2:]) / 2, math.sqrt(float(np.prod(size)))


def _shape_distance(previous, current, shape):
    factor = [shape[1], shape[0]]
    source, target = previous[list(ANCHORS), :2] * factor, current[list(ANCHORS), :2] * factor
    source, target = source - source.mean(axis=0), target - target.mean(axis=0)
    source_norm, target_norm = float(np.linalg.norm(source)), float(np.linalg.norm(target))
    if min(source_norm, target_norm) < 1 or np.linalg.matrix_rank(source) < 2 or np.linalg.matrix_rank(target) < 2:
        raise ValueError("degenerate tracking anchors")
    # Similarity alignment removes translation, isotropic scale and roll. It does
    # not infer depth or claim that a face rather than a photograph moved.
    source, target = source / source_norm, target / target_norm
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(2)
    correction[1, 1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = u @ correction @ vt
    residual = np.linalg.norm(source @ rotation - target, axis=1)
    # Robust to a few bad anchors, but not to a systematic shape change.
    return float(np.quantile(residual, 0.8))


def _appearance(frame, points):
    height, width = frame.shape[:2]
    source = (points[[33, 263, 1], :2] * [width, height]).astype(np.float32)
    if abs(float(np.linalg.det(np.column_stack((source[1] - source[0], source[2] - source[0]))))) < 25:
        raise ValueError("degenerate eye/nose alignment")
    target = np.array([[12, 12], [36, 12], [24, 30]], np.float32)
    transform = cv2.getAffineTransform(source, target)
    # Sample only a small skin region within the face; no full previous frame is
    # retained. Coordinates outside the image cannot be supplied by padded pixels.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    patch = cv2.warpAffine(gray, transform, (48, 48), flags=cv2.INTER_LINEAR)
    face_mask = np.zeros(gray.shape, np.uint8)
    hull = cv2.convexHull((points[:468, :2] * [width, height]).astype(np.int32))
    cv2.fillConvexPoly(face_mask, hull, 1)
    coverage = cv2.warpAffine(face_mask, transform, (48, 48), flags=cv2.INTER_NEAREST)
    if not coverage[20:36, 8:40].all():
        return None
    patch = cv2.GaussianBlur(patch, (5, 5), 1)[20:36, 8:40].astype(np.float64).ravel()
    patch -= patch.mean()
    deviation = float(patch.std())
    if deviation < 3:
        return None  # Missing texture is not a spoof or a tracking failure.
    return patch / np.linalg.norm(patch)


def _appearance_distance(source, target):
    if source is None or target is None:
        return None
    return float((1 - np.clip(np.dot(source, target), -1, 1)) / 2)


class FaceTracker:
    def __init__(self, policy=TrackingPolicy()):
        self.policy = policy
        self._history = deque(maxlen=policy.history_frames)
        self._next_id = 0
        self._first_timestamp = None
        self._anchor = self._anchor_appearance = self._previous_appearance = None
        self.result = TrackingResult()

    @property
    def history(self) -> tuple[TrackedSample, ...]:
        return tuple(self._history)

    def reset(self, reason, *, state=TrackingState.LOST, terminal=False):
        if self.result.state in (TrackingState.REJECTED, TrackingState.CLOSED) and not terminal:
            raise ValueError("terminal tracker cannot be reopened")
        active = bool(self._history)
        self._history.clear()
        self._first_timestamp = None
        self._anchor = self._anchor_appearance = self._previous_appearance = None
        # Keep scalar mismatch diagnostics for a rejected track, never its vectors.
        metrics = {} if terminal else dict(center_displacement=None, scale_change=None,
                                          shape_residual=None, appearance_change=None)
        self.result = replace(
            self.result, state=state, track_id=None, continuous_frames=0,
            history_frame_count=0, track_duration_ms=0, continuity_score=0.0,
            appearance_available=False, reset_count=self.result.reset_count + int(active),
            history_reset=(active or self.result.history_reset) if terminal else active,
            event=reason, reset_reason=reason, **metrics,
        )
        return self.result

    def expire(self, now):
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            self.reset("invalid_tracking_clock", state=TrackingState.REJECTED, terminal=True)
            raise ValueError("invalid tracking clock")
        if self._history and now - self._history[-1].timestamp > self.policy.max_gap_seconds:
            return self.reset("tracking_stale")
        return self.result

    def update(self, frame, points, timestamp):
        try:
            return self._update(frame, points, timestamp)
        except Exception:
            self.reset("tracking_error", state=TrackingState.REJECTED, terminal=True)
            raise

    def _update(self, frame, points, timestamp):
        # Also enforce the boundary for callers using the tracker directly.
        validate_frame(frame)
        FaceObservation((points,)).validate()
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)):
            raise ValueError("invalid tracking timestamp")
        if self.result.state in (TrackingState.REJECTED, TrackingState.CLOSED):
            raise ValueError("terminal tracker cannot be reused")
        shape = frame.shape[:2]
        center, scale = _geometry(points, shape)
        _shape_distance(points, points, shape)  # Reject degenerate anchors even on frame one.
        appearance = _appearance(frame, points)
        if not self._history:
            self._next_id += 1
            self._first_timestamp = timestamp
            self._anchor = _owned_points(points)
            self._anchor_appearance = appearance
            self.result = replace(self.result, state=TrackingState.ACQUIRING, track_id=self._next_id,
                                  continuous_frames=0, history_reset=False, event="track_started")
        else:
            previous = self._history[-1]
            gap = timestamp - previous.timestamp
            if gap <= 0:
                raise ValueError("non-monotonic tracking timestamp")
            if shape != previous.frame_shape:
                return self.reset("frame_shape_changed", state=TrackingState.REJECTED, terminal=True)
            if gap > self.policy.max_gap_seconds:
                self.reset("frame_gap")
                restarted = self.update(frame, points, timestamp)
                self.result = replace(restarted, history_reset=True, event="track_restarted")
                return self.result
            old_center, old_scale = _geometry(previous.landmarks, shape)
            displacement = float(np.linalg.norm(center - old_center) / min(scale, old_scale))
            scale_change = abs(math.log(scale / old_scale))
            residual = _shape_distance(previous.landmarks, points, shape)
            anchor_residual = _shape_distance(self._anchor, points, shape)
            distances = (_appearance_distance(self._previous_appearance, appearance),
                         _appearance_distance(self._anchor_appearance, appearance))
            available_distances = [distance for distance in distances if distance is not None]
            appearance_change = max(available_distances) if available_distances else None
            displacement_limit, scale_limit = 0.08 + 1.5 * gap, 0.12 + 1.5 * gap
            self.result = replace(self.result, center_displacement=displacement, scale_change=scale_change,
                                  shape_residual=max(residual, anchor_residual), appearance_change=appearance_change,
                                  history_reset=False)
            reason = None
            if displacement > displacement_limit:
                reason = "position_discontinuity"
            elif scale_change > scale_limit:
                reason = "scale_discontinuity"
            elif residual > self.policy.max_shape_residual or anchor_residual > self.policy.max_anchor_shape_residual:
                reason = "shape_discontinuity"
            elif appearance_change is not None and appearance_change > self.policy.max_appearance_change:
                reason = "appearance_discontinuity"
            if reason:
                return self.reset(reason, state=TrackingState.REJECTED, terminal=True)
            margins = (displacement / displacement_limit, scale_change / scale_limit,
                       residual / self.policy.max_shape_residual,
                       anchor_residual / self.policy.max_anchor_shape_residual)
            score = max(0, 1 - max(margins))
            if appearance_change is not None:
                score = min(score, max(0, 1 - appearance_change / self.policy.max_appearance_change))
            else:
                score = min(score, 0.5)
            self.result = replace(self.result, continuity_score=score)
        sample = TrackedSample(self.result.track_id, timestamp, _owned_points(points), shape)
        self._history.append(sample)
        while self._history and timestamp - self._history[0].timestamp > self.policy.history_seconds:
            self._history.popleft()
        self._previous_appearance = appearance
        span = timestamp - self._first_timestamp
        duration = max(0, int(round(span * 1000)))
        frames = self.result.continuous_frames + 1
        stable = frames >= self.policy.stable_frames and span >= self.policy.stable_seconds
        old_state = self.result.state
        self.result = replace(
            self.result, state=TrackingState.STABLE if stable else TrackingState.ACQUIRING,
            continuous_frames=frames, history_frame_count=len(self._history), track_duration_ms=duration,
            appearance_available=appearance is not None,
            event="track_stabilized" if stable and old_state != TrackingState.STABLE else "track_continues",
        )
        if frames == 1:
            self.result = replace(self.result, event="track_started", continuity_score=0.0)
        return self.result
