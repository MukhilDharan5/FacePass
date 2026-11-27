"""Conservative bilateral EAR sequence detector; never an authentication decision."""

from dataclasses import replace
import math

import numpy as np

from .landmarks import FaceObservation
from .results import BlinkDiagnostics, SignalResult


# MediaPipe contour indices, in six-point EAR order. These are image landmarks,
# not an eye-visibility or physical eyelid sensor.
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
# Initial engineering gates only. No enrollment or adaptive threshold lowering.
OPEN_EAR = 0.23
CLOSED_EAR = 0.15
MIN_EYE_WIDTH_PX = 18.0
MAX_SAMPLE_GAP = 0.22
MIN_CONFIRM_SECONDS = 0.06
MAX_SEQUENCE_SECONDS = 0.65
EVIDENCE_SECONDS = 2.0


def eye_measurement(points, shape, indices):
    xy = points[list(indices), :2]
    if not np.all((xy >= 0) & (xy <= 1)):
        return None
    xy = xy * [shape[1], shape[0]]
    axis = xy[3] - xy[0]
    width = float(np.linalg.norm(axis))
    if width < MIN_EYE_WIDTH_PX:
        return None
    axis /= width
    normal = np.array([-axis[1], axis[0]])
    along = (xy[[1, 2, 4, 5]] - xy[0]) @ axis / width
    if not np.all((along > 0.05) & (along < 0.95)):
        return None
    gaps = xy[[5, 4]] - xy[[1, 2]]
    if np.any(np.abs(gaps @ axis) > width * 0.12) or np.any(gaps @ normal < -width * 0.015):
        return None  # Crossed/displaced eyelids are tracking uncertainty.
    lengths = np.linalg.norm(gaps, axis=1) / width
    ear = float(lengths.mean())
    if ear > 0.6 or abs(float(lengths[0] - lengths[1])) > 0.12:
        return None
    return ear, width


class BlinkAnalyzer:
    """Constant-memory state machine on one uninterrupted quality-stable track.

    Thresholds are conservative and uncalibrated. Fast blinks can be missed at
    webcam cadence. Missing/no blink and sustained closure never imply SPOOF.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._track_id = self._last_timestamp = self._first_timestamp = None
        self._open_since = self._closed_since = self._cycle_since = None
        self._first_reopen = self._last_completed = self._last_duration = None
        self._open_frames = self._closed_frames = self._frames = 0
        self._blinks = self._rejections = 0
        self._previous_widths = None
        self._state = "ACQUIRING_OPEN"
        self._diagnostics = BlinkDiagnostics()
        self._signal = SignalResult(diagnostic_information="blink_not_observed")

    def _clear_cycle(self):
        self._open_since = self._closed_since = self._cycle_since = self._first_reopen = None
        self._open_frames = self._closed_frames = 0

    def _reject(self):
        self._rejections += 1
        self._last_completed = self._last_duration = None
        self._clear_cycle()
        self._state = "ACQUIRING_OPEN"

    def _missing(self, event, track_id=None, timestamp=None):
        self.reset()
        self._track_id, self._last_timestamp = track_id, timestamp
        self._diagnostics = BlinkDiagnostics(status="observing", state="ACQUIRING_OPEN", event=event)
        self._signal = SignalResult(diagnostic_information=event)
        return self._diagnostics, self._signal

    def _open_sample(self, timestamp):
        if self._open_since is None:
            self._open_since = timestamp
        self._open_frames += 1
        return self._open_frames >= 2 and timestamp - self._open_since >= MIN_CONFIRM_SECONDS

    def _score(self, now):
        stats = self._diagnostics
        if not stats.eyes_available or stats.observed_frame_count < 3 or stats.observation_duration_ms < 150:
            return SignalResult(diagnostic_information="blink_observation_insufficient")
        if self._state == "OPEN" and self._last_completed is not None and now - self._last_completed <= EVIDENCE_SECONDS:
            return SignalResult(0.65, 0.35, True, "weak_bilateral_blink_sequence")
        reason = "eyes_held_closed" if self._state == "SUSTAINED_CLOSED" else "no_completed_blink"
        return SignalResult(0.5, 0.15, True, reason)

    def current_result(self, now):
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            self.reset()
            raise ValueError("invalid blink clock")
        if self._last_timestamp is not None and now - self._last_timestamp > MAX_SAMPLE_GAP:
            return self._missing("blink_sampling_gap")
        if self._last_completed is not None:
            self._diagnostics = replace(self._diagnostics,
                                        last_blink_age_ms=max(0, int(round((now - self._last_completed) * 1000))))
        self._signal = self._score(now) if self._diagnostics.eyes_available else self._signal
        return self._diagnostics, self._signal

    def update(self, history, *, stable):
        try:
            return self._update(history, stable=stable)
        except Exception:
            self.reset()
            raise

    def _update(self, history, *, stable):
        if not history or not stable:
            return self._missing("blink_tracking_not_stable")
        current = history[-1]
        timestamp = current.timestamp
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid blink timestamp")
        if current.track_id != self._track_id:
            self.reset()
            self._track_id = current.track_id
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("non-monotonic blink timestamp")
            if timestamp - self._last_timestamp > MAX_SAMPLE_GAP:
                self.reset()
                self._track_id = current.track_id
        FaceObservation((current.landmarks,)).validate()
        left = eye_measurement(current.landmarks, current.frame_shape, LEFT_EYE)
        right = eye_measurement(current.landmarks, current.frame_shape, RIGHT_EYE)
        if left is None or right is None:
            return self._missing("eye_geometry_unreliable", current.track_id, timestamp)
        if abs(left[0] - right[0]) > 0.10 or max(left[1], right[1]) / min(left[1], right[1]) > 1.8:
            return self._missing("eye_bilateral_disagreement", current.track_id, timestamp)
        widths = np.array([left[1], right[1]])
        if self._previous_widths is not None and np.max(np.abs(np.log(widths / self._previous_widths))) > 0.2:
            return self._missing("eye_geometry_jump", current.track_id, timestamp)
        self._previous_widths = widths
        self._last_timestamp = timestamp
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._frames += 1
        is_open = min(left[0], right[0]) >= OPEN_EAR
        is_closed = max(left[0], right[0]) <= CLOSED_EAR
        event = "observing_eyes"

        if self._cycle_since is not None and timestamp - self._cycle_since > MAX_SEQUENCE_SECONDS:
            closed_since = self._closed_since
            self._reject()
            event = "blink_sequence_too_long"
            if is_closed:
                self._state = "SUSTAINED_CLOSED"
                self._closed_since = closed_since

        if self._state in ("ACQUIRING_OPEN", "SUSTAINED_CLOSED"):
            if is_open:
                self._closed_since = None
                if self._open_sample(timestamp):
                    self._state = "OPEN"
                    self._clear_cycle()
                    event = "open_baseline_ready"
            else:
                self._open_since, self._open_frames = None, 0
                if is_closed:
                    if self._closed_since is None:
                        self._closed_since = timestamp
                    if timestamp - self._closed_since > MAX_SEQUENCE_SECONDS:
                        self._state = "SUSTAINED_CLOSED"
                        event = "eyes_held_closed"
                else:
                    self._closed_since = None
        elif self._state == "OPEN":
            if not is_open:
                self._clear_cycle()
                self._cycle_since = timestamp
                self._state = "CLOSING"
                event = "closing_started"
                if is_closed:
                    self._closed_since, self._closed_frames = timestamp, 1
        elif self._state in ("CLOSING", "CLOSED"):
            if is_closed:
                if self._closed_since is None:
                    self._closed_since = timestamp
                self._closed_frames += 1
                if self._closed_frames >= 2 and timestamp - self._closed_since >= MIN_CONFIRM_SECONDS:
                    self._state = "CLOSED"
                    event = "closure_confirmed"
            elif is_open:
                if self._state == "CLOSED":
                    self._state = "OPENING"
                    self._first_reopen = timestamp
                    self._open_sample(timestamp)
                    event = "reopening_started"
                else:
                    self._reject()
                    self._open_sample(timestamp)
                    event = "closure_not_confirmed"
            elif self._state == "CLOSED":
                self._state = "OPENING"
                event = "reopening_started"
            else:
                # Closed observations must be consecutive, not accumulated
                # from isolated noise spikes separated by intermediate samples.
                self._closed_since, self._closed_frames = None, 0
        elif self._state == "OPENING":
            if is_closed:
                self._reject()
                event = "reopening_interrupted"
            elif is_open:
                if self._first_reopen is None:
                    self._first_reopen = timestamp
                if self._open_sample(timestamp):
                    self._blinks += 1
                    self._last_completed = timestamp
                    self._last_duration = int(round((self._first_reopen - self._cycle_since) * 1000))
                    self._state = "OPEN"
                    self._clear_cycle()
                    event = "blink_completed"
            else:
                self._open_since, self._open_frames = None, 0
                self._first_reopen = None

        self._diagnostics = BlinkDiagnostics(
            status="observing", state=self._state, eyes_available=True,
            left_ear=left[0], right_ear=right[0], left_eye_width_px=left[1], right_eye_width_px=right[1],
            observed_frame_count=self._frames,
            observation_duration_ms=max(0, int(round((timestamp - self._first_timestamp) * 1000))),
            blink_count=self._blinks, rejected_sequence_count=self._rejections,
            closed_duration_ms=None if self._closed_since is None else int(round((timestamp - self._closed_since) * 1000)),
            last_blink_duration_ms=self._last_duration,
            last_blink_age_ms=None if self._last_completed is None else int(round((timestamp - self._last_completed) * 1000)),
            event=event,
        )
        self._signal = self._score(timestamp)
        return self._diagnostics, self._signal
