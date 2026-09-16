"""Fail-closed capture/tracking/motion session, independent from authentication."""

from dataclasses import replace
from enum import Enum
import math
import time
from typing import Callable

from .landmarks import FaceObservation
from .quality import assess_quality, validate_frame
from .results import Decision, LivenessResult, MotionDiagnostics, QualityResult, SIGNAL_NAMES, SignalResult, TrackingState
from .motion import MotionAnalyzer
from .tracking import FaceTracker, TrackingPolicy


class SessionState(str, Enum):
    NEW = "NEW"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class LivenessSession:
    """Timestamps are capture times in time.monotonic() seconds.

    One session per attempt. Backend calls are synchronous: callers must supervise
    this engine in a cancellable worker, as the supplied harness does.
    """

    def __init__(self, backend_factory: Callable, *, max_duration=5.0, clock=time.monotonic,
                 tracking_policy=TrackingPolicy()):
        if isinstance(max_duration, bool) or not math.isfinite(max_duration) or not 0 < max_duration <= 30:
            raise ValueError("max_duration must be in (0, 30] seconds")
        self._factory = backend_factory
        self._clock = clock
        self._max_duration = max_duration
        self._backend = None
        self._started = None
        self._last_timestamp = None
        self._observation = None
        self._seen_face = False
        self._tracker = FaceTracker(tracking_policy)
        self._motion = MotionAnalyzer()
        self.state = SessionState.NEW
        self._result = LivenessResult()

    @property
    def observation(self) -> FaceObservation | None:
        """Transient geometry for in-memory detectors/preview, never logging."""
        self.current_result()
        return self._observation

    @property
    def history(self):
        """Bounded, immutable landmark samples from one uninterrupted quality track.

        This is private biometric input for later detectors, not logging/IPC data.
        """
        self.current_result()
        return self._tracker.history

    def _clear_signals(self):
        self._motion.reset()
        self._result = replace(self._result, motion=MotionDiagnostics(),
                               signals={name: SignalResult() for name in SIGNAL_NAMES})

    def start(self):
        if self.state != SessionState.NEW:
            return self.fail("invalid_session_state")
        try:
            self._backend = self._factory()
            self._started = self._clock()
            self.state = SessionState.RUNNING
            self._result = replace(self._result, reason="observing")
        except Exception:
            self.fail("model_initialization_failed")
        return self.current_result()

    def _elapsed(self):
        return 0 if self._started is None else max(0, int((self._clock() - self._started) * 1000))

    def _release(self, tracking_state=TrackingState.CLOSED, tracking_reason="session_closed",
                 preserve_diagnostics=False):
        snapshot = self._result
        self._observation = None
        tracking = self._tracker.reset(tracking_reason, state=tracking_state, terminal=True)
        self._result = replace(self._result, tracking=tracking)
        self._clear_signals()
        if preserve_diagnostics and snapshot.motion.window_frame_count:
            # Historical scalar diagnostics are not reusable live evidence.
            signals = {name: replace(score, available=False, reliability=0.0,
                                     diagnostic_information="historical_" + score.diagnostic_information)
                       if name in ("head_motion", "facial_geometry") else score
                       for name, score in snapshot.signals.items()}
            self._result = replace(self._result, signals=signals,
                                   motion=replace(snapshot.motion, historical=True, status="historical_diagnostics"))
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.close()
            except Exception:
                self.state = SessionState.FAILED
                self._clear_signals()
                self._result = replace(self._result, decision=Decision.UNCERTAIN,
                                       confidence=0.0, reason="model_cleanup_failed")

    def fail(self, reason):
        self.state = SessionState.FAILED
        self._result = replace(self._result, decision=Decision.UNCERTAIN, confidence=0.0,
                               quality=QualityResult(), reason=reason, terminal=True,
                               observation_duration_ms=self._elapsed(), face_count=0, landmark_count=0)
        self._release(TrackingState.REJECTED, reason)
        return self.current_result()

    def cancel(self):
        if self.state == SessionState.RUNNING or self.state == SessionState.NEW:
            self.fail("cancelled")
            self.state = SessionState.CANCELLED
        return self.current_result()

    def process_frame(self, frame, timestamp):
        if self.state != SessionState.RUNNING:
            return self.fail("invalid_session_state")
        if self._elapsed() >= self._max_duration * 1000:
            return self.finish("observation_timeout")
        self._result = replace(self._result, frame_count=self._result.frame_count + 1)
        self._observation = None
        try:
            validate_frame(frame)
            now = self._clock()
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp) or timestamp < self._started
                    or not -0.1 <= now - timestamp <= 0.5
                    or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
                return self.fail("invalid_timestamp")
            if self._last_timestamp is not None and timestamp - self._last_timestamp > 1.0:
                return self.fail("frame_gap")
            self._last_timestamp = timestamp
            observation = self._backend.detect(frame, timestamp)
            if not isinstance(observation, FaceObservation):
                return self.fail("malformed_model_output")
            observation.validate()
            if self._elapsed() >= self._max_duration * 1000:
                return self.finish("observation_timeout")
            count = len(observation.faces)
            if count > 1:
                self._result = replace(self._result, face_count=count,
                                       flags=("liveness_not_implemented", "multiple_faces"))
                result = self.fail("multiple_faces")
                self._result = replace(result, face_count=count)
                return self.current_result()
            flags = ["liveness_not_implemented", "occlusion_not_measured"]
            if not count:
                if self._seen_face:
                    flags.append("tracking_lost")
                quality = QualityResult(reasons=("no_face",))
                tracking = self._tracker.reset("no_face", state=TrackingState.LOST if self._seen_face
                                               else TrackingState.SEARCHING)
                self._clear_signals()
            else:
                self._seen_face = True
                quality = assess_quality(frame, observation.faces[0])
                self._observation = observation
                if quality.acceptable:
                    tracking = self._tracker.update(frame, observation.faces[0], timestamp)
                    if tracking.state == TrackingState.REJECTED:
                        self._result = replace(self._result, tracking=tracking,
                                               flags=tuple(flags + ["tracking_discontinuity"]))
                        result = self.fail(tracking.event)
                        self._result = replace(result, face_count=count)
                        return self.current_result()
                    if tracking.state == TrackingState.ACQUIRING:
                        flags.append("tracking_acquiring")
                    if tracking.history_reset:
                        flags.append("tracking_reset")
                        self._clear_signals()
                    stats, head_signal, geometry_signal = self._motion.update(
                        frame, self._tracker.history, stable=tracking.state == TrackingState.STABLE,
                    )
                    signals = dict(self._result.signals)
                    signals.update(head_motion=head_signal, facial_geometry=geometry_signal)
                    self._result = replace(self._result, motion=stats, signals=signals)
                    if stats.common_motion_suppressed:
                        flags.append("motion_shared_with_environment")
                else:
                    tracking = self._tracker.reset("insufficient_quality")
                    self._clear_signals()
                    flags.append("tracking_lost")
                if "underexposure" in quality.reasons or "overexposure" in quality.reasons:
                    flags.append("poor_lighting")
            if tracking.history_reset:
                flags.append("history_reset")
            self._result = replace(
                self._result,
                decision=Decision.UNCERTAIN if quality.acceptable else Decision.INSUFFICIENT_QUALITY,
                confidence=0.0, quality=quality, tracking=tracking, flags=tuple(flags), face_count=count,
                landmark_count=478 if count else 0,
                reason=("tracking_acquiring" if tracking.state == TrackingState.ACQUIRING
                        else "liveness_not_implemented") if quality.acceptable else "insufficient_quality",
                observation_duration_ms=self._elapsed(),
            )
        except Exception:
            return self.fail("frame_or_model_error")
        return self.current_result()

    def current_result(self):
        if self.state == SessionState.RUNNING:
            if self._elapsed() >= self._max_duration * 1000:
                return self.finish("observation_timeout")
            self._result = replace(self._result, observation_duration_ms=self._elapsed())
            tracking = self._tracker.expire(self._clock())
            if tracking != self._result.tracking:
                self._observation = None
                self._clear_signals()
                old_flags = tuple(flag for flag in self._result.flags
                                  if flag not in ("tracking_acquiring", "tracking_reset"))
                flags = tuple(dict.fromkeys((*old_flags, "tracking_lost", "history_reset")))
                self._result = replace(self._result, decision=Decision.UNCERTAIN, confidence=0.0,
                                       tracking=tracking, quality=QualityResult(reasons=("stale_capture",)),
                                       face_count=0, landmark_count=0, flags=flags, reason="tracking_stale")
        # Do not give callers mutable ownership of the internal signal mapping.
        return replace(self._result, signals=dict(self._result.signals))

    def finish(self, reason="observation_finished"):
        if self.state == SessionState.NEW:
            return self.fail("not_started")
        if self.state == SessionState.RUNNING:
            self.state = SessionState.FINISHED
            self._result = replace(self._result, reason=reason, terminal=True,
                                   observation_duration_ms=self._elapsed())
            # A new capture may finish inference after timeout before it has
            # updated tracking/motion. It must not refresh the older summary.
            history = self._tracker.history
            fresh = bool(history) and 0 <= self._clock() - history[-1].timestamp <= 0.35
            self._release(preserve_diagnostics=fresh and self._result.quality.acceptable
                          and self._result.tracking.state == TrackingState.STABLE)
        return self.current_result()
