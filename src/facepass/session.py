"""Fail-closed capture and temporal evidence, independent from authentication."""

from dataclasses import replace
from enum import Enum
import math
import time
from typing import Callable

from .landmarks import FaceObservation
from .quality import assess_quality, prepare_landmark_frame, quality_policy_for, validate_frame
from .results import BlinkDiagnostics, Decision, FusionDiagnostics, GazeDiagnostics, IlluminationDiagnostics, LivenessResult, MotionDiagnostics, ObservationDiagnostics, QualityResult, ReplayDiagnostics, SIGNAL_NAMES, SignalResult, TemporalDiagnostics, TrackingState
from .blink import BlinkAnalyzer
from .gaze import GazeAnalyzer
from .illumination import IlluminationAnalyzer
from .motion import MotionAnalyzer
from .replay import ReplayArtifactAnalyzer
from .temporal import TemporalConsistencyAnalyzer
from .observation import AdaptiveObservationPolicy
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
                 tracking_policy=TrackingPolicy(), adaptive=True, minimum_duration=1.5,
                 extension_duration=2.0, quality_profile="tolerant"):
        if isinstance(max_duration, bool) or not isinstance(max_duration, (int, float)) or not math.isfinite(max_duration) or not 0 < max_duration <= 30:
            raise ValueError("max_duration must be in (0, 30] seconds")
        if isinstance(adaptive, bool) is False:
            raise ValueError("adaptive must be boolean")
        if (isinstance(minimum_duration, bool) or not isinstance(minimum_duration, (int, float))
                or not math.isfinite(minimum_duration) or minimum_duration < 0):
            raise ValueError("minimum_duration must be non-negative")
        minimum_duration = min(minimum_duration, max_duration)
        if (isinstance(extension_duration, bool) or not isinstance(extension_duration, (int, float))
                or not math.isfinite(extension_duration) or not 0 <= extension_duration <= 5):
            raise ValueError("extension_duration must be in [0, 5]")
        self._factory = backend_factory
        self._quality_policy = quality_policy_for(quality_profile)
        self._clock = clock
        self._max_duration = max_duration
        self._adaptive_enabled = adaptive
        self._policy = AdaptiveObservationPolicy(
            minimum_duration_ms=int(round(minimum_duration * 1000)),
            base_limit_ms=int(round(max_duration * 1000)),
            extension_limit_ms=int(round(extension_duration * 1000)),
            max_extensions=1 if adaptive and extension_duration else 0,
        )
        self._effective_limit_ms = self._policy.base_limit_ms
        self._extension_count = 0
        self._quality_frame_count = 0
        self._stable_frame_count = 0
        self._backend = None
        self._started = None
        self._last_timestamp = None
        self._observation = None
        self._seen_face = False
        self._tracker = FaceTracker(tracking_policy)
        self._motion = MotionAnalyzer()
        self._blink = BlinkAnalyzer()
        self._gaze = GazeAnalyzer()
        self._illumination = IlluminationAnalyzer()
        self._replay = ReplayArtifactAnalyzer()
        self._temporal = TemporalConsistencyAnalyzer()
        self.state = SessionState.NEW
        self._result = LivenessResult(quality=QualityResult(profile=quality_profile))

    def _empty_observation(self):
        return ObservationDiagnostics(
            minimum_duration_ms=self._policy.minimum_duration_ms,
            base_limit_ms=self._policy.base_limit_ms,
            effective_limit_ms=self._effective_limit_ms,
            extension_limit_ms=self._policy.extension_limit_ms,
            extension_count=self._extension_count,
        )

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

    def _clear_signals(self, reason="signal_history_cleared"):
        self._motion.reset()
        self._blink.reset()
        self._gaze.reset()
        self._illumination.reset()
        self._replay.reset()
        self._temporal.reset()
        self._stable_frame_count = 0
        self._result = replace(self._result, motion=MotionDiagnostics(), blink=BlinkDiagnostics(), gaze=GazeDiagnostics(),
                               illumination=IlluminationDiagnostics(), replay=ReplayDiagnostics(),
                               temporal=TemporalDiagnostics(), fusion=FusionDiagnostics(),
                               observation=self._empty_observation(),
                               signals={name: SignalResult(diagnostic_information=reason) for name in SIGNAL_NAMES})

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
        historical_temporal, historical_fusion = self._temporal.historical()
        historical_observation = self._policy.historical(snapshot.observation)
        self._observation = None
        tracking = self._tracker.reset(tracking_reason, state=tracking_state, terminal=True)
        self._result = replace(self._result, tracking=tracking)
        self._clear_signals("quality_not_acceptable" if snapshot.decision == Decision.INSUFFICIENT_QUALITY
                            else "session_ended")
        if preserve_diagnostics and (snapshot.motion.window_frame_count or snapshot.blink.observed_frame_count
                                     or snapshot.gaze.window_frame_count or snapshot.illumination.window_frame_count
                                     or snapshot.replay.window_frame_count or snapshot.temporal.window_frame_count
                                     or snapshot.observation.elapsed_ms):
            # Historical scalar diagnostics are not reusable live evidence.
            signals = {name: replace(score, available=False, reliability=0.0,
                                     diagnostic_information="historical_" + score.diagnostic_information)
                                   if name in ("head_motion", "facial_geometry", "blink", "gaze", "illumination_consistency",
                                   "replay_artifact_detection", "temporal_consistency") else score
                       for name, score in snapshot.signals.items()}
            self._result = replace(self._result, signals=signals,
                                   motion=replace(snapshot.motion, historical=True, status="historical_diagnostics")
                                   if snapshot.motion.window_frame_count else MotionDiagnostics(),
                                   blink=replace(snapshot.blink, historical=True, status="historical_diagnostics")
                                   if snapshot.blink.observed_frame_count else BlinkDiagnostics(),
                                   gaze=replace(snapshot.gaze, historical=True, status="historical_diagnostics")
                                   if snapshot.gaze.window_frame_count else GazeDiagnostics(),
                                   illumination=replace(snapshot.illumination, historical=True, status="historical_diagnostics")
                                   if snapshot.illumination.window_frame_count else IlluminationDiagnostics(),
                                   replay=replace(snapshot.replay, historical=True, status="historical_diagnostics")
                                   if snapshot.replay.window_frame_count else ReplayDiagnostics(),
                                   temporal=replace(historical_temporal, historical=True, status="historical_diagnostics")
                                   if snapshot.temporal.window_frame_count else TemporalDiagnostics(),
                                   fusion=replace(historical_fusion, historical=True, status="historical_diagnostics")
                                   if snapshot.fusion.window_frame_count else FusionDiagnostics(),
                                   observation=historical_observation)
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
                               quality=QualityResult(profile=self._quality_policy.profile), reason=reason, terminal=True,
                               observation_duration_ms=self._elapsed(), face_count=0, landmark_count=0)
        self._release(TrackingState.REJECTED, reason)
        return self.current_result()

    def _update_observation_diagnostics(self):
        diagnostics = self._policy.evaluate(
            elapsed_ms=self._elapsed(),
            frame_count=self._result.frame_count,
            quality_acceptable=self._result.quality.acceptable,
            tracking_state=self._result.tracking.state,
            quality_frame_count=self._quality_frame_count,
            stable_frame_count=self._stable_frame_count,
            temporal=self._result.temporal,
            fusion=self._result.fusion,
            temporal_signal=self._result.signals.get("temporal_consistency", SignalResult()),
            extension_count=self._extension_count,
            effective_limit_ms=self._effective_limit_ms,
        )
        self._result = replace(self._result, observation=diagnostics)
        return diagnostics

    def _maybe_extend_observation(self):
        if not self._adaptive_enabled or self._extension_count >= self._policy.max_extensions:
            return False
        now = self._clock()
        history = self._tracker.history
        if (not history
                or not 0 <= now - history[-1].timestamp <= min(self._tracker.policy.max_gap_seconds, self._temporal.MAX_GAP)
                or self._elapsed() >= self._policy.base_limit_ms + self._policy.extension_limit_ms):
            return False
        diagnostics = self._update_observation_diagnostics()
        if not diagnostics.extension_requested:
            return False
        self._extension_count += 1
        self._effective_limit_ms = self._policy.base_limit_ms + self._extension_count * self._policy.extension_limit_ms
        self._update_observation_diagnostics()
        return True

    def _at_observation_limit(self):
        if self._elapsed() < self._effective_limit_ms:
            return False
        if not self._maybe_extend_observation():
            return True
        return self._elapsed() >= self._effective_limit_ms

    def cancel(self):
        if self.state == SessionState.RUNNING or self.state == SessionState.NEW:
            self.fail("cancelled")
            self.state = SessionState.CANCELLED
        return self.current_result()

    def process_frame(self, frame, timestamp):
        if self.state != SessionState.RUNNING:
            return self.fail("invalid_session_state")
        if self._at_observation_limit():
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
            model_frame, enhanced = prepare_landmark_frame(frame, self._quality_policy)
            observation = self._backend.detect(model_frame, timestamp)
            del model_frame
            if not isinstance(observation, FaceObservation):
                return self.fail("malformed_model_output")
            observation.validate()
            if self._at_observation_limit():
                return self.finish("observation_timeout")
            count = len(observation.faces)
            if count > 1:
                self._result = replace(self._result, face_count=count,
                                       flags=("liveness_not_implemented", "multiple_faces"))
                result = self.fail("multiple_faces")
                self._result = replace(result, face_count=count)
                return self.current_result()
            flags = ["liveness_not_implemented", "occlusion_not_measured"]
            if enhanced:
                flags.append("landmark_low_light_enhanced")
            if not count:
                if self._seen_face:
                    flags.append("tracking_lost")
                quality = QualityResult(reasons=("no_face",))
                tracking = self._tracker.reset("no_face", state=TrackingState.LOST if self._seen_face
                                               else TrackingState.SEARCHING)
                self._clear_signals()
            else:
                self._seen_face = True
                quality = assess_quality(frame, observation.faces[0], self._quality_policy)
                self._observation = observation
                if quality.acceptable:
                    self._quality_frame_count += 1
                    tracking = self._tracker.update(frame, observation.faces[0], timestamp)
                    if tracking.state == TrackingState.REJECTED:
                        self._result = replace(self._result, tracking=tracking,
                                               flags=tuple(flags + ["tracking_discontinuity"]))
                        result = self.fail(tracking.event)
                        self._result = replace(result, face_count=count)
                        return self.current_result()
                    if tracking.state == TrackingState.ACQUIRING:
                        flags.append("tracking_acquiring")
                    if tracking.state == TrackingState.STABLE:
                        self._stable_frame_count += 1
                    if tracking.history_reset:
                        flags.append("tracking_reset")
                        self._clear_signals()
                    stats, head_signal, geometry_signal = self._motion.update(
                        frame, self._tracker.history, stable=tracking.state == TrackingState.STABLE,
                    )
                    blink_stats, blink_signal = self._blink.update(
                        self._tracker.history, stable=tracking.state == TrackingState.STABLE,
                    )
                    gaze_stats, gaze_signal = self._gaze.update(
                        self._tracker.history, stable=tracking.state == TrackingState.STABLE,
                    )
                    illumination_stats, illumination_signal = self._illumination.update(
                        frame, self._tracker.history, stable=tracking.state == TrackingState.STABLE, motion=stats,
                    )
                    replay_stats, replay_signal = self._replay.update(
                        frame, self._tracker.history, stable=tracking.state == TrackingState.STABLE, motion=stats,
                    )
                    signals = dict(self._result.signals)
                    signals.update(head_motion=head_signal, facial_geometry=geometry_signal,
                                   blink=blink_signal, gaze=gaze_signal, illumination_consistency=illumination_signal,
                                   replay_artifact_detection=replay_signal)
                    temporal_stats, fusion_stats, temporal_signal = self._temporal.update(
                        timestamp, tracking, stats, blink_stats, gaze_stats, illumination_stats, replay_stats,
                        signals, stable=tracking.state == TrackingState.STABLE,
                    )
                    signals["temporal_consistency"] = temporal_signal
                    self._result = replace(self._result, motion=stats, blink=blink_stats, gaze=gaze_stats,
                                           illumination=illumination_stats, replay=replay_stats,
                                           temporal=temporal_stats, fusion=fusion_stats, signals=signals)
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
            quality = replace(quality, profile=self._quality_policy.profile, landmark_enhanced=enhanced)
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

    def _refresh_temporal_signals(self):
        blink, blink_signal = self._blink.current_result(self._clock())
        gaze, gaze_signal = self._gaze.current_result(self._clock())
        illumination, illumination_signal = self._illumination.current_result(self._clock())
        replay, replay_signal = self._replay.current_result(self._clock())
        signals = dict(self._result.signals)
        signals.update(blink=blink_signal, gaze=gaze_signal, illumination_consistency=illumination_signal,
                       replay_artifact_detection=replay_signal)
        if self._result.tracking.state == TrackingState.STABLE and self._result.quality.acceptable:
            temporal, fusion, temporal_signal = self._temporal.refresh(
                self._last_timestamp if self._last_timestamp is not None else self._clock(),
                self._result.motion, blink, gaze, illumination, replay, signals,
            )
        else:
            temporal, fusion, temporal_signal = self._temporal.result()
            temporal_signal = SignalResult()
        signals["temporal_consistency"] = temporal_signal
        self._result = replace(self._result, blink=blink, gaze=gaze, illumination=illumination,
                               replay=replay, temporal=temporal, fusion=fusion, signals=signals)

    def current_result(self):
        if self.state == SessionState.RUNNING:
            if self._at_observation_limit():
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
            if tracking.state == TrackingState.STABLE and self._result.quality.acceptable:
                try:
                    self._refresh_temporal_signals()
                except Exception:
                    return self.fail("temporal_evidence_error")
            diagnostics = self._update_observation_diagnostics()
            if self._adaptive_enabled and diagnostics.early_completion:
                return self.finish("observation_early_complete")
        # Do not give callers mutable ownership of the internal signal mapping.
        return replace(self._result, signals=dict(self._result.signals))

    def finish(self, reason="observation_finished"):
        if self.state == SessionState.NEW:
            return self.fail("not_started")
        if self.state == SessionState.RUNNING:
            try:
                self._refresh_temporal_signals()
            except Exception:
                return self.fail("temporal_evidence_error")
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
