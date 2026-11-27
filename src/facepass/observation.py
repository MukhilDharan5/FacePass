"""Fail-closed adaptive observation policy for milestone 9."""

from dataclasses import replace
import math

from .results import ObservationDiagnostics, SignalResult, TrackingState


class AdaptiveObservationPolicy:
    """Decide whether to collect, extend once, or finish an observation.

    This policy only controls capture duration.  It never returns LIVE/SPOOF and
    it never turns a heuristic score into an authentication result.
    """

    def __init__(self, *, minimum_duration_ms=1500, base_limit_ms=5000,
                 extension_limit_ms=2000, max_extensions=1):
        values = (minimum_duration_ms, base_limit_ms, extension_limit_ms, max_extensions)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("observation policy values must be integers")
        if minimum_duration_ms < 0 or base_limit_ms <= 0 or extension_limit_ms < 0:
            raise ValueError("invalid observation policy duration")
        if minimum_duration_ms > base_limit_ms or max_extensions < 0 or max_extensions > 1:
            raise ValueError("invalid observation policy bounds")
        self.minimum_duration_ms = minimum_duration_ms
        self.base_limit_ms = base_limit_ms
        self.extension_limit_ms = extension_limit_ms
        self.max_extensions = max_extensions

    @staticmethod
    def _value(value, default=0):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return default
        return value

    def _ready(self, elapsed_ms, quality_acceptable, tracking_state, temporal,
               fusion, temporal_signal):
        return (
            elapsed_ms >= self.minimum_duration_ms
            and quality_acceptable
            and tracking_state == TrackingState.STABLE
            and temporal.status == "observing"
            and temporal.window_frame_count >= 8
            and temporal.window_duration_ms >= 600
            and temporal.positive_group_count >= 2
            and temporal.support_overlap >= 0.5
            and temporal.cross_group_agreement >= 0.65
            and temporal.consistent_behavior_observed
            and temporal.motion_context_available
            and not temporal.conflict_observed
            and not temporal.motion_suppressed
            and temporal_signal.available
            and temporal_signal.confidence > 0.5
            and not fusion.liveness_candidate
            and not fusion.spoof_candidate
        )

    def _extension_eligible(self, quality_acceptable, tracking_state, temporal, fusion,
                            temporal_signal, extension_count):
        return (
            extension_count < self.max_extensions
            and quality_acceptable
            and tracking_state == TrackingState.STABLE
            and temporal.status == "observing"
            and temporal.window_frame_count >= 8
            and temporal.window_duration_ms >= 600
            and temporal.positive_group_count >= 1
            and not temporal.conflict_observed
            and not temporal.motion_suppressed
            and not fusion.spoof_candidate
            and (not temporal_signal.available or temporal_signal.confidence <= 0.5)
        )

    def evaluate(self, *, elapsed_ms, frame_count, quality_acceptable,
                 tracking_state, quality_frame_count, stable_frame_count,
                 temporal, fusion, temporal_signal, extension_count,
                 effective_limit_ms=None):
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or elapsed_ms < 0:
            raise ValueError("invalid observation elapsed time")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("invalid observation frame count")
        if isinstance(quality_frame_count, bool) or not isinstance(quality_frame_count, int) or not 0 <= quality_frame_count <= frame_count:
            raise ValueError("invalid observation quality count")
        if isinstance(stable_frame_count, bool) or not isinstance(stable_frame_count, int) or stable_frame_count < 0:
            raise ValueError("invalid observation stable count")
        if isinstance(extension_count, bool) or not isinstance(extension_count, int) or not 0 <= extension_count <= self.max_extensions:
            raise ValueError("invalid observation extension count")
        limit = self.base_limit_ms + extension_count * self.extension_limit_ms
        if effective_limit_ms is not None:
            if isinstance(effective_limit_ms, bool) or not isinstance(effective_limit_ms, int):
                raise ValueError("invalid observation effective limit")
            limit = effective_limit_ms
        ready = self._ready(elapsed_ms, quality_acceptable, tracking_state, temporal, fusion, temporal_signal)
        extension = self._extension_eligible(quality_acceptable, tracking_state, temporal,
                                             fusion, temporal_signal, extension_count)
        if ready:
            status, event = "early_complete", "validated_observation_window"
        elif elapsed_ms >= self.base_limit_ms and extension:
            status, event = "extension_pending", "bounded_extension_available"
        elif elapsed_ms >= limit:
            status, event = "hard_timeout", "observation_hard_limit"
        elif tracking_state != TrackingState.STABLE:
            status, event = "observing", "awaiting_stable_track"
        elif not quality_acceptable:
            status, event = "observing", "awaiting_acceptable_quality"
        elif temporal.conflict_observed:
            status, event = "observing", "evidence_conflict"
        else:
            status, event = "observing", "awaiting_diverse_evidence"
        fraction = quality_frame_count / max(1, frame_count)
        return ObservationDiagnostics(
            status=status,
            elapsed_ms=elapsed_ms,
            minimum_duration_ms=self.minimum_duration_ms,
            base_limit_ms=self.base_limit_ms,
            effective_limit_ms=limit,
            extension_limit_ms=self.extension_limit_ms,
            extension_count=extension_count,
            quality_frame_count=quality_frame_count,
            stable_frame_count=stable_frame_count,
            quality_frame_fraction=max(0.0, min(1.0, fraction)),
            temporal_ready=ready,
            positive_group_count=max(0, min(3, temporal.positive_group_count)),
            conflict_count=max(0, min(64, temporal.conflict_count)),
            early_completion=status == "early_complete",
            extension_requested=status == "extension_pending",
            challenge_state="disabled",
            event=event,
        )

    def historical(self, diagnostics):
        return replace(diagnostics, historical=True, status="historical_diagnostics",
                       early_completion=False, extension_requested=False)
