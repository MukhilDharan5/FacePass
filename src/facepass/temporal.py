"""Bounded temporal consistency over explicit detector dependence groups."""

from dataclasses import replace
import math

from .fusion import EvidenceSnapshot, fuse_snapshots
from .results import FusionDiagnostics, SignalResult, TemporalDiagnostics, TrackingState


_LANDMARK_SIGNALS = ("head_motion", "facial_geometry", "blink", "gaze")


def _finite(value, default=0.0):
    return default if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) else float(value)


def _support(score):
    """Normalize a positive signal without treating reliability as probability."""
    if score is None or not getattr(score, "available", False) or _finite(getattr(score, "reliability", 0)) <= 0:
        return 0.0
    confidence = _finite(getattr(score, "confidence", 0.5), 0.5)
    reliability = max(0.0, min(1.0, _finite(getattr(score, "reliability", 0))))
    strength = max(0.0, min(1.0, (confidence - 0.5) * 2.0))
    return strength * (0.5 + 0.5 * reliability)


def _suspicion(score):
    if score is None or not getattr(score, "available", False) or _finite(getattr(score, "reliability", 0)) <= 0:
        return 0.0
    confidence = _finite(getattr(score, "confidence", 0.5), 0.5)
    reliability = max(0.0, min(1.0, _finite(getattr(score, "reliability", 0))))
    return max(0.0, min(1.0, (0.5 - confidence) * 2.0)) * (0.5 + 0.5 * reliability)


class TemporalConsistencyAnalyzer:
    """Keep at most 64 scalar snapshots covering at most two seconds."""

    MAX_FRAMES = 64
    MAX_DURATION = 2.0
    MAX_GAP = 0.35

    def __init__(self):
        self._snapshots = []
        self._last_timestamp = None
        self._last_temporal = TemporalDiagnostics()
        self._last_fusion = FusionDiagnostics()
        self._last_signal = SignalResult()

    @property
    def snapshot_count(self):
        return len(self._snapshots)

    def reset(self):
        self._snapshots.clear()
        self._last_timestamp = None
        self._last_temporal = TemporalDiagnostics()
        self._last_fusion = FusionDiagnostics()
        self._last_signal = SignalResult()

    def _make_snapshot(self, timestamp, motion, blink, gaze, illumination, replay, signals):
        scores = signals or {}
        landmark_values = [_support(scores.get(name)) for name in _LANDMARK_SIGNALS]
        landmark_available = any(getattr(scores.get(name), "available", False) for name in _LANDMARK_SIGNALS)
        photo_score = _support(scores.get("illumination_consistency"))
        replay_score = _suspicion(scores.get("replay_artifact_detection"))
        photo_available = bool(getattr(scores.get("illumination_consistency"), "available", False))
        replay_available = bool(getattr(scores.get("replay_artifact_detection"), "available", False))
        motion_suppressed = bool(getattr(motion, "common_motion_suppressed", False)
                                 or getattr(illumination, "motion_suppressed", False)
                                 or getattr(replay, "motion_suppressed", False)
                                 or getattr(gaze, "head_motion_suppressed", False))
        motion_context = bool(getattr(illumination, "motion_context_available", False)
                              or getattr(replay, "motion_context_available", False)
                              or getattr(motion, "background_available", False))
        consistent = bool(getattr(illumination, "consistent_response_observed", False))
        replay_artifact = bool(getattr(replay, "replay_artifact_observed", False))
        mismatch = bool(getattr(illumination, "response_mismatch", False))
        return EvidenceSnapshot(
            timestamp=float(timestamp),
            landmark_support=max(landmark_values, default=0.0),
            photometric_support=photo_score,
            replay_suspicion=replay_score,
            landmark_available=landmark_available,
            photometric_available=photo_available,
            replay_available=replay_available,
            motion_context_available=motion_context,
            motion_suppressed=motion_suppressed,
            consistent_behavior=consistent,
            replay_artifact=replay_artifact,
            response_mismatch=mismatch,
            supporting_signal_count=sum(value >= 0.08 for value in landmark_values)
                                  + int(photo_score >= 0.08),
        )

    def _append(self, row):
        self._snapshots.append(row)
        while len(self._snapshots) > self.MAX_FRAMES:
            self._snapshots.pop(0)
        while self._snapshots and row.timestamp - self._snapshots[0].timestamp > self.MAX_DURATION:
            self._snapshots.pop(0)
        self._last_timestamp = row.timestamp

    def update(self, timestamp, tracking, motion, blink, gaze, illumination, replay, signals, *, stable):
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid temporal timestamp")
        if not stable or getattr(tracking, "state", None) != TrackingState.STABLE:
            self.reset()
            return self.result()
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("non-increasing temporal timestamp")
            if timestamp - self._last_timestamp > self.MAX_GAP:
                self.reset()
        row = self._make_snapshot(timestamp, motion, blink, gaze, illumination, replay, signals)
        self._append(row)
        return self._evaluate()

    def refresh(self, timestamp, motion, blink, gaze, illumination, replay, signals):
        """Refresh the most recent row after detector polling/expiry."""
        if not self._snapshots:
            return self.result()
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid temporal timestamp")
        if timestamp - self._snapshots[-1].timestamp > self.MAX_GAP:
            self.reset()
            return self.result()
        row = self._make_snapshot(self._snapshots[-1].timestamp, motion, blink, gaze, illumination, replay, signals)
        self._snapshots[-1] = row
        return self._evaluate()

    def expire(self, timestamp, tracking):
        if (self._snapshots and (getattr(tracking, "state", None) != TrackingState.STABLE
                                 or timestamp - self._snapshots[-1].timestamp > self.MAX_GAP)):
            self.reset()
        return self.result()

    def _evaluate(self):
        fusion = fuse_snapshots(self._snapshots)
        rows = self._snapshots
        if not rows:
            self._last_temporal, self._last_fusion, self._last_signal = TemporalDiagnostics(), FusionDiagnostics(), SignalResult()
            return self.result()
        landmark_rows = [row for row in rows if row.landmark_available]
        photo_rows = [row for row in rows if row.photometric_available]
        landmark_supporting = [row for row in rows if row.landmark_support >= 0.08]
        photo_supporting = [row for row in rows if row.photometric_support >= 0.08]
        overlap_rows = sum(row.landmark_support >= 0.08 and row.photometric_support >= 0.08 for row in rows)
        support_overlap = overlap_rows / max(1, min(len(landmark_supporting), len(photo_supporting)))
        if landmark_supporting and photo_supporting:
            agreement = max(0.0, min(1.0, 1.0 - abs(fusion.landmark_group_score - fusion.photometric_group_score)))
        else:
            agreement = 0.0
        changes = [abs(rows[index].landmark_support - rows[index - 1].landmark_support)
                   + abs(rows[index].photometric_support - rows[index - 1].photometric_support)
                   for index in range(1, len(rows))]
        stability = max(0.0, min(1.0, 1.0 - (sum(changes) / max(1, len(changes)))))
        motion_context = any(row.motion_context_available for row in rows)
        motion_suppressed = any(row.motion_suppressed for row in rows)
        consistent = any(row.consistent_behavior for row in rows)
        duration = fusion.window_duration_ms
        enough = len(rows) >= 8 and duration >= 600
        conflict = fusion.conflict_count
        positive_groups = fusion.positive_group_count
        event = "short_window"
        if enough and conflict:
            event = "conflict"
        elif enough and positive_groups >= 2 and support_overlap >= 0.5 and agreement >= 0.65:
            event = "cross_group_support"
        elif enough:
            event = "partial_group"
        if not enough:
            signal = SignalResult(diagnostic_information="temporal_window_insufficient")
        elif conflict:
            signal = SignalResult(0.5, 0.03, True, "temporal_conflict")
        elif positive_groups >= 2 and support_overlap >= 0.5 and agreement >= 0.65 and not motion_suppressed and motion_context and consistent:
            signal = SignalResult(0.56, 0.18, True, "weak_dependence_aware_temporal_consistency")
        else:
            signal = SignalResult(0.5, 0.06, True, "temporal_consistency_partial_groups")
        temporal = TemporalDiagnostics(
            status="observing",
            window_frame_count=fusion.window_frame_count,
            window_duration_ms=duration,
            landmark_group_available=bool(landmark_rows),
            photometric_group_available=bool(photo_rows),
            replay_group_available=any(row.replay_available for row in rows),
            motion_context_available=motion_context,
            landmark_group_support=fusion.landmark_group_score,
            photometric_group_support=fusion.photometric_group_score,
            replay_suspicion=fusion.replay_group_score,
            positive_group_count=positive_groups,
            supporting_signal_count=fusion.individual_signal_count,
            support_overlap=max(0.0, min(1.0, support_overlap)),
            cross_group_agreement=agreement,
            temporal_stability=stability,
            conflict_count=fusion.conflict_count,
            conflict_observed=bool(conflict),
            dependence_adjusted_support=fusion.adjusted_combined_score,
            missing_group_count=fusion.missing_group_count,
            motion_suppressed=motion_suppressed,
            consistent_behavior_observed=consistent,
            event=event,
        )
        self._last_temporal, self._last_fusion, self._last_signal = temporal, fusion, signal
        return self.result()

    def result(self):
        return self._last_temporal, self._last_fusion, self._last_signal

    def current_result(self, timestamp=None, tracking=None):
        """Return the current scalar summary, expiring a stale track if asked."""
        if timestamp is not None:
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError("invalid temporal timestamp")
            if tracking is not None:
                self.expire(timestamp, tracking)
            elif self._snapshots and timestamp - self._snapshots[-1].timestamp > self.MAX_GAP:
                self.reset()
        return self.result()

    def historical(self):
        if not self._snapshots:
            return TemporalDiagnostics(), FusionDiagnostics()
        temporal, fusion = self._last_temporal, self._last_fusion
        return (replace(temporal, historical=True, status="historical_diagnostics"),
                replace(fusion, historical=True, status="historical_diagnostics"))
