"""Small, explicit dependence groups for temporal evidence.

The fusion summary is an audit aid.  It deliberately does not produce an
authentication decision or a calibrated probability.
"""

from dataclasses import dataclass
import math

from .results import FusionDiagnostics


@dataclass(frozen=True)
class EvidenceSnapshot:
    timestamp: float
    landmark_support: float = 0.0
    photometric_support: float = 0.0
    replay_suspicion: float = 0.0
    landmark_available: bool = False
    photometric_available: bool = False
    replay_available: bool = False
    motion_context_available: bool = False
    motion_suppressed: bool = False
    consistent_behavior: bool = False
    replay_artifact: bool = False
    response_mismatch: bool = False
    supporting_signal_count: int = 0


def _clip(value):
    return min(1.0, max(0.0, float(value)))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _duration_ms(snapshots):
    if len(snapshots) < 2:
        return 0
    return max(0, min(2000, int(round((snapshots[-1].timestamp - snapshots[0].timestamp) * 1000))))


def fuse_snapshots(snapshots) -> FusionDiagnostics:
    """Summarize a bounded sequence of scalar evidence snapshots.

    The four facial landmark signals are represented by one group so that a
    single landmark/model failure cannot be counted four times.  Photometric
    and replay groups remain separate, and replay suspicion is never positive
    liveness support.
    """

    rows = tuple(snapshots)
    if not rows:
        return FusionDiagnostics()
    if any(not isinstance(row.timestamp, (int, float)) or not math.isfinite(row.timestamp)
           for row in rows):
        raise ValueError("invalid temporal timestamp")
    landmark_values = [row.landmark_support for row in rows if row.landmark_available]
    photo_values = [row.photometric_support for row in rows if row.photometric_available]
    replay_values = [row.replay_suspicion for row in rows if row.replay_available]
    landmark = _mean(landmark_values)
    photo = _mean(photo_values)
    replay = _mean(replay_values)
    landmark_available = any(row.landmark_available for row in rows)
    photo_available = any(row.photometric_available for row in rows)
    replay_available = any(row.replay_available for row in rows)
    available = [landmark_available, photo_available, replay_available]
    active = [landmark >= 0.08, photo >= 0.08, replay >= 0.08]
    positive = [landmark >= 0.08, photo >= 0.08]
    suspicious = replay >= 0.08
    conflict = sum(1 for row in rows if row.replay_artifact or row.response_mismatch
                   or (row.replay_suspicion >= 0.20
                       and (row.landmark_support >= 0.08 or row.photometric_support >= 0.08)))
    # A single group is deliberately penalized more heavily than independent
    # cross-group support.  This is a transparent dependence penalty, not a
    # probability calibration.
    positive_scores = [score for score, present in ((landmark, landmark_available), (photo, photo_available)) if present]
    raw = _mean(positive_scores) if positive_scores else 0.0
    if positive[0] and positive[1]:
        penalty = 0.25
    elif any(positive):
        penalty = 0.55
    else:
        penalty = 0.75
    adjusted = _clip(raw * (1.0 - penalty))
    if positive[0] and positive[1]:
        agreement = _clip(1.0 - abs(landmark - photo))
    else:
        agreement = 0.0
    event = "conflict" if conflict else ("cross_group_support" if sum(positive) >= 2 else "partial_group")
    return FusionDiagnostics(
        status="observing",
        window_frame_count=min(64, len(rows)),
        window_duration_ms=_duration_ms(rows),
        landmark_group_score=_clip(landmark),
        photometric_group_score=_clip(photo),
        replay_group_score=_clip(replay),
        active_group_count=sum(active),
        available_group_count=sum(available),
        missing_group_count=3 - sum(available),
        positive_group_count=sum(positive),
        suspicious_group_count=int(suspicious),
        conflict_count=min(64, conflict),
        dependence_penalty=penalty,
        raw_combined_score=_clip(raw),
        adjusted_combined_score=adjusted,
        coverage=sum(available) / 3.0,
        group_agreement=agreement,
        individual_signal_count=max(0, min(256, max(row.supporting_signal_count for row in rows))),
        event=event,
    )
