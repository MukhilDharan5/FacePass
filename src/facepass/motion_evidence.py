"""Replaceable weak evidence producers. Neither one decides authentication."""

from .results import SignalResult


def head_motion_evidence(stats):
    if stats.window_frame_count < 6 or stats.window_duration_ms < 450:
        return SignalResult(diagnostic_information="motion_observation_too_short")
    if not stats.pose_available or stats.rotation_span_degrees is None:
        return SignalResult(diagnostic_information="pose_unreliable")
    if stats.rotation_span_degrees < 0.05 or stats.temporal_coherence < 0.55:
        return SignalResult(0.5, 0.2, True, "still_or_incoherent_motion")
    if stats.common_motion_suppressed:
        return SignalResult(0.5, 0.25, True, "motion_shared_with_environment")
    if not stats.background_available:
        return SignalResult(0.5, 0.15, True, "background_motion_unavailable")
    if (not stats.planar_available or stats.nonplanar_fraction is None
            or stats.nonplanar_fraction < 0.10):
        return SignalResult(0.5, 0.25, True, "planar_or_insufficient_perspective")
    # Heuristic confidence/reliability caps, awaiting held-out physical evaluation.
    strength = min(1.0, stats.rotation_span_degrees / 0.3) * stats.temporal_coherence
    return SignalResult(0.5 + 0.18 * strength, 0.4 * stats.background_inlier_fraction,
                        True, "weak_coherent_head_motion")


def geometry_evidence(stats):
    if stats.window_frame_count < 6 or stats.window_duration_ms < 450 or not stats.planar_available:
        return SignalResult(diagnostic_information="geometry_observation_insufficient")
    if (stats.landmark_motion_px is None or stats.landmark_motion_px < 0.15
            or stats.temporal_coherence < 0.55 or stats.common_motion_suppressed):
        return SignalResult(0.5, 0.15, True, "geometry_indeterminate")
    if not stats.background_available or not stats.pose_available:
        return SignalResult(0.5, 0.15, True, "geometry_confounders_unresolved")
    fraction = stats.nonplanar_fraction
    if fraction is None:
        return SignalResult(diagnostic_information="planar_fit_unreliable")
    if fraction < 0.10:
        return SignalResult(0.5, 0.2, True, "motion_explained_by_plane")
    # Nonplanar model-landmark motion can also be tracking noise or replay video.
    strength = min(1.0, fraction / 0.4) * stats.temporal_coherence
    return SignalResult(0.5 + 0.18 * strength, 0.35 * stats.background_inlier_fraction,
                        True, "weak_nonplanar_landmark_motion")
