"""Replaceable experimental illumination response, never an authentication rule."""

from .results import SignalResult


def illumination_signal(stats):
    if (not stats.face_regions_available or not stats.ambient_available or not stats.pose_available
            or stats.window_frame_count < 8 or stats.window_duration_ms < 600):
        return SignalResult(diagnostic_information="illumination_observation_insufficient")
    if not stats.motion_context_available or stats.motion_suppressed:
        return SignalResult(0.5, 0.03, True, "illumination_motion_confounded")
    if stats.consistent_response_observed:
        return SignalResult(0.58, 0.20, True, "weak_multiregion_illumination_response")
    if stats.response_mismatch:
        return SignalResult(0.5, 0.05, True, "illumination_response_mismatch_unvalidated")
    return SignalResult(0.5, 0.08, True, "no_diverse_illumination_response")
