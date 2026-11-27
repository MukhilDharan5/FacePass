"""Replaceable weak replay-artifact evidence; absence is never live evidence."""

from .results import SignalResult


def replay_signal(stats):
    if (not stats.regions_available or not stats.background_available
            or stats.window_frame_count < 6 or stats.window_duration_ms < 450):
        return SignalResult(diagnostic_information="replay_observation_insufficient")
    if not stats.motion_context_available or stats.motion_suppressed:
        return SignalResult(0.5, 0.03, True, "replay_texture_motion_confounded")
    if stats.replay_artifact_observed:
        return SignalResult(0.35, 0.25, True, "weak_persistent_multifamily_replay_artifact")
    return SignalResult(0.5, 0.10, True, "no_persistent_replay_artifact_observed")
