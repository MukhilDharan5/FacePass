"""Replaceable, weak relative-iris evidence; no identity or LIVE decision."""

from .results import SignalResult


def gaze_signal(stats):
    if (not stats.left_eye_available or not stats.right_eye_available or not stats.pose_available
            or stats.window_frame_count < 6 or stats.window_duration_ms < 450):
        return SignalResult(diagnostic_information="gaze_observation_insufficient")
    if stats.head_motion_suppressed:
        return SignalResult(0.5, 0.05, True, "gaze_head_motion_confounded")
    if stats.independent_motion_observed:
        return SignalResult(0.62, 0.30, True, "weak_bilateral_relative_iris_motion")
    return SignalResult(0.5, 0.10, True, "no_reliable_relative_iris_motion")
