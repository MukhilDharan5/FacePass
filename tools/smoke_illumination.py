"""Generated temporal lighting fields, not physical faces or attack evaluation."""

import argparse
from dataclasses import asdict
import json

from facepass.illumination import IlluminationAnalyzer
from facepass.results import Decision, MotionDiagnostics
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from synthetic_illumination import illumination_frame
from synthetic_motion import gaze_mesh, SyntheticBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Show scalar illumination diagnostics and signal results")
    options = parser.parse_args()
    cases = ("static_side_lighting", "uniform_exposure", "common_white_balance", "spatial_response",
             "spatial_with_common_gains", "background_only", "opposed_response", "planar_region_response",
             "color_response_mismatch", "tiny_lighting_change", "missing_scene_context", "head_rotation",
             "camera_motion", "sampling_gap", "identical_replay_pattern")
    for case in cases:
        analyzer = IlluminationAnalyzer()
        history = []
        timestamp = 100.0
        for i in range(12):
            amplitude, light = i * .035, {}
            motion = dict(pose_available=True, pitch_degrees=0, yaw_degrees=0, roll_degrees=0,
                          background_available=True, common_motion_px=0)
            if case == "static_side_lighting":
                amplitude, light = 0, {"face_amplitude": .3}
            elif case == "uniform_exposure":
                amplitude, light = 0, {"global_log_gain": i * .025}
            elif case == "common_white_balance":
                amplitude, light = 0, {"channel_log_gain": (i * .02, -i * .01, i * .005)}
            elif case == "spatial_with_common_gains":
                light = dict(global_log_gain=i * .01, channel_log_gain=(i * .01, -i * .005, 0))
            elif case == "background_only":
                light = {"face_amplitude": 0}
            elif case == "opposed_response":
                light = {"face_amplitude": -amplitude}
            elif case == "planar_region_response":
                light = {"nonlinear": False}
            elif case == "color_response_mismatch":
                light = {"face_color": (.1, .5, 1.4)}
            elif case == "tiny_lighting_change":
                amplitude = i * .001
            elif case == "missing_scene_context":
                motion.update(background_available=False, common_motion_px=None)
            elif case == "head_rotation":
                motion.update(yaw_degrees=i * .3)
            elif case == "camera_motion":
                motion.update(common_motion_px=i * .3)
            timestamp += .4 if case == "sampling_gap" and i == 7 else .1
            history.append(TrackedSample(1, timestamp, gaze_mesh(), (480, 640)))
            stats, score = analyzer.update(illumination_frame(amplitude, **light), tuple(history),
                                           stable=True, motion=MotionDiagnostics(**motion))
        positive = case in ("spatial_response", "spatial_with_common_gains", "identical_replay_pattern")
        if (score.confidence > .5) != positive:
            raise RuntimeError("unexpected generated lighting response")
        if options.json:
            print(json.dumps({"synthetic_scenario": case, "illumination": asdict(stats), "signal": asdict(score)}, allow_nan=False))
        else:
            print(f"{case}: confidence={score.confidence:.2f} reliability={score.reliability:.2f} "
                  f"available={score.available} consistent={stats.consistent_response_observed} "
                  f"mismatch={stats.response_mismatch} suppressed={stats.motion_suppressed}")
        analyzer.reset()

    timestamp = 100.0
    backend = SyntheticBackend()
    session = LivenessSession(lambda: backend, clock=lambda: timestamp)
    session.start()
    try:
        for i in range(14):
            timestamp += .1
            backend.points = gaze_mesh()
            result = session.process_frame(illumination_frame(i * .035), timestamp)
            validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
            if result.decision != Decision.UNCERTAIN or result.confidence != 0:
                raise RuntimeError("unexpected authentication result")
        if result.signals["illumination_consistency"].confidence != .58:
            raise RuntimeError("missing generated regional response")
        result = session.finish()
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        if not result.illumination.historical or result.signals["illumination_consistency"].available or session.history:
            raise RuntimeError("unexpected retained illumination evidence")
        if options.json:
            print(json.dumps({"synthetic_session_final": result.to_dict()}, allow_nan=False))
        else:
            print(f"session final: {result.decision.value} confidence={result.confidence:.1f} "
                  f"illumination_historical={result.illumination.historical} retained_history={len(session.history)}")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
