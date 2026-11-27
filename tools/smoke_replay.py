"""Generated replay-artifact diagnostics, not physical media or attack accuracy."""

import argparse
from dataclasses import asdict, replace
import json

from facepass.replay import ReplayArtifactAnalyzer
from facepass.results import Decision, MotionDiagnostics
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from synthetic_motion import SyntheticBackend, gaze_mesh
from synthetic_replay import replay_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Show scalar replay diagnostics and signal results")
    options = parser.parse_args()
    cases = ("natural_texture", "display_grid", "print_halftone", "global_sensor_banding",
             "background_grid", "localized_glare", "global_sharpening", "low_texture",
             "missing_motion_context", "camera_motion", "sampling_gap", "transient_pattern",
             "matching_bona_fide_pattern")
    suspicious_cases = {"display_grid", "print_halftone", "matching_bona_fide_pattern"}
    for case in cases:
        analyzer = ReplayArtifactAnalyzer()
        history = []
        timestamp = 100.0
        for index in range(8):
            kind = case if case in ("natural_texture", "display_grid", "print_halftone",
                                    "global_sensor_banding", "background_grid", "localized_glare",
                                    "global_sharpening", "low_texture", "matching_bona_fide_pattern") else "natural_texture"
            if case == "transient_pattern" and index < 2:
                kind = "display_grid"
            motion = MotionDiagnostics(pose_available=True, pitch_degrees=0, yaw_degrees=0, roll_degrees=0,
                                       background_available=True, common_motion_px=0)
            if case == "missing_motion_context":
                motion = replace(motion, background_available=False, common_motion_px=None)
            elif case == "camera_motion":
                motion = replace(motion, common_motion_px=4)
                kind = "display_grid"
            timestamp += .4 if case == "sampling_gap" and index == 5 else .1
            points = gaze_mesh()
            history.append(TrackedSample(1, timestamp, points, (480, 640)))
            stats, score = analyzer.update(replay_frame(kind, points=points), tuple(history),
                                           stable=True, motion=motion)
        if (score.confidence < .5) != (case in suspicious_cases):
            raise RuntimeError("unexpected generated replay response")
        if options.json:
            print(json.dumps({"synthetic_scenario": case, "replay": asdict(stats),
                              "signal": asdict(score)}, allow_nan=False))
        else:
            print(f"{case}: confidence={score.confidence:.2f} reliability={score.reliability:.2f} "
                  f"available={score.available} suspicious={stats.replay_artifact_observed} "
                  f"families={stats.artifact_family_count} affected={stats.affected_face_regions} "
                  f"background={stats.background_artifact_score}")
        analyzer.reset()

    timestamp = 100.0
    backend = SyntheticBackend()
    session = LivenessSession(lambda: backend, clock=lambda: timestamp)
    session.start()
    try:
        for _ in range(10):
            timestamp += .1
            backend.points = gaze_mesh()
            result = session.process_frame(replay_frame("display_grid"), timestamp)
            validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
            if result.decision != Decision.UNCERTAIN or result.confidence != 0:
                raise RuntimeError("unexpected authentication result")
        if result.signals["replay_artifact_detection"].confidence != .35:
            raise RuntimeError("missing generated multifamily artifact")
        result = session.finish()
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        if (not result.replay.historical or result.signals["replay_artifact_detection"].available
                or session.history or session._replay._samples):
            raise RuntimeError("unexpected retained replay evidence")
        if options.json:
            print(json.dumps({"synthetic_session_final": result.to_dict()}, allow_nan=False))
        else:
            print(f"session final: {result.decision.value} confidence={result.confidence:.1f} "
                  f"replay_historical={result.replay.historical} retained_history={len(session.history)}")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
