"""Offline relative-iris patterns; no physical gaze or replay accuracy claim."""

import argparse
from dataclasses import asdict
import json

import numpy as np

from facepass.gaze import GazeAnalyzer
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from synthetic_motion import SyntheticBackend, gaze_mesh, planar_mesh, synthetic_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print scalar diagnostics and signal results")
    options = parser.parse_args()
    cases = ("open_stillness", "relative_horizontal", "relative_vertical", "confirmed_fixation_step",
             "whole_scene_motion", "head_rotation", "unilateral_motion", "single_frame_noise",
             "low_iris_resolution", "partial_eye_failure", "eye_closure", "sampling_gap", "identical_replay_pattern")
    for case in cases:
        analyzer = GazeAnalyzer()
        history = []
        timestamp = 100.0
        for index in range(10):
            x = 0.4 + index / 9 * 0.15 if case in ("relative_horizontal", "head_rotation", "unilateral_motion",
                                                  "sampling_gap", "identical_replay_pattern") else 0.5
            if case == "confirmed_fixation_step":
                x = 0.45 if index < 5 else 0.53
            if case == "single_frame_noise" and index == 5:
                x = 0.6
            geometry = {"horizontal": x}
            if case == "relative_vertical":
                geometry.update(vertical=-0.04 + index / 9 * 0.08, ear=0.4)
            if case == "head_rotation":
                geometry.update(yaw=index * 0.4)
            if case == "unilateral_motion":
                geometry.update(right_horizontal=0.5)
            if case == "low_iris_resolution":
                geometry.update(iris_diameter=4)
            if case == "eye_closure":
                geometry.update(ear=0.10)
            points = gaze_mesh(**geometry)
            if case == "partial_eye_failure":
                points[469, 0] = -0.01
            if case == "whole_scene_motion":
                points = planar_mesh(points, np.array([[1, 0, index], [0, 1, index * 0.5], [0, 0, 1]], dtype=float))
            timestamp += 0.3 if case == "sampling_gap" and index == 5 else 0.1
            history.append(TrackedSample(1, timestamp, points, (480, 640)))
            stats, score = analyzer.update(tuple(history), stable=True)
        positive = case in ("relative_horizontal", "relative_vertical", "confirmed_fixation_step", "identical_replay_pattern")
        if (score.confidence > 0.5) != positive:
            raise RuntimeError("unexpected generated relative-iris evidence")
        if options.json:
            print(json.dumps({"synthetic_scenario": case, "gaze": asdict(stats), "signal": asdict(score)}, allow_nan=False))
        else:
            print(f"{case}: confidence={score.confidence:.2f} reliability={score.reliability:.2f} "
                  f"available={score.available} independent_motion={stats.independent_motion_observed} "
                  f"event={stats.event}")
        analyzer.reset()

    timestamp = 100.0
    backend = SyntheticBackend()
    session = LivenessSession(lambda: backend, clock=lambda: timestamp)
    session.start()
    try:
        frame = synthetic_frame()
        for x in [0.4] * 3 + list(np.linspace(0.4, 0.55, 10)):
            timestamp += 0.1
            backend.points = gaze_mesh(horizontal=x)
            result = session.process_frame(frame, timestamp)
            validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
            if result.decision != Decision.UNCERTAIN or result.confidence != 0:
                raise RuntimeError("unexpected authentication result")
        if result.signals["gaze"].confidence != 0.62:
            raise RuntimeError("missing generated gaze evidence")
        result = session.finish()
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        if not result.gaze.historical or result.signals["gaze"].available or session.history:
            raise RuntimeError("unexpected retained evidence")
        if options.json:
            print(json.dumps({"synthetic_session_final": result.to_dict()}, allow_nan=False))
        else:
            print(f"session final: {result.decision.value} confidence={result.confidence:.1f} "
                  f"gaze_historical={result.gaze.historical} retained_history={len(session.history)}")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
