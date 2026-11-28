"""Offline generated blink patterns; no physical blink or replay accuracy claim."""

import argparse
from dataclasses import asdict
import json

from facepass.blink import BlinkAnalyzer
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from synthetic_motion import SyntheticBackend, blink_mesh, synthetic_frame


SEQUENCE = [0.30, 0.30, 0.30, 0.19, 0.10, 0.10, 0.19, 0.30, 0.30]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Show full scalar diagnostics and individual signal values")
    options = parser.parse_args()
    cases = {
        "bilateral_blink": SEQUENCE,
        "open_stillness": [0.30] * 15,
        "sustained_closure": [0.30] * 3 + [0.10] * 12,
        "single_frame_noise": [0.30] * 3 + [0.10] + [0.30] * 3,
        "asymmetric_eye": SEQUENCE,
        "low_eye_resolution": SEQUENCE,
        "blink_crossing_gap": [0.30] * 3 + [0.10, 0.10] + [0.30] * 3,
        "identical_replay_pattern": SEQUENCE,
    }
    for case, sequence in cases.items():
        analyzer = BlinkAnalyzer()
        history, phases = [], []
        timestamp = 100.0
        for index, ear in enumerate(sequence):
            timestamp += 0.3 if case == "blink_crossing_gap" and index == 5 else 0.1
            points = blink_mesh(left_ear=ear, right_ear=0.30 if case == "asymmetric_eye" else ear,
                                eye_width=10 if case == "low_eye_resolution" else 36)
            history.append(TrackedSample(1, timestamp, points, (480, 640)))
            stats, score = analyzer.update(tuple(history), stable=True)
            if not phases or phases[-1] != stats.state:
                phases.append(stats.state)
        expected_blink = case in ("bilateral_blink", "identical_replay_pattern")
        if bool(stats.blink_count) != expected_blink or (score.confidence > 0.5) != expected_blink:
            raise RuntimeError("unexpected generated blink result")
        if options.json:
            print(json.dumps({"synthetic_scenario": case, "blink": asdict(stats),
                              "signal": asdict(score), "phases": phases}, allow_nan=False))
        else:
            print(f"{case}: count={stats.blink_count} confidence={score.confidence:.2f} "
                  f"reliability={score.reliability:.2f} available={score.available} "
                  f"phases={' -> '.join(phases)}")
        analyzer.reset()

    timestamp = 100.0
    backend = SyntheticBackend()
    session = LivenessSession(lambda: backend, clock=lambda: timestamp)
    session.start()
    try:
        frame = synthetic_frame()
        for ear in [0.30] * 3 + SEQUENCE:
            timestamp += 0.1
            backend.points = blink_mesh(left_ear=ear, right_ear=ear)
            result = session.process_frame(frame, timestamp)
            validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
            if result.decision != Decision.UNCERTAIN or result.confidence != 0:
                raise RuntimeError("unexpected authentication result")
        result = session.finish()
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        if result.blink.blink_count != 1 or result.signals["blink"].available or result.confidence != 0:
            raise RuntimeError("unexpected finished blink diagnostics")
        if options.json:
            print(json.dumps({"synthetic_session_final": result.to_dict()}, allow_nan=False))
        else:
            print(f"session final: {result.decision.value} confidence={result.confidence:.1f} "
                  f"blink_historical={result.blink.historical} retained_history={len(session.history)}")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
