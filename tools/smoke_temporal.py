"""Generated temporal-consistency/fusion scenarios; not physical PAD validation."""

import argparse
from dataclasses import asdict
import json
from types import SimpleNamespace

from facepass.results import SignalResult, TrackingResult, TrackingState
from facepass.temporal import TemporalConsistencyAnalyzer


TRACK = TrackingResult(state=TrackingState.STABLE, track_id=1, continuous_frames=20,
                       history_frame_count=20, track_duration_ms=1800, continuity_score=.9)


def row_context(*, context=True, consistent=True, replay=False, mismatch=False):
    return (
        SimpleNamespace(background_available=context, common_motion_suppressed=False),
        SimpleNamespace(),
        SimpleNamespace(head_motion_suppressed=False),
        SimpleNamespace(motion_context_available=context, motion_suppressed=False,
                        consistent_response_observed=consistent, response_mismatch=mismatch),
        SimpleNamespace(motion_context_available=context, motion_suppressed=False,
                        replay_artifact_observed=replay),
    )


def scores(landmark=True, photo=True, replay=False):
    return {
        "head_motion": SignalResult(.64, .35 if landmark else 0, landmark, "motion"),
        "facial_geometry": SignalResult(.62, .30 if landmark else 0, landmark, "geometry"),
        "blink": SignalResult(.65, .35 if landmark else 0, landmark, "blink"),
        "gaze": SignalResult(.62, .30 if landmark else 0, landmark, "gaze"),
        "illumination_consistency": SignalResult(.58, .20 if photo else 0, photo, "illumination"),
        "replay_artifact_detection": SignalResult(.34, .25 if replay else 0, replay, "replay"),
    }


def run_case(name, *, landmark=True, photo=True, replay=False, context=True, consistent=True,
             mismatch=False, gap=False):
    analyzer = TemporalConsistencyAnalyzer()
    result = None
    for index in range(10):
        timestamp = index * .1 + (.5 if gap and index >= 6 else 0)
        motion, blink, gaze, illumination, replay_diag = row_context(
            context=context, consistent=consistent, replay=replay, mismatch=mismatch)
        result = analyzer.update(timestamp, TRACK, motion, blink, gaze, illumination, replay_diag,
                                 scores(landmark=landmark, photo=photo, replay=replay), stable=True)
    temporal, fusion, signal = result
    expected_positive = name == "aligned_two_groups"
    if expected_positive and signal.confidence != .56:
        raise RuntimeError(f"{name}: expected weak cross-group support")
    if not expected_positive and signal.confidence > .5:
        raise RuntimeError(f"{name}: unexpected positive temporal evidence")
    return {"synthetic_scenario": name, "temporal": asdict(temporal),
            "fusion": asdict(fusion), "signal": asdict(signal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()
    cases = [
        ("landmark_only", dict(photo=False)),
        ("photometric_only", dict(landmark=False)),
        ("aligned_two_groups", dict()),
        ("replay_conflict", dict(replay=True)),
        ("missing_context", dict(context=False, consistent=False)),
        ("sampling_gap", dict(gap=True)),
    ]
    for name, kwargs in cases:
        record = run_case(name, **kwargs)
        if options.json:
            print(json.dumps(record, allow_nan=False))
        else:
            temporal, fusion, signal = record["temporal"], record["fusion"], record["signal"]
            print(f"{name}: temporal={signal['confidence']:.2f}/{signal['reliability']:.2f} "
                  f"groups={temporal['positive_group_count']} overlap={temporal['support_overlap']:.2f} "
                  f"conflict={temporal['conflict_observed']} adjusted={fusion['adjusted_combined_score']:.2f}")
    if options.json:
        print(json.dumps({"status": "all_generated_temporal_cases_passed"}, allow_nan=False))
    else:
        print("all generated temporal cases passed; global LIVE/SPOOF remains disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
