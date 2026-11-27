"""Generated adaptive-observation policy scenarios; no authentication is performed."""

import argparse
from dataclasses import asdict
import json
from types import SimpleNamespace

from facepass.observation import AdaptiveObservationPolicy
from facepass.results import SignalResult, TrackingState


def evidence(groups=0, *, conflict=False, ready=False):
    temporal = SimpleNamespace(
        status="observing", window_frame_count=8, window_duration_ms=700,
        positive_group_count=groups, support_overlap=.8 if groups >= 2 else 0,
        cross_group_agreement=.8 if groups >= 2 else 0,
        consistent_behavior_observed=ready, motion_context_available=ready,
        conflict_observed=conflict, motion_suppressed=False, conflict_count=int(conflict),
    )
    fusion = SimpleNamespace(liveness_candidate=False, spoof_candidate=False)
    signal = SignalResult(.56, .18, True, "weak_dependence_aware_temporal_consistency") if ready else SignalResult()
    return temporal, fusion, signal


def run_case(name, elapsed, *, groups=0, conflict=False, ready=False, extensions=0):
    policy = AdaptiveObservationPolicy(minimum_duration_ms=1000, base_limit_ms=2000,
                                       extension_limit_ms=500, max_extensions=1)
    temporal, fusion, signal = evidence(groups, conflict=conflict, ready=ready)
    result = policy.evaluate(
        elapsed_ms=elapsed, frame_count=12, quality_acceptable=True,
        tracking_state=TrackingState.STABLE, quality_frame_count=12,
        stable_frame_count=10, temporal=temporal, fusion=fusion,
        temporal_signal=signal, extension_count=extensions,
        effective_limit_ms=2000 + extensions * 500,
    )
    if name == "early_completion" and result.status != "early_complete":
        raise RuntimeError("early completion policy case failed")
    if name == "bounded_extension" and result.status != "extension_pending":
        raise RuntimeError("bounded extension policy case failed")
    if name in ("hard_timeout", "conflict_timeout") and result.status != "hard_timeout":
        raise RuntimeError("hard timeout policy case failed")
    return {"synthetic_scenario": name, "observation": asdict(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()
    cases = [
        ("collecting", 700, dict()),
        ("early_completion", 1200, dict(groups=2, ready=True)),
        ("bounded_extension", 2000, dict(groups=1)),
        ("hard_timeout", 2500, dict(groups=1, extensions=1)),
        ("conflict_timeout", 2000, dict(groups=2, conflict=True)),
    ]
    for name, elapsed, kwargs in cases:
        record = run_case(name, elapsed, **kwargs)
        if options.json:
            print(json.dumps(record, allow_nan=False))
        else:
            observation = record["observation"]
            print(f"{name}: status={observation['status']} event={observation['event']} "
                  f"limit={observation['effective_limit_ms']}ms")
    if options.json:
        print(json.dumps({"status": "all_generated_observation_cases_passed"}, allow_nan=False))
    else:
        print("all generated observation cases passed; authentication remains disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
