"""Milestone 9 adaptive observation policy tests."""

import json
from types import SimpleNamespace
import unittest

from facepass.observation import AdaptiveObservationPolicy
from facepass.results import SignalResult, TrackingState, failure_result
from facepass.supervisor import validate_result


def evidence(*, groups=0, conflict=False, ready=False):
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


class ObservationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AdaptiveObservationPolicy(minimum_duration_ms=1000,
                                                base_limit_ms=2000,
                                                extension_limit_ms=500,
                                                max_extensions=1)

    def evaluate(self, elapsed, *, groups=0, conflict=False, ready=False, extension_count=0,
                 tracking=TrackingState.STABLE, quality=True, frame_count=10,
                 quality_count=10, stable_count=8):
        temporal, fusion, signal = evidence(groups=groups, conflict=conflict, ready=ready)
        return self.policy.evaluate(
            elapsed_ms=elapsed, frame_count=frame_count,
            quality_acceptable=quality, tracking_state=tracking,
            quality_frame_count=quality_count, stable_frame_count=stable_count,
            temporal=temporal, fusion=fusion, temporal_signal=signal,
            extension_count=extension_count,
            effective_limit_ms=2000 + extension_count * 500,
        )

    def test_collects_before_minimum(self):
        result = self.evaluate(900, groups=2, ready=True)
        self.assertEqual(result.status, "observing")
        self.assertFalse(result.early_completion)
        self.assertEqual(result.event, "awaiting_diverse_evidence")

    def test_ready_window_finishes_early(self):
        result = self.evaluate(1200, groups=2, ready=True)
        self.assertEqual(result.status, "early_complete")
        self.assertTrue(result.temporal_ready)
        self.assertTrue(result.early_completion)

    def test_partial_evidence_gets_one_bounded_extension(self):
        result = self.evaluate(2000, groups=1)
        self.assertEqual(result.status, "extension_pending")
        self.assertTrue(result.extension_requested)
        extended = self.evaluate(2100, groups=1, extension_count=1)
        self.assertEqual(extended.status, "observing")
        self.assertEqual(extended.effective_limit_ms, 2500)

    def test_extension_limit_hard_times_out(self):
        result = self.evaluate(2500, groups=1, extension_count=1)
        self.assertEqual(result.status, "hard_timeout")
        self.assertEqual(result.event, "observation_hard_limit")

    def test_conflict_never_finishes_early_or_extends(self):
        result = self.evaluate(2000, groups=2, conflict=True, ready=False)
        self.assertEqual(result.status, "hard_timeout")
        self.assertFalse(result.temporal_ready)

    def test_unstable_or_bad_quality_waits(self):
        for kwargs in ({"tracking": TrackingState.ACQUIRING}, {"quality": False}):
            result = self.evaluate(800, **kwargs)
            self.assertEqual(result.event, "awaiting_stable_track" if kwargs.get("tracking") else "awaiting_acceptable_quality")

    def test_session_timeout_emits_policy_diagnostics(self):
        from tests.test_session import Backend, Clock, good_frame
        clock = Clock()
        session = __import__("facepass.session", fromlist=["LivenessSession"]).LivenessSession(
            lambda: Backend(), clock=clock, max_duration=1.0, adaptive=True)
        session.start()
        clock.now += .1
        session.process_frame(good_frame(), clock.now)
        clock.now += 2
        result = session.current_result()
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "observation_timeout")
        self.assertFalse(result.observation.historical)
        validate_result(json.loads(json.dumps(result.to_dict())))

    def test_adaptive_can_be_disabled(self):
        from tests.test_session import Backend, Clock, good_frame
        from facepass.session import LivenessSession
        clock = Clock()
        session = LivenessSession(lambda: Backend(), clock=clock, max_duration=1.0, adaptive=False)
        session.start()
        clock.now += .1
        session.process_frame(good_frame(), clock.now)
        clock.now += 2
        self.assertEqual(session.current_result().reason, "observation_timeout")

    def test_schema_rejects_enabled_challenges_and_bad_states(self):
        value = json.loads(json.dumps(failure_result("example")))
        value["observation"]["challenge_state"] = "ISSUED"
        with self.assertRaises(ValueError):
            validate_result(value)
        value = json.loads(json.dumps(failure_result("example")))
        value["observation"].update(status="early_complete", early_completion=False)
        with self.assertRaises(ValueError):
            validate_result(value)

    def test_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveObservationPolicy(minimum_duration_ms=3000, base_limit_ms=2000)
        with self.assertRaises(ValueError):
            AdaptiveObservationPolicy(minimum_duration_ms=0, base_limit_ms=2000, max_extensions=3)


if __name__ == "__main__":
    unittest.main()
