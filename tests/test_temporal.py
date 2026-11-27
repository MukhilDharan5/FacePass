"""Milestone 8 temporal consistency and dependence-aware fusion tests."""

import json
from types import SimpleNamespace
import unittest

from facepass.results import SignalResult, TrackingResult, TrackingState
from facepass.supervisor import validate_result
from facepass.temporal import TemporalConsistencyAnalyzer


def diagnostics(*, context=True, consistent=True, replay=False, mismatch=False):
    motion = SimpleNamespace(background_available=context, common_motion_suppressed=False)
    blink = SimpleNamespace()
    gaze = SimpleNamespace(head_motion_suppressed=False)
    illumination = SimpleNamespace(motion_context_available=context,
                                   motion_suppressed=False,
                                   consistent_response_observed=consistent,
                                   response_mismatch=mismatch)
    replay_diag = SimpleNamespace(motion_context_available=context,
                                  motion_suppressed=False,
                                  replay_artifact_observed=replay)
    return motion, blink, gaze, illumination, replay_diag


def scores(*, landmark=True, photo=True, replay=False):
    return {
        "head_motion": SignalResult(.64, .35 if landmark else 0, landmark, "motion"),
        "facial_geometry": SignalResult(.62, .30 if landmark else 0, landmark, "geometry"),
        "blink": SignalResult(.65, .35 if landmark else 0, landmark, "blink"),
        "gaze": SignalResult(.62, .30 if landmark else 0, landmark, "gaze"),
        "illumination_consistency": SignalResult(.58, .20 if photo else 0, photo, "lighting"),
        "replay_artifact_detection": SignalResult(.34, .25 if replay else 0, replay, "replay"),
    }


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.tracking = TrackingResult(state=TrackingState.STABLE, track_id=1,
                                       continuous_frames=20, history_frame_count=20,
                                       track_duration_ms=1800, continuity_score=.9)

    def observe(self, analyzer, count=10, *, landmark=True, photo=True, replay=False,
                context=True, consistent=True, mismatch=False, start=0.0):
        motion, blink, gaze, illumination, replay_diag = diagnostics(
            context=context, consistent=consistent, replay=replay, mismatch=mismatch)
        result = None
        for index in range(count):
            result = analyzer.update(start + index * .1, self.tracking, motion, blink, gaze,
                                     illumination, replay_diag,
                                     scores(landmark=landmark, photo=photo, replay=replay),
                                     stable=True)
        return result

    def test_short_window_is_unavailable(self):
        temporal, fusion, signal = self.observe(TemporalConsistencyAnalyzer(), 7)
        self.assertFalse(signal.available)
        self.assertEqual(temporal.event, "short_window")
        self.assertEqual(fusion.window_frame_count, 7)

    def test_aligned_two_groups_are_weak_temporal_support_only(self):
        temporal, fusion, signal = self.observe(TemporalConsistencyAnalyzer())
        self.assertTrue(signal.available)
        self.assertEqual(signal.confidence, .56)
        self.assertEqual(signal.reliability, .18)
        self.assertEqual(temporal.positive_group_count, 2)
        self.assertGreaterEqual(temporal.support_overlap, .5)
        self.assertFalse(fusion.liveness_candidate)
        self.assertFalse(fusion.spoof_candidate)

    def test_landmark_only_and_photo_only_never_claim_cross_group_support(self):
        for kwargs in ({"landmark": True, "photo": False}, {"landmark": False, "photo": True}):
            temporal, fusion, signal = self.observe(TemporalConsistencyAnalyzer(), **kwargs)
            self.assertLessEqual(signal.confidence, .5)
            self.assertLess(temporal.positive_group_count, 2)
            self.assertEqual(fusion.positive_group_count, 1)

    def test_replay_conflict_cannot_become_positive(self):
        temporal, fusion, signal = self.observe(TemporalConsistencyAnalyzer(), replay=True)
        self.assertTrue(temporal.conflict_observed)
        self.assertEqual(signal.confidence, .5)
        self.assertLessEqual(signal.reliability, .03)
        self.assertFalse(fusion.liveness_candidate)

    def test_mismatch_and_suppressed_context_are_neutral(self):
        temporal, _, signal = self.observe(TemporalConsistencyAnalyzer(), mismatch=True)
        self.assertTrue(temporal.conflict_observed)
        self.assertLessEqual(signal.confidence, .5)
        temporal, _, signal = self.observe(TemporalConsistencyAnalyzer(), context=False, consistent=False)
        self.assertEqual(signal.confidence, .5)
        self.assertFalse(temporal.motion_context_available)

    def test_gap_clears_history_and_nonstable_resets(self):
        analyzer = TemporalConsistencyAnalyzer()
        self.observe(analyzer, 8)
        temporal, fusion, signal = self.observe(analyzer, 1, start=1.4)
        self.assertEqual(temporal.window_frame_count, 1)
        self.assertFalse(signal.available)
        self.assertEqual(fusion.window_frame_count, 1)
        temporal, _, _ = analyzer.update(1.1, TrackingResult(), *diagnostics(), scores(), stable=False)
        self.assertEqual(temporal.window_frame_count, 0)

    def test_window_is_bounded(self):
        analyzer = TemporalConsistencyAnalyzer()
        self.observe(analyzer, 100)
        self.assertLessEqual(analyzer.snapshot_count, 64)
        temporal, _, _ = analyzer.result()
        self.assertLessEqual(temporal.window_frame_count, 64)
        self.assertLessEqual(temporal.window_duration_ms, 2000)

    def test_invalid_and_duplicate_timestamps_fail_closed(self):
        analyzer = TemporalConsistencyAnalyzer()
        with self.assertRaises(ValueError):
            self.observe(analyzer, 1, start=float("nan"))
        self.observe(analyzer, 1)
        motion, blink, gaze, illumination, replay = diagnostics()
        with self.assertRaises(ValueError):
            analyzer.update(0.0, self.tracking, motion, blink, gaze, illumination, replay, scores(), stable=True)

    def test_session_payload_is_scalar_and_strictly_validated(self):
        temporal, fusion, signal = self.observe(TemporalConsistencyAnalyzer())
        payload = {
            "decision": "UNCERTAIN", "confidence": 0.0,
            "signals": {name: {"confidence": .5, "reliability": 0.0,
                               "available": False, "diagnostic_information": "x"}
                        for name in ("blink", "gaze", "head_motion", "facial_geometry",
                                     "illumination_consistency", "replay_artifact_detection",
                                     "temporal_consistency")},
        }
        self.assertEqual(signal.confidence, .56)
        self.assertEqual(temporal.window_frame_count, 10)
        self.assertEqual(fusion.window_frame_count, 10)
        self.assertIsInstance(json.loads(json.dumps({"temporal": temporal.__dict__, "fusion": fusion.__dict__})), dict)


if __name__ == "__main__":
    unittest.main()
