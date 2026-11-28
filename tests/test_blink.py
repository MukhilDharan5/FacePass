"""Generated blink contours test sequencing, confounders and fail-closed resets."""

from collections import deque
import json
import unittest
from unittest.mock import patch

import numpy as np

from facepass.blink import BlinkAnalyzer, LEFT_EYE, RIGHT_EYE, eye_measurement
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from tools.synthetic_motion import SyntheticBackend, blink_mesh, synthetic_frame
from tests.test_session import Clock


BLINK_SEQUENCE = [0.30, 0.30, 0.30, 0.19, 0.10, 0.10, 0.19, 0.30, 0.30]


class BlinkTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = BlinkAnalyzer()
        self.history = deque(maxlen=64)
        self.timestamp = 100.0

    def update(self, ear=0.30, *, right=None, gap=0.1, track_id=1, stable=True, points=None):
        self.timestamp += gap
        mesh = blink_mesh(left_ear=ear, right_ear=ear if right is None else right) if points is None else points
        self.history.append(TrackedSample(track_id, self.timestamp, mesh, (480, 640)))
        return self.analyzer.update(tuple(self.history), stable=stable)

    def observe(self, sequence=BLINK_SEQUENCE):
        phases = []
        for ear in sequence:
            stats, score = self.update(ear)
            phases.append(stats.state)
        return stats, score, phases

    def test_ear_uses_pixels_and_is_invariant_to_rigid_roll_and_scale(self):
        points = blink_mesh()
        self.assertAlmostEqual(eye_measurement(points, (480, 640), LEFT_EYE)[0], 0.30)
        angle = np.deg2rad(20)
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * 0.9
        points[:, :2] = ((points[:, :2] * [640, 480] - [320, 240]) @ matrix.T + [320, 240]) / [640, 480]
        self.assertAlmostEqual(eye_measurement(points, (480, 640), LEFT_EYE)[0], 0.30)
        self.assertAlmostEqual(eye_measurement(points, (480, 640), RIGHT_EYE)[0], 0.30)

    def test_full_bilateral_sequence_supplies_only_weak_evidence(self):
        stats, score, phases = self.observe()
        for phase in ("ACQUIRING_OPEN", "OPEN", "CLOSING", "CLOSED", "OPENING"):
            self.assertIn(phase, phases)
        self.assertEqual(stats.blink_count, 1)
        self.assertEqual(stats.event, "blink_completed")
        self.assertEqual(stats.state, "OPEN")
        self.assertEqual(stats.last_blink_duration_ms, 400)
        self.assertEqual(score.confidence, 0.65)
        self.assertEqual(score.reliability, 0.35)

    def test_open_stillness_never_implies_spoof_or_requires_blink(self):
        stats, score, _ = self.observe([0.30] * 20)
        self.assertEqual(stats.blink_count, 0)
        self.assertTrue(score.available)
        self.assertEqual(score.confidence, 0.5)

    def test_starting_closed_then_open_is_not_a_blink(self):
        stats, score, _ = self.observe([0.10] * 10 + [0.30] * 3)
        self.assertEqual(stats.blink_count, 0)
        self.assertEqual(stats.state, "OPEN")
        self.assertEqual(score.confidence, 0.5)

    def test_sustained_closure_and_delayed_reopening_are_not_blinks(self):
        stats, score, _ = self.observe([0.30] * 3 + [0.10] * 12)
        self.assertEqual(stats.state, "SUSTAINED_CLOSED")
        self.assertEqual(score.confidence, 0.5)
        self.assertEqual(stats.blink_count, 0)
        self.assertEqual(stats.closed_duration_ms, 1100)
        stats, score, _ = self.observe([0.30] * 3)
        self.assertEqual(stats.blink_count, 0)

    def test_single_frame_collapse_and_separated_spikes_are_not_blinks(self):
        for sequence in ([0.30] * 3 + [0.10] + [0.30] * 3,
                         [0.30] * 3 + [0.10, 0.19, 0.10] + [0.30] * 3):
            self.analyzer.reset()
            stats, score, _ = self.observe(sequence)
            self.assertEqual(stats.blink_count, 0)
            self.assertEqual(score.confidence, 0.5)
            self.assertGreater(stats.rejected_sequence_count, 0)

    def test_squint_hysteresis_noise_without_confirmed_closure_is_neutral(self):
        stats, score, _ = self.observe([0.30] * 3 + [0.22, 0.17, 0.22, 0.24, 0.30, 0.30])
        self.assertEqual(stats.blink_count, 0)
        self.assertEqual(score.confidence, 0.5)

    def test_wink_or_asymmetric_tracking_breaks_the_sequence(self):
        self.observe([0.30] * 3)
        stats, score = self.update(0.10, right=0.30)
        self.assertFalse(stats.eyes_available)
        self.assertFalse(score.available)
        self.assertEqual(stats.event, "eye_bilateral_disagreement")
        stats, score, _ = self.observe([0.10, 0.10, 0.30, 0.30])
        self.assertEqual(stats.blink_count, 0)

    def test_small_degenerate_clipped_crossed_and_displaced_eyes_are_unavailable(self):
        meshes = [blink_mesh(eye_width=10), blink_mesh(eye_width=0)]
        clipped = blink_mesh()
        clipped[385, 0] = -0.01
        meshes.append(clipped)
        crossed = blink_mesh()
        crossed[[385, 387], 1] += 0.04
        meshes.append(crossed)
        displaced = blink_mesh()
        displaced[385, 0] += 0.05
        meshes.append(displaced)
        for mesh in meshes:
            with self.subTest():
                stats, score = self.update(points=mesh)
                self.assertFalse(stats.eyes_available)
                self.assertFalse(score.available)

    def test_eye_width_jump_clears_previous_positive_evidence(self):
        self.observe()
        stats, score = self.update(points=blink_mesh(eye_width=48))
        self.assertEqual(stats.event, "eye_geometry_jump")
        self.assertEqual(stats.blink_count, 0)
        self.assertFalse(score.available)

    def test_short_closed_interval_at_high_cadence_is_not_confirmed(self):
        self.observe([0.30] * 3)
        self.update(0.10, gap=0.01)
        self.update(0.10, gap=0.01)
        stats, score = self.update(0.30, gap=0.01)
        self.assertEqual(stats.blink_count, 0)
        self.assertEqual(score.confidence, 0.5)

    def test_sampling_gap_track_change_unstable_track_and_expiry_clear_state(self):
        for cause in ("gap", "track", "unstable", "expire", "no_history"):
            with self.subTest(cause=cause):
                self.analyzer.reset()
                self.observe()
                if cause == "expire":
                    stats, score = self.analyzer.current_result(self.timestamp + 0.25)
                elif cause == "no_history":
                    stats, score = self.analyzer.update((), stable=True)
                else:
                    stats, score = self.update(gap=0.3 if cause == "gap" else 0.1,
                                               track_id=2 if cause == "track" else 1,
                                               stable=cause != "unstable")
                self.assertEqual(stats.blink_count, 0)
                self.assertFalse(score.available)
                self.assertIsNone(self.analyzer._last_completed)

    def test_partial_blink_cannot_bridge_sampling_gap(self):
        self.observe([0.30] * 3 + [0.10, 0.10])
        self.update(0.30, gap=0.3)
        stats, score, _ = self.observe([0.30] * 3)
        self.assertEqual(stats.blink_count, 0)
        self.assertEqual(score.confidence, 0.5)

    def test_positive_credit_expires_during_continuous_open_observation(self):
        self.observe()
        stats, score, _ = self.observe([0.30] * 21)
        self.assertEqual(stats.blink_count, 1)
        self.assertEqual(score.confidence, 0.5)

    def test_polling_expires_credit_while_the_latest_sample_is_fresh(self):
        self.observe()
        self.observe([0.30] * 19)
        stats, score = self.analyzer.current_result(self.timestamp + 0.15)
        self.assertEqual(stats.blink_count, 1)
        self.assertTrue(score.available)
        self.assertGreater(stats.last_blink_age_ms, 2000)
        self.assertEqual(score.confidence, 0.5)

    def test_bad_timestamps_nan_landmarks_and_exceptions_reset_detector(self):
        for cause in ("timestamp", "nan", "exception"):
            with self.subTest(cause=cause):
                self.analyzer.reset()
                self.observe()
                if cause == "timestamp":
                    with self.assertRaises(ValueError):
                        self.analyzer.update(tuple(self.history), stable=True)
                elif cause == "nan":
                    points = blink_mesh()
                    points[385, 0] = np.nan
                    with self.assertRaises(ValueError):
                        self.update(points=points)
                else:
                    with patch("facepass.blink.eye_measurement", side_effect=RuntimeError("simulated failure")):
                        with self.assertRaises(RuntimeError):
                            self.update()
                self.assertIsNone(self.analyzer._last_completed)
                self.assertIsNone(self.analyzer._previous_widths)

    def test_repeated_cycles_do_not_raise_the_score_or_reliability_caps(self):
        for _ in range(4):
            stats, score, _ = self.observe()
            self.assertEqual(score.confidence, 0.65)
            self.assertEqual(score.reliability, 0.35)
        self.assertEqual(stats.blink_count, 4)


class BlinkSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = SyntheticBackend()
        self.backend.points = blink_mesh()
        self.frame = synthetic_frame()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def update(self, ear=0.30, frame=None):
        self.clock.now += 0.1
        self.backend.points = blink_mesh(left_ear=ear, right_ear=ear)
        result = self.session.process_frame(self.frame if frame is None else frame, self.clock.now)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        self.assertEqual(result.confidence, 0)
        self.assertNotEqual(result.decision, Decision.LIVE)
        return result

    def observe(self):
        # Tracker acquires first; the detector must observe its own open baseline
        # after the track is STABLE rather than reuse pre-stability observations.
        for ear in [0.30] * 3 + BLINK_SEQUENCE:
            result = self.update(ear)
        self.assertEqual(result.blink.blink_count, 1)
        self.assertGreater(result.signals["blink"].confidence, 0.5)
        return result

    def test_blink_is_never_authentication_and_finish_is_historical_only(self):
        self.observe()
        result = self.session.finish()
        self.assertTrue(result.blink.historical)
        self.assertFalse(result.signals["blink"].available)
        self.assertEqual(result.signals["blink"].reliability, 0)
        self.assertEqual(result.confidence, 0)
        self.assertIsNone(self.session._blink._last_completed)
        self.assertEqual(len(self.session.history), 0)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))

    def test_face_loss_quality_failure_stale_capture_cancel_and_failure_clear_blink(self):
        for cause in ("loss", "quality", "stale", "cancel", "failure", "multiple_faces"):
            with self.subTest(cause=cause):
                self.session.finish()
                self.backend = SyntheticBackend()
                self.session = LivenessSession(lambda: self.backend, clock=self.clock)
                self.session.start()
                self.observe()
                if cause == "loss":
                    self.backend.visible = False
                    result = self.update()
                elif cause == "quality":
                    result = self.update(frame=np.zeros_like(self.frame))
                elif cause == "stale":
                    self.clock.now += 0.5
                    result = self.session.current_result()
                elif cause == "cancel":
                    result = self.session.cancel()
                elif cause == "multiple_faces":
                    from facepass.landmarks import FaceObservation
                    self.clock.now += 0.1
                    with patch.object(self.backend, "detect", return_value=FaceObservation((blink_mesh(), blink_mesh()))):
                        result = self.session.process_frame(self.frame, self.clock.now)
                else:
                    result = self.session.fail("camera_read_failed")
                self.assertFalse(result.signals["blink"].available)
                self.assertEqual(result.blink.blink_count, 0)
                self.assertFalse(result.blink.historical)
                self.assertIsNone(self.session._blink._last_completed)
                self.session.finish()

    def test_no_blink_is_neutral_and_no_gaze_detector_is_enabled(self):
        for _ in range(15):
            result = self.update()
        self.assertEqual(result.signals["blink"].confidence, 0.5)
        self.assertTrue(result.signals["blink"].available)
        self.assertFalse(result.signals["gaze"].available)

    def test_finish_during_closure_never_completes_a_blink(self):
        for ear in [0.30] * 6 + [0.10, 0.10]:
            result = self.update(ear)
        self.assertEqual(result.blink.state, "CLOSED")
        result = self.session.finish()
        self.assertTrue(result.blink.historical)
        self.assertEqual(result.blink.blink_count, 0)
        self.assertFalse(result.signals["blink"].available)
        self.assertEqual(result.confidence, 0)

    def test_detector_exception_and_cleanup_failure_clear_positive_credit(self):
        self.observe()
        self.clock.now += 0.1
        with patch("facepass.blink.eye_measurement", side_effect=RuntimeError("simulated detector failure")):
            result = self.session.process_frame(self.frame, self.clock.now)
        self.assertTrue(result.terminal)
        self.assertEqual(result.blink.blink_count, 0)
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()
        self.observe()
        with patch.object(self.backend, "close", side_effect=RuntimeError("simulated cleanup failure")):
            result = self.session.finish()
        self.assertEqual(result.reason, "model_cleanup_failed")
        self.assertEqual(result.blink.blink_count, 0)
        self.assertFalse(result.blink.historical)

    def test_cadence_expiry_without_a_new_frame_clears_blink_before_track_expiry(self):
        self.observe()
        self.clock.now += 0.25
        result = self.session.current_result()
        self.assertEqual(result.tracking.state.value, "STABLE")
        self.assertFalse(result.signals["blink"].available)
        self.assertEqual(result.blink.blink_count, 0)

    def test_blink_ipc_rejects_nan_vectors_bad_types_context_and_excess_credit(self):
        base = self.observe().to_dict()
        mutations = [("left_ear", float("nan")), ("ear_history", [0.30, 0.10]),
                     ("eyes_available", 1), ("state", "UNKNOWN"),
                     ("observed_frame_count", None), ("blink_count", 0),
                     ("status", "not_observed"), ("left_eye_width_px", 2.0), ("left_ear", 0.10),
                     ("last_blink_age_ms", 2001), ("last_blink_duration_ms", 900),
                     ("historical", True), ("state", "CLOSED")]
        for key, value in mutations:
            candidate = json.loads(json.dumps(base))
            candidate["blink"][key] = value
            with self.assertRaises(ValueError):
                validate_result(candidate)
        for signal in ({"confidence": 1}, {"reliability": 1}):
            candidate = json.loads(json.dumps(base))
            candidate["signals"]["blink"].update(signal)
            with self.assertRaises(ValueError):
                validate_result(candidate)
        candidate = json.loads(json.dumps(base))
        candidate["signals"]["gaze"].update(available=True, reliability=0.2)
        with self.assertRaises(ValueError):
            validate_result(candidate)
        candidate = json.loads(json.dumps(base))
        candidate.update(decision="LIVE", confidence=1)
        with self.assertRaises(ValueError):
            validate_result(candidate)


if __name__ == "__main__":
    unittest.main()
