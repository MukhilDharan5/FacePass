"""Ideal iris geometry tests; no physical gaze or presentation-attack claim."""

from collections import deque
import json
import unittest
from unittest.mock import patch

import numpy as np

from facepass.blink import LEFT_EYE
from facepass.gaze import GazeAnalyzer, LEFT_IRIS, iris_measurement
from facepass.head_pose import PoseEstimate
from facepass.landmarks import FaceObservation
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from tools.synthetic_motion import SyntheticBackend, gaze_mesh, planar_mesh, synthetic_frame
from tests.test_session import Clock


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = GazeAnalyzer()
        self.timestamp = 100.0
        self.history = deque(maxlen=64)

    def update(self, x=0.5, *, y=0.0, gap=0.1, track_id=1, stable=True, points=None, **geometry):
        self.timestamp += gap
        points = gaze_mesh(horizontal=x, vertical=y, **geometry) if points is None else points
        self.history.append(TrackedSample(track_id, self.timestamp, points, (480, 640)))
        return self.analyzer.update(tuple(self.history), stable=stable)

    def motion(self):
        for x in np.linspace(0.4, 0.55, 10):
            stats, signal = self.update(x)
        return stats, signal

    def test_pixel_space_position_and_rigid_transform_invariance(self):
        points = gaze_mesh(horizontal=0.4, vertical=0.02)
        m = iris_measurement(points, (480, 640), LEFT_EYE, LEFT_IRIS)
        np.testing.assert_allclose(m[:4], (0.4, 0.02, 36, 8))
        angle = np.deg2rad(15)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * 0.95
        points[:, :2] = ((points[:, :2] * [640, 480] - [320, 240]) @ rotation.T + [325, 243]) / [640, 480]
        m = iris_measurement(points, (480, 640), LEFT_EYE, LEFT_IRIS)
        np.testing.assert_allclose(m[:4], (0.4, 0.02, 34.2, 7.6), atol=1e-10)

    def test_bilateral_motion_is_weak_capped_evidence(self):
        stats, score = self.motion()
        self.assertTrue(stats.independent_motion_observed)
        self.assertAlmostEqual(stats.bilateral_agreement, 1)
        self.assertAlmostEqual(stats.motion_coherence, 1)
        self.assertEqual(score.confidence, 0.62)
        self.assertEqual(score.reliability, 0.30)
        self.assertTrue(score.available)

    def test_vertical_movement_can_supply_evidence(self):
        for y in np.linspace(-0.04, 0.04, 10):
            stats, score = self.update(y=y, ear=0.4)
        self.assertTrue(stats.independent_motion_observed)
        self.assertGreater(stats.vertical_span, 0.06)
        self.assertEqual(score.confidence, 0.62)

    def test_confirmed_step_between_fixations_can_supply_evidence(self):
        for x in [0.45] * 3 + [0.53] * 3:
            stats, score = self.update(x)
        self.assertTrue(stats.independent_motion_observed)
        self.assertEqual(score.confidence, 0.62)

    def test_still_eyes_neutral_without_a_required_direction(self):
        for x in (0.35, 0.5, 0.65):
            self.analyzer.reset()
            for _ in range(10):
                stats, score = self.update(x)
            self.assertFalse(stats.independent_motion_observed)
            self.assertEqual(score.confidence, 0.5)
            self.assertTrue(score.available)

    def test_short_window_and_tiny_movement_do_not_supply_evidence(self):
        for x in [0.4, 0.44, 0.48, 0.52, 0.56]:
            stats, score = self.update(x)
        self.assertFalse(score.available)
        self.analyzer.reset()
        for x in np.linspace(0.48, 0.52, 10):
            stats, score = self.update(x)
        self.assertEqual(score.confidence, 0.5)
        self.assertFalse(stats.independent_motion_observed)

    def test_single_frame_spike_and_jitter_are_not_rewarded(self):
        for xs in ([0.5] * 5 + [0.6] + [0.5] * 5, [0.46, 0.54] * 6):
            self.analyzer.reset()
            for x in xs:
                stats, score = self.update(x)
                self.assertLessEqual(score.confidence, 0.5)
            self.assertFalse(stats.independent_motion_observed)

    def test_unilateral_or_opposed_movement_is_not_rewarded(self):
        for mode in ("fixed", "opposed"):
            self.analyzer.reset()
            for x in np.linspace(0.4, 0.55, 10):
                stats, score = self.update(x, right_horizontal=0.5 if mode == "fixed" else 1 - x)
            self.assertEqual(score.confidence, 0.5)
            self.assertFalse(stats.independent_motion_observed)

    def test_head_rotation_confounds_apparent_iris_motion(self):
        for i, x in enumerate(np.linspace(0.4, 0.55, 10)):
            stats, score = self.update(x, yaw=i * 0.4)
        self.assertTrue(stats.head_motion_suppressed)
        self.assertGreater(stats.head_rotation_span_degrees, 3)
        self.assertEqual(score.confidence, 0.5)
        self.assertEqual(score.reliability, 0.05)

    def test_whole_frame_movement_does_not_become_eye_motion(self):
        for i in range(10):
            transform = np.array([[1, 0, i], [0, 1, i * 0.5], [0, 0, 1]], dtype=float)
            stats, score = self.update(points=planar_mesh(gaze_mesh(), transform))
        self.assertLess(stats.robust_displacement, 1e-9)
        self.assertEqual(score.confidence, 0.5)

    def test_missing_or_severe_pose_clears_earlier_evidence(self):
        for pose in (None, PoseEstimate(0, 30, 0, 0.1)):
            self.motion()
            with patch("facepass.gaze.estimate_pose", return_value=pose):
                stats, score = self.update()
            self.assertFalse(score.available)
            self.assertFalse(stats.pose_available)
            self.assertEqual(len(self.analyzer._samples), 0)

    def test_small_eye_or_iris_and_closure_are_unavailable(self):
        for geometry in ({"eye_width": 24}, {"iris_diameter": 4}, {"ear": 0.10}, {"ear": 0.19}):
            self.motion()
            stats, score = self.update(**geometry)
            self.assertFalse(score.available)
            self.assertFalse(stats.independent_motion_observed)
            self.assertEqual(len(self.analyzer._samples), 0)

    def test_partial_eye_failure_does_not_supply_positive_evidence(self):
        points = gaze_mesh()
        points[469, 0] = -0.1
        stats, score = self.update(points=points)
        self.assertTrue(stats.left_eye_available)
        self.assertFalse(stats.right_eye_available)
        self.assertFalse(score.available)

    def test_malformed_ring_or_center_outside_aperture_is_unavailable(self):
        collapsed = gaze_mesh()
        collapsed[list(LEFT_IRIS)] = collapsed[474]
        displaced = gaze_mesh()
        displaced[474, 0] += 0.02
        crossed = gaze_mesh()
        crossed[[475, 476]] = crossed[[476, 475]]
        for points in (collapsed, displaced, crossed, gaze_mesh(vertical=0.20)):
            with self.subTest():
                stats, score = self.update(points=points)
                self.assertFalse(stats.left_eye_available)
                self.assertFalse(score.available)

    def test_sudden_eye_or_iris_change_clears_evidence(self):
        for geometry in ({"eye_width": 48}, {"iris_diameter": 11}, {"horizontal": 0.75}):
            self.motion()
            stats, score = self.update(points=gaze_mesh(**geometry))
            self.assertEqual(stats.event, "gaze_geometry_jump")
            self.assertFalse(score.available)

    def test_gap_track_change_unstable_and_no_history_reset_state(self):
        for cause in ("gap", "track", "unstable", "missing"):
            self.motion()
            stats, score = (self.analyzer.update((), stable=True) if cause == "missing" else
                            self.update(gap=0.3 if cause == "gap" else 0.1,
                                        track_id=2 if cause == "track" else 1, stable=cause != "unstable"))
            self.assertFalse(score.available)
            self.assertFalse(stats.independent_motion_observed)
            self.assertLessEqual(len(self.analyzer._samples), 1)

    def test_polling_sampling_expiry_clears_all_private_history(self):
        self.motion()
        stats, score = self.analyzer.current_result(self.timestamp + 0.25)
        self.assertFalse(score.available)
        self.assertEqual(stats.event, "gaze_sampling_gap")
        self.assertEqual(len(self.analyzer._samples), 0)
        self.assertIsNone(self.analyzer._previous_positions)

    def test_two_second_and_frame_limits_bound_private_history(self):
        for _ in range(200):
            stats, _ = self.update(gap=0.01)
        self.assertEqual(len(self.analyzer._samples), 64)
        self.assertEqual(stats.window_frame_count, 64)
        for _ in range(30):
            stats, _ = self.update()
        self.assertLessEqual(stats.window_duration_ms, 2000)
        self.assertLessEqual(self.timestamp - self.analyzer._samples[0][0], 2)

    def test_polling_trims_window_with_a_fresh_latest_sample(self):
        for _ in range(21):
            self.update()
        stats, score = self.analyzer.current_result(self.timestamp + 0.15)
        self.assertLessEqual(self.timestamp + 0.15 - self.analyzer._samples[0][0], 2)
        self.assertTrue(score.available)

    def test_old_eye_movement_expires_during_still_observation(self):
        self.motion()
        for _ in range(25):
            stats, score = self.update(0.55)
        self.assertEqual(score.confidence, 0.5)
        self.assertFalse(stats.independent_motion_observed)

    def test_invalid_times_nan_and_native_exceptions_release_state(self):
        for cause in ("timestamp", "nan", "exception", "pose_nan", "clock"):
            self.motion()
            with self.assertRaises((ValueError, RuntimeError)):
                if cause == "timestamp":
                    self.analyzer.update(tuple(self.history), stable=True)
                elif cause == "nan":
                    points = gaze_mesh()
                    points[474, 0] = np.nan
                    self.update(points=points)
                elif cause == "clock":
                    self.analyzer.current_result(float("nan"))
                else:
                    with patch("facepass.gaze.estimate_pose", **({"side_effect": RuntimeError("native failure")} if cause == "exception"
                               else {"return_value": PoseEstimate(0, float("nan"), 0, 0)})):
                        self.update()
            self.assertEqual(len(self.analyzer._samples), 0)
            self.assertIsNone(self.analyzer._previous_positions)

    def test_native_failure_during_polling_releases_private_state(self):
        self.motion()
        with patch("facepass.gaze.np.linalg.lstsq", side_effect=RuntimeError("native failure")):
            with self.assertRaises(RuntimeError):
                self.analyzer.current_result(self.timestamp)
        self.assertEqual(len(self.analyzer._samples), 0)
        self.assertIsNone(self.analyzer._previous_positions)


class GazeSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = SyntheticBackend()
        self.frame = synthetic_frame()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def update(self, x=0.5, *, frame=None, **geometry):
        self.clock.now += 0.1
        self.backend.points = gaze_mesh(horizontal=x, **geometry)
        result = self.session.process_frame(self.frame if frame is None else frame, self.clock.now)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        self.assertEqual(result.confidence, 0)
        self.assertNotEqual(result.decision, Decision.LIVE)
        return result

    def observe(self):
        for _ in range(3):
            self.update(0.4)
        for x in np.linspace(0.4, 0.55, 10):
            result = self.update(x)
        self.assertEqual(result.signals["gaze"].confidence, 0.62)
        return result

    def test_finish_retains_only_unavailable_historical_scalars(self):
        self.observe()
        result = self.session.finish()
        self.assertTrue(result.gaze.historical)
        self.assertFalse(result.signals["gaze"].available)
        self.assertEqual(result.signals["gaze"].reliability, 0)
        self.assertEqual(len(self.session._gaze._samples), 0)
        self.assertEqual(len(self.session.history), 0)
        validate_result(json.loads(json.dumps(result.to_dict())))

    def test_failure_and_tracking_reset_paths_clear_gaze(self):
        for cause in ("loss", "quality", "stale", "cancel", "failure", "multiple", "discontinuity"):
            self.session.finish()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.backend.visible = True
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
            elif cause == "multiple":
                self.clock.now += 0.1
                with patch.object(self.backend, "detect", return_value=FaceObservation((gaze_mesh(), gaze_mesh()))):
                    result = self.session.process_frame(self.frame, self.clock.now)
            elif cause == "discontinuity":
                self.backend.points = gaze_mesh()
                self.backend.points[:, 0] += 0.3
                self.clock.now += 0.1
                result = self.session.process_frame(self.frame, self.clock.now)
            else:
                result = self.session.fail("camera_read_failed")
            self.assertFalse(result.signals["gaze"].available)
            self.assertFalse(result.gaze.historical)
            self.assertEqual(result.gaze.window_frame_count, 0)
            self.assertEqual(len(self.session._gaze._samples), 0)

    def test_unreliable_eye_geometry_keeps_the_face_track_usable(self):
        self.observe()
        result = self.update(ear=0.10)
        self.assertTrue(result.quality.acceptable)
        self.assertEqual(result.tracking.state.value, "STABLE")
        self.assertFalse(result.signals["gaze"].available)
        self.assertFalse(result.terminal)

    def test_cadence_expiry_precedes_track_expiry(self):
        self.observe()
        self.clock.now += 0.25
        result = self.session.current_result()
        self.assertEqual(result.tracking.state.value, "STABLE")
        self.assertFalse(result.signals["gaze"].available)
        self.assertEqual(result.gaze.window_frame_count, 0)

    def test_default_and_faster_cadence_produce_valid_json(self):
        for rate in (12, 30):
            self.session.finish()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            for i in range(30):
                self.clock.now += 1 / rate
                self.backend.points = gaze_mesh(horizontal=0.45 + i * 0.006)
                result = self.session.process_frame(self.frame, self.clock.now)
                validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
                self.assertIsInstance(result.gaze.independent_motion_observed, bool)
                self.assertFalse(result.terminal)
                self.assertEqual(result.confidence, 0)

    def test_detector_and_cleanup_failure_clear_positive_evidence(self):
        self.observe()
        with patch("facepass.gaze.iris_measurement", side_effect=RuntimeError("native failure")):
            result = self.update()
        self.assertTrue(result.terminal)
        self.assertEqual(result.gaze.window_frame_count, 0)
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()
        self.observe()
        with patch.object(self.backend, "close", side_effect=RuntimeError("cleanup failure")):
            result = self.session.finish()
        self.assertEqual(result.reason, "model_cleanup_failed")
        self.assertFalse(result.gaze.historical)
        self.assertEqual(result.gaze.window_frame_count, 0)

    def test_stillness_neutral_and_temporal_fusion_stays_non_positive(self):
        for _ in range(12):
            result = self.update()
        self.assertEqual(result.signals["gaze"].confidence, 0.5)
        self.assertTrue(result.signals["gaze"].available)
        self.assertLessEqual(result.signals["temporal_consistency"].confidence, 0.5)
        self.assertFalse(result.fusion.liveness_candidate)

    def test_native_poll_or_finish_failure_terminates_without_historical_evidence(self):
        for operation in ("poll", "finish"):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            self.observe()
            with patch("facepass.gaze.np.linalg.lstsq", side_effect=RuntimeError("native failure")):
                result = self.session.current_result() if operation == "poll" else self.session.finish()
            self.assertTrue(result.terminal)
            self.assertEqual(result.reason, "temporal_evidence_error")
            self.assertFalse(result.gaze.historical)
            self.assertEqual(result.gaze.window_frame_count, 0)
            self.assertEqual(len(self.session.history), 0)
            self.assertTrue(self.backend.closed)
            self.assertEqual(result.confidence, 0)

    def test_inference_finishing_after_timeout_does_not_refresh_old_gaze(self):
        self.session.finish()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock, max_duration=1.5)
        self.session.start()
        self.observe()
        original = self.backend.detect

        def delayed(frame, timestamp):
            self.clock.now += 0.4
            return original(frame, timestamp)

        with patch.object(self.backend, "detect", side_effect=delayed):
            result = self.update(0.55)
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "observation_timeout")
        self.assertFalse(result.gaze.historical)
        self.assertEqual(result.gaze.window_frame_count, 0)
        self.assertFalse(result.signals["gaze"].available)

    def test_ipc_rejects_bad_values_vectors_context_and_excessive_scores(self):
        base = self.observe().to_dict()
        mutations = (("left_horizontal_ratio", float("nan")), ("iris_history", [0.4, 0.5]),
                     ("left_eye_available", 1), ("window_frame_count", None), ("motion_coherence", None),
                     ("left_eye_width_px", 10), ("left_iris_diameter_px", 3), ("pose_available", False),
                     ("independent_motion_observed", False), ("head_motion_suppressed", True),
                     ("head_rotation_span_degrees", 2), ("robust_displacement", 0.001),
                     ("motion_coherence", 0.5), ("bilateral_agreement", 0.5),
                     ("historical", True), ("status", "not_observed"))
        for key, value in mutations:
            candidate = json.loads(json.dumps(base))
            candidate["gaze"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(candidate)
        for key in ("confidence", "reliability"):
            candidate = json.loads(json.dumps(base))
            candidate["signals"]["gaze"][key] = 1
            with self.assertRaises(ValueError):
                validate_result(candidate)
        candidate = json.loads(json.dumps(base))
        candidate.update(decision="LIVE", confidence=1)
        with self.assertRaises(ValueError):
            validate_result(candidate)


if __name__ == "__main__":
    unittest.main()
