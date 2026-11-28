"""Test motion math and fail-closed boundaries with generated pixels/geometry."""

from collections import deque
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from facepass.head_pose import PoseEstimate, estimate_pose
from facepass.motion import MotionAnalyzer, environment_image, planar_comparison, temporal_rotation
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from tools.synthetic_motion import SyntheticBackend, planar_mesh, projected_mesh, synthetic_frame
from tests.test_session import Clock


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MotionAnalyzer()
        self.history = deque(maxlen=64)
        self.frame = synthetic_frame()

    def update(self, points, index, frame=None, track_id=1):
        self.history.append(TrackedSample(track_id, 100 + index * 0.1, points, (480, 640)))
        return self.analyzer.update(self.frame if frame is None else frame, tuple(self.history), stable=index >= 2)

    def test_pose_recovers_synthetic_small_yaw_pitch_roll(self):
        pose = estimate_pose(projected_mesh(yaw=0.2, pitch=0.1, roll=0.05), (480, 640))
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.yaw, 0.2, delta=0.005)
        self.assertAlmostEqual(pose.pitch, 0.1, delta=0.005)
        self.assertAlmostEqual(pose.roll, 0.05, delta=0.005)
        self.assertLess(pose.reprojection_error_px, 0.01)

    def test_single_frame_and_short_window_supply_no_evidence(self):
        for index in range(5):
            stats, head, geometry = self.update(projected_mesh(yaw=index * 0.05), index)
            self.assertFalse(head.available)
            self.assertFalse(geometry.available)
        self.assertIsNone(stats.rotation_span_degrees)

    def test_still_face_has_neutral_motion_not_spoof(self):
        for index in range(8):
            stats, head, geometry = self.update(projected_mesh(), index)
        self.assertTrue(head.available)
        self.assertEqual(head.confidence, 0.5)
        self.assertEqual(geometry.confidence, 0.5)
        self.assertAlmostEqual(stats.rotation_span_degrees, 0.0, places=4)

    def test_small_coherent_3d_rotation_has_weak_bounded_evidence(self):
        for index in range(10):
            stats, head, geometry = self.update(projected_mesh(yaw=index * 0.05), index)
        self.assertTrue(stats.background_available)
        self.assertGreater(stats.temporal_coherence, 0.9)
        self.assertGreater(stats.rotation_span_degrees, 0.2)
        self.assertGreater(stats.nonplanar_fraction, 0.1)
        self.assertGreater(head.confidence, 0.5)
        self.assertLessEqual(head.confidence, 0.68)
        self.assertLessEqual(head.reliability, 0.4)
        self.assertGreater(geometry.confidence, 0.5)

    def test_planar_homography_predicts_held_out_landmarks(self):
        source = projected_mesh()
        matrix = np.array([[1.002, 0.001, 1.0], [-0.001, 1.001, 0.5], [1e-5, -1e-5, 1]], np.float64)
        residual = planar_comparison(source, planar_mesh(source, matrix), (480, 640))
        self.assertIsNotNone(residual)
        self.assertLess(residual, 0.001)

    def test_moved_planar_media_does_not_gain_positive_motion_evidence(self):
        source = projected_mesh()
        for index in range(10):
            matrix = np.array([[1, 0, index * 0.2], [0, 1, index * 0.1], [0, 0, 1]], np.float64)
            stats, head, geometry = self.update(planar_mesh(source, matrix), index)
        self.assertTrue(stats.planar_available)
        self.assertLess(stats.nonplanar_fraction, 0.01)
        self.assertLessEqual(head.confidence, 0.5)
        self.assertLessEqual(geometry.confidence, 0.5)

    def test_shared_scene_camera_translation_is_suppressed(self):
        source = projected_mesh()
        for index in range(8):
            matrix = np.array([[1, 0, index * 0.6], [0, 1, index * 0.3], [0, 0, 1]], np.float64)
            frame = cv2.warpPerspective(self.frame, matrix, (640, 480))
            stats, head, geometry = self.update(planar_mesh(source, matrix), index, frame)
        self.assertTrue(stats.background_available)
        self.assertTrue(stats.common_motion_suppressed)
        # Subpixel image warping/downsampling introduces flow bias. Check that
        # most motion is explained and that neither producer rewards it.
        self.assertLess(stats.independent_motion_px / stats.common_motion_px, 0.2)
        self.assertLessEqual(head.confidence, 0.5)
        self.assertLessEqual(geometry.confidence, 0.5)

    def test_shared_projective_scene_motion_is_not_positive_evidence(self):
        source = projected_mesh()
        for index in range(8):
            matrix = np.array([[1 + index * 0.001, index * 0.001, index * 0.6],
                               [-index * 0.001, 1, index * 0.3],
                               [index * 1e-5, -index * 1e-5, 1]], np.float64)
            frame = cv2.warpPerspective(self.frame, matrix, (640, 480))
            stats, head, geometry = self.update(planar_mesh(source, matrix), index, frame)
        self.assertTrue(stats.background_available)
        self.assertTrue(stats.common_motion_suppressed)
        self.assertLessEqual(head.confidence, 0.5)
        self.assertLessEqual(geometry.confidence, 0.5)

    def test_missing_background_texture_reduces_evidence(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        for index in range(10):
            stats, head, geometry = self.update(projected_mesh(yaw=index * 0.1), index, flat)
        self.assertFalse(stats.background_available)
        self.assertEqual(head.confidence, 0.5)
        self.assertLessEqual(head.reliability, 0.15)
        self.assertEqual(geometry.confidence, 0.5)

    def test_unstructured_pose_jitter_has_no_positive_motion_score(self):
        rng = np.random.default_rng(31)
        for index in range(12):
            stats, head, geometry = self.update(projected_mesh(yaw=float(rng.normal(0, 0.08))), index)
        self.assertLess(stats.temporal_coherence, 0.55)
        self.assertEqual(head.confidence, 0.5)
        self.assertEqual(geometry.confidence, 0.5)

    def test_unreliable_pose_breaks_rotation_history(self):
        for index in range(7):
            self.update(projected_mesh(yaw=index * 0.1), index)
        with patch("facepass.motion.estimate_pose", return_value=None):
            stats, head, geometry = self.update(projected_mesh(), 7)
        self.assertFalse(head.available)
        self.assertEqual(stats.window_frame_count, 1)
        stats, head, geometry = self.update(projected_mesh(), 8)
        self.assertFalse(head.available)

    def test_track_change_and_reset_release_buffers(self):
        self.update(projected_mesh(), 0)
        self.assertIsNotNone(self.analyzer._image)
        self.history.clear()
        stats, head, geometry = self.update(projected_mesh(), 1, track_id=2)
        self.assertEqual(stats.window_frame_count, 1)
        self.assertFalse(head.available)
        self.analyzer.reset()
        self.assertIsNone(self.analyzer._image)
        self.assertIsNone(self.analyzer._mask)
        self.assertEqual(len(self.analyzer._samples), 0)
        self.assertEqual(len(self.analyzer._flows), 0)

    def test_environment_buffer_masks_face_pixels_and_is_small(self):
        image, mask = environment_image(self.frame, projected_mesh())
        self.assertLessEqual(image.shape[1], 320)
        self.assertLessEqual(image.shape[0], 240)
        self.assertTrue(np.all(image[mask == 0] == 0))
        self.assertEqual(image[120, 160], 0)

    def test_temporal_history_is_bounded(self):
        for index in range(40):
            self.update(projected_mesh(), index)
        self.assertLessEqual(len(self.analyzer._samples), 64)
        self.assertLessEqual(self.analyzer._samples[-1][0] - self.analyzer._samples[0][0], 2)

    def test_nan_pose_output_is_rejected(self):
        with patch("facepass.head_pose.cv2.solvePnP", return_value=(True, np.full((3, 1), np.nan), np.ones((3, 1)))):
            with self.assertRaises(ValueError):
                estimate_pose(projected_mesh(), (480, 640))

    def test_exception_after_good_motion_clears_private_buffers(self):
        self.update(projected_mesh(), 0)
        with patch("facepass.motion.estimate_pose", side_effect=RuntimeError("simulated native error")):
            with self.assertRaises(RuntimeError):
                self.update(projected_mesh(), 1)
        self.assertIsNone(self.analyzer._image)
        self.assertEqual(len(self.analyzer._samples), 0)

    def test_temporal_rotation_stationary_and_noise_differ_from_trend(self):
        still = [(100 + i * 0.1, PoseEstimate(0, 0, 0, 0)) for i in range(8)]
        self.assertEqual(temporal_rotation(still)[1], 0)
        trend = [(100 + i * 0.1, PoseEstimate(0, i * 0.05, 0, 0)) for i in range(8)]
        self.assertGreater(temporal_rotation(trend)[1], 0.99)


class MotionSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = SyntheticBackend()
        self.frame = synthetic_frame()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def observe(self, count=10):
        for index in range(count):
            self.clock.now += 0.1
            self.backend.points = projected_mesh(yaw=index * 0.05)
            result = self.session.process_frame(self.frame, self.clock.now)
            validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
            self.assertEqual(result.decision, Decision.UNCERTAIN)
            self.assertEqual(result.confidence, 0)
        return result

    def test_weak_motion_evidence_cannot_authenticate(self):
        result = self.observe()
        self.assertGreater(result.signals["head_motion"].confidence, 0.5)
        self.assertFalse(result.signals["blink"].available)
        self.assertEqual(result.confidence, 0)

    def test_normal_finish_keeps_only_historical_scalar_diagnostics(self):
        live_measurements = self.observe()
        result = self.session.finish()
        self.assertTrue(result.motion.historical)
        self.assertEqual(result.signals["head_motion"].confidence, live_measurements.signals["head_motion"].confidence)
        self.assertFalse(result.signals["head_motion"].available)
        self.assertEqual(result.signals["head_motion"].reliability, 0)
        self.assertIsNone(self.session._motion._image)
        self.assertEqual(len(self.session._motion._samples), 0)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))

    def test_timeout_during_inference_does_not_refresh_old_summary(self):
        self.session.finish()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock, max_duration=1.4)
        self.session.start()
        self.observe()
        self.clock.now += 0.15
        capture_time = self.clock.now
        detect = self.backend.detect

        def delayed_detect(frame, timestamp):
            self.clock.now += 0.3
            return detect(frame, timestamp)

        with patch.object(self.backend, "detect", side_effect=delayed_detect):
            result = self.session.process_frame(self.frame, capture_time)
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "observation_timeout")
        self.assertEqual(result.confidence, 0)
        self.assertFalse(result.motion.historical)
        self.assertEqual(result.motion.window_frame_count, 0)
        self.assertFalse(result.signals["head_motion"].available)
        self.assertIsNone(self.session._motion._image)

    def test_loss_quality_staleness_cancellation_and_failure_clear_motion(self):
        for cause in ("loss", "quality", "stale", "cancel", "failure"):
            with self.subTest(cause=cause):
                self.backend = SyntheticBackend()
                self.session = LivenessSession(lambda: self.backend, clock=self.clock)
                self.session.start()
                self.observe()
                if cause == "cancel":
                    result = self.session.cancel()
                elif cause == "failure":
                    result = self.session.fail("camera_read_failed")
                elif cause == "stale":
                    self.clock.now += 0.5
                    result = self.session.current_result()
                else:
                    self.clock.now += 0.1
                    if cause == "loss":
                        self.backend.visible = False
                        frame = self.frame
                    else:
                        frame = np.zeros_like(self.frame)
                    result = self.session.process_frame(frame, self.clock.now)
                self.assertFalse(result.signals["head_motion"].available)
                self.assertEqual(result.signals["head_motion"].reliability, 0)
                self.assertFalse(result.motion.historical)
                self.assertEqual(result.motion.window_frame_count, 0)
                self.assertIsNone(self.session._motion._image)
                self.session.finish()

    def test_detector_exception_and_cleanup_failure_fail_closed(self):
        self.observe()
        self.clock.now += 0.1
        with patch("facepass.motion.estimate_pose", side_effect=RuntimeError("simulated failure")):
            result = self.session.process_frame(self.frame, self.clock.now)
        self.assertTrue(result.terminal)
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.motion.window_frame_count, 0)
        self.assertIsNone(self.session._motion._image)
        self.backend = SyntheticBackend()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()
        self.observe()
        with patch.object(self.backend, "close", side_effect=RuntimeError("simulated close failure")):
            result = self.session.finish()
        self.assertEqual(result.reason, "model_cleanup_failed")
        self.assertEqual(result.motion.window_frame_count, 0)
        self.assertFalse(result.signals["head_motion"].available)

    def test_nan_motion_and_unknown_vectors_are_rejected_at_ipc(self):
        result = self.observe().to_dict()
        for key, value in (("yaw_degrees", float("nan")), ("background_available", 1),
                           ("pose_vector", [1, 2, 3]), ("window_frame_count", 10000)):
            candidate = json.loads(json.dumps(result))
            candidate["motion"][key] = value
            with self.assertRaises(ValueError):
                validate_result(candidate)
        candidate = json.loads(json.dumps(result))
        candidate["signals"]["blink"].update(available=True, reliability=1)
        with self.assertRaises(ValueError):
            validate_result(candidate)
        candidate = json.loads(json.dumps(result))
        candidate.update(decision="LIVE", confidence=1)
        with self.assertRaises(ValueError):
            validate_result(candidate)


if __name__ == "__main__":
    unittest.main()
