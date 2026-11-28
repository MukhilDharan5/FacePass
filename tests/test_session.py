"""Security boundary tests use synthetic geometry, never stored user images."""

import json
import math
import unittest

import numpy as np

from facepass.landmarks import FaceObservation
from facepass.results import Decision, SignalResult
from facepass.session import LivenessSession, SessionState


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def points():
    angle = np.linspace(0, 2 * np.pi, 478)
    face = np.column_stack((0.5 + 0.2 * np.cos(angle), 0.5 + 0.3 * np.sin(angle), angle * 0))
    face[33] = [0.4, 0.4, 0]
    face[263] = [0.6, 0.4, 0]
    face[1] = [0.5, 0.5, 0]
    return face


def good_frame():
    # Deterministic textured patch: sufficient quality does not prove liveness.
    rng = np.random.default_rng(42)
    return rng.integers(70, 190, size=(480, 640, 3), dtype=np.uint8)


class Backend:
    def __init__(self, faces=None):
        self.faces = (points(),) if faces is None else faces
        self.closed = False
        self.error = False
        self.cleanup_error = False

    def detect(self, frame, timestamp):
        if self.error:
            raise RuntimeError("simulated model failure")
        return FaceObservation(self.faces)

    def close(self):
        self.closed = True
        if self.cleanup_error:
            raise RuntimeError("simulated cleanup failure")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = Backend()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.frame = good_frame()
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def process(self):
        self.clock.now += 0.1
        return self.session.process_frame(self.frame, self.clock.now)

    def assert_non_success(self, result):
        self.assertNotEqual(result.decision, Decision.LIVE)
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(all((name == "temporal_consistency" and score.confidence <= 0.5)
                            or (not score.available and score.reliability == 0)
                            for name, score in result.signals.items()))

    def test_quality_is_never_liveness(self):
        for _ in range(20):
            result = self.process()
            self.assertTrue(result.quality.acceptable)
            self.assert_non_success(result)

    def test_multiple_faces_is_terminal_even_if_second_small(self):
        second = points()
        second[:, :2] = second[:, :2] * 0.2 + 0.1
        self.backend.faces = (points(), second)
        result = self.process()
        self.assertEqual(result.reason, "multiple_faces")
        self.assertEqual(result.face_count, 2)
        self.assertTrue(result.terminal)
        self.assertIn("multiple_faces", result.flags)
        self.assertIsNone(self.session.observation)
        self.assert_non_success(result)

    def test_face_enters_leaves_and_returns(self):
        self.backend.faces = ()
        self.assertEqual(self.process().quality.reasons, ("no_face",))
        self.backend.faces = (points(),)
        self.assertTrue(self.process().quality.acceptable)
        self.backend.faces = ()
        result = self.process()
        self.assertIn("tracking_lost", result.flags)
        self.assertIsNone(self.session.observation)
        self.assertIsNone(result.quality.blur_variance)
        self.backend.faces = (points(),)
        self.assert_non_success(self.process())

    def test_dark_bright_blurred_frames_are_quality_failures(self):
        for level, reason in ((0, "underexposure"), (255, "overexposure"), (128, "blur")):
            self.frame = np.full((480, 640, 3), level, dtype=np.uint8)
            result = self.process()
            self.assertEqual(result.decision, Decision.INSUFFICIENT_QUALITY)
            self.assertIn(reason, result.quality.reasons)
            self.assert_non_success(result)

    def test_blur_before_timeout_clears_evidence_without_claiming_detectors_are_missing(self):
        for _ in range(8):
            observed = self.process()
        self.assertTrue(observed.quality.acceptable)
        self.assertGreater(observed.tracking.history_frame_count, 0)
        self.frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        poor = self.process()
        self.assertIn("blur", poor.quality.reasons)
        self.clock.now = 105.1
        result = self.session.current_result()
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "observation_timeout")
        self.assertEqual(result.decision, Decision.INSUFFICIENT_QUALITY)
        self.assertTrue(self.backend.closed)
        self.assertEqual(self.session.history, ())
        self.assertIsNone(self.session.observation)
        self.assertTrue(all(score.diagnostic_information == "quality_not_acceptable"
                            and not score.available and score.reliability == 0
                            for score in result.signals.values()))
        self.assert_non_success(result)
        from facepass.supervisor import validate_result
        validate_result(json.loads(json.dumps(result.to_dict())))

    def test_partial_face_and_side_orientation_fail_quality(self):
        self.backend.faces = (points() - [0.4, 0, 0],)
        self.assertIn("face_clipped", self.process().quality.reasons)
        side = points()
        side[1, 0] = 0.7
        self.backend.faces = (side,)
        self.assertIn("face_orientation", self.process().quality.reasons)

    def test_malformed_frames_fail_closed(self):
        for frame in (None, [], np.zeros((480, 640), np.uint8), np.zeros((480, 640, 3)),
                      np.zeros((0, 640, 3), np.uint8), self.frame[:, ::2]):
            with self.subTest(shape=getattr(frame, "shape", None)):
                session = LivenessSession(lambda: Backend(), clock=self.clock)
                session.start()
                result = session.process_frame(frame, self.clock.now)
                self.assertTrue(result.terminal)
                self.assert_non_success(result)

    def test_nan_and_malformed_landmarks_fail_closed(self):
        for face in (np.full((478, 3), np.nan), np.full((478, 3), np.inf), np.zeros((12, 3)),
                     np.full((478, 3), 20.0), np.zeros((478, 3), dtype=int)):
            with self.subTest(shape=face.shape):
                session = LivenessSession(lambda: Backend((face,)), clock=self.clock)
                session.start()
                result = session.process_frame(self.frame, self.clock.now)
                self.assertTrue(result.terminal)
                self.assert_non_success(result)

    def test_model_failure_is_terminal(self):
        self.backend.error = True
        result = self.process()
        self.assertTrue(self.backend.closed)
        self.assertTrue(result.terminal)
        self.assert_non_success(result)

    def test_model_load_failure(self):
        def broken():
            raise RuntimeError("failed to load model")
        result = LivenessSession(broken).start()
        self.assertEqual(result.reason, "model_initialization_failed")
        self.assert_non_success(result)

    def test_camera_failure_external_report(self):
        self.process()
        result = self.session.fail("camera_read_failed")
        self.assertTrue(result.terminal)
        self.assert_non_success(result)

    def test_bad_timestamps_and_frame_gap(self):
        for timestamp in (float("nan"), float("inf"), True, "100", 99.0, 101.0):
            session = LivenessSession(lambda: Backend(), clock=self.clock)
            session.start()
            result = session.process_frame(self.frame, timestamp)
            self.assertEqual(result.reason, "invalid_timestamp")
            self.assert_non_success(result)
        self.process()
        result = self.session.process_frame(self.frame, self.clock.now)
        self.assertEqual(result.reason, "invalid_timestamp")
        session = LivenessSession(lambda: Backend(), clock=self.clock)
        session.start()
        session.process_frame(self.frame, self.clock.now)
        self.clock.now += 1.1
        self.assertEqual(session.process_frame(self.frame, self.clock.now).reason, "frame_gap")

    def test_timeout_even_without_new_frames(self):
        self.process()
        self.clock.now += 5
        result = self.session.current_result()
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "observation_timeout")
        self.assertTrue(self.backend.closed)
        self.assertIsNone(self.session.observation)
        self.assert_non_success(result)

    def test_cancellation_releases_geometry_and_backend(self):
        self.process()
        self.assertIsNotNone(self.session.observation)
        result = self.session.cancel()
        self.assertEqual(self.session.state, SessionState.CANCELLED)
        self.assertTrue(self.backend.closed)
        self.assertIsNone(self.session.observation)
        self.assert_non_success(result)

    def test_cleanup_failure_invalidates_result(self):
        self.process()
        self.backend.cleanup_error = True
        result = self.session.finish()
        self.assertEqual(result.reason, "model_cleanup_failed")
        self.assert_non_success(result)

    def test_result_has_no_raw_geometry_and_no_mutable_signal_ownership(self):
        result = self.process()
        result.signals.clear()
        current = self.session.current_result()
        self.assertEqual(len(current.signals), 7)
        serialized = json.dumps(current.to_dict(), allow_nan=False)
        self.assertNotIn("face_landmarks", serialized)
        self.assertIsNone(current.quality.landmark_confidence)

    def test_invalid_signal_values_rejected(self):
        for value in (math.nan, math.inf, -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                SignalResult(confidence=value)
        with self.assertRaises(ValueError):
            SignalResult(reliability=0.9, available=False)

    def test_finished_session_cannot_restart_or_reuse_frames(self):
        self.session.finish()
        self.assert_non_success(self.session.start())
        self.assert_non_success(self.session.process_frame(self.frame, self.clock.now))


if __name__ == "__main__":
    unittest.main()
