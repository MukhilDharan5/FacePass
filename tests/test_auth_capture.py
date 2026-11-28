"""Camera-worker integration and raw low-light/quality boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from facepass.auth_capture import AuthFrame, CaptureVerifier, capture_attempt
from facepass.auth_challenge import ActiveChallenge
from tests.test_authentication import Clock, FixedRandom, vector
from tests.test_session import Backend, good_frame, points


class CaptureTests(unittest.TestCase):
    def attempt(self, mode="authenticate", *, change=None, no_face=False):
        clock = Clock()
        backend = Backend()
        class Camera:
            closed = False
            def read(self):
                clock.value += .1
                return np.zeros((480, 640, 3), np.uint8)
            def close(self):
                self.closed = True
        camera = Camera()
        class Identity:
            def feature(self, *_):
                return vector()
        class Store:
            def read(self, uid):
                return {"templates": [vector()] * 8}
        class ScriptedObservation:
            def __init__(self, *_):
                self.count = 0
            def process(self, frame, timestamp):
                self.count += 1
                count = self.count
                blinks = 0 if count < 14 else 2 if count < 28 else 4
                yaw = -18 if 16 <= count <= 20 else 18 if 30 <= count <= 34 else 0
                value = AuthFrame(timestamp, 1, True, yaw, blinks, True, vector())
                if no_face:
                    return AuthFrame(timestamp)
                if change and count == 25:
                    setattr(value, *change)
                return value
        with patch("facepass.auth_capture.CaptureVerifier", ScriptedObservation), \
             patch("facepass.auth_capture.ActiveChallenge", lambda c: ActiveChallenge(c, rng=FixedRandom())), \
             patch("facepass.auth_capture.time.sleep"):
            value = capture_attempt(mode, 1000, clock=clock, store=Store(),
                                    camera_factory=lambda *_: camera, backend_factory=lambda *_: backend,
                                    identity_factory=lambda *_: Identity())
        self.assertTrue(camera.closed)
        self.assertTrue(backend.closed)
        return value

    def test_real_challenge_can_complete_authentication_and_enrollment(self):
        value = self.attempt()
        self.assertTrue(value["success"])
        self.assertEqual(value["reason"], "identity_and_active_challenge_verified")
        self.assertNotIn("templates", value)
        enrollment = self.attempt("enroll")
        self.assertTrue(enrollment["success"])
        self.assertGreaterEqual(len(enrollment["templates"]), 8)
        self.assertLessEqual(len(enrollment["templates"]), 12)

    def test_identity_switch_face_loss_quality_loss_or_track_switch_fail(self):
        for change in (("feature", vector(1)), ("feature", None), ("stable", False), ("track_id", 2)):
            with self.subTest(change=change[0]):
                self.assertFalse(self.attempt(change=change)["success"])

    def test_no_face_times_out_and_closes_camera(self):
        self.assertEqual(self.attempt(no_face=True)["reason"], "face_or_identity_unavailable")

    def test_camera_initialization_failure_closes_landmarker(self):
        backend = Backend()
        def fail(*_):
            raise OSError("camera unavailable")
        with self.assertRaises(OSError):
            capture_attempt("enroll", 1000, backend_factory=lambda _: backend,
                            identity_factory=lambda _: object(), camera_factory=fail)
        self.assertTrue(backend.closed)

    def test_multiple_faces_are_rejected_before_identity(self):
        class Identity:
            def feature(self, *_):
                raise AssertionError("must not run")
        verifier = CaptureVerifier(Backend((points(), points())), Identity())
        with self.assertRaisesRegex(ValueError, "multiple_faces"):
            verifier.process(good_frame(), 1)

    def test_lowlight_is_enhanced_for_detection_and_original_used_for_identity(self):
        raw = (good_frame().astype(float) * .25).astype(np.uint8)
        original = raw.copy()
        seen = []
        class Detector(Backend):
            def detect(self, image, timestamp):
                seen.append(image.copy())
                return super().detect(image, timestamp)
        identities = []
        class Identity:
            def feature(self, image, mesh):
                identities.append(image.copy())
                return vector()
        verifier = CaptureVerifier(Detector(), Identity())
        for index in range(8):
            verifier.process(raw, 1 + index * .1)
        self.assertTrue(identities)
        self.assertGreater(seen[0].mean(), raw.mean())
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(identities[0], original)
        result = verifier.process(np.zeros_like(raw), 2)
        self.assertFalse(result.stable)
        self.assertIsNone(result.feature)


if __name__ == "__main__":
    unittest.main()
