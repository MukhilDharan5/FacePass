"""Synthetic usability regressions; no real face images or skin-tone claims."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from facepass.quality import assess_quality, prepare_landmark_frame, quality_policy_for
from facepass.results import Decision, TrackingState
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from tests.test_session import Backend, Clock, good_frame, points


def softened_frame(gain=1.0, sigma=1.5):
    image = cv2.GaussianBlur(good_frame(), (0, 0), sigma)
    return (image.astype(np.float64) * gain).astype(np.uint8)


class QualityTests(unittest.TestCase):
    def test_moderate_blur_is_a_warning_and_strict_retains_original_gate(self):
        frame = softened_frame()
        tolerant = assess_quality(frame, points())
        strict = assess_quality(frame, points(), quality_policy_for("strict"))
        self.assertTrue(tolerant.acceptable)
        self.assertIn("moderate_blur", tolerant.warnings)
        self.assertLess(tolerant.blur_variance, 45)
        self.assertFalse(strict.acceptable)
        self.assertIn("blur", strict.reasons)

    def test_dim_detail_is_usable_across_exposure_levels(self):
        for gain in (.15, .25, .5, 1.0):
            with self.subTest(gain=gain):
                frame = (good_frame().astype(np.float64) * gain).astype(np.uint8)
                q = assess_quality(frame, points())
                self.assertTrue(q.acceptable, q.reasons)
                if gain <= .25:
                    self.assertLess(q.mean_luminance, 40)
                    self.assertIn("low_luminance", q.warnings)
                    self.assertIn("underexposure", assess_quality(frame, points(), quality_policy_for("strict")).reasons)

    def test_dim_and_moderately_blurred_detail_can_be_used_together(self):
        q = assess_quality(softened_frame(.25), points())
        self.assertTrue(q.acceptable, q.reasons)
        self.assertIn("low_luminance", q.warnings)
        self.assertIn("moderate_blur", q.warnings)
        self.assertGreater(q.evaluated_blur_variance, q.blur_variance)
        self.assertLessEqual(q.evaluated_blur_variance, q.blur_variance * 16)

    def test_lower_resolution_compressed_capture_with_detail_is_usable(self):
        reduced = cv2.resize(good_frame(), (320, 240), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", reduced, [cv2.IMWRITE_JPEG_QUALITY, 25])
        self.assertTrue(ok)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        for frame in (compressed, (compressed.astype(float) * .25).astype(np.uint8)):
            q = assess_quality(frame, points())
            self.assertTrue(q.acceptable, q.reasons)

    def test_bright_background_does_not_disqualify_a_dim_face(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        dim = (good_frame().astype(np.float64) * .2).astype(np.uint8)
        frame[96:385, 192:449] = dim[96:385, 192:449]
        q = assess_quality(frame, points())
        self.assertTrue(q.acceptable, q.reasons)
        self.assertIn("low_luminance", q.warnings)

    def test_smaller_but_resolved_face_is_usable(self):
        mesh = points()
        mesh[:, :2] = .5 + (mesh[:, :2] - .5) * .42
        q = assess_quality(good_frame(), mesh)
        self.assertTrue(q.acceptable, q.reasons)
        self.assertIn("small_face", q.warnings)
        self.assertIn("face_too_small", assess_quality(good_frame(), mesh, quality_policy_for("strict")).reasons)
        mesh[:, :2] = .5 + (mesh[:, :2] - .5) * .5
        self.assertIn("face_too_small", assess_quality(good_frame(), mesh).reasons)

    def test_severe_blur_and_missing_contrast_still_fail(self):
        for frame in (softened_frame(sigma=6), softened_frame(.25, sigma=4),
                      np.full((480, 640, 3), 128, np.uint8)):
            with self.subTest(mean=float(frame.mean())):
                q = assess_quality(frame, points())
                self.assertFalse(q.acceptable)
                self.assertIn("low_contrast", q.reasons)

    def test_black_white_and_near_black_still_fail(self):
        frames = [(np.zeros((480, 640, 3), np.uint8), "underexposure"),
                  (np.full((480, 640, 3), 255, np.uint8), "overexposure"),
                  ((good_frame().astype(float) * .04).astype(np.uint8), "underexposure")]
        for frame, reason in frames:
            q = assess_quality(frame, points())
            self.assertFalse(q.acceptable)
            self.assertIn(reason, q.reasons)

    def test_geometry_gates_are_retained(self):
        self.assertIn("face_clipped", assess_quality(good_frame(), points() - [.4, 0, 0]).reasons)
        mesh = points()
        mesh[1, 0] = .7
        self.assertIn("face_orientation", assess_quality(good_frame(), mesh).reasons)

    def test_low_light_enhancement_preserves_original_and_full_frame(self):
        frame = softened_frame(.25)
        original = frame.copy()
        output, enhanced = prepare_landmark_frame(frame)
        self.assertTrue(enhanced)
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.shares_memory(frame, output))
        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(output.flags.c_contiguous)
        self.assertGreater(output.mean(), frame.mean())
        self.assertTrue(np.all(output.astype(int) <= frame.astype(int) * 3))

    def test_enhancement_is_skipped_for_strict_bright_and_flat_frames(self):
        for frame, policy in ((softened_frame(.25), quality_policy_for("strict")),
                              (good_frame(), quality_policy_for("tolerant")),
                              (np.full((480, 640, 3), 25, np.uint8), quality_policy_for("tolerant")),
                              (np.zeros((480, 640, 3), np.uint8), quality_policy_for("tolerant"))):
            output, enhanced = prepare_landmark_frame(frame, policy)
            self.assertFalse(enhanced)
            self.assertIs(output, frame)

    def test_gamma_is_monotonic_and_does_not_brighten_black_or_clip_white(self):
        frame = softened_frame(.25)
        frame[0, :256] = np.arange(256, dtype=np.uint8)[:, None]
        output, enhanced = prepare_landmark_frame(frame)
        self.assertTrue(enhanced)
        self.assertTrue(np.all(np.diff(output[0, :256, 0].astype(int)) >= 0))
        self.assertEqual(int(output[0, 0, 0]), 0)
        self.assertEqual(int(output[0, 255, 0]), 255)

    def test_quality_diagnostics_validate_and_reject_malformed_warnings(self):
        from facepass.results import failure_result
        value = json.loads(json.dumps(failure_result("test")))
        value["quality"] = json.loads(json.dumps(assess_quality(softened_frame(.25), points()).__dict__))
        validate_result(value)
        for key, invalid in (("profile", "unknown"), ("landmark_enhanced", 1),
                             ("warnings", ["private face text"]), ("contrast_span", float("nan")),
                             ("evaluated_blur_variance", float("inf"))):
            with self.subTest(key=key):
                bad = json.loads(json.dumps(value))
                bad["quality"][key] = invalid
                with self.assertRaises(ValueError):
                    validate_result(bad)


class TolerantSessionTests(unittest.TestCase):
    def run_frames(self, frames, profile="tolerant", backend=None):
        clock, backend = Clock(), backend or Backend()
        session = LivenessSession(lambda: backend, clock=clock, quality_profile=profile)
        session.start()
        results = []
        try:
            for frame in frames:
                clock.now += .1
                result = session.process_frame(frame, clock.now)
                validate_result(json.loads(json.dumps(result.to_dict())))
                results.append(result)
            return results
        finally:
            session.finish()

    def test_moderate_blur_keeps_the_same_track_and_history(self):
        results = self.run_frames([softened_frame(sigma=1)] * 4 + [softened_frame()] * 4)
        self.assertTrue(all(r.quality.acceptable for r in results))
        self.assertEqual(results[-1].tracking.state, TrackingState.STABLE)
        self.assertEqual(results[-1].tracking.track_id, results[0].tracking.track_id)
        self.assertEqual(results[-1].tracking.reset_count, 0)
        self.assertEqual(results[-1].tracking.history_frame_count, 8)
        self.assertTrue(all(r.decision == Decision.UNCERTAIN and r.confidence == 0 for r in results))

    def test_dim_capture_reaches_stable_and_analysis_receives_original(self):
        frame = softened_frame(.25)
        backend = Backend()
        original_detect = backend.detect
        model_means = []

        def detect(image, timestamp):
            model_means.append(float(image.mean()))
            return original_detect(image, timestamp)

        backend.detect = detect
        raw_assess = assess_quality
        with patch("facepass.session.assess_quality", wraps=raw_assess) as quality, \
                patch("facepass.session.IlluminationAnalyzer.update", autospec=True) as lighting, \
                patch("facepass.session.ReplayArtifactAnalyzer.update", autospec=True) as replay:
            from facepass.results import IlluminationDiagnostics, ReplayDiagnostics, SignalResult
            lighting.return_value = (IlluminationDiagnostics(), SignalResult())
            replay.return_value = (ReplayDiagnostics(), SignalResult())
            results = self.run_frames([frame] * 4, backend=backend)
            self.assertTrue(all(call.args[0] is frame for call in quality.call_args_list))
            self.assertTrue(all(call.args[1] is frame for call in lighting.call_args_list))
            self.assertTrue(all(call.args[1] is frame for call in replay.call_args_list))
        self.assertTrue(all(mean > frame.mean() for mean in model_means))
        self.assertEqual(results[-1].tracking.state, TrackingState.STABLE)
        self.assertTrue(results[-1].quality.landmark_enhanced)
        self.assertIn("landmark_low_light_enhanced", results[-1].flags)
        self.assertEqual(results[-1].confidence, 0)

    def test_strict_mode_disables_enhancement_and_rejects_dim_capture(self):
        results = self.run_frames([softened_frame(.25)], "strict")
        self.assertEqual(results[0].quality.profile, "strict")
        self.assertFalse(results[0].quality.landmark_enhanced)
        self.assertEqual(results[0].decision, Decision.INSUFFICIENT_QUALITY)

    def test_severe_quality_loss_still_clears_track_and_evidence(self):
        results = self.run_frames([softened_frame()] * 4 + [np.zeros((480, 640, 3), np.uint8)])
        self.assertEqual(results[-2].tracking.state, TrackingState.STABLE)
        self.assertEqual(results[-1].decision, Decision.INSUFFICIENT_QUALITY)
        self.assertEqual(results[-1].tracking.history_frame_count, 0)
        self.assertTrue(all(not s.available and s.reliability == 0 for s in results[-1].signals.values()))

    def test_dim_multiple_face_capture_still_terminates(self):
        results = self.run_frames([softened_frame(.25)], backend=Backend((points(), points())))
        self.assertTrue(results[0].terminal)
        self.assertEqual(results[0].reason, "multiple_faces")


if __name__ == "__main__":
    unittest.main()
