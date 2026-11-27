"""Controlled region photometry and fail-closed integration, not physical PAD."""

from collections import deque
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from facepass.illumination import IlluminationAnalyzer, extract_photometry, linear_rgb, region_masks, face_patch_centers
from facepass.results import Decision, MotionDiagnostics
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from facepass.landmarks import FaceObservation
from tools.synthetic_illumination import encode_rgb, illumination_frame
from tools.synthetic_motion import gaze_mesh, SyntheticBackend
from tests.test_session import Clock


def still_motion(**changes):
    return replace(MotionDiagnostics(pose_available=True, pitch_degrees=0, yaw_degrees=0, roll_degrees=0,
                                     background_available=True, common_motion_px=0), **changes)


class PhotometryTests(unittest.TestCase):
    def test_inverse_srgb_channel_order_and_round_trip(self):
        rgb = np.array([[[.18, .22, .28], [.001, .002, .003]]])
        np.testing.assert_allclose(linear_rgb(encode_rgb(rgb)), rgb, atol=.002)
        np.testing.assert_allclose(linear_rgb(np.array([[[0, 0, 255]]], np.uint8)), [[[1, 0, 0]]])

    def test_regions_have_support_and_background_excludes_dilated_face(self):
        points = gaze_mesh()
        size, faces, ambient = region_masks(points, (480, 640))
        self.assertEqual(size, (320, 240))
        self.assertEqual((len(faces), len(ambient)), (7, 4))
        hull = np.zeros((240, 320), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull((points[:468, :2] * [320, 240]).astype(np.int32)), 255)
        excluded = cv2.dilate(hull, np.ones((25, 25), np.uint8)) != 0
        for mask in ambient:
            self.assertFalse(np.any(mask & excluded))
            self.assertGreaterEqual(int(mask.sum()), 128)
        for index, mask in enumerate(faces):
            self.assertGreaterEqual(int(mask.sum()), 24)
            for other in faces[index + 1:]:
                self.assertFalse(np.any(mask & other))

    def test_photometry_distinguishes_skin_background_and_seven_regions(self):
        p = extract_photometry(illumination_frame(.3), gaze_mesh())
        self.assertEqual(p.face_rgb.shape, (7, 3))
        self.assertEqual(p.ambient_rgb.shape, (4, 3))
        self.assertGreater(p.face_rgb[:, 0].mean(), p.face_rgb[:, 2].mean())
        self.assertGreater(p.ambient_rgb[:, 2].mean(), p.ambient_rgb[:, 0].mean())
        self.assertGreater(np.ptp(p.face_rgb[:, 0]), .05)

    def test_clipped_dark_and_unsupported_regions_are_unavailable(self):
        points = gaze_mesh()
        for intensity in (0, 255):
            frame = illumination_frame()
            center = face_patch_centers(points, frame.shape)[0][0]
            cv2.circle(frame, tuple(np.rint(center).astype(int)), 15, (intensity,) * 3, -1)
            self.assertIsNone(extract_photometry(frame, points))
        enlarged = points.copy()
        enlarged[:, :2] = (enlarged[:, :2] - .5) * 3 + .5
        self.assertIsNone(region_masks(enlarged, (480, 640)))

    def test_malformed_pixels_and_nonfinite_landmarks_are_errors(self):
        points = gaze_mesh()
        points[1, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_photometry(illumination_frame(), points)
        with self.assertRaises(ValueError):
            extract_photometry(np.zeros((480, 640, 3), np.float32), gaze_mesh())


class IlluminationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = IlluminationAnalyzer()
        self.timestamp = 100.0
        self.history = deque(maxlen=64)

    def update(self, amplitude=0, *, gap=.1, track_id=1, stable=True, frame=None, motion=None, points=None, **light):
        self.timestamp += gap
        points = gaze_mesh() if points is None else points
        self.history.append(TrackedSample(track_id, self.timestamp, points, (480, 640)))
        return self.analyzer.update(illumination_frame(amplitude, points=points, **light) if frame is None else frame,
                                    tuple(self.history), stable=stable, motion=still_motion() if motion is None else motion)

    def response(self, **light):
        for i in range(12):
            stats, score = self.update(i * .035, **light)
        return stats, score

    def test_directional_multiregion_response_is_only_weak_evidence(self):
        stats, score = self.response()
        self.assertTrue(stats.consistent_response_observed)
        self.assertGreaterEqual(stats.changing_face_regions, 4)
        self.assertGreater(stats.regional_response_residual, .012)
        self.assertGreater(stats.response_coherence, .8)
        self.assertEqual((score.confidence, score.reliability), (.58, .20))
        json.dumps(asdict(stats), allow_nan=False)

    def test_static_side_lighting_and_different_face_background_colors_are_neutral(self):
        for _ in range(12):
            stats, score = self.update(0, face_amplitude=.3)
        self.assertTrue(score.available)
        self.assertEqual(score.confidence, .5)
        self.assertFalse(stats.response_mismatch)
        self.assertIsNone(stats.ambient_response_direction_degrees)

    def test_uniform_exposure_change_is_removed_and_never_rewarded(self):
        for i in range(12):
            stats, score = self.update(global_log_gain=i * .025)
        self.assertGreater(stats.global_gain_ev, .3)
        self.assertLess(stats.ambient_spatial_change, .02)
        self.assertFalse(stats.consistent_response_observed)
        self.assertEqual(score.confidence, .5)

    def test_common_white_balance_change_is_removed_and_never_rewarded(self):
        for i in range(12):
            stats, score = self.update(channel_log_gain=(i * .02, -i * .01, i * .005))
        self.assertGreater(stats.common_color_shift, .2)
        self.assertLess(stats.ambient_spatial_change, .02)
        self.assertEqual(score.confidence, .5)

    def test_directional_response_survives_common_gain_and_white_balance(self):
        for i in range(12):
            stats, score = self.update(i * .035, global_log_gain=i * .01,
                                       channel_log_gain=(i * .01, -i * .005, 0))
        self.assertTrue(stats.consistent_response_observed)
        self.assertEqual(score.confidence, .58)

    def test_background_only_and_opposed_response_are_unvalidated_mismatches(self):
        for mode in ("static", "opposed"):
            self.analyzer.reset()
            for i in range(12):
                stats, score = self.update(i * .035, face_amplitude=0 if mode == "static" else -i * .035)
            self.assertTrue(stats.response_mismatch)
            self.assertEqual(score.confidence, .5)
            self.assertFalse(stats.consistent_response_observed)

    def test_planar_regional_gradient_without_richer_response_is_not_rewarded(self):
        stats, score = self.response(nonlinear=False)
        self.assertLess(stats.regional_response_residual, .012)
        self.assertFalse(stats.consistent_response_observed)
        self.assertEqual(score.confidence, .5)

    def test_inconsistent_spectral_response_stays_neutral(self):
        stats, score = self.response(face_color=(.1, .5, 1.4))
        self.assertGreater(stats.color_response_error, .35)
        self.assertTrue(stats.response_mismatch)
        self.assertEqual(score.confidence, .5)

    def test_small_and_short_lighting_changes_cannot_earn_credit(self):
        for i in range(7):
            stats, score = self.update(i * .04)
        self.assertFalse(score.available)
        self.analyzer.reset()
        for i in range(12):
            stats, score = self.update(i * .001)
        self.assertEqual(score.confidence, .5)

    def test_unavailable_scene_motion_context_reduces_reliability(self):
        for i in range(12):
            stats, score = self.update(i * .035, motion=still_motion(background_available=False, common_motion_px=None))
        self.assertFalse(stats.consistent_response_observed)
        self.assertFalse(stats.motion_context_available)
        self.assertEqual((score.confidence, score.reliability), (.5, .03))

    def test_head_or_background_motion_suppresses_response_credit(self):
        for mode in ("head", "background"):
            self.analyzer.reset()
            for i in range(12):
                context = still_motion(yaw_degrees=i * .3) if mode == "head" else still_motion(common_motion_px=i * .3)
                stats, score = self.update(i * .035, motion=context)
            self.assertTrue(stats.motion_suppressed)
            self.assertEqual(score.confidence, .5)

    def test_missing_or_nonfrontal_pose_clears_history(self):
        for context in (still_motion(pose_available=False), still_motion(yaw_degrees=30)):
            self.response()
            stats, score = self.update(motion=context)
            self.assertEqual(stats.event, "illumination_pose_unreliable")
            self.assertFalse(score.available)
            self.assertEqual(len(self.analyzer._samples), 0)

    def test_radiometric_jump_clears_credit_without_lowering_thresholds(self):
        self.response()
        stats, score = self.update(global_log_gain=.6)
        self.assertEqual(stats.event, "illumination_radiometric_jump")
        self.assertFalse(score.available)
        self.assertEqual(len(self.analyzer._samples), 0)

    def test_gap_track_change_unstable_and_loss_do_not_join_old_response(self):
        for cause in ("gap", "track", "unstable", "loss"):
            self.response()
            stats, score = (self.analyzer.update(illumination_frame(), (), stable=True, motion=still_motion()) if cause == "loss"
                            else self.update(gap=.4 if cause == "gap" else .1, track_id=2 if cause == "track" else 1,
                                             stable=cause != "unstable"))
            self.assertFalse(score.available)
            self.assertFalse(stats.consistent_response_observed)
            self.assertLessEqual(len(self.analyzer._samples), 1)

    def test_history_and_polling_freshness_are_bounded(self):
        for _ in range(80):
            stats, _ = self.update(gap=.02)
        self.assertEqual(len(self.analyzer._samples), 64)
        for _ in range(22):
            stats, _ = self.update()
        self.assertLessEqual(stats.window_duration_ms, 2000)
        self.analyzer.current_result(self.timestamp + .2)
        self.assertLessEqual(self.timestamp + .2 - self.analyzer._samples[0][0], 2)
        stats, score = self.analyzer.current_result(self.timestamp + .4)
        self.assertFalse(score.available)
        self.assertEqual(len(self.analyzer._samples), 0)

    def test_old_lighting_response_expires_during_stationary_observation(self):
        self.response()
        for _ in range(25):
            stats, score = self.update(.385)
        self.assertFalse(stats.consistent_response_observed)
        self.assertEqual(score.confidence, .5)

    def test_bad_times_native_errors_and_polling_errors_clear_private_state(self):
        for cause in ("timestamp", "native", "poll", "clock"):
            self.response()
            with self.assertRaises((ValueError, RuntimeError)):
                if cause == "timestamp":
                    self.analyzer.update(illumination_frame(), tuple(self.history), stable=True, motion=still_motion())
                elif cause == "clock":
                    self.analyzer.current_result(float("nan"))
                else:
                    with patch("facepass.illumination.np.linalg.lstsq", side_effect=RuntimeError("native failure")):
                        self.analyzer.current_result(self.timestamp) if cause == "poll" else self.update()
            self.assertEqual(len(self.analyzer._samples), 0)


class IlluminationSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = SyntheticBackend()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def update(self, amplitude=0, **light):
        self.clock.now += .1
        self.backend.points = gaze_mesh()
        result = self.session.process_frame(illumination_frame(amplitude, **light), self.clock.now)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        self.assertNotEqual(result.decision, Decision.LIVE)
        self.assertEqual(result.confidence, 0)
        return result

    def observe(self):
        for i in range(14):
            result = self.update(i * .035)
        self.assertEqual(result.signals["illumination_consistency"].confidence, .58)
        return result

    def test_historical_finish_releases_private_summaries_and_images(self):
        self.observe()
        result = self.session.finish()
        self.assertTrue(result.illumination.historical)
        self.assertFalse(result.signals["illumination_consistency"].available)
        self.assertEqual(result.signals["illumination_consistency"].reliability, 0)
        self.assertEqual(len(self.session._illumination._samples), 0)
        self.assertEqual(len(self.session.history), 0)
        validate_result(json.loads(json.dumps(result.to_dict())))

    def test_face_loss_quality_stale_cancel_failure_multiple_and_discontinuity_clear_evidence(self):
        for cause in ("loss", "quality", "stale", "cancel", "failure", "multiple", "discontinuity"):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            self.observe()
            if cause == "loss":
                self.backend.visible = False
                result = self.update()
            elif cause == "quality":
                self.clock.now += .1
                result = self.session.process_frame(np.zeros((480, 640, 3), np.uint8), self.clock.now)
            elif cause == "stale":
                self.clock.now += .4
                result = self.session.current_result()
            elif cause == "cancel":
                result = self.session.cancel()
            elif cause == "multiple":
                self.clock.now += .1
                with patch.object(self.backend, "detect", return_value=FaceObservation((gaze_mesh(), gaze_mesh()))):
                    result = self.session.process_frame(illumination_frame(), self.clock.now)
            elif cause == "discontinuity":
                self.backend.points = gaze_mesh()
                self.backend.points[:, 0] += .15
                self.clock.now += .1
                result = self.session.process_frame(illumination_frame(), self.clock.now)
                self.assertTrue(result.terminal)
            else:
                result = self.session.fail("camera_read_failed")
            self.assertFalse(result.signals["illumination_consistency"].available)
            self.assertFalse(result.illumination.historical)
            self.assertEqual(result.illumination.window_frame_count, 0)
            self.assertEqual(len(self.session._illumination._samples), 0)

    def test_static_and_uniform_gain_are_not_success_and_temporal_fusion_stays_non_positive(self):
        for i in range(14):
            result = self.update(global_log_gain=i * .015)
        self.assertEqual(result.signals["illumination_consistency"].confidence, .5)
        self.assertLessEqual(result.signals["temporal_consistency"].confidence, 0.5)
        self.assertFalse(result.fusion.liveness_candidate)

    def test_native_update_poll_finish_and_cleanup_failures_release_everything(self):
        for operation in ("update", "poll", "finish", "cleanup"):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            self.observe()
            if operation == "cleanup":
                with patch.object(self.backend, "close", side_effect=RuntimeError("cleanup failure")):
                    result = self.session.finish()
            else:
                with patch.object(self.session._illumination, "_summarize", side_effect=RuntimeError("native failure")):
                    result = self.update() if operation == "update" else self.session.current_result() if operation == "poll" else self.session.finish()
            self.assertTrue(result.terminal)
            self.assertFalse(result.illumination.historical)
            self.assertEqual(result.illumination.window_frame_count, 0)
            self.assertEqual(len(self.session._illumination._samples), 0)
            self.assertEqual(result.confidence, 0)

    def test_real_default_and_faster_cadence_produce_scalar_json(self):
        for rate in (12, 30):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            for i in range(30):
                self.clock.now += 1 / rate
                self.backend.points = gaze_mesh()
                result = self.session.process_frame(illumination_frame(i * .012), self.clock.now)
                validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
                self.assertFalse(result.terminal)
                self.assertIsInstance(result.illumination.consistent_response_observed, bool)

    def test_timeout_after_inference_does_not_refresh_old_illumination(self):
        self.session.finish()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock, max_duration=1.6)
        self.session.start()
        self.observe()
        original = self.backend.detect

        def delayed(frame, timestamp):
            self.clock.now += .4
            return original(frame, timestamp)

        with patch.object(self.backend, "detect", side_effect=delayed):
            result = self.update(.455)
        self.assertTrue(result.terminal)
        self.assertFalse(result.illumination.historical)
        self.assertEqual(result.illumination.window_frame_count, 0)

    def test_strict_ipc_rejects_vectors_nan_context_caps_and_unsupported_success(self):
        base = self.observe().to_dict()
        mutations = (("ambient_red_balance", float("nan")), ("regional_rgb", [[.2, .3, .4]]),
                     ("window_frame_count", None), ("ambient_available", 1), ("ambient_region_count", 3),
                     ("consistent_response_observed", False), ("response_mismatch", True),
                     ("motion_context_available", False), ("motion_suppressed", True),
                     ("ambient_spatial_change", .01), ("regional_response_residual", .001),
                     ("changing_face_regions", 3), ("response_coherence", .5), ("direction_agreement", .5),
                     ("color_response_error", .5), ("background_motion_px", 3), ("head_rotation_span_degrees", 2),
                     ("historical", True), ("status", "not_observed"))
        for key, value in mutations:
            candidate = json.loads(json.dumps(base))
            candidate["illumination"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(candidate)
        for key in ("confidence", "reliability"):
            candidate = json.loads(json.dumps(base))
            candidate["signals"]["illumination_consistency"][key] = 1
            with self.assertRaises(ValueError):
                validate_result(candidate)
        candidate = json.loads(json.dumps(base))
        candidate.update(decision="LIVE", confidence=1)
        with self.assertRaises(ValueError):
            validate_result(candidate)


if __name__ == "__main__":
    unittest.main()
