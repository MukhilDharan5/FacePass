"""Controlled texture diagnostics and fail-closed integration, not physical PAD."""

from collections import deque
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from facepass.landmarks import FaceObservation
from facepass.replay import (FAMILY_NAMES, ReplayArtifactAnalyzer, artifact_regions,
                             extract_artifacts, patch_features)
from facepass.results import Decision, MotionDiagnostics
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import TrackedSample
from tools.synthetic_motion import SyntheticBackend, gaze_mesh
from tools.synthetic_replay import replay_frame
from tests.test_session import Clock


def still_motion(**changes):
    return replace(MotionDiagnostics(pose_available=True, pitch_degrees=0, yaw_degrees=0, roll_degrees=0,
                                     background_available=True, common_motion_px=0), **changes)


class ArtifactFeatureTests(unittest.TestCase):
    def test_regions_have_equal_support_and_background_excludes_face_margin(self):
        points = gaze_mesh()
        face, background, width = artifact_regions(replay_frame(), points)
        self.assertEqual((len(face), len(background), width), (5, 4, 25))
        self.assertTrue(all(patch.shape == (32, 32, 3) for patch in (*face, *background)))
        # Brightening only the face must not leak into the outside-face controls.
        base = extract_artifacts(replay_frame(), points)
        bright = replay_frame()
        hull = np.zeros(bright.shape[:2], np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull((points[:468, :2] * [640, 480]).astype(np.int32)), 255)
        bright[hull != 0] = np.clip(bright[hull != 0].astype(int) + 30, 0, 255)
        changed = extract_artifacts(bright.astype(np.uint8), points)
        np.testing.assert_allclose(base.background_scores, changed.background_scores)

    def test_patch_features_are_bounded_and_separate_pattern_families(self):
        natural = artifact_regions(replay_frame(), gaze_mesh())[0][0]
        patterned = artifact_regions(replay_frame("display_grid"), gaze_mesh())[0][0]
        a, contrast = patch_features(natural)
        b, _ = patch_features(patterned)
        self.assertEqual(a.shape, (7,))
        self.assertTrue(np.all((a >= 0) & (a <= 1)))
        self.assertGreater(contrast, .01)
        self.assertGreater(b[FAMILY_NAMES.index("periodic")], a[FAMILY_NAMES.index("periodic")])
        self.assertGreater(b[FAMILY_NAMES.index("chroma_grid")], a[FAMILY_NAMES.index("chroma_grid")])
        self.assertGreater(b[FAMILY_NAMES.index("resampling")], a[FAMILY_NAMES.index("resampling")])

    def test_face_and_background_artifacts_are_kept_separate(self):
        display = extract_artifacts(replay_frame("display_grid"), gaze_mesh())
        background = extract_artifacts(replay_frame("background_grid"), gaze_mesh())
        self.assertGreater(np.median(display.face_scores[:, 0]), np.median(display.background_scores[:, 0]))
        self.assertGreater(np.max(background.background_scores), np.max(np.median(background.face_scores, axis=0)))

    def test_low_texture_and_invalid_inputs_are_unavailable_or_errors(self):
        self.assertIsNone(extract_artifacts(replay_frame("low_texture"), gaze_mesh()))
        small = gaze_mesh()
        small[:, :2] = (small[:, :2] - .5) * .2 + .5
        self.assertIsNone(artifact_regions(replay_frame(), small))
        bad = gaze_mesh()
        bad[1, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_artifacts(replay_frame(), bad)
        with self.assertRaises(ValueError):
            extract_artifacts(np.zeros((480, 640, 3), np.float32), gaze_mesh())


class ReplayAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = ReplayArtifactAnalyzer()
        self.timestamp = 100.0
        self.history = deque(maxlen=64)

    def update(self, kind="natural_texture", *, gap=.1, track_id=1, stable=True, motion=None, points=None):
        self.timestamp += gap
        points = gaze_mesh() if points is None else points
        self.history.append(TrackedSample(track_id, self.timestamp, points, (480, 640)))
        return self.analyzer.update(replay_frame(kind, points=points), tuple(self.history), stable=stable,
                                    motion=still_motion() if motion is None else motion)

    def observe(self, kind="display_grid", **options):
        for _ in range(8):
            stats, score = self.update(kind, **options)
        return stats, score

    def test_display_and_print_patterns_supply_only_weak_suspicion(self):
        for kind in ("display_grid", "print_halftone"):
            self.analyzer.reset()
            stats, score = self.observe(kind)
            self.assertTrue(stats.replay_artifact_observed)
            self.assertGreaterEqual(stats.artifact_family_count, 2)
            self.assertGreaterEqual(stats.affected_face_regions, 3)
            self.assertGreaterEqual(stats.artifact_persistence, .75)
            self.assertEqual((score.confidence, score.reliability), (.35, .25))
            json.dumps(asdict(stats), allow_nan=False)

    def test_natural_texture_and_single_local_glare_stay_neutral(self):
        for kind in ("natural_texture", "localized_glare", "global_sharpening"):
            self.analyzer.reset()
            stats, score = self.observe(kind)
            self.assertFalse(stats.replay_artifact_observed)
            self.assertEqual((score.confidence, score.reliability), (.5, .10))

    def test_global_camera_artifacts_and_background_pattern_stay_neutral(self):
        for kind in ("global_sensor_banding", "background_grid"):
            self.analyzer.reset()
            stats, score = self.observe(kind)
            self.assertGreater(stats.background_artifact_score, .45)
            self.assertFalse(stats.replay_artifact_observed)
            self.assertEqual(score.confidence, .5)

    def test_same_generated_pattern_cannot_reveal_its_origin(self):
        suspicious, score = self.observe("matching_bona_fide_pattern")
        self.assertTrue(suspicious.replay_artifact_observed)
        self.assertEqual(score.confidence, .35)

    def test_short_or_textureless_observations_are_unavailable(self):
        for _ in range(5):
            stats, score = self.update("display_grid")
        self.assertFalse(score.available)
        stats, score = self.update("low_texture")
        self.assertFalse(score.available)
        self.assertEqual(stats.event, "replay_regions_unreliable")
        self.assertEqual(len(self.analyzer._samples), 0)

    def test_missing_motion_context_and_motion_suppress_suspicion(self):
        for context in (still_motion(background_available=False, common_motion_px=None),
                        still_motion(common_motion_px=4)):
            self.analyzer.reset()
            stats, score = self.observe("display_grid", motion=context)
            self.assertFalse(stats.replay_artifact_observed)
            self.assertEqual((score.confidence, score.reliability), (.5, .03))

    def test_gap_track_change_unstable_and_loss_do_not_join_old_artifacts(self):
        for cause in ("gap", "track", "unstable", "loss"):
            self.observe()
            stats, score = (self.analyzer.update(replay_frame(), (), stable=True, motion=still_motion()) if cause == "loss"
                            else self.update("display_grid", gap=.4 if cause == "gap" else .1,
                                             track_id=2 if cause == "track" else 1,
                                             stable=cause != "unstable"))
            self.assertFalse(score.available)
            self.assertFalse(stats.replay_artifact_observed)
            self.assertLessEqual(len(self.analyzer._samples), 1)

    def test_history_polling_and_artifact_expiry_are_bounded(self):
        for _ in range(80):
            stats, _ = self.update("display_grid", gap=.02)
        self.assertEqual(len(self.analyzer._samples), 64)
        for _ in range(24):
            stats, score = self.update("natural_texture")
        self.assertLessEqual(stats.window_duration_ms, 2000)
        self.assertFalse(stats.replay_artifact_observed)
        self.assertEqual(score.confidence, .5)
        stats, score = self.analyzer.current_result(self.timestamp + .4)
        self.assertFalse(score.available)
        self.assertEqual(len(self.analyzer._samples), 0)

    def test_bad_times_and_native_update_poll_errors_clear_private_state(self):
        for cause in ("timestamp", "clock", "update", "poll"):
            self.observe()
            with self.assertRaises((ValueError, RuntimeError)):
                if cause == "timestamp":
                    self.analyzer.update(replay_frame(), tuple(self.history), stable=True, motion=still_motion())
                elif cause == "clock":
                    self.analyzer.current_result(float("nan"))
                elif cause == "update":
                    with patch("facepass.replay.np.fft.fft2", side_effect=RuntimeError("native failure")):
                        self.update()
                else:
                    with patch.object(self.analyzer, "_summarize", side_effect=RuntimeError("native failure")):
                        self.analyzer.current_result(self.timestamp)
            self.assertEqual(len(self.analyzer._samples), 0)


class ReplaySessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = SyntheticBackend()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def update(self, kind="display_grid"):
        self.clock.now += .1
        self.backend.points = gaze_mesh()
        result = self.session.process_frame(replay_frame(kind), self.clock.now)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        self.assertNotEqual(result.decision, Decision.LIVE)
        self.assertNotEqual(result.decision, Decision.SPOOF)
        self.assertEqual(result.confidence, 0)
        return result

    def observe(self):
        for _ in range(10):
            result = self.update()
        self.assertEqual(result.signals["replay_artifact_detection"].confidence, .35)
        return result

    def test_historical_finish_releases_private_summaries_and_images(self):
        self.observe()
        result = self.session.finish()
        self.assertTrue(result.replay.historical)
        self.assertFalse(result.signals["replay_artifact_detection"].available)
        self.assertEqual(result.signals["replay_artifact_detection"].reliability, 0)
        self.assertEqual(len(self.session._replay._samples), 0)
        self.assertEqual(len(self.session.history), 0)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))

    def test_loss_quality_stale_cancel_failure_multiple_and_discontinuity_clear_evidence(self):
        for cause in ("loss", "quality", "stale", "cancel", "failure", "multiple", "discontinuity"):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            self.observe()
            if cause == "loss":
                self.backend.visible = False
                result = self.update("natural_texture")
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
                    result = self.session.process_frame(replay_frame(), self.clock.now)
            elif cause == "discontinuity":
                self.backend.points = gaze_mesh()
                self.backend.points[:, 0] += .15
                self.clock.now += .1
                result = self.session.process_frame(replay_frame(), self.clock.now)
            else:
                result = self.session.fail("camera_read_failed")
            self.assertFalse(result.signals["replay_artifact_detection"].available)
            self.assertFalse(result.replay.historical)
            self.assertEqual(result.replay.window_frame_count, 0)
            self.assertEqual(len(self.session._replay._samples), 0)

    def test_temporal_consistency_is_bounded_and_neutral_without_two_groups(self):
        result = self.observe()
        self.assertLessEqual(result.signals["temporal_consistency"].confidence, 0.5)
        self.assertIsInstance(result.temporal.conflict_observed, bool)
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
                with patch.object(self.session._replay, "_summarize", side_effect=RuntimeError("native failure")):
                    result = self.update() if operation == "update" else self.session.current_result() if operation == "poll" else self.session.finish()
            self.assertTrue(result.terminal)
            self.assertFalse(result.replay.historical)
            self.assertEqual(result.replay.window_frame_count, 0)
            self.assertEqual(len(self.session._replay._samples), 0)
            self.assertEqual(result.confidence, 0)

    def test_default_and_faster_cadence_produce_scalar_json(self):
        for rate in (12, 30):
            self.session.finish()
            self.backend = SyntheticBackend()
            self.session = LivenessSession(lambda: self.backend, clock=self.clock)
            self.session.start()
            for _ in range(25):
                self.clock.now += 1 / rate
                self.backend.points = gaze_mesh()
                result = self.session.process_frame(replay_frame("display_grid"), self.clock.now)
                validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
                self.assertIsInstance(result.replay.replay_artifact_observed, bool)

    def test_timeout_after_inference_does_not_refresh_old_replay_result(self):
        self.session.finish()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock, max_duration=1.2)
        self.session.start()
        self.observe()
        original = self.backend.detect

        def delayed(frame, timestamp):
            self.clock.now += .4
            return original(frame, timestamp)

        with patch.object(self.backend, "detect", side_effect=delayed):
            result = self.update()
        self.assertTrue(result.terminal)
        self.assertFalse(result.replay.historical)
        self.assertEqual(result.replay.window_frame_count, 0)

    def test_strict_ipc_rejects_vectors_nan_context_caps_and_unsupported_decisions(self):
        base = self.observe().to_dict()
        mutations = (("periodic_peak_score", float("nan")), ("regional_spectra", [[.2, .3]]),
                     ("window_frame_count", None), ("regions_available", 1),
                     ("face_region_count", 4), ("background_region_count", 3),
                     ("replay_artifact_observed", False), ("motion_context_available", False),
                     ("motion_suppressed", True), ("affected_face_regions", 2),
                     ("artifact_family_count", 1), ("artifact_persistence", .5),
                     ("background_artifact_score", .8), ("face_motion_px", 4),
                     ("historical", True), ("status", "not_observed"))
        for key, value in mutations:
            candidate = json.loads(json.dumps(base))
            candidate["replay"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(candidate)
        for key in ("confidence", "reliability"):
            candidate = json.loads(json.dumps(base))
            candidate["signals"]["replay_artifact_detection"][key] = 1
            with self.assertRaises(ValueError):
                validate_result(candidate)
        for decision in ("LIVE", "SPOOF"):
            candidate = json.loads(json.dumps(base))
            candidate.update(decision=decision, confidence=1)
            with self.assertRaises(ValueError):
                validate_result(candidate)


if __name__ == "__main__":
    unittest.main()
