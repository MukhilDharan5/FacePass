"""Continuity regressions use synthetic frames/meshes, not collected biometrics."""

from dataclasses import replace
import json
import unittest

import numpy as np

from facepass.results import Decision, SignalResult, TrackingState
from facepass.session import LivenessSession
from facepass.supervisor import validate_result
from facepass.tracking import ANCHORS, FaceTracker, TrackingPolicy
from tests.test_session import Backend, Clock, good_frame, points


def textured_frame():
    y, x = np.mgrid[:480, :640]
    # Smooth structure survives a 48-pixel alignment; fine texture passes quality.
    gray = (125 + 35 * np.sin(x / 19) + 28 * np.cos(y / 17) + 9 * np.sin(x * 1.5)).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = FaceTracker()
        self.frame = textured_frame()
        self.points = points()

    def update(self, timestamp, points_=None, frame=None):
        return self.tracker.update(self.frame if frame is None else frame,
                                   self.points if points_ is None else points_, timestamp)

    def test_still_face_stabilizes_without_blink_or_motion(self):
        first = self.update(100)
        self.assertEqual(first.state, TrackingState.ACQUIRING)
        self.assertEqual(self.update(100.1).state, TrackingState.ACQUIRING)
        stable = self.update(100.2)
        self.assertEqual(stable.state, TrackingState.STABLE)
        self.assertEqual(stable.track_id, first.track_id)
        self.assertEqual(stable.continuous_frames, 3)
        self.assertEqual(stable.event, "track_stabilized")

    def test_high_fps_does_not_stabilize_from_three_near_simultaneous_frames(self):
        for timestamp in (100, 100.01, 100.02):
            self.assertEqual(self.update(timestamp).state, TrackingState.ACQUIRING)
        self.assertEqual(self.update(100.1499).state, TrackingState.ACQUIRING)
        self.assertEqual(self.update(100.2).state, TrackingState.STABLE)

    def test_small_landmark_jitter_keeps_track_and_unsmoothed_history(self):
        self.frame = good_frame()  # Insufficient aligned texture is gracefully unavailable.
        first = self.update(100)
        rng = np.random.default_rng(91)
        for index in range(1, 15):
            jittered = self.points + rng.normal(0, 0.0005, self.points.shape)
            result = self.update(100 + index * 0.1, jittered)
            self.assertEqual(result.track_id, first.track_id)
            np.testing.assert_array_equal(self.tracker.history[-1].landmarks, jittered)
        self.assertEqual(result.state, TrackingState.STABLE)

    def test_gradual_translation_scale_and_roll_keep_geometric_continuity(self):
        # Uniform pixels deliberately make appearance unavailable. This checks
        # geometry tolerance, not PAD performance or tracking confidence.
        self.frame = np.full((480, 640, 3), 128, np.uint8)
        first = self.update(100)
        center = np.array([320, 240])
        for index in range(1, 10):
            angle = index * 0.003
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            moved = self.points.copy()
            moved[:, :2] = (((moved[:, :2] * [640, 480] - center) @ rotation)
                             * (1 + index * 0.004) + center + [index * 0.5, index * 0.2]) / [640, 480]
            result = self.update(100 + index * 0.1, moved)
            self.assertEqual(result.track_id, first.track_id)
        self.assertEqual(result.state, TrackingState.STABLE)
        self.assertLessEqual(result.continuity_score, 0.5)

    def test_blink_and_iris_motion_do_not_break_tracking(self):
        self.update(100)
        changed = self.points.copy()
        non_anchors = [i for i in (159, 145, 386, 374, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477)
                       if i not in ANCHORS]
        changed[non_anchors, :2] += [0.001, 0.005]
        self.update(100.1, changed)
        result = self.update(100.2)
        self.assertEqual(result.state, TrackingState.STABLE)
        np.testing.assert_array_equal(self.tracker.history[1].landmarks, changed)

    def test_abrupt_position_and_scale_changes_reject_and_clear_history(self):
        for kind in ("position", "scale"):
            with self.subTest(kind=kind):
                self.tracker = FaceTracker()
                self.frame = np.full((480, 640, 3), 128, np.uint8)
                self.update(100)
                changed = self.points.copy()
                if kind == "position":
                    changed[:, 0] += 0.25
                else:
                    changed[:, :2] = (changed[:, :2] - 0.5) * 1.6 + 0.5
                result = self.update(100.1, changed)
                self.assertEqual(result.state, TrackingState.REJECTED)
                self.assertEqual(result.event, f"{kind}_discontinuity")
                self.assertEqual(self.tracker.history, ())
                self.assertEqual(result.continuity_score, 0)
                with self.assertRaises(ValueError):
                    self.update(100.2)
                with self.assertRaises(ValueError):
                    self.tracker.reset("no_face")

    def test_same_position_shape_substitution_rejects(self):
        self.update(100)
        changed = self.points.copy()
        selected = list(ANCHORS[6:])
        # Keep the bounding box and eye/nose appearance alignment unchanged.
        selected = [i for i in selected if i not in (1, 33, 263)]
        changed[selected, :2] = 1 - changed[selected, :2]
        result = self.update(100.1, changed)
        self.assertEqual(result.state, TrackingState.REJECTED)
        self.assertEqual(result.event, "shape_discontinuity")
        self.assertEqual(self.tracker.history, ())

    def test_same_geometry_appearance_substitution_rejects(self):
        first = self.update(100)
        self.assertTrue(first.appearance_available)
        changed = (255 - self.frame).astype(np.uint8)
        result = self.update(100.1, frame=changed)
        self.assertEqual(result.state, TrackingState.REJECTED)
        self.assertEqual(result.event, "appearance_discontinuity")
        self.assertGreater(result.appearance_change, 0.4)
        self.assertEqual(self.tracker.history, ())

    def test_uniform_brightness_and_contrast_changes_do_not_break_track(self):
        self.update(100)
        changed = np.clip(self.frame.astype(float) * 0.8 + 25, 0, 255).astype(np.uint8)
        self.update(100.1, frame=changed)
        result = self.update(100.2, frame=changed)
        self.assertEqual(result.state, TrackingState.STABLE)
        self.assertLess(result.appearance_change, 0.02)

    def test_gap_restarts_track_without_bridging_history(self):
        first = self.update(100)
        self.update(100.1)
        result = self.update(100.6)
        self.assertNotEqual(result.track_id, first.track_id)
        self.assertEqual(result.state, TrackingState.ACQUIRING)
        self.assertEqual(result.reset_reason, "frame_gap")
        self.assertTrue(result.history_reset)
        self.assertEqual(len(self.tracker.history), 1)

    def test_expiry_without_new_frames_clears_history(self):
        self.update(100)
        result = self.tracker.expire(100.5)
        self.assertEqual(result.state, TrackingState.LOST)
        self.assertEqual(result.event, "tracking_stale")
        self.assertEqual(self.tracker.history, ())

    def test_frame_resolution_change_rejects(self):
        self.update(100)
        result = self.update(100.1, frame=self.frame[::2, ::2].copy())
        self.assertEqual(result.state, TrackingState.REJECTED)
        self.assertEqual(result.event, "frame_shape_changed")
        self.assertEqual(self.tracker.history, ())

    def test_history_is_bounded_by_time_and_frame_count(self):
        self.tracker = FaceTracker(TrackingPolicy(history_frames=8, history_seconds=0.5))
        for index in range(40):
            self.update(100 + index * 0.03)
            self.assertLessEqual(len(self.tracker.history), 8)
        history = self.tracker.history
        self.assertLessEqual(history[-1].timestamp - history[0].timestamp, 0.5)
        self.tracker = FaceTracker(TrackingPolicy(history_seconds=0.3))
        for index in range(20):
            self.update(100 + index * 0.1)
        history = self.tracker.history
        self.assertLessEqual(history[-1].timestamp - history[0].timestamp, 0.3)
        self.assertEqual(self.tracker.result.continuous_frames, 20)

    def test_history_owns_immutable_copy_of_backend_points(self):
        self.update(100)
        retained = self.tracker.history[-1].landmarks
        expected = retained.copy()
        self.points[:] = 0
        np.testing.assert_array_equal(retained, expected)
        with self.assertRaises(ValueError):
            retained[0, 0] = 0
        with self.assertRaises(ValueError):
            retained.setflags(write=True)

    def test_degenerate_first_geometry_and_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.update(100, np.full((478, 3), 0.5))
        for kwargs in ({"stable_frames": 1}, {"stable_seconds": float("nan")},
                       {"history_frames": 10000}, {"max_gap_seconds": 2}):
            with self.assertRaises(ValueError):
                TrackingPolicy(**kwargs)

    def test_invalid_input_after_stability_clears_history(self):
        for failure in ("timestamp", "frame", "clock"):
            with self.subTest(failure=failure):
                self.tracker = FaceTracker()
                for timestamp in (100, 100.1, 100.2):
                    self.update(timestamp)
                with self.assertRaises(ValueError):
                    if failure == "clock":
                        self.tracker.expire(float("nan"))
                    elif failure == "timestamp":
                        self.update(float("nan"))
                    else:
                        self.update(100.3, frame=np.zeros((480, 640), np.uint8))
                self.assertEqual(self.tracker.result.state, TrackingState.REJECTED)
                self.assertEqual(self.tracker.history, ())


class TrackingSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = Backend()
        self.frame = good_frame()
        self.session = LivenessSession(lambda: self.backend, clock=self.clock)
        self.session.start()

    def tearDown(self):
        self.session.finish()

    def process(self):
        self.clock.now += 0.1
        result = self.session.process_frame(self.frame, self.clock.now)
        validate_result(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        return result

    def stabilize(self):
        for _ in range(3):
            result = self.process()
        self.assertEqual(result.tracking.state, TrackingState.STABLE)
        return result

    def test_stable_session_never_returns_live(self):
        result = self.stabilize()
        self.assertEqual(result.decision, Decision.UNCERTAIN)
        self.assertEqual(result.confidence, 0)
        self.assertTrue(all(not signal.available for signal in result.signals.values()))
        self.assertNotIn("landmarks", result.to_dict()["tracking"])

    def test_face_loss_and_reentry_clear_history_and_reacquire(self):
        first = self.stabilize()
        self.backend.faces = ()
        result = self.process()
        self.assertEqual(self.session.history, ())
        self.assertTrue(result.tracking.history_reset)
        self.backend.faces = (points(),)
        returned = self.process()
        self.assertEqual(returned.tracking.state, TrackingState.ACQUIRING)
        self.assertNotEqual(returned.tracking.track_id, first.tracking.track_id)
        self.assertEqual(len(self.session.history), 1)

    def test_quality_failure_clears_history_then_starts_new_track(self):
        first = self.stabilize()
        self.frame = np.zeros((480, 640, 3), np.uint8)
        result = self.process()
        self.assertEqual(result.decision, Decision.INSUFFICIENT_QUALITY)
        self.assertEqual(self.session.history, ())
        self.frame = good_frame()
        returned = self.process()
        self.assertNotEqual(returned.tracking.track_id, first.tracking.track_id)
        self.assertEqual(returned.tracking.state, TrackingState.ACQUIRING)

    def test_session_appearance_switch_ends_attempt_uncertain(self):
        self.frame = textured_frame()
        self.stabilize()
        self.frame = 255 - self.frame
        result = self.process()
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "appearance_discontinuity")
        self.assertIn("tracking_discontinuity", result.flags)
        self.assertEqual(result.decision, Decision.UNCERTAIN)
        self.assertEqual(self.session.history, ())
        self.assertIsNone(self.session.observation)

    def test_current_result_and_history_access_expire_stale_capture(self):
        self.stabilize()
        self.clock.now += 0.5
        self.assertEqual(self.session.history, ())
        result = self.session.current_result()
        self.assertEqual(result.tracking.state, TrackingState.LOST)
        self.assertEqual(result.reason, "tracking_stale")
        self.assertFalse(result.quality.acceptable)
        self.assertIsNone(self.session.observation)
        self.assertEqual(self.process().tracking.state, TrackingState.ACQUIRING)

    def test_short_frame_gap_clears_evidence_and_restarts_acquisition(self):
        first = self.stabilize()
        self.clock.now += 0.4
        result = self.process()
        self.assertNotEqual(result.tracking.track_id, first.tracking.track_id)
        self.assertEqual(result.tracking.reset_reason, "frame_gap")
        self.assertEqual(len(self.session.history), 1)

    def test_every_terminal_path_clears_history_and_geometry(self):
        for action in ("cancel", "finish", "camera_failure", "multiple_faces", "model_failure", "malformed_frame"):
            with self.subTest(action=action):
                self.backend = Backend()
                self.session = LivenessSession(lambda: self.backend, clock=self.clock)
                self.session.start()
                self.frame = good_frame()
                self.stabilize()
                if action in ("cancel", "finish"):
                    result = getattr(self.session, action)()
                elif action == "camera_failure":
                    result = self.session.fail("camera_read_failed")
                else:
                    if action == "multiple_faces":
                        self.backend.faces = (points(), points())
                    elif action == "model_failure":
                        self.backend.error = True
                    else:
                        self.frame = None
                    result = self.process()
                self.assertTrue(result.terminal)
                self.assertEqual(result.tracking.history_frame_count, 0)
                self.assertEqual(self.session.history, ())
                self.assertIsNone(self.session.observation)
                self.assertTrue(self.backend.closed)

    def test_loss_clears_any_retained_evidence_scores(self):
        self.stabilize()
        # Inject a hypothetical future detector's evidence to verify reset wiring.
        signals = dict(self.session._result.signals)
        signals["blink"] = SignalResult(confidence=0.9, reliability=0.8, available=True)
        self.session._result = replace(self.session._result, signals=signals)
        self.backend.faces = ()
        result = self.process()
        self.assertFalse(result.signals["blink"].available)
        self.assertEqual(result.signals["blink"].reliability, 0)


if __name__ == "__main__":
    unittest.main()
