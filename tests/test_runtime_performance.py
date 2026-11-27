import json
from dataclasses import replace
import sys
import time
import unittest
from unittest.mock import patch

from facepass.performance import CaptureProfile, distribution
from facepass.results import TemporalDiagnostics, failure_result
from facepass.runtime import CaptureOptions
from facepass.session import LivenessSession
from facepass.supervisor import supervise, validate_result
from tests.test_session import Backend, Clock, good_frame


class RuntimeTests(unittest.TestCase):
    def test_validated_options_and_worker_command_preserve_the_hard_bound(self):
        options = CaptureOptions(duration=.5, extension_duration=1)
        self.assertEqual(options.maximum_extension, 1)
        self.assertIn("--extension-duration", options.worker_command("model.task"))
        fixed = CaptureOptions(fixed_window=True)
        self.assertEqual(fixed.maximum_extension, 0)
        self.assertIn("--fixed-window", fixed.worker_command("model.task"))
        self.assertEqual(options.quality_profile, "tolerant")
        strict = CaptureOptions(quality_profile="strict").worker_command("model.task")
        self.assertEqual(strict[strict.index("--quality-profile") + 1], "strict")
        for kwargs in ({"duration": float("inf")}, {"duration": True}, {"fps": 12.5},
                       {"fps": True}, {"device": "/tmp/video"}, {"extension_duration": 6}, {"fixed_window": "yes"},
                       {"quality_profile": "always_accept"}, {"quality_profile": True}, {"quality_profile": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CaptureOptions(**kwargs)

    def test_extended_session_processes_frames_past_original_deadline(self):
        clock, backend = Clock(), Backend()
        session = LivenessSession(lambda: backend, clock=clock, max_duration=1, extension_duration=1)
        session.start()
        try:
            frame = good_frame()
            for step in range(5):
                clock.now += .1
                session.process_frame(frame, clock.now)
            session._extension_count = 1
            session._effective_limit_ms = 2000
            clock.now = 101.1
            result = session.process_frame(frame, clock.now)
            self.assertFalse(result.terminal)
            self.assertEqual(result.observation.extension_count, 1)
            validate_result(json.loads(json.dumps(result.to_dict())))
            clock.now = 102.1
            self.assertTrue(session.current_result().terminal)
        finally:
            session.finish()

    def test_real_policy_extension_requires_a_fresh_track(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                clock, backend = Clock(), Backend()
                session = LivenessSession(lambda: backend, clock=clock, max_duration=.5, extension_duration=1)
                session.start()
                try:
                    frame = good_frame()
                    for step in range(4):
                        clock.now += .1
                        session.process_frame(frame, clock.now)
                    session._result = replace(session._result, temporal=TemporalDiagnostics(
                        status="observing", window_frame_count=8, window_duration_ms=700,
                        positive_group_count=1))
                    clock.now = 101.0 if stale else 100.55
                    result = session.current_result()
                    self.assertEqual(result.terminal, stale)
                    self.assertEqual(session._extension_count, 0 if stale else 1)
                finally:
                    session.finish()

    def test_watchdog_allows_explicit_extension_and_remains_bounded(self):
        value = json.dumps({"kind": "final", "result": failure_result("finished")})
        source = f"import time; print('{{\"kind\":\"started\"}}', flush=True); time.sleep(1.1); print({value!r}, flush=True)"
        command = [sys.executable, "-u", "-c", source]
        result = supervise(command, duration=.5, extension_duration=.5, frame_timeout=2, startup_timeout=1)
        self.assertEqual(result["reason"], "finished")
        result = supervise(command, duration=.5, frame_timeout=2, startup_timeout=1)
        self.assertEqual(result["reason"], "worker_observation_timeout")

    def test_invalid_watchdog_limits_do_not_spawn_worker(self):
        with patch("facepass.supervisor.subprocess.Popen") as spawn:
            for value in (-1, True, float("nan"), 10):
                result = supervise([], duration=1, extension_duration=value)
                self.assertEqual(result["reason"], "invalid_observation_limits")
            spawn.assert_not_called()

    def test_longest_supported_capture_fits_ipc_and_record_counters(self):
        value = json.loads(json.dumps(failure_result("long_capture")))
        value["frame_count"] = 1050
        value["observation"].update(base_limit_ms=30000, effective_limit_ms=35000,
                                    extension_limit_ms=5000, extension_count=1)
        validate_result(value)
        value["frame_count"] = 1101
        with self.assertRaises(ValueError):
            validate_result(value)

    def test_inconsistent_or_second_extensions_are_rejected(self):
        for changes in ({"extension_count": 2}, {"effective_limit_ms": 6000}, {"minimum_duration_ms": 6000}):
            value = json.loads(json.dumps(failure_result("example")))
            value["observation"].update(changes)
            with self.assertRaises(ValueError):
                validate_result(value)

    def test_empty_and_interpolated_timing_distributions(self):
        self.assertIsNone(distribution([])["p95"])
        result = distribution([10, 20, 30, 40])
        self.assertEqual(result["p50"], 25)
        self.assertEqual(result["maximum"], 40)

    def test_profile_is_bounded_and_handles_missing_updates(self):
        clock = Clock()
        profile = CaptureProfile(clock=clock)
        clock.now += .5
        result = profile.finish(failure_result("camera_open_failed"))
        self.assertIsNone(result["first_update_ms"])
        self.assertEqual(result["wall_ms"], 500)
        update = failure_result("observing")
        for index in range(1200):
            clock.now += .01
            update["frame_count"] = min(index, 1000)
            profile.update(update)
        self.assertEqual(len(profile.intervals), 1100)
        result = profile.finish(update)
        self.assertEqual(result["frames"], 1000)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
