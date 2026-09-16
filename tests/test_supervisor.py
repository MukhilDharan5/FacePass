import json
import sys
import time
import unittest

from facepass.results import failure_result
from facepass.supervisor import supervise, validate_result


def command(source):
    return [sys.executable, "-u", "-c", source]


class SupervisorTests(unittest.TestCase):
    def run_worker(self, source, **kwargs):
        return supervise(command(source), duration=0.5, startup_timeout=0.3, frame_timeout=0.2, **kwargs)

    def test_clean_final_result(self):
        message = json.dumps({"kind": "final", "result": failure_result("no_face")})
        result = self.run_worker(f"print({message!r})")
        self.assertEqual(result["reason"], "no_face")

    def test_startup_hang_is_bounded(self):
        before = time.monotonic()
        result = self.run_worker("import time; time.sleep(60)")
        self.assertEqual(result["reason"], "worker_startup_timeout")
        self.assertLess(time.monotonic() - before, 2)

    def test_camera_or_model_hang(self):
        result = self.run_worker('import time; print(\'{"kind":"started"}\'); time.sleep(60)')
        self.assertEqual(result["reason"], "camera_or_model_stalled")

    def test_partial_ipc_message_does_not_block_watchdog(self):
        result = self.run_worker('import sys,time; sys.stdout.write(\'{"kind":\'); sys.stdout.flush(); time.sleep(60)')
        self.assertEqual(result["reason"], "worker_startup_timeout")

    def test_crash_after_final_invalidates_result(self):
        message = json.dumps({"kind": "final", "result": failure_result("no_face")})
        result = self.run_worker(f"import os; print({message!r}); os._exit(7)")
        self.assertEqual(result["reason"], "worker_crashed")

    def test_missing_final_and_unknown_messages(self):
        for source in ("pass", "print('{}')", "print('not json')", "print('{\"kind\":\"unexpected\"}')"):
            with self.subTest(source=source):
                self.assertEqual(self.run_worker(source)["confidence"], 0)

    def test_cancellation_kills_hung_worker(self):
        before = time.monotonic()
        result = self.run_worker("import time; time.sleep(60)", pump=lambda: False)
        self.assertEqual(result["reason"], "cancelled")
        self.assertLess(time.monotonic() - before, 2)

    def test_unexpected_success_and_nan_cannot_pass_ipc(self):
        for mutation in ({"decision": "LIVE"}, {"confidence": float("nan")}, {"decision": "UNKNOWN"}):
            value = failure_result("example")
            value.update(mutation)
            with self.assertRaises(ValueError):
                validate_result(json.loads(json.dumps(value)))
            message = json.dumps({"kind": "final", "result": value})
            result = self.run_worker(f"print({message!r})")
            self.assertEqual(result["decision"], "UNCERTAIN")
            self.assertEqual(result["confidence"], 0)

    def test_extra_biometric_fields_and_bad_quality_rejected(self):
        value = failure_result("example")
        value["embedding"] = [0.2, 0.3]
        with self.assertRaises(ValueError):
            validate_result(json.loads(json.dumps(value)))
        value = json.loads(json.dumps(failure_result("example")))
        value["quality"]["blur_variance"] = float("nan")
        with self.assertRaises(ValueError):
            validate_result(value)

    def test_oversized_message(self):
        result = self.run_worker("print('x' * 2100000)")
        self.assertEqual(result["reason"], "oversized_worker_message")

    def test_invalid_tracking_data_and_vectors_rejected(self):
        for key, bad in (("state", "UNKNOWN"), ("continuity_score", float("nan")),
                         ("history_frame_count", 99999), ("track_id", [1, 2]),
                         ("landmarks", [[0.2, 0.3]]), ("appearance_change", float("inf"))):
            with self.subTest(key=key):
                value = json.loads(json.dumps(failure_result("example")))
                value["tracking"][key] = bad
                with self.assertRaises(ValueError):
                    validate_result(value)

    def test_stable_tracking_cannot_override_live_prohibition(self):
        value = json.loads(json.dumps(failure_result("example")))
        value["tracking"].update(state="STABLE", track_id=1, continuous_frames=10,
                                 history_frame_count=10, continuity_score=1)
        value.update(decision="LIVE", confidence=1)
        with self.assertRaises(ValueError):
            validate_result(value)


if __name__ == "__main__":
    unittest.main()
