import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from facepass.daemon import MAX_REQUEST_BYTES, PROTOCOL_VERSION, ResearchServer, private_socket_parent, stream_request, validate_request
from facepass.results import failure_result
from facepass.supervisor import supervise


def request(operation="status", request_id="test", **fields):
    return {"version": PROTOCOL_VERSION, "request_id": request_id, "operation": operation, **fields}


class RequestSchemaTests(unittest.TestCase):
    def test_version_types_unknown_fields_and_capture_bypasses_rejected(self):
        invalid = [request(version=True), request(version=1), request(version=PROTOCOL_VERSION + 1), request(operation="unlock"),
                   request(request_id="bad id"), request(embedding=[1, 2]),
                   request("cancel", target_id=""), request("observe", options={"device": "http://camera"}),
                   request("observe", options={"model": "/tmp/arbitrary.task"}),
                   request("observe", options={"preview": True}), request("observe", options={"duration": float("nan")}),
                   request("observe", options={"fixed_window": 1}),
                   request("observe", options={"quality_profile": "always_accept"})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request(value)
        validate_request(request("observe", options={}))
        validate_request(request("observe", options={"quality_profile": "strict"}))
        validate_request(request("cancel", target_id="active"))

    def test_private_directory_existing_socket_and_symlinks_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            public = root / "public"
            public.mkdir(mode=0o755)
            with self.assertRaises(ValueError):
                private_socket_parent(public / "daemon.sock")
            link = root / "linked"
            link.symlink_to(public, target_is_directory=True)
            with self.assertRaises(ValueError):
                private_socket_parent(link / "daemon.sock")
            existing = root / "existing.sock"
            existing.write_text("keep me")
            with self.assertRaises(FileExistsError):
                private_socket_parent(existing)
            self.assertEqual(existing.read_text(), "keep me")


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "daemon.sock"
        self.started = threading.Event()
        self.complete = threading.Event()
        self.calls = 0
        self.capture_options = []

        def runner(options, *, on_update, pump):
            self.calls += 1
            self.capture_options.append(options)
            self.started.set()
            value = failure_result("observing")
            value["terminal"] = False
            on_update(json.loads(json.dumps(value)), None)
            deadline = time.monotonic() + 3
            while pump() and not self.complete.wait(.01):
                if time.monotonic() > deadline:
                    return failure_result("test_timeout")
            return failure_result("finished" if self.complete.is_set() else "cancelled")

        self.server = ResearchServer(self.path, "model.task", runner=runner)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()

    def test_quality_profile_reaches_the_capture_runner(self):
        self.complete.set()
        messages = list(stream_request(self.path, request("observe", options={"quality_profile": "strict"})))
        self.assertEqual(messages[-1]["kind"], "final")
        self.assertEqual(self.capture_options[0].quality_profile, "strict")

    def tearDown(self):
        self.server.request_stop()
        self.complete.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def query(self, value):
        return list(stream_request(self.path, value, timeout=3))

    def connect_observe(self, request_id="attempt"):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(3)
        connection.connect(str(self.path))
        source = connection.makefile("rb")
        connection.sendall((json.dumps(request("observe", request_id, options={})) + "\n").encode())
        self.assertEqual(json.loads(source.readline())["kind"], "accepted")
        self.assertTrue(self.started.wait(1))
        return connection, source

    def test_idle_status_opens_no_pipeline_and_socket_is_private(self):
        status = self.query(request())[0]
        self.assertEqual(status["kind"], "status")
        self.assertFalse(status["authentication_enabled"])
        self.assertIsNone(status["active_request_id"])
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_streaming_observation_final_is_valid_non_success(self):
        self.complete.set()
        messages = self.query(request("observe", options={}))
        self.assertEqual([item["kind"] for item in messages], ["accepted", "update", "final"])
        self.assertEqual(messages[-1]["result"]["decision"], "UNCERTAIN")
        self.assertEqual(messages[-1]["result"]["reason"], "finished")
        self.assertNotIn("landmarks", messages[-1]["result"])

    def test_second_capture_is_busy_and_cancellation_is_correlated(self):
        connection, source = self.connect_observe()
        try:
            busy = self.query(request("observe", "second", options={}))[0]
            self.assertEqual(busy["error"], "busy")
            self.assertEqual(self.query(request("status", "status"))[0]["active_request_id"], "attempt")
            unknown = self.query(request("cancel", "unknown", target_id="different"))[0]
            self.assertEqual(unknown["error"], "unknown_attempt")
            acknowledgement = self.query(request("cancel", "cancel", target_id="attempt"))[0]
            self.assertEqual(acknowledgement["kind"], "cancel_requested")
            messages = [json.loads(line) for line in source]
            self.assertEqual(messages[-1]["result"]["reason"], "cancelled")
            self.assertEqual(self.calls, 1)
        finally:
            source.close()
            connection.close()

    def test_client_disconnect_cancels_and_releases_active_slot(self):
        connection, source = self.connect_observe()
        source.close()
        connection.close()
        deadline = time.monotonic() + 2
        while self.server.active_id is not None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIsNone(self.server.active_id)
        self.complete.set()
        self.started.clear()
        self.assertEqual(self.query(request("observe", "new", options={}))[-1]["kind"], "final")

    def test_malformed_oversized_duplicate_and_pipelined_requests_never_capture(self):
        data = [b"not json\n", b'{"version":1,"version":1}\n',
                b"x" * (MAX_REQUEST_BYTES + 1),
                (json.dumps(request()) + "\n" + json.dumps(request()) + "\n").encode()]
        for payload in data:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(3)
                connection.connect(str(self.path))
                connection.sendall(payload)
                response = json.loads(connection.recv(4096))
                self.assertEqual(response["kind"], "error")
        self.assertEqual(self.calls, 0)

    def test_failed_or_success_forging_runner_is_fail_closed(self):
        for mode in ("exception", "success", "nonterminal"):
            def bad_runner(options, **kwargs):
                if mode == "exception":
                    raise RuntimeError("native error")
                value = failure_result("forged")
                if mode == "success":
                    value.update(decision="LIVE", confidence=1)
                else:
                    value["terminal"] = False
                return value
            self.server.runner = bad_runner
            final = self.query(request("observe", mode, options={}))[-1]
            self.assertEqual(final["result"]["reason"], "daemon_worker_error")
            self.assertEqual(final["result"]["confidence"], 0)

    def test_different_peer_uid_is_refused(self):
        class Peer:
            def getsockopt(self, *args):
                return struct.pack("3i", 123, os.getuid() + 1, 123)
        self.assertFalse(self.server.verify_request(Peer(), ""))

    def test_partial_request_deadline_is_not_refreshed_by_dripping_bytes(self):
        with patch("facepass.daemon.REQUEST_TIMEOUT", .12), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(str(self.path))
            before = time.monotonic()
            connection.sendall(b"{")
            time.sleep(.07)
            connection.sendall(b'"')
            # This second fragment must not reset the original receive deadline.
            self.assertEqual(connection.recv(4096), b"")
            self.assertLess(time.monotonic() - before, .4)
            self.assertEqual(self.calls, 0)

    def test_daemon_cancellation_terminates_a_hung_real_subprocess(self):
        import sys
        def hung_runner(options, *, on_update, pump):
            self.started.set()
            return supervise([sys.executable, "-u", "-c", 'import time; print(\'{"kind":"started"}\'); time.sleep(60)'],
                             duration=1, pump=pump, startup_timeout=1, frame_timeout=2)
        self.server.runner = hung_runner
        connection, source = self.connect_observe()
        before = time.monotonic()
        try:
            self.query(request("cancel", "cancel", target_id="attempt"))
            messages = [json.loads(line) for line in source]
            self.assertEqual(messages[-1]["result"]["reason"], "cancelled")
            self.assertLess(time.monotonic() - before, 2)
        finally:
            source.close()
            connection.close()

    def test_shutdown_cancels_active_attempt_and_removes_owned_socket(self):
        connection, source = self.connect_observe()
        try:
            self.server.request_stop()
            messages = [json.loads(line) for line in source]
            self.assertEqual(messages[-1]["result"]["reason"], "cancelled")
            self.server.server_close()
            self.assertFalse(self.path.exists())
        finally:
            source.close()
            connection.close()


if __name__ == "__main__":
    unittest.main()
