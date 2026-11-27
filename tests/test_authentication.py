"""Face-unlock boundary tests; synthetic features are not biometric calibration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from facepass.auth_challenge import ActiveChallenge
from facepass.auth_cli import auth_request
from facepass.auth_cli import _enroll_and_verify
from facepass.auth_models import IdentityModel, identity_similarity, normalized_feature, verify_auth_model
from facepass.auth_service import AuthServer, parse_auth_request, supervised_auth_capture
from facepass.auth_store import EnrollmentStore, validate_template
from tools.auth_integration import patch_main, patch_messages, patch_pam, remove_block


def vector(index=0):
    result = np.zeros(128)
    result[index] = 1
    return result


class FeatureTests(unittest.TestCase):
    def test_matching_requires_three_supporting_templates(self):
        self.assertEqual(identity_similarity(vector(), [vector()] * 8), 1)
        self.assertAlmostEqual(identity_similarity(vector(), [vector()] + [vector(1)] * 7), 1 / 3)
        self.assertEqual(identity_similarity(vector(), [vector(1)] * 8), 0)

    def test_invalid_features_fail(self):
        for value in ([0] * 128, [1] * 127, [True] * 128, [float("nan")] * 128,
                      [float("inf")] * 128, [1e308] * 128, ["1"] * 128):
            with self.subTest(value=str(value)[:20]), np.errstate(over="ignore"):
                with self.assertRaises(ValueError):
                    normalized_feature(value)

    def test_model_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "face_detection_yunet_2023mar.onnx"
            path.write_bytes(b"not a verified model")
            with self.assertRaises(ValueError):
                verify_auth_model(path)

    def test_real_models_reject_blank_frame(self):
        models = Path(__file__).resolve().parents[1] / "models"
        if not (models / "face_recognition_sface_2021dec.onnx").exists():
            self.skipTest("optional verified identity models absent")
        import cv2
        cv2.setNumThreads(1)
        identity = IdentityModel(models)
        self.assertIsNone(identity.feature(np.zeros((480, 640, 3), np.uint8), np.full((478, 3), .5)))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.store = EnrollmentStore(self.path, owner_uid=os.geteuid())
        self.uid = max(1000, os.getuid())
        self.target = self.path / f"{self.uid}.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_write_replace_and_remove(self):
        self.store.write(self.uid, [vector()] * 8)
        self.assertTrue(self.store.enrolled(self.uid))
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.target.stat().st_nlink, 1)
        with self.assertRaises(FileExistsError):
            self.store.write(self.uid, [vector(1)] * 8)
        self.assertEqual(self.store.read(self.uid)["templates"][0][0], 1)
        self.store.write(self.uid, [vector(1)] * 8, replace=True)
        self.assertEqual(self.store.read(self.uid)["templates"][0][1], 1)
        self.store.remove(self.uid)
        self.assertFalse(self.store.enrolled(self.uid))

    def test_schema_and_model_binding(self):
        self.store.write(self.uid, [vector()] * 8)
        value = self.store.read(self.uid)
        for key, invalid in (("uid", self.uid + 1), ("schema_version", True), ("models", {}),
                             ("created_utc", "2026-01-01T00:00:00"), ("templates", [vector().tolist()] * 3)):
            candidate = deepcopy(value)
            candidate[key] = invalid
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                validate_template(candidate, self.uid)

    def test_nonprivate_directory_or_template_fails(self):
        self.store.write(self.uid, [vector()] * 8)
        self.target.chmod(0o644)
        self.assertFalse(self.store.enrolled(self.uid))
        self.target.chmod(0o600)
        self.path.chmod(0o755)
        self.assertFalse(self.store.enrolled(self.uid))

    def test_symlink_hardlink_and_oversized_files_fail(self):
        self.store.write(self.uid, [vector()] * 8)
        os.link(self.target, self.path / "linked")
        self.assertFalse(self.store.enrolled(self.uid))
        (self.path / "linked").unlink()
        self.target.rename(self.path / "original")
        self.target.symlink_to(self.path / "original")
        self.assertFalse(self.store.enrolled(self.uid))
        with self.assertRaises(OSError):
            self.store.write(self.uid, [vector()] * 8, replace=True)
        self.target.unlink()
        self.target.write_bytes(b"x" * 80_001)
        self.target.chmod(0o600)
        self.assertFalse(self.store.enrolled(self.uid))

    def test_invalid_update_preserves_old_enrollment(self):
        self.store.write(self.uid, [vector()] * 8)
        old = self.target.read_bytes()
        with self.assertRaises(ValueError):
            self.store.write(self.uid, [[0] * 128] * 8, replace=True)
        self.assertEqual(self.target.read_bytes(), old)
        self.assertFalse(any(path.name.startswith(".enroll-") for path in self.path.iterdir()))

    def enrollment(self, verified, connected=lambda: True):
        self.store.write(self.uid, [vector()] * 8)
        candidate_path = self.path / "candidate"
        candidate_path.mkdir(mode=0o700)
        candidate = EnrollmentStore(candidate_path, owner_uid=os.geteuid())
        capture = {"success": True, "reason": "enrollment_captured", "templates": [vector(1)] * 8}
        with patch("facepass.auth_cli.supervised_auth_capture", side_effect=[capture, {"success": verified}]) as runner:
            value = _enroll_and_verify(self.uid, lambda _: None, self.store, candidate,
                                       replace=True, preview=True, connected=connected)
        self.assertEqual(runner.call_args_list[1].args[0], "authenticate")
        self.assertIn("--template-directory", runner.call_args_list[1].kwargs["command"])
        return value

    def test_second_challenge_required_before_committing_enrollment(self):
        self.assertFalse(self.enrollment(False)["success"])
        self.assertEqual(self.store.read(self.uid)["templates"][0][0], 1)

    def test_cancelled_enrollment_cannot_replace_templates(self):
        self.assertFalse(self.enrollment(True, lambda: False)["success"])
        self.assertEqual(self.store.read(self.uid)["templates"][0][0], 1)

    def test_verified_enrollment_commits_and_exposes_no_features(self):
        value = self.enrollment(True)
        self.assertEqual(value, {"success": True, "reason": "enrollment_saved_and_verified"})
        self.assertEqual(self.store.read(self.uid)["templates"][0][1], 1)


class Clock:
    value = 0.0
    def __call__(self):
        return self.value


class FixedRandom:
    def choice(self, values):
        return 2
    def shuffle(self, values):
        pass


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.challenge = ActiveChallenge(self.clock, rng=FixedRandom())
        self.blinks = 0

    def update(self, yaw=0, **overrides):
        self.clock.value += .1
        values = dict(track_id=1, yaw=yaw, blink_count=self.blinks, eyes_open=True,
                      identity_match=True, quality_ok=True)
        values.update(overrides)
        return self.challenge.update(self.clock(), **values)

    def center(self):
        for _ in range(5):
            self.update()

    def turn(self, yaw):
        for _ in range(5):
            self.update(yaw)
        for _ in range(5):
            self.update()

    def test_requires_both_blinks_and_both_turns_then_center(self):
        self.center()
        self.assertEqual(self.challenge.stage, 0)
        self.blinks = 1
        self.update()
        self.assertEqual(self.challenge.stage, 0)
        self.blinks = 2
        self.update()
        self.turn(-18)
        self.assertEqual(self.challenge.stage, 2)
        self.blinks = 4
        self.update()
        self.assertFalse(self.challenge.done)
        self.turn(18)
        self.assertTrue(self.challenge.done)
        self.assertFalse(self.update())  # A completed challenge cannot issue a second success.

    def test_still_face_or_wrong_direction_never_completes(self):
        self.center()
        for _ in range(10):
            self.update()
        self.assertFalse(self.challenge.done)
        self.blinks = 2
        self.update()
        self.turn(18)
        self.assertEqual(self.challenge.stage, 1)

    def test_account_identity_quality_and_track_changes_fail(self):
        for change in ({"identity_match": False}, {"quality_ok": False}, {"track_id": 2},
                       {"conflict": True}, {"eyes_open": 1}, {"track_id": True},
                       {"blink_count": -1}, {"blink_count": True}):
            self.setUp()
            self.center()
            self.update(**change)
            self.assertTrue(self.challenge.failed, change)

    def test_stale_duplicate_future_and_gapped_frames_fail(self):
        for timestamp in (-1, 1, .1, .9, float("nan")):
            self.setUp()
            self.update()
            self.clock.value = .5
            self.challenge.update(timestamp, track_id=1, yaw=0, blink_count=0, eyes_open=True,
                                  identity_match=True, quality_ok=True)
            self.assertTrue(self.challenge.failed, timestamp)

    def test_deadline_fails(self):
        self.center()
        self.clock.value = 26
        self.update()
        self.assertTrue(self.challenge.failed)


class ProtocolTests(unittest.TestCase):
    def test_peer_and_request_validation(self):
        nonce = "a" * 32
        self.assertEqual(parse_auth_request(f"AUTH 1000 kde {nonce}\n".encode(), 1000), ("AUTH", 1000, "kde", nonce))
        for request, uid in ((f"AUTH 1000 kde {nonce}\n", 1001), (f"AUTH 0 kde {nonce}\n", 0),
                             (f"AUTH 1000 sudo {nonce}\n", 1000), (f"AUTH 1000 kde {nonce}\nextra", 1000),
                             (f"AUTH 1000 kde {nonce}\n\n", 1000), ("AUTH 1000 kde bogus\n", 1000)):
            with self.subTest(request=request), self.assertRaises((ValueError, PermissionError)):
                parse_auth_request(request.encode(), uid)

    def worker(self, script, connected=lambda: True, timeout=2):
        return supervised_auth_capture("authenticate", 1000, lambda _: None, connected,
                                       command=[sys.executable, "-I", "-c", script], timeout=timeout)

    def test_worker_accepts_exact_success_and_exit(self):
        message = json.dumps({"kind": "final", "success": True, "reason": "identity_and_active_challenge_verified"})
        self.assertTrue(self.worker(f"print({message!r}, flush=True)")["success"])
        self.assertFalse(self.worker(f"import sys; print({message!r}, flush=True); sys.exit(2)")["success"])

    def test_worker_rejects_spoofed_malformed_or_trailing_results(self):
        for value in ({"kind": "final", "success": True, "reason": "live"},
                      {"kind": "final", "success": 1, "reason": "identity_and_active_challenge_verified"},
                      {"kind": "final", "success": True, "reason": "identity_and_active_challenge_verified", "extra": 1}):
            self.assertFalse(self.worker(f"print({json.dumps(value)!r})")["success"])
        self.assertFalse(self.worker("print('not JSON')")["success"])
        self.assertFalse(self.worker("print('{\"kind\":\"started\"}',end='')")["success"])
        final = json.dumps({"kind": "final", "success": True, "reason": "identity_and_active_challenge_verified"})
        self.assertFalse(self.worker(f"print({final!r}); print({final!r})")["success"])

    def test_worker_hard_timeout_and_disconnect_kill_process_group(self):
        start = time.monotonic()
        self.assertFalse(self.worker("import time; time.sleep(60)", timeout=.1)["success"])
        self.assertLess(time.monotonic() - start, 2)
        self.assertFalse(self.worker("import time; time.sleep(60)", connected=lambda: False)["success"])


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.uid = max(1000, os.getuid())
        class Store:
            def enrolled(_, uid):
                return uid == self.uid
        self.runner_result = {"success": True, "reason": "identity_and_active_challenge_verified"}
        self.calls = 0
        def runner(*args):
            self.calls += 1
            return self.runner_result
        self.server = AuthServer(self.path / "auth.sock", store=Store(), runner=runner, owner_uid=os.getuid())
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse((self.path / "auth.sock").exists())
        self.temporary.cleanup()

    def request(self, operation):
        return list(auth_request(operation, self.uid, path=self.server.path, owner_uid=os.getuid()))[-1]["success"]

    def test_status_never_captures_auth_requires_verified_result(self):
        self.assertTrue(self.request("STATUS"))
        self.assertEqual(self.calls, 0)
        self.assertTrue(self.request("AUTH"))
        self.runner_result = {"success": True, "reason": "live"}
        self.assertFalse(self.request("AUTH"))
        self.runner_result = {"success": False, "reason": "identity_and_active_challenge_verified"}
        self.assertFalse(self.request("AUTH"))

    def test_busy_and_rate_limited_attempts_fail(self):
        self.server.capture_lock.acquire()
        self.assertFalse(self.request("AUTH"))
        self.server.capture_lock.release()
        for _ in range(4):
            self.assertTrue(self.request("AUTH"))
        self.assertFalse(self.request("AUTH"))
        self.assertEqual(self.calls, 4)

    def test_camera_lock_symlink_fails_before_capture(self):
        (self.path / "camera.lock").symlink_to(self.path / "target")
        with self.assertRaises(ValueError):
            self.request("AUTH")
        self.assertEqual(self.calls, 0)

    def test_production_client_rejects_nonroot_broker(self):
        with self.assertRaises(PermissionError):
            list(auth_request("STATUS", self.uid, path=self.server.path))

    def test_insecure_socket_parent_refused(self):
        self.path.chmod(0o777)
        with self.assertRaises(ValueError):
            AuthServer(self.path / "another.sock", owner_uid=os.getuid())
        self.path.chmod(0o700)

    def test_live_socket_cannot_be_replaced_and_stale_socket_can(self):
        with self.assertRaises(FileExistsError):
            AuthServer(self.server.path, owner_uid=os.getuid())
        path = self.path / "stale.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(path))
        recovered = AuthServer(path, owner_uid=os.getuid())
        recovered.server_close()
        self.assertFalse(path.exists())


class IntegrationEditTests(unittest.TestCase):
    def test_pam_edit_is_idempotent_preserves_account_session_and_reversible(self):
        original = "#%PAM-1.0\nauth substack password-auth\naccount include password-auth\nsession include postlogin\n"
        edited = patch_pam(original)
        self.assertEqual(patch_pam(edited), edited)
        self.assertEqual(remove_block(edited), original)
        self.assertIn("account include password-auth\nsession include postlogin", edited)
        with self.assertRaises(ValueError):
            patch_pam("auth required pam_deny.so\n")

    def test_kde_edits_are_reversible_and_keep_password_flow(self):
        directory = Path("/usr/share/plasma/shells/org.kde.plasma.desktop/contents/lockscreen")
        if not directory.exists():
            self.skipTest("KDE shell not installed")
        for name, transform in (("MainBlock.qml", patch_main), ("LockScreenUi.qml", patch_messages)):
            original = (directory / name).read_text()
            edited = transform(original)
            self.assertEqual(transform(edited), edited)
            self.assertEqual(remove_block(edited, "qml").strip(), original.strip())
        self.assertIn("passwordResult(password);", patch_main((directory / "MainBlock.qml").read_text()))


if __name__ == "__main__":
    unittest.main()
