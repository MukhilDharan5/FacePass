"""Exercise the native module through real libpam in isolated temporary configs."""

import ctypes as C
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest


class Message(C.Structure):
    _fields_ = [("style", C.c_int), ("text", C.c_char_p)]


class Response(C.Structure):
    _fields_ = [("text", C.c_void_p), ("code", C.c_int)]


CALLBACK = C.CFUNCTYPE(C.c_int, C.c_int, C.POINTER(C.POINTER(Message)),
                     C.POINTER(C.POINTER(Response)), C.c_void_p)


class Conversation(C.Structure):
    _fields_ = [("callback", CALLBACK), ("data", C.c_void_p)]


class FakeBroker:
    def __init__(self, path, result="SUCCESS", available=True, info=False):
        self.path, self.result, self.available, self.info = path, result, available, info
        self.stopped = threading.Event()
        self.operations = []
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(path))
        self.socket.listen(8)
        self.socket.settimeout(.1)
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()

    def serve(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.socket.accept()
            except socket.timeout:
                continue
            with client:
                client.settimeout(1)
                try:
                    row = client.makefile("rb").readline(256).decode().split()
                    if len(row) != 4:
                        continue
                    operation, uid, service, nonce = row
                    self.operations.append(operation)
                    result = ("ENROLLED" if self.available else "UNAVAILABLE") if operation == "STATUS" else self.result
                    if operation == "AUTH" and self.info:
                        client.sendall(b"INFO FacePass: blink slowly 1 time.\n")
                    if result == "STALL":
                        self.stopped.wait(.6)
                    else:
                        suffix = "0" * 32 if result == "WRONG_NONCE" else nonce
                        result = "SUCCESS" if result == "WRONG_NONCE" else result
                        client.sendall(f"{result} {suffix}\n".encode())
                except (OSError, ValueError):
                    pass

    def close(self):
        self.stopped.set()
        self.thread.join(2)
        self.socket.close()
        self.path.unlink()


class PamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("gcc") or os.getuid() < 1000:
            raise unittest.SkipTest("native ABI tests require gcc and an ordinary local account")
        cls.temporary = tempfile.TemporaryDirectory(prefix="facepass-pam-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.socket_path = cls.directory / "auth.sock"
        root = Path(__file__).resolve().parents[1]
        common = ["gcc", "-std=c11", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror",
                  "-I", str(root / "native/include"), str(root / "native/pam_facepass.c"),
                  "-l:libpam.so.0", '-DFACEPASS_SOCKET="' + str(cls.socket_path) + '"', "-DFACEPASS_AUTH_TIMEOUT=.3"]
        cls.production_peer = cls.directory / "production-peer.so"
        cls.test_peer = cls.directory / "test-peer.so"
        subprocess.run(common + ["-o", str(cls.production_peer)], check=True, capture_output=True)
        subprocess.run(common + [f"-DFACEPASS_PEER_UID={os.getuid()}", "-o", str(cls.test_peer)], check=True, capture_output=True)
        # A downstream test module checks that FacePass hands the exact supplied
        # password to the original stack instead of authenticating it itself.
        source = cls.directory / "check_token.c"
        source.write_text('#include <security/pam_modules.h>\n#include <string.h>\n'
                          'PAM_EXTERN int pam_sm_authenticate(pam_handle_t *p,int f,int a,const char **v)'
                          '{const void *x=0;(void)f;(void)a;(void)v;'
                          'return pam_get_item(p,PAM_AUTHTOK,&x)==PAM_SUCCESS && x && !strcmp(x,"normal-password") ? PAM_SUCCESS:PAM_AUTH_ERR;}\n')
        cls.token_module = cls.directory / "check_token.so"
        subprocess.run(["gcc", "-fPIC", "-shared", "-I", str(root / "native/include"), str(source),
                        "-l:libpam.so.0", "-o", str(cls.token_module)], check=True, capture_output=True)
        cls.pam = C.CDLL("libpam.so.0")
        cls.pam.pam_start_confdir.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(Conversation), C.c_char_p, C.POINTER(C.c_void_p)]
        cls.pam.pam_authenticate.argtypes = [C.c_void_p, C.c_int]
        cls.pam.pam_end.argtypes = [C.c_void_p, C.c_int]
        cls.libc = C.CDLL("libc.so.6")
        cls.libc.calloc.argtypes = [C.c_size_t, C.c_size_t]
        cls.libc.calloc.restype = C.c_void_p
        cls.libc.strdup.argtypes = [C.c_char_p]
        cls.libc.strdup.restype = C.c_void_p

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def call(self, *, answer="", module=None, tail="pam_deny.so", face_only=False, username=None, service="kde"):
        module = module or self.test_peer
        (self.directory / service).write_text(f"auth [success=done ignore=ignore default=die] {module} {'face_only' if face_only else ''}\nauth required {tail}\n")
        self.messages = []
        @CALLBACK
        def conversation(count, messages, responses, data):
            array = C.cast(self.libc.calloc(count, C.sizeof(Response)), C.POINTER(Response))
            for index in range(count):
                message = messages[index].contents
                self.messages.append((message.style, message.text.decode()))
                if message.style in (1, 2):
                    array[index].text = self.libc.strdup(answer.encode())
            responses[0] = array
            return 0
        conv = Conversation(conversation, None)
        handle = C.c_void_p()
        result = self.pam.pam_start_confdir(service.encode(), (username or pwd.getpwuid(os.getuid()).pw_name).encode(),
                                           C.byref(conv), str(self.directory).encode(), C.byref(handle))
        self.assertEqual(result, 0)
        try:
            result = self.pam.pam_authenticate(handle, 0)
        finally:
            self.pam.pam_end(handle, result)
        return result

    def broker(self, **kwargs):
        result = FakeBroker(self.socket_path, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_empty_password_authenticates_only_verified_broker_success(self):
        broker = self.broker(info=True)
        self.assertEqual(self.call(), 0)
        self.assertEqual(broker.operations, ["STATUS", "AUTH"])
        self.assertTrue(any("blink slowly" in text for _, text in self.messages))

    def test_password_passes_unchanged_to_original_stack_without_camera(self):
        broker = self.broker()
        self.assertEqual(self.call(answer="normal-password", tail=str(self.token_module)), 0)
        self.assertEqual(broker.operations, ["STATUS"])

    def test_bad_password_cannot_succeed_through_face_module(self):
        broker = self.broker()
        self.assertNotEqual(self.call(answer="wrong-password", tail=str(self.token_module)), 0)
        self.assertEqual(broker.operations, ["STATUS"])

    def test_failed_face_keeps_pam_failure_even_if_fallback_would_permit(self):
        self.broker(result="FAILURE")
        self.assertNotEqual(self.call(tail="pam_permit.so"), 0)

    def test_wrong_nonce_and_control_characters_fail(self):
        broker = self.broker(result="WRONG_NONCE")
        self.assertNotEqual(self.call(), 0)
        broker.result = "SUCCESS\t"
        self.assertNotEqual(self.call(), 0)

    def test_unavailable_enrollment_preserves_password_stack(self):
        broker = self.broker(available=False)
        self.assertEqual(self.call(tail="pam_permit.so"), 0)
        self.assertEqual(broker.operations, ["STATUS"])
        self.assertEqual(self.messages, [])

    def test_missing_broker_preserves_password_stack(self):
        self.assertEqual(self.call(tail="pam_permit.so"), 0)
        self.assertEqual(self.messages, [])

    def test_production_peer_check_rejects_user_owned_socket(self):
        self.broker()
        self.assertNotEqual(self.call(module=self.production_peer, face_only=True), 0)

    def test_face_only_requires_success_and_stalled_broker_fails(self):
        broker = self.broker(result="STALL")
        self.assertNotEqual(self.call(face_only=True), 0)
        broker.result = "SUCCESS"
        self.assertEqual(self.call(face_only=True), 0)
        self.assertFalse(any(style == 1 for style, _ in self.messages))

    def test_root_or_unrelated_service_cannot_use_face_module(self):
        broker = self.broker()
        self.assertNotEqual(self.call(username="root", face_only=True), 0)
        self.assertNotEqual(self.call(service="sudo", face_only=True), 0)
        self.assertEqual(broker.operations, [])


if __name__ == "__main__":
    unittest.main()
