"""Enrollment, verification and status for the optional system face authenticator."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import secrets
import select
import signal
import socket
import stat
import struct
import sys
import tempfile

from .auth_service import AUTH_SOCKET, supervised_auth_capture
from .auth_store import EnrollmentStore, checked_uid

SYSTEM_PYTHON = Path("/opt/facepass-auth/current/venv/bin/python")


def auth_request(operation, uid, *, path=AUTH_SOCKET, owner_uid=0):
    checked_uid(uid)
    nonce = secrets.token_hex(16)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(48 if operation == "AUTH" else 2)
        connection.connect(str(path))
        _, peer, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if peer != owner_uid:
            raise PermissionError("authentication broker has the wrong owner")
        connection.sendall(f"{operation} {uid} kde {nonce}\n".encode("ascii"))
        with connection.makefile("rb") as stream:
            for _ in range(64):
                data = stream.readline(512)
                if not data or not data.endswith(b"\n"):
                    raise ValueError("authentication broker disconnected")
                text = data.decode("ascii").rstrip("\n")
                if text.startswith("INFO ") and operation == "AUTH":
                    yield {"kind": "info", "message": text[5:]}
                elif text in (f"SUCCESS {nonce}", f"ENROLLED {nonce}", f"FAILURE {nonce}", f"UNAVAILABLE {nonce}"):
                    yield {"kind": "final", "success": text in (f"SUCCESS {nonce}", f"ENROLLED {nonce}")}
                    return
                else:
                    raise ValueError("unexpected authentication response")
    raise ValueError("too many authentication responses")


def status(uid):
    ready = False
    enrolled = False
    try:
        enrolled = list(auth_request("STATUS", uid))[-1]["success"]
        ready = True
    except (OSError, ValueError):
        pass
    pam = Path("/etc/pam.d/kde")
    integrated = pam.exists() and "# BEGIN FACEPASS" in pam.read_text()
    return {"installed": SYSTEM_PYTHON.exists(), "broker_available": ready,
            "enrolled": enrolled, "kde_pam_integrated": integrated,
            "authentication_enabled": ready and enrolled and integrated,
            "experimental_rgb_authentication": True, "password_fallback": True}


def enroll(uid, emit, *, replace=False, preview=False, connected=lambda: True):
    if os.geteuid() != 0:
        raise PermissionError("Enroll through FacePass Setup and the administrator dialog.")
    user = pwd.getpwuid(checked_uid(uid))
    if user.pw_shell in ("/sbin/nologin", "/usr/sbin/nologin", "/bin/false"):
        raise ValueError("account cannot log in")
    store = EnrollmentStore()
    if store.enrolled(uid) and not replace:
        raise FileExistsError("Account already enrolled; choose Replace enrollment explicitly.")
    lock_fd = os.open(AUTH_SOCKET.parent / "camera.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(lock_fd)
        if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("insecure camera lock")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(dir=AUTH_SOCKET.parent, prefix="enrollment-") as directory:
            return _enroll_and_verify(uid, emit, store, EnrollmentStore(directory),
                                      replace=replace, preview=preview, connected=connected)
    finally:
        os.close(lock_fd)


def _enroll_and_verify(uid, emit, store, candidate, *, replace, preview, connected):
    value = supervised_auth_capture("enroll", uid, emit, connected, preview=preview)
    if not value["success"]:
        return value
    # Keep the old enrollment until a second fresh challenge verifies the new
    # templates. The candidate store enforces the same private-file schema.
    candidate.write(uid, value["templates"])
    emit({"kind": "info", "message": "Enrollment captured. FacePass is checking a fresh challenge before saving."})
    command = [str(SYSTEM_PYTHON), "-I", "-m", "facepass.auth_capture", "--mode", "authenticate",
               "--uid", str(uid), "--template-directory", str(candidate.path)] + (["--preview"] if preview else [])
    verified = supervised_auth_capture("authenticate", uid, emit, connected, preview=preview, command=command)
    if not verified["success"]:
        return {"success": False, "reason": "new_enrollment_verification_failed"}
    if not connected():
        return {"success": False, "reason": "enrollment_cancelled"}
    store.write(uid, value["templates"], replace=replace)
    return {"success": True, "reason": "enrollment_saved_and_verified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("status", "verify", "enroll", "remove"):
        item = sub.add_parser(operation)
        item.add_argument("--uid", type=int, default=os.getuid())
        if operation == "enroll":
            item.add_argument("--replace", action="store_true")
            item.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    def connected():
        # A closed preview pipe cancels root capture even when the desktop user
        # cannot send a signal to the administrator process started by pkexec.
        poller = select.poll()
        poller.register(sys.stdout.fileno(), select.POLLERR | select.POLLHUP)
        return not poller.poll(0)

    def emit(value):
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        uid = checked_uid(args.uid)
        if os.geteuid() != 0 and uid != os.getuid():
            raise PermissionError("Only your own account can be queried.")
        if args.operation == "status":
            emit(status(uid))
            return 0
        if args.operation == "verify":
            final = None
            for value in auth_request("AUTH", uid):
                emit(value)
                final = value
            return 0 if final and final["success"] else 2
        if args.operation == "remove":
            EnrollmentStore().remove(uid)
            emit({"kind": "final", "success": True, "reason": "enrollment_removed"})
            return 0
        value = enroll(uid, emit, replace=args.replace, preview=args.preview, connected=connected)
        emit({"kind": "final", **value})
        return 0 if value["success"] else 2
    except (Exception, KeyboardInterrupt) as error:
        # Never include private vectors or captured data in diagnostic errors.
        if connected():
            emit({"kind": "final", "success": False, "reason": "operation_failed",
                  "message": str(error) if isinstance(error, (PermissionError, FileExistsError)) else "FacePass setup failed; use password login and check installation status."})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
