"""Administrator-owned local authentication broker, separate from research IPC."""

import argparse
import errno
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import threading
import time

from .auth_store import EnrollmentStore, checked_uid
from .evaluation import strict_json

AUTH_SOCKET = Path("/run/facepass/auth.sock")
AUTH_SERVICES = ("kde", "kde-face", "plasmalogin", "sddm")
MAX_WORKER_MESSAGE = 2_000_000


def parse_auth_request(data, peer_uid):
    if len(data) > 256 or not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ValueError("invalid authentication request")
    text = data.decode("ascii")
    parts = text.rstrip("\n").split(" ")
    if (len(parts) != 4 or parts[0] not in ("STATUS", "AUTH")
            or not re.fullmatch(r"[0-9]{4,10}", parts[1]) or parts[2] not in AUTH_SERVICES
            or not re.fullmatch(r"[a-f0-9]{32}", parts[3])):
        raise ValueError("invalid authentication request")
    uid = checked_uid(int(parts[1]))
    if peer_uid not in (0, uid):
        raise PermissionError("authentication account differs from peer")
    return parts[0], uid, parts[2], parts[3]


def supervised_auth_capture(mode, uid, emit, connected, *, preview=False, command=None,
                            timeout=45):
    command = command or [sys.executable, "-I", "-m", "facepass.auth_capture", "--mode", mode,
                          "--uid", str(uid)] + (["--preview"] if preview else [])
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8",
                                    "MPLCONFIGDIR": "/run/facepass/mpl", "OMP_NUM_THREADS": "1",
                                    "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    buffer = bytearray()
    final = None
    deadline = time.monotonic() + timeout
    try:
        fd = process.stdout.fileno()
        os.set_blocking(fd, False)
        while time.monotonic() < deadline and connected():
            ready, _, _ = select.select([fd], [], [], .1)
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > MAX_WORKER_MESSAGE:
                    break
                while b"\n" in buffer:
                    line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    value = strict_json(line)
                    if not isinstance(value, dict):
                        raise ValueError("invalid worker result")
                    if final is not None:
                        raise ValueError("worker output after final")
                    if value.get("kind") == "info" and set(value) == {"kind", "message"}:
                        message = value["message"]
                        if not isinstance(message, str) or not re.fullmatch(r"[a-zA-Z0-9 :.,'-]{1,200}", message):
                            raise ValueError("invalid challenge prompt")
                        emit(value)
                    elif value.get("kind") == "started" and set(value) == {"kind"}:
                        pass
                    elif preview and value.get("kind") == "preview" and set(value) == {"kind", "png"}:
                        if not isinstance(value["png"], str) or len(value["png"]) > 1_900_000:
                            raise ValueError("invalid preview")
                        emit(value)
                    elif value.get("kind") == "final":
                        fields = {"kind", "success", "reason"} | ({"templates"} if mode == "enroll" and value.get("success") is True else set())
                        if (set(value) != fields or type(value.get("success")) is not bool
                                or not isinstance(value.get("reason"), str)
                                or not re.fullmatch(r"[a-z_]{1,80}", value["reason"])):
                            raise ValueError("invalid final authentication result")
                        expected = "enrollment_captured" if mode == "enroll" else "identity_and_active_challenge_verified"
                        if value["success"] and value["reason"] != expected:
                            raise ValueError("invalid authentication success")
                        final = value
                    else:
                        raise ValueError("unknown authentication worker message")
            elif process.poll() is not None:
                break
        if buffer:
            raise ValueError("partial worker result")
        process.wait(timeout=1)
        if (not connected() or final is None or process.returncode != (0 if final["success"] else 2)):
            return {"success": False, "reason": "worker_failed"}
        return {key: value for key, value in final.items() if key != "kind"}
    except (Exception, KeyboardInterrupt):
        return {"success": False, "reason": "worker_failed"}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
        process.stdout.close()


class AuthServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, path=AUTH_SOCKET, *, store=None, runner=None, owner_uid=0):
        self.path = Path(path)
        self.owner_uid = owner_uid
        info = self.path.parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid
                or stat.S_IMODE(info.st_mode) & 0o022 or self.path.parent.resolve() != self.path.parent.absolute()):
            raise ValueError("authentication socket directory is insecure")
        if self.path.exists() or self.path.is_symlink():
            info = self.path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != owner_uid:
                raise FileExistsError("unexpected authentication socket path")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(.2)
                try:
                    probe.connect(str(self.path))
                except OSError as error:
                    if error.errno != errno.ECONNREFUSED:
                        raise FileExistsError("cannot safely replace authentication socket") from error
                else:
                    raise FileExistsError("authentication broker already running")
            self.path.unlink()  # Root-owned stale socket after an interrupted service.
        self.store = store or EnrollmentStore()
        self.runner = runner or supervised_auth_capture
        self.capture_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.attempts = {}
        self.stopping = threading.Event()
        self.slots = threading.BoundedSemaphore(8)
        self.socket_identity = None
        super().__init__(str(self.path), AuthHandler)
        info = self.path.lstat()
        self.socket_identity = (info.st_dev, info.st_ino)
        os.chmod(self.path, 0o666)  # Peer credentials, rather than arbitrary groups, select the account.

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, address):
        pass  # Never dump requests or private enrollment values into logs.

    def rate_allowed(self, uid):
        with self.rate_lock:
            now = time.monotonic()
            self.attempts = {key: rows for key, rows in self.attempts.items() if rows and now - rows[-1] < 60}
            if uid not in self.attempts and len(self.attempts) >= 128:
                return False
            rows = self.attempts.setdefault(uid, deque(maxlen=5))
            while rows and now - rows[0] >= 60:
                rows.popleft()
            if len(rows) == 5:
                return False
            rows.append(now)
            return True

    def server_close(self):
        self.stopping.set()
        super().server_close()
        try:
            info = self.path.lstat()
            if (info.st_dev, info.st_ino) == self.socket_identity:
                self.path.unlink()
        except FileNotFoundError:
            pass


class AuthHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        try:
            _, peer_uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            data = bytearray()
            while b"\n" not in data and len(data) <= 256:
                chunk = self.request.recv(257 - len(data))
                if not chunk:
                    return
                data.extend(chunk)
            operation, uid, service, nonce = parse_auth_request(bytes(data), peer_uid)
            if not self.server.store.enrolled(uid):
                self.request.sendall(f"UNAVAILABLE {nonce}\n".encode())
                return
            if operation == "STATUS":
                self.request.sendall(f"ENROLLED {nonce}\n".encode())
                return
            if not self.server.rate_allowed(uid) or not self.server.capture_lock.acquire(blocking=False):
                self.request.sendall(f"FAILURE {nonce}\n".encode())
                return
            lock_fd = None
            try:
                lock_fd = os.open(self.server.path.parent / "camera.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                info = os.fstat(lock_fd)
                if (info.st_uid != self.server.owner_uid or stat.S_IMODE(info.st_mode) != 0o600
                        or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                    raise ValueError("insecure camera lock")
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                def connected():
                    if self.server.stopping.is_set():
                        return False
                    ready, _, _ = select.select([self.request], [], [], 0)
                    if ready:
                        return False  # EOF or unsolicited input cancels capture.
                    return True

                def emit(value):
                    self.request.sendall(("INFO " + value["message"] + "\n").encode("ascii"))

                value = self.server.runner("authenticate", uid, emit, connected)
                if connected():
                    outcome = "SUCCESS" if value.get("success") is True and value.get("reason") == "identity_and_active_challenge_verified" else "FAILURE"
                    self.request.sendall(f"{outcome} {nonce}\n".encode())
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                self.server.capture_lock.release()
        except (OSError, ValueError, PermissionError):
            return


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("Authentication broker must run as the installed system service.")
    os.umask(0o077)
    with AuthServer() as server:
        def stop(*_):
            server.stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
