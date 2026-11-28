"""Foreground, same-user Unix-socket research service. Never authenticates."""

import argparse
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
import threading
import time

from . import __version__
from .evaluation import strict_json
from .results import failure_result
from .runtime import CaptureOptions
from .supervisor import supervise, validate_result


PROTOCOL_VERSION = 2  # Exposure-aware quality diagnostics extend the strict result schema.
MAX_REQUEST_BYTES = 4096
REQUEST_TIMEOUT = 3.0


def default_socket():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path("/tmp")
    return base / ("facepass" if runtime else f"facepass-{os.getuid()}") / "daemon.sock"


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) is not None


def validate_request(value):
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported_protocol")
    if not identifier(value.get("request_id")):
        raise ValueError("invalid_request_id")
    operation = value.get("operation")
    base = {"version", "operation", "request_id"}
    if operation == "observe":
        if set(value) != base | {"options"} or not isinstance(value["options"], dict):
            raise ValueError("invalid_request")
        try:
            CaptureOptions(**value["options"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid_capture_options") from error
    elif operation == "cancel":
        if set(value) != base | {"target_id"} or not identifier(value["target_id"]):
            raise ValueError("invalid_cancel_request")
    elif operation == "status":
        if set(value) != base:
            raise ValueError("invalid_request")
    else:
        raise ValueError("unknown_operation")
    return value


def private_socket_parent(path):
    path = Path(os.path.abspath(path))
    if len(os.fsencode(path)) > 100 or path.parent.resolve() != path.parent:
        raise ValueError("socket path must be short and have no symlink parents")
    try:
        path.parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("socket parent must be an owned private 0700 directory on a POSIX filesystem")
    if path.exists() or path.is_symlink():
        raise FileExistsError("socket path already exists; stop its owner before removing a stale socket")
    return path


class ResearchServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """At most one worker and eight connections; idle never opens a camera/model."""

    daemon_threads = False
    block_on_close = True
    request_queue_size = 8

    def __init__(self, path, model, *, runner=None):
        self.path = private_socket_parent(path)
        self.model = str(Path(model).resolve())
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.active_id = self.cancel_event = None
        self.slots = threading.BoundedSemaphore(8)
        self.runner = runner or self._run_capture
        self.socket_identity = None
        super().__init__(str(self.path), RequestHandler)

    def server_bind(self):
        super().server_bind()
        info = self.path.lstat()
        self.socket_identity = (info.st_dev, info.st_ino)
        os.chmod(self.path, 0o600)
        info = self.path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("socket permissions are not private")

    def verify_request(self, request, client_address):
        try:
            _, uid, _ = struct.unpack("3i", request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            return uid == os.getuid() and not self.stopping.is_set()
        except OSError:
            return False

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Avoid arbitrary request content or biometric data in tracebacks/logs.
        pass

    def _run_capture(self, options, *, on_update, pump):
        return supervise(options.worker_command(self.model), duration=options.duration,
                         extension_duration=options.maximum_extension,
                         on_update=on_update, pump=pump)

    def request_stop(self):
        self.stopping.set()
        with self.lock:
            if self.cancel_event is not None:
                self.cancel_event.set()

    def server_close(self):
        self.request_stop()
        super().server_close()
        try:
            info = self.path.lstat()
            if (info.st_dev, info.st_ino) == self.socket_identity:
                self.path.unlink()
        except FileNotFoundError:
            pass


class RequestHandler(socketserver.BaseRequestHandler):
    def send(self, kind, **fields):
        message = {"version": PROTOCOL_VERSION, "request_id": self.request_id, "kind": kind, **fields}
        self.request.sendall((json.dumps(message, allow_nan=False) + "\n").encode())

    def handle(self):
        self.request_id = None
        deadline = time.monotonic() + REQUEST_TIMEOUT
        self.request.settimeout(REQUEST_TIMEOUT)
        try:
            data = bytearray()
            while b"\n" not in data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.request.settimeout(remaining)
                chunk = self.request.recv(min(1024, MAX_REQUEST_BYTES + 1 - len(data)))
                if not chunk:
                    return
                data.extend(chunk)
                if len(data) > MAX_REQUEST_BYTES:
                    raise ValueError("oversized_request")
            line, _, remainder = data.partition(b"\n")
            if remainder:
                raise ValueError("one_request_per_connection")
            request = validate_request(strict_json(line))
            self.request_id = request["request_id"]
            self.request.settimeout(.5)
            if request["operation"] == "status":
                with self.server.lock:
                    self.send("status", engine_version=__version__, authentication_enabled=False,
                              active_request_id=self.server.active_id)
            elif request["operation"] == "cancel":
                with self.server.lock:
                    found = self.server.active_id == request["target_id"]
                    if found:
                        self.server.cancel_event.set()
                self.send("cancel_requested" if found else "error", **({} if found else {"error": "unknown_attempt"}))
            else:
                self.observe(request)
        except (ValueError, UnicodeError, RecursionError) as error:
            try:
                reason = str(error)
                self.send("error", error=reason if re.fullmatch(r"[a-z_]{1,80}", reason) else "invalid_json_or_request")
            except OSError:
                pass
        except OSError:
            pass

    def observe(self, request):
        cancelled = threading.Event()
        with self.server.lock:
            if self.server.active_id is not None or self.server.stopping.is_set():
                self.send("error", error="busy")
                return
            self.server.active_id = self.request_id
            self.server.cancel_event = cancelled

        def connected():
            if cancelled.is_set() or self.server.stopping.is_set():
                return False
            readable, _, _ = select.select([self.request], [], [], 0)
            # Any additional bytes or EOF violate the one-request streaming contract.
            return not readable

        def update(result, preview):
            self.send("update", result=validate_result(result))

        try:
            self.send("accepted", authentication_enabled=False)
            try:
                result = self.server.runner(CaptureOptions(**request["options"]), on_update=update, pump=connected)
                result = validate_result(strict_json(json.dumps(result, allow_nan=False)))
                if not result["terminal"]:
                    raise ValueError("missing terminal result")
            except Exception:
                result = failure_result("daemon_worker_error")
            # The supervised worker has exited before publishing its final
            # result. Make the capture slot available for the next attempt.
            with self.server.lock:
                if self.server.cancel_event is cancelled:
                    self.server.active_id = self.server.cancel_event = None
            self.send("final", result=result)
        finally:
            cancelled.set()
            with self.server.lock:
                if self.server.cancel_event is cancelled:
                    self.server.active_id = self.server.cancel_event = None


def stream_request(path, request, *, timeout=60.0):
    """Yield JSON responses from one local connection. Do not half-close observe."""
    validate_request(request)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        if uid != os.getuid():
            raise ValueError("daemon peer must have the same user ID")
        connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
        with connection.makefile("rb") as source:
            while True:
                line = source.readline(100_001)
                if not line:
                    raise ValueError("daemon disconnected without a terminal response")
                if len(line) > 100_000 or not line.endswith(b"\n"):
                    raise ValueError("invalid daemon response size")
                message = strict_json(line)
                if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION or message.get("request_id") != request["request_id"]:
                    raise ValueError("invalid daemon response")
                kind = message.get("kind")
                base = {"version", "request_id", "kind"}
                expected = {"update": {"result"}, "final": {"result"}, "accepted": {"authentication_enabled"},
                            "status": {"engine_version", "authentication_enabled", "active_request_id"},
                            "cancel_requested": set(), "error": {"error"}}
                if kind not in expected or set(message) != base | expected[kind]:
                    raise ValueError("unknown daemon response schema")
                if kind in ("accepted", "status") and message["authentication_enabled"] is not False:
                    raise ValueError("unsupported authentication service")
                if kind in ("update", "final"):
                    validate_result(message["result"])
                    if kind == "final" and not message["result"]["terminal"]:
                        raise ValueError("nonterminal final response")
                elif kind not in ("accepted", "status", "cancel_requested", "error"):
                    raise ValueError("unknown daemon response")
                yield message
                if kind in ("final", "status", "cancel_requested", "error"):
                    return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket())
    from .model_assets import default_model_path
    parser.add_argument("--model", default=default_model_path())
    args = parser.parse_args(argv)
    try:
        with ResearchServer(args.socket, args.model) as server:
            def stop(signum, frame):
                server.request_stop()
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            server.timeout = .1
            print(json.dumps({"socket": str(server.path), "version": PROTOCOL_VERSION,
                              "engine_version": __version__, "authentication_enabled": False}), flush=True)
            while not server.stopping.is_set():
                server.handle_request()
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f"daemon error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
