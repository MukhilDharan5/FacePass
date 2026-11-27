"""Exercise the real daemon/supervisor/model path using a nonexistent capture node."""

import argparse
import json
from pathlib import Path
import tempfile
import threading

from facepass.daemon import PROTOCOL_VERSION, ResearchServer, stream_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    args = parser.parse_args(argv)
    device = "/dev/video999999"
    if Path(device).exists():
        raise RuntimeError("smoke check requires its capture node to be absent")
    with tempfile.TemporaryDirectory(prefix="facepass-daemon-smoke-") as directory:
        path = Path(directory) / "daemon.sock"
        with ResearchServer(path, args.model) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05})
            thread.start()
            try:
                status = list(stream_request(path, {"version": PROTOCOL_VERSION, "operation": "status", "request_id": "status"}))[-1]
                if status["active_request_id"] is not None or status["authentication_enabled"]:
                    raise RuntimeError("invalid idle service state")
                messages = list(stream_request(path, {"version": PROTOCOL_VERSION, "operation": "observe", "request_id": "missing_camera",
                                                     "options": {"device": device, "duration": .5, "fixed_window": True}}))
                final = messages[-1]
                if final["kind"] != "final" or final["result"]["reason"] != "camera_open_failed" or final["result"]["confidence"] != 0:
                    raise RuntimeError("real missing-camera daemon path failed")
                print(json.dumps({"daemon_protocol_verified": True, "real_worker_verified": True,
                                  "decision": final["result"]["decision"], "reason": final["result"]["reason"],
                                  "camera_opened": False, "authentication_enabled": False}))
            finally:
                server.request_stop()
                server.shutdown()
                thread.join(timeout=2)
        if path.exists():
            raise RuntimeError("daemon socket was not cleaned up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
