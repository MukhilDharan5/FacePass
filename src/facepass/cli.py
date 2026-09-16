"""Run from GNOME Terminal; optional in-memory Tk preview via XWayland."""

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from .results import failure_result


class Preview:
    def __init__(self):
        import tkinter as tk
        self._tk = tk
        self._root = tk.Tk()
        self._root.title("FacePass — motion diagnostics, no authentication")
        self._label = tk.Label(self._root, text="Starting local capture…")
        self._label.pack()
        self._status = tk.Label(self._root, text="Experimental motion; LIVE disabled", wraplength=640)
        self._status.pack()
        self._open = True
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda event: self.close())

    def close(self):
        if self._open:
            self._open = False
            self._label.configure(image="")
            self._root.destroy()
            self._image = None

    def pump(self):
        if self._open:
            self._root.update()
        return self._open

    def update(self, result, encoded):
        if encoded and self._open:
            self._image = self._tk.PhotoImage(data=encoded)
            self._label.configure(image=self._image)
            self._status.configure(text=f"{result['decision']}: {result['reason']} | "
                                        f"faces={result['face_count']} | "
                                        f"track={result['tracking']['track_id']} "
                                        f"{result['tracking']['state']} | "
                                        f"quality={','.join(result['quality']['reasons']) or 'acceptable'} | "
                                        f"head={result['signals']['head_motion']['confidence']:.2f} "
                                        f"geometry={result['signals']['facial_geometry']['confidence']:.2f} | "
                                        "LIVE disabled")


def doctor(model):
    packages = {}
    for name in ("mediapipe", "numpy", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps({
        "python": sys.version.split()[0], "platform": sys.platform,
        "session_type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
        "model_present": Path(model).is_file(), "packages": packages,
        "tk_preview_available": importlib.util.find_spec("tkinter") is not None,
        "cameras": [{"device": device, "readable": os.access(device, os.R_OK),
                     "writable": os.access(device, os.W_OK)} for device in sorted(glob.glob("/dev/video[0-9]*"))],
        "milestone": 3, "authentication_enabled": False,
    }, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Fedora V4L2 tracking/motion diagnostic harness. Never authenticates.")
    parser.add_argument("--doctor", action="store_true", help="Inspect dependencies and device access without opening camera")
    parser.add_argument("--model", default="models/face_landmarker.task", help="Existing local MediaPipe model")
    parser.add_argument("--device", default="/dev/video0", help="Local /dev/videoN webcam; no URLs or video files")
    parser.add_argument("--duration", type=float, default=5.0, help="Observation limit in seconds (0.5 to 30)")
    parser.add_argument("--fps", type=int, default=12, help="Processing rate (5 to 30)")
    parser.add_argument("--preview", action="store_true", help="Display transient annotated frames in an optional Tk window")
    parser.add_argument("--json", action="store_true", help="Print scalar diagnostic JSON updates and final result")
    parser.add_argument("--save-diagnostic-frames", metavar="NEW_PRIVATE_DIRECTORY",
                        help="WARNING: retain biometric PNGs, up to one per second; disabled by default")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if not 0.5 <= options.duration <= 30 or not 5 <= options.fps <= 30:
        parser.error("duration must be 0.5..30 seconds and fps must be 5..30")
    if options.doctor:
        return doctor(options.model)
    if options.worker:
        from .worker import run
        run(options)
        return 0
    from .supervisor import supervise

    if options.save_diagnostic_frames:
        print("WARNING: BIOMETRIC IMAGES WILL BE RETAINED on disk in "
              f"{options.save_diagnostic_frames}. Disable --save-diagnostic-frames for normal use.", file=sys.stderr)
    preview = None
    if options.preview:
        try:
            preview = Preview()
        except Exception:
            print(json.dumps(failure_result("preview_unavailable")))
            print("Preview needs tkinter and a GNOME desktop display; use terminal mode or "
                  "install Fedora's python3-tkinter package.", file=sys.stderr)
            return 2
    forwarded = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", "facepass", "--worker", *forwarded]
    last_print = float("-inf")

    def update(result, encoded):
        nonlocal last_print
        if preview:
            preview.update(result, encoded)
        if options.json:
            print(json.dumps({"kind": "update", "result": result}, allow_nan=False), flush=True)
        elif time.monotonic() - last_print >= 0.5:
            quality = result["quality"]
            tracking = result["tracking"]
            motion = result["motion"]
            print(f"{result['observation_duration_ms']:4d}ms {result['decision']} "
                  f"faces={result['face_count']} landmarks={result['landmark_count']} "
                  f"track={tracking['track_id']}:{tracking['state']} "
                  f"history={tracking['history_frame_count']} resets={tracking['reset_count']} "
                  f"quality={','.join(quality['reasons']) or 'acceptable'} "
                  f"head={result['signals']['head_motion']['confidence']:.2f} "
                  f"geometry={result['signals']['facial_geometry']['confidence']:.2f} "
                  f"relative_motion_px={motion['independent_motion_px']} "
                  f"planar_error_px={motion['planar_holdout_error_px']}", flush=True)
            last_print = time.monotonic()

    if not options.json:
        print("Milestone 3: capture, tracking and experimental motion evidence. LIVE is unavailable. Ctrl+C cancels.", flush=True)
    try:
        result = supervise(command, duration=options.duration, on_update=update,
                           pump=preview.pump if preview else lambda: True)
    finally:
        if preview:
            preview.close()
    if options.json:
        print(json.dumps({"kind": "final", "result": result}, allow_nan=False))
    else:
        print(f"Final: {result['decision']} ({result['reason']}), confidence={result['confidence']}")
        tracking = result["tracking"]
        print(f"  tracking: {tracking['state']}, retained_history={tracking['history_frame_count']}, "
              f"resets={tracking['reset_count']}, reset_reason={tracking['reset_reason']}")
        for name, score in result["signals"].items():
            print(f"  {name}: confidence={score['confidence']:.2f}, reliability={score['reliability']:.2f}, "
                  f"available={score['available']} ({score['diagnostic_information']})")
    # No successful authentication exit status, including for clean observations.
    return 2
