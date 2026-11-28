"""Fedora GNOME/KDE terminal harness; optional in-memory Tk X11/XWayland preview."""

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
from .runtime import add_capture_arguments, options_from_args
from .desktop import preview_diagnostics, preview_failure_hint, session_diagnostics
from .model_assets import default_model_path


class Preview:
    def __init__(self):
        import tkinter as tk
        self._tk = tk
        self._root = tk.Tk()
        self._root.title("FacePass — adaptive observation diagnostics")
        self._label = tk.Label(self._root, text="Starting local capture…")
        self._label.pack()
        self._status = tk.Label(self._root, text="Adaptive observation/temporal diagnostics; LIVE disabled", wraplength=640)
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
                                        f"warnings={','.join(result['quality']['warnings']) or 'none'} | "
                                        f"head={result['signals']['head_motion']['confidence']:.2f} "
                                        f"geometry={result['signals']['facial_geometry']['confidence']:.2f} "
                                        f"blink={result['signals']['blink']['confidence']:.2f} "
                                        f"{result['blink']['state']} count={result['blink']['blink_count']} | "
                                        f"gaze={result['signals']['gaze']['confidence']:.2f} "
                                        f"available={result['signals']['gaze']['available']} | "
                                        f"lighting={result['signals']['illumination_consistency']['confidence']:.2f} "
                                        f"available={result['signals']['illumination_consistency']['available']} | "
                                        f"replay={result['signals']['replay_artifact_detection']['confidence']:.2f} "
                                        f"suspect={result['replay']['replay_artifact_observed']} | "
                                        f"temporal={result['signals']['temporal_consistency']['confidence']:.2f} "
                                        f"groups={result['temporal']['positive_group_count']} "
                                        f"conflict={result['temporal']['conflict_observed']} | "
                                        f"observation={result['observation']['status']} "
                                        f"elapsed={result['observation']['elapsed_ms']}ms | "
                                        "LIVE disabled")


def doctor(model):
    packages = {}
    for name in ("mediapipe", "numpy", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tk_available = importlib.util.find_spec("tkinter") is not None
    print(json.dumps({
        "python": sys.version.split()[0], "platform": sys.platform,
        **session_diagnostics(),
        "model_present": Path(model).is_file(), "packages": packages,
        "tk_preview_available": tk_available,
        "preview": preview_diagnostics(tk_available),
        "cameras": [{"device": device, "readable": os.access(device, os.R_OK),
                     "writable": os.access(device, os.W_OK)} for device in sorted(glob.glob("/dev/video[0-9]*"))],
        "milestone": 12, "authentication_enabled": False,
    }, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Fedora V4L2 adaptive-observation diagnostics. Never authenticates.")
    parser.add_argument("--doctor", action="store_true", help="Inspect dependencies and device access without opening camera")
    parser.add_argument("--model", default=default_model_path(), help="Existing local MediaPipe model")
    add_capture_arguments(parser)
    parser.add_argument("--profile", action="store_true", help="Report bounded scalar capture timing and CPU diagnostics")
    parser.add_argument("--preview", action="store_true", help="Display transient annotated frames in an optional Tk window")
    parser.add_argument("--json", action="store_true", help="Print scalar diagnostic JSON updates and final result")
    parser.add_argument("--save-diagnostic-frames", metavar="NEW_PRIVATE_DIRECTORY",
                        help="WARNING: retain biometric PNGs, up to one per second; disabled by default")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    try:
        capture = options_from_args(options)
    except ValueError as error:
        parser.error(str(error))
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
        except Exception as error:
            print(json.dumps(failure_result("preview_unavailable")))
            print(preview_failure_hint(error), file=sys.stderr)
            return 2
    forwarded = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", "facepass", "--worker", *forwarded]
    last_print = float("-inf")
    from .performance import CaptureProfile
    profile = CaptureProfile() if options.profile else None

    def update(result, encoded):
        nonlocal last_print
        if profile:
            profile.update(result)
        if preview:
            preview.update(result, encoded)
        if options.json:
            print(json.dumps({"kind": "update", "result": result}, allow_nan=False), flush=True)
        elif time.monotonic() - last_print >= 0.5:
            quality = result["quality"]
            tracking = result["tracking"]
            motion = result["motion"]
            blink = result["blink"]
            gaze = result["gaze"]
            lighting = result["illumination"]
            replay = result["replay"]
            temporal = result["temporal"]
            fusion = result["fusion"]
            observation = result["observation"]
            print(f"{result['observation_duration_ms']:4d}ms {result['decision']} "
                  f"faces={result['face_count']} landmarks={result['landmark_count']} "
                  f"track={tracking['track_id']}:{tracking['state']} "
                  f"history={tracking['history_frame_count']} resets={tracking['reset_count']} "
                  f"quality={','.join(quality['reasons']) or 'acceptable'} "
                  f"warnings={','.join(quality['warnings']) or 'none'} "
                  f"head={result['signals']['head_motion']['confidence']:.2f} "
                  f"geometry={result['signals']['facial_geometry']['confidence']:.2f} "
                  f"blink={result['signals']['blink']['confidence']:.2f} "
                  f"blink_state={blink['state']} blinks={blink['blink_count']} "
                  f"blink_available={result['signals']['blink']['available']} "
                  f"gaze={result['signals']['gaze']['confidence']:.2f} "
                  f"gaze_available={result['signals']['gaze']['available']} "
                  f"gaze_motion={gaze['independent_motion_observed']} "
                  f"lighting={result['signals']['illumination_consistency']['confidence']:.2f} "
                  f"lighting_available={result['signals']['illumination_consistency']['available']} "
                  f"lighting_response={lighting['consistent_response_observed']} "
                  f"replay={result['signals']['replay_artifact_detection']['confidence']:.2f} "
                  f"replay_available={result['signals']['replay_artifact_detection']['available']} "
                  f"replay_suspect={replay['replay_artifact_observed']} "
                  f"temporal={result['signals']['temporal_consistency']['confidence']:.2f} "
                  f"temporal_available={result['signals']['temporal_consistency']['available']} "
                  f"groups={temporal['positive_group_count']}/{fusion['available_group_count']} "
                  f"overlap={temporal['support_overlap']:.2f} conflict={temporal['conflict_observed']} "
                  f"fusion_adjusted={fusion['adjusted_combined_score']:.2f} "
                  f"observation={observation['status']} "
                  f"limit={observation['effective_limit_ms']}ms "
                  f"extensions={observation['extension_count']} "
                  f"relative_motion_px={motion['independent_motion_px']} "
                  f"planar_error_px={motion['planar_holdout_error_px']}", flush=True)
            last_print = time.monotonic()

    if not options.json:
        print("FacePass research capture: bounded adaptive observation; authentication disabled. "
              f"Quality profile: {capture.quality_profile}. Ctrl+C cancels.", flush=True)
    try:
        result = supervise(command, duration=options.duration, on_update=update,
                           pump=preview.pump if preview else lambda: True,
                           extension_duration=capture.maximum_extension)
    finally:
        if preview:
            preview.close()
    if options.json:
        print(json.dumps({"kind": "final", "result": result}, allow_nan=False))
    else:
        print(f"Final: {result['decision']} ({result['reason']}), confidence={result['confidence']}")
        quality = result["quality"]
        print(f"  quality: {','.join(quality['reasons']) or 'acceptable'}")
        print(f"  quality profile: {quality['profile']}; "
              f"warnings: {','.join(quality['warnings']) or 'none'}; "
              f"landmark low-light enhancement: {quality['landmark_enhanced']}")
        measurements = [f"{name}={quality[name]:.2f}" for name in
                        ("blur_variance", "evaluated_blur_variance", "contrast_span", "mean_luminance", "face_size_px")
                        if quality[name] is not None]
        if measurements:
            print("  quality measurements: " + ", ".join(measurements))
        if result["reason"] == "observation_timeout" and not quality["acceptable"]:
            print("  Observation ended with insufficient capture quality; detector history was cleared.")
        if "blur" in quality["reasons"]:
            print("  Try brighter, even lighting, a clean camera lens, and keeping your face steady.")
        tracking = result["tracking"]
        print(f"  tracking: {tracking['state']}, retained_history={tracking['history_frame_count']}, "
              f"resets={tracking['reset_count']}, reset_reason={tracking['reset_reason']}")
        for name, score in result["signals"].items():
            print(f"  {name}: confidence={score['confidence']:.2f}, reliability={score['reliability']:.2f}, "
                  f"available={score['available']} ({score['diagnostic_information']})")
    if profile:
        print(json.dumps({"kind": "performance", "performance": profile.finish(result)}, allow_nan=False))
    # No successful authentication exit status, including for clean observations.
    return 2
