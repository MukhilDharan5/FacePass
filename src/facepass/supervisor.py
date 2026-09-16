"""Bound model/camera hangs and validate diagnostics before exposing them."""

import json
import math
import os
import re
import selectors
import subprocess
import time

from .results import SIGNAL_NAMES, TrackingState, failure_result


MAX_MESSAGE_BYTES = 2_000_000


def _code(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,80}", value) is not None


def _number(value, low=0, high=1):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and low <= value <= high)


def validate_result(value):
    template = failure_result("template")
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError("unknown result schema")
    # Fusion is unimplemented: individual weak evidence can never enable LIVE.
    if (value["decision"] not in ("UNCERTAIN", "INSUFFICIENT_QUALITY")
            or not _number(value["confidence"]) or value["confidence"] != 0.0):
        raise ValueError("unsupported decision")
    if not _code(value["reason"]) or not isinstance(value["terminal"], bool):
        raise ValueError("invalid result state")
    for key, maximum in (("frame_count", 1000), ("face_count", 4), ("landmark_count", 478),
                         ("observation_duration_ms", 60_000)):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or not 0 <= value[key] <= maximum:
            raise ValueError("invalid result counter")
    if (not isinstance(value["flags"], list) or len(value["flags"]) > 20
            or not all(_code(flag) for flag in value["flags"])
            or "liveness_not_implemented" not in value["flags"]):
        raise ValueError("invalid flags")
    scores = value["signals"]
    if not isinstance(scores, dict) or set(scores) != set(SIGNAL_NAMES):
        raise ValueError("unknown signals")
    for name, score in scores.items():
        if (not isinstance(score, dict) or set(score) != set(template["signals"]["blink"])
                or not _number(score["confidence"]) or not _number(score["reliability"])
                or not isinstance(score["available"], bool)
                or not _code(score["diagnostic_information"])):
            raise ValueError("invalid evidence")
        if not score["available"] and score["reliability"] != 0:
            raise ValueError("unavailable evidence cannot have reliability")
        if name not in ("head_motion", "facial_geometry") and (score["available"] or score["reliability"] != 0):
            raise ValueError("unimplemented detector evidence")
        if name in ("head_motion", "facial_geometry") and (score["confidence"] > 0.68 + 1e-9 or score["reliability"] > 0.4):
            raise ValueError("motion evidence exceeds experimental caps")
    quality = value["quality"]
    if not isinstance(quality, dict) or set(quality) != set(template["quality"]):
        raise ValueError("unknown quality fields")
    for key, measurement in quality.items():
        if key == "acceptable":
            if not isinstance(measurement, bool):
                raise ValueError("invalid quality state")
        elif key == "orientation_acceptable":
            if measurement is not None and not isinstance(measurement, bool):
                raise ValueError("invalid orientation")
        elif key == "occlusion":
            if measurement != "not_measured":
                raise ValueError("unsupported occlusion evidence")
        elif key == "reasons":
            if not isinstance(measurement, list) or len(measurement) > 20 or not all(_code(r) for r in measurement):
                raise ValueError("invalid quality reasons")
        elif measurement is not None and not _number(measurement, 0, 1_000_000):
            raise ValueError("invalid quality measurement")
    tracking = value["tracking"]
    if not isinstance(tracking, dict) or set(tracking) != set(template["tracking"]):
        raise ValueError("unknown tracking fields")
    if tracking["state"] not in {state.value for state in TrackingState}:
        raise ValueError("unknown tracking state")
    active = tracking["state"] in ("ACQUIRING", "STABLE")
    track_id = tracking["track_id"]
    if active:
        if not isinstance(track_id, int) or isinstance(track_id, bool) or not 1 <= track_id <= 1000:
            raise ValueError("invalid track identifier")
        if value["terminal"] or value["face_count"] != 1 or not quality["acceptable"]:
            raise ValueError("invalid active track")
    elif track_id is not None:
        raise ValueError("inactive track cannot identify a current track")
    for key, maximum in (("continuous_frames", 1000), ("history_frame_count", 128),
                         ("track_duration_ms", 60_000), ("reset_count", 1000)):
        if not isinstance(tracking[key], int) or isinstance(tracking[key], bool) or not 0 <= tracking[key] <= maximum:
            raise ValueError("invalid tracking counter")
    if active:
        if not 1 <= tracking["history_frame_count"] <= tracking["continuous_frames"]:
            raise ValueError("invalid tracking history")
        if tracking["state"] == "STABLE" and tracking["continuous_frames"] < 3:
            raise ValueError("invalid stable track")
    elif any(tracking[key] for key in ("history_frame_count", "continuous_frames", "track_duration_ms")):
        raise ValueError("inactive track retains history")
    for key in ("appearance_available", "history_reset"):
        if not isinstance(tracking[key], bool):
            raise ValueError("invalid tracking availability")
    if not _code(tracking["event"]) or not _code(tracking["reset_reason"]):
        raise ValueError("invalid tracking event")
    if not _number(tracking["continuity_score"]):
        raise ValueError("invalid tracking score")
    for key, maximum in (("center_displacement", 100), ("scale_change", 100),
                         ("shape_residual", 2), ("appearance_change", 1)):
        if tracking[key] is not None and not _number(tracking[key], 0, maximum):
            raise ValueError("invalid tracking measurement")
    motion = value["motion"]
    if not isinstance(motion, dict) or set(motion) != set(template["motion"]):
        raise ValueError("unknown motion fields")
    for key, measurement in motion.items():
        if key == "status":
            if measurement not in ("not_observed", "observing", "historical_diagnostics"):
                raise ValueError("invalid motion status")
        elif isinstance(template["motion"][key], bool):
            if not isinstance(measurement, bool):
                raise ValueError("invalid motion availability")
        elif key in ("window_frame_count", "window_duration_ms", "background_point_count"):
            limit = {"window_frame_count": 64, "window_duration_ms": 2000, "background_point_count": 80}[key]
            if isinstance(measurement, bool) or not isinstance(measurement, int) or not 0 <= measurement <= limit:
                raise ValueError("invalid motion counter")
        elif measurement is not None:
            low, high = (-180, 180) if key.endswith("degrees") else (0, 100_000)
            if key in ("temporal_coherence", "background_inlier_fraction", "nonplanar_fraction"):
                low, high = 0, 1
            if not _number(measurement, low, high):
                raise ValueError("invalid motion measurement")
    if motion["historical"]:
        if not value["terminal"] or tracking["state"] != "CLOSED" or motion["status"] != "historical_diagnostics":
            raise ValueError("invalid historical motion diagnostics")
    for name in ("head_motion", "facial_geometry"):
        if scores[name]["available"] and (value["terminal"] or tracking["state"] != "STABLE"
                                         or not quality["acceptable"] or motion["historical"]
                                         or motion["window_frame_count"] < 6 or motion["window_duration_ms"] < 450):
            raise ValueError("motion evidence outside valid observation")
    return value


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def supervise(command, *, duration, on_update=lambda result, preview: None,
              pump=lambda: True, startup_timeout=20.0, frame_timeout=2.0):
    """Private worker diagnostics only; never an authentication IPC endpoint."""
    process = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        created = last_update = time.monotonic()
        started = None
        final = None
        buffer = bytearray()
        while True:
            if not pump():
                return failure_result("cancelled")
            now = time.monotonic()
            if started is None and now - created > startup_timeout:
                return failure_result("worker_startup_timeout")
            if started is not None:
                if now - started > duration + 0.5:
                    return failure_result("worker_observation_timeout")
                if final is None and now - last_update > frame_timeout:
                    return failure_result("camera_or_model_stalled")
            if not selector.select(timeout=0.05):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                if buffer or final is None:
                    return failure_result("worker_exited_without_result")
                try:
                    exit_code = process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    return failure_result("worker_cleanup_stalled")
                return final if exit_code == 0 else failure_result("worker_crashed")
            buffer.extend(chunk)
            if len(buffer) > MAX_MESSAGE_BYTES:
                return failure_result("oversized_worker_message")
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                message = json.loads(line)
                if not isinstance(message, dict) or final is not None:
                    return failure_result("invalid_worker_protocol")
                kind = message.get("kind")
                if kind == "started" and set(message) == {"kind"} and started is None:
                    started = last_update = time.monotonic()
                elif kind in ("update", "final"):
                    if set(message) not in ({"kind", "result"}, {"kind", "result", "preview"}):
                        return failure_result("invalid_worker_protocol")
                    result = validate_result(message["result"])
                    if kind == "final":
                        if not result["terminal"] or "preview" in message:
                            return failure_result("invalid_worker_protocol")
                        final = result
                    else:
                        if started is None:
                            return failure_result("invalid_worker_protocol")
                        preview = message.get("preview")
                        if preview is not None and (not isinstance(preview, str) or len(preview) > 1_800_000):
                            return failure_result("invalid_preview")
                        last_update = time.monotonic()
                        on_update(result, preview)
                else:
                    return failure_result("invalid_worker_protocol")
    except KeyboardInterrupt:
        return failure_result("cancelled")
    except Exception:
        return failure_result("harness_or_ipc_error")
    finally:
        selector.close()
        if process is not None:
            stop_worker(process)
            process.stdout.close()
