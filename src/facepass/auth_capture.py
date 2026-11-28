"""Bounded authentication worker. Private enrollment vectors never reach PAM."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import signal
import time

import cv2
import numpy as np

from .auth_challenge import ActiveChallenge
from .auth_models import AUTH_MODEL_DIR, IdentityModel, identity_similarity
from .auth_store import EnrollmentStore, checked_uid
from .blink import BlinkAnalyzer
from .camera import V4L2Camera
from .head_pose import estimate_pose
from .landmarks import FaceObservation, MediaPipeLandmarker
from .quality import assess_quality, prepare_landmark_frame, validate_frame
from .results import TrackingState
from .tracking import FaceTracker


@dataclass
class AuthFrame:
    timestamp: float
    track_id: int | None = None
    stable: bool = False
    yaw: float | None = None
    blink_count: int = 0
    eyes_open: bool = False
    feature: np.ndarray | None = None
    issue: str = "acquiring"


class CaptureVerifier:
    def __init__(self, backend, identity):
        self.backend, self.identity = backend, identity
        self.tracker = FaceTracker()
        self.blink = BlinkAnalyzer()

    def process(self, frame, timestamp):
        validate_frame(frame)
        image, _ = prepare_landmark_frame(frame)
        faces = self.backend.detect(image, timestamp)
        if not isinstance(faces, FaceObservation):
            raise ValueError("invalid face model output")
        faces.validate()
        if len(faces.faces) > 1:
            raise ValueError("multiple_faces")
        if not faces.faces:
            self.tracker.reset("no_face")
            self.blink.reset()
            return AuthFrame(timestamp, issue="face_not_found")
        mesh = faces.faces[0]
        quality = assess_quality(frame, mesh)
        if not quality.acceptable:
            self.tracker.reset("quality_lost")
            self.blink.reset()
            return AuthFrame(timestamp, issue="capture_quality_insufficient")
        tracking = self.tracker.update(frame, mesh, timestamp)
        if tracking.state == TrackingState.REJECTED:
            raise ValueError("identity_track_changed")
        if tracking.history_reset:
            self.blink.reset()
        stable = tracking.state == TrackingState.STABLE
        blink, _ = self.blink.update(self.tracker.history, stable=stable)
        pose = estimate_pose(mesh, frame.shape)
        feature = self.identity.feature(frame, mesh) if stable else None
        return AuthFrame(timestamp, tracking.track_id, stable,
                         pose.yaw if pose else None, blink.blink_count,
                         blink.eyes_available and blink.state == "OPEN", feature,
                         "observing" if stable else "acquiring")


def capture_attempt(mode, uid, *, model_directory=AUTH_MODEL_DIR, device="/dev/video0",
                    emit=lambda message: None, preview=False, store=None,
                    clock=time.monotonic, camera_factory=V4L2Camera,
                    backend_factory=MediaPipeLandmarker, identity_factory=IdentityModel):
    if mode not in ("authenticate", "enroll"):
        raise ValueError("unknown capture mode")
    checked_uid(uid)
    store = store or EnrollmentStore()
    templates = store.read(uid)["templates"] if mode == "authenticate" else []
    backend = camera = None
    try:
        cv2.setNumThreads(1)
        emit({"kind": "info", "message": "FacePass: starting the camera."})
        backend = backend_factory(Path(model_directory) / "face_landmarker.task")
        identity = identity_factory(model_directory)
        camera = camera_factory(device, 12)
        verifier = CaptureVerifier(backend, identity)
        started = clock()
        challenge = None
        last_prompt = None
        last_preview = last_sample = float("-inf")
        anchor = []
        collected = []
        matches = 0
        emit({"kind": "started"})
        while clock() - started < 30:
            tick = clock()
            frame = camera.read()
            timestamp = clock()
            sample = verifier.process(frame, timestamp)
            if mode == "enroll" and sample.stable and sample.feature is not None:
                if len(anchor) < 3:
                    anchor.append(sample.feature)
                references = anchor
            else:
                references = templates
            matched = (sample.feature is not None and len(references) >= 3
                       and identity_similarity(sample.feature, references) >= (.70 if mode == "enroll" else .60))
            if matched and sample.stable:
                matches += 1
                if challenge is None and matches >= 3 and sample.eyes_open:
                    challenge = ActiveChallenge(clock)
            elif challenge is None:
                matches = 0
            if challenge is not None and not challenge.done:
                challenge.update(timestamp, track_id=sample.track_id, yaw=sample.yaw,
                                 blink_count=sample.blink_count, eyes_open=sample.eyes_open,
                                 identity_match=matched, quality_ok=sample.stable)
                if challenge.failed:
                    return {"success": False, "reason": "challenge_or_identity_lost"}
            if challenge is not None and challenge.done and (
                    not matched or not sample.stable or sample.track_id != challenge.track_id):
                return {"success": False, "reason": "challenge_or_identity_lost"}
            prompt = challenge.prompt if challenge else (
                "FacePass: face the camera, move closer if needed, and keep your eyes open.")
            if prompt != last_prompt:
                emit({"kind": "info", "message": prompt})
                last_prompt = prompt
            if mode == "enroll" and matched and sample.eyes_open and timestamp - last_sample >= .35 and len(collected) < 12:
                collected.append(sample.feature)
                last_sample = timestamp
            if preview and timestamp - last_preview >= .2:
                import base64
                reduced = cv2.resize(frame, (640, 480))
                ok, png = cv2.imencode(".png", reduced)
                if ok:
                    emit({"kind": "preview", "png": base64.b64encode(png).decode("ascii")})
                last_preview = timestamp
            if challenge is not None and challenge.done and matched and sample.stable and matches >= 12:
                if mode == "authenticate":
                    return {"success": True, "reason": "identity_and_active_challenge_verified"}
                if len(collected) >= 8:
                    return {"success": True, "reason": "enrollment_captured",
                            "templates": [row.tolist() for row in collected]}
            if challenge is None and timestamp - started > 7:
                return {"success": False, "reason": "face_or_identity_unavailable"}
            # Inference/capture are isolated under the supervisor's hard timeout.
            remaining = tick + 1 / 12 - clock()
            if remaining > 0:
                time.sleep(min(remaining, .1))
        return {"success": False, "reason": "challenge_timeout"}
    finally:
        if camera is not None:
            camera.close()
        if backend is not None:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("authenticate", "enroll"), required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--template-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.exit(2, "Authentication capture requires the installed administrator-owned service.\n")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    def emit(value):
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        value = capture_attempt(args.mode, args.uid, emit=emit, preview=args.preview,
                                store=EnrollmentStore(args.template_directory) if args.template_directory else None)
    except (Exception, KeyboardInterrupt):
        value = {"success": False, "reason": "capture_failed"}
    emit({"kind": "final", **value})
    return 0 if value["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
