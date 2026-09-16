"""Offline synthetic continuity demo; no webcam, recognition or PAD evaluation."""

import json

import numpy as np

from facepass.landmarks import FaceObservation
from facepass.results import Decision, TrackingState
from facepass.session import LivenessSession


class SyntheticBackend:
    def __init__(self):
        angle = np.linspace(0, 2 * np.pi, 478)
        self.points = np.column_stack((0.5 + 0.2 * np.cos(angle), 0.5 + 0.3 * np.sin(angle), angle * 0))
        self.points[33], self.points[263], self.points[1] = [0.4, 0.4, 0], [0.6, 0.4, 0], [0.5, 0.5, 0]
        self.visible = True

    def detect(self, frame, timestamp):
        return FaceObservation((self.points,) if self.visible else ())

    def close(self):
        pass


def main():
    backend = SyntheticBackend()
    current_time = 100.0
    session = LivenessSession(lambda: backend, clock=lambda: current_time)
    y, x = np.mgrid[:480, :640]
    gray = (125 + 35 * np.sin(x / 19) + 28 * np.cos(y / 17) + 9 * np.sin(x * 1.5)).astype(np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    session.start()

    def step(scenario):
        nonlocal current_time
        current_time += 0.1
        result = session.process_frame(frame, current_time)
        if result.decision == Decision.LIVE or result.confidence != 0:
            raise RuntimeError("unexpected liveness success")
        print(json.dumps({"scenario": scenario, "result": result.to_dict()}, allow_nan=False))
        return result

    try:
        step("still_face")
        step("still_face")
        stable = step("still_face")
        if stable.tracking.state != TrackingState.STABLE:
            raise RuntimeError("track failed to stabilize")
        backend.visible = False
        lost = step("face_leaves")
        if session.history or lost.tracking.state != TrackingState.LOST:
            raise RuntimeError("face loss retained history")
        backend.visible = True
        returned = step("face_returns")
        if returned.tracking.track_id == stable.tracking.track_id or len(session.history) != 1:
            raise RuntimeError("reentry reused old track")
        frame = 255 - frame
        rejected = step("appearance_substitution")
        if not rejected.terminal or session.history or rejected.reason != "appearance_discontinuity":
            raise RuntimeError("substitution failed to terminate/clear")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
