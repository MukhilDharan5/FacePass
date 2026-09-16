"""Offline real-model smoke check on synthetic pixels; never opens a webcam."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="facepass-model-smoke-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        from facepass.landmarks import MediaPipeLandmarker
        from facepass.session import LivenessSession
        import time

        session = LivenessSession(lambda: MediaPipeLandmarker(options.model))
        if session.start().terminal:
            raise RuntimeError("model initialization failed")
        try:
            result = session.process_frame(np.full((480, 640, 3), 128, np.uint8), time.monotonic())
            if result.face_count != 0 or result.reason != "insufficient_quality":
                raise RuntimeError("unexpected blank-frame inference result")
        finally:
            session.finish()
        if session.current_result().reason == "model_cleanup_failed":
            raise RuntimeError("model close failed")
        print(json.dumps({"model_loaded": True, "synthetic_frame_processed": True,
                          "face_count": result.face_count, "decision": result.decision.value,
                          "camera_opened": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
