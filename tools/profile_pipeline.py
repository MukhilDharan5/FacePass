"""Reproducible CPU/timing benchmarks without a webcam or retained biometric data."""

import argparse
import cProfile
import json
import os
import platform
from pathlib import Path
import resource
import sys
import time

# Allow invocation from another working directory as well as the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facepass import __version__
from facepass.performance import distribution


def measure(operation, frames):
    samples = []
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for index in range(frames):
        before = time.perf_counter()
        operation(index)
        samples.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return {"frames": frames, "wall_ms": round(elapsed * 1000, 3), "cpu_ms": round(cpu * 1000, 3),
            "average_cpu_cores": round(cpu / elapsed, 3) if elapsed else None,
            "throughput_fps": round(frames / elapsed, 3) if elapsed else None,
            "frame_ms": distribution(samples), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(frames, model=None):
    import cv2
    import numpy as np
    from facepass.landmarks import FaceObservation, MediaPipeLandmarker
    from facepass.session import LivenessSession
    from tools.synthetic_motion import gaze_mesh, synthetic_frame

    cv2.setNumThreads(1)
    reports = []
    if model:
        before = time.perf_counter()
        backend = MediaPipeLandmarker(model)
        initialization = (time.perf_counter() - before) * 1000
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            backend.detect(blank, time.monotonic())  # warm up native inference
            report = measure(lambda index: backend.detect(blank, time.monotonic()), frames)
            reports.append({"workload": "real_model_blank_frame", "initialization_ms": round(initialization, 3), **report})
        finally:
            backend.close()

    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

    class Backend:
        def detect(self, frame, timestamp):
            return FaceObservation((gaze_mesh(horizontal=.5 + .06 * np.sin((timestamp - 100) * 3)),))

        def close(self):
            pass

    frame, clock = synthetic_frame(), Clock()
    session = None

    def process(index):
        nonlocal session
        # Each bounded attempt uses simulated 12-fps capture timestamps. This
        # measures detector math with synthetic landmarks, not model/camera cost.
        if session is None or session.current_result().terminal:
            session = LivenessSession(Backend, clock=clock, max_duration=30, adaptive=False)
            session.start()
        clock.now += 1 / 12
        result = session.process_frame(frame, clock.now)
        if result.reason in ("frame_or_model_error", "temporal_evidence_error"):
            raise RuntimeError("synthetic benchmark pipeline failed")

    try:
        for index in range(8):
            process(index)
        reports.append({"workload": "synthetic_full_detector_pipeline", **measure(process, frames)})
    finally:
        if session:
            session.finish()
    return {"schema_version": 1, "engine_version": __version__, "authentication_enabled": False,
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            "numpy": np.__version__, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads()},
            "physical_capture_measured": False, "benchmarks": reports,
            "interpretation": "Synthetic landmarks and blank model input measure engineering cost only. RSS is the process high-water mark; throughput excludes webcam pacing. Use facepass --profile on physical captures for end-to-end timing."}


def main(argv=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--model", type=Path, help="Include verified real-model blank-frame inference")
    parser.add_argument("--cprofile", type=Path, help="Optional developer function profile (no images)")
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 1000:
        parser.error("frames must be 1..1000")
    profiler = cProfile.Profile() if args.cprofile else None
    if profiler:
        profiler.enable()
    result = run(args.frames, args.model)
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.cprofile)
    print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
