"""Finite capture worker; private JSON pipe is only for the standalone harness."""

import base64
import json
import os
import signal
import sys
import time
import resource


def emit(kind, **fields):
    sys.stdout.write(json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n")
    sys.stdout.flush()


def run(options):
    # Prevent a worker crash from saving camera memory in a core dump.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Keep numerical runtimes from starting large CPU thread pools.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.umask(0o077)
    cancelled = False

    def stop(signum, stack):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    camera = session = None
    try:
        import cv2
        from .camera import V4L2Camera
        from .diagnostics import DiagnosticFrames
        from .landmarks import MediaPipeLandmarker
        from .quality import face_bounds
        from .session import LivenessSession

        cv2.setNumThreads(1)
        writer = DiagnosticFrames(options.save_diagnostic_frames) if options.save_diagnostic_frames else None
        session = LivenessSession(lambda: MediaPipeLandmarker(options.model), max_duration=options.duration,
                                  adaptive=not options.fixed_window,
                                  extension_duration=options.extension_duration,
                                  quality_profile=options.quality_profile)
        if session.start().terminal:
            emit("final", result=session.current_result().to_dict())
            return
        if cancelled:
            emit("final", result=session.cancel().to_dict())
            return
        try:
            camera = V4L2Camera(options.device, options.fps)
        except Exception:
            emit("final", result=session.fail("camera_open_failed").to_dict())
            return
        emit("started")
        while not cancelled:
            tick = time.monotonic()
            if session.current_result().terminal:
                break
            try:
                frame = camera.read()
            except Exception:
                session.fail("camera_read_failed")
                break
            timestamp = time.monotonic()
            result = session.process_frame(frame, timestamp)
            if writer:
                writer.maybe_save(frame, timestamp)
            payload = {"result": result.to_dict()}
            if options.preview:
                annotated = frame.copy()
                observation = session.observation
                if observation:
                    height, width = frame.shape[:2]
                    for points in observation.faces:
                        for x, y in points[:, :2]:
                            cv2.circle(annotated, (int(x * width), int(y * height)), 1, (0, 220, 0), -1)
                        x0, y0, x1, y1 = face_bounds(points, frame.shape)
                        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 220, 0), 1)
                cv2.putText(annotated, "ADAPTIVE OBSERVATION/TEMPORAL - LIVE DISABLED", (12, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
                if annotated.shape[1] > 640:
                    annotated = cv2.resize(annotated, (640, int(annotated.shape[0] * 640 / annotated.shape[1])))
                ok, encoded = cv2.imencode(".png", annotated)
                if not ok:
                    raise RuntimeError("preview encoding failed")
                payload["preview"] = base64.b64encode(encoded).decode("ascii")
            # Preview work also consumes time; don't emit stale quality/tracking
            # from a result captured before encoding or timeout/expiry checks.
            result = session.current_result()
            payload["result"] = result.to_dict()
            emit("update", **payload)
            del frame
            if result.terminal:
                break
            # Sleep is interruptible in short increments; no capture at idle.
            deadline = tick + 1 / options.fps
            while not cancelled and time.monotonic() < deadline:
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        result = session.cancel() if cancelled else session.finish()
        camera.close()
        camera = None
        emit("final", result=result.to_dict())
    except Exception:
        from .results import failure_result
        result = session.fail("worker_error").to_dict() if session else failure_result("worker_error")
        emit("final", result=result)
    finally:
        if camera is not None:
            camera.close()
        if session is not None:
            session.finish()
