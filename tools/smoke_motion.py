"""Offline synthetic milestone-3 demo; not measured physical PAD performance."""

import argparse
import json

import cv2
import numpy as np

from facepass.motion import MotionAnalyzer
from facepass.results import Decision
from facepass.session import LivenessSession
from facepass.tracking import TrackedSample
from synthetic_motion import SyntheticBackend, planar_mesh, projected_mesh, synthetic_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Show full scalar signal and motion diagnostics")
    options = parser.parse_args()
    base_frame = synthetic_frame()
    base_points = projected_mesh()
    for scenario in ("still", "subtle_3d_rotation", "moved_plane", "shared_camera_motion", "no_background_texture"):
        analyzer = MotionAnalyzer()
        history = []
        for index in range(10):
            frame = base_frame
            if scenario in ("subtle_3d_rotation", "no_background_texture"):
                points = projected_mesh(yaw=index * 0.05)
                if scenario == "no_background_texture":
                    frame = np.full_like(base_frame, 128)
            elif scenario in ("moved_plane", "shared_camera_motion"):
                matrix = np.array([[1, 0, index * 0.6], [0, 1, index * 0.3], [0, 0, 1]], np.float64)
                points = planar_mesh(base_points, matrix)
                if scenario == "shared_camera_motion":
                    frame = cv2.warpPerspective(base_frame, matrix, (640, 480))
            else:
                points = base_points
            history.append(TrackedSample(1, 100 + index * 0.1, points, (480, 640)))
            stats, head, geometry = analyzer.update(frame, tuple(history), stable=index >= 2)
        if scenario == "subtle_3d_rotation":
            if head.confidence <= 0.5 or geometry.confidence <= 0.5:
                raise RuntimeError("expected weak synthetic motion evidence")
        elif head.confidence > 0.5 or geometry.confidence > 0.5:
            raise RuntimeError("confounder produced positive motion evidence")
        if scenario == "shared_camera_motion" and not stats.common_motion_suppressed:
            raise RuntimeError("shared motion was not suppressed")
        if options.json:
            from dataclasses import asdict
            print(json.dumps({"synthetic_scenario": scenario, "head_motion": asdict(head),
                              "facial_geometry": asdict(geometry), "motion": asdict(stats)}, allow_nan=False))
        else:
            print(f"{scenario}: head={head.confidence:.3f} reliability={head.reliability:.3f}; "
                  f"geometry={geometry.confidence:.3f}; background={stats.background_available}; "
                  f"shared_motion={stats.common_motion_suppressed}")
        analyzer.reset()

    current_time = 100.0
    backend = SyntheticBackend()
    session = LivenessSession(lambda: backend, clock=lambda: current_time)
    session.start()
    try:
        for index in range(10):
            current_time += 0.1
            backend.points = projected_mesh(yaw=index * 0.05)
            result = session.process_frame(base_frame, current_time)
            if result.decision != Decision.UNCERTAIN or result.confidence != 0:
                raise RuntimeError("motion demo produced unexpected authentication result")
        result = session.finish()
        if result.decision == Decision.LIVE or result.confidence != 0:
            raise RuntimeError("finished demo produced authentication success")
        if options.json:
            print(json.dumps({"synthetic_session_final": result.to_dict()}, allow_nan=False))
        else:
            print(f"session final: {result.decision.value}, confidence={result.confidence:.1f}, "
                  f"motion_historical={result.motion.historical}, retained_history={len(session.history)}")
    finally:
        session.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
