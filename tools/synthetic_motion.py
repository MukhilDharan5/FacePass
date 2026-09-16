"""Synthetic projection fixtures only. Not camera input or a runtime backend."""

import cv2
import numpy as np

from facepass.head_pose import POSE_LANDMARKS, POSE_TEMPLATE, approximate_camera_matrix
from facepass.landmarks import FaceObservation


def projected_mesh(*, yaw=0.0, pitch=0.0, roll=0.0):
    angle = np.linspace(0, 2 * np.pi, 478)
    model = np.column_stack((65 * np.cos(angle), 75 * np.sin(angle) + 5, angle * 0 - 22))
    for index, coordinates in zip(POSE_LANDMARKS, POSE_TEMPLATE):
        model[index] = coordinates
    anchors = {
        10: (0, -65, -22), 234: (-65, -3, -25), 454: (65, -3, -25),
        168: (0, -18, -3), 127: (-56, -35, -24), 356: (56, -35, -24),
        172: (-45, 60, -20), 397: (45, 60, -20), 132: (-60, 35, -22),
        93: (-53, 12, -21), 323: (53, 12, -21), 361: (58, 40, -21),
        4: (0, 8, -2), 102: (-16, 7, -8), 331: (16, 7, -8),
        98: (-14, 12, -5), 327: (14, 12, -5), 6: (0, -12, -4),
    }
    for index, coordinates in anchors.items():
        model[index] = coordinates
    rotation = np.deg2rad(np.array([pitch, yaw, roll], dtype=np.float64))
    pixels, _ = cv2.projectPoints(model, rotation, np.array([0, 0, 420.0]),
                                  approximate_camera_matrix((480, 640)), None)
    # No synthetic depth is supplied to the engine: motion works only from XY.
    return np.column_stack((pixels[:, 0] / [640, 480], np.zeros(478)))


def planar_mesh(points, matrix):
    pixels = points[:, :2] * [640, 480]
    projected = cv2.perspectiveTransform(pixels.astype(np.float64)[None], matrix)[0]
    return np.column_stack((projected / [640, 480], np.zeros(478)))


def synthetic_frame():
    rng = np.random.default_rng(1729)
    gray = rng.integers(40, 210, size=(480, 640), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    return np.repeat(gray[:, :, None], 3, axis=2)


class SyntheticBackend:
    def __init__(self):
        self.points = projected_mesh()
        self.visible = True
        self.closed = False

    def detect(self, frame, timestamp):
        return FaceObservation((self.points,) if self.visible else ())

    def close(self):
        self.closed = True
