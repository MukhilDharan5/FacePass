"""Approximate monocular pose from a generic template, never measured depth."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
# Generic engineering template, in arbitrary consistent units. Not an enrolled
# face, anatomical calibration, or a measured depth map. Camera Y points down.
POSE_TEMPLATE = np.array([[0, 0, 0], [0, 70, -10], [-35, -30, -15],
                          [35, -30, -15], [-25, 30, -10], [25, 30, -10]], np.float64)


@dataclass(frozen=True)
class PoseEstimate:
    pitch: float
    yaw: float
    roll: float
    reprojection_error_px: float


def approximate_camera_matrix(shape):
    height, width = shape[:2]
    focal = float(max(width, height))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)


def estimate_pose(points, shape):
    image_points = np.ascontiguousarray(points[list(POSE_LANDMARKS), :2] * [shape[1], shape[0]])
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2:
        return None
    camera = approximate_camera_matrix(shape)
    ok, rotation, translation = cv2.solvePnP(POSE_TEMPLATE, image_points, camera, None,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("non-finite pose model output")
    if translation[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(POSE_TEMPLATE, rotation, translation, camera, None)
    error = float(np.sqrt(np.mean(np.sum((projected[:, 0] - image_points) ** 2, axis=1))))
    face_width = float(np.ptp(points[:468, 0]) * shape[1])
    if error > max(2.0, face_width * 0.03):
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    if not all(math.isfinite(angle) for angle in angles):
        raise ValueError("non-finite pose angles")
    pitch, yaw, roll = (float(angle) for angle in angles)
    if abs(pitch) > 50 or abs(yaw) > 50 or abs(roll) > 35:
        return None
    return PoseEstimate(pitch, yaw, roll, error)
