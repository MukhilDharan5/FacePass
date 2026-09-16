"""Experimental quality gates; thresholds need diverse-camera evaluation."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .results import QualityResult


@dataclass(frozen=True)
class QualityPolicy:
    min_face_width_px: int = 120
    min_face_area_fraction: float = 0.025
    min_blur_variance: float = 45.0
    min_luminance: float = 40.0
    max_luminance: float = 230.0
    max_clipped_fraction: float = 0.4
    max_roll_degrees: float = 30.0


def validate_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("expected uint8 BGR frame")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected H x W x 3 frame")
    height, width = frame.shape[:2]
    if not 64 <= height <= 2160 or not 64 <= width <= 3840:
        raise ValueError("invalid frame dimensions")
    if not frame.flags.c_contiguous:
        raise ValueError("expected contiguous frame")


def face_bounds(points, shape):
    height, width = shape[:2]
    xy = points[:468, :2]
    x0, y0 = np.floor(xy.min(axis=0) * [width, height]).astype(int)
    x1, y1 = np.ceil(xy.max(axis=0) * [width, height]).astype(int)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def assess_quality(frame, points, policy=QualityPolicy()):
    x0, y0, x1, y1 = face_bounds(points, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return QualityResult(reasons=("invalid_face_bounds",))
    height, width = frame.shape[:2]
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    # Use the mesh hull, rather than counting background in the face box.
    local = points[:468, :2] * [width, height] - [x0, y0]
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(local.astype(np.int32)), 255)
    pixels = roi[mask != 0]
    if pixels.size < 64:
        return QualityResult(reasons=("invalid_face_region",))
    mean = float(pixels.mean())
    dark = float(np.mean(pixels <= 10))
    bright = float(np.mean(pixels >= 245))
    # Ignore the hull edge to avoid artificially increasing sharpness.
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) != 0
    if not inner.any():
        return QualityResult(reasons=("invalid_face_region",))
    blur = float(cv2.Laplacian(roi, cv2.CV_64F)[inner].var())
    valid = float(np.mean(np.all((points[:, :2] >= 0) & (points[:, :2] <= 1), axis=1)))
    eyes = points[[33, 263], :2] * [width, height]
    eye_vector = eyes[1] - eyes[0]
    norm = float(np.dot(eye_vector, eye_vector))
    nose = points[1, :2] * [width, height]
    nose_position = float(np.dot(nose - eyes[0], eye_vector) / norm) if norm > 1 else -1
    roll = abs(math.degrees(math.atan2(eye_vector[1], eye_vector[0])))
    roll = min(roll, 180 - roll)
    # Nose/eye asymmetry is only a rough orientation proxy, not measured 3D pose.
    orientation = norm > 25 and 0.2 <= nose_position <= 0.8 and roll <= policy.max_roll_degrees
    area = float((x1 - x0) * (y1 - y0) / (width * height))
    reasons = []
    if x1 - x0 < policy.min_face_width_px or area < policy.min_face_area_fraction:
        reasons.append("face_too_small")
    if valid < 0.98 or min(x0, y0) == 0 or x1 == width or y1 == height:
        reasons.append("face_clipped")
    if blur < policy.min_blur_variance:
        reasons.append("blur")
    if mean < policy.min_luminance or dark > policy.max_clipped_fraction:
        reasons.append("underexposure")
    if mean > policy.max_luminance or bright > policy.max_clipped_fraction:
        reasons.append("overexposure")
    if not orientation:
        reasons.append("face_orientation")
    return QualityResult(
        acceptable=not reasons, face_size_px=float(x1 - x0), face_area_fraction=area,
        blur_variance=blur, mean_luminance=mean, dark_fraction=dark, bright_fraction=bright,
        landmark_valid_fraction=valid, orientation_acceptable=orientation,
        # FaceLandmarker does not expose a calibrated per-mesh confidence score.
        landmark_confidence=None, reasons=tuple(reasons),
    )
