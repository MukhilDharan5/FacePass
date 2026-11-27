"""Exposure-aware capture usability; these heuristics are not PAD evidence."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .results import QualityResult


@dataclass(frozen=True)
class QualityPolicy:
    profile: str = "tolerant"
    min_face_width_px: int = 96
    min_face_area_fraction: float = 0.015
    min_blur_variance: float = 8.0
    min_luminance: float = 12.0
    max_luminance: float = 245.0
    max_clipped_fraction: float = 0.65
    max_roll_degrees: float = 30.0
    min_contrast_span: float = 8.0
    min_raw_contrast_span: float = 3.0
    normalize_for_luminance: bool = True
    enhance_low_light: bool = True


def quality_policy_for(profile):
    if profile == "tolerant":
        return QualityPolicy()
    if profile == "strict":
        # Retain the pre-0.13 capture gates for comparisons with older trials.
        return QualityPolicy(profile="strict", min_face_width_px=120,
                             min_face_area_fraction=0.025, min_blur_variance=45.0,
                             min_luminance=40.0, max_luminance=230.0,
                             max_clipped_fraction=0.4, min_contrast_span=0.0,
                             min_raw_contrast_span=0.0, normalize_for_luminance=False,
                             enhance_low_light=False)
    raise ValueError("quality profile must be tolerant or strict")


def prepare_landmark_frame(frame, policy=QualityPolicy()):
    """One bounded, full-frame gamma pass for dim landmark input only.

    Never sharpen, invent texture, crop away other faces, or send the enhanced
    image to quality, appearance, illumination or replay measurements. Pixels
    stay in memory; the caller retains its original frame unchanged.
    """
    if not policy.enhance_low_light:
        return frame, False
    height, width = frame.shape[:2]
    sample = frame[::max(1, math.ceil(height / 120)), ::max(1, math.ceil(width / 160))]
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    low, high = np.percentile(gray, (5, 95))
    if not 8 <= median < 65 or high - low < 3:
        return frame, False
    gamma = max(0.55, min(1.0, math.log(85 / 255) / math.log(median / 255)))
    levels = np.arange(256, dtype=np.float64)
    # A threefold gain cap prevents extreme near-black/noise amplification.
    lut = np.minimum(np.rint(255 * (levels / 255) ** gamma), levels * 3).astype(np.uint8)
    return cv2.LUT(frame, lut), True


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
        return QualityResult(profile=policy.profile, reasons=("invalid_face_bounds",))
    height, width = frame.shape[:2]
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    # Use the mesh hull, rather than counting background in the face box.
    local = points[:468, :2] * [width, height] - [x0, y0]
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(local.astype(np.int32)), 255)
    pixels = roi[mask != 0]
    if pixels.size < 64:
        return QualityResult(profile=policy.profile, reasons=("invalid_face_region",))
    mean = float(pixels.mean())
    dark = float(np.mean(pixels <= 10))
    bright = float(np.mean(pixels >= 245))
    # Ignore the hull edge to avoid artificially increasing sharpness.
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) != 0
    if not inner.any():
        return QualityResult(profile=policy.profile, reasons=("invalid_face_region",))
    blur = float(cv2.Laplacian(roi, cv2.CV_64F)[inner].var())
    low, high = np.percentile(pixels, (5, 95))
    contrast = float(high - low)
    # A dark face has a lower absolute Laplacian variance even with identical
    # spatial detail. Evaluate a bounded reference-exposure equivalent while
    # publishing the raw measurement too. This does not estimate skin tone.
    gain = min(4.0, max(1.0, 128 / max(mean, 16))) if policy.normalize_for_luminance else 1.0
    evaluated_blur = blur * gain ** 2
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
    if evaluated_blur < policy.min_blur_variance:
        reasons.append("blur")
    if contrast < policy.min_raw_contrast_span or contrast * gain < policy.min_contrast_span:
        reasons.append("low_contrast")
    if mean < policy.min_luminance or dark > policy.max_clipped_fraction:
        reasons.append("underexposure")
    if mean > policy.max_luminance or bright > policy.max_clipped_fraction:
        reasons.append("overexposure")
    if not orientation:
        reasons.append("face_orientation")
    warnings = []
    if policy.profile == "tolerant":
        if blur < 45 and "blur" not in reasons:
            warnings.append("moderate_blur")
        if mean < 40 and "underexposure" not in reasons:
            warnings.append("low_luminance")
        if contrast < 20 and "low_contrast" not in reasons:
            warnings.append("limited_contrast")
        if (x1 - x0 < 120 or area < 0.025) and "face_too_small" not in reasons:
            warnings.append("small_face")
    return QualityResult(
        acceptable=not reasons, profile=policy.profile, warnings=tuple(warnings),
        face_size_px=float(x1 - x0), face_area_fraction=area,
        blur_variance=blur, evaluated_blur_variance=evaluated_blur, contrast_span=contrast,
        mean_luminance=mean, dark_fraction=dark, bright_fraction=bright,
        landmark_valid_fraction=valid, orientation_acceptable=orientation,
        # FaceLandmarker does not expose a calibrated per-mesh confidence score.
        landmark_confidence=None, reasons=tuple(reasons),
    )
