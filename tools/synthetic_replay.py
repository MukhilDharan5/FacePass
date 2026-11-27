"""Controlled pixel artifacts, not physical replay media or PAD ground truth."""

import cv2
import numpy as np

if __package__:
    from .synthetic_motion import gaze_mesh, synthetic_frame
else:
    from synthetic_motion import gaze_mesh, synthetic_frame


def replay_frame(kind="natural_texture", *, points=None):
    points = gaze_mesh() if points is None else points
    frame = synthetic_frame()
    height, width = frame.shape[:2]
    rows, columns = np.indices((height, width))
    hull = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull((points[:468, :2] * [width, height]).astype(np.int32)), 255)
    face = hull != 0

    if kind in ("display_grid", "matching_bona_fide_pattern"):
        luminance = 28 * np.sin(2 * np.pi * rows / 8)
        red = 130 + luminance + 48 * np.sin(2 * np.pi * columns / 4)
        green = 130 + luminance + 48 * np.sin(2 * np.pi * (columns + 1) / 4)
        blue = 130 + luminance + 48 * np.sin(2 * np.pi * (columns + 2) / 4)
        pattern = np.stack((blue, green, red), axis=2)
        frame[face] = np.clip(pattern[face], 0, 255).astype(np.uint8)
    elif kind == "print_halftone":
        ink = (columns % 3 == 0) | (rows % 3 == 0)
        value = np.where(ink, 40, 200).astype(np.uint8)
        pattern = np.repeat(value[:, :, None], 3, axis=2)
        frame[face] = pattern[face]
    elif kind == "global_sensor_banding":
        gain = (35 * np.sin(2 * np.pi * rows / 8))[:, :, None]
        frame = np.clip(frame.astype(float) + gain, 0, 255).astype(np.uint8)
    elif kind == "background_grid":
        pattern = (110 + 65 * np.sign(np.sin(2 * np.pi * columns / 4)))[:, :, None]
        pattern = np.repeat(pattern, 3, axis=2)
        frame[~face] = np.clip(pattern[~face], 0, 255).astype(np.uint8)
    elif kind == "localized_glare":
        center = tuple(np.rint(points[1, :2] * [width, height]).astype(int))
        cv2.circle(frame, center, 18, (255, 255, 255), -1)
    elif kind == "global_sharpening":
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
        frame = cv2.addWeighted(frame, 2.2, blurred, -1.2, 0)
    elif kind == "low_texture":
        frame[:] = 128
    elif kind != "natural_texture":
        raise ValueError("unknown synthetic replay fixture")
    return np.ascontiguousarray(frame)
