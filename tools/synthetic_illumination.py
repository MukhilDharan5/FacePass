"""Controlled photometry fixtures, not a physically rendered face or camera input."""

import cv2
import numpy as np

from facepass.illumination import FACE_COORDINATES, face_patch_centers, linear_rgb
if __package__:
    from .synthetic_motion import gaze_mesh, synthetic_frame
else:
    from synthetic_motion import gaze_mesh, synthetic_frame


def encode_rgb(rgb):
    encoded = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    return np.ascontiguousarray(np.rint(np.clip(encoded[:, :, ::-1], 0, 1) * 255).astype(np.uint8))


def illumination_frame(amplitude=0.0, *, face_amplitude=None, nonlinear=True, global_log_gain=0.0,
                       channel_log_gain=(0, 0, 0), face_color=(1, 1, 1), points=None):
    """RGB gain fields with known regional responses; no ground-truth live label."""
    points = gaze_mesh() if points is None else points
    texture = linear_rgb(synthetic_frame())[:, :, 0]
    texture = texture / texture.mean()
    rgb = texture[:, :, None] * [.18, .22, .28]
    rows, columns = np.indices(texture.shape)
    direction = np.where(columns < 320, -1, 1) + .3 * np.where(rows < 240, -1, 1)
    rgb *= np.exp(amplitude * direction[:, :, None])
    hull = np.zeros(texture.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull((points[:468, :2] * [640, 480]).astype(np.int32)), 255)
    rgb[hull != 0] = texture[hull != 0, None] * [.30, .20, .13]
    patch = face_patch_centers(points, (480, 640))
    if patch is None:
        raise ValueError("fixture geometry unavailable")
    centers, _ = patch
    f = amplitude if face_amplitude is None else face_amplitude
    gains = .7 * (FACE_COORDINATES[:, 0] + .3 * FACE_COORDINATES[:, 1])
    if nonlinear:
        gains += [.3, .6, 0, 0, -.3, -.3, .3]
    for center, gain in zip(centers, gains):
        disk = np.zeros_like(hull)
        cv2.circle(disk, tuple(np.rint(center).astype(int)), 12, 255, -1)
        selected = (disk != 0) & (hull != 0)
        rgb[selected] *= np.exp(f * gain * np.array(face_color))
    rgb *= np.exp(global_log_gain + np.array(channel_log_gain))
    return encode_rgb(rgb)
