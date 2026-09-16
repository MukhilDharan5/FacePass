"""Short-window head motion, planar fit and sparse environment-motion comparison."""

from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .head_pose import estimate_pose
from .motion_evidence import geometry_evidence, head_motion_evidence
from .quality import validate_frame
from .results import MotionDiagnostics, SignalResult
from .tracking import ANCHORS


# A held-out spatial split prevents four fitted landmarks from defining their own
# geometry evidence. Avoid eyelids/iris/mouth motion in this comparison.
PLANE_TRAIN = (10, 152, 234, 454, 33, 263, 168, 127, 356, 172, 397, 132)
PLANE_TEST = (1, 6, 98, 327, 93, 323, 361, 4, 102, 331)


def _project(matrix, points):
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid motion transform")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError("singular motion transform")
    projected = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.isfinite(projected).all():
        raise ValueError("non-finite projected motion")
    return projected


def planar_comparison(previous, current, shape):
    factor = [shape[1], shape[0]]
    source, target = previous[list(PLANE_TRAIN), :2] * factor, current[list(PLANE_TRAIN), :2] * factor
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        return None
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 0.5, maxIters=300, confidence=0.99)
    if matrix is None or mask is None or float(mask.mean()) < 0.65:
        return None
    test_source, test_target = previous[list(PLANE_TEST), :2] * factor, current[list(PLANE_TEST), :2] * factor
    residual = float(np.median(np.linalg.norm(_project(matrix, test_source) - test_target, axis=1)))
    return residual


@dataclass(frozen=True)
class BackgroundEstimate:
    transform: np.ndarray | None = None  # Private, in full input-pixel coordinates.
    point_count: int = 0
    inlier_fraction: float = 0.0


def environment_image(frame, points):
    height, width = frame.shape[:2]
    scale = min(1.0, 320 / width, 240 / height)
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    mask = np.full(small.shape, 255, np.uint8)
    hull = cv2.convexHull((points[:468, :2] * [small.shape[1], small.shape[0]]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 0)
    # Exclude hair, face-edge flow windows and nearby facial shadows as well.
    mask = cv2.erode(mask, np.ones((25, 25), np.uint8))
    small[mask == 0] = 0  # Retained environment buffer contains no detected face pixels.
    return small, mask


def background_motion(previous_image, current_image, previous_mask, current_mask, original_shape):
    # Do not detect corners manufactured by the edge of the black face mask.
    # The feature detector's support must lie inside real environment pixels.
    mask = cv2.erode(cv2.bitwise_and(previous_mask, current_mask), np.ones((17, 17), np.uint8))
    points = cv2.goodFeaturesToTrack(previous_image, 80, 0.02, 8, mask=mask, blockSize=5)
    if points is None or len(points) < 12:
        return BackgroundEstimate()
    options = dict(winSize=(15, 15), maxLevel=2,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, points, None, **options)
    if forward is None or status is None:
        return BackgroundEstimate()
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current_image, previous_image, forward, None, **options)
    if backward is None or back_status is None:
        return BackgroundEstimate()
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("non-finite environment flow")
    source, target = points[:, 0], forward[:, 0]
    good = (status[:, 0] != 0) & (back_status[:, 0] != 0)
    good &= np.linalg.norm(backward[:, 0] - source, axis=1) < 0.5
    height, width = current_image.shape
    ix, iy = np.rint(target[:, 0]).astype(int), np.rint(target[:, 1]).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    good &= inside
    valid_mask = np.zeros(len(points), dtype=bool)
    valid_mask[inside] = current_mask[iy[inside], ix[inside]] != 0
    good &= valid_mask
    source, target = source[good], target[good]
    if len(source) < 12:
        return BackgroundEstimate(point_count=len(source))
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 0.6, maxIters=300, confidence=0.99)
    if matrix is None or inliers is None:
        return BackgroundEstimate(point_count=len(source))
    # Prefer a simpler scene transform when it explains the same flow. A noisy
    # eight-parameter homography can accumulate artificial perspective changes
    # under tiny camera translation. Keep projective motion only when it clearly
    # improves fit; this does not change the liveness scoring/security caps.
    affine, affine_inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                       ransacReprojThreshold=0.6, maxIters=300,
                                                       confidence=0.99, refineIters=10)
    if affine is not None and affine_inliers is not None:
        simpler = np.vstack((affine, [0, 0, 1]))
        simple_error = float(np.median(np.linalg.norm(_project(simpler, source) - target, axis=1)))
        projective_error = float(np.median(np.linalg.norm(_project(matrix, source) - target, axis=1)))
        if not (simple_error - projective_error > 0.05 and projective_error < 0.6 * simple_error):
            matrix, inliers = simpler, affine_inliers
    selected = inliers[:, 0] != 0
    ratio = float(selected.mean())
    spread = source[selected]
    if len(spread) < 12 or ratio < 0.7:
        return BackgroundEstimate(point_count=len(source), inlier_fraction=ratio)
    area = float(cv2.contourArea(cv2.convexHull(spread))) / (width * height)
    quadrants = {(int(x >= width / 2), int(y >= height / 2)) for x, y in spread}
    if area < 0.15 or len(quadrants) < 3:
        return BackgroundEstimate(point_count=len(source), inlier_fraction=ratio)
    factor = np.diag([width / original_shape[1], height / original_shape[0], 1.0])
    transform = np.linalg.inv(factor) @ matrix @ factor
    _project(transform, np.array([[0.0, 0.0], [original_shape[1], original_shape[0]]]))
    return BackgroundEstimate(transform, len(source), ratio)


def temporal_rotation(samples):
    valid = [(timestamp, pose) for timestamp, pose in samples if pose is not None]
    if len(valid) < 6:
        return None, 0.0
    times = np.array([item[0] for item in valid])
    angles = np.array([[pose.pitch, pose.yaw, pose.roll] for _, pose in valid])
    span = float(np.linalg.norm(np.percentile(angles, 90, axis=0) - np.percentile(angles, 10, axis=0)))
    centered = angles - angles.mean(axis=0)
    total = float(np.sum(centered ** 2))
    if total < 1e-8:
        return span, 0.0
    times -= times.mean()
    trend = np.outer(times, times @ centered / float(times @ times))
    coherence = float(np.clip(1 - np.sum((centered - trend) ** 2) / total, 0, 1))
    return span, coherence


class MotionAnalyzer:
    def __init__(self):
        self._samples = deque(maxlen=64)  # Scalar pose history only, at most two seconds.
        self._flows = deque(maxlen=64)  # Small private background transforms, no images.
        self._image = self._mask = self._track_id = None

    def reset(self):
        self._samples.clear()
        self._flows.clear()
        self._image = self._mask = self._track_id = None

    def update(self, frame, history, *, stable):
        try:
            return self._update(frame, history, stable=stable)
        except Exception:
            self.reset()
            raise

    def _update(self, frame, history, *, stable):
        validate_frame(frame)
        if not history:
            self.reset()
            return MotionDiagnostics(), SignalResult(), SignalResult()
        current = history[-1]
        if current.track_id != self._track_id:
            self.reset()
            self._track_id = current.track_id
        if self._samples and current.timestamp <= self._samples[-1][0]:
            raise ValueError("non-monotonic motion timestamp")
        pose = estimate_pose(current.landmarks, frame.shape)
        if pose is None:
            # Do not build a rotation trend across missing/unreliable estimates.
            self._samples.clear()
        self._samples.append((current.timestamp, pose))
        while self._samples and current.timestamp - self._samples[0][0] > 2.0:
            self._samples.popleft()
        image, mask = environment_image(frame, current.landmarks)
        stats = MotionDiagnostics(status="observing", window_frame_count=len(self._samples),
                                  window_duration_ms=max(0, int((current.timestamp - self._samples[0][0]) * 1000)))
        if pose is not None:
            stats = replace(stats, pose_available=True, pitch_degrees=pose.pitch, yaw_degrees=pose.yaw,
                            roll_degrees=pose.roll, pose_reprojection_error_px=pose.reprojection_error_px)
        if len(history) >= 2 and self._image is not None:
            previous = history[-2]
            if previous.track_id != current.track_id or current.timestamp - previous.timestamp > 0.35:
                raise ValueError("motion history crosses discontinuity")
            # Accumulate a short baseline: subpixel natural motion should not
            # disappear beneath single-frame landmark noise/perspective changes.
            recent = [sample for sample in history if sample.timestamp >= self._samples[0][0]]
            reference = recent[0]
            factor = [frame.shape[1], frame.shape[0]]
            source, target = reference.landmarks[list(ANCHORS), :2] * factor, current.landmarks[list(ANCHORS), :2] * factor
            landmark_motion = float(np.median(np.linalg.norm(target - source, axis=1)))
            background = background_motion(self._image, image, self._mask, mask, frame.shape)
            self._flows.append((current.timestamp, background))
            while self._flows and current.timestamp - self._flows[0][0] > 2.0:
                self._flows.popleft()
            residual = planar_comparison(reference.landmarks, current.landmarks, frame.shape)
            stats = replace(stats, landmark_motion_px=landmark_motion,
                            background_point_count=background.point_count,
                            background_inlier_fraction=background.inlier_fraction,
                            planar_available=residual is not None, planar_holdout_error_px=residual,
                            nonplanar_fraction=None if residual is None else float(np.clip(residual / max(landmark_motion, 0.15), 0, 1)))
            steps = [flow for timestamp, flow in self._flows if timestamp > reference.timestamp]
            if steps and len(steps) == len(recent) - 1 and all(step.transform is not None for step in steps):
                transform = np.eye(3)
                for step in steps:
                    transform = step.transform @ transform
                predicted = _project(transform, source)
                common = float(np.median(np.linalg.norm(predicted - source, axis=1)))
                independent = float(np.median(np.linalg.norm(target - predicted, axis=1)))
                suppressed = common > 0.1 and independent < max(0.15, 0.25 * landmark_motion)
                stats = replace(stats, background_available=True, common_motion_px=common,
                                independent_motion_px=independent, common_motion_suppressed=suppressed,
                                background_inlier_fraction=min(step.inlier_fraction for step in steps))
        self._image, self._mask = image, mask
        span, coherence = temporal_rotation(self._samples)
        stats = replace(stats, rotation_span_degrees=span, temporal_coherence=coherence)
        if not stable:
            return stats, SignalResult(diagnostic_information="tracking_not_stable"), SignalResult(diagnostic_information="tracking_not_stable")
        return stats, head_motion_evidence(stats), geometry_evidence(stats)
