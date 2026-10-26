"""Background-masked affine parallax gating.

Rigid global motion is estimated from static-background features only. Dynamic
object and hand boxes are masked out first, which is what separates a genuine
camera pan from a moving object or a still frame with detector jitter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from chronicle.perception.config import MotionConfig
from chronicle.types import BBox, CameraState, MotionEstimate


class AffineMotionEstimator:
    """Estimate ``(dx, dy, theta)`` between consecutive frames with RANSAC."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self._scale = max(0.25, min(1.0, float(self.config.scale)))
        self._frame_scale = self._scale
        self._prev_gray: np.ndarray | None = None
        self._state = CameraState.UNKNOWN

    def reset(self) -> None:
        self._prev_gray = None
        self._state = CameraState.UNKNOWN

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _mask(self, shape: tuple[int, int], boxes: Iterable[BBox], scale: float | None = None) -> np.ndarray:
        height, width = shape
        mask = np.full(shape, 255, dtype=np.uint8)
        active_scale = self._scale if scale is None else max(0.05, float(scale))
        pad = max(0, int(round(self.config.mask_padding * active_scale)))
        for bbox in boxes:
            x1 = max(0, int(round(bbox.x1 * active_scale)) - pad)
            y1 = max(0, int(round(bbox.y1 * active_scale)) - pad)
            x2 = min(width, int(round(bbox.x2 * active_scale)) + pad)
            y2 = min(height, int(round(bbox.y2 * active_scale)) + pad)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _classify(self, magnitude: float, rotation_degrees: float) -> CameraState:
        cfg = self.config
        if magnitude > cfg.pan_threshold or abs(rotation_degrees) > cfg.pan_rotation_degrees:
            return CameraState.PANNING
        if magnitude < cfg.still_threshold and abs(rotation_degrees) < cfg.still_rotation_degrees:
            return CameraState.STATIONARY
        return CameraState.MOVING

    def estimate(
        self,
        frame: np.ndarray,
        exclude: Iterable[BBox] = (),
        *,
        timestamp: float = 0.0,
    ) -> MotionEstimate:
        """Return motion from the previous frame to ``frame``.

        ``exclude`` should contain the current object/hand boxes. The pipeline
        also passes the previous frame's boxes so a missed detection still
        cannot poison the background feature set.
        """
        gray_full = self._to_gray(frame)
        width = gray_full.shape[1]
        max_width = max(0, int(getattr(self.config, "max_width", 0) or 0))
        scale = self._scale
        if max_width and width * scale > max_width:
            scale = max_width / max(1, width)
        scale = max(0.15, min(1.0, scale))
        self._frame_scale = scale
        if scale < 0.999:
            gray = cv2.resize(
                gray_full,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(gray_full.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray = gray_full

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            self._state = CameraState.UNKNOWN
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        mask = self._mask(gray.shape[:2], exclude, scale)
        previous_points = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=max(8, int(self.config.max_corners)),
            qualityLevel=max(1e-5, float(self.config.quality)),
            minDistance=max(1.0, float(self.config.min_distance)),
            mask=mask,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < self.config.min_features:
            self._prev_gray = gray.copy()
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            self._prev_gray = gray.copy()
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        # Forward-backward consistency removes points that drift onto moving
        # objects or occlusion boundaries.
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self._prev_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is None or back_status is None:
            self._prev_gray = gray.copy()
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        valid = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
        if errors is not None:
            valid &= errors.reshape(-1) < 30.0
        forward_backward = np.linalg.norm(
            previous_points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1
        )
        valid &= forward_backward <= float(self.config.fb_error)

        source_points = previous_points.reshape(-1, 2)[valid]
        target_points = next_points.reshape(-1, 2)[valid]
        if len(source_points) < self.config.min_features:
            self._prev_gray = gray.copy()
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        matrix, inliers = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.config.ransac_threshold),
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        self._prev_gray = gray.copy()
        if matrix is None:
            return MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        inlier_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None
        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else len(source_points)
        inlier_ratio = inlier_count / max(1, len(source_points))
        if inlier_count < self.config.min_features or inlier_ratio < 0.25:
            return MotionEstimate(
                valid=False,
                state=CameraState.UNKNOWN,
                inlier_ratio=inlier_ratio,
                tracked_points=len(source_points),
            )

        dx = float(matrix[0, 2]) / self._frame_scale
        dy = float(matrix[1, 2]) / self._frame_scale
        theta = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        magnitude = math.hypot(dx, dy)
        state = self._classify(magnitude, math.degrees(theta))

        transition: str | None = None
        if state is CameraState.PANNING and self._state is not CameraState.PANNING:
            transition = "pan_start"
        elif self._state is CameraState.PANNING and state is CameraState.STATIONARY:
            transition = "pan_end"
        self._state = state

        return MotionEstimate(
            dx=dx,
            dy=dy,
            theta=theta,
            inlier_ratio=inlier_ratio,
            tracked_points=len(source_points),
            valid=True,
            state=state,
            transition=transition,
        )
