from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.config import MotionConfig
from chronicle.perception.motion import AffineMotionEstimator
from chronicle.types import BBox, CameraState


def texture(width: int = 320, height: int = 240, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (3, 3), 0)


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def test_first_frame_is_invalid_and_still_frame_is_stationary() -> None:
    estimator = AffineMotionEstimator(MotionConfig(scale=1.0))
    frame = texture()
    assert estimator.estimate(frame).valid is False
    motion = estimator.estimate(frame)
    assert motion.valid is True
    assert motion.state is CameraState.STATIONARY
    assert motion.magnitude < 1.0


def test_translation_is_recovered() -> None:
    estimator = AffineMotionEstimator(MotionConfig(scale=1.0))
    frame = texture()
    estimator.estimate(frame)
    motion = estimator.estimate(shifted(frame, 5.0, -4.0))
    assert motion.valid is True
    assert abs(motion.dx - 5.0) < 1.0
    assert abs(motion.dy + 4.0) < 1.0
    assert motion.state is CameraState.MOVING


def test_pan_transition_emits_settle_keyframe() -> None:
    estimator = AffineMotionEstimator(
        MotionConfig(scale=1.0, pan_threshold=8.0, still_threshold=2.5)
    )
    frame = texture()
    panned = shifted(frame, 14.0, 0.0)
    estimator.estimate(frame)
    started = estimator.estimate(panned)
    assert started.state is CameraState.PANNING
    assert started.transition == "pan_start"
    settled = estimator.estimate(panned)
    assert settled.state is CameraState.STATIONARY
    assert settled.transition == "pan_end"
    assert settled.keyframe is True


def test_rotation_is_treated_as_pan_even_with_tiny_translation() -> None:
    estimator = AffineMotionEstimator(
        MotionConfig(scale=1.0, pan_threshold=100.0, pan_rotation_degrees=1.8)
    )
    frame = texture()
    height, width = frame.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 4.0, 1.0)
    rotated = cv2.warpAffine(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    estimator.estimate(frame)
    motion = estimator.estimate(rotated)
    assert motion.valid is True
    # theta follows the image-coordinate convention (y down), so the sign is
    # opposite to OpenCV's display-angle argument; only magnitude is compared.
    assert abs(abs(motion.rotation_degrees) - 4.0) < 0.8
    assert motion.state is CameraState.PANNING


def test_masked_dynamic_object_does_not_make_background_move() -> None:
    estimator = AffineMotionEstimator(MotionConfig(scale=1.0))
    background = texture()
    first = background.copy()
    cv2.rectangle(first, (20, 20), (100, 100), (255, 255, 255), -1)
    second = background.copy()
    cv2.rectangle(second, (200, 120), (300, 220), (255, 255, 255), -1)
    estimator.estimate(first)
    motion = estimator.estimate(
        second,
        exclude=[BBox(20, 20, 100, 100), BBox(200, 120, 300, 220)],
    )
    assert motion.valid is True
    assert motion.state is CameraState.STATIONARY
