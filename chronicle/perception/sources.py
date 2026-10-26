"""Video / webcam / frame-iterable source normalisation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Yield ``(frame, presentation_time, frame_index)`` with monotonic times."""

    def __init__(
        self,
        source: str | Path | int | Iterable[np.ndarray],
        *,
        fps_override: float | None = None,
    ) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._iterator: Iterator[np.ndarray] | None = None
        self._path: Path | None = None
        self._camera: int | None = None
        self._fps_override = fps_override
        self._default_fps = 30.0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.fps = float(fps_override or self._default_fps)
        self.name = "frames"

        if isinstance(source, (str, Path)):
            self._path = Path(source).expanduser().resolve()
            if not self._path.exists():
                raise FileNotFoundError(f"video not found: {self._path}")
            self.name = self._path.name
            self._capture = cv2.VideoCapture(str(self._path))
            self._read_metadata()
        elif isinstance(source, int):
            self._camera = source
            self.name = f"camera_{source}"
            self._capture = cv2.VideoCapture(source)
            self._read_metadata(camera=True)
        elif hasattr(source, "__iter__"):
            self._iterator = iter(source)
        else:
            raise TypeError(f"unsupported video source: {type(source)!r}")

        if fps_override is not None and fps_override > 0:
            self.fps = float(fps_override)

    def _read_metadata(self, *, camera: bool = False) -> None:
        assert self._capture is not None
        if not self._capture.isOpened():
            target = self._path or self._camera
            raise RuntimeError(f"could not open video source: {target}")
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = width
        self.height = height
        self.frame_count = 0 if camera else max(0, count)
        if not camera and 0.1 < fps < 1000:
            self.fps = fps

    @property
    def duration(self) -> float | None:
        if self.frame_count > 0 and self.fps > 0:
            return self.frame_count / self.fps
        return None

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        index = 0
        last_timestamp = -1.0
        metadata_start = time.monotonic()
        while True:
            if self._capture is not None:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    break
                position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                if self._camera is not None:
                    timestamp = time.monotonic() - metadata_start
                elif position_ms > 0:
                    timestamp = position_ms / 1000.0
                else:
                    timestamp = index / self.fps
            else:
                assert self._iterator is not None
                try:
                    frame = next(self._iterator)
                except StopIteration:
                    break
                timestamp = index / self.fps

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1.0 / max(1e-3, self.fps)
            last_timestamp = timestamp
            yield frame, timestamp, index
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._iterator = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
