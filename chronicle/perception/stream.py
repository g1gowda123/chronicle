"""Non-blocking delivery of perception results for the telemetry UI."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.types import FrameResult


class PerceptionStream:
    """Run a pipeline on a worker thread and expose a bounded result queue.

    The UI can drop frames while it repaints; the graph/state consumer should
    use ``queue_size=0`` (unbounded) or call :meth:`PerceptionPipeline.run`
    directly if every frame must be processed in order.
    """

    _SENTINEL = None

    def __init__(
        self,
        pipeline: PerceptionPipeline,
        source: str | Path | int | Iterable[np.ndarray],
        *,
        queue_size: int = 32,
        drop_oldest: bool = True,
        fps_override: float | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.source = source
        self.fps_override = fps_override
        self.drop_oldest = drop_oldest
        self.queue: queue.Queue[FrameResult | None] = queue.Queue(
            maxsize=0 if queue_size <= 0 else queue_size
        )
        self.errors: queue.Queue[BaseException] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> PerceptionStream:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="chronicle-perception", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for result in self.pipeline.iter_results(
                self.source,
                stop_event=self._stop,
                fps_override=self.fps_override,
            ):
                self._put(result)
                if self._stop.is_set():
                    break
        except BaseException as exc:  # surfaced to the consumer thread
            self.errors.put(exc)
        finally:
            self._put(self._SENTINEL)

    def _put(self, item: FrameResult | None) -> None:
        try:
            self.queue.put_nowait(item)
            return
        except queue.Full:
            if not self.drop_oldest:
                while not self._stop.is_set():
                    try:
                        self.queue.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue
                return
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __iter__(self) -> Iterator[FrameResult]:
        while True:
            item = self.queue.get()
            if item is self._SENTINEL:
                break
            yield item

    def close(self) -> None:
        self.stop()
        self.join(timeout=5.0)
        self.pipeline.close()

    def __enter__(self) -> PerceptionStream:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
