"""Thread-safe telemetry fan-out for the perception pipeline."""

from __future__ import annotations

import queue
import threading
from collections import deque

from chronicle.types import TelemetryEvent


class TelemetryBus:
    """Fan-out bus with bounded subscriber queues.

    Ingestion never blocks on a slow consumer: when a subscriber is full, its
    oldest event is dropped so the newest one still arrives.
    """

    def __init__(self, history: int = 2000) -> None:
        self._lock = threading.RLock()
        self._subscribers: set[queue.Queue[TelemetryEvent]] = set()
        self._history: deque[TelemetryEvent] = deque(maxlen=max(1, history))

    def subscribe(self, maxsize: int = 256) -> queue.Queue[TelemetryEvent]:
        subscriber: queue.Queue[TelemetryEvent] = queue.Queue(maxsize=max(1, maxsize))
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[TelemetryEvent]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._history.append(event)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass

    def __call__(self, event: TelemetryEvent) -> None:
        self.publish(event)

    def recent(self, count: int = 100) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._history)[-max(1, count):]

    @staticmethod
    def drain(subscriber: queue.Queue[TelemetryEvent], limit: int = 200) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        for _ in range(max(0, limit)):
            try:
                events.append(subscriber.get_nowait())
            except queue.Empty:
                break
        return events


class TelemetryPrinter:
    """Console sink matching the spec's ``[GEOM]`` / ``[TRACK]`` log format."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, event: TelemetryEvent) -> None:
        if self.enabled:
            print(event.text, flush=True)
