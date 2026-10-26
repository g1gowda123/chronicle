"""In-process telemetry bus for the NiceGUI dashboard.

The ingestion thread must never block on the UI.  Role B therefore emits plain
immutable events into a thread-safe bus; Role D subscribes and forwards them to
``ui.log`` / WebSocket clients.  Subscriber failures are swallowed so a crashed
dashboard can never stall perception.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TelemetryTag(str, Enum):
    """Log channel labels rendered as ``[TAG]`` in the dashboard terminal."""

    GEOM = "GEOM"
    STATE = "STATE"
    VLM = "VLM"
    GRAPH = "GRAPH"
    QUERY = "QUERY"
    STORE = "STORE"
    SYSTEM = "SYSTEM"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class TelemetryEvent:
    tag: TelemetryTag
    message: str
    t: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        stamp = f"t={self.t:8.3f}s  " if self.t is not None else ""
        extras = ""
        if self.data:
            extras = "  " + " ".join(f"{k}={_short(v)}" for k, v in self.data.items())
        return f"[{self.tag.value}] {stamp}{self.message}{extras}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag.value,
            "message": self.message,
            "t": self.t,
            "data": dict(self.data),
        }


def _short(value: Any, limit: int = 60) -> str:
    text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


Subscriber = Callable[[TelemetryEvent], None]


class EventBus:
    """Thread-safe pub/sub with a replayable ring buffer."""

    def __init__(self, history_size: int = 1000) -> None:
        self._lock = threading.RLock()
        self._subscribers: list[Subscriber] = []
        self._history: deque[TelemetryEvent] = deque(maxlen=history_size)

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            self.unsubscribe(callback)

        return unsubscribe

    def unsubscribe(self, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(
        self,
        tag: TelemetryTag | str,
        message: str,
        t: float | None = None,
        **data: Any,
    ) -> TelemetryEvent:
        event = TelemetryEvent(tag=TelemetryTag(tag), message=message, t=t, data=dict(data))
        with self._lock:
            self._history.append(event)
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # pragma: no cover - defensive isolation
                continue
        return event

    # Convenience channel helpers -------------------------------------------------
    def geom(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.GEOM, message, t, **data)

    def state(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.STATE, message, t, **data)

    def graph(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.GRAPH, message, t, **data)

    def store(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.STORE, message, t, **data)

    def query(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.QUERY, message, t, **data)

    def vlm(self, message: str, t: float | None = None, **data: Any) -> TelemetryEvent:
        return self.emit(TelemetryTag.VLM, message, t, **data)

    # Replay ----------------------------------------------------------------------
    def history(
        self, tag: TelemetryTag | str | None = None, limit: int | None = None
    ) -> list[TelemetryEvent]:
        with self._lock:
            events = list(self._history)
        if tag is not None:
            wanted = TelemetryTag(tag)
            events = [e for e in events if e.tag is wanted]
        if limit is not None:
            events = events[-limit:]
        return events

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
