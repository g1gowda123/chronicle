"""Bounded verification queue with backpressure, de-duplication and telemetry.

Role A can emit interaction clips faster than Florence-2 + the action classifier
can verify them.  An unbounded queue silently grows the demo's memory and adds
minutes of latency to the graph.  This module keeps the queue small and makes
the trade-off explicit:

* **bounded** — at most ``max_pending`` clips wait for verification;
* **backpressure** — ``drop_oldest`` (freshest state wins) or ``reject``;
* **de-duplicated** — the same ``clip_id`` is never verified twice;
* **observable** — ``QUEUE_DEPTH`` / ``QUEUE_BACKPRESSURE`` /
  ``VERIFICATION_LATENCY`` events go to the Role C telemetry sink, and
  ``stats()`` is safe to read from the dashboard thread.

The runtime wires the queue into its ingestion loop; ``VerificationService`` is
the convenience wrapper for standalone runs and demos.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger("chronicle.semantic.queue")

EventCallback = Callable[[str, str, str, dict[str, Any]], None]


def clip_key(clip: Any) -> str:
    """Stable de-duplication key; falls back to identity for bare buffers."""
    clip_id = getattr(clip, "clip_id", None)
    if clip_id:
        return str(clip_id)
    return f"anon-{id(clip)}"


@dataclass
class QueueStats:
    depth: int = 0
    max_depth: int = 0
    max_pending: int = 0
    submitted: int = 0
    accepted: int = 0
    verified: int = 0
    failed: int = 0
    duplicates: int = 0
    dropped: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    policy: str = "drop_oldest"

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.verified if self.verified else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "max_depth": self.max_depth,
            "max_pending": self.max_pending,
            "submitted": self.submitted,
            "accepted": self.accepted,
            "verified": self.verified,
            "failed": self.failed,
            "duplicates": self.duplicates,
            "dropped": self.dropped,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "policy": self.policy,
        }


class VerificationQueue:
    """Thread-safe bounded FIFO with clip de-duplication and metrics."""

    def __init__(
        self,
        *,
        max_pending: int = 8,
        policy: str = "drop_oldest",
        dedup_window: int = 1024,
        on_event: EventCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        if policy not in {"drop_oldest", "reject"}:
            raise ValueError(f"unknown policy {policy!r}")
        if dedup_window < 0:
            raise ValueError("dedup_window must be >= 0")

        self.max_pending = int(max_pending)
        self.policy = policy
        self.dedup_window = int(dedup_window)
        self._on_event = on_event
        self._clock = clock
        self._pending: deque[tuple[Any, float]] = deque()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._stats = QueueStats(max_pending=self.max_pending, policy=policy)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

    # ------------------------------------------------------------------ events
    def _emit(self, event: str, message: str, **data: Any) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event("VLM", event, message, data)
        except Exception:  # noqa: BLE001 - telemetry must never block ingestion
            logger.exception("verification queue event sink failed")

    # ------------------------------------------------------------------ producer
    def submit(self, clip: Any, *, now: float | None = None) -> bool:
        """Queue a clip.  Returns False when de-duplicated or rejected."""
        key = clip_key(clip)
        submitted_at = self._clock() if now is None else float(now)
        with self._condition:
            if key in self._seen:
                self._stats.duplicates += 1
                self._emit(
                    "CLIP_DEDUPLICATED",
                    f"skipped duplicate clip {key}",
                    clip=key,
                    duplicates=self._stats.duplicates,
                )
                return False

            if len(self._pending) >= self.max_pending:
                if self.policy == "drop_oldest":
                    dropped_clip, _ = self._pending.popleft()
                    dropped_key = clip_key(dropped_clip)
                    self._stats.dropped += 1
                    self._emit(
                        "QUEUE_BACKPRESSURE",
                        f"queue full; dropped oldest clip {dropped_key}",
                        clip=dropped_key,
                        depth=len(self._pending),
                        max_pending=self.max_pending,
                        dropped=self._stats.dropped,
                        policy=self.policy,
                    )
                else:
                    self._stats.dropped += 1
                    self._emit(
                        "QUEUE_BACKPRESSURE",
                        f"queue full; rejected clip {key}",
                        clip=key,
                        depth=len(self._pending),
                        max_pending=self.max_pending,
                        dropped=self._stats.dropped,
                        policy=self.policy,
                    )
                    return False

            self._seen[key] = None
            if self.dedup_window and len(self._seen) > self.dedup_window:
                self._seen.popitem(last=False)
            self._pending.append((clip, submitted_at))
            self._stats.submitted += 1
            self._stats.accepted += 1
            self._stats.depth = len(self._pending)
            self._stats.max_depth = max(self._stats.max_depth, self._stats.depth)
            self._emit(
                "QUEUE_DEPTH",
                f"verification queue depth {self._stats.depth}/{self.max_pending}",
                depth=self._stats.depth,
                max_pending=self.max_pending,
                clip=key,
            )
            self._condition.notify()
            return True

    # ------------------------------------------------------------------ consumer
    def get_nowait(self) -> tuple[Any, float] | None:
        with self._condition:
            if not self._pending:
                return None
            clip, submitted_at = self._pending.popleft()
            self._stats.depth = len(self._pending)
            return clip, submitted_at

    def get(self, timeout: float | None = None) -> tuple[Any, float] | None:
        """Blocking pop used by :class:`VerificationService` workers."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while not self._pending:
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            clip, submitted_at = self._pending.popleft()
            self._stats.depth = len(self._pending)
            return clip, submitted_at

    def drain(self) -> Iterator[tuple[Any, float]]:
        while True:
            item = self.get_nowait()
            if item is None:
                return
            yield item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    # ------------------------------------------------------------------ results
    def mark_verified(self, latency_s: float) -> None:
        self._mark(latency_s, failed=False)

    def mark_failed(self, latency_s: float) -> None:
        self._mark(latency_s, failed=True)

    def _mark(self, latency_s: float, *, failed: bool) -> None:
        latency_ms = max(0.0, float(latency_s) * 1000.0)
        with self._lock:
            if failed:
                self._stats.failed += 1
            else:
                self._stats.verified += 1
                self._stats.total_latency_ms += latency_ms
                self._stats.max_latency_ms = max(
                    self._stats.max_latency_ms, latency_ms
                )
            self._stats.last_latency_ms = latency_ms
            event = "VERIFICATION_FAILED" if failed else "VERIFICATION_LATENCY"
            message = (
                f"clip verification failed after {latency_ms:.1f} ms"
                if failed
                else f"clip verified in {latency_ms:.1f} ms"
            )
            self._emit(
                event,
                message,
                latency_ms=round(latency_ms, 2),
                avg_latency_ms=round(self._stats.avg_latency_ms, 2),
                depth=self._stats.depth,
                verified=self._stats.verified,
                failed=self._stats.failed,
            )

    # ------------------------------------------------------------------ metrics
    def stats(self) -> QueueStats:
        with self._lock:
            return replace(self._stats)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(clip_key(clip) for clip, _ in self._pending)

    def __len__(self) -> int:
        return self.pending

    def empty(self) -> bool:
        return self.pending == 0


class VerificationService:
    """Drain a :class:`VerificationQueue` through a ``SemanticEngine``.

    The engine is resolved lazily because ``ChronicleRuntime`` creates its
    semantic layer on first use.  Pass ``start()`` to verify on a background
    thread, or call ``drain()`` inline for deterministic demos/tests.
    """

    def __init__(
        self,
        engine_provider: Callable[[], Any | None],
        *,
        max_pending: int = 8,
        policy: str = "drop_oldest",
        dedup_window: int = 1024,
        on_event: EventCallback | None = None,
        on_verified: Callable[[Any], None] | None = None,
    ) -> None:
        self._engine_provider = engine_provider
        self._on_event = on_event
        self._on_verified = on_verified
        self.last_error: str | None = None
        self.queue = VerificationQueue(
            max_pending=max_pending,
            policy=policy,
            dedup_window=dedup_window,
            on_event=self._emit,
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active = 0
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------ telemetry
    def _emit(self, tag: str, event: str, message: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(tag, event, message, data)
            return
        engine = self._engine_provider()
        if engine is not None:
            engine.emit(tag, event, message, **data)

    # ------------------------------------------------------------------ producer
    def submit(self, clip: Any) -> bool:
        return self.queue.submit(clip)

    def get_nowait(self) -> tuple[Any, float] | None:
        return self.queue.get_nowait()

    # ------------------------------------------------------------------ consumer
    def verify_one(self, item: tuple[Any, float]) -> Any | None:
        clip, submitted_at = item
        engine = self._engine_provider()
        if engine is None:
            self.queue.mark_failed(self._elapsed(submitted_at))
            self.last_error = "semantic engine unavailable"
            return None
        try:
            result = engine.verify_interaction(clip)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the worker
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("clip verification failed")
            self.queue.mark_failed(self._elapsed(submitted_at))
            return None
        self.queue.mark_verified(self._elapsed(submitted_at))
        if self._on_verified is not None:
            try:
                self._on_verified(result)
            except Exception:  # noqa: BLE001 - callback isolation
                logger.exception("on_verified callback failed")
        return result

    def drain(self) -> int:
        processed = 0
        while True:
            item = self.queue.get_nowait()
            if item is None:
                return processed
            self.verify_one(item)
            processed += 1

    def _elapsed(self, submitted_at: float) -> float:
        return max(0.0, time.monotonic() - float(submitted_at))

    # ------------------------------------------------------------------ worker
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="chronicle-verify", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self.queue.get(timeout=0.1)
            if item is None:
                continue
            with self._state_lock:
                self._active += 1
            try:
                self.verify_one(item)
            finally:
                with self._state_lock:
                    self._active -= 1

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            with self._state_lock:
                active = self._active
            if self.queue.empty() and active == 0:
                return True
            time.sleep(0.01)
        return False

    def stats(self) -> QueueStats:
        return self.queue.stats()
