"""Role C: bounded verification queue, backpressure and telemetry."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from chronicle.integration import ChronicleRuntime
from chronicle.semantic import (
    SemanticConfig,
    SemanticEngine,
    VerificationQueue,
    VerificationService,
)
from chronicle.semantic.florence import StubCaptioner
from chronicle.semantic.temporal_strip import FrameSample


def make_clip(clip_id: str, label: str = "keys") -> SimpleNamespace:
    frames = [
        FrameSample(image=None, timestamp=float(index), index=index)
        for index in range(5)
    ]
    return SimpleNamespace(
        clip_id=clip_id,
        track_id=1,
        label=label,
        hand="Right",
        frames=frames,
        contact_index=2,
        contact_ts=2.0,
        trigger="ios+dwell",
        max_ios=0.7,
        metadata={},
    )


def make_visual_clip(clip_id: str, label: str = "keys") -> SimpleNamespace:
    """Clip with real pixels so the offline semantic pipeline can verify it."""

    clip = make_clip(clip_id, label)
    clip.frames = [
        FrameSample(
            image=Image.new("RGB", (24, 24), (40 + index, 80, 120)),
            timestamp=float(index),
            index=index,
        )
        for index in range(6)
    ]
    return clip


class FakeEngine:
    """Minimal engine used by the service tests (no models, no frames)."""

    def __init__(self, fail_ids: tuple[str, ...] = ()) -> None:
        self.fail_ids = set(fail_ids)
        self.calls: list[str] = []
        self.events: list[dict] = []

    def verify_interaction(self, clip: SimpleNamespace) -> SimpleNamespace:
        self.calls.append(clip.clip_id)
        if clip.clip_id in self.fail_ids:
            raise RuntimeError("VLM unavailable")
        return SimpleNamespace(clip_id=clip.clip_id, action="PICK_UP")

    def emit(self, tag: str, event: str, message: str, **data) -> None:
        self.events.append(
            {"tag": tag, "event": event, "message": message, "data": data}
        )


class VerificationQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def event_sink(self, tag: str, event: str, message: str, data: dict) -> None:
        self.events.append((event, message, data))

    def test_same_clip_id_is_not_queued_twice(self) -> None:
        queue = VerificationQueue(max_pending=4, on_event=self.event_sink)
        self.assertTrue(queue.submit(make_clip("clip-1")))
        self.assertFalse(queue.submit(make_clip("clip-1")))
        self.assertEqual(queue.pending, 1)
        self.assertEqual(queue.stats().duplicates, 1)
        self.assertIn("CLIP_DEDUPLICATED", [event for event, _, _ in self.events])

    def test_bounded_queue_drops_oldest_under_backpressure(self) -> None:
        queue = VerificationQueue(max_pending=2, on_event=self.event_sink)
        for clip_id in ("a", "b", "c"):
            queue.submit(make_clip(clip_id))
        self.assertEqual(queue.pending_ids(), ("b", "c"))
        stats = queue.stats()
        self.assertEqual(stats.dropped, 1)
        self.assertEqual(stats.max_depth, 2)
        backpressure = [data for event, _, data in self.events if event == "QUEUE_BACKPRESSURE"]
        self.assertEqual(len(backpressure), 1)
        self.assertEqual(backpressure[0]["clip"], "a")

    def test_reject_policy_keeps_oldest_and_refuses_new(self) -> None:
        queue = VerificationQueue(max_pending=1, policy="reject")
        self.assertTrue(queue.submit(make_clip("first")))
        self.assertFalse(queue.submit(make_clip("second")))
        self.assertEqual(queue.pending_ids(), ("first",))
        self.assertEqual(queue.stats().dropped, 1)

    def test_fifo_order(self) -> None:
        queue = VerificationQueue(max_pending=4)
        for clip_id in ("one", "two", "three"):
            queue.submit(make_clip(clip_id))
        order = [queue.get_nowait()[0].clip_id for _ in range(3)]
        self.assertEqual(order, ["one", "two", "three"])
        self.assertIsNone(queue.get_nowait())

    def test_latency_statistics_are_emitted(self) -> None:
        queue = VerificationQueue(on_event=self.event_sink)
        queue.mark_verified(0.2)
        queue.mark_verified(0.4)
        queue.mark_failed(0.1)
        stats = queue.stats()
        self.assertEqual(stats.verified, 2)
        self.assertEqual(stats.failed, 1)
        self.assertAlmostEqual(stats.avg_latency_ms, 300.0, places=3)
        self.assertAlmostEqual(stats.max_latency_ms, 400.0, places=3)
        self.assertAlmostEqual(stats.last_latency_ms, 100.0, places=3)
        latency_events = [event for event, _, _ in self.events if event == "VERIFICATION_LATENCY"]
        self.assertEqual(len(latency_events), 2)
        self.assertIn("VERIFICATION_FAILED", [event for event, _, _ in self.events])

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            VerificationQueue(max_pending=0)
        with self.assertRaises(ValueError):
            VerificationQueue(policy="block")


class VerificationServiceTest(unittest.TestCase):
    def test_drain_verifies_and_records_metrics(self) -> None:
        engine = FakeEngine()
        service = VerificationService(lambda: engine, max_pending=4)
        service.submit(make_clip("a"))
        service.submit(make_clip("b"))
        service.submit(make_clip("a"))  # duplicate
        self.assertEqual(service.drain(), 2)
        self.assertEqual(engine.calls, ["a", "b"])
        stats = service.stats()
        self.assertEqual(stats.verified, 2)
        self.assertEqual(stats.duplicates, 1)
        self.assertEqual(stats.depth, 0)
        events = [event["event"] for event in engine.events]
        self.assertEqual(events.count("VERIFICATION_LATENCY"), 2)

    def test_failure_marks_failed_and_keeps_draining(self) -> None:
        engine = FakeEngine(fail_ids=("bad",))
        service = VerificationService(lambda: engine)
        service.submit(make_clip("bad"))
        service.submit(make_clip("good"))
        self.assertEqual(service.drain(), 2)
        stats = service.stats()
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.verified, 1)
        self.assertIn("VLM unavailable", service.last_error or "")

    def test_missing_engine_marks_failed(self) -> None:
        service = VerificationService(lambda: None)
        service.submit(make_clip("a"))
        service.drain()
        self.assertEqual(service.stats().failed, 1)

    def test_background_worker(self) -> None:
        engine = FakeEngine()
        service = VerificationService(lambda: engine)
        service.start()
        try:
            service.submit(make_clip("a"))
            self.assertTrue(service.wait_idle(timeout=5.0))
            self.assertEqual(engine.calls, ["a"])
            self.assertEqual(service.stats().verified, 1)
        finally:
            service.stop()

    def test_engine_exposes_a_service(self) -> None:
        engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template"),
            captioner=StubCaptioner("the hand picks up the keys"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
            on_event=lambda payload: None,
        )
        clip = make_clip("clip-1")
        clip.frames = [
            FrameSample(
                image=Image.new("RGB", (24, 24), (40 + index, 80, 120)),
                timestamp=float(index),
                index=index,
            )
            for index in range(6)
        ]
        self.assertTrue(engine.submit_clip(clip))
        self.assertFalse(engine.submit_clip(clip))
        self.assertEqual(engine.drain_verifications(), 1)
        stats = engine.queue_stats()
        self.assertEqual(stats["verified"], 1)
        self.assertEqual(stats["duplicates"], 1)
        self.assertEqual(engine.verification_service().stats().max_pending, 8)


class RuntimeQueueSeamTest(unittest.TestCase):
    def test_runtime_uses_the_role_c_queue(self) -> None:
        engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template"),
            captioner=StubCaptioner("the hand picks up the keys"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
        )
        runtime = ChronicleRuntime(
            ":memory:", semantic=engine, reset=True, load=False
        )
        try:
            clip = make_clip("runtime-1")
            clip.frames = [
                FrameSample(
                    image=Image.new("RGB", (24, 24), (40 + index, 80, 120)),
                    timestamp=float(index),
                    index=index,
                )
                for index in range(6)
            ]
            runtime._queue_clip(clip)
            runtime._queue_clip(clip)
            self.assertEqual(runtime._clip_queue.pending, 1)
            runtime._drain_clips()
            stats = runtime.verification_stats
            self.assertEqual(stats["verified"], 1)
            self.assertEqual(stats["duplicates"], 1)
        finally:
            runtime.close()

    def test_snapshot_exposes_verification_queue_stats(self) -> None:
        config = SemanticConfig(llm_backend="template", verification_queue_size=1)
        engine = SemanticEngine(
            config=config,
            captioner=StubCaptioner("the hand picks up the keys"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
        )
        runtime = ChronicleRuntime(
            ":memory:",
            semantic=engine,
            semantic_config=config,
            reset=True,
            load=False,
        )
        try:
            runtime._queue_clip(make_visual_clip("queue-a"))
            runtime._queue_clip(make_visual_clip("queue-b"))

            snapshot = runtime.snapshot()
            verification = snapshot["verification_stats"]
            self.assertEqual(verification["depth"], 1)
            self.assertEqual(verification["dropped"], 1)
            self.assertEqual(verification["max_pending"], 1)
            self.assertIn("avg_latency_ms", verification)
            self.assertEqual(snapshot["stats"]["verification_stats"], verification)

            runtime._drain_clips()
            after = runtime.snapshot()["stats"]["verification_stats"]
            self.assertEqual(after["depth"], 0)
            self.assertEqual(after["verified"], 1)
            self.assertGreaterEqual(after["avg_latency_ms"], 0.0)
        finally:
            runtime.close()

    def test_runtime_snapshot_and_export_include_verification_telemetry(self) -> None:
        engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template"),
            captioner=StubCaptioner("the hand picks up the keys"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
        )
        runtime = ChronicleRuntime(
            ":memory:", semantic=engine, reset=True, load=False
        )
        try:
            clip = make_clip("runtime-test-1")
            clip.frames = [
                FrameSample(
                    image=Image.new("RGB", (24, 24), (50, 50, 50)),
                    timestamp=float(i),
                    index=i,
                )
                for i in range(4)
            ]
            runtime._queue_clip(clip)
            snapshot = runtime.snapshot()
            self.assertIn("verification", snapshot)
            self.assertEqual(snapshot["stats"]["verification_queue_depth"], 1)
            self.assertEqual(snapshot["verification"]["depth"], 1)

            # JSON export
            exported_json = runtime.export_snapshot_json()
            self.assertIn('"verification_queue_depth": 1', exported_json)

            # Drain and verify stats update
            runtime._drain_clips()
            snapshot2 = runtime.snapshot()
            self.assertEqual(snapshot2["stats"]["verification_queue_depth"], 0)
            self.assertEqual(snapshot2["stats"]["verification_verified"], 1)

            # CSV exports
            csv_entities = runtime.export_entities_csv()
            self.assertIn("id,label,state,zone,anchor,container,span", csv_entities)
            csv_actions = runtime.export_actions_csv()
            self.assertIn("id,timestamp,action,entity,caption,source", csv_actions)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
