"""Telemetry bus behaviour used by Role D's dashboard."""

from __future__ import annotations

from chronicle.telemetry import EventBus, TelemetryTag


def test_events_are_tagged_and_replayable() -> None:
    bus = EventBus(history_size=10)
    bus.state("NEW -> VISIBLE_STATIC", 1.25, entity="cup")
    bus.graph("RESTING_ON edge", 1.25, anchor="anchor:table")

    events = bus.history()
    assert [event.tag for event in events] == [TelemetryTag.STATE, TelemetryTag.GRAPH]
    assert bus.history(TelemetryTag.GRAPH)[0].data["anchor"] == "anchor:table"
    assert bus.history(limit=1)[0].tag is TelemetryTag.GRAPH


def test_format_is_dashboard_ready() -> None:
    bus = EventBus()
    event = bus.geom("camera pan detected", 12.3, magnitude_px=31.0)
    rendered = event.format()
    assert rendered.startswith("[GEOM]")
    assert "t=  12.300s" in rendered
    assert "magnitude_px=31.0" in rendered


def test_subscribers_are_isolated_and_removable() -> None:
    bus = EventBus()
    received: list[str] = []

    def good(event) -> None:
        received.append(event.message)

    def bad(event) -> None:
        raise RuntimeError("dashboard exploded")

    unsubscribe_bad = bus.subscribe(bad)
    unsubscribe_good = bus.subscribe(good)
    bus.emit("STATE", "hello", 1.0)
    assert received == ["hello"]

    unsubscribe_bad()
    unsubscribe_good()
    bus.emit("STATE", "after", 2.0)
    assert received == ["hello"]


def test_ring_buffer_is_bounded() -> None:
    bus = EventBus(history_size=3)
    for index in range(10):
        bus.store("sweep", float(index))
    assert len(bus.history()) == 3
    assert bus.history()[-1].t == 9.0
