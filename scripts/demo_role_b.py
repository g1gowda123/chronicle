"""Synthetic Role B demo: static -> interact -> lift -> place -> ask -> restart.

No video, no models, no GPU: frames are generated geometrically so the demo is
fully deterministic.  Run it from the repository root:

    python scripts/demo_role_b.py

The script prints the same telemetry Roles C and D subscribe to, then shows the
dual-graph statistics, the 2-hop retrieval packet and a persisted answer after
reopening the SQLite database.
"""

from __future__ import annotations

# ruff: noqa: E402 - allow direct `python scripts/demo_role_b.py` from the repo root
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shapely.geometry import Polygon

from chronicle import (
    ChronicleMemory,
    EventBus,
    FrameContext,
    HandObservation,
    ObjectObservation,
    SupportSurface,
    TelemetryTag,
)

FPS = 30
DT = 1.0 / FPS
SHOW_TAGS = {TelemetryTag.STATE, TelemetryTag.GRAPH, TelemetryTag.QUERY}

TABLE = SupportSurface(
    anchor_id="dining_table",
    label="dining table",
    polygon=Polygon([(0, 380), (400, 380), (400, 600), (0, 600)]),
)
COUNTER = SupportSurface(
    anchor_id="kitchen_counter",
    label="kitchen counter",
    polygon=Polygon([(400, 380), (800, 380), (800, 600), (400, 600)]),
)


def hand_at(x: float, y: float, size: float = 80.0) -> HandObservation:
    points = [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x + size / 2, y + size / 2),
    ]
    return HandObservation.from_landmarks(points, hand_id="hand:0")


def scripted_scene(t: float) -> tuple[tuple[float, float, float, float], HandObservation | None]:
    """Piecewise cup/hand trajectory: rest, grasp, lift, carry, lower, release."""

    if t < 1.2:  # rest on the dining table
        return (100.0, 300.0, 180.0, 380.0), None
    if t < 1.6:  # hand closes on the cup: INTERACTED after 0.3 s dwell
        return (100.0, 300.0, 180.0, 380.0), hand_at(100.0, 300.0)
    if t < 2.2:  # lift 120 px while the hand stays locked to the cup
        dy = (t - 1.6) / 0.6 * 120.0
        return (100.0, 300.0 - dy, 180.0, 380.0 - dy), hand_at(100.0, 300.0 - dy)
    if t < 2.8:  # carry 400 px to the right, still held
        dx = (t - 2.2) / 0.6 * 400.0
        return (100.0 + dx, 180.0, 180.0 + dx, 260.0), hand_at(100.0 + dx, 180.0)
    if t < 3.4:  # lower onto the kitchen counter
        dy = (t - 2.8) / 0.6 * 120.0
        return (500.0, 180.0 + dy, 580.0, 260.0 + dy), hand_at(500.0, 180.0 + dy)
    return (500.0, 300.0, 580.0, 380.0), None


def run_scene(memory: ChronicleMemory, until: float) -> None:
    steps = round(until / DT)
    for index in range(steps):
        t = index * DT
        bbox, hand = scripted_scene(t)
        frame_objects = [ObjectObservation(entity_id="cup", label="cup", bbox=bbox)]
        frame_hands = [hand] if hand is not None else []
        memory.observe(
            memory_frame(t, frame_objects, frame_hands),
            on_transition=vlm_hook,
        )


def seed_legacy_duplicate(memory: ChronicleMemory, t0: float = 5.0) -> None:
    """Emit a second id for the same physical cup, as an old track would."""

    steps = round(0.7 / DT)
    for index in range(steps):
        t = t0 + index * DT
        memory.observe(
            memory_frame(
                t,
                [
                    ObjectObservation(
                        entity_id="cup-legacy",
                        label="cup",
                        bbox=(500.0, 300.0, 580.0, 380.0),
                        track_id=99,
                    )
                ],
                [],
            )
        )


def memory_frame(t, objects, hands):
    return FrameContext(
        t=t,
        objects=objects,
        hands=hands,
        supports=(TABLE, COUNTER),
        frame_size=(800, 600),
        zone_id="kitchen",
        attributes={"zone_label": "kitchen"},
    )


def vlm_hook(transition, frame) -> None:
    """Stand-in for Role C's Florence-2 verification queue."""

    if transition.to_state.value in {"INTERACTED", "HELD"}:
        print(
            f"      [VLM] enqueue verify {transition.tag} "
            f"entity={transition.entity_id} t={transition.t_event:.2f}s"
        )


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="chronicle-demo-"))
    db_path = temp_dir / "chronicle.db"

    bus = EventBus()
    bus.subscribe(lambda event: print(event.format()) if event.tag in SHOW_TAGS else None)

    print("Chronicle Role B demo - nine-state lifecycle + dual temporal graph")
    print(f"SQLite: {db_path}\n")

    try:
        with ChronicleMemory(db_path, bus=bus) as memory:
            run_scene(memory, until=4.6)

            print("\n--- dual-graph state ---")
            stats = memory.stats()
            print(
                "background graph: "
                f"{memory.background.graph.number_of_nodes()} nodes / "
                f"{memory.background.graph.number_of_edges()} edges "
                "(zones + furniture anchors)"
            )
            print(
                "foreground graph: "
                f"{memory.foreground.graph.number_of_nodes()} nodes / "
                f"{memory.foreground.graph.number_of_edges()} edges "
                "(entities + time-intervaled relations)"
            )
            print(f"SQLite projection: {json.dumps(stats, sort_keys=True)}")

            print("\n--- retrieval ---")
            result = memory.query("where are my cups?")
            if result is None:
                print("no entity resolved")
                return
            print(
                f"resolved: {result.entity_id} "
                f"(match={result.match_score:.2f}, "
                f"confidence={result.confidence:.2f}, "
                f"reason={result.match_reason}, state={result.state})"
            )
            print(f"answer: {result.fallback_answer()}\n")
            print("2-hop subgraph:")
            print(result.to_table())
            anchor = result.terminal_anchor or {}
            print(
                f"\nterminal anchor retained: {anchor.get('anchor_id')} "
                f"({anchor.get('label')}) t={anchor.get('t_start')}"
            )

            print("\n--- duplicate consolidation ---")
            seed_legacy_duplicate(memory)
            candidates = memory.consolidator.find_candidates(5.7)
            print(
                "candidates: "
                + json.dumps(
                    [candidate.as_dict() for candidate in candidates],
                    sort_keys=True,
                )
            )
            planned = memory.merge_entities("cup", "cup-legacy", dry_run=True)
            print(f"dry run: {planned.status} ({planned.reason})")
            report = memory.merge_entities("cup", "cup-legacy", t=5.7)
            print(f"merge: {report.status} preserved={json.dumps(report.preserved)}")
            print(
                f"legacy id now resolves to: {memory.resolve_entity_alias('cup-legacy')}"
            )
            print(f"merge stats: {json.dumps(memory.stats()['merges'], sort_keys=True)}")

            print("\n--- process restart ---")

        with ChronicleMemory(db_path, bus=bus) as restored:
            result = restored.query("where is my cup?")
            print(f"restored answer: {result.fallback_answer() if result else 'none'}")
            alias = restored.query_entity("cup-legacy")
            print(f"restored legacy alias -> {alias.entity_id if alias else 'none'}")
    finally:
        for child in sorted(temp_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
        temp_dir.rmdir()


if __name__ == "__main__":
    main()
