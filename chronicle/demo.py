"""Model-free Chronicle demo.

This path exists so a new user can see the memory layer work before installing
torch, YOLO-World, MediaPipe, Florence-2, or the NiceGUI dashboard.  Frames are
generated geometrically and fed straight into :class:`ChronicleMemory`; no video
decoder, model weights, GPU, or network access are involved.

The scene is intentionally small: a cup is grasped on a dining table, carried
to a kitchen counter, and released.  The demo then runs two graph queries and
reopens the SQLite database to show that the answer survives a process restart.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

from chronicle.graph.memory import ChronicleMemory
from chronicle.types import (
    FrameContext,
    HandObservation,
    ObjectObservation,
    SupportSurface,
)

FPS = 30
DT = 1.0 / FPS
SCENE_SECONDS = 4.6
QUESTIONS = ("where is my cup?", "what happened to my cup?")

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


def _hand_at(x: float, y: float, size: float = 80.0) -> HandObservation:
    points = [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x + size / 2, y + size / 2),
    ]
    return HandObservation.from_landmarks(points, hand_id="hand:0")


def _scene(t: float) -> tuple[tuple[float, float, float, float], HandObservation | None]:
    """Piecewise cup/hand trajectory: rest, grasp, lift, carry, lower, release."""

    if t < 1.2:  # rest on the dining table
        return (100.0, 300.0, 180.0, 380.0), None
    if t < 1.6:  # hand closes on the cup: INTERACTED after a short dwell
        return (100.0, 300.0, 180.0, 380.0), _hand_at(100.0, 300.0)
    if t < 2.2:  # lift 120 px while the hand stays locked to the cup
        dy = (t - 1.6) / 0.6 * 120.0
        return (100.0, 300.0 - dy, 180.0, 380.0 - dy), _hand_at(100.0, 300.0 - dy)
    if t < 2.8:  # carry 400 px to the right, still held
        dx = (t - 2.2) / 0.6 * 400.0
        return (100.0 + dx, 180.0, 180.0 + dx, 260.0), _hand_at(100.0 + dx, 180.0)
    if t < 3.4:  # lower onto the kitchen counter
        dy = (t - 2.8) / 0.6 * 120.0
        return (500.0, 180.0 + dy, 580.0, 260.0 + dy), _hand_at(500.0, 180.0 + dy)
    return (500.0, 300.0, 580.0, 380.0), None


def _frame(
    t: float,
    objects: list[ObjectObservation],
    hands: list[HandObservation],
) -> FrameContext:
    return FrameContext(
        t=t,
        objects=tuple(objects),
        hands=tuple(hands),
        supports=(TABLE, COUNTER),
        frame_size=(800, 600),
        zone_id="kitchen",
        attributes={"zone_label": "kitchen"},
    )


def _answer(result: Any) -> dict[str, Any]:
    anchor = result.terminal_anchor or {}
    return {
        "answer": result.fallback_answer(),
        "entity": result.entity_id,
        "state": getattr(result.state, "value", str(result.state)),
        "confidence": round(float(result.confidence), 3),
        "anchor": {
            "id": anchor.get("anchor_id"),
            "label": anchor.get("label"),
            "t_start": anchor.get("t_start"),
        },
    }


def run_demo(db_path: Path | str) -> dict[str, Any]:
    """Run the synthetic scene and return a JSON-serialisable result summary."""

    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    transitions = 0
    with ChronicleMemory(path, load=False) as memory:
        for index in range(round(SCENE_SECONDS / DT)):
            t = index * DT
            bbox, hand = _scene(t)
            objects = [ObjectObservation(entity_id="cup", label="cup", bbox=bbox)]
            hands = [hand] if hand is not None else []
            transitions += len(memory.observe(_frame(t, objects, hands)))
            frames += 1

        stats = memory.stats()
        queries = {}
        for question in QUESTIONS:
            result = memory.query(question)
            queries[question] = (
                _answer(result)
                if result is not None
                else {"answer": "I don't have that in my memory.", "entity": None}
            )

    # Reopen from disk: this is the important claim behind "compile once".
    with ChronicleMemory(path, load=True) as restored:
        restored_result = restored.query("where is my cup?")
        restored_answer = (
            restored_result.fallback_answer() if restored_result is not None else None
        )
        restored_stats = restored.stats()

    return {
        "db": str(path),
        "frames": frames,
        "transitions": transitions,
        "seconds": round(SCENE_SECONDS, 1),
        "stats": stats,
        "queries": queries,
        "restored_answer": restored_answer,
        "restored_entities": restored_stats.get("entities"),
    }


def _print_report(result: dict[str, Any], *, temporary: bool) -> None:
    stats = result["stats"]
    states = stats.get("states", {})
    print("Chronicle model-free demo")
    print("No video, no models, no GPU.")
    print(
        f"Scene: {result['seconds']:.1f} s synthetic pick-and-place, "
        f"{result['frames']} frames at {FPS} fps."
    )
    print(f"SQLite: {result['db']}\n")

    print("Ingest")
    print(f"  lifecycle transitions: {result['transitions']}")
    print(f"  entities: {stats.get('entities', 0)}  observations: {stats.get('observations', 0)}")
    print("  states: " + ", ".join(f"{key}={value}" for key, value in sorted(states.items())))
    print(
        "  foreground graph: "
        f"{stats.get('edges', 0)} relation edges across "
        f"{stats.get('zones', 0)} zone(s)"
    )

    print("\nQuery")
    for question, answer in result["queries"].items():
        print(f"  Q: {question}")
        print(f"  A: {answer['answer']}")
        if answer.get("anchor", {}).get("label"):
            anchor = answer["anchor"]
            print(
                f"     evidence: {anchor['label']} at t={anchor.get('t_start')}s "
                f"({answer['state']})"
            )

    print("\nPersistence")
    print(f"  reopened SQLite -> {result['restored_answer']}")
    if temporary:
        print(
            "\nThe demo database was temporary. "
            "Keep one with `chronicle demo --db runs/demo.db`."
        )
    else:
        print(
            "\nNext: chronicle ask \"where is my cup?\" "
            f"--db {result['db']} --llm template"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the model-free Chronicle demo (no model weights required)."
    )
    parser.add_argument("--db", help="persist the demo memory to this SQLite path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite --db if it already exists",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args(argv)

    temporary = args.db is None
    if temporary:
        temp_dir = Path(tempfile.mkdtemp(prefix="chronicle-demo-"))
        db_path = temp_dir / "chronicle.db"
    else:
        db_path = Path(args.db).expanduser().resolve()
        if db_path.exists() and not args.force:
            parser.error(f"{db_path} already exists; pass --force to overwrite it")
        if args.force:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(db_path) + suffix)
                if candidate.exists():
                    candidate.unlink()

    try:
        result = run_demo(db_path)
        if args.json:
            print(json.dumps({**result, "temporary": temporary}, indent=2))
        else:
            _print_report(result, temporary=temporary)
        return 0
    finally:
        if temporary:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
