"""Model-free Chronicle demo.

This path exists so a new user can see the memory layer work before installing
torch, YOLO-World, MediaPipe, Florence-2, or the NiceGUI dashboard.  Frames are
generated geometrically and fed straight into :class:`ChronicleMemory`; no video
decoder, model weights, GPU, or network access are involved.

The scene is intentionally small: a cup is carried from the dining table to
the kitchen counter, keys make the reverse trip, a bottle moves to the counter,
and a notebook never moves.  The demo then runs graph queries and reopens the
SQLite database to show that the answer survives a process restart.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
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
SCENE_SECONDS = 10.6
QUESTIONS = (
    "where is my cup?",
    "what happened to my cup?",
    "where are my keys?",
    "where is my bottle?",
)

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


@dataclass(frozen=True)
class _PickAndPlace:
    """One rigid-box pick-and-place with a fixed 2.2 s hand trajectory."""

    entity_id: str
    label: str
    start: tuple[float, float, float, float]
    end: tuple[float, float, float, float]
    grasp_at: float

    def pose(self, t: float) -> tuple[tuple[float, float, float, float], HandObservation | None]:
        x0, y0, x1, y1 = self.start
        ex0, ey0, ex1, ey1 = self.end
        if t < self.grasp_at:
            return self.start, None
        if t < self.grasp_at + 0.4:  # dwell on the object before lifting
            return self.start, _hand_at(x0, y0)
        if t < self.grasp_at + 1.0:  # lift 120 px in lockstep with the hand
            dy = (t - self.grasp_at - 0.4) / 0.6 * 120.0
            return (x0, y0 - dy, x1, y1 - dy), _hand_at(x0, y0 - dy)
        if t < self.grasp_at + 1.6:  # carry horizontally while held
            progress = (t - self.grasp_at - 1.0) / 0.6
            dx = (ex0 - x0) * progress
            dy = (ey0 - y0) * progress
            return (
                x0 + dx,
                y0 - 120.0 + dy,
                x1 + dx,
                y1 - 120.0 + dy,
            ), _hand_at(x0 + dx, y0 - 120.0 + dy)
        if t < self.grasp_at + 2.2:  # lower onto the destination support
            progress = (t - self.grasp_at - 1.6) / 0.6
            dy = 120.0 * (1.0 - progress)
            return (ex0, ey0 - dy, ex1, ey1 - dy), _hand_at(ex0, ey0 - dy)
        return self.end, None


CUP = _PickAndPlace(
    "cup", "cup", (100.0, 300.0, 180.0, 380.0), (500.0, 300.0, 580.0, 380.0), 1.2
)
KEYS = _PickAndPlace(
    "keys", "keys", (600.0, 300.0, 680.0, 380.0), (300.0, 300.0, 380.0, 380.0), 4.4
)
BOTTLE = _PickAndPlace(
    "bottle", "bottle", (200.0, 300.0, 280.0, 380.0), (400.0, 300.0, 480.0, 380.0), 7.6
)
NOTEBOOK = _PickAndPlace(
    "notebook", "notebook", (20.0, 300.0, 100.0, 380.0), (20.0, 300.0, 100.0, 380.0), 99.9
)
MOTIONS = (CUP, KEYS, BOTTLE, NOTEBOOK)

APPEARANCE: dict[str, dict[str, Any]] = {
    "cup": {"colour": "white", "shape": "round", "size": "small"},
    "keys": {"colour": "grey", "shape": "flat", "size": "small"},
    "bottle": {"colour": "blue", "shape": "tall", "size": "medium"},
    "notebook": {"colour": "red", "shape": "rectangular", "size": "medium"},
}


def _scene(t: float) -> tuple[list[ObjectObservation], list[HandObservation]]:
    """All four objects are observed every frame; only the active one moves."""

    objects: list[ObjectObservation] = []
    hands: list[HandObservation] = []
    for motion in MOTIONS:
        bbox, hand = motion.pose(t)
        objects.append(
            ObjectObservation(
                entity_id=motion.entity_id,
                label=motion.label,
                bbox=bbox,
                attributes=APPEARANCE.get(motion.entity_id, {}),
            )
        )
        if hand is not None:
            hands.append(hand)
    return objects, hands


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
            objects, hands = _scene(t)
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
            t_start = anchor.get("t_start")
            t_text = f"{float(t_start):.1f}" if t_start is not None else "unknown"
            print(
                f"     evidence: {anchor['label']} at t={t_text}s "
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
