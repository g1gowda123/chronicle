"""Persistent indexed storage for the dual graph.

SQLite is used purely as a durable projection of the in-memory NetworkX graph:
every mutation is written through this class inside the ingestion thread, while
the dashboard reads via short, read-only queries (WAL mode + ``busy_timeout``
keeps those readers non-blocking).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from chronicle.storage.records import (
    AnchorRecord,
    EdgeRecord,
    EntityRecord,
    FurnitureRecord,
    StorageSnapshot,
    TransitionRecord,
    ZoneRecord,
)
from chronicle.telemetry import EventBus

SCHEMA_VERSION = "1"
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _maybe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _maybe_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, (list, tuple)) or len(loaded) != 4:
        return None
    return tuple(float(v) for v in loaded)  # type: ignore[return-value]


def _encode_embedding(embedding: Sequence[float] | None) -> bytes | None:
    if embedding is None:
        return None
    return array("f", (float(v) for v in embedding)).tobytes()


def _decode_embedding(blob: bytes | None) -> tuple[float, ...] | None:
    if not blob:
        return None
    values = array("f")
    values.frombytes(blob)
    return tuple(float(v) for v in values)


class SQLiteStore:
    """Thread-safe, WAL-enabled SQLite projection of the episodic graph."""

    def __init__(
        self,
        path: str | Path = "data/chronicle.db",
        *,
        bus: EventBus | None = None,
        initialize: bool = True,
    ) -> None:
        self.path = str(path)
        self.bus = bus or EventBus()
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        if initialize:
            self.initialize()

    # ------------------------------------------------------------------ plumbing
    def initialize(self) -> None:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._lock, self._conn:
            self._conn.executescript(schema)
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (SCHEMA_VERSION,),
            )
        self.bus.store("schema ready", path=self.path, version=SCHEMA_VERSION)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        with self._lock, suppress(sqlite3.ProgrammingError):  # pragma: no cover - double close
            self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ zones
    def upsert_zone(
        self,
        zone_id: str,
        label: str,
        t: float,
        embedding: Sequence[float] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO zones(id, label, embedding, created_at, last_seen, visit_count, attrs)
                VALUES(?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    embedding=COALESCE(excluded.embedding, zones.embedding),
                    last_seen=excluded.last_seen,
                    visit_count=zones.visit_count + 1,
                    attrs=excluded.attrs
                """,
                (
                    zone_id,
                    label,
                    _encode_embedding(embedding),
                    float(t),
                    float(t),
                    json.dumps(attrs or {}, sort_keys=True),
                ),
            )

    def load_zones(self) -> list[ZoneRecord]:
        rows = self._fetchall("SELECT * FROM zones ORDER BY created_at")
        return [
            ZoneRecord(
                id=row["id"],
                label=row["label"],
                embedding=_decode_embedding(row["embedding"]),
                created_at=row["created_at"],
                last_seen=row["last_seen"],
                visit_count=row["visit_count"],
                attrs=_maybe_json(row["attrs"]),
            )
            for row in rows
        ]

    # --------------------------------------------------------------- furniture
    def upsert_furniture(
        self,
        anchor_id: str,
        label: str,
        zone_id: str | None,
        t: float,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT attrs FROM furniture WHERE id = ?", (anchor_id,)
            ).fetchone()
            existing: dict[str, Any] = {}
            if row is not None and row["attrs"]:
                try:
                    loaded = json.loads(row["attrs"])
                except (TypeError, ValueError):
                    loaded = {}
                if isinstance(loaded, dict):
                    existing = loaded
            merged = {**existing, **dict(attrs or {})}
            self._conn.execute(
                """
                INSERT INTO furniture(id, label, zone_id, created_at, attrs)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    zone_id=COALESCE(excluded.zone_id, furniture.zone_id),
                    attrs=excluded.attrs
                """,
                (anchor_id, label, zone_id, float(t), json.dumps(merged, sort_keys=True)),
            )

    def load_furniture(self) -> list[FurnitureRecord]:
        rows = self._fetchall("SELECT * FROM furniture ORDER BY created_at")
        return [
            FurnitureRecord(
                id=row["id"],
                label=row["label"],
                zone_id=row["zone_id"],
                created_at=row["created_at"],
                attrs=_maybe_json(row["attrs"]),
            )
            for row in rows
        ]

    # ---------------------------------------------------------------- entities
    def upsert_entity(
        self,
        entity_id: str,
        label: str,
        state: str,
        state_since: float,
        first_seen: float,
        last_seen: float,
        bbox: Sequence[float] | None = None,
        embedding: Sequence[float] | None = None,
        track_id: int | None = None,
        zone_id: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        bbox_json = json.dumps(list(bbox)) if bbox is not None else None
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO entities(
                    id, label, state, state_since, first_seen, last_seen,
                    bbox, embedding, track_id, zone_id, attrs
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    state=excluded.state,
                    state_since=excluded.state_since,
                    last_seen=excluded.last_seen,
                    bbox=COALESCE(excluded.bbox, entities.bbox),
                    embedding=COALESCE(excluded.embedding, entities.embedding),
                    track_id=COALESCE(excluded.track_id, entities.track_id),
                    zone_id=COALESCE(excluded.zone_id, entities.zone_id),
                    attrs=excluded.attrs
                """,
                (
                    entity_id,
                    label,
                    state,
                    float(state_since),
                    float(first_seen),
                    float(last_seen),
                    bbox_json,
                    _encode_embedding(embedding),
                    track_id,
                    zone_id,
                    json.dumps(attrs or {}, sort_keys=True),
                ),
            )

    def load_entities(self) -> list[EntityRecord]:
        rows = self._fetchall("SELECT * FROM entities ORDER BY first_seen")
        return [
            EntityRecord(
                id=row["id"],
                label=row["label"],
                state=row["state"],
                state_since=row["state_since"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                bbox=_maybe_bbox(row["bbox"]),
                embedding=_decode_embedding(row["embedding"]),
                track_id=row["track_id"],
                zone_id=row["zone_id"],
                attrs=_maybe_json(row["attrs"]),
            )
            for row in rows
        ]

    # -------------------------------------------------------------- transitions
    def add_transition(
        self,
        entity_id: str,
        from_state: str,
        to_state: str,
        t_event: float,
        evidence: dict[str, Any] | None = None,
        clip_ref: str | None = None,
    ) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO state_transitions(
                    entity_id, from_state, to_state, t_event, evidence, clip_ref
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    from_state,
                    to_state,
                    float(t_event),
                    json.dumps(evidence or {}, sort_keys=True, default=str),
                    clip_ref,
                ),
            )
            return int(cursor.lastrowid or 0)

    def load_transitions(self, entity_id: str | None = None) -> list[TransitionRecord]:
        if entity_id:
            rows = self._fetchall(
                "SELECT * FROM state_transitions WHERE entity_id=? ORDER BY t_event, id",
                (entity_id,),
            )
        else:
            rows = self._fetchall("SELECT * FROM state_transitions ORDER BY t_event, id")
        return [
            TransitionRecord(
                id=row["id"],
                entity_id=row["entity_id"],
                from_state=row["from_state"],
                to_state=row["to_state"],
                t_event=row["t_event"],
                evidence=_maybe_json(row["evidence"]),
                clip_ref=row["clip_ref"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------- edges
    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        t_start: float | None,
        t_end: float | None = None,
        *,
        is_terminal: bool = False,
        props: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO edges(src, dst, relation, t_start, t_end, is_terminal, props)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    src,
                    dst,
                    relation,
                    t_start,
                    t_end,
                    int(is_terminal),
                    json.dumps(props or {}, sort_keys=True, default=str),
                ),
            )
            return int(cursor.lastrowid or 0)

    def close_open_edges(self, src: str, relation: str, t_end: float) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE edges SET t_end=?
                WHERE src=? AND relation=? AND t_end IS NULL
                """,
                (float(t_end), src, relation),
            )
            return int(cursor.rowcount or 0)

    def open_edges(self, src: str | None = None, relation: str | None = None) -> list[EdgeRecord]:
        clauses = ["t_end IS NULL"]
        params: list[Any] = []
        if src:
            clauses.append("src=?")
            params.append(src)
        if relation:
            clauses.append("relation=?")
            params.append(relation)
        rows = self._fetchall(
            f"SELECT * FROM edges WHERE {' AND '.join(clauses)} ORDER BY t_start, id",
            params,
        )
        return [self._edge_from_row(row) for row in rows]

    def load_edges(self) -> list[EdgeRecord]:
        rows = self._fetchall("SELECT * FROM edges ORDER BY t_start, id")
        return [self._edge_from_row(row) for row in rows]

    def latest_edge(self, src: str, relation: str) -> EdgeRecord | None:
        row = self._fetchone(
            """
            SELECT * FROM edges WHERE src=? AND relation=?
            ORDER BY t_start DESC, id DESC LIMIT 1
            """,
            (src, relation),
        )
        return self._edge_from_row(row) if row else None

    @staticmethod
    def _edge_from_row(row: sqlite3.Row) -> EdgeRecord:
        return EdgeRecord(
            id=row["id"],
            src=row["src"],
            dst=row["dst"],
            relation=row["relation"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            is_terminal=bool(row["is_terminal"]),
            props=_maybe_json(row["props"]),
        )

    # ----------------------------------------------------------------- anchors
    def add_anchor(
        self,
        entity_id: str,
        anchor_id: str,
        relation: str,
        t_start: float,
        confidence: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> int:
        """Record a placement anchor; previous terminal anchors stay retained."""

        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE anchors SET is_terminal=0
                WHERE entity_id=? AND is_terminal=1
                """,
                (entity_id,),
            )
            # A new placement implies the previous open placement ended.
            self._conn.execute(
                """
                UPDATE anchors SET t_end=COALESCE(t_end, ?)
                WHERE entity_id=? AND t_end IS NULL
                """,
                (float(t_start), entity_id),
            )
            cursor = self._conn.execute(
                """
                INSERT INTO anchors(
                    entity_id, anchor_id, relation, t_start, is_terminal,
                    retained, confidence, evidence
                ) VALUES(?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (
                    entity_id,
                    anchor_id,
                    relation,
                    float(t_start),
                    float(confidence),
                    json.dumps(evidence or {}, sort_keys=True, default=str),
                ),
            )
            return int(cursor.lastrowid or 0)

    def close_open_anchors(self, entity_id: str, relation: str, t_end: float) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE anchors SET t_end=COALESCE(t_end, ?)
                WHERE entity_id=? AND relation=? AND t_end IS NULL
                """,
                (float(t_end), entity_id, relation),
            )
            return int(cursor.rowcount or 0)

    def terminal_anchor(self, entity_id: str) -> AnchorRecord | None:
        row = self._fetchone(
            """
            SELECT * FROM anchors
            WHERE entity_id=? AND is_terminal=1
            ORDER BY t_start DESC LIMIT 1
            """,
            (entity_id,),
        )
        return self._anchor_from_row(row) if row else None

    def anchors_for(self, entity_id: str) -> list[AnchorRecord]:
        rows = self._fetchall(
            "SELECT * FROM anchors WHERE entity_id=? ORDER BY t_start, id",
            (entity_id,),
        )
        return [self._anchor_from_row(row) for row in rows]

    def load_anchors(self) -> list[AnchorRecord]:
        rows = self._fetchall("SELECT * FROM anchors ORDER BY t_start, id")
        return [self._anchor_from_row(row) for row in rows]

    @staticmethod
    def _anchor_from_row(row: sqlite3.Row) -> AnchorRecord:
        return AnchorRecord(
            id=row["id"],
            entity_id=row["entity_id"],
            anchor_id=row["anchor_id"],
            relation=row["relation"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            is_terminal=bool(row["is_terminal"]),
            retained=bool(row["retained"]),
            confidence=row["confidence"],
            evidence=_maybe_json(row["evidence"]),
        )

    # ------------------------------------------------------------ observations
    def add_observation(
        self,
        entity_id: str,
        t: float,
        bbox: Sequence[float] | None,
        state: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO observations(entity_id, t, bbox, state) VALUES(?, ?, ?, ?)",
                (
                    entity_id,
                    float(t),
                    json.dumps(list(bbox)) if bbox is not None else None,
                    state,
                ),
            )

    def observations_for(
        self, entity_id: str, limit: int = 1000
    ) -> list[tuple[float, tuple[float, float, float, float] | None, str | None]]:
        rows = self._fetchall(
            """
            SELECT t, bbox, state FROM observations
            WHERE entity_id=? ORDER BY t DESC LIMIT ?
            """,
            (entity_id, int(limit)),
        )
        return [(row["t"], _maybe_bbox(row["bbox"]), row["state"]) for row in reversed(rows)]

    def purge_trajectories(self, now: float, retention_s: float) -> int:
        """Delete bounding-box trajectories older than the retention window."""

        cutoff = float(now) - float(retention_s)
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM observations WHERE t < ?", (cutoff,))
            deleted = int(cursor.rowcount or 0)
        if deleted:
            self.bus.store(
                "trajectory retention sweep",
                now,
                deleted=deleted,
                retention_s=retention_s,
            )
        return deleted

    # ------------------------------------------------------------------- misc
    def load_snapshot(self) -> StorageSnapshot:
        return StorageSnapshot(
            zones=tuple(self.load_zones()),
            furniture=tuple(self.load_furniture()),
            entities=tuple(self.load_entities()),
            transitions=tuple(self.load_transitions()),
            edges=tuple(self.load_edges()),
            anchors=tuple(self.load_anchors()),
        )

    def stats(self) -> dict[str, int | str]:
        tables = (
            "zones",
            "furniture",
            "entities",
            "state_transitions",
            "edges",
            "anchors",
            "observations",
        )
        result: dict[str, int | str] = {"path": self.path}
        for table in tables:
            row = self._fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            result[table] = int(row["n"]) if row else 0
        return result

    def count(self, table: str) -> int:
        if table not in {
            "zones",
            "furniture",
            "entities",
            "state_transitions",
            "edges",
            "anchors",
            "observations",
        }:
            raise ValueError(f"unknown table {table!r}")
        row = self._fetchone(f"SELECT COUNT(*) AS n FROM {table}")
        return int(row["n"]) if row else 0

    def recent_transitions(self, limit: int = 50) -> list[TransitionRecord]:
        rows = self._fetchall(
            "SELECT * FROM state_transitions ORDER BY t_event DESC, id DESC LIMIT ?",
            (int(limit),),
        )
        return [
            TransitionRecord(
                id=row["id"],
                entity_id=row["entity_id"],
                from_state=row["from_state"],
                to_state=row["to_state"],
                t_event=row["t_event"],
                evidence=_maybe_json(row["evidence"]),
                clip_ref=row["clip_ref"],
            )
            for row in rows
        ]

    def iter_entities(self) -> Iterable[EntityRecord]:
        yield from self.load_entities()
