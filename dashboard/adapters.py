"""Live ingestion adapters for Role D.

Supported transports (all optional, selected by environment variables):

* ``CHRONICLE_EVENTS_JSONL`` — tail a newline-delimited JSON event log.
* ``CHRONICLE_DB``           — poll a SQLite database written by Role B.
* ``POST /api/ingest``       — push events from any process (always enabled).

Both adapters are read-only with respect to teammate artifacts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .runtime import Runtime

# --------------------------------------------------------------------------- #
# JSONL tail
# --------------------------------------------------------------------------- #


async def tail_jsonl(runtime: Runtime, path: Path, *, interval: float = 0.35) -> None:
    """Tail ``path`` and feed complete JSON lines into the runtime."""
    path = Path(path)
    offset = 0
    buffer = ""
    from_start = os.getenv("CHRONICLE_JSONL_FROM_START", "1").strip() not in {"0", "false", "no"}
    announced = False
    while True:
        try:
            if not path.exists():
                await asyncio.sleep(0.6)
                continue
            size = path.stat().st_size
            if not announced and not from_start:
                offset = size
            announced = True
            if size < offset:
                offset = 0
                buffer = ""
            if size == offset:
                await asyncio.sleep(interval)
                continue
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            buffer += chunk.decode("utf-8", errors="replace")
            *lines, buffer = buffer.split("\n")
            for line in lines:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    runtime.ingest_errors.append(f"jsonl: {exc}")
                    continue
                runtime.ingest(payload, origin=f"jsonl:{path.name}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never kill the dashboard
            runtime.ingest_errors.append(f"jsonl: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)
        await asyncio.sleep(interval)


# --------------------------------------------------------------------------- #
# SQLite polling
# --------------------------------------------------------------------------- #

#: Best-effort column aliases per logical table.  Role B can expose views with
#: these names, or simply use the canonical names documented in docs/INTEGRATION.md.
_TABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "entities": ("entities", "entity", "objects", "foreground_nodes", "nodes"),
    "edges": ("edges", "relations", "graph_edges", "foreground_edges"),
    "zones": ("zones", "places", "background_nodes", "anchors"),
    "state_history": ("state_history", "state_transitions", "transitions", "states"),
    "interactions": ("interactions", "actions", "vlm_actions"),
    "telemetry": ("telemetry", "logs", "events"),
    "metrics": ("metrics", "pipeline_metrics"),
    "session": ("session", "episode", "metadata"),
}

_ENTITY_COLUMNS = {
    "id": ("id", "entity_id", "track_id", "node_id"),
    "label": ("label", "name", "class_name", "class"),
    "cls": ("class", "cls", "category"),
    "state": ("state", "current_state", "lifecycle_state"),
    "bbox": ("bbox", "bbox_json", "box", "bbox_norm"),
    "confidence": ("confidence", "score", "det_conf"),
    "clip_sim": ("clip_sim", "similarity", "clip_similarity", "reid_sim"),
    "crop_url": ("crop_url", "crop", "crop_path", "thumbnail_url"),
    "zone_id": ("zone_id", "zone", "anchor_id"),
    "track_id": ("track_id", "bytetrack_id"),
    "first_seen": ("first_seen", "t_first", "first_seen_t"),
    "last_seen": ("last_seen", "t_last", "last_seen_t", "updated_at"),
    "state_since": ("state_since", "state_t"),
    "attributes": ("attributes", "meta", "attributes_json"),
}

_EDGE_COLUMNS = {
    "id": ("id", "edge_id"),
    "source": ("source", "subject", "from_entity", "src"),
    "target": ("target", "object", "to_entity", "dst"),
    "relation": ("relation", "predicate", "type", "label"),
    "t_start": ("t_start", "start_t", "t0", "created_at"),
    "t_end": ("t_end", "end_t", "t1"),
    "active": ("active", "is_active"),
    "clip_sim": ("clip_sim", "similarity", "confidence"),
    "meta": ("meta", "attributes", "data_json"),
}

_STATE_COLUMNS = {
    "id": ("id", "rowid"),
    "entity_id": ("entity_id", "object_id", "track_id"),
    "from_state": ("from_state", "from", "old_state", "previous_state"),
    "to_state": ("to_state", "to", "new_state", "state"),
    "t_event": ("t_event", "t", "timestamp", "time", "created_at"),
    "reason": ("reason", "trigger", "description"),
    "data": ("data", "trigger", "evidence", "data_json"),
    "confidence": ("confidence",),
}

_TELEMETRY_COLUMNS = {
    "id": ("id", "rowid"),
    "t_event": ("t_event", "t", "timestamp", "time", "created_at"),
    "channel": ("channel", "tag", "category"),
    "level": ("level", "severity"),
    "message": ("message", "msg", "text"),
    "entity_id": ("entity_id", "object_id"),
    "data": ("data", "meta", "data_json"),
}

_INTERACTION_COLUMNS = {
    "id": ("id", "rowid"),
    "entity_id": ("entity_id", "object_id", "track_id"),
    "t_event": ("t_event", "t", "timestamp", "time"),
    "action": ("action", "label", "verb", "action_type"),
    "caption": ("caption", "description", "text"),
    "clip_sim": ("clip_sim", "similarity", "score", "confidence"),
    "strip_url": ("strip_url", "composite_url", "image_url"),
    "crop_url": ("crop_url", "crop", "thumbnail_url"),
    "model": ("model", "vlm", "model_name"),
    "latency_ms": ("latency_ms", "latency"),
}

_ZONE_COLUMNS = {
    "id": ("id", "zone_id", "place_id", "node_id"),
    "label": ("label", "name", "zone_name"),
    "clip_sim": ("clip_sim", "similarity", "score"),
    "keyframe_url": ("keyframe_url", "keyframe", "image_url", "crop_url"),
    "bbox": ("bbox", "bbox_json", "box"),
    "first_seen": ("first_seen", "t_first"),
    "last_seen": ("last_seen", "t_last", "updated_at"),
    "confidence": ("confidence", "score"),
    "attributes": ("attributes", "meta"),
}


class _SQLiteMirror:
    def __init__(self, runtime: Runtime, path: Path) -> None:
        self.runtime = runtime
        self.path = path
        self.tables: dict[str, str] = {}
        self.columns: dict[str, set[str]] = {}
        self.marks: dict[str, int] = {}
        self._discovered = False

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        return connection

    def discover(self) -> bool:
        if self._discovered:
            return True
        if not self.path.exists():
            return False
        try:
            with self._connect() as connection:
                names = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                    )
                ]
                lower = {name.lower(): name for name in names}
                for logical, aliases in _TABLE_ALIASES.items():
                    for alias in aliases:
                        if alias in lower:
                            self.tables[logical] = lower[alias]
                            break
                for logical, table in self.tables.items():
                    self.columns[logical] = {
                        row[1].lower() for row in connection.execute(f'PRAGMA table_info("{table}")')
                    }
                # Role B may store session as key/value rows.
                self._discovered = True
                return True
        except sqlite3.Error as exc:
            self.runtime.ingest_errors.append(f"sqlite discover: {exc}")
            return False

    @staticmethod
    def _value(row: sqlite3.Row, mapping: dict[str, tuple[str, ...]], key: str) -> Any:
        for candidate in mapping.get(key, ()):
            if candidate in row.keys() and row[candidate] is not None:
                return row[candidate]
        return None

    @staticmethod
    def _json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    def _select_new(self, connection: sqlite3.Connection, logical: str) -> list[sqlite3.Row]:
        table = self.tables.get(logical)
        if not table:
            return []
        columns = self.columns.get(logical, set())
        if "rowid" not in columns:
            # WITHOUT ROWID tables still expose it unless explicitly omitted.
            pass
        mark = self.marks.get(logical, 0)
        try:
            cursor = connection.execute(
                f'SELECT rowid AS _chronicle_rowid, * FROM "{table}" WHERE rowid > ? ORDER BY rowid LIMIT 400',
                (mark,),
            )
            return cursor.fetchall()
        except sqlite3.Error:
            try:
                cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT 400')
                return cursor.fetchall()
            except sqlite3.Error as exc:
                self.runtime.ingest_errors.append(f"sqlite read {logical}: {exc}")
                return []

    def poll(self) -> int:
        if not self.discover():
            return 0
        accepted = 0
        try:
            with self._connect() as connection:
                accepted += self._poll_zones(connection)
                accepted += self._poll_entities(connection)
                accepted += self._poll_edges(connection)
                accepted += self._poll_states(connection)
                accepted += self._poll_interactions(connection)
                accepted += self._poll_telemetry(connection)
                accepted += self._poll_metrics(connection)
                accepted += self._poll_session(connection)
        except sqlite3.Error as exc:
            self.runtime.ingest_errors.append(f"sqlite poll: {exc}")
        return accepted

    def _advance(self, logical: str, rows: Iterable[sqlite3.Row]) -> None:
        last = self.marks.get(logical, 0)
        for row in rows:
            value = row["_chronicle_rowid"] if "_chronicle_rowid" in row.keys() else 0
            last = max(last, int(value or 0))
        self.marks[logical] = last

    def _poll_entities(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "entities")
        accepted = 0
        has_state_table = "state_history" in self.tables
        for row in rows:
            data: dict[str, Any] = {}
            entity_id = self._value(row, _ENTITY_COLUMNS, "id")
            if entity_id is None:
                continue
            data["id"] = str(entity_id)
            for key in ("label", "cls", "clip_sim", "crop_url", "zone_id", "track_id", "confidence"):
                value = self._value(row, _ENTITY_COLUMNS, key)
                if value is not None:
                    data[key] = self._json(value) if key == "attributes" else value
            if not has_state_table:
                state = self._value(row, _ENTITY_COLUMNS, "state")
                if state is not None:
                    data["state"] = state
            bbox = self._value(row, _ENTITY_COLUMNS, "bbox")
            if bbox is not None:
                data["bbox"] = self._json(bbox)
            for key in ("first_seen", "last_seen"):
                value = self._value(row, _ENTITY_COLUMNS, key)
                if value is not None:
                    data[key] = value
            attributes = self._value(row, _ENTITY_COLUMNS, "attributes")
            parsed = self._json(attributes)
            if isinstance(parsed, dict):
                data["attributes"] = parsed
            t = data.get("last_seen")
            for candidate in ("last_seen", "first_seen"):
                if data.get(candidate) is not None:
                    t = data[candidate]
                    break
            if self.runtime.ingest({"kind": "entity", "t": t or 0, "data": data}, origin=self._origin()):
                accepted += 1
        self._advance("entities", rows)
        return accepted

    def _poll_states(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "state_history")
        accepted = 0
        for row in rows:
            entity_id = self._value(row, _STATE_COLUMNS, "entity_id")
            if entity_id is None:
                continue
            payload: dict[str, Any] = {
                "entity_id": str(entity_id),
                "from": self._value(row, _STATE_COLUMNS, "from_state"),
                "to": self._value(row, _STATE_COLUMNS, "to_state"),
                "reason": self._value(row, _STATE_COLUMNS, "reason"),
            }
            trigger = self._json(self._value(row, _STATE_COLUMNS, "data"))
            if isinstance(trigger, dict):
                payload["trigger"] = trigger
            confidence = self._value(row, _STATE_COLUMNS, "confidence")
            if confidence is not None:
                payload["confidence"] = confidence
            t = self._value(row, _STATE_COLUMNS, "t_event") or 0
            if self.runtime.ingest({"kind": "state", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("state_history", rows)
        return accepted

    def _poll_edges(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "edges")
        accepted = 0
        for row in rows:
            payload: dict[str, Any] = {}
            for key in ("id", "source", "target", "relation", "t_start", "t_end", "active", "clip_sim"):
                value = self._value(row, _EDGE_COLUMNS, key)
                if value is not None:
                    payload[key] = value
            meta = self._json(self._value(row, _EDGE_COLUMNS, "meta"))
            if isinstance(meta, dict):
                payload["meta"] = meta
            t = payload.get("t_start") or payload.get("t_end") or 0
            if self.runtime.ingest({"kind": "edge", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("edges", rows)
        return accepted

    def _poll_zones(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "zones")
        accepted = 0
        for row in rows:
            payload: dict[str, Any] = {}
            for key in ("id", "label", "clip_sim", "keyframe_url", "first_seen", "last_seen", "confidence"):
                value = self._value(row, _ZONE_COLUMNS, key)
                if value is not None:
                    payload[key] = value
            bbox = self._json(self._value(row, _ZONE_COLUMNS, "bbox"))
            if bbox is not None:
                payload["bbox"] = bbox
            attributes = self._json(self._value(row, _ZONE_COLUMNS, "attributes"))
            if isinstance(attributes, dict):
                payload["attributes"] = attributes
            t = payload.get("last_seen") or payload.get("first_seen") or 0
            if self.runtime.ingest({"kind": "zone", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("zones", rows)
        return accepted

    def _poll_interactions(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "interactions")
        accepted = 0
        for row in rows:
            payload: dict[str, Any] = {}
            for key in (
                "entity_id",
                "action",
                "caption",
                "clip_sim",
                "strip_url",
                "crop_url",
                "model",
                "latency_ms",
            ):
                value = self._value(row, _INTERACTION_COLUMNS, key)
                if value is not None:
                    payload[key] = value
            t = self._value(row, _INTERACTION_COLUMNS, "t_event") or 0
            if self.runtime.ingest({"kind": "interaction", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("interactions", rows)
        return accepted

    def _poll_telemetry(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "telemetry")
        accepted = 0
        for row in rows:
            payload: dict[str, Any] = {}
            for key in ("channel", "level", "message", "entity_id"):
                value = self._value(row, _TELEMETRY_COLUMNS, key)
                if value is not None:
                    payload[key] = value
            data = self._json(self._value(row, _TELEMETRY_COLUMNS, "data"))
            if isinstance(data, dict):
                payload.update({k: v for k, v in data.items() if k not in payload})
            t = self._value(row, _TELEMETRY_COLUMNS, "t_event") or 0
            if self.runtime.ingest({"kind": "telemetry", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("telemetry", rows)
        return accepted

    def _poll_metrics(self, connection: sqlite3.Connection) -> int:
        rows = self._select_new(connection, "metrics")
        accepted = 0
        columns = self.columns.get("metrics", set())
        for row in rows:
            t = 0
            payload: dict[str, Any] = {}
            if "data" in columns or "payload" in columns:
                raw = row["data"] if "data" in columns else row["payload"]
                parsed = self._json(raw)
                if isinstance(parsed, dict):
                    payload.update(parsed)
            for key in row.keys():
                if key == "_chronicle_rowid":
                    continue
                if key in {"data", "payload"}:
                    continue
                if key in {"t_event", "t", "timestamp", "time"}:
                    t = row[key] or t
                else:
                    payload.setdefault(key, row[key])
            if self.runtime.ingest({"kind": "metrics", "t": t, "data": payload}, origin=self._origin()):
                accepted += 1
        self._advance("metrics", rows)
        return accepted

    def _poll_session(self, connection: sqlite3.Connection) -> int:
        table = self.tables.get("session")
        if not table:
            return 0
        columns = self.columns.get("session", set())
        payload: dict[str, Any] = {}
        try:
            if {"key", "value"} <= columns:
                for row in connection.execute(f'SELECT key, value FROM "{table}"'):
                    value = self._json(row["value"])
                    payload[str(row["key"])] = value
            else:
                cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT 1')
                row = cursor.fetchone()
                if row is not None:
                    for key in row.keys():
                        payload[str(key)] = self._json(row[key])
        except sqlite3.Error:
            return 0
        if not payload:
            return 0
        # Session rows are idempotent; publish only once per unique payload.
        signature = json.dumps(payload, sort_keys=True, default=str)
        if getattr(self, "_session_signature", None) == signature:
            return 0
        self._session_signature = signature
        t = payload.get("t", payload.get("t_event", 0)) or 0
        accepted = 1 if self.runtime.ingest({"kind": "session", "t": t, "data": payload}, origin=self._origin()) else 0
        return accepted

    def _origin(self) -> str:
        return f"sqlite:{self.path.name}"


async def poll_sqlite(runtime: Runtime, path: Path, *, interval: float = 0.5) -> None:
    path = Path(path)
    mirror = _SQLiteMirror(runtime, path)
    # Give Role B a moment to create the DB on first boot.
    for _ in range(20):
        if path.exists():
            break
        await asyncio.sleep(0.5)
    while True:
        try:
            await asyncio.to_thread(mirror.poll)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime.ingest_errors.append(f"sqlite: {type(exc).__name__}: {exc}")
        await asyncio.sleep(interval)
