"""Compact storage primitives for the VFM vault.

Three ideas borrowed from model compression:

* **Dictionary encoding** — repeated labels/zones/relations become uint16 ids.
* **Packed arrays** — events become a struct-of-arrays blob (~16 bytes/event),
  not one SQLite row per event.
* **Int8 quantization** — embeddings keep one float32 scale plus int8 values
  (4x smaller than float32, ~2x smaller than float16 with a scale).

Nothing here depends on numpy; `array`/`struct`/`zlib` are enough.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

MAGIC = b"VFM2"
_VERSION = 1

# Event kinds are a fixed enum: no string storage per event.
KIND_TO_ID = {
    "state": 1,
    "anchor": 2,
    "relation": 3,
    "zone": 4,
    "keyframe": 5,
    "interaction": 6,
}
ID_TO_KIND = {value: key for key, value in KIND_TO_ID.items()}

#: t_ms, kind, subject_id, relation_id, object_id, zone_id, conf_q, flags, evidence_id
HEADER = struct.Struct("<4sII")
EVENT_STRUCT = struct.Struct("<IBHHHHBBH")
EVENT_SIZE = EVENT_STRUCT.size


class StringTable:
    """Bidirectional string <-> uint16 id table."""

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {"": 0}
        self._to_value: list[str] = [""]

    def add(self, value: str | None) -> int:
        text = str(value or "")
        existing = self._to_id.get(text)
        if existing is not None:
            return existing
        if len(self._to_value) >= 65535:
            raise ValueError("string table exhausted (65535 entries)")
        index = len(self._to_value)
        self._to_id[text] = index
        self._to_value.append(text)
        return index

    def get(self, index: int | None) -> str:
        try:
            return self._to_value[int(index or 0)]
        except (IndexError, TypeError, ValueError):
            return ""

    def rows(self) -> list[tuple[int, str]]:
        return list(enumerate(self._to_value))

    def load(self, rows: Iterable[tuple[int, str]]) -> None:
        self._to_id = {}
        self._to_value = []
        for index, value in rows:
            index = int(index)
            while len(self._to_value) <= index:
                self._to_value.append("")
            self._to_value[index] = str(value)
        self._to_id = {value: index for index, value in enumerate(self._to_value)}


def pack_events(events: Iterable[dict[str, Any]], table: StringTable) -> bytes:
    """Pack events into a struct-of-arrays blob. Returns ~16 bytes/event."""

    rows = list(events)
    packed = bytearray(HEADER.pack(MAGIC, len(rows), _VERSION))
    for event in rows:
        t_ms = max(0, min(0xFFFFFFFF, int(round(float(event.get("t") or 0.0) * 1000.0))))
        kind = KIND_TO_ID.get(str(event.get("kind") or ""), 0)
        subject = table.add(event.get("subject"))
        relation = table.add(event.get("relation"))
        obj = table.add(event.get("object"))
        zone = table.add(event.get("zone"))
        confidence = max(0, min(100, int(round(float(event.get("confidence") or 0.0) * 100.0))))
        flags = int(event.get("flags") or 0) & 0xFF
        evidence = int(event.get("evidence_id") or 0) & 0xFFFF
        packed += EVENT_STRUCT.pack(
            t_ms, kind, subject, relation, obj, zone, confidence, flags, evidence
        )
    return bytes(packed)


def unpack_events(blob: bytes, table: StringTable) -> list[dict[str, Any]]:
    """Unpack a struct-of-arrays blob into event dicts."""

    if not blob:
        return []
    magic, count, version = HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a VFM2 event blob: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported VFM event version: {version}")
    offset = HEADER.size
    events: list[dict[str, Any]] = []
    for _ in range(int(count)):
        (
            t_ms,
            kind,
            subject,
            relation,
            obj,
            zone,
            confidence,
            flags,
            evidence,
        ) = EVENT_STRUCT.unpack_from(blob, offset)
        offset += EVENT_SIZE
        events.append(
            {
                "t": t_ms / 1000.0,
                "kind": ID_TO_KIND.get(kind, "unknown"),
                "subject": table.get(subject),
                "relation": table.get(relation),
                "object": table.get(obj),
                "zone": table.get(zone),
                "confidence": confidence / 100.0,
                "flags": flags,
                "evidence_id": evidence,
            }
        )
    return events


def encode_events(events: Iterable[dict[str, Any]], table: StringTable, mode: str = "packed") -> bytes:
    """Encode an event ledger with the requested storage layout.

    ``packed`` is the production layout (~16 B/event).  ``zlib-json`` and
    ``json`` exist so the storage ablation can measure the packed layout
    against the obvious baselines without changing the query reader.
    """

    rows = list(events)
    if mode == "packed":
        return pack_events(rows, table)
    import json

    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if mode == "zlib-json":
        return zlib.compress(raw, 9)
    if mode == "json":
        return raw
    raise ValueError(f"unknown event encoding: {mode!r}")


def decode_events(blob: bytes, table: StringTable) -> list[dict[str, Any]]:
    """Decode any layout produced by :func:`encode_events`.

    The leading ``VFM2`` magic distinguishes the packed struct layout from the
    JSON baselines, so a vault remains self-describing.
    """

    if not blob:
        return []
    if blob[:4] == MAGIC:
        return unpack_events(blob, table)
    import json

    try:
        raw = zlib.decompress(blob)
    except zlib.error:
        raw = blob
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON event blob must be a list")
    return payload


def compress_json(payload: Any) -> bytes:
    """zlib-compress a JSON-serialisable payload."""

    import json

    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return zlib.compress(raw, 9)


def decompress_json(blob: bytes | None) -> Any:
    if not blob:
        return None
    import json

    return json.loads(zlib.decompress(blob).decode("utf-8"))


@dataclass(frozen=True, slots=True)
class QuantizedVector:
    scale: float
    data: bytes

    def to_blob(self) -> bytes:
        return struct.pack("<f", self.scale) + self.data


def quantize_int8(values: Iterable[float]) -> QuantizedVector:
    """Symmetric int8 quantization with one float32 scale.

    512-d float32 = 2048 B -> 516 B (3.97x smaller). Dequantisation error is
    bounded by scale/2, which is well below the retrieval tolerance of CLIP.
    """

    vector = [float(value) for value in values]
    if not vector:
        return QuantizedVector(0.0, b"")
    peak = max(abs(value) for value in vector)
    scale = peak / 127.0 if peak > 0 else 0.0
    if scale <= 0:
        return QuantizedVector(0.0, bytes(len(vector)))
    quantized = bytes(max(-127, min(127, int(round(value / scale)))) & 0xFF for value in vector)
    return QuantizedVector(scale, quantized)


def dequantize_int8(blob: bytes) -> list[float]:
    if len(blob) < 4:
        return []
    scale = struct.unpack_from("<f", blob, 0)[0]
    raw = blob[4:]
    return [float((byte if byte < 128 else byte - 256) * scale) for byte in raw]


def compression_ratio(raw_size: int, stored_size: int) -> float:
    return round(raw_size / stored_size, 2) if stored_size else 0.0
