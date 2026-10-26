"""Validation helpers for importing an exported dashboard snapshot."""

from __future__ import annotations

import json
from typing import Any


class SnapshotImportError(ValueError):
    """Raised when an uploaded file is not a usable Chronicle snapshot."""


def parse_snapshot(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Decode and validate an exported ``chronicle_snapshot.json`` payload."""

    if isinstance(payload, (bytes, bytearray)):
        try:
            text = bytes(payload).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SnapshotImportError("file is not valid UTF-8") from exc
    else:
        text = str(payload).lstrip("\ufeff")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SnapshotImportError(f"invalid JSON at line {exc.lineno}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise SnapshotImportError("snapshot root must be a JSON object")

    missing = [key for key in ("nodes", "entities") if key not in data]
    if missing:
        raise SnapshotImportError(
            "missing " + ", ".join(f'"{key}"' for key in missing) + " (not a Chronicle snapshot)"
        )
    for key in ("nodes", "entities"):
        if not isinstance(data[key], list):
            raise SnapshotImportError(f'"{key}" must be a list')
    return data


def snapshot_summary(payload: dict[str, Any]) -> str:
    """One-line import receipt for the UI notification."""

    entities = len(payload.get("entities") or [])
    actions = len(payload.get("actions") or [])
    zones = sum(
        1
        for node in payload.get("nodes") or []
        if isinstance(node, dict) and node.get("group") == "zone"
    )
    parts = [f"{entities} entities", f"{actions} actions"]
    if zones:
        parts.append(f"{zones} zones")
    return "Imported " + " · ".join(parts)


__all__ = ["SnapshotImportError", "parse_snapshot", "snapshot_summary"]
