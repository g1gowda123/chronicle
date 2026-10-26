"""Canonical node identifiers.

Raw entity / anchor / zone ids are namespaced so a detector label can never
collide with a room zone inside the same NetworkX graph or Vis.js canvas.
"""

from __future__ import annotations

ENTITY_PREFIX = "entity:"
ANCHOR_PREFIX = "anchor:"
ZONE_PREFIX = "zone:"
AGENT_NODE = "agent:user"


def _prefix(value: str, prefix: str) -> str:
    value = str(value).strip()
    return value if value.startswith(prefix) else f"{prefix}{value}"


def entity_node(entity_id: str) -> str:
    return _prefix(entity_id, ENTITY_PREFIX)


def anchor_node(anchor_id: str) -> str:
    return _prefix(anchor_id, ANCHOR_PREFIX)


def zone_node(zone_id: str) -> str:
    return _prefix(zone_id, ZONE_PREFIX)


def strip_prefix(node_id: str, prefix: str) -> str:
    return node_id.removeprefix(prefix)


def raw_entity_id(node_id: str) -> str:
    return strip_prefix(node_id, ENTITY_PREFIX)


def raw_anchor_id(node_id: str) -> str:
    return strip_prefix(node_id, ANCHOR_PREFIX)


def raw_zone_id(node_id: str) -> str:
    return strip_prefix(node_id, ZONE_PREFIX)
