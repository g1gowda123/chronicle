"""Dual-graph memory (Role B)."""

from chronicle.graph.adapter import frame_context_from_result
from chronicle.graph.background import BackgroundGraph
from chronicle.graph.foreground import ForegroundGraph
from chronicle.graph.ids import (
    AGENT_NODE,
    ANCHOR_PREFIX,
    ENTITY_PREFIX,
    ZONE_PREFIX,
    anchor_node,
    entity_node,
    raw_anchor_id,
    raw_entity_id,
    raw_zone_id,
    zone_node,
)
from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.query import GraphQuery, QueryResult

__all__ = [
    "AGENT_NODE",
    "ANCHOR_PREFIX",
    "ENTITY_PREFIX",
    "ZONE_PREFIX",
    "BackgroundGraph",
    "ChronicleMemory",
    "ForegroundGraph",
    "GraphQuery",
    "QueryResult",
    "anchor_node",
    "entity_node",
    "frame_context_from_result",
    "raw_anchor_id",
    "raw_entity_id",
    "raw_zone_id",
    "zone_node",
]
