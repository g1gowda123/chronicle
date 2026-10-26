"""Pure ECharts option builders for the Chronicle dashboard.

Keeping these functions free of NiceGUI/FastAPI imports means the graph and
timeline payloads can be unit-tested on CI machines without the ``ui`` extra.
"""

from __future__ import annotations

import math
from typing import Any

STATE_ORDER = [
    "NEW",
    "VISIBLE_STATIC",
    "INTERACTED",
    "VISIBLE_MOVING",
    "HELD",
    "CONTAINED",
    "OCCLUDED",
    "OUT_OF_VIEW",
    "LOST",
]
STATE_COLORS = {
    "NEW": "#94a3b8",
    "VISIBLE_STATIC": "#22c55e",
    "INTERACTED": "#f59e0b",
    "VISIBLE_MOVING": "#38bdf8",
    "HELD": "#f97316",
    "CONTAINED": "#a855f7",
    "OCCLUDED": "#64748b",
    "OUT_OF_VIEW": "#475569",
    "LOST": "#ef4444",
}
GROUP_COLORS = {
    "entity": "#38bdf8",
    "furniture": "#f59e0b",
    "zone": "#2dd4bf",
    "agent": "#e879f9",
}
# Stable palette so the same room keeps the same accent across refreshes.
ZONE_PALETTE = (
    "#2dd4bf",
    "#60a5fa",
    "#c084fc",
    "#f472b6",
    "#facc15",
    "#4ade80",
    "#fb923c",
    "#22d3ee",
)
ACTIVE_ZONE_BORDER = "#f8fafc"


def _zone_id(node: dict[str, Any], known_zones: set[str]) -> str | None:
    """Resolve a node's zone to the namespaced ``zone:<id>`` graph node."""

    raw = node.get("zone_id") or node.get("zone")
    if raw in (None, ""):
        return None
    raw = str(raw)
    if raw in known_zones:
        return raw
    prefixed = f"zone:{raw}"
    if prefixed in known_zones:
        return prefixed
    return None


# ECharts graph coordinates are not automatically centred; explicit seed
# positions near (0, 0) are clipped by the card's top-left corner.  Seed the
# ring around the middle of the graph panel instead.
GRAPH_ORIGIN = (360.0, 190.0)


def _zone_centers(zone_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Place zone anchors on a fixed ring so force layout forms room clusters."""

    count = len(zone_ids)
    if not count:
        return {}
    origin_x, origin_y = GRAPH_ORIGIN
    if count == 1:
        return {zone_ids[0]: (origin_x, origin_y)}
    if count == 2:
        return {
            zone_ids[0]: (origin_x - 130.0, origin_y),
            zone_ids[1]: (origin_x + 130.0, origin_y),
        }
    radius = min(190.0, 110.0 + 25.0 * count)
    start = -math.pi / 2
    return {
        zone: (
            round(origin_x + radius * math.cos(start + 2 * math.pi * index / count), 2),
            round(origin_y + radius * math.sin(start + 2 * math.pi * index / count), 2),
        )
        for index, zone in enumerate(zone_ids)
    }


def _member_position(
    center: tuple[float, float], rank: int, *, phase: float = 0.0
) -> tuple[float, float]:
    """Seed a member node just outside its zone anchor to help force settle."""

    ring = 70.0 + 36.0 * (rank // 9)
    angle = phase + 2 * math.pi * (rank % 9) / 9
    return (
        round(center[0] + ring * math.cos(angle), 2),
        round(center[1] + ring * math.sin(angle), 2),
    )


def graph_options(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a zone-clustered ECharts force graph from a runtime snapshot."""

    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])
    categories = [
        {"name": "entity"},
        {"name": "furniture"},
        {"name": "zone"},
        {"name": "agent"},
    ]
    category_index = {category["name"]: index for index, category in enumerate(categories)}

    zone_nodes = {
        str(node["id"]): node
        for node in nodes
        if str(node.get("group") or "") == "zone" and node.get("id")
    }
    zone_ids = sorted(
        zone_nodes,
        key=lambda zone_id: (
            str(zone_nodes[zone_id].get("label", zone_id)).lower(),
            zone_id,
        ),
    )
    zone_colors = {
        zone_id: ZONE_PALETTE[index % len(ZONE_PALETTE)]
        for index, zone_id in enumerate(zone_ids)
    }
    zone_labels = {
        zone_id: str(zone_nodes[zone_id].get("label", zone_id)) for zone_id in zone_ids
    }
    centers = _zone_centers(zone_ids)
    active_zone = str((snapshot.get("stats") or {}).get("active_zone") or "")

    resolved_zones: dict[str, str | None] = {
        str(node.get("id")): _zone_id(node, set(zone_ids))
        for node in nodes
        if node.get("id")
    }
    for zone_id in zone_ids:
        resolved_zones[zone_id] = zone_id
    members: dict[str, list[str]] = {zone_id: [] for zone_id in zone_ids}
    for node in nodes:
        node_id = str(node.get("id") or "")
        zone_id = resolved_zones.get(node_id)
        if node_id and zone_id in members and zone_id not in {node_id}:
            members[zone_id].append(node_id)

    data: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_type = str(node.get("group") or "entity")
        state = str(node.get("state") or "")
        zone_id = node_id if node_type == "zone" else resolved_zones.get(node_id)
        zone_name = zone_labels.get(zone_id or "", "")
        zone_color = zone_colors.get(zone_id or "")
        label = str(node.get("label", node_id))
        color = STATE_COLORS.get(state) if node_type == "entity" else GROUP_COLORS.get(node_type)

        detail = label
        if state:
            detail += f" · {state}"
        appearance = node.get("attributes") or {}
        appearance_bits = [
            str(appearance.get(key))
            for key in ("colour", "shape", "size")
            if appearance.get(key)
        ]
        if appearance_bits:
            detail += f" · {' · '.join(appearance_bits)}"
        if zone_name:
            detail += f" · zone {zone_name}"

        if node_type == "furniture":
            symbol = "roundRect"
            size: Any = [46, 24]
            item_style: dict[str, Any] = {
                "color": color or "#f59e0b",
                "borderColor": "#fbbf24",
                "borderWidth": 1.5,
            }
            if zone_color:
                item_style["shadowColor"] = zone_color
                item_style["shadowBlur"] = 10
            label_cfg = {"show": True, "fontSize": 10, "color": "#fef08a"}
        elif node_type == "zone":
            symbol = "diamond"
            size = 52 if node_id == active_zone else 44
            item_style = {
                "color": zone_color or color or "#2dd4bf",
                "borderColor": ACTIVE_ZONE_BORDER if node_id == active_zone else "#0f766e",
                "borderWidth": 2.5 if node_id == active_zone else 1.2,
                "shadowColor": zone_color or "#2dd4bf",
                "shadowBlur": 20,
            }
            label_cfg = {
                "show": True,
                "fontSize": 12,
                "fontWeight": "bold",
                "color": "#e2fdf7",
                "position": "bottom",
                "backgroundColor": "rgba(2,6,13,.72)",
                "padding": [2, 5],
                "borderRadius": 4,
            }
        elif node_type == "agent":
            symbol = "pin"
            size = 34
            item_style = {
                "color": color or "#e879f9",
                "borderColor": "#f5d0fe",
                "borderWidth": 1.2,
            }
            label_cfg = {"show": True, "fontSize": 11, "color": "#f5d0fe"}
        else:
            symbol = "circle"
            size = 34
            item_style = {"color": color or "#94a3b8"}
            if zone_color:
                item_style["borderColor"] = zone_color
                item_style["borderWidth"] = 2.2
            # A detected dominant colour is a stronger identity cue than the
            # room border, so use it as the ring when we have one.
            colour_hex = node.get("colour_hex")
            if colour_hex:
                item_style["borderColor"] = str(colour_hex)
                item_style["borderWidth"] = 2.8
            label_cfg = {"show": True, "fontSize": 11, "color": "#e2e8f0"}

        entry: dict[str, Any] = {
            "id": node_id,
            "name": label,
            "category": category_index.get(node_type, 0),
            "symbol": symbol,
            "symbolSize": size,
            "itemStyle": item_style,
            "label": label_cfg,
            "tooltip": {"show": True, "formatter": detail},
        }
        if node_type == "zone":
            center = centers.get(node_id, (0.0, 0.0))
            entry["x"] = center[0]
            entry["y"] = center[1]
            entry["fixed"] = True
            entry["zoneId"] = node_id
            entry["zoneName"] = label
        elif zone_id:
            center = centers.get(zone_id, (0.0, 0.0))
            rank = members.get(zone_id, []).index(node_id) if node_id in members.get(
                zone_id, []
            ) else 0
            x, y = _member_position(center, rank, phase=0.25 * zone_ids.index(zone_id))
            entry["x"] = x
            entry["y"] = y
            entry["zoneId"] = zone_id
            entry["zoneName"] = zone_name
        elif node_type != "agent":
            # Unmapped nodes live on the outside so they do not blur room clusters.
            angle = 2 * math.pi * len(data) / max(len(nodes), 1)
            entry["x"] = round(480.0 * math.cos(angle), 2)
            entry["y"] = round(480.0 * math.sin(angle), 2)
        data.append(entry)

    links: list[dict[str, Any]] = []
    linked_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if not source or not target:
            continue
        label = str(edge.get("label", edge.get("kind", "")))
        # State transitions are rendered on the Timeline tab.  Their self-loops
        # would add 100+ invisible edges to the force simulation and make the
        # graph cluster collapse off-canvas.
        if source == target and label == "STATE_TRANSITION":
            continue
        linked_pairs.add((source, target))
        link: dict[str, Any] = {
            "source": source,
            "target": target,
            "value": label,
        }
        if label == "IN_ZONE":
            zone_id = target if target in zone_colors else source
            link["lineStyle"] = {
                "color": zone_colors.get(zone_id, "#475569"),
                "width": 1.6,
                "opacity": 0.75,
            }
        links.append(link)

    # Force uses links as springs. Entities have a zone_id but no IN_ZONE edge;
    # add an invisible spring so each item stays in its room cluster.
    for node_id, zone_id in resolved_zones.items():
        if not node_id or not zone_id or zone_id == node_id:
            continue
        if (node_id, zone_id) in linked_pairs or (zone_id, node_id) in linked_pairs:
            continue
        links.append(
            {
                "source": node_id,
                "target": zone_id,
                "value": "",
                "lineStyle": {"opacity": 0.0, "width": 0.5},
                "label": {"show": False},
                "tooltip": {"show": False},
                "silent": True,
            }
        )

    return {
        "backgroundColor": "transparent",
        "tooltip": {"formatter": "{b}"},
        "legend": [
            {
                "data": [category["name"] for category in categories],
                "textStyle": {"color": "#94a3b8"},
                "top": 0,
            }
        ],
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "categories": categories,
                "data": data,
                "links": links,
                "force": {
                    "repulsion": 160,
                    "edgeLength": [45, 130],
                    "gravity": 0.12,
                    "friction": 0.2,
                    "layoutAnimation": True,
                },
                "label": {
                    "show": True,
                    "position": "right",
                    "color": "#e2e8f0",
                    "fontSize": 11,
                },
                "lineStyle": {"color": "#475569", "curveness": 0.12, "opacity": 0.8},
                "edgeLabel": {
                    "show": True,
                    "formatter": "{c}",
                    "fontSize": 9,
                    "color": "#94a3b8",
                },
                "emphasis": {"focus": "adjacency", "lineStyle": {"width": 2}},
            }
        ],
    }


def timeline_options(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the state-transition scatter used by the Timeline tab."""

    points: list[dict[str, Any]] = []
    for row in reversed(transitions):
        state = str(row.get("to_state") or "NEW")
        y = STATE_ORDER.index(state) if state in STATE_ORDER else 0
        points.append(
            {
                "value": [
                    round(float(row.get("timestamp") or 0.0), 2),
                    y,
                    row.get("label") or "",
                ],
                "itemStyle": {"color": STATE_COLORS.get(state, "#94a3b8")},
            }
        )
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 90, "right": 24, "top": 24, "bottom": 40},
        "tooltip": {"formatter": "t={@[0]}s<br/>{@[2]} -> {b}"},
        "xAxis": {
            "type": "value",
            "name": "seconds",
            "axisLabel": {"color": "#94a3b8"},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "yAxis": {
            "type": "category",
            "data": STATE_ORDER,
            "axisLabel": {"color": "#94a3b8", "fontSize": 10},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "series": [
            {
                "type": "scatter",
                "symbolSize": 9,
                "data": points,
                "emphasis": {"scale": 1.4},
            }
        ],
    }


__all__ = [
    "ACTIVE_ZONE_BORDER",
    "GROUP_COLORS",
    "STATE_COLORS",
    "STATE_ORDER",
    "ZONE_PALETTE",
    "graph_options",
    "timeline_options",
]
