"""Model-free tests for the zone-clustered ECharts graph payload."""

from __future__ import annotations

import math

from chronicle.ui.graph_view import (
    ACTIVE_ZONE_BORDER,
    graph_options,
    timeline_options,
)


def _snapshot() -> dict:
    return {
        "stats": {"active_zone": "zone:kitchen"},
        "nodes": [
            {"id": "zone:kitchen", "label": "Kitchen", "group": "zone"},
            {"id": "zone:living-room", "label": "Living Room", "group": "zone"},
            {
                "id": "entity:cup",
                "label": "cup",
                "group": "entity",
                "state": "VISIBLE_STATIC",
                "zone_id": "kitchen",
            },
            {
                "id": "entity:keys",
                "label": "keys",
                "group": "entity",
                "state": "HELD",
                "zone_id": "zone:living-room",
            },
            {
                "id": "anchor:table",
                "label": "table",
                "group": "furniture",
                "zone_id": "zone:kitchen",
            },
            {"id": "agent:user", "label": "USER", "group": "agent"},
        ],
        "edges": [
            {
                "from": "anchor:table",
                "to": "zone:kitchen",
                "label": "IN_ZONE",
                "arrows": "to",
            },
            {
                "from": "entity:cup",
                "to": "anchor:table",
                "label": "RESTING_ON",
                "arrows": "to",
            },
        ],
    }


def test_zone_nodes_are_fixed_and_share_a_cluster_with_their_members() -> None:
    options = graph_options(_snapshot())
    series = options["series"][0]
    by_id = {node["id"]: node for node in series["data"]}

    kitchen = by_id["zone:kitchen"]
    living = by_id["zone:living-room"]
    assert kitchen["fixed"] is True
    assert living["fixed"] is True
    assert (kitchen["x"], kitchen["y"]) != (living["x"], living["y"])

    cup = by_id["entity:cup"]
    keys = by_id["entity:keys"]
    assert cup["zoneId"] == "zone:kitchen"
    assert keys["zoneId"] == "zone:living-room"

    kitchen_distance = math.dist((cup["x"], cup["y"]), (kitchen["x"], kitchen["y"]))
    living_distance = math.dist((cup["x"], cup["y"]), (living["x"], living["y"]))
    assert kitchen_distance < living_distance

    assert cup["itemStyle"]["borderColor"] == kitchen["itemStyle"]["color"]
    assert keys["itemStyle"]["borderColor"] == living["itemStyle"]["color"]
    assert kitchen["itemStyle"]["borderColor"] == ACTIVE_ZONE_BORDER
    assert living["itemStyle"]["borderColor"] != ACTIVE_ZONE_BORDER

    assert cup["tooltip"]["formatter"].endswith("zone Kitchen")


def test_invisible_zone_springs_are_added_once_per_member() -> None:
    series = graph_options(_snapshot())["series"][0]
    springs = [
        link
        for link in series["links"]
        if link.get("lineStyle", {}).get("opacity") == 0.0
    ]
    pairs = {(link["source"], link["target"]) for link in springs}
    assert ("entity:cup", "zone:kitchen") in pairs
    assert ("entity:keys", "zone:living-room") in pairs
    # The table already has a real IN_ZONE edge; do not duplicate it.
    assert ("anchor:table", "zone:kitchen") not in pairs


def test_empty_and_unzoned_snapshots_stay_renderable() -> None:
    empty = graph_options({"nodes": [], "edges": []})
    assert empty["series"][0]["data"] == []
    assert empty["series"][0]["links"] == []

    unzoned = graph_options(
        {
            "nodes": [{"id": "entity:cup", "label": "cup", "group": "entity"}],
            "edges": [],
        }
    )
    item = unzoned["series"][0]["data"][0]
    assert "zoneId" not in item
    assert "fixed" not in item


def test_timeline_options_keep_state_order_and_colours() -> None:
    options = timeline_options(
        [
            {
                "timestamp": 1.25,
                "to_state": "HELD",
                "label": "cup",
            }
        ]
    )
    point = options["series"][0]["data"][0]
    assert point["value"][0] == 1.25
    assert point["value"][2] == "cup"
    assert point["itemStyle"]["color"] == "#f97316"
