"""Legacy duplicate consolidation: safety gates, retention and audit."""

from __future__ import annotations

from pathlib import Path

from conftest import DT, cup, feed, make_frame

from chronicle.config import FurnitureConfig
from chronicle.graph.memory import ChronicleMemory
from chronicle.telemetry import TelemetryTag
from chronicle.types import ObjectObservation, Relation

SOFA_BBOX = (100.0, 100.0, 300.0, 250.0)


def offline_memory() -> ChronicleMemory:
    """Memory with periodic consolidation disabled for deterministic setup."""

    return ChronicleMemory(
        ":memory:",
        furniture_config=FurnitureConfig(consolidation_enabled=False),
    )


def add_duplicate_cups(memory: ChronicleMemory) -> None:
    feed(memory, 0.0, 1.0, DT, lambda t: [cup("cup-a")])
    feed(memory, 5.0, 6.0, DT, lambda t: [cup("cup-b")])


def wallet(entity_id: str, bbox: tuple[float, float, float, float]) -> ObjectObservation:
    return ObjectObservation(entity_id=entity_id, label="wallet", bbox=bbox)


def seed_sofa(
    memory: ChronicleMemory,
    anchor_id: str,
    first: float,
    last: float,
    *,
    bbox: tuple[float, float, float, float] = SOFA_BBOX,
) -> str:
    zone = memory.enter_zone("living-room", first)
    attrs = {
        "bbox": list(bbox),
        "embedding": [1.0, 0.0, 0.0],
        "first_seen": first,
        "last_seen": last,
        "confirmed": True,
        "hits": 2,
        "track_ids": [7],
    }
    node = memory.background.ensure_furniture(
        anchor_id,
        "sofa",
        zone,
        first,
        bbox=bbox,
        embedding=attrs["embedding"],
        attrs=attrs,
    )
    memory.background.update_furniture(node, t=last)
    memory.store.upsert_furniture(node, "sofa", zone, first, attrs=attrs)
    memory.furniture.load_attributes([(anchor_id, "sofa", "living-room", attrs)])
    return node


def test_manual_entity_merge_preserves_all_history() -> None:
    memory = offline_memory()
    try:
        add_duplicate_cups(memory)
        memory.record_action("cup-b", "lift", 5.2, confidence=0.91, clip_ref="clip-1")

        before = {
            "transitions": len(memory.store.load_transitions("cup-a"))
            + len(memory.store.load_transitions("cup-b")),
            "anchors": len(memory.store.anchors_for("cup-a"))
            + len(memory.store.anchors_for("cup-b")),
            "observations": len(memory.store.observations_for("cup-a", limit=1_000_000))
            + len(memory.store.observations_for("cup-b", limit=1_000_000)),
        }

        candidate = next(
            item
            for item in memory.consolidator.find_candidates(6.0)
            if item.kind == "entity"
        )
        assert candidate.keep == "cup-a"
        assert candidate.remove == "cup-b"
        assert candidate.score > 0.9

        report = memory.merge_entities("cup-a", "cup-b", t=6.0)

        assert report.merged is True
        assert report.status == "merged"
        assert memory.store.get_entity("cup-b") is None
        assert memory.store.get_entity("cup-a") is not None
        assert len(memory.store.load_transitions("cup-a")) == before["transitions"]
        assert len(memory.store.anchors_for("cup-a")) == before["anchors"]
        assert (
            len(memory.store.observations_for("cup-a", limit=1_000_000))
            == before["observations"]
        )
        assert report.preserved["transitions"] == before["transitions"]
        assert report.preserved["anchors"] == before["anchors"]
        assert report.preserved["observations"] == before["observations"]

        anchor = memory.store.terminal_anchor("cup-a")
        assert anchor is not None
        assert anchor.retained is True
        assert anchor.is_terminal is True
        assert anchor.anchor_id == "anchor:dining_table"

        actions = [
            edge
            for edge in memory.store.load_edges()
            if edge.relation == Relation.ACTION.value and edge.dst == "entity:cup-a"
        ]
        assert len(actions) == 1
        assert actions[0].props["action"] == "lift"
        assert actions[0].props["clip_ref"] == "clip-1"

        assert memory.foreground.has_entity("cup-a")
        assert not memory.foreground.has_entity("cup-b")
        assert memory.resolve_entity_alias("cup-b") == "cup-a"
        result = memory.query_entity("cup-b")
        assert result is not None
        assert result.entity_id == "cup-a"
        assert result.state == "VISIBLE_STATIC"

        event = next(
            event
            for event in memory.bus.history(TelemetryTag.GRAPH)
            if event.message == "entity merged"
        )
        assert event.format().startswith("[GRAPH]")
        assert event.data["keep"] == "cup-a"
        assert event.data["removed"] == "cup-b"
        assert event.data["transitions"] == before["transitions"]
        assert event.data["anchors"] == before["anchors"]
        assert memory.stats()["entities_merged"] == 1
    finally:
        memory.close()


def test_merge_dry_run_is_a_noop() -> None:
    memory = offline_memory()
    try:
        add_duplicate_cups(memory)
        report = memory.merge_entities("cup-a", "cup-b", dry_run=True)
        assert report.merged is False
        assert report.status == "would_merge"
        assert memory.foreground.has_entity("cup-a")
        assert memory.foreground.has_entity("cup-b")
        assert memory.store.get_entity("cup-b") is not None
        assert memory.resolve_entity_alias("cup-b") == "cup-b"
        assert memory.stats()["entities_merged"] == 0
        assert any(
            event.message == "entity merge planned"
            for event in memory.bus.history(TelemetryTag.GRAPH)
        )
        assert not any(
            event.message == "entity merged"
            for event in memory.bus.history(TelemetryTag.GRAPH)
        )
    finally:
        memory.close()


def test_covisible_entities_are_never_merged() -> None:
    memory = offline_memory()
    try:
        feed(
            memory,
            0.0,
            1.0,
            DT,
            lambda t: [
                cup("cup-a", bbox=(100.0, 300.0, 180.0, 380.0)),
                cup("cup-b", bbox=(500.0, 300.0, 580.0, 380.0)),
            ],
        )
        assert memory.consolidator.find_candidates(1.0) == []

        report = memory.merge_entities("cup-a", "cup-b", t=1.0)
        assert report.merged is False
        assert report.error is not None
        assert "co-visible" in report.error
        assert memory.foreground.has_entity("cup-a")
        assert memory.foreground.has_entity("cup-b")

        # The manual override exists for operators who have independently
        # confirmed that two tracks are one object.
        forced = memory.consolidator.merge_entities(
            "cup-a", "cup-b", t=1.0, allow_covisible=True
        )
        assert forced.merged is True
    finally:
        memory.close()


def test_alias_labels_share_one_consolidation_group() -> None:
    memory = offline_memory()
    try:
        feed(
            memory,
            0.0,
            1.0,
            DT,
            lambda t: [ObjectObservation(entity_id="wallet-a", label="wallet", bbox=(100.0, 300.0, 180.0, 380.0))],
        )
        feed(
            memory,
            5.0,
            6.0,
            DT,
            lambda t: [ObjectObservation(entity_id="purse-b", label="purse", bbox=(100.0, 300.0, 180.0, 380.0))],
        )

        candidate = next(
            item
            for item in memory.consolidator.find_candidates(6.0)
            if item.kind == "entity"
        )
        assert {candidate.keep, candidate.remove} == {"wallet-a", "purse-b"}

        report = memory.merge_entities("wallet-a", "purse-b", t=6.0)
        assert report.merged is True
        result = memory.query("where is my purse?")
        assert result is not None
        assert result.entity_id == "wallet-a"
        assert result.match_reason == "label alias"
    finally:
        memory.close()


def test_auto_sweep_merges_sequential_duplicates() -> None:
    memory = ChronicleMemory(":memory:")
    try:
        add_duplicate_cups(memory)
        # The first sweep ran at t=0 and found nothing.  Advance past the
        # configured interval so the facade runs a real sweep.
        reports = memory.maybe_consolidate(20.0)
        merged = [report for report in reports if report.merged]
        assert len(merged) == 1
        assert merged[0].kind == "entity"
        assert merged[0].keep == "cup-a"
        assert memory.stats()["entities_merged"] == 1
        assert memory.stats()["merges"]["sweeps"] >= 1
        assert any(
            event.message == "consolidation sweep"
            for event in memory.bus.history(TelemetryTag.GRAPH)
        )
    finally:
        memory.close()


def test_auto_sweep_keeps_covisible_sofas_separate() -> None:
    memory = ChronicleMemory(":memory:")
    try:
        seed_sofa(memory, "sofa-left", 1.0, 5.0, bbox=(100.0, 100.0, 200.0, 200.0))
        seed_sofa(memory, "sofa-right", 1.0, 5.0, bbox=(500.0, 100.0, 600.0, 200.0))

        assert memory.consolidator.find_candidates(5.0) == []
        reports = memory.consolidator.maybe_consolidate(20.0, force=True)
        assert all(report.merged is False for report in reports)
        assert {record.anchor_id for record in memory.furniture.records} == {
            "sofa-left",
            "sofa-right",
        }
        assert memory.background.has_node("anchor:sofa-left")
        assert memory.background.has_node("anchor:sofa-right")

        report = memory.merge_furniture("sofa-left", "sofa-right", t=5.0)
        assert report.merged is False
        assert report.error is not None
        assert "co-visible" in report.error
    finally:
        memory.close()


def test_merge_furniture_preserves_registry_and_aliases() -> None:
    memory = offline_memory()
    try:
        seed_sofa(memory, "sofa-1", 1.0, 2.0)
        seed_sofa(memory, "sofa-2", 100.0, 101.0)

        report = memory.merge_furniture("sofa-1", "sofa-2", t=101.0)
        assert report.merged is True
        assert report.status == "merged"
        assert report.preserved["edges"] >= 1
        assert memory.resolve_furniture_alias("sofa-2") == "sofa-1"
        assert memory.store.get_furniture("sofa-1") is not None
        assert memory.store.get_furniture("sofa-2") is None
        assert not memory.background.has_node("anchor:sofa-2")
        assert memory.background.has_node("anchor:sofa-1")
        assert [record.anchor_id for record in memory.furniture.records] == ["sofa-1"]
        assert memory.stats()["furniture_merged"] == 1

        merged = memory.store.get_furniture("sofa-1")
        assert merged is not None
        assert "sofa-2" in merged.attrs["merged_from"]

        event = next(
            event
            for event in memory.bus.history(TelemetryTag.GRAPH)
            if event.message == "furniture merged"
        )
        assert event.data["keep"] == "anchor:sofa-1"
        assert event.data["removed"] == "anchor:sofa-2"
    finally:
        memory.close()


def test_aliases_survive_reopen_and_block_recreation(tmp_path: Path) -> None:
    db_path = tmp_path / "chronicle.db"
    memory = ChronicleMemory(
        db_path,
        furniture_config=FurnitureConfig(consolidation_enabled=False),
    )
    try:
        add_duplicate_cups(memory)
        assert memory.merge_entities("cup-a", "cup-b", t=6.0).merged is True
    finally:
        memory.close()

    reopened = ChronicleMemory(db_path)
    try:
        assert reopened.resolve_entity_alias("cup-b") == "cup-a"
        result = reopened.query_entity("cup-b")
        assert result is not None
        assert result.entity_id == "cup-a"

        # Role A may still emit the retired track id once; the facade must fold
        # it back onto the keeper instead of minting a second node.
        reopened.observe(make_frame(6.0, objects=[cup("cup-b")]))
        assert [record["node_id"] for record in reopened.foreground.entities()] == [
            "entity:cup-a"
        ]
    finally:
        reopened.close()


def test_query_aliases_and_confidence_disambiguate_duplicates() -> None:
    memory = offline_memory()
    try:
        feed(
            memory,
            0.0,
            1.0,
            DT,
            lambda t: [wallet("wallet-a", (100.0, 300.0, 180.0, 380.0))],
        )
        feed(
            memory,
            5.0,
            6.0,
            DT,
            lambda t: [wallet("wallet-b", (500.0, 300.0, 580.0, 380.0))],
        )

        result = memory.query("where is my purse?")
        assert result is not None
        assert result.entity_id == "wallet-b"
        assert result.match_reason == "label alias"
        assert result.match_score >= 0.8
        assert 0.0 < result.confidence <= result.match_score
        assert len(result.candidates) == 2
        assert result.candidates[0]["entity_id"] == "wallet-b"
        assert result.candidates[0]["visible"] is True
        assert result.candidates[1]["entity_id"] == "wallet-a"
    finally:
        memory.close()
