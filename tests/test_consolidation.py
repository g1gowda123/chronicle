"""Legacy duplicate consolidation: safety gates, retention and audit."""

from __future__ import annotations

from pathlib import Path

from conftest import DT, cup, feed, make_frame

from chronicle.config import FurnitureConfig
from chronicle.geometry import bbox_to_polygon
from chronicle.graph.memory import ChronicleMemory
from chronicle.telemetry import TelemetryTag
from chronicle.types import ObjectObservation, Relation, SupportSurface

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


def test_merge_furniture_promotes_graph_only_keeper_with_namespaced_id() -> None:
    memory = offline_memory()
    try:
        zone = memory.enter_zone("living-room", 1.0)
        # Legacy graph-only keeper: present in the background graph but never
        # mirrored into SQLite.
        memory.background.ensure_furniture(
            "sofa-1",
            "sofa",
            zone,
            1.0,
            bbox=SOFA_BBOX,
            attrs={"first_seen": 1.0, "last_seen": 2.0},
        )
        seed_sofa(memory, "sofa-2", 10.0, 11.0)

        report = memory.merge_furniture("sofa-1", "sofa-2", t=11.5)
        assert report.merged is True

        rows = memory.store.load_furniture()
        assert len(rows) == 1
        assert rows[0].id == "anchor:sofa-1"
        assert memory.store.get_furniture("sofa-1") is not None
    finally:
        memory.close()


def test_auto_sweep_merges_pan_reentry_furniture_duplicate() -> None:
    memory = ChronicleMemory(
        ":memory:",
        furniture_config=FurnitureConfig(consolidation_enabled=False),
    )
    try:
        # One physical sofa: visible before the pan, then re-entering the
        # frustum ten seconds later.  An older association pass minted a second
        # anchor for the re-entry; the sweep must fold it back into the keeper.
        seed_sofa(memory, "sofa-1", 1.0, 2.0)
        seed_sofa(memory, "sofa-2", 10.0, 11.0)

        candidates = [
            candidate
            for candidate in memory.consolidator.find_candidates(11.5)
            if candidate.kind == "furniture"
        ]
        assert len(candidates) == 1
        assert candidates[0].keep == "sofa-1"
        assert candidates[0].remove == "sofa-2"

        reports = memory.consolidator.maybe_consolidate(11.5, force=True)
        merged = [
            report
            for report in reports
            if report.kind == "furniture" and report.merged
        ]
        assert len(merged) == 1
        assert merged[0].keep == "sofa-1"
        assert merged[0].remove == "sofa-2"
        assert memory.resolve_furniture_alias("sofa-2") == "sofa-1"
        assert [record.anchor_id for record in memory.furniture.records] == ["sofa-1"]
        assert [row["anchor_id"] for row in memory.background.furniture()] == [
            "anchor:sofa-1"
        ]
        assert not memory.background.has_node("anchor:sofa-2")
        assert memory.stats()["furniture_merged"] == 1
    finally:
        memory.close()


def test_furniture_alias_survives_reopen_and_blocks_reminting(tmp_path: Path) -> None:
    db_path = tmp_path / "chronicle.db"
    memory = ChronicleMemory(
        db_path,
        furniture_config=FurnitureConfig(consolidation_enabled=False),
    )
    try:
        seed_sofa(memory, "sofa-1", 1.0, 2.0)
        seed_sofa(memory, "sofa-2", 10.0, 11.0)
        assert memory.merge_furniture("sofa-1", "sofa-2", t=11.5).merged is True
    finally:
        memory.close()

    reopened = ChronicleMemory(
        db_path,
        furniture_config=FurnitureConfig(consolidation_enabled=False),
    )
    try:
        assert reopened.resolve_furniture_alias("anchor:sofa-2") == "sofa-1"
        assert [record.anchor_id for record in reopened.furniture.records] == ["sofa-1"]
        assert [row["anchor_id"] for row in reopened.background.furniture()] == [
            "anchor:sofa-1"
        ]
        assert reopened.store.get_furniture("sofa-2") is None
        assert reopened.store.get_furniture("sofa-1") is not None

        # A stale Role A support carrying the retired anchor id must be folded
        # onto the keeper, not used to mint a second background node.
        old_support = SupportSurface(
            anchor_id="sofa-2",
            label="sofa",
            polygon=bbox_to_polygon(SOFA_BBOX),
            zone_id="living-room",
        )
        reopened.observe(
            make_frame(
                12.0,
                supports=[old_support],
                zone_id="living-room",
                zone_label="living room",
            )
        )
        assert reopened.background.has_node("anchor:sofa-1")
        assert not reopened.background.has_node("anchor:sofa-2")
        assert [record.anchor_id for record in reopened.furniture.records] == ["sofa-1"]

        # The pan/re-entry detection reuses the keeper through geometry.
        feed(
            reopened,
            12.1,
            12.2,
            DT,
            lambda t: [
                ObjectObservation(
                    entity_id="sofa-track",
                    label="sofa",
                    bbox=SOFA_BBOX,
                    track_id=9,
                )
            ],
            supports_fn=lambda t: (),
            zone_id="living-room",
        )
        assert [row["anchor_id"] for row in reopened.background.furniture()] == [
            "anchor:sofa-1"
        ]
        assert [record.anchor_id for record in reopened.furniture.records] == ["sofa-1"]
        assert reopened.store.get_furniture("sofa-2") is None
        assert reopened.resolve_furniture_alias("sofa-2") == "sofa-1"
    finally:
        reopened.close()


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


def test_dual_furniture_consolidation_sweep_and_sqlite_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "chronicle.db"
    config = FurnitureConfig(
        consolidation_enabled=True,
        consolidation_interval_s=10.0,
        min_hits=1,
    )
    memory = ChronicleMemory(db_path, furniture_config=config)
    try:
        # 1. Two distinct co-visible sofas in living-room: sofa-left and sofa-right
        seed_sofa(memory, "sofa-left", 1.0, 5.0, bbox=(50.0, 100.0, 250.0, 300.0))
        seed_sofa(memory, "sofa-right", 1.0, 5.0, bbox=(550.0, 100.0, 750.0, 300.0))

        # 2. Add an object resting on sofa-left
        from chronicle.types import ObjectState
        memory.foreground.upsert_entity(
            "cushion-1",
            "cushion",
            state=ObjectState.VISIBLE_STATIC,
            state_since=5.0,
            first_seen=5.0,
            last_seen=5.0,
            zone_id="living-room",
        )
        memory.store.upsert_entity(
            "cushion-1",
            "cushion",
            "VISIBLE_STATIC",
            5.0,
            5.0,
            5.0,
            zone_id="living-room",
        )
        memory.foreground.open_resting("cushion-1", "anchor:sofa-left", 5.0)
        memory.store.add_anchor(
            "cushion-1",
            "anchor:sofa-left",
            Relation.RESTING_ON.value,
            5.0,
        )

        # 3. Later observation: re-entry after pan mints duplicate anchor for sofa-left
        seed_sofa(memory, "sofa-left-reentry", 50.0, 55.0, bbox=(52.0, 101.0, 248.0, 299.0))

        # Run periodic consolidation sweep at t=60
        reports = memory.maybe_consolidate(60.0)
        merged = [r for r in reports if r.merged and r.kind == "furniture"]

        assert len(merged) == 1
        assert merged[0].keep == "sofa-left"
        assert merged[0].remove == "sofa-left-reentry"
        # sofa-left and sofa-right must remain separate
        assert memory.background.has_node("anchor:sofa-left")
        assert memory.background.has_node("anchor:sofa-right")
        assert not memory.background.has_node("anchor:sofa-left-reentry")
        assert memory.resolve_furniture_alias("sofa-left-reentry") == "sofa-left"
        assert memory.stats()["furniture_merged"] == 1
    finally:
        memory.close()

    # 4. Reopen SQLite database and verify persistence & alias retention
    reopened = ChronicleMemory(db_path, furniture_config=config)
    try:
        assert reopened.resolve_furniture_alias("sofa-left-reentry") == "sofa-left"
        assert reopened.background.has_node("anchor:sofa-left")
        assert reopened.background.has_node("anchor:sofa-right")
        assert not reopened.background.has_node("anchor:sofa-left-reentry")
        assert {r.anchor_id for r in reopened.furniture.records} == {"sofa-left", "sofa-right"}

        # Observe frame with support referencing merged alias: must canonicalize to sofa-left
        from shapely.geometry import Polygon

        from chronicle.types import SupportSurface
        surf = SupportSurface(
            anchor_id="sofa-left-reentry",
            label="sofa",
            polygon=Polygon([(50.0, 100.0), (250.0, 100.0), (250.0, 300.0), (50.0, 300.0)]),
        )
        reopened.observe(make_frame(70.0, supports=(surf,)))
        # No new anchor created; both sofas still intact
        assert not reopened.background.has_node("anchor:sofa-left-reentry")
        assert reopened.background.has_node("anchor:sofa-left")
        assert reopened.background.has_node("anchor:sofa-right")
        assert len(reopened.furniture.records) == 2
    finally:
        reopened.close()

