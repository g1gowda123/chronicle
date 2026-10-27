"""Entity matching and temporal wording for caption-first VFM memories."""

from __future__ import annotations

from chronicle.vfm import query
from chronicle.vfm.query import EventStore


def _entity(label: str, first: float, last: float | None = None) -> dict:
    slug = label.replace(" ", "-")
    return {
        "id": "obj:" + slug,
        "label": label,
        "state": "SEEN",
        "zone": "outdoors",
        "first_seen": first,
        "last_seen": last if last is not None else first,
    }


def _store() -> EventStore:
    man = _entity("man wearing red hat", 96.0)
    bus = _entity("bus parked near the curb", 705.0)
    event = {
        "t": 96.0,
        "kind": "state",
        "relation": "SEEN",
        "subject": "man wearing red hat",
        "object": "man wearing red hat",
        "zone": "outdoors",
    }
    return EventStore(
        events=[event],
        entities=[man, bus],
        captions=[],
        con=None,  # type: ignore[arg-type]
        by_subject={"man wearing red hat": [event]},
        ledger_text="a hallway with arched doorways leading outside",
    )


def test_stopwords_do_not_match_an_unrelated_entity() -> None:
    assert query._match_entity(_store(), "where is the city arch") is None


def test_red_hat_still_matches_its_entity() -> None:
    entity = query._match_entity(_store(), "where is the man wearing a red hat")
    assert entity is not None
    assert entity["label"] == "man wearing red hat"


def test_when_appear_uses_first_seen_bounds() -> None:
    store = _store()
    entity = query._match_entity(store, "when did the man wearing red hat appear")
    assert entity is not None
    result = query.answer_when(store, "when did the man wearing red hat appear", entity, "see")
    assert "first seen at t=96.0s" in result.answer


def test_seen_events_satisfy_the_see_action() -> None:
    assert query._matches_action({"kind": "state", "relation": "SEEN"}, "see")


def test_before_first_sighting_does_not_claim_absence() -> None:
    store = _store()
    entity = query._match_entity(store, "where is the man wearing a red hat")
    assert entity is not None
    result = query.answer_where(store, "where is the man wearing a red hat", entity, now=50.0)
    assert "has not appeared yet" not in result.answer
    assert "earliest recorded sighting is at t=96.0s" in result.answer
    assert result.citations and result.citations[0]["t"] == 96.0


def test_arch_matches_arched_but_not_architecture() -> None:
    store = _store()
    assert query._ledger_has(store, "where is the arch")
    store.ledger_text = "modern architecture and glass facades"
    assert not query._ledger_has(store, "where is the arch")


def test_later_sighting_does_not_imply_visible_now() -> None:
    store = _store()
    entity = _entity("man wearing red hat", 96.0, 167.0)
    events = [
        {
            "t": t,
            "kind": "state",
            "relation": "SEEN",
            "subject": entity["label"],
            "object": entity["label"],
            "zone": "outdoors",
        }
        for t in (96.0, 167.0)
    ]
    store.entities = [entity]
    store.events = events
    store.by_subject = {entity["label"]: events}

    near = query.answer_where(store, "where is the man wearing red hat", entity, now=98.0)
    assert "is visible" in near.answer

    far = query.answer_where(store, "where is the man wearing red hat", entity, now=120.0)
    assert "is not visible" in far.answer
    assert "last seen at t=96.0s" in far.answer


def test_fragmented_red_hat_labels_are_merged() -> None:
    person = _entity("person in red hat", 96.0)
    man = _entity("man wearing red hat", 157.0)

    def _event(label: str, t: float) -> dict:
        return {
            "t": t,
            "kind": "state",
            "relation": "SEEN",
            "subject": label,
            "object": label,
            "zone": "outdoors",
        }

    store = _store()
    store.entities = [person, man]
    store.by_subject = {
        "person in red hat": [_event("person in red hat", 96.0)],
        "man wearing red hat": [_event("man wearing red hat", 157.0)],
    }
    store.events = [*store.by_subject["person in red hat"], *store.by_subject["man wearing red hat"]]

    where = query.answer(
        store, "where is the man wearing a red hat", use_cache=False, now=98.3
    )
    assert "is visible" in where.answer

    when = query.answer(
        store, "when did the man wearing a red hat appear", use_cache=False, now=200.0
    )
    assert "first seen at t=96.0s" in when.answer


def test_rare_keyword_beats_common_city_captions() -> None:
    store = _store()
    store.ledger = {
        "windows": [
            {"t_start": 870.0, "t_end": 900.0, "caption": "A blurred city street scene."},
            {
                "t_start": 540.0,
                "t_end": 570.0,
                "caption": "A city hallway.",
                "frames": [{"t": 555.0, "caption": "A hallway with arched doorways."}],
            },
        ]
    }
    result = query.answer(store, "where is the city arch", use_cache=False)
    assert result.citations[0]["t"] == 555.0
