"""Question parsing for people and ordinary objects."""

from chronicle.perception.spore import extract_terms


def test_person_location_is_queryable() -> None:
    assert extract_terms("Where is the person?") == ["person"]


def test_people_location_is_canonicalized() -> None:
    assert extract_terms("Where are the people?") == ["person"]


def test_person_does_not_replace_an_object_term() -> None:
    assert extract_terms("Where is the cup?") == ["cup"]
