from chronicle.perception.spore import (
    Binding,
    _binding_matches_support,
    extract_support_qualifier,
    extract_terms,
)
from chronicle.types import BBox


def test_where_contraction_is_normalised() -> None:
    assert extract_terms("Where's the knife?") == ["knife"]


def test_which_surface_question_extracts_object() -> None:
    assert extract_terms("Which surface is the knife on?") == ["knife"]


def test_where_can_i_find_is_supported() -> None:
    assert extract_terms("Where can I find the cup?") == ["cup"]


def test_polite_where_question_extracts_object() -> None:
    assert extract_terms("Can you tell me where the knife is?") == ["knife"]


def test_located_suffix_is_supported() -> None:
    assert extract_terms("Where is the knife located?") == ["knife"]


def test_colour_of_question_extracts_object() -> None:
    assert extract_terms("What's the colour of the cup?") == ["cup"]
    assert extract_terms("What is the color of the cup?") == ["cup"]


def test_when_was_first_touched_extracts_object() -> None:
    assert extract_terms("When was the cup first touched?") == ["cup"]


def test_where_did_end_up_extracts_object() -> None:
    assert extract_terms("Where did the plate end up?") == ["plate"]


def test_possessive_colour_question_uses_person() -> None:
    assert extract_terms("What colour is the person's shirt?") == ["person"]


def test_first_see_question_extracts_object() -> None:
    assert extract_terms("When did I first see the person?") == ["person"]


def test_which_object_on_surface_extracts_object() -> None:
    assert extract_terms("Which cup is on the shelf?") == ["cup"]
    assert extract_terms("Is the cup on the shelf?") == ["cup"]


def test_support_qualifier_extracts_known_surface() -> None:
    assert extract_support_qualifier("Where is the cup on the shelf?") == "shelf"
    assert extract_support_qualifier("Which cup is on the table?") == "table"
    assert extract_support_qualifier("Where is the cup?") is None
    assert extract_support_qualifier("Where is the cup on the left?") is None


def test_support_qualifier_matches_detector_aliases() -> None:
    binding = Binding(
        term="cup",
        label="cup",
        capsule_id="capsule-1",
        confidence=0.8,
        timestamp=1.0,
        bbox=BBox(0, 0, 10, 10),
        support="countertop",
    )
    assert _binding_matches_support(binding, "table")
    assert not _binding_matches_support(binding, "shelf")
    assert not _binding_matches_support(binding, "unknown")
