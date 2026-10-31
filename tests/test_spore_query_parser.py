from chronicle.perception.spore import extract_terms


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
