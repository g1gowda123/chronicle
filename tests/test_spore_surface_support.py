from chronicle.perception.surfaces import choose_support


def test_wall_mounted_object_gets_wall() -> None:
    assert choose_support("tv", "shelf", {"wall": 0.62, "shelf": 0.20}) == "wall"


def test_furniture_on_desk_gets_floor() -> None:
    assert choose_support("sofa", "desk", {"floor": 0.71}) == "floor"
    assert choose_support("chair", "desk", {"floor": 0.10}) == "floor"


def test_tabletop_object_on_desk_gets_table() -> None:
    assert choose_support("plate", "desk", {"table": 0.58}) == "table"


def test_specific_horizontal_support_is_not_replaced() -> None:
    assert choose_support("toaster", "countertop", {"shelf": 0.92}) is None
    assert choose_support("switch console", "work table", {"floor": 0.92}) is None


def test_weak_wall_fraction_is_ignored() -> None:
    assert choose_support("tv", "shelf", {"wall": 0.10}) is None


def test_empty_support_can_take_strong_surface() -> None:
    assert choose_support("plate", None, {"table": 0.60}) == "table"


def test_outdoor_object_can_take_road() -> None:
    assert choose_support("car", None, {"road": 0.75}) == "road"
    assert choose_support("person", None, {"road": 0.58}) == "road"
    assert choose_support("person", None, {"road": 0.20}) is None
