"""Geometry primitives: IoS, support contact and velocity estimation."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from chronicle.geometry import (
    bbox_to_polygon,
    bottom_edge_samples,
    center_outside_frame,
    cosine_similarity,
    ema_time_aware,
    ios,
    iou,
    point_inside,
    polyline_velocity,
    support_contact,
    to_polygon,
)


def test_ios_is_one_when_smaller_polygon_is_contained() -> None:
    outer = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    inner = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    assert ios(inner, outer) == pytest.approx(1.0)
    assert ios(outer, inner) == pytest.approx(1.0)


def test_ios_partial_overlap() -> None:
    left = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])
    # intersection is 5x10=50, smaller area is 100
    assert ios(left, right) == pytest.approx(0.5)


def test_ios_disjoint_and_empty_are_zero() -> None:
    left = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right = Polygon([(100, 0), (110, 0), (110, 10), (100, 10)])
    assert ios(left, right) == 0.0
    assert ios(left, Polygon()) == 0.0


def test_iou_differs_from_ios_for_containment() -> None:
    outer = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    inner = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    assert iou(inner, outer) == pytest.approx(100 / 10000)


def test_to_polygon_accepts_bbox_and_ring() -> None:
    assert to_polygon((0, 0, 10, 20)).area == pytest.approx(200)
    ring = [(0, 0), (10, 0), (10, 10)]
    assert to_polygon(ring).area == pytest.approx(50)
    with pytest.raises(ValueError):
        to_polygon(Polygon())


def test_support_contact_edge_and_gap() -> None:
    table = Polygon([(0, 380), (800, 380), (800, 600), (0, 600)])
    resting = bbox_to_polygon((100, 300, 180, 380))
    contact = support_contact(resting, table, "table", tolerance_px=0.0)
    assert contact.supported
    assert contact.edge_overlap == pytest.approx(1.0)
    assert contact.gap_px == pytest.approx(0.0)

    hovering = bbox_to_polygon((100, 250, 180, 330))
    hover = support_contact(hovering, table, "table", tolerance_px=5.0)
    assert not hover.supported
    assert hover.gap_px == pytest.approx(50.0)


def test_support_contact_accepts_fully_enclosed_box() -> None:
    table = Polygon([(0, 380), (800, 380), (800, 600), (0, 600)])
    # a box wholly inside the surface polygon (e.g. detector box covers the table)
    contained = bbox_to_polygon((100, 420, 180, 500))
    contact = support_contact(contained, table, "table", tolerance_px=0.0)
    assert contact.supported
    assert contact.ios == pytest.approx(1.0)


def test_bottom_edge_samples_span_bbox_width() -> None:
    samples = bottom_edge_samples(bbox_to_polygon((10, 20, 30, 40)), samples=5)
    assert samples[0] == (10.0, 40.0)
    assert samples[-1] == (30.0, 40.0)
    assert len(samples) == 5


def test_point_inside_uses_boundary() -> None:
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert point_inside(poly, (5, 5))
    assert point_inside(poly, (0, 5))  # boundary counts
    assert not point_inside(poly, (11, 5))


def test_cosine_similarity_edge_cases() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    assert cosine_similarity(None, [1, 0]) == 0.0
    assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)


def test_ema_is_time_aware() -> None:
    one_second = ema_time_aware(0.0, 1.0, 1.0, 1.0)
    assert one_second == pytest.approx(1 - math.exp(-1.0))
    half = ema_time_aware(0.0, 1.0, 0.5, 1.0)
    two_halves = ema_time_aware(half, 1.0, 0.5, 1.0)
    assert two_halves == pytest.approx(one_second)
    assert ema_time_aware(None, 5.0, 0.1, 0.1) == 5.0


def test_polyline_velocity() -> None:
    vx, vy, speed = polyline_velocity([(0.0, 0.0, 0.0), (1.0, 30.0, 40.0)])
    assert (vx, vy) == (30.0, 40.0)
    assert speed == pytest.approx(50.0)
    assert polyline_velocity([(0.0, 0.0, 0.0)]) == (0.0, 0.0, 0.0)
    assert polyline_velocity([(1.0, 0.0, 0.0), (1.0, 5.0, 5.0)]) == (0.0, 0.0, 0.0)


def test_center_outside_frame() -> None:
    poly = bbox_to_polygon((700, 10, 750, 50))
    assert not center_outside_frame(poly, (800, 600))
    assert center_outside_frame(poly, (400, 400))
