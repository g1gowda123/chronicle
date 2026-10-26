"""Tunable thresholds for the deterministic state machine.

Every value is expressed in wall-clock seconds or pixels so that behaviour does
not depend on the capture frame rate.  Defaults follow the Chronicle technical
specification (section 4) where the specification fixes a value; the remaining
values are documented engineering choices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Geometry, motion and dwell gates for :class:`LifecycleEngine`."""

    # --- camera / parallax gating (pixels per frame) -----------------------
    tau_pan_px: float = 12.0
    """Translation magnitude above which a pan event is dispatched."""

    tau_still_px: float = 2.5
    """Translation magnitude below which the camera counts as stationary."""

    camera_smoothing_s: float = 0.15
    """Time constant of the exponential smoother applied to camera motion."""

    # --- background-relative velocity (pixels / second) --------------------
    tau_move_px_s: float = 25.0
    """Relative speed above which an object counts as independently moving."""

    tau_static_px_s: float = 6.0
    """Relative speed below which an object counts as at rest on its support."""

    velocity_window_s: float = 0.30
    """Centroid history window used to estimate velocity."""

    velocity_min_span_s: float = 0.10
    """Minimum history span before a velocity estimate is trusted."""

    # --- interaction / containment geometry (Intersection over Smaller Area) -
    ios_interact: float = 0.60
    """IoS(hand, object) required to flag an interaction."""

    ios_held: float = 0.50
    """IoS(hand, object) required for the HELD state."""

    ios_contained: float = 0.85
    """IoS(object, container) required for the CONTAINED state."""

    ios_hand_free: float = 0.10
    """IoS below which the hand is considered uninvolved."""

    held_velocity_cosine: float = 0.80
    """Cosine similarity between object and hand velocity for HELD."""

    support_tolerance_px: float = 12.0
    """Pixel tolerance when testing ymax contact against a support surface."""

    support_min_edge_overlap: float = 0.60
    """Fraction of the object's bottom edge that must touch the support."""

    support_min_ios: float = 0.50
    """Alternative support test for boxes wholly overlapping a surface."""

    occluder_min_ios: float = 0.50
    """IoS(last bounding box, occluder) required to call OCCLUDED."""

    # --- wall-clock dwell gates (seconds) ----------------------------------
    static_dwell_s: float = 0.50
    """Spec: object at rest on a support for > 0.5 s before VISIBLE_STATIC."""

    interaction_dwell_s: float = 0.30
    """Spec: hand contact sustained for >= 0.3 s before INTERACTED."""

    moving_dwell_s: float = 0.20
    """Debounce for independent displacement (VISIBLE_MOVING)."""

    held_dwell_s: float = 0.20
    """Debounce for lock-step hand tracking (HELD)."""

    contained_dwell_s: float = 0.15
    """Debounce for enclosure (CONTAINED)."""

    occlusion_dwell_s: float = 0.20
    """Grace period before a missing track is reported OCCLUDED/OUT_OF_VIEW."""

    exit_dwell_s: float = 0.15
    """Debounce before leaving the CONTAINED state when the object reappears."""

    lost_after_s: float = 600.0
    """Spec: OUT_OF_VIEW for more than 10 minutes becomes LOST."""

    # --- identity & place recognition (cosine similarity) ------------------
    reid_cosine: float = 0.82
    """CLIP re-ID threshold for foreground entity reconciliation."""

    place_cosine: float = 0.85
    """CLIP visual place recognition threshold for background zone dedup."""

    # --- data retention ----------------------------------------------------
    trajectory_retention_s: float = 600.0
    """Spec: bounding-box trajectories are purged after 10 minutes."""

    purge_interval_s: float = 30.0
    """Minimum wall-clock gap between retention sweeps."""


DEFAULT_THRESHOLDS = Thresholds()

CONTAINER_LABELS: frozenset[str] = frozenset(
    {
        "backpack",
        "bag",
        "basket",
        "bin",
        "bowl",
        "box",
        "bucket",
        "cabinet",
        "case",
        "container",
        "cup",
        "drawer",
        "fridge",
        "jar",
        "mug",
        "pot",
        "shelf",
        "trash can",
    }
)
"""Entity labels that may enclose another entity (CONTAINED)."""
