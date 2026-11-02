"""Configuration objects for the video-free memory (VFM) compiler.

The compiler used to expose its policies as module-level constants.  Research
ablations need the same code path with one component switched off, so the
policy is now a frozen dataclass that can be copied and varied:

    >>> from chronicle.vfm.config import CompileConfig
    >>> cfg = CompileConfig(include_anchors=False, counts_mode="tracks")
    >>> compile_vault(source_db, out_db, config=cfg)

Every field is serialised into the vault ``meta`` table, so a result file can
always be traced back to the exact compiler configuration that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

#: Transition reasons that are visibility noise, not memory-worthy events.
DEFAULT_NOISE_REASONS = frozenset(
    {
        "camera_pan",
        "occluder_overlap",
        "line_of_sight_lost",
        "reacquired_without_support_geometry",
        "resting_no_surface_detector",
        "independent_motion",
        "independent_motion_relaxed",
    }
)

#: Support values for ``counts_mode``.
COUNTS_MODES = ("tracks", "peak", "dedup")
#: Support values for ``compression``.
COMPRESSION_MODES = ("packed", "zlib-json", "json")


@dataclass(frozen=True, slots=True)
class CompileConfig:
    """Policy for turning a Chronicle source DB into a VFM vault.

    Parameters are deliberately small and orthogonal so that a research
    ablation changes exactly one thing.  ``name`` is a human-readable label
    stored in the vault; it does not affect behaviour.
    """

    name: str = "full"

    # Caption / periodic-anchor cadence.
    window_s: float = 30.0
    periodic_window_s: float | None = None

    # Salience coalescing.
    interaction_gap_s: float = 6.0
    placement_gap_s: float = 8.0
    drop_noise_reasons: bool = True
    noise_reasons: frozenset[str] = DEFAULT_NOISE_REASONS

    # Event sources.
    include_appearances: bool = True
    include_lifecycle: bool = True
    include_anchors: bool = True
    include_relations: bool = True
    include_periodic_anchors: bool = True
    include_caption_windows: bool = True

    # Count reconciliation.
    counts_mode: str = "dedup"

    # Storage layout.
    compression: str = "packed"

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.periodic_window_s is not None and self.periodic_window_s <= 0:
            raise ValueError("periodic_window_s must be positive")
        if self.counts_mode not in COUNTS_MODES:
            raise ValueError(f"counts_mode must be one of {COUNTS_MODES}")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"compression must be one of {COMPRESSION_MODES}")

    @property
    def periodic_step(self) -> float:
        return float(self.periodic_window_s or self.window_s)

    def replace(self, **changes: Any) -> CompileConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["noise_reasons"] = sorted(self.noise_reasons)
        payload["periodic_step"] = self.periodic_step
        return payload


#: The configuration used by the paper's main system and all existing tests.
DEFAULT_COMPILE_CONFIG = CompileConfig()
