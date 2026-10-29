"""Configuration for the Role A perception pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_VOCABULARY: tuple[str, ...] = (
    # everyday egocentric objects
    "keys",
    "phone",
    "mug",
    "cup",
    "bottle",
    "backpack",
    "bag",
    "wallet",
    "notebook",
    "pen",
    "book",
    "glasses",
    # furniture / room
    "table",
    "desk",
    "chair",
    "sofa",
    "shelf",
    "cabinet",
    "door",
    "window",
    "trash can",
    "whiteboard",
    # computer-lab kit
    "monitor",
    "laptop",
    "mouse",
    "keyboard",
    "headphones",
    "charger",
    "cable",
    "router",
    "tv",
    "person",
)

CS_LAB_VOCABULARY: tuple[str, ...] = (
    "person",
    "monitor",
    "keyboard",
    "mouse",
    "phone",
    "laptop",
    "computer tower",
    "bag",
    "backpack",
    "chair",
    "desk",
    "table",
    "bottle",
    "cup",
    "charger",
    "cable",
    "headphones",
    "book",
    "notebook",
    "bus",
    "door",
    "window",
)

DEFAULT_ALIASES: dict[str, str] = {
    "key": "keys",
    "keys ": "keys",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "cellphone": "phone",
    "bag": "backpack",
    "hand bag": "backpack",
    "computer": "computer tower",
    "pc": "computer tower",
    "desktop": "computer tower",
    "tower": "computer tower",
    "notebook": "laptop",
    "couch": "sofa",
    "screen": "monitor",
    "display": "monitor",
    "computer monitor": "monitor",
    "computer screen": "monitor",
    "computer mouse": "mouse",
    "computer keyboard": "keyboard",
    "laptop computer": "laptop",
    "notebook computer": "laptop",
    "back pack": "backpack",
    "handbag": "bag",
    "tote bag": "bag",
    "shoulder bag": "bag",
    "notepad": "notebook",
    "headphone": "headphones",
    "earphones": "headphones",
    "power adapter": "charger",
    "usb cable": "cable",
    "charging cable": "cable",
    "white board": "whiteboard",
    "garbage can": "trash can",
    "bin": "trash can",
}

#: Opt-in broad prompt set for uploaded clips.  YOLO-World is open-vocabulary
#: but prompt-driven: it will not discover objects it was not asked about.
#: This covers ~100 everyday egocentric/indoor objects at roughly the same
#: per-frame cost as the starter list, at the price of more false positives.
BROAD_VOCABULARY: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "bench",
            "backpack",
            "umbrella",
            "handbag",
            "suitcase",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "monitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "toothbrush",
            "desk",
            "door",
            "window",
            "wall",
            "floor",
            "shelf",
            "cabinet",
            "drawer",
            "box",
            "bag",
            "keys",
            "mug",
            "paper",
            "pen",
            "pencil",
            "notebook",
            "table",
            "countertop",
            "headphones",
            "speaker",
            "router",
            "cable",
            "charger",
            "plant",
            "picture frame",
            "trash can",
            "towel",
            "soap",
            "mirror",
            "lamp",
            "pillow",
            "blanket",
            "curtain",
            "clothes",
            "shoe",
            "hat",
            "glasses",
            "wallet",
            "watch",
            "ring",
            "food",
            "fruit",
            "vegetable",
            "pan",
            "pot",
            "plate",
            "glass",
            "jar",
        )
    )
)


@dataclass(slots=True)
class MotionConfig:
    enabled: bool = True
    still_threshold: float = 2.5  # px; spec: < 2.5 px is STATIONARY
    still_rotation_degrees: float = 0.6
    pan_threshold: float = 8.0  # px; spec: > tau_pan dispatches a pan event
    pan_rotation_degrees: float = 1.8
    min_features: int = 24
    mask_padding: int = 8
    max_corners: int = 600
    quality: float = 0.01
    min_distance: float = 8.0
    ransac_threshold: float = 2.5
    scale: float = 0.5  # estimate motion on a half-resolution grayscale frame
    max_width: int = 640  # never estimate motion above this width (caps 1080p/4K cost)
    fb_error: float = 3.0  # forward-backward LK consistency gate


@dataclass(slots=True)
class DetectorConfig:
    backend: str = "auto"  # auto | yolo-world | color
    model: str = "yolov8s-worldv2.pt"
    # Optional multi-model ensemble.  Different YOLO-World scales disagree on
    # open-vocabulary classes (a large model grounds wooden boards, a small one
    # keeps small bowls), so running both and merging their boxes recovers
    # recall that either model misses alone.  Empty means "just `model`".
    models: tuple[str, ...] = ()
    confidence: float = 0.30
    iou: float = 0.45
    imgsz: int = 640
    max_detections: int = 50
    prompts: tuple[str, ...] | None = None
    label_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    # Prompt ensembling: an open-vocabulary detector may need a different
    # surface form than the label a user asks for (``mug`` rather than ``cup``).
    # Each canonical vocabulary term maps to fixed extra prompts; detections
    # from those prompts are renamed back to the canonical term.  This is
    # prompt engineering over a fixed class list, not test-set vocabulary.
    prompt_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min_label_area: dict[str, float] = field(default_factory=dict)
    # Multi-scale tiling for small-object recall.  A 2x2 grid gives each
    # quarter ~4x the pixels at the detector input; this is what recovers
    # cards, straps and buttons that vanish in a full-frame resize.  Tiling is
    # restricted to ``tile_labels`` so large-object tracks are unchanged.
    tile_grid: int = 1
    tile_overlap: float = 0.15
    tile_confidence: float | None = None
    tile_labels: tuple[str, ...] = ()
    dedup_iou: float = 0.55
    dedup_ios: float = 0.80
    max_per_label: int = 3


@dataclass(slots=True)
class HandConfig:
    backend: str = "auto"  # auto | mediapipe | hybrid | color | off
    confidence: float = 0.35
    max_hands: int = 2
    model: str = "hand_landmarker.task"
    upscale: float = 1.0
    max_width: int = 960  # downscale before MediaPipe; landmarks stay normalised
    interval: int = 2  # reuse landmarks on skipped frames


@dataclass(slots=True)
class EmbedderConfig:
    backend: str = "auto"  # auto | openclip | color
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    dimension: int = 512
    refresh_interval: int = 30  # refresh active-track appearance every N frames
    cache_size: int = 512


@dataclass(slots=True)
class TrackerConfig:
    reid_threshold: float = 0.82  # spec: cosine similarity > 0.82
    high_confidence: float = 0.45
    low_confidence: float = 0.10
    min_hits: int = 2
    iou_gate: float = 0.02
    iou_low_gate: float = 0.10
    inactive_ttl: float = 120.0  # seconds an inactive track stays re-identifiable
    spatial_reid_distance: float = 260.0  # px radius for spatial re-binding
    spatial_reid_window: float = 60.0  # seconds since last seen
    max_center_distance: float = 220.0
    smooth: float = 0.65
    refresh_interval: int = 30  # refresh active-track embeddings every N frames


@dataclass(slots=True)
class InteractionConfig:
    enabled: bool = True
    ios_threshold: float = 0.60  # spec: IoS(Hand, Object) > 0.60
    dwell_seconds: float = 0.30  # spec: sustained for delta t > 0.3 s
    release_grace: float = 0.15  # tolerate one/two dropped hand frames
    fingertip_indices: tuple[int, ...] = (4, 8)  # thumb + index tips
    fingertip_tolerance_px: float = 2.0
    min_hand_area: float = 12.0  # degenerate hand hull guard
    ignore_labels: tuple[str, ...] = (
        "hand",
        "table",
        "desk",
        "counter",
        "countertop",
        "sofa",
        "couch",
        "shelf",
        "floor",
        "bed",
    )


@dataclass(slots=True)
class ClipConfig:
    enabled: bool = True
    ring_size: int = 30  # spec: 30-frame RAM ring buffer
    pre_roll: int = 10
    length: int = 30
    post_roll: int = 5
    max_width: int = 640  # downscale stored frames to keep the ring buffer cheap
    max_pending: int = 8
    save_frames: bool = False


@dataclass(slots=True)
class AttributesConfig:
    """Cheap per-crop appearance facts (colour, shape, size, texture)."""

    enabled: bool = True
    interval: int = 3  # recompute every N frames; the tracker retains the last value
    min_crop_px: int = 8
    colour_clusters: int = 3


@dataclass(slots=True)
class SupportConfig:
    """Role A static-support candidate extraction.

    Role A does not claim to *verify* physical support; it emits persistent
    furniture candidates so Role B can reason about ``RESTING_ON`` without
    waiting for a separate surface detector.  Association is deliberately
    conservative and camera-compensated, mirroring the graph-level furniture
    registry.
    """

    enabled: bool = True
    labels: tuple[str, ...] = (
        "table",
        "desk",
        "counter",
        "countertop",
        "sofa",
        "couch",
        "shelf",
        "floor",
        "bed",
        "chair",
        "bench",
        "stool",
        "tv",
        "television",
        "sink",
        "stove",
        "oven",
    )
    min_hits: int = 2  # emit after the candidate is confirmed across frames
    iou_threshold: float = 0.25
    center_distance_ratio: float = 0.12
    match_threshold: float = 0.38
    max_missed_s: float = 45.0
    camera_compensation: bool = True


@dataclass(slots=True)
class PerceptionConfig:
    vocabulary: tuple[str, ...] = DEFAULT_VOCABULARY
    device: str = "auto"  # auto | cpu | cuda:0
    model_dir: Path = field(default_factory=lambda: Path("models"))
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    hands: HandConfig = field(default_factory=HandConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    attributes: AttributesConfig = field(default_factory=AttributesConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    frame_stride: int = 1
    max_frames: int | None = None
    demo_tokens: tuple[str, ...] = ("demo", "synthetic")
    telemetry_interval: int = 15  # periodic [PERCEPTION] summary cadence

    def __post_init__(self) -> None:
        self.vocabulary = tuple(dict.fromkeys(v.strip().lower() for v in self.vocabulary if v.strip()))
        self.model_dir = Path(self.model_dir).expanduser()

    def resolve_model(self, name: str, *, required: bool = False) -> str:
        """Resolve a model name against ``model_dir`` before letting the backend search.

        Relative names that exist under ``model_dir`` are returned as a path.
        Otherwise the bare name is returned so Ultralytics can auto-download it.
        """
        path = Path(name).expanduser()
        if path.is_absolute() or path.exists():
            return str(path)
        candidate = self.model_dir / path
        if candidate.exists():
            return str(candidate)
        if required:
            raise FileNotFoundError(
                f"model '{name}' was not found under {self.model_dir}. "
                "Run `python scripts/fetch_assets.py` from the repository root."
            )
        return name

    def resolve_hand_model(self) -> str:
        return self.resolve_model(self.hands.model, required=True)

    def is_demo_source(self, source_name: str) -> bool:
        lowered = source_name.lower()
        return any(token in lowered for token in self.demo_tokens)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_dir"] = str(self.model_dir)
        return data
