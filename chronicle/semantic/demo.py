"""Offline end-to-end demo for Role C.

Run from the repository root:

    python -m chronicle.semantic.demo

It uses synthetic frames so no camera, GPU, model download, or Ollama daemon is
required.  Every component runs through the same public API the real pipeline
uses, making this a useful smoke test before wiring up Florence-2/OpenCLIP.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from .actions import ActionParseError, parse_action_json
from .calibration import calibrate_rule_classifier
from .config import SemanticConfig
from .embedders import DeterministicImageEmbedder
from .engine import SemanticEngine
from .evaluation import evaluate_graphrag
from .florence import StubCaptioner
from .temporal_strip import FrameSample


def _noise(image: Image.Image, rng: np.random.Generator, sigma: float) -> Image.Image:
    if sigma <= 0:
        return image
    array = np.asarray(image, dtype=np.float32)
    array = array + rng.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _room(kind: str, rng: np.random.Generator, noise: float = 3.0) -> Image.Image:
    if kind == "kitchen":
        wall, floor, furniture, accent = (214, 196, 161), (126, 101, 76), (120, 90, 66), (188, 64, 54)
    else:
        wall, floor, furniture, accent = (150, 176, 205), (90, 82, 76), (96, 68, 48), (56, 122, 176)

    image = Image.new("RGB", (640, 480), wall)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 330, 640, 480), fill=floor)
    draw.rectangle((0, 300, 640, 330), fill=(235, 235, 235))

    if kind == "kitchen":
        draw.rectangle((60, 170, 580, 310), fill=furniture)
        draw.rectangle((90, 190, 210, 300), fill=(70, 70, 70))
        draw.ellipse((300, 220, 340, 260), fill=accent)  # kettle
        draw.rectangle((420, 180, 560, 300), fill=(205, 205, 205))  # fridge
    else:
        draw.rectangle((120, 260, 560, 330), fill=furniture)
        draw.rectangle((150, 120, 400, 260), fill=(45, 55, 70))  # monitor
        draw.rectangle((170, 140, 380, 240), fill=(90, 140, 190))
        draw.ellipse((470, 220, 520, 270), fill=accent)  # mug
        draw.rectangle((450, 300, 560, 330), fill=(160, 120, 80))  # drawer

    return _noise(image, rng, noise)


def _interaction_frame(step: int, total: int) -> FrameSample:
    """A mug moves from the desk into an open drawer; contact around step 14."""
    base = _room("desk", np.random.default_rng(7), noise=0.0)
    draw = ImageDraw.Draw(base)
    progress = min(1.0, step / max(total - 1, 1))
    x = int(485 + 15 * math.sin(progress * math.pi))
    y = int(240 + 75 * progress)
    draw.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(196, 58, 58))
    draw.arc((x - 30, y - 42, x + 30, y - 14), 200, 340, fill=(30, 30, 30), width=4)
    score = math.exp(-((step - 14) / 3.0) ** 2)
    return FrameSample(image=base, timestamp=12.0 + step / 30.0, index=step, contact_score=score)


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rng = np.random.default_rng(42)
    config = SemanticConfig(
        llm_backend="template",
        vpr_threshold=0.85,
        strip_height=240,
        strip_panel_width=240,
    )

    events: list[dict] = []
    engine = SemanticEngine(
        config=config,
        embedder=DeterministicImageEmbedder(),
        captioner=StubCaptioner(),
        chat_client=None,
        embedder_backend="deterministic",
        captioner_backend="stub",
        on_event=events.append,
    )

    _banner("1. OpenCLIP VPR (deterministic stand-in offline)")
    kitchen = _room("kitchen", rng)
    desk = _room("desk", rng)
    matches = [
        engine.identify_zone(kitchen, caption="kitchen counter", timestamp=0.0),
        engine.identify_zone(desk, caption="work desk", timestamp=4.0),
        engine.identify_zone(_room("kitchen", rng, noise=5.0), timestamp=9.0),
        engine.identify_zone(_room("desk", rng, noise=5.0), timestamp=13.0),
    ]
    for match in matches:
        print(
            f"{match.event_type:18s} {match.zone.zone_id:24s} "
            f"sim={match.score:.3f} label={match.zone.label!r}"
        )
    print(f"zones in background graph: {len(engine.vpr)}")

    _banner("2. 1x3 temporal strip from a 30-frame buffer")
    buffer = [_interaction_frame(step, 30) for step in range(30)]
    interaction = engine.verify_interaction(
        buffer,
        "mug",
        container="drawer",
        context_hint="The hand grasps the red mug and places it inside the open drawer.",
        timestamp=14.2,
    )
    strip = interaction.strip
    print(f"strip size={strip.image.size} panels={strip.panel_indices} contact={strip.contact_index}")
    print(f"panel labels={strip.panel_labels}")
    print(json.dumps(interaction.action.to_dict(), indent=2))
    print(f"preview data URL: {strip.to_data_url()[:78]}...")

    _banner("3. Strict action JSON boundary")
    fenced = """```json
{"action": "INSERT_INTO", "object": "keys", "container": "drawer",
 "confidence": 0.93, "evidence": "hand lowers keys into drawer", "timestamp": 14.2}
```"""
    parsed = parse_action_json(fenced, default_object="keys")
    print(f"accepted fenced JSON -> {parsed.action.value} ({parsed.confidence:.2f})")
    try:
        parse_action_json('{"action": "TELEPORT", "confidence": 0.9, "evidence": "x"}')
    except ActionParseError as exc:
        print(f"rejected invalid action -> {exc}")

    _banner("4. GraphRAG answer with citation")
    context = {
        "entity": {"id": "mug", "label": "mug", "state": "CONTAINED", "zone": "work desk"},
        "facts": [
            {
                "subject": "mug",
                "relation": "RESTING_ON",
                "object": "work desk",
                "timestamp": 12.0,
                "zone": "work desk",
                "confidence": 0.91,
            },
            {
                "subject": "user",
                "relation": "PICK_UP",
                "object": "mug",
                "timestamp": 14.2,
                "zone": "work desk",
                "confidence": 0.88,
            },
            {
                "subject": "mug",
                "relation": "INSERT_INTO",
                "object": "drawer",
                "timestamp": 16.4,
                "zone": "work desk",
                "confidence": 0.86,
            },
        ],
    }
    answer = engine.answer_query("Where is my mug?", context)
    print(answer.answer)
    print(f"source={answer.source} grounded={answer.grounded} citations={[c.to_dict() for c in answer.citations]}")

    _banner("5. Zone deduplication pass")
    merged = engine.merge_duplicate_zones()
    print(f"merged={merged or 'none'} zones={len(engine.vpr)}")

    _banner("6. Bounded verification queue (dedup + backpressure)")
    service = engine.verification_service(max_pending=2)
    clip_frames = buffer

    def make_clip(clip_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            clip_id=clip_id,
            track_id=3,
            label="mug",
            hand="Right",
            frames=clip_frames,
            contact_index=14,
            contact_ts=12.0 + 14 / 30.0,
            trigger="ios+dwell",
            max_ios=0.74,
            metadata={"container": "drawer"},
        )

    print(f"submit clip-a -> {service.submit(make_clip('clip-a'))}")
    print(f"submit clip-b -> {service.submit(make_clip('clip-b'))}")
    print(f"duplicate clip-a -> {service.submit(make_clip('clip-a'))}")
    print(f"overflow clip-c -> {service.submit(make_clip('clip-c'))}")
    print(f"pending={service.queue.pending_ids()}")
    print(f"verified inline={service.drain()}")
    stats = service.stats().to_dict()
    print(
        "stats: "
        f"verified={stats['verified']} dropped={stats['dropped']} "
        f"duplicates={stats['duplicates']} avg_latency_ms={stats['avg_latency_ms']}"
    )

    _banner("7. Calibrated action confidence + GraphRAG evaluation")
    calibration = calibrate_rule_classifier()
    report = calibration.report
    print(
        f"calibration: precision={report.precision:.2f} recall={report.recall:.2f} "
        f"accepted={report.accepted}/{report.samples}"
    )
    eval_report = evaluate_graphrag(engine)
    for line in eval_report.summary_lines():
        print(line)

    _banner("Telemetry events emitted for Role D")
    for event in events:
        prints = event["data"]
        suffix = " ".join(
            f"{key}={value}"
            for key, value in prints.items()
            if key in {"zone_id", "similarity", "action", "confidence", "grounded", "source"}
        )
        print(f"[{event['tag']}] {event['message']}" + (f"  ({suffix})" if suffix else ""))


if __name__ == "__main__":
    main()
