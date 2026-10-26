# Role C — Semantic Engine & Multimodal Verification

This document is the integration contract for **Role C** of the Chronicle
specification:

| Spec requirement | Implementation |
| --- | --- |
| OpenCLIP Visual Place Recognition (VPR), 0.85 cosine threshold, zone dedup | `chronicle/semantic/vpr.py`, `embedders.py` |
| 1×3 temporal strip compositor for 30-frame interaction buffers | `chronicle/semantic/temporal_strip.py` |
| Florence-2 `<MORE_DETAILED_CAPTION>` + strict JSON action classification | `chronicle/semantic/florence.py`, `actions.py`, `verification.py` |
| GraphRAG interface between serialized subgraphs and a local LLM | `chronicle/semantic/graphrag.py`, `llm.py` |
| One facade + telemetry events | `chronicle/semantic/engine.py` |

The code has **offline fallbacks** for every heavy dependency, so the pipeline,
demo, and tests run on a laptop with no GPU, no model weights, and no Ollama.
Production runs automatically use OpenCLIP, Florence-2, and Ollama when they are
available.

> The dashboard/CLI use `ChronicleRuntime` rather than calling
> `SemanticEngine` directly. See [`INTEGRATION.md`](INTEGRATION.md) for the
> clip-queue ordering and GraphContext normalization.

---

## 1. Install & run

### Offline / demo machine (only NumPy + Pillow)

```powershell
pip install -r requirements-semantic.txt
python -m chronicle.semantic.demo
python -m unittest discover -s tests -v
```

### Production / GPU machine

```powershell
pip install -r requirements-semantic-models.txt
ollama pull qwen2.5:3b-instruct   # optional but recommended
```

`torch` chooses CUDA automatically; override with `CHRONICLE_DEVICE=cpu|cuda|mps`.
The first Florence-2 / OpenCLIP call downloads weights from Hugging Face.

### Demo output

The demo uses synthetic kitchen/desk frames and a synthetic mug interaction. It
exercises the real public API end to end:

1. two zones are created, then relocalized from noisy keyframes;
2. a 30-frame buffer becomes a 1×3 strip (`PRE / CONTACT / POST`);
3. the strip is captioned and classified into a validated `INSERT_INTO`;
4. a 2-hop subgraph produces a cited answer (`[t=16.4s]`);
5. Role D-style telemetry events are printed.

---

## 2. Quick start

```python
from chronicle.semantic import SemanticConfig, SemanticEngine

engine = SemanticEngine(
    config=SemanticConfig.from_env(),
    on_event=lambda event: print(event["tag"], event["message"]),
)
```

`SemanticEngine` lazily loads models:

* `identify_zone()` triggers OpenCLIP (or the deterministic fallback) and
  Florence only when a **new** zone is created.
* `verify_interaction()` triggers Florence once for the strip.
* `answer_query()` calls Ollama if reachable, otherwise a grounded template.

---

## 3. Role A integration — perception & 30-frame buffer

Role A owns YOLO-World, MediaPipe, ByteTrack, and the ring buffer. Role C
consumes Role A's native `chronicle.types.InteractionClip` directly, so no
frame re-wrapping is needed:

```python
from chronicle.types import InteractionClip

result = engine.verify_interaction(clip)   # clip: InteractionClip

action = result.action              # ActionVerification (validated JSON)
strip = result.strip                # TemporalStrip (PIL image + metadata)
result.clip_id, result.track_id     # copied from the clip for Role B/D correlation
```

When a clip is passed, Role C reads `label`, `contact_index`, `contact_ts`,
`trigger`, `hand`, `max_ios`, and `metadata["container"]` /
`metadata["context_hint"]` automatically. Any argument you pass explicitly
(`entity_label`, `contact_index`, `container`, `context_hint`, `timestamp`)
overrides the inferred value.

The lower-level call is still available for raw buffers:

```python
result = engine.verify_interaction(
    frames,                         # 30-frame deque/list
    entity_label="keys",
    contact_index=contact_frame,    # index into that list; omit to auto-pick
    container=None,                 # e.g. "drawer" if known from geometry
    context_hint="hand lowers keys into the drawer",
    timestamp=video_time_s,
)
```

### Accepted frame formats

```python
# 1. Role A InteractionClip (preferred)
InteractionClip(frames=[ClipFrame(...), ...], contact_index=14, label="keys", ...)

# 2. explicit
FrameSample(image=pil_or_cv2_array, timestamp=12.47, index=14, contact_score=0.93)

# 3. dict (keys are flexible)
{"frame": image, "timestamp": 12.47, "index": 14, "ios": 0.93}

# 4. raw PIL image or OpenCV BGR ndarray
```

`ClipFrame` images are BGR `np.ndarray` (OpenCV order). Build the engine with
`channel_order="bgr"` (the default) so colour conversion is correct.

### Sharing one OpenCLIP model with Role A

Role A's `EmbeddingService` / `OpenClipEmbedder` expose a single-image
`embed_image()`. Pass either object as `embedder=`; Role C wraps it in
`PerceptionEmbedderAdapter` to satisfy the batch `ImageEmbedder` protocol:

```python
from chronicle.perception import EmbeddingService
from chronicle.semantic import SemanticEngine

embedder = EmbeddingService(perception_config)
engine = SemanticEngine(embedder=embedder)   # no second OpenCLIP load
```

`adapt_embedder()` is the public helper if you need the adapter directly. Plain
batch embedders (objects with `embed_images`) pass through unchanged.

`contact_index` is absolute within the supplied list. If omitted, the frame
with the highest `contact_score` / `ios` is used, otherwise the middle frame.
Selection is spread around contact: `contact ± max(1, len(buffer)//6)`.

### Buffers are not mutated

`verify_interaction` copies references into a `TemporalStrip`; Role A can clear
its deque immediately afterwards.

---

## 4. Role B integration — graph & storage

### 4.1 Zone relocalization

```python
match = engine.identify_zone(keyframe, timestamp=video_time_s)

match.zone.zone_id          # stable id, e.g. "zone_kitchen_counter"
match.zone.label            # "kitchen counter"
match.is_new                # True -> insert node in background graph
match.score                 # cosine similarity
match.as_event()            # JSON-ready payload for SQLite/UI
```

`as_event()` returns:

```json
{
  "event": "ZONE_CREATED",
  "tag": "VLM",
  "zone_id": "zone_kitchen_counter",
  "label": "kitchen counter",
  "caption": "a kitchen counter with ...",
  "similarity": 1.0,
  "runner_up": null,
  "is_new": true,
  "timestamp": 0.0
}
```

Only create a background node when `is_new` is true (or when the id is not in
the graph). Merge pass:

```python
engine.merge_duplicate_zones()   # returns [(removed_id, kept_id), ...]
engine.save_zones("data/zones.json")
engine.load_zones("data/zones.json")
```

### 4.2 GraphRAG queries

Either pass a NetworkX-like graph directly:

```python
from chronicle.semantic import serialize_subgraph

context = serialize_subgraph(graph, "keys", hops=2)
answer = engine.answer_query("Where are my keys?", context)
```

or the serialized mapping (preferred if Role B already has a SQLite projection):

```python
context = {
    "entity": {
        "id": "keys",
        "label": "keys",
        "state": "CONTAINED",
        "zone": "work desk",
        "last_seen": 16.4,
    },
    "facts": [
        {
            "subject": "keys",
            "relation": "INSERT_INTO",
            "object": "drawer",
            "timestamp": 16.4,
            "zone": "work desk",
            "confidence": 0.88,
        }
    ],
}
answer = engine.answer_query("Where are my keys?", context)
```

#### Fact aliases accepted

| Canonical | Accepted aliases |
| --- | --- |
| `subject` | `source`, `from`, `u` |
| `relation` | `predicate`, `type`, `label`, `action`, `edge_type` |
| `object` | `object_label`, `target`, `destination`, `to`, `v` |
| `timestamp` | `t`, `tevent`, `time`, `start` |
| `zone` | `location`, `place` |
| `confidence` | `score` |

NetworkX edge attributes are read the same way. Node `label` / `name` are used
for human-readable facts. The terminal spatial anchor should still live in
SQLite exactly as Role B designed; Role C never deletes graph data.

#### Answer object

```python
answer.answer       # "Keys was last seen inside the drawer ... [t=16.4s]."
answer.source       # "llm" | "template"
answer.grounded     # True if at least one citation maps to a fact
answer.citations    # (Citation(timestamp, relation, zone, fact), ...)
answer.to_dict()    # JSON-ready for the dashboard
```

A citation is only accepted if its timestamp maps to a fact within 0.35 s.
Hallucinated timestamps are dropped and an uncited LLM answer automatically
falls back to the deterministic template, so the judge never sees an
ungrounded location.

---

## 5. Role D integration — telemetry & UI

Pass `on_event` to the engine. Events are emitted from the ingestion thread and
exceptions in the sink are swallowed (telemetry must never block ingestion).

```python
def push(event: dict) -> None:
    ui.log(f"[{event['tag']}] {event['message']}")
    ui.emit("chronicle_event", event)

engine = SemanticEngine(..., on_event=push)
```

Payload:

```json
{
  "tag": "VLM | GRAPH",
  "event": "ZONE_CREATED | ZONE_RELOCALIZED | ACTION_VERIFIED | QUERY_ANSWERED | ZONES_MERGED",
  "message": "INSERT_INTO mug (conf=0.80, accepted=True)",
  "data": {"action": "INSERT_INTO", "confidence": 0.8, "...": "..."},
  "emitted_at": 1770000000.0
}
```

### Visual strip preview

```python
result = engine.verify_interaction(...)
result.strip.save("crops/strip_14.jpg")
result.strip.to_data_url()          # data:image/jpeg;base64,... for <img src=...>
result.to_dict(include_preview=True)
```

### Actions needing review

`action.accepted` is false when the action is `OTHER` or confidence is below
`CHRONICLE_ACTION_MIN_CONFIDENCE` (default `0.45`). Use it to flag nodes in the
drawer instead of writing low-quality action edges.

---

## 6. Strict action JSON schema

Florence-2 captions the strip; a local LLM (JSON mode) or the deterministic
rule classifier returns:

```json
{
  "action": "INSERT_INTO",
  "object": "keys",
  "container": "drawer",
  "confidence": 0.91,
  "evidence": "the hand lowers the keys into the open drawer",
  "timestamp": 14.2
}
```

Allowed `action` values:

`PICK_UP`, `PUT_DOWN`, `INSERT_INTO`, `REMOVE_FROM`, `OPEN`, `CLOSE`, `PUSH`,
`PULL`, `HAND_OFF`, `HOLD`, `OTHER`

Validation rules enforced by `parse_action_json`:

* response may be bare JSON, fenced JSON, or short prose + JSON;
* `action` must be one of the enum values (aliases such as `"pick up"` are
  accepted);
* `object` required unless the caller supplies a default;
* `confidence` must be a number in `[0, 1]` (booleans rejected);
* `evidence` must be non-empty;
* malformed output is retried once with the parser error, then falls back to the
  rule classifier; the graph only ever receives validated objects.

Use it directly if needed:

```python
from chronicle.semantic import ActionParseError, parse_action_json

try:
    action = parse_action_json(raw_text, default_object="keys")
except ActionParseError as exc:
    ...
```

---

## 7. Configuration

Every field of `SemanticConfig` can be set with
`CHRONICLE_<UPPERCASE_FIELD_NAME>`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHRONICLE_VPR_THRESHOLD` | `0.85` | zone relocalization threshold |
| `CHRONICLE_VPR_MIN_MARGIN` | `0.0` | optional top-2 separation |
| `CHRONICLE_VPR_PROTOTYPE_EMA` | `0.85` | zone prototype blending |
| `CHRONICLE_CLIP_MODEL_NAME` | `ViT-B-32` | OpenCLIP architecture |
| `CHRONICLE_CLIP_PRETRAINED` | `laion2b_s34b_b79k` | OpenCLIP weights |
| `CHRONICLE_DEVICE` | `auto` | `cpu`, `cuda`, `mps` |
| `CHRONICLE_FLORENCE_MODEL_ID` | `microsoft/Florence-2-base` | captioner |
| `CHRONICLE_ACTION_MIN_CONFIDENCE` | `0.45` | accept/review cutoff |
| `CHRONICLE_LLM_BACKEND` | `auto` | `auto`, `ollama`, `transformers`, `template`, `none` |
| `CHRONICLE_LLM_MODEL` | `qwen2.5:3b-instruct` | Ollama / HF model id |
| `CHRONICLE_OLLAMA_URL` | `http://127.0.0.1:11434` | daemon URL |
| `CHRONICLE_LLM_TIMEOUT_S` | `60` | generation timeout |
| `CHRONICLE_REQUIRE_CITATIONS` | `true` | reject uncited LLM answers |
| `CHRONICLE_STRIP_HEIGHT` | `336` | strip panel height |
| `CHRONICLE_STRIP_PANEL_WIDTH` | `336` | strip panel width |
| `CHRONICLE_STRIP_GAP` | `6` | separator width |
| `CHRONICLE_STRIP_LABELS` | `true` | draw PRE/CONTACT/POST labels |

`auto` LLM mode **only probes Ollama**. It never downloads a multi-GB
transformers model during ingestion. Set `llm_backend="transformers"` explicitly
to load `CHRONICLE_LLM_MODEL` with Hugging Face `pipeline("text-generation")`.

---

## 8. Fallback matrix

| Component | Missing dependency / failure | Result |
| --- | --- | --- |
| OpenCLIP | torch/open_clip/weights unavailable | deterministic HSV embedder (demo only) |
| Florence-2 | transformers/weights unavailable | stub caption + pipeline `context_hint` |
| Ollama | daemon down / model missing | deterministic action rules + grounded template answers |
| LLM action JSON | invalid twice | rule classifier; `last_parse_error` set |
| LLM query answer | no verifiable citation | template answer with a real graph citation |
| Dashboard sink | exception | logged; ingestion continues |

---

## 9. Tests

```powershell
python -m unittest discover -s tests -v
```

58 tests cover:

* VPR create/relocalize/threshold/dedup/persistence;
* temporal strip selection, sizing, dict/PIL inputs, edge cases;
* every strict JSON rejection path, rule classifier variants, LLM repair/fallback;
* graph serialization (NetworkX-like + mapping), citation validation, GraphRAG
  template/LLM/fallback paths;
* the end-to-end facade and telemetry contract;
* Role A integration: `InteractionClip`/`ClipFrame` ingestion and the embedder
  adapter.

The suite has **no model downloads and no network access**.

---

## 10. Known limitations / next steps

* Action classification currently uses Florence-2's `<MORE_DETAILED_CAPTION>`
  followed by a text-only JSON classifier. Swapping in a vision-language model
  is a matter of replacing `ActionVerifier.chat_client` with a vision client;
  the schema and fallbacks stay unchanged.
* The deterministic embedder is intentionally not CLIP-quality. It exists for
  offline demos/tests; production must install `requirements-semantic-models.txt`.
* Zone prototypes are in-memory + JSON by default. Role B can persist
  `match.as_event()` / `Zone.to_dict()` in SQLite if the background graph needs
  a single source of truth.
* GraphRAG grounds timestamps and zones, but does not yet re-rank candidate
  entities with CLIP text-to-node similarity; Role B currently supplies the
  entity context.
