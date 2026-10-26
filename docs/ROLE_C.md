# Role C — Semantic Engine & Multimodal Verification

This document is the integration contract for **Role C** of the Chronicle
specification:

| Spec requirement | Implementation |
| --- | --- |
| OpenCLIP Visual Place Recognition (VPR), 0.85 cosine threshold, zone dedup | `chronicle/semantic/vpr.py`, `embedders.py` |
| 1×3 temporal strip compositor for 30-frame interaction buffers | `chronicle/semantic/temporal_strip.py` |
| Florence-2 `<MORE_DETAILED_CAPTION>` + strict JSON action classification | `chronicle/semantic/florence.py`, `actions.py`, `verification.py` |
| GraphRAG interface between serialized subgraphs and a local LLM | `chronicle/semantic/graphrag.py`, `llm.py` |
| Bounded verification queue, calibration, action-timeline and GraphRAG eval sets | `chronicle/semantic/queue.py`, `calibration.py`, `timeline.py`, `evaluation.py` |
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
ollama pull qwen2.5:7b-instruct   # optional but recommended
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
* `submit_clip()` / `drain_verifications()` run the same verification through
  the bounded queue (dedup + backpressure); `queue_stats()` is dashboard-safe.
* `answer_query()` calls Ollama if reachable, otherwise a grounded template.

```python
service = engine.verification_service()      # bounded, dedup, telemetry
service.queue.max_pending                    # 8 by default
service.submit(clip)                         # False if duplicate / rejected
service.drain()                              # or service.start() / stop()
```


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

#### Multi-hop / "before" questions

Temporal questions such as *"Where was the cup before it was placed on the
table?"* are answered with a backward walk over the entity's facts:

1. locate the reference event from the query (placement/action wording plus the
   target object, e.g. `PLACED table`), or from an explicit `t=9s` / `9 seconds`;
2. keep the facts strictly earlier than that timestamp;
3. answer with the latest earlier location fact, citing its timestamp
   (`"Before it was placed on the table, the cup had been resting on the work
   desk at [t=5.0s]."`).

If no earlier location exists the template returns an ungrounded
`"I don't have an earlier location for ..."` answer instead of inventing one.
The same path is exercised by the `before-placement` evaluation case.

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

`action.accepted` is false when the action is `OTHER`, the caption is
ambiguous/hedged, or confidence is below the per-action threshold (never below
`CHRONICLE_ACTION_MIN_CONFIDENCE`, default `0.45`). Use it to flag nodes in the
drawer instead of writing low-quality action edges.

### Queue depth and verification latency

`VerificationQueue` emits one event per state change through the same
`on_event` sink Role D already consumes:

| Event | Payload |
| --- | --- |
| `QUEUE_DEPTH` | `depth`, `max_pending`, `clip` |
| `QUEUE_BACKPRESSURE` | `clip`, `depth`, `dropped`, `policy` |
| `CLIP_DEDUPLICATED` | `clip`, `duplicates` |
| `VERIFICATION_LATENCY` | `latency_ms`, `avg_latency_ms`, `verified` |
| `VERIFICATION_FAILED` | `latency_ms`, `failed` |

The runtime also exposes `runtime.verification_stats` (a JSON-safe snapshot) if
the dashboard prefers polling to events. `ChronicleRuntime.snapshot()` embeds
the same payload at `snapshot()["verification_stats"]` and
`snapshot()["stats"]["verification_stats"]`, including `depth`, `dropped` and
`avg_latency_ms`. Latency is measured from submission to graph update, so it
includes queue wait, not just model time.

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

### Container extraction

When the caller does not supply a container, Role C extracts it from the
caption (and then the pipeline hint):

```python
from chronicle.semantic import extract_container

extract_container("puts the keys into the open drawer")      # "open drawer"
extract_container("takes the charger out of the drawer",
                  action=ActionPrimitive.REMOVE_FROM)        # "drawer"
extract_container("within the bag")                          # "bag"
extract_container("moves into the frame")                    # None
```

Priority is explicit argument > LLM JSON > caption > `context_hint`. The
extractor is action-aware, so "out of the drawer ... into the pocket" resolves
to `drawer` for `REMOVE_FROM` and `pocket` for `INSERT_INTO`.

### Confidence calibration

`DEFAULT_ACTION_THRESHOLDS` is a per-action acceptance cutoff (for example
`HOLD` 0.62, `INSERT_INTO` 0.80, `OTHER` never). The shipped values are
calibrated on the real HoloAssist Joy-Con clip — see
[`docs/CALIBRATION.md`](CALIBRATION.md) for the labels, report and provenance.
`ActionCalibrator` applies the cutoff, the global floor, and the ambiguity
penalty. Captions containing "appears", "maybe", "unclear", "hard to tell" or a
question mark are never accepted as graph facts, even at 0.95 confidence.

To calibrate on labelled clips:

```python
from chronicle.semantic import calibrate_rule_classifier

result = calibrate_rule_classifier()        # 16-caption offline seed set
print(result.report.precision, result.report.recall)
config = SemanticConfig(action_thresholds=result.thresholds)
```

`calibrate_rule_classifier(labels=[(caption, ActionPrimitive.PICK_UP), ...])`
accepts your own labelled captions; `suggest_thresholds()` works on raw
`CalibrationSample(action, confidence, correct)` values.

For real HoloLens clips, keep the labels outside the code and point the CLI at
them. The shipped defaults come from
[`docs/calibration_labels.holoassist.json`](calibration_labels.holoassist.json):

```bash
python -m chronicle.semantic.calibration \
  --labels docs/calibration_labels.holoassist.json \
  --target-precision 0.9 \
  --write docs/calibration.holoassist.report.json
# labels=19 accepted=14 rejected_correct=5 precision=1.00 recall=1.00
# CHRONICLE_ACTION_THRESHOLDS=CLOSE=0.6,HAND_OFF=0.6,HOLD=0.62,...

python -m chronicle.semantic.calibration \
  --labels runs/hololens_labels.jsonl --target-precision 0.95 \
  --write runs/calibration.json --json
```

`docs/calibration_labels.example.json` is the small smoke-test fixture. See
[`docs/CALIBRATION.md`](CALIBRATION.md) for the label provenance, the
per-action table and the recommended thresholds. NB-18 removed the pre-grasp
`PICK_UP` false positive with negation-aware rule matching, so the real-label
report is now precision 1.00 / recall 1.00.

Accepted label shapes are a list of `{"caption", "action"}` objects,
`(caption, action)` pairs, `{"labels": [...]}`, or
`{"INSERT_INTO": ["caption", ...]}`; `.jsonl`/`.ndjson` files are read line by
line. Actions accept `REMOVE_FROM`, `remove from` or `remove-from`. The command
prints the `CHRONICLE_ACTION_THRESHOLDS=...` value to copy into the environment,
and exits non-zero if the requested precision cannot be reached on those labels
(so CI can gate a regression).

### Action timeline evaluation

The Role C acceptance bar is that the verified actions for a clip match a
hand-labelled timeline within tolerance. `timeline.py` measures exactly that
against the same `snapshot()["actions"]` rows the dashboard consumes.

Label file (also accepted as `{"labels": [...]}`):

```json
[
  {"timestamp": 1.9, "action": "PICK_UP", "entity": "keys"},
  {"timestamp": 4.2, "action": "INSERT_INTO", "entity": "keys", "container": "drawer"}
]
```

```python
from chronicle.semantic import (
    evaluate_action_timeline,
    load_action_labels,
    load_annotation_actions,
    predictions_from_snapshot,
)

labels = load_action_labels("runs/hololens_timeline.json")
report = evaluate_action_timeline(
    predictions_from_snapshot(runtime.snapshot()),   # ingest the clip first
    labels,
    tolerance=2.0,                                   # seconds, inclusive
)

report.passed                     # precision/recall = 1, entity/container ok
report.precision, report.recall
report.mean_timestamp_error       # seconds over matched labels
report.entity_accuracy            # None when no prediction has an entity
report.container_accuracy         # None when neither side names a container
print("\n".join(report.summary_lines()))
```

HoloAssist annotations can be scored without converting them first.
`load_annotation_actions()` reads `ground_truth.actions` (or a bare action
list), takes the first candidate in each row's `actions` list that maps to a
Chronicle primitive (`PICK_UP`, `PLACE` -> `PUT_DOWN`, `INSERT_INTO`,
`REMOVE_FROM`, ...), and skips movement-only rows (`MOVE`/`VISIBLE_MOVING`).
`data/real/holoassist_switch_34s.json` yields 7 scorable labels.

Matching is greedy by nearest `|Δt|` among predictions with the same action
primitive. Wrong-action or out-of-window predictions count as extras (they lower
precision); unmatched labels lower recall. Entity/container strings are compared
case- and separator-insensitively (`keys` matches `the keys`); when only one
side has a container the pair is reported as **not comparable** rather than
wrong. Predictions with `accepted=false` are skipped by default so rejected
actions cannot inflate recall.

```bash
# one-shot score of an exported snapshot
python -m chronicle.semantic.timeline \
  --labels docs/action_timeline.example.json \
  --snapshot runs/chronicle_snapshot.json --tolerance 2.0

# score a predictions file (useful before the runtime exists), or include rejects
python -m chronicle.semantic.timeline \
  --labels docs/action_timeline.example.json \
  --predictions docs/action_predictions.example.json --json

# HoloLens benchmark: export a snapshot, then score HoloAssist annotations
python -m chronicle.cli run --source tabletop \
  --db runs/hololens.db --reset --snapshot-out runs/hololens_snapshot.json
python -m chronicle.semantic.timeline \
  --annotations data/real/holoassist_switch_34s.json \
  --snapshot runs/hololens_snapshot.json --tolerance 2.2 --ignore-entity
```

`--annotations` is mutually exclusive with `--labels`. `--ignore-entity` still
matches actions and timestamps but reports entity/container accuracy as `None`
and drops those checks from the PASS gate; use it when the ground truth and the
runtime detector use different vocabularies (for example HoloAssist `joy-con`
vs. the colour fallback's `cup`/`mug`).

The command exits 0 only when every label is matched, there are no extra
predictions, and (where comparable) entity/container names agree; otherwise it
exits 1 and prints `MISS` / `EXTRA` lines, so CI can gate the "done when" bar.
The bundled demo example matches `PICK_UP keys @ 1.967 s` against a label at
1.9 s (mean error 0.07 s). Snapshot action rows do not currently include a
`container` column, so container accuracy only becomes comparable when scoring
a predictions file (or after Role B adds the field).

**Real HoloAssist timeline (NB-07).** The hand-labelled file is
[`docs/action_timeline.holoassist.json`](action_timeline.holoassist.json) (7
scorable events from the published annotations) and the scored report is
[`docs/action_timeline.holoassist.report.json`](action_timeline.holoassist.report.json):

```bash
python -m chronicle.semantic.timeline \
  --labels docs/action_timeline.holoassist.json \
  --snapshot runs/nb07/hololens_snapshot.json \
  --tolerance 2.2 --ignore-entity
# 1/7 matched · precision 6.3% · recall 14.3% · mean |dt| 0.06 s · FAIL
```

The timestamp is right; the action class is wrong. This run used the fallback
`StubCaptioner` (Florence-2 is not installed), which emits
`"the hand grasps the joy-con"` for every strip, so the rule classifier can
only produce `PICK_UP`. Re-run the same command with Florence-2 or a vision LLM
before treating the result as production accuracy; that work is NB-19 on the
notice board.

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
| `CHRONICLE_ACTION_MIN_CONFIDENCE` | `0.45` | global accept/review floor |
| `CHRONICLE_ACTION_REJECT_AMBIGUOUS` | `true` | reject hedged captions |
| `CHRONICLE_ACTION_THRESHOLDS` | calibrated defaults ([`CALIBRATION.md`](CALIBRATION.md)) | `HOLD=0.62,PICK_UP=0.74,INSERT_INTO=0.8` |
| `CHRONICLE_VERIFICATION_QUEUE_SIZE` | `8` | max clips waiting for the VLM |
| `CHRONICLE_VERIFICATION_QUEUE_POLICY` | `drop_oldest` | `drop_oldest`, `reject` |
| `CHRONICLE_VERIFICATION_DEDUP_WINDOW` | `1024` | clip ids remembered |
| `CHRONICLE_LLM_BACKEND` | `auto` | `auto`, `ollama`, `transformers`, `template`, `none` |
| `CHRONICLE_LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama / HF model id |
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

## 9. Tests and evaluation

```powershell
python -m unittest discover -s tests -v   # 128 model-free tests
python -m pytest                          # 250 collected (dashboard test skips without the ui extra)
python -m chronicle.semantic.demo         # offline end-to-end demo
python -m chronicle.semantic.evaluation   # GraphRAG eval set (or -m chronicle-semantic-eval)
python -m chronicle.semantic.timeline `
  --labels docs/action_timeline.example.json `
  --predictions docs/action_predictions.example.json   # labelled action timeline (section 6)
```

The Role C suite covers:

* VPR create/relocalize/threshold/dedup/persistence;
* temporal strip selection, sizing, dict/PIL inputs, edge cases;
* every strict JSON rejection path, rule classifier variants, LLM repair/fallback;
* graph serialization (NetworkX-like + mapping), citation validation, GraphRAG
  template/LLM/fallback paths;
* the end-to-end facade and telemetry contract;
* Role A integration: `InteractionClip`/`ClipFrame` ingestion and the embedder
  adapter;
* bounded queue: de-duplication, both backpressure policies, FIFO order,
  latency stats, background worker, runtime seam;
* container extraction for `INSERT_INTO` / `REMOVE_FROM`, ambiguity rejection,
  per-action thresholds and threshold suggestion;
* labelled-caption files: JSON list/mapping/`labels` wrapper, JSONL, alias
  action names, invalid-label rejection, CLI `--json`/`--write` output;
* action timeline: tolerance boundaries, nearest-match pairing, extra / missing /
  wrong-action accounting, entity/container aliases, entity-agnostic scoring,
  rejected predictions, snapshot/prediction/annotation loaders and CLI exit
  codes;
* GraphRAG evaluation: 9 cases (location, time, action, temporal before-placement,
  remove, lost, zone-only, unknown), 100 % citation precision,
  hallucinated-timestamp fallback.

The suite has **no model downloads and no network access**.

---

## 10. Known limitations / next steps

* Action classification currently uses Florence-2's `<MORE_DETAILED_CAPTION>`
  followed by a text-only JSON classifier. Swapping in a vision-language model
  is a matter of replacing `ActionVerifier.chat_client` with a vision client;
  the schema and fallbacks stay unchanged.
* Calibration now ships with real HoloAssist labels
  (`docs/calibration_labels.holoassist.json`, 19 labels, precision 1.00 /
  recall 1.00) and the resulting per-action defaults. NB-18 closed the pre-grasp
  `PICK_UP` false positive with negation-aware rule matching. The
  action-timeline labels are committed (`docs/action_timeline.holoassist.json`)
  but the real score is still 1/7 because Florence-2 is unavailable; NB-19
  reruns with real VLM captions before the Role C “done when” line in
  `TEAM_STATUS.md` is ticked.
* The deterministic embedder is intentionally not CLIP-quality. It exists for
  offline demos/tests; production must install `requirements-semantic-models.txt`.
* Zone prototypes are in-memory + JSON by default. Role B can persist
  `match.as_event()` / `Zone.to_dict()` in SQLite if the background graph needs
  a single source of truth.
* GraphRAG grounds timestamps and zones, but does not yet re-rank candidate
  entities with CLIP text-to-node similarity; Role B currently supplies the
  entity context.
* "Before" questions can only use facts inside the serialized 2-hop context. If
  Role B's retrieval window drops the earlier location, GraphRAG returns the
  ungrounded "no earlier location" fallback; widen `max_facts`/`hops` rather
  than inventing a location.
