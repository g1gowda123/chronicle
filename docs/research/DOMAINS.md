# Held-out egocentric domains

SPORE's development set already contains four real clips:

| clip | domain | QA |
|---|---|---|
| `aria_cook_45s.mp4` | kitchen cooking | 10 hand-written temporal questions |
| `aria_play_45s.mp4` | living room / play | 4 object-location questions |
| `aria_walk_60s.mp4` | outdoor walk | none (scene-recall probe) |
| `holoassist_switch_34s.mp4` | tabletop assembly | 8 hand-written questions |

The risk is that a perception layer tuned on cooking plus Joy-Con assembly does
not generalise. The Aria Gen 2 Pilot release adds eating, cleaning, a second
social scene, and outdoor walking. The full videos stay local; only the
manifest, fetch/trim helper, and derived reports are committed.

## Manifest

`data/aria_gen2pilot.json` records the four held-out windows:

| name | domain | window | why |
|---|---|---|---|
| `eat_0` | eating | 30–75 s | three people at a dining table, hands, plates, cups |
| `clean_0` | cleaning | 40–85 s | living room sorting/cleaning, cluttered coffee table |
| `play_3` | social | 40–85 s | two people talking over a board game |
| `walk_0` | outdoor | 40–85 s | campus path, building, hedges, hydrant, strong sun |

The windows were chosen from sampled frames before any SPORE run. Scene
metadata (2D/3D boxes, object categories, hand-object interactions) is
available from the same API and can provide provenance for future questions.

## Fetch

The Dataset Explorer API returns short-lived signed URLs. The preview files
are ~483 MB each with `moov` at the end, so remote `ffmpeg -ss` seeking is
unreliable. Download once into a cache, then trim locally:

```bash
python scripts/fetch_aria_clip.py --sequence eat_0 --dry-run
python scripts/fetch_aria_clip.py --sequence eat_0
python scripts/fetch_aria_clip.py --sequence clean_0 --sequence play_3
```

The trimmed clip is written to the path in the manifest under `data/footage/`
(ignored by git). Full previews stay in
`~/.cache/chronicle/aria_gen2pilot/`.

## Evaluate

`scripts/spore_autoeval.py` is fully self-supervised: it tracks dense
YOLO-World detections and asks only about objects seen repeatedly, so no human
picks the question, timestamp, or label. `scripts/spore_domains.py` runs that
evaluator across the manifest and aggregates by domain:

```bash
# inspect the plan without touching the GPU
.venv/bin/python scripts/spore_domains.py --dry-run

# run whatever clips are present; skip missing ones
.venv/bin/python scripts/spore_domains.py --allow-missing

# one domain
.venv/bin/python scripts/spore_domains.py --sequence eat_0
```

Use a Python environment with Ultralytics, MediaPipe, and OpenCLIP installed.
On this machine that is `/home/g1/projects/chronicle/.venv/bin/python`; the
SPORE project venv is not sufficient for real footage.

The older Cook/Play/Walk/Switch demo clips are Git LFS pointers in a plain
checkout. Restore them without installing `git-lfs`:

```bash
python scripts/fetch_lfs_clips.py --dry-run
python scripts/fetch_lfs_clips.py --all
```

The fetcher verifies the pointer's SHA-256 and byte size. The four new Aria
clips above are fetched separately with `scripts/fetch_aria_clip.py`.

Results go to `runs/spore_domains/` (ignored): per-clip JSON plus
`summary.json` / `summary.md`. Metrics are semantic accuracy, localization at
IoU .30, mean IoU, detector calls per query, and dense calls. Because the
questions come from the dense detector, they measure memory/retention and
instance selection, not open-world recall of undetected objects.

## First held-out run (2026-10-31)

Each clip was trimmed to a 45 s window from the manifest, sampled at 4 fps
(150 dense frames), and evaluated with 12 automatically generated questions.
The detector budget was 8 capsules per query.

| domain | clip | questions | semantic | loc@.30 | mean IoU | calls/query | dense calls |
|---|---|---:|---:|---:|---:|---:|---:|
| eating | `aria_eat_0_45s.mp4` | 12 | 1.000 | .917 | .711 | 3.33 | 150 |
| cleaning | `aria_clean_0_45s.mp4` | 12 | 1.000 | .833 | .678 | 4.50 | 150 |
| social | `aria_play_3_45s.mp4` | 12 | 1.000 | .917 | .719 | 4.75 | 150 |
| outdoor | `aria_walk_0_45s.mp4` | 12 | 1.000 | .833 | .617 | 5.25 | 150 |
| **all four** | | **48** | **1.000** | **.875** | **.681** | **4.46** | **150/clip** |

The outdoor clip includes seven `person` questions. Before the run, `person`
was a stopword in the question parser, so those queries returned no bindings;
the parser now treats a bare “person/people” location question as a people
query, and all seven are grounded. This is a general capability, not a
clip-specific rule.

These numbers are deliberately **not** a headline accuracy claim:

- Dense YOLO-World supplies both the questions and the reference boxes. A
  semantic score of 1.000 means SPORE retained and re-bound the detector's
  labels, not that it discovered objects the detector missed.
- Localization is against the dense track box at the question timestamp. It
  measures instance selection and retention, not semantic correctness against
  human annotation.
- Support relations are noisy. Some clean/play answers say “sofa on the
  basket” or “chair on the basket” because the support detector chose a nearby
  box; the current metric does not score relations.
- The four clips are short windows from one capture rig. Report them as
  held-out development domains, not a benchmark.

## Human QA check (eat_0)

The automatic questions above are generated from the same dense detector that
supplies their reference boxes. To get one independent check, six questions
were written from visual inspection of `aria_eat_0_45s` and committed
(`data/qa/aria_eat_0_45s.json`, `ec501bb`) **before** the first SPORE run on
the clip. The clip has three people at a dining table, plates, pink/orange
cups, a wall-mounted TV, and a window with blinds.

| question | expected | SPORE answer | IoU | outcome |
|---|---|---|---|---:|---|
| Where is the table? | table | `table at 20.0s` | .807 | pass |
| Where is the cup? | table | `cup on the table (countertop) at 19.2s` | .772 | pass |
| Where is the plate? | table | `plate on the desk (work table) at 20.0s` | .238 | wrong support; semantic substring accepts “table” |
| What colour is the cup? | red/orange | `cup is red at 19.2s` | .772 | pass |
| Where is the TV? | wall | `tv on the desk (work table) at 20.0s` | .609 | detected, support wrong |
| Where is the window? | wall | `window on the table (countertop) at 19.6s` | .000 | CLIP fallback, no dense target |

Aggregate: lenient semantic **.667 (4/6)**, human-judged **.50 (3/6)**,
loc@.30/.50 **.80**, mean IoU **.64**, **4.5 calls/query** (27 total vs 150
dense), 30.6 s ingest, 5.24 MB in-memory store, 1.34 GB peak VRAM.

This is a transfer check, not a benchmark estimate. Object memory transfers to
the new room (the TV is detected at IoU .609), but the support vocabulary does
not contain `wall`, so the TV relation is forced onto a desk; the window is a
CLIP-fallback hallucination with no detector target. The plate answer exposes
a metric weakness: “table” is a substring of “work table,” so the lenient
matcher accepts a wrong support. Report both the lenient and human-judged
scores, and do not tune these questions after the run. A wall-support change
must be evaluated on a different clip before it is claimed as a gain.

### Human QA check (play_3)

A third six-question set was written for the held-out `aria_play_3_45s`
social/board-game clip and committed (`3f74c24`) before the surface-head run.
It probes sofa/chair→floor, TV→shelf, book→table, window→wall, and sofa colour.
The TV sits on a media console; the row deliberately checks that the wall rule
does not over-fire.

| question | expected | baseline | surface head | outcome |
|---|---|---|---|---|
| Where is the sofa? | floor | `sofa on desk` | `sofa on floor` | fixed |
| Where is the chair? | floor | `chair on desk` | `chair on floor` | fixed |
| Where is the TV? | shelf | `tv on shelf` | `tv on wall` | **regressed** |
| Where is the book? | table | `book on desk` | `book on desk` | not fixed |
| Where is the window? | wall | `window on desk` | `window on wall` | fixed |
| What colour is the sofa? | grey | orange | orange | not fixed |

Lenient semantic **2/6→4/6**; human-judged **1/6→3/6**. The TV is truncated at
the left frame edge, its black console is not segmented, and the wall fraction
dominates; the game box’s table fraction is only .045. These are limits of the
dense surface head, not rows to tune. The original four-question Play clip is
unchanged by the head.

### Human QA check (walk_0)

A five-question set was written for the held-out outdoor `aria_walk_0_45s`
clip and committed (`2b61849`) before its first run. It probes the person,
shirt colour, first appearance, people count, and car. The default engine
abstains on all five (**0/5 semantic**); the car is detected at 8.0 s (IoU
.661) but its expected support (`parking lot`/`road`) is outside the indoor
support ontology, and the parser extracts `person's shirt` and `first see` as
object terms.

| question | expected | baseline | post-hoc fallback + road |
|---|---|---|---|
| Where is the person? | parking lot/road | abstain | `person on the road at 24.0s` |
| What colour is the person's shirt? | green/teal | abstain | `person is blue at 24.0s` (wrong) |
| When did I first see the person? | person | abstain | `first interaction with person at 0.8s` (wrong time) |
| How many people did I see? | 3 | abstain | `2 person` |
| Where is the car? | parking lot/road | abstain | `car on the road at 8.0s` |

The post-hoc path combines opt-in `--context-fallback`, `--surface-support`
with a road/sidewalk rule, and parser fixes. It scores **2/5 semantic**
(person and car) with loc@.30/.50 .75/.75, IoU .614, and 8.4 calls/query.
Because the parser and outdoor rule were written after seeing these questions,
this is a **post-hoc diagnostic, not a clean held-out estimate**; the first
appearance (0.8 s), shirt colour (blue), and count (2) remain wrong. Do not
tune further on this clip; validate on a second outdoor clip.

### Wall-support probe (negative)

A second six-question human set was written for `aria_clean_0_45s` (living
room: TV, window, sofa, table, chair) and committed as `c85be76` before the
change. Baseline: semantic **0/6**; TV supported by a shelf, sofa/chair by a
desk, window abstained. Adding `wall` to `SporeConfig.support_labels` gave
only **2/6** (sofa and chair moved to `floor`), while TV still used the shelf
and window still abstained. It also regressed Cook **.70→.60**, IoU
**.826→.792**, and cup count **2→1**. Reverted. This is evidence that support
selection, not just the support vocabulary, is the bottleneck.

## Comparison with the original `main` perception path

The same human-written QA sets were run with SPORE on the real clips recovered
from Git LFS. The `main` numbers are the research-branch results reported
before SPORE (Cook 8/10, Switch 7/8); the Play set has only four questions.

| clip | main | SPORE | SPORE calls/query | dense calls | capsule store |
|---|---:|---:|---:|---:|---:|
| Cook | 8/10 | 7/10 | 7.9 | 150 | 6.46 MB |
| Switch | 7/8 | 7/8 | 7.38 | 128 | 4.69 MB |
| Play | n/a | 4/4 | 5.0 | 150 | 7.74 MB |

SPORE is **not an accuracy improvement** over `main` on this comparison. It is
one question worse on Cook (cup support and plate destination resolve to the
shelf), equal on Switch, and perfect on the tiny four-question Play set. Its
measured advantage is compute and storage: roughly 5–8 detector calls per
query instead of one call per sampled frame (128–150), with 4.7–7.7 MB capsule
stores. Ingest is not free: SPORE spends ~19–27 s building the store per 34–45 s
clip, while the dense pass takes 5–6 s.

The Cook failures are informative rather than a reason to tune on Cook: the
support classifier confuses a shelf with the countertop, and the plate
destination tracker follows the plate to the shelf. Those are general
support/track problems, not Aria-specific quirks.

## Independent annotation check (not yet valid)

`scripts/aria_scene_eval.py` asks questions selected from Aria's scene
annotations instead of dense YOLO tracks. On the three clips with downloaded
metadata it scored 33 questions at semantic .848, loc@.30 .061, mean IoU .078,
and center .212. Those numbers are **not** an accuracy verdict: the released
Aria 2D boxes are projections of 3D scene cuboids, the boxes are coarse, and
the `214-1` stream/camera alignment with the preview is unverified. Overlay
checks show a `table` box over the TV/window and a `sofa` box covering the rug
and coffee table. Keep this path as an alignment diagnostic; do not quote it as
SPORE localization accuracy.

## Reading the result

- Report domains separately. A single micro-average over clips with different
  vocabularies hides failure.
- `walk_0` has almost no manipulable objects. It tests people, vehicles,
  street furniture, scene retention, and the detector-free scene path.
- The `clean_0` and `play_3` windows overlap in room appearance. Treat their
  results as social/indoor evidence, not two independent households.
- These are four short windows from one capture rig, not a benchmark claim.
  The paper should call them held-out development domains.

## License

Aria Gen 2 Pilot is **CC BY-NC 4.0**, research/demo use only. Do not
redistribute the downloaded videos. Cite the Aria Gen 2 Pilot dataset and keep
the source URL in derived reports.
