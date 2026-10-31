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
