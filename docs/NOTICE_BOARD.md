# Chronicle Notice Board

**Last synced:** 2026-10-27
**Source of truth for scope:** `docs/chronicle-spec-condensed.tex` + current code  
**Detailed status / history:** [`TEAM_STATUS.md`](TEAM_STATUS.md)

This is the working board for the team. Pick one `READY` task, change its owner
to your name, move it to `IN PROGRESS`, and open a small PR. Keep one task per
PR so parallel work does not collide.

## How to use this board

1. Pick the highest-priority `READY` task that does not overlap your current files.
2. Claim it by editing the task card:
   - `Status: READY` → `IN PROGRESS (your name)`
   - `Owner: unclaimed` → your name
3. Work only inside the listed files unless the task says otherwise.
4. Run the verification commands in the task before marking it `DONE`.
5. Move the card to **Done** with the commit hash.

**Priority meanings**

| Priority | Meaning |
| --- | --- |
| `P0` | Demo blocker / correctness bug. Do before judges or recording. |
| `P1` | Important quality or feature gap. Do after P0. |
| `P2` | Polish / stretch. Safe to leave. |

**Suggested work lanes** (to avoid conflicts)

| Lane | Owns | Avoid touching |
| --- | --- | --- |
| Perception | `chronicle/perception/` | graph, UI |
| Graph / state | `chronicle/graph/`, `chronicle/storage/`, `chronicle/state_machine.py` | UI |
| Semantic | `chronicle/semantic/` | graph internals |
| Dashboard / UI | `chronicle/ui/`, `dashboard/` | graph internals |
| Demo / docs | `docs/`, `chronicle.sh`, `run_dashboard.py` | runtime code |

> **Coordination rule:** `chronicle/integration.py` is shared by every lane.
> Announce edits there before starting. Additive changes only.

---

## Board summary

| ID | Pri | Task | Lane | Status | Owner |
| --- | --- | --- | --- | --- | --- |
| NB-01 | P0 | Clickable citations seek the video | UI | READY | unclaimed |
| NB-02 | P0 | Show the evidence frame under each answer | UI | READY | unclaimed |
| NB-03 | P0 | Grounding / rejection provenance in the query panel | UI | READY | unclaimed |
| NB-04 | P0 | Demo preflight script and checklist | Ops | DONE | opencode (AI) |
| NB-05 | P0 | Verify every bundled source end to end | QA | DONE | opencode (AI) |
| NB-06 | P0 | Real-clip evidence images for demo answers | Perception | DONE | opencode (AI) |
| NB-07 | P1 | Label the HoloLens action timeline and run the scorer | Semantic | DONE | opencode (AI) |
| NB-19 | P1 | Re-run HoloAssist action timeline with Florence-2 captions | Semantic | READY | unclaimed |
| NB-08 | P1 | Calibrate action thresholds against real labels | Semantic | DONE | opencode (AI) |
| NB-09 | P1 | Review drawer for rejected / ambiguous actions | UI | READY | unclaimed |
| NB-10 | P1 | Tune consolidation on the living-room clip | Graph | READY | unclaimed |
| NB-11 | P1 | Reduce per-frame support/SQLite write cost | Graph | READY | unclaimed |
| NB-12 | P1 | Add appearance attributes to the spec + evaluation | Docs / Semantic | READY | unclaimed |
| NB-13 | P2 | Retire or clearly separate the duplicate dashboard | UI / Docs | READY | unclaimed |
| NB-14 | P2 | Search and filter actions / transitions | UI | READY | unclaimed |
| NB-15 | P2 | Colour filter dropdown for entities | UI | READY | unclaimed |
| NB-16 | P2 | Export evidence frames with the snapshot | UI / Integration | READY | unclaimed |
| NB-17 | P2 | Responsive and accessibility pass | UI | READY | unclaimed |
| NB-18 | P1 | Reduce the pre-grasp PICK_UP false positive | Semantic | DONE | opencode (AI) |

---

## P0 — demo blockers

### NB-01 · Clickable citations seek the video
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** the spec promises “the citation is clickable and seeks the video to
  16.4 s”. The operator dashboard currently prints citation times as text only.
- **Scope:**
  - Render each citation in `chronicle/ui/app.py` as a small button/chip.
  - Clicking it seeks the `ui.video` player to `citation["timestamp"]` and
    optionally pauses.
  - Keep the metadata line for debugging.
- **Files:** `chronicle/ui/app.py`, `tests/test_dashboard_smoke.py`
- **Acceptance:** clicking a citation changes the video position; a headless
  test asserts the handler exists; no layout regression at 1280/1440 px.
- **Spec:** §Query; §Ten-minute demo script step 2.

### NB-02 · Show the evidence frame under each answer
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** `ChronicleRuntime._answer_frame()` already produces a supporting
  still frame, but the dashboard ignores `RuntimeAnswer.frame`.
- **Scope:**
  - Display `answer.frame["url"]` in the query card.
  - Show the frame kind and timestamp.
  - Hide gracefully when `frame` is `None`.
- **Files:** `chronicle/ui/app.py`, `tests/test_dashboard_smoke.py`
- **Acceptance:** grounded answers with a frame render an image preview;
  missing frames leave the panel usable; no broken image icon on failure.
- **Spec:** §Query, evidence frame paragraph.

### NB-03 · Grounding / rejection provenance in the query panel
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** `RuntimeAnswer` exposes `grounded`, `source`, `method` and citations,
  but the UI only shows the answer text and a metadata string. Judges cannot
  see when an answer is ungrounded or came from a fallback.
- **Scope:**
  - Add a grounded / ungrounded / fallback badge.
  - Show `source` (`template`, `llm`, `graph`) and citation count.
  - If `grounded` is false, style the answer distinctly and explain why.
- **Files:** `chronicle/ui/app.py`, `chronicle/ui/graph_view.py` only if helpers move
- **Acceptance:** a known unknown query shows an ungrounded state; a template
  answer shows `template`; no hallucinated timestamp is presented as grounded.
- **Spec:** §Query step 4.

---

## P1 — quality and correctness

### NB-19 · Re-run HoloAssist action timeline with Florence-2 captions
- **Status:** READY · **Owner:** unclaimed · **Lane:** Semantic
- **Why:** NB-07 labelled the timeline and scored the production run. The
  timestamps are excellent (mean |Δt| 0.06 s) but every prediction is
  `PICK_UP` because Florence-2 is unavailable and `StubCaptioner` emits
  `"the hand grasps the joy-con"` for every strip.
- **Scope:** install/point at Florence-2 (or a vision LLM), run
  `chronicle run --source tabletop` with real captions, then rescore:
  ```bash
  python -m chronicle.semantic.timeline \
    --labels docs/action_timeline.holoassist.json \
    --snapshot runs/nb07/hololens_snapshot.json \
    --tolerance 2.2 --ignore-entity
  ```
- **Files:** `chronicle/semantic/florence.py`, `docs/action_timeline.holoassist.report.json`,
  `docs/PERCEPTION.md`, `docs/TEAM_STATUS.md`
- **Acceptance:** at least `PUT_DOWN`, `INSERT_INTO` and `REMOVE_FROM` appear
  in predictions; document the new precision/recall and any threshold change.
  If the VLM is still unavailable, record the blocker and stop claiming action
  accuracy.
- **Spec:** §Quality; §Ten-minute demo script step 5.

### NB-09 · Review drawer for rejected / ambiguous actions
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** rejected actions are currently only visible in telemetry. The
  suggestions note asks for a review queue with strip preview.
- **Scope:** list rejected actions with reason, confidence, caption and clip
  reference; optionally preview the strip.
- **Files:** `chronicle/ui/app.py`, `docs/SUGGESTIONS.txt` reference only
- **Acceptance:** rejected actions are inspectable without reading the log.
- **Spec:** §Quality (verification queue); SUGGESTIONS 5.2.

### NB-10 · Tune consolidation on the living-room clip
- **Status:** READY · **Owner:** unclaimed · **Lane:** Graph
- **Why:** the “done when” bar for Role B is one anchor per physical sofa after
  replaying the living-room clip.
- **Scope:** run the clip, inspect `memory.stats()["merges"]`, tune association
  and merge gates if duplicates remain; add a regression fixture.
- **Files:** `chronicle/graph/consolidation.py`,
  `chronicle/graph/furniture.py`, `tests/test_consolidation.py`
- **Acceptance:** exactly one anchor per real sofa and no merged co-visible
  instances; tests pass.
- **Spec:** §Data model; TEAM_STATUS Role B “Done when”.

### NB-11 · Reduce per-frame support / SQLite write cost
- **Status:** READY · **Owner:** unclaimed · **Lane:** Graph
- **Why:** TEAM_STATUS lists repeated support updates and per-frame SQLite
  writes as the remaining runtime-efficiency work.
- **Scope:** deduplicate unchanged support writes, batch entity/observation
  writes, and keep snapshots cheap enough for 30 fps.
- **Files:** `chronicle/graph/memory.py`,
  `chronicle/storage/sqlite_store.py`, benchmark script
- **Acceptance:** measurable write reduction on the walking clip with identical
  snapshot content and passing tests.
- **Spec:** §Storage; TEAM_STATUS “Runtime efficiency”.

### NB-12 · Add appearance attributes to the spec + evaluation
- **Status:** READY · **Owner:** unclaimed · **Lane:** Docs / Semantic
- **Why:** colour/shape/size/texture attributes are implemented and used by the
  UI and GraphRAG, but the LaTeX spec and evaluation set do not mention them.
- **Scope:**
  - Add an “Object appearance attributes” subsection to
    `docs/chronicle-spec-condensed.tex` (formula + thresholds).
  - Add colour/size/describe cases to the GraphRAG evaluation set.
  - Rebuild the PDF with `docs/build-spec.sh`.
- **Files:** `docs/chronicle-spec-condensed.tex`,
  `chronicle/semantic/evaluation.py`, `docs/chronicle-spec-condensed.pdf`
- **Acceptance:** spec builds; evaluation passes with the new cases and reports
  citation precision.
- **Spec:** §Perception; new feature in `chronicle/perception/attributes.py`.

---

## P2 — polish

### NB-13 · Retire or clearly separate the duplicate dashboard
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI / Docs
- **Why:** the spec lists both `chronicle serve` (operator dashboard) and
  `run_dashboard.py` (mission-control dashboard). Keeping both without labels
  confuses teammates and judges.
- **Scope:** document which is canonical for the demo; either retire the other
  or label it clearly and document its unique features.
- **Files:** `README.md`, `docs/INTEGRATION.md`, `run_dashboard.py`
- **Acceptance:** one obvious entry point in the README and demo docs.
- **Spec:** §Interfaces and operations.

### NB-14 · Search and filter actions / transitions
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** tables only paginate; real episodes can have hundreds of rows.
- **Scope:** add text filter and action-type/state filters to the Actions and
  Timeline tabs.
- **Files:** `chronicle/ui/app.py`
- **Acceptance:** filters work without breaking live refresh.
- **Spec:** §Interfaces, dashboard purpose.

### NB-15 · Colour filter dropdown for entities
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** the entity filter is free-text; a colour dropdown makes the new
  appearance data discoverable for judges.
- **Scope:** add a colour selector populated from current entity attributes,
  combined with the existing text filter.
- **Files:** `chronicle/ui/app.py`
- **Acceptance:** selecting “red” shows only red entities; clearing restores all.
- **Spec:** appearance attributes feature.

### NB-16 · Export evidence frames with the snapshot
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI / Integration
- **Why:** snapshots contain the graph but not the still images that support
  answers, so an imported snapshot cannot show evidence.
- **Scope:** optional `frames` bundle in the export: answer frames keyed by
  entity/time, or a documented manifest with relative paths.
- **Files:** `chronicle/integration.py`, `chronicle/ui/app.py`,
  `chronicle/ui/snapshot_io.py`
- **Acceptance:** exported/imported snapshot can render at least one evidence
  frame without re-ingesting video.
- **Spec:** §Storage “Export is portable”; §Query evidence frame.

### NB-17 · Responsive and accessibility pass
- **Status:** READY · **Owner:** unclaimed · **Lane:** UI
- **Why:** current layout is tuned for 1440 px; judges may use a laptop or
  projector.
- **Scope:** verify 1280/1440/1920 px; keyboard focus for query and tabs;
  sufficient contrast for muted text; labels for icon-only buttons.
- **Files:** `chronicle/ui/app.py`, `chronicle/ui/graph_view.py`
- **Acceptance:** no clipped controls at 1280 px; keyboard-only query works;
  icon buttons have tooltips and accessible labels.
- **Spec:** §Interfaces.

---

## Done (recent)

| ID | Task | Done by | Evidence |
| --- | --- | --- | --- |
| NB-D01 | Graph clipped off-canvas / not rendering | orchestrator | `chronicle/ui/graph_view.py`; screenshot verified |
| NB-D02 | Colour + simple object attributes end to end | orchestrator | `tests/test_attributes.py`; 314 tests pass |
| NB-D03 | Appearance in snapshot, CSV, GraphRAG and UI | orchestrator | `docs/INTEGRATION.md`, `docs/PERCEPTION.md` |
| NB-D04 | Dashboard video upload + drag-and-drop | orchestrator | `tests/test_uploads.py` |
| NB-D05 | Snapshot import / export | orchestrator | `tests/test_snapshot_io.py`, `tests/test_dashboard_smoke.py` |
| NB-08 | Calibrate action thresholds against real labels | opencode (AI) | `9ed6735`; `docs/CALIBRATION.md`; 19 labels, precision 1.00 / recall 1.00 |
| NB-04 | Demo preflight script and checklist | opencode (AI) | `3b74ef5`; `scripts/demo_preflight.py`; `docs/DEMO_PREFLIGHT.md` |
| NB-05 | Verify every bundled source end to end | opencode (AI) | `f8fcd0e`; `docs/PERCEPTION.md` NB-05 tables; 5/5 sources 9/9 checks |
| NB-06 | Real-clip evidence images for demo answers | opencode (AI) | `f967a71`; `data/*/evidence/`; `scripts/build_evidence_fixtures.py` |
| NB-07 | Label the HoloLens action timeline and run the scorer | opencode (AI) | `68b1432`; `docs/action_timeline.holoassist.json`; 1/7, 6.3% precision |
| NB-18 | Reduce the pre-grasp PICK_UP false positive | opencode (AI) | `dfcd4d7`; negation-aware rule matching; precision 1.00 / recall 1.00 |

### NB-18 details

- **Branch:** `task/nb-18-pickup-negation` · **Commit:** `dfcd4d7`
- **Fix:** `RuleBasedActionClassifier` rejects negated action verbs before the
  positive patterns, so a pre-grasp "reaches … does not grasp" caption becomes
  `OTHER`; true pickups still match.
- **Result:** real-label calibration is now 19 labels, 14 accepted, 5 correct
  rejections, **precision 1.000 / recall 1.000**; thresholds unchanged
  (`PICK_UP` 0.74). This is a phrasing fix, not a cutoff change.
- **Docs/tests:** `docs/CALIBRATION.md`, `docs/ROLE_C.md`, report JSON,
  `tests/test_actions.py` (negation + true action with "without dropping") and
  `tests/test_calibration.py` (PICK_UP precision/recall).

### NB-07 details

- **Branch:** `task/nb-07-holoassist-timeline` · **Commit:** `68b1432`
- **Labels:** `docs/action_timeline.holoassist.json` (7 scorable events from
  the published HoloAssist annotations, with source event strings).
- **Run:** full 1019-frame tabletop clip with YOLO-World + MediaPipe + OpenCLIP
  (4 entities, 16 actions, 16 clips); snapshot and report are local under
  `runs/nb07/` and `docs/action_timeline.holoassist.report.json`.
- **Score:** 1/7 matched, precision 6.3%, recall 14.3%, mean |Δt| 0.06 s,
  **FAIL**.
- **Root cause:** no Florence-2 → `StubCaptioner` returns one generic
  `PICK_UP` caption for every strip. Timestamp extraction is excellent; action
  classification is the gap.
- **Follow-up:** **NB-19** reruns with real VLM captions.

### NB-06 details

- **Branch:** `task/nb-06-evidence-fixtures` · **Commit:** `f967a71`
- **Delivered:** `scripts/build_evidence_fixtures.py`, stored
  `evidence/index.jsonl` + images for all five sources, stored-evidence
  selection in `chronicle/frames.py`, and `_answer_frame()` wiring in
  `chronicle/integration.py`.
- **Selection order:** stored strip/crop/zone image first, ranked by kind,
  entity/label, citation time and playhead; video-frame extraction only when no
  stored image fits (uploaded clips still use the fallback).
- **Encoding:** the chosen stored image is returned as a
  `data:image/...;base64,...` URL, so the CLI and future UI need no media route.
- **Verified:** `where are my keys?` → keys `PICK_UP` strip at 11.15 s;
  `where is the joy-con?` → Joy-Con strip at 6.55 s; `what did I see?` → newest
  Joy-Con strip. Preflight now reports **5/5 sources have stored evidence**.
- **Coverage:** `tests/test_query_frames.py` (index loading, missing files,
  label ranking, data URLs, runtime preference, bundled indexes).
- **Spec:** §Query evidence frame; §Demo pitfalls (iv).

### NB-05 details

- **Branch:** `task/nb-05-source-verification` · **Commit:** `f8fcd0e`
- **Full-clip fallback:** all five sources completed with 9/9 snapshot
  invariants (`synthetic` 360 frames, `tabletop` 1019, `walking` 600, `cook`
  450, `play` 450).
- **Bounded production:** 180 frames with YOLO-World + MediaPipe + OpenCLIP on
  all four real clips; all 9/9 (`tabletop` 4 actions, `cook` 1, `play` 0,
  `walking` 0).
- **Dashboard:** headless `/source-video` check returns HTTP 200 + `video/*` for
  every registered clip; the interactive dropdown/player remains the later UI
  pass.
- **Acceptance:** every source passes; the only documented gap is Aria action
  accuracy (no labels), not ingestion.
- **Artifacts:** `runs/benchmark/nb05-*.json` (local, not committed) and the
  tables in `docs/PERCEPTION.md`.

### NB-04 details

- **Branch:** `task/nb-04-demo-preflight` · **Commit:** `3b74ef5`
- **Delivered:** `scripts/demo_preflight.py`, `./chronicle.sh preflight`,
  `docs/DEMO_PREFLIGHT.md`, and `tests/test_demo_preflight.py`.
- **Checks:** Python/venv, data dir, dashboard deps, every source's first +
  middle frame, SQLite WAL/`quick_check`/checkpoint, detector/hands/embedder on
  a real frame, stored evidence or fallback extraction, Ollama/model warm state,
  `pytest`, and 30-frame synthetic ingestion.
- **Usability:** silent colour fallbacks become `WARN`; `--strict` promotes
  warnings; `--json` is CI-friendly; `--warm-llm` loads Qwen; `--skip-*` makes
  a fast venue check possible.
- **Acceptance:** one command prints a `PASS`/`WARN`/`FAIL` table with exact
  next commands. First full run on this machine: 15 PASS, 1 WARN (no stored
  evidence yet; video-frame fallback works), 0 FAIL.
- **Result:** `READY` on the current tree.

### NB-08 details

- **Branch:** `task/nb-08-real-calibration` · **Merged:** `9ed6735..d5870b3` on `main`
- **Why:** shipped thresholds came from a 16-caption seed set.
- **Delivered:** `docs/calibration_labels.holoassist.json` (19 traceable labels
  from the published HoloAssist timeline), `docs/calibration.holoassist.report.json`,
  `docs/CALIBRATION.md`, calibrated defaults in `actions.py`, and fixes for the
  two regex gaps the real labels exposed ("slide …" → `PUSH`, "withdraw …" →
  `REMOVE_FROM`).
- **Acceptance:** precision target 0.9 met (now 1.000 after NB-18), threshold
  source documented, model-free regression tests added.
- **Known follow-up:** the pre-grasp "reaches toward …" caption is still a
  `PICK_UP` false positive. Raising the threshold would reject true pickups, so
  it needs a classifier/phrasing fix rather than a threshold change.

---

## Verification commands

Run before marking any task DONE:

```bash
.venv/bin/pytest -q
.venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check chronicle tests
.venv/bin/python -m chronicle.semantic.evaluation
```

For UI tasks also:

```bash
.venv/bin/python -m pytest -q tests/test_dashboard_smoke.py
./chronicle.sh restart
```

---

## Claim template

Copy this into the task card when you start:

```text
Status: IN PROGRESS (<name>)
Owner: <name>
Started: <date>
Branch: <branch>
Notes:
- ...
```

## AI-agent handoff template

When handing a task to another agent, paste the task card and this guardrail:

```text
Repository: /home/g1/projects/chronicle
Task: <NB-ID and title>
Read: docs/NOTICE_BOARD.md, docs/TEAM_STATUS.md, and the task's listed files.
Scope: only edit the files listed in the task card; all other changes must be additive.
Before finishing: run `.venv/bin/pytest -q`, `.venv/bin/python -m unittest discover -s tests`,
and `.venv/bin/ruff check chronicle tests`.
Report: commit hash, files changed, tests run, and any remaining risk.
Do not revert another teammate's uncommitted work.
```
