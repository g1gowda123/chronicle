# Agent board — parallel work claims

**Updated:** 2026-10-27
**Branch:** `experiment/video-free-memory`
**Worktree:** `/home/g1/projects/chronicle-vfm`
**Main repo:** `/home/g1/projects/chronicle` — read-only for this experiment.

This board is the coordination point when more than one agent works on the handoff at the same
time. Update your row **before** editing code and again when you finish. Keep one task per file
area where possible.

## Status legend

- `READY` — unclaimed, safe to pick up.
- `IN PROGRESS` — claimed; do not edit the listed files without talking to the owner.
- `BLOCKED` — waiting on a dependency or a measurement.
- `DONE` — code/tests/docs landed; include the commit hash.

## Live presence

Agents: post one line here when you start and when you stop. Timestamps are IST.

| Agent | Present since | Working on | Files locked | Last update |
| --- | --- | --- | --- | --- |
| opencode (DeepSeek-V4.1-Flash) | 2026-10-27 03:27 IST | P2-PACK finish + verification, then P1-COUNTS research | `prototype/memory_pack.py`, `prototype/nlq.py`, `prototype/studio_app.py`, `tests/test_pack.py`, `README.md` | 03:27 IST |
| opencode (AI) | 2026-10-27 ~03:00 IST | P0-VFM-UI main dashboard integration | `chronicle/vfm/`, `chronicle/ui/app.py`, `chronicle/ui/uploads.py`, `chronicle/cli.py`, `tests/test_vfm.py`, `tests/test_dashboard_smoke.py` | 03:00 IST |

> Presence is informational, not a lock. Check the **Claims** table before editing shared files.
> If you see a stale row (no update for >45 min), announce it here before taking the files over.

## Claims

| ID | Pri | Task | Files owned | Status | Owner |
| --- | --- | --- | --- | --- | --- |
| P0-VFM-UI | P0 | Wire VFM compile/export/ask into the existing dashboard | `chronicle/vfm/`, `chronicle/ui/app.py`, `chronicle/ui/uploads.py`, `chronicle/cli.py`, `tests/test_vfm.py`, `tests/test_dashboard_smoke.py` | IN PROGRESS | opencode (AI) |
| P1-COUNTS | P1 | Physical object counts / track fragmentation | `prototype/`, `benchmark/` | READY | unclaimed |
| P1-EXTERNAL | P1 | External benchmark (EgoSchema / Ego4D NLQ / CapMem) | `prototype/`, `benchmark/`, `docs/RESEARCH.md` | READY | unclaimed |
| P1-GEMINI | P1 | Paid/faster online VLM baseline | `prototype/online_baseline.py` | READY | unclaimed |
| P1-FLORENCE | P1 | Florence-2 vs Qwen2.5-VL A/B on 30 s windows | `prototype/`, `docs/JOURNAL.md` | READY | unclaimed |
| P2-RETENTION | P2 | Snapshot dedup + 24 h / 7 day retention test | `prototype/retention.py`, `tests/` | READY | unclaimed |
| P2-PACK | P2 | Token-budget selector (2k / 4k / 8k) | `prototype/memory_pack.py`, `prototype/studio_app.py`, `tests/test_pack.py` | DONE (see journal Session 7) | opencode/DeepSeek-V4.1-Flash |
| QUICK-WINDOW | Quick | Caption-window fine-tune (10/15/20/30/45/60 s) | `prototype/caption_ledger.py`, `docs/JOURNAL.md` | READY | unclaimed |

## Rules for parallel agents

1. **One owner per file area.** If you need a shared file, add an entry here first.
2. **Do not edit `dashboard/`** from this branch. Main dashboard work belongs on `main` and the
   handoff explicitly warns that it may be dirty.
3. **Do not replace the existing upload button, player, progress or Export menu.** Add VFM behind
   them.
4. **Do not make video deletion the default.** `--delete-video` is opt-in and only applies to
   files under `<data-dir>/uploads/`.
5. Run before claiming done:
   ```bash
   cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
   /home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests
   cd /home/g1/projects/chronicle-vfm
   /home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests/test_vfm.py tests/test_dashboard_smoke.py
   git diff --check
   ```
6. Append a dated entry to `docs/JOURNAL.md` with what changed, what was measured, and what is
   still open. Update `HANDOFF.md` "What still needs doing" if you close or change a task.
