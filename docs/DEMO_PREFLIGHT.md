# Demo preflight

Run this **before** the ten-minute demo, not during it. One command checks the
stage pitfalls the spec calls out: missing clips, a locked SQLite WAL, a silent
fallback to the colour detector, a cold Qwen model, missing evidence images,
and a broken test suite or ingestion run.

```bash
./chronicle.sh preflight
```

The default run checks every bundled source, loads the configured production
backends once, probes Ollama, runs the test suite, and ingests 30 synthetic
frames into a throwaway database. It prints a `PASS` / `WARN` / `FAIL` table
and ends with the exact next command.

## What it checks

| Check | What it proves |
| --- | --- |
| `Python` | The venv interpreter and version are the ones the team uses. |
| `Data dir` | `data/` exists and is writable, so SQLite and uploads can be written. |
| `Dashboard` | NiceGUI + FastAPI are importable, so `chronicle serve` can start. |
| `Video: <source>` | Each registered source exists; first and middle frames decode. |
| `SQLite` | Each episode DB opens, is in WAL mode, passes `quick_check`, and can checkpoint. Missing DBs are created in a scratch dir to prove writes work. |
| `Detector` | The configured detector (YOLO-World by default on real clips) loads and runs on a real frame. |
| `Hands` | MediaPipe/hybrid hands load and run. |
| `Embedder` | OpenCLIP loads and returns a 512-D vector. |
| `Evidence` | Stored evidence frames exist, or the video-frame fallback extracts one. |
| `LLM` | Ollama is up, the configured model is present, and whether it is warm or cold. |
| `Tests` | `pytest -q` passes. |
| `Smoke run` | `chronicle run --source synthetic --max-frames 30` passes with a throwaway DB. |

## Reading the result

| Status | Meaning | Action |
| --- | --- | --- |
| `PASS` | Ready for the demo. | None. |
| `WARN` | The demo still runs, but a fallback or cold start will show. | Run the printed next command if the demo depends on that path. |
| `FAIL` | A required path is broken. | Fix before recording or going on stage. |
| `SKIP` | The check was disabled or had no input. | Usually only with explicit flags. |

`WARN` does not change the exit code. Use `--strict` in CI or a final rehearsal
to make warnings fail too.

## Example

```text
Chronicle demo preflight
Data dir: /home/g1/projects/chronicle/data

PASS  Python                  3.12.13 · /home/g1/projects/chronicle/.venv/bin/python
PASS  Data dir                /home/g1/projects/chronicle/data is writable
PASS  Dashboard               NiceGUI + FastAPI importable
PASS  Video: synthetic        chronicle_demo.mp4 · 640×360 · 30.00 fps · 12.0s · first+middle frame ok
PASS  Video: tabletop         holoassist_switch_34s.mp4 · 454×256 · 29.94 fps · 34.0s · first+middle frame ok
PASS  SQLite                  5 database(s) · WAL ok · checkpoint ok
PASS  Detector                yolo-world · 0 detection(s) · 3.1s
PASS  Hands                   hybrid · 0 hand(s) · 0.0s
PASS  Embedder                openclip · shape (512,) · 1.6s
PASS  Evidence                5/5 sources have stored evidence frames
PASS  LLM                     qwen2.5:7b-instruct loaded and warm · Ollama 0.40.10
PASS  Tests                   pytest exit 0 · 3.0s
PASS  Smoke run               synthetic · 30 frames · 117.6 fps · 4 entities · 4 transitions · 0.7s

Summary: 16 PASS · 0 WARN · 0 FAIL · 0 SKIP
Result: READY
Next:
  ./chronicle.sh pipeline tabletop
```

If `Evidence` warns, the answers will use on-demand video-frame extraction
instead of pre-stored images. That works, but it is slower and can fail if the
clip is moved. Run `python scripts/build_evidence_fixtures.py` (or the suggested
bounded ingestion) to write `evidence/` beside the episode database.

## Flags

| Flag | Use |
| --- | --- |
| `--source tabletop` | Check one source; repeat for several. Default: every available source. |
| `--detector`, `--hands`, `--embedder` | Force a backend, e.g. `--detector color` for an offline rehearsal. |
| `--llm template` | Skip Ollama and use deterministic answers. |
| `--warm-llm` | Send a tiny prompt to load Qwen before the demo. |
| `--require-llm` | Treat an offline/missing Ollama model as `FAIL`, not `WARN`. |
| `--smoke-source`, `--max-frames` | Change the ingestion smoke test (default `synthetic`, 30 frames). |
| `--strict` | Exit non-zero when any `WARN` is present. |
| `--json` | Machine-readable output for CI or a dashboard. |
| `--no-color` | Plain text for logs and projectors. |
| `--verbose` | Show TensorFlow/MediaPipe/model-loader output. |
| `--skip-*` | Disable a check, e.g. `--skip-tests` during a quick venue check. |

## Stage-day checklist

1. **30 minutes before:** `./chronicle.sh preflight --warm-llm`
2. **Fix every `FAIL`.** A `WARN` is acceptable only if you know which fallback
   will show on screen.
3. **Start the canonical operator dashboard:** `./chronicle.sh pipeline tabletop`
4. **Open** <http://127.0.0.1:8080> and confirm the source dropdown lists
   `synthetic`, `tabletop`, `cook`, `play`, `walking`. The zine mission control
   (`./chronicle.sh mission`, port 8081) is optional and separate; it is not
   part of the operator-dashboard preflight.
5. **Switch to each clip** you plan to show and confirm the video plays.
6. **Ask one grounded question per clip**; confirm the answer has a citation
   and a supporting frame.
7. **Export the snapshot** once to confirm the download path works.
8. **Set the LLM policy:** use `--no-llm` if the venue network or GPU is
   unreliable; the template path is grounded and deterministic.
9. **Re-run** `./chronicle.sh preflight --strict` immediately before going on
   stage if anything changed.

## If a check fails

| Failure | Try |
| --- | --- |
| Missing video | `git lfs pull` |
| Dashboard deps | `uv sync --extra ui` |
| YOLO / hands model missing | `python scripts/fetch_assets.py` |
| SQLite checkpoint busy | `./chronicle.sh stop`, then rerun preflight |
| Ollama offline | `ollama serve` |
| Model not pulled | `ollama pull qwen2.5:7b-instruct` |
| Cold model | `./chronicle.sh preflight --warm-llm` |
| Tests fail | `.venv/bin/pytest -q` and fix before the demo |
| Smoke run fails | `python -m chronicle.cli run --source synthetic --max-frames 30` |

## Limitations

- The preflight does not click through the browser. Step 4–7 above are still
  manual.
- It loads the backends once on a single real frame; it does not benchmark
  throughput. Use `chronicle/benchmark.py` or NB-05 for a full source pass.
- The evidence check proves that extraction works; it does not prove that a
  particular demo question will return a stored strip. Check the questions you
  plan to ask.
- `--skip-tests` and `--skip-run` make the command fast for a venue sanity
  check; do not use them for the final rehearsal.

## Related

- [`NOTICE_BOARD.md`](NOTICE_BOARD.md) — NB-04 (this script) and NB-05 (full
  source verification).
- [`TEAM_STATUS.md`](TEAM_STATUS.md) — current backend and benchmark status.
- [`PERCEPTION.md`](PERCEPTION.md) — detector/hand/embedder details.
- [`CALIBRATION.md`](CALIBRATION.md) — action thresholds used by the smoke run.
