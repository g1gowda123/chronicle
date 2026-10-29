# Documentation guide

This directory contains both current documentation and historical design notes.
They are not all describing the same version of Chronicle. Use the table below
before quoting a number from a document.

## Current

These documents describe the code on `main` today.

| Document | Scope |
|---|---|
| [`../README.md`](../README.md) | Install, quickstart, CLI, architecture, evaluation summary |
| [`BENCHMARKS.md`](BENCHMARKS.md) | 13 handcrafted GraphRAG text cases and LLM baselines |
| [`RESEARCH_RESULTS.md`](RESEARCH_RESULTS.md) | Research video QA, ablations and caveats |
| [`INTEGRATION.md`](INTEGRATION.md) | Runtime integration contract |
| [`PERCEPTION.md`](PERCEPTION.md) | Perception contract |
| [`ROLE_B.md`](ROLE_B.md) | Memory and graph contract |
| [`ROLE_C.md`](ROLE_C.md) | Semantic verification contract |
| [`CALIBRATION.md`](CALIBRATION.md) | Action calibration notes |
| [`DEMO_PREFLIGHT.md`](DEMO_PREFLIGHT.md) | Demo environment checks |
| [`RELEASING.md`](RELEASING.md) | Release checklist |
| [`EVENT_CLIPS.md`](EVENT_CLIPS.md) | Event clip evidence format |
| [`TASK_REAL_FOOTAGE.md`](TASK_REAL_FOOTAGE.md) | Real-footage task notes |

The authoritative evaluation summary is in the root README under **What the
evaluation numbers mean**. In short:

| Suite | Cases | Where it lives |
|---|---:|---|
| `chronicle eval` product checks | 60 | `main` |
| GraphRAG text cases | 13 | `main` |
| Research video QA | 18 | `research/vfm-merged` |
| HoloAssist long-video diagnostic | 40 | `research/long-video-qa` |

The 60-case product suite is a deterministic smoke test over the bundled demo
memory. It is **not** a general video-understanding benchmark.

## Historical prototype material

These documents describe the earlier VFM / 30-question prototype or record
project history. They are kept for context, but their evaluation numbers and
architecture claims must not be presented as current results.

| Document | Why it is historical |
|---|---|
| [`HACKATHON_PITCH.md`](HACKATHON_PITCH.md) | 30-question pitch, 95 KB memory, older latency numbers |
| [`CHRONICLE_LEARNING_GUIDE.tex`](CHRONICLE_LEARNING_GUIDE.tex) | Combined VFM architecture guide with the 30-question suite |
| [`CHRONICLE_SHORT.tex`](CHRONICLE_SHORT.tex) | Short VFM guide with the 30-question suite |
| [`VFM_ARCHITECTURE.tex`](VFM_ARCHITECTURE.tex) | VFM architecture note and 30-question cost table |
| [`chronicle-spec-condensed.tex`](chronicle-spec-condensed.tex) | Condensed VFM/graph specification |
| [`NOTICE_BOARD.md`](NOTICE_BOARD.md) | Historical task board and commit log |
| [`TEAM_STATUS.md`](TEAM_STATUS.md) | Historical team status log |
| [`SUGGESTIONS.txt`](SUGGESTIONS.txt) | Historical brainstorming notes |

The committed PDFs were generated from these TeX sources before the current
product suite existed. They are intentionally left in place as a record; the
source files now carry a historical-note banner.
