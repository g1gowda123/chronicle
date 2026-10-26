# Literature review — memory instead of video

Scope: papers that (a) compile long video into a reusable memory, (b) retrieve/answer without
re-running the video model, or (c) push memory into the model itself. This is a working review,
not a formal survey. Links were checked on 2026-10-26.

## The short version

- **Captions are a real memory substrate.** CapMem shows that timestamped 30–60 s captions beat
  direct VideoQA on long egocentric videos for most models, and a retrieve-and-verify loop adds
  more. The win comes from coverage and reuse, not from caption elegance.
- **Graphs are the index, not the answer.** EgoGraph, VL-KnG, R4DSG and Chronicle all converge on
  persistent entities + anchor-relative relations + a timestamped event ledger. The graph makes
  temporal and identity questions cheap.
- **Query-agnostic construction, query-specific composition.** MERIT and FocusGraph both argue
  for building a high-recall memory before knowing the question, then composing/expanding at
  query time. This is the opposite of "run a video LLM per question".
- **Frames on demand.** Caption-once / frames-on-demand and FocusGraph keep pixels out of the
  loop until a text-first answer is insufficient. That is the right failure mode for a demo:
  cheap by default, visual only when needed.
- **Memory hierarchy is mature enough to copy.** LightMem's sensory → short-term → long-term
  split, HippoRAG's graph retrieval, and EM-LLM's event segmentation all map cleanly onto a
  three-tier Chronicle vault.
- **The radical end is model-native memory.** Titans/MemoryLLM store memory in parameters or a
  latent pool. This is exciting but high-risk for a hackathon; treat it as a north star, not the
  first implementation.

## Papers

| # | Work | Venue / year | Core idea | Why it matters here |
|---|------|--------------|-----------|---------------------|
| 1 | [R4DSG](https://arxiv.org/abs/2608.11017) | ACM MM 2026 | Relative 4D scene-graph memory; stable anchors vs dynamic objects; compact queryable entries; no global 3D reconstruction | Validates anchor-relative, video-free memory; Chronicle already has the dual graph and terminal anchors |
| 2 | [VL-KnG](https://arxiv.org/abs/2510.01483) | 2025/2026 | Persistent spatio-temporal KG from monocular egocentric video; LLM-based object association; graph-enhanced retrieval; latency decoupled from video length | The closest architectural cousin; Chronicle needs the same query-time independence and a stronger association story |
| 3 | [EgoGraph](https://arxiv.org/abs/2602.23709) | 2026 | Training-free temporal KG with an egocentric schema (people, objects, locations, events); cross-entity temporal modelling; SOTA on EgoLifeQA/EgoR1 | Direct blueprint for the vault schema: entities + events + time, not clips |
| 4 | [CapMem](https://arxiv.org/abs/2609.17688) | EMNLP 2026 | 75 videos / 33.7 h, 1,000 MCQs; caption-only QA beats direct VideoQA on >20 min videos for 10/12 (30 s) and 8/12 (60 s) models; retrieve-and-verify +5.3 pts | Strongest evidence that a caption ledger is not a downgrade; sets the caption-window experiment |
| 5 | [EgoCITE](https://arxiv.org/pdf/2608.12627v2) | 2026 | Dense visual captions + speech transcripts as memory entries; context-augmented indexing; time-aware evidence retrieval | Add audio/transcripts to the ledger; query by time, not just embedding similarity |
| 6 | [MERIT](https://arxiv.org/abs/2608.07663) | ECCV 2026 (oral) | Query-agnostic memory build; episodic multi-key representation; high-recall key matching; temporal expansion only at inference | Memory construction must not guess the question; defer relation composition to query time |
| 7 | [FocusGraph](https://arxiv.org/abs/2603.04349) | 2026 | Scene-graph LLM selector picks query-relevant clips; patch-wise sparse-flow keyframes; no raw frame sequences at question time | The "text-first, pixels-second" router; useful for the low-confidence fallback |
| 8 | [VideoMem](https://arxiv.org/abs/2512.04540) | 2025/2026 | Adaptive global memory buffer for ultra-long video; progressive state + PRPO training | Compression/retention policy inspiration; we can do it without training |
| 9 | [EM-LLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c05144b635df16ac9bbf8246bbbd55ca-Abstract-Conference.html) | ICLR 2025 | Episodic events from Bayesian surprise + graph boundary refinement; similarity + temporal-contiguity retrieval | Event segmentation and two-stage retrieval for the caption ledger |
| 10 | [LightMem](https://arxiv.org/abs/2510.18866) | ICLR 2026 | Sensory filtering → short-term → long-term; offline consolidation; up to 38× fewer tokens / 30× fewer calls | Copy the tiering and the offline/online split; our ingest is the offline phase |
| 11 | [HippoRAG](https://arxiv.org/abs/2405.14831) | NeurIPS 2024 | KG + Personalized PageRank for long-term memory; 10–30× cheaper than iterative retrieval | Graph retrieval is a cheap alternative to iterative VLM calls |
| 12 | [EgoMemory / EgoRetriever](https://aclanthology.org/2026.findings-acl.362.pdf) | ACL Findings 2026 | Personal memory bank + reflective CoT; training-free retrieval over long egocentric video | Personalisation and intent inference; 639 human-curated queries |
| 13 | [Continuous Semantic Caching](https://research.ece.cmu.edu/lions/Papers/Semantic_LLM_Caching.pdf) | 2026 | Principled semantic cache eviction/reuse under uncertainty | A QA cache is a memory tier; answers can be reused across paraphrases |
| 14 | [Titans](https://arxiv.org/abs/2501.00663) | 2024/2025 | Neural long-term memory module that learns at test time; attention = short-term, neural memory = long-term | North star: memory in weights, not just a DB |
| 15 | [MemoryLLM](https://arxiv.org/abs/2402.04624) | ICML 2024 | Fixed-size latent memory pool inside a transformer; self-updates with new knowledge | Another route to "the model remembers the episode" without retrieval |
| 16 | [Caption-once, Frames-on-Demand](https://aclanthology.org/2026.emnlp-main.422.pdf) | EMNLP 2026 | Cache captions once; fetch frames only when confidence is low; bounded FIFO visual working memory | Exactly the query-time cost model we want: text first, pixels on demand |

## Detailed notes

### CapMem — the strongest empirical argument

The benchmark defines *Episodic Memory Video Caption QA*: the model answers **only from
timestamped captions**, never from the video. On 33.7 hours across 75 videos, full-coverage
caption memory with 30 s and 60 s windows beats direct VideoQA on long videos for 10/12 and 8/12
models. A matched-frame control shows the gain is not just "more frames". The retrieve-and-verify
harness improves accuracy by up to 5.3 points.

Implication: a dense caption ledger is a legitimate memory format. The risk moves from "did we
sample the right frame?" to "did we caption enough, and can we retrieve it?".

### EgoCITE — time is a first-class key

EgoCITE turns video + audio into dense captions and transcripts, then indexes them with context
and retrieves with time-aware evidence. The lesson is simple: users ask "what did I do after
lunch?" as often as "where is the cup?". A pure vector index loses the temporal ordering that
makes those questions answerable.

Implication: the vault needs an explicit event timeline and SQL time filters, not just embeddings.

### EgoGraph — the schema

EgoGraph's egocentric schema unifies people, objects, locations and events, and models temporal
dependencies across entities. It is training-free and accumulates memory over multiple days.

Implication: do not invent a new ontology. Use the existing Chronicle entities/edges/transitions,
add `events` and `captions`, and make time an indexed column.

### MERIT — build for unknown questions

MERIT deliberately avoids expensive high-level relation construction at build time. It stores a
simple multi-key episodic memory, retrieves with high recall, and expands temporally only around
the retrieved segments at inference.

Implication: the compiler should be cheap and over-inclusive; the query engine composes relations.
This also means we should not precompute only the questions we expect.

### FocusGraph / Caption-once — the fallback ladder

Both papers keep raw pixels out of the default path. FocusGraph selects clips from graph-based
captions, then extracts a small keyframe set. Caption-once retrieves captions first and only
fetches frames when confidence is low, with a bounded FIFO working memory.

Implication: implement a **three-step query ladder**:
1. answer from the structured ledger,
2. if confidence is low, retrieve 1–3 keyframes and ask a small VLM,
3. if still low, say "not in memory" rather than hallucinate.

### LightMem / HippoRAG / EM-LLM — the memory stack

- LightMem: filter sensory input, keep short-term context, consolidate into long-term memory
  offline. Up to 38× token reduction.
- HippoRAG: graph + Personalized PageRank retrieves multi-hop evidence cheaply.
- EM-LLM: segment streams into events using surprise/boundaries and retrieve with similarity +
  temporal contiguity.

Implication: the vault should have tiers, a graph index, and event segmentation — all three are
already close to Chronicle's data model.

### Titans / MemoryLLM — the radical end

Titans learns a neural long-term memory at test time; MemoryLLM maintains a fixed-size latent
memory pool inside the transformer. Both point at a future where the episode is not a database
row but a change in the model's weights/state.

Implication: keep as a stretch experiment. A LoRA/cartridge per episode or a memory-pool adapter
could answer common questions with zero retrieval, but evaluation and catastrophic forgetting are
real risks. Do not put this on the demo critical path.

## Standardized benchmarks: what to run, and what they miss

There is no standard benchmark for "delete the video, query the compiled memory, and return a few
evidence JPEGs." The closest external tests are:

| Benchmark | What it measures | Fit / caveat |
|---|---|---|
| **CapMem — Episodic Memory Video Caption QA** | MCQs answered from timestamped 30/60 s captions only; 75 videos / 33.7 h / 1,000 MCQs per the paper | **Closest task match.** Captions are the only input; no video at answer time. Use if the release is usable. |
| **EgoSchema** | 5,000+ three-minute egocentric clips, long-form multiple-choice QA | Public and standardized. Compile clips to vaults and answer from captions/events; run a 100–200 clip subset first. |
| **Ego4D Episodic Memory — NLQ** | Natural-language queries grounded to time intervals | Directly tests "when" questions. Compile 5–10 min windows; measure temporal IoU/recall@1. Requires Ego4D access. |
| **EgoLifeQA / EgoR1-bench** | Long-horizon, multi-day egocentric memory/reasoning | Best fit for the multi-hour claim; check release/licensing. |
| **EgoMemory / EgoRetriever** | 639 human-curated personal-memory queries over a memory bank | Good retrieval test; less about compiled-ledger structure. |
| **Video-MME / LongVideoBench / MLVU / LVBench** | General long-video MCQ | Standard and public, but not egocentric and usually assume video/frames. Adaptable, lower relevance. |

None of them measure video deletion, evidence-JPEG sufficiency, semantic-cache hit rate, or
bytes/hour retention. Those need the internal harness in `benchmark/questions.json` and
`retention.py`. The honest framing: the 30-question set is a regression suite; CapMem/EgoSchema
would test questions we did not design.

### Baseline economics

A useful external benchmark needs **both** the original system and a video-to-LLM baseline, not
just vault accuracy. Measured on the same 30 questions:

- original Chronicle graph + Qwen2.5 7B: 14/30, 847 ms p50, 4.6 s p95, requires the DB/video;
- 9-frame timestamped contact sheet + Qwen2.5-VL: 7/30, 4.46 s p50, 764 tokens/question,
  19.24 MB of video at query;
- VFM vault: 30/30, 0.045 ms p50, 0 tokens, no video, 404 KB at rest.

The original is more fluent but reads terminal anchors, so it gives stale locations/timestamps.
The vault wins on exact temporal lookup at the cost of natural-language richness. Counts remain
track counts, not verified physical counts.

The advantage grows with duration: a real video LLM must either scale frames/tokens with length
or keep a fixed frame budget and lose coverage. The vault grows with events (~42 KB/hour) plus a
bounded snapshot budget (12–40/hour), so the video-to-memory ratio grows roughly linearly with
episode length.

## What Chronicle already has that these papers do not

- A deterministic 9-state lifecycle with wall-clock gates and support/containment reasoning.
- A real graph + SQLite projection with terminal anchors and 10-minute trajectory retention.
- A verification layer: temporal strips, Florence-2 action JSON, calibration, queue telemetry.
- A UI that shows the evidence frame and seeks the video.

The experiment should **reuse** these, not replace them. The missing pieces are:
dense caption ledger, event snapshots as the only retained pixels, a video-free query engine,
and a query/answer cache.
