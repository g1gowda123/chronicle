# Chronicle — hackathon pitch

> **Historical material.** This pitch is kept for context. Its 30-question suite,
> 95 KB memory, and cost/latency numbers refer to the earlier VFM prototype, not
> the current `chronicle eval` product suite (60 deterministic checks) or the
> research results on `research/vfm-merged`.

**Winning frame:** every other team is making video models faster. You made the video
unnecessary.

---

## The one-liner (memorize this)

> "We turn a 15-minute, 249 MB video into a 95 KB searchable memory. Ask it anything — it answers in
> under a millisecond, with a timestamp and the exact frame. No video, no video model."

Backup line if the demo fails:

> "The point isn't the answer. The point is that the answer came from 95 KB of text, not from
> re-watching the video."

---

## 60-second pitch (open with the demo, not the architecture)

**[Hold up / play the video]**
"Judges — this is a 15-minute, 249-megabyte video. Every normal system answers a question about it by
re-watching it: decode, sample frames, call a vision model. Seconds and dollars per question."

**[Point to the 95 KB memory]**
"We compile it once. This is the entire searchable memory: 95 kilobytes. A graph index, timestamped
captions, a compressed event ledger, and a few evidence frames. After this, the video is optional."

**[Type] "Where are my keys?"**
"Answer, timestamp, frame."

**[Type] "What happened around 10 minutes?"**
"Caption window, citation, evidence still."

**[Type] "How many bowls?"**
"Track, peak, physical — with duplicate plates reconciled."

**[Stop the LLM / switch it off]**
"And with the language model off, the deterministic path still answers. No model in the answer path."

**[Close]**
"Everyone here is making video understanding faster. We made the video unnecessary — compile once,
query in microseconds, show the evidence."

---

## 3-minute pitch

### 0:00–0:20 — Hook with the object
Show the 249 MB file and the 95 KB memory side by side.

> "One is the episode. One is everything you need to answer questions about it. The difference is the
> product."

### 0:20–0:50 — Problem
- Every question re-decodes and re-samples the video.
- Cost and latency scale with video length, not question difficulty.
- Coverage depends on which frames you happened to sample.
- Long egocentric video is the worst case: huge file, tiny relevant moment.

### 0:50–1:40 — Live demo (three questions)
1. **Where are my keys?** → exact location, timestamp, clickable citation, frame chip.
2. **What happened around 10 minutes?** → caption window + evidence still.
3. **How many bowls?** → counts with duplicate reconciliation; explain 4 tracks → zero physical
   bowls because two were double-labelled plates.
4. **Kill the LLM** and repeat one query. "Still works. Deterministic."

### 1:40–2:20 — How it works (one diagram, no jargon)
- Compile once: VLM captions every 30 s, the graph emits a salience event ledger, evidence stills are
  saved, everything is packed into a small SQLite vault.
- Query: intent router → deterministic vault lookup → citations + evidence. Miss → graph fallback.
- The LLM is optional phrasing, never the answer source.

### 2:20–2:45 — Proof and honesty
- 249 MB → 95 KB; 0.05–0.06 ms vault p50; 0.89 ms cold on the 15-minute walk.
- Baselines on your own 30-question suite: vault 30/30, Qwen over pack 30/30, graph+Qwen 14/30,
  9-frame VLM 7/30. **Say "internal regression suite", not "benchmark".**
- One honest limit: "Captions are lossy — fine motion, audio and tiny objects. Evidence makes errors
  inspectable, not invisible."

### 2:45–3:00 — Close
> "Not another video LLM. The memory layer that makes one unnecessary for repeated questions. The
> video is optional."

---

## Slide deck (5 slides max)

1. **Demo first.** No title slide. Play the clip, ask one question, show the frame.
2. **The re-watch tax.** One bullet per problem: slow, expensive, coverage depends on sampling.
3. **Compile once, query many.** The 5-box diagram: video → compile → memory → question → answer.
   Under it: graph = index, vault = memory, evidence = trust.
4. **Proof.** 249 MB → 95 KB · 0.05 ms · 0 video · 30/30 internal · 14/30 and 7/30 baselines.
5. **Limits + close.** What is lost, what is next (external benchmark, audio, live anomaly), one-liner.

Put the architecture diagram on the slide; do not read it. Point at it once.

---

## Demo setup checklist

- Pre-compile the demo clip. Never rely on a live compile on stage.
- Have two clips: one graph-backed (shows fallback + counts) and one caption-first (shows long-video
  coverage).
- Preload the LLM, then test with it **off**.
- Rehearse exactly three questions. Do not improvise a fourth.
- Open the evidence lightbox and fullscreen once before going on stage.
- Record a 60-second screen capture of the working demo as backup.
- Put the size comparison on screen: 249 MB file vs 95 KB vault.
- If a query misses: "That's the graph fallback" — then show it. A visible fallback is a feature.
- If the network dies: switch to the offline deterministic path and say so.

---

## Judge Q&A (short answers)

**"Isn't this just captions / RAG?"**
No. Captions alone cannot answer "when", "how many" or "where" cleanly. We add a salience event
ledger, graph counts, deterministic intents, and evidence.

**"Why not just use a video LLM?"**
We do — at compile time. Query time is the win. A per-question video model is the baseline we
measured against.

**"What's novel?"**
The system: dual memory (graph index + caption/event vault), a deterministic query path, graph
fallback, and evidence-backed answers. The algorithms are prior work; the integration and cost profile
are the contribution.

**"Is 30/30 a benchmark?"**
No — an internal 30-question regression suite. External benchmarks (EgoSchema, Ego4D, CapMem-style)
are the next step.

**"What is lost?"**
Fine motion, audio, tiny text, some rare objects. We state it and keep evidence for audit.

**"Does it need the video?"**
No. Only for playback and screenshots. That is the product.

**"Why will people use it?"**
Repeated questions over long recordings: field ops, training review, compliance, accessibility,
robotics, coaching. The second question is nearly free.

---

## Do not say

- "First" or "state of the art."
- "30/30 benchmark" — say "internal regression suite."
- "Replaces video LLMs" — say "makes one unnecessary for repeated questions."
- "Zero hallucination" — say "auditable, with citations and evidence."
- "Real-time" unless you have measured it on the demo machine.

---

## The moment that wins

Stop the language model mid-demo, ask again, and show the deterministic answer with a citation.
Most teams demo a model. You demo a system that does not need one at query time.
