# Chronicle evaluation summary

- **Method:** `chronicle`
- **Video:** `/home/g1/projects/chronicle-longvideo/data/real/holoassist_z210_257s.mp4`
- **QA file:** `/home/g1/projects/chronicle-longvideo/data/qa/holoassist_z210_40q.json`
- **Questions:** 40
- **Recall@1:** 12.5% (5/40)
- **Answered before video end:** 40
- **Unreachable timestamps:** 0
- **Average GraphRAG query latency:** 4.9222 ms

## Efficiency

| Metric | Value |
|---|---:|
| Frames processed | 2565 |
| VLM calls (Florence triggers) | 54 |
| VLM call reduction | 97.89% |
| Effective Florence calls | 54 |
| Text-LLM calls | 0 |
| Text-LLM latency (mean) | — |
| Text-LLM models |  |
| Average frame latency | 30.4455 ms |
| Steady-state average | 29.3689 ms |
| p50 / p95 latency | 26.2615 / 48.6099 ms |
| Peak VRAM | 1960.66 MB |
| Foreground graph | 14 nodes / 577 edges |
| Background graph | 2 nodes / 1 edges |

## Recall@1 by category

| Category | Correct | Total | Recall@1 |
|---|---:|---:|---:|
| activity | 1 | 6 | 16.7% |
| counting | 1 | 6 | 16.7% |
| error_correction | 0 | 8 | 0.0% |
| object_identity | 0 | 8 | 0.0% |
| object_location | 2 | 6 | 33.3% |
| temporal_order | 1 | 6 | 16.7% |

## Per-question results

| # | Question | Expected | Predicted | Correct | Answered at | Source | Latency (ms) |
|---:|---|---|---|---:|---:|---|---:|
| 1 | What did the person attach the blue controller to first? | comfort grip | I don't remember any blue objects in this episode. | no | 250.0751427712654 | graph | 0.2663 |
| 2 | What did the person attach the controllers to after the comfort grip? | Nintendo Switch | The last recorded event for comfort grip was state OUT_OF_VIEW in the Work Table at [t=168.2s]. | no | 250.0751427712654 | template | 2.0659 |
| 3 | What did the person attach after the Nintendo Switch? | wrist straps | The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.1191 |
| 4 | What did the instructor open on the back of the Switch? | kickstand | The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.0868 |
| 5 | What did the person remove from the game card slot? | old game card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.4298 |
| 6 | What did the person insert after removing the old game card? | new game card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.5128 |
| 7 | What did the person replace in the kickstand compartment? | SD card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.287 |
| 8 | What did the person press to turn on the Switch? | power button | The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.0098 |
| 9 | Where did the person put the joy-con controllers at the start? | comfort grip | The last recorded event for joy-con was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.4095 |
| 10 | Where was the Nintendo Switch placed after the kickstand was opened? | Work Table | The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | yes | 250.0751427712654 | template | 3.4048 |
| 11 | Where is the game card slot? | back of the Nintendo Switch | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.5117 |
| 12 | Where is the SD card slot? | kickstand compartment | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.3241 |
| 13 | How many joy-con controllers did the person use? | 2 | There are 4 joy-cons in my memory (resolved from 4 detector tracks). | no | 250.0751427712654 | graph | 0.8958 |
| 14 | How many comfort grips did the person use? | 1 | There are 1 comfort grip in my memory (resolved from 1 detector tracks). | yes | 250.0751427712654 | graph | 0.2824 |
| 15 | How many wrist straps did the person attach? | 2 | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.1734 |
| 16 | How many game cards did the person handle? | 2 | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.1695 |
| 17 | How many SD cards did the person handle? | 2 | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.1579 |
| 18 | How many times did the person press the power button? | 1 | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.1336 |
| 19 | What happened first, attaching the controllers to the comfort grip or to the wrist straps? | comfort grip | The last recorded event for comfort grip was state OUT_OF_VIEW in the Work Table at [t=168.2s]. | yes | 250.0751427712654 | template | 1.1646 |
| 20 | What happened after the person removed the old game card? | inserted the new game card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.2698 |
| 21 | What happened before the instructor opened the kickstand? | attached the wrist straps | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.463 |
| 22 | Which came first, the game card replacement or the SD card replacement? | game card | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.4878 |
| 23 | Which came last, opening the kickstand or pressing the power button? | power button | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.4924 |
| 24 | What did the person do between removing the old game card and inserting the new one? | rotated the new game card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.6945 |
| 25 | What did the person do at the beginning? | picked up the comfort grip | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.5037 |
| 26 | What did the person do after attaching the blue controller? | attached the red controller | I don't remember any blue objects in this episode. | no | 250.0751427712654 | graph | 0.0624 |
| 27 | What did the person do at 67 seconds? | grabbed the Nintendo Switch | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.0878 |
| 28 | What did the person do at 166 seconds? | opened the kickstand | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.1529 |
| 29 | What did the person do at 194 seconds? | inserted the new game card | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 124.9755 |
| 30 | What mistake did the person make with the blue controller at 25 seconds? | inserted it through the middle | I don't remember any blue objects in this episode. | no | 250.0751427712654 | graph | 0.0607 |
| 31 | How was the person corrected at 25 seconds? | verbally | Person was picked up in the Work Table at [t=247.7s]. | no | 250.0751427712654 | template | 3.142 |
| 32 | What did the person do wrong at 106 seconds? | inserted the red controller | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.6008 |
| 33 | What did the instructor say at 42 seconds? | No, top, very very top | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.4262 |
| 34 | What did the person do after the instructor said no, top, very very top? | slid the blue controller | The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s]. | no | 250.0751427712654 | template | 3.5979 |
| 35 | How many actions were corrected by the instructor verbally? | 10 | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.162 |
| 36 | What was the last wrong action? | inserted the new game card | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.3603 |
| 37 | What was the first wrong action? | slid the blue controller | I don't have that in my memory. | no | 250.0751427712654 | graph | 0.3775 |
| 38 | Where are the joy-con controllers at the end? | wrist straps | The last recorded event for nintendo switch was state OUT_OF_VIEW in the Work Table at [t=253.6s]. | no | 256.08656447249774 | template | 3.4103 |
| 39 | Where is the Nintendo Switch at the end? | Work Table | The last recorded event for nintendo switch was state OUT_OF_VIEW in the Work Table at [t=253.6s]. | yes | 256.08656447249774 | template | 3.0964 |
| 40 | What did the person do at the end? | turned on the Nintendo Switch | The last recorded event for nintendo switch was state OUT_OF_VIEW in the Work Table at [t=253.6s]. | yes | 256.08656447249774 | template | 3.0597 |

## Failures

- **What did the person attach the blue controller to first?** — token recall 0.00 — `I don't remember any blue objects in this episode.`
- **What did the person attach the controllers to after the comfort grip?** — token recall 0.00 — `The last recorded event for comfort grip was state OUT_OF_VIEW in the Work Table at [t=168.2s].`
- **What did the person attach after the Nintendo Switch?** — token recall 0.00 — `The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the instructor open on the back of the Switch?** — token recall 0.00 — `The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person remove from the game card slot?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person insert after removing the old game card?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person replace in the kickstand compartment?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person press to turn on the Switch?** — token recall 0.00 — `The last recorded event for nintendo switch was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **Where did the person put the joy-con controllers at the start?** — token recall 0.00 — `The last recorded event for joy-con was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **Where is the game card slot?** — prediction says unknown — `I don't have that in my memory.`
- **Where is the SD card slot?** — prediction says unknown — `I don't have that in my memory.`
- **How many joy-con controllers did the person use?** — expected number missing — `There are 4 joy-cons in my memory (resolved from 4 detector tracks).`
- **How many wrist straps did the person attach?** — prediction says unknown — `I don't have that in my memory.`
- **How many game cards did the person handle?** — prediction says unknown — `I don't have that in my memory.`
- **How many SD cards did the person handle?** — prediction says unknown — `I don't have that in my memory.`
- **How many times did the person press the power button?** — prediction says unknown — `I don't have that in my memory.`
- **What happened after the person removed the old game card?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What happened before the instructor opened the kickstand?** — prediction says unknown — `I don't have that in my memory.`
- **Which came first, the game card replacement or the SD card replacement?** — prediction says unknown — `I don't have that in my memory.`
- **Which came last, opening the kickstand or pressing the power button?** — prediction says unknown — `I don't have that in my memory.`
- **What did the person do between removing the old game card and inserting the new one?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person do at the beginning?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person do after attaching the blue controller?** — token recall 0.00 — `I don't remember any blue objects in this episode.`
- **What did the person do at 67 seconds?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person do at 166 seconds?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the person do at 194 seconds?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What mistake did the person make with the blue controller at 25 seconds?** — token recall 0.00 — `I don't remember any blue objects in this episode.`
- **How was the person corrected at 25 seconds?** — token recall 0.00 — `Person was picked up in the Work Table at [t=247.7s].`
- **What did the person do wrong at 106 seconds?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **What did the instructor say at 42 seconds?** — prediction says unknown — `I don't have that in my memory.`
- **What did the person do after the instructor said no, top, very very top?** — token recall 0.00 — `The last recorded event for person was state VISIBLE_STATIC in the Work Table at [t=248.8s].`
- **How many actions were corrected by the instructor verbally?** — prediction says unknown — `I don't have that in my memory.`
- **What was the last wrong action?** — prediction says unknown — `I don't have that in my memory.`
- **What was the first wrong action?** — prediction says unknown — `I don't have that in my memory.`
- **Where are the joy-con controllers at the end?** — token recall 0.00 — `The last recorded event for nintendo switch was state OUT_OF_VIEW in the Work Table at [t=253.6s].`

## Notes

- Recall@1 uses deterministic token/number matching against the QA answer.
- VLM calls count captioner triggers; `effective Florence calls` excludes stub fallbacks.
- Text-LLM calls count GraphRAG phrasing requests; template answers make zero LLM calls.
- Peak VRAM is `torch.cuda.max_memory_allocated()` after resetting peak stats at run start.
- The first `warmup_frames` are excluded from steady-state latency.
