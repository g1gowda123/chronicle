# Corrected research ablations

All runs use the leak-free scene vocabulary and the deterministic planner.
`caption_stride: 4` unless noted. Raw runs: `runs/research/`.

## Cumulative changes

| Stage | Detector | Planner | Scene vocab | Strict resolution | Cook | Holo |
|---|---|:--:|:--:|:--:|---:|---:|
| Leak-free baseline | YOLO-World-S | no | no | no | 4/10 (40.0%) | 5/8 (62.5%) |
| + deterministic planner | YOLO-World-S | yes | no | no | 4/10 (40.0%) | 8/8* (100%) |
| + scene vocabulary | YOLO-World-S | yes | yes | no | 5/10 (50.0%) | 8/8* (100%) |
| + strict resolution, fixed synonyms, bowl alias | YOLO-World-S | yes | yes | yes | 7/10 (70.0%) | 7/8 (87.5%) |
| + large detector (final) | YOLO-World-L | yes | yes | yes | 8/10 (80.0%) | 7/8 (87.5%) |
| + two-scale ensemble (rejected) | S + L | yes | yes | yes | 8/10 (80.0%) | 5/8 (62.5%) |

`*` includes the false "switch console → joy-con" pass removed by strict
resolution.

## Detector scale

| Detector | Cook | Holo | Pooled | VLM calls (C/H) | Wall time (C/H) | Peak VRAM | Strict Cook location |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO-World-S | 7/10 (70.0%) | 7/8 (87.5%) | 14/18 (77.8%) | 25 / 9 | 27.5 / 25.2 s | 1830 MB | 2/6 |
| YOLO-World-L (final) | 8/10 (80.0%) | 7/8 (87.5%) | 15/18 (83.3%) | 26 / 13 | 35.8 / 38.5 s | 1960 MB | 1/6 |
| S + L ensemble | 8/10 (80.0%) | 5/8 (62.5%) | 13/18 (72.2%) | 30 / 14 | 43.7 / 46.3 s | 2009 MB | 1/6 |

## Question-leakage ablation

| Detector vocabulary | Cook | Holo | Notes |
|---|---:|---:|---|
| Scene vocabulary (final) | 8/10 | 7/8 | fixed 12-15 class list |
| QA-derived vocabulary | 7/10 | 5/8 | corrupted classes ("person joy-con", "colour joy-con"); Holo count = 6 from 13 tracks |

## VLM budget

| Dataset | off | every 8th | every 4th | every 2nd | every trigger |
|---|---:|---:|---:|---:|---:|
| Cook 45 s | 80.0% (0) | 80.0% (13) | 80.0% (26) | 80.0% (51) | 80.0% (102) |
| HoloAssist 34 s | 87.5% (0) | 87.5% (7) | 87.5% (13) | 87.5% (26) | 87.5% (51) |

Numbers in parentheses are Florence-2 calls. Dense 1 FPS baselines use 45/34
calls and reach 40.0%/12.5%.
