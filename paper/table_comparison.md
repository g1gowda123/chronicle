| Dataset | Method | Recall@1 | VLM Calls | Wall time (s) | Query latency (ms) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|
| Cook 45 s | Chronicle (final, stride 4) | **80.0%** (8/10) | 26 | 35.8 | 1.13 | 1960.4 |
| Cook 45 s | Dense 1 FPS | 40.0% (4/10) | 45 | 15.0 | 262.9 | 819.2 |
| HoloAssist 34 s | Chronicle (final, stride 4) | **87.5%** (7/8) | 13 | 38.5 | 0.69 | 1960.7 |
| HoloAssist 34 s | Dense 1 FPS | 12.5% (1/8) | 34 | 10.9 | 228.8 | 817.2 |

Chronicle query latency is the GraphRAG answer time (sub-millisecond). The
dense baseline's latency is per sampled caption. Chronicle still processes
every frame for detection/tracking, so its end-to-end wall time is higher; the
efficiency claim is about VLM calls, not total compute. The dense numbers were
produced by the same harness before the perception fixes and are unchanged.
