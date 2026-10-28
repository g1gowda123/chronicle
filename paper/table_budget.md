| Dataset | VLM budget | VLM calls | Recall@1 | 95% CI (exact) |
|---|---|---:|---:|---:|
| Cook 45 s | off (stride 0) | 0 | 80.0% (8/10) | [44.4%, 97.5%] |
| Cook 45 s | every 8th trigger | 13 | 80.0% (8/10) | [44.4%, 97.5%] |
| Cook 45 s | every 4th trigger | 26 | 80.0% (8/10) | [44.4%, 97.5%] |
| Cook 45 s | every 2nd trigger | 51 | 80.0% (8/10) | [44.4%, 97.5%] |
| Cook 45 s | every trigger | 102 | 80.0% (8/10) | [44.4%, 97.5%] |
| HoloAssist 34 s | off (stride 0) | 0 | 87.5% (7/8) | [47.3%, 99.7%] |
| HoloAssist 34 s | every 8th trigger | 7 | 87.5% (7/8) | [47.3%, 99.7%] |
| HoloAssist 34 s | every 4th trigger | 13 | 87.5% (7/8) | [47.3%, 99.7%] |
| HoloAssist 34 s | every 2nd trigger | 26 | 87.5% (7/8) | [47.3%, 99.7%] |
| HoloAssist 34 s | every trigger | 51 | 87.5% (7/8) | [47.3%, 99.7%] |

Measured with the final YOLO-World-L detector, scene vocabulary and fixed
synonyms.  The dense 1 FPS baselines use 45 (Cook) and 34 (HoloAssist)
captioner calls and reach 40.0% and 12.5%, respectively.  On this grounded QA
suite, Chronicle's recall is invariant to the Florence-2 budget; the
recommended default is stride 4, which uses 26 and 13 calls.
