# Third-party notices

Chronicle itself is released under the [MIT License](LICENSE). This file lists
the third-party libraries, tools, model weights and datasets the project uses,
with links and the licences they are distributed under.

This file is informational, not legal advice. Always check the upstream project
for the authoritative licence text and version.

## Runtime and core libraries

| Project | Used for | Licence |
| --- | --- | --- |
| [Python](https://www.python.org/) | Runtime (3.11–3.13) | PSF-2.0 |
| [NumPy](https://github.com/numpy/numpy) | Numerical arrays | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv) | Video I/O, image ops, geometry | Apache-2.0 |
| [SciPy](https://github.com/scipy/scipy) | Optimisation / linear algebra | BSD-3-Clause |
| [Shapely](https://github.com/shapely/shapely) | IoS and containment geometry | BSD-3-Clause |
| [NetworkX](https://github.com/networkx/networkx) | Dual-graph memory | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | Image encode/decode | MIT-CMU (HPND) |
| [SQLite](https://sqlite.org/) | Episode DB, VFM vault | Public domain |

## Perception (optional production backends)

| Project | Used for | Licence |
| --- | --- | --- |
| [Ultralytics YOLO / YOLO-World](https://github.com/ultralytics/ultralytics) | Open-vocabulary detection | **AGPL-3.0** |
| [YOLO-World paper code](https://github.com/AILab-CVC/YOLO-World) | Reference implementation | Apache-2.0 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | 21-point hand landmarks | Apache-2.0 |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Place recognition and visual re-ID | MIT |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Tracking design reference (not vendored) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) | Model runtime | BSD-3-Clause |
| [torchvision](https://github.com/pytorch/vision) | Image transforms | BSD-3-Clause |

> **AGPL note:** Ultralytics YOLO-World is optional. The demo and the model-free
> test suite use the colour/palette fallback and do not require Ultralytics. If
> you distribute a combined work or run a hosted service that includes
> Ultralytics code, review the AGPL-3.0 obligations with counsel.

## Semantic engine and local models (optional)

| Project | Used for | Licence |
| --- | --- | --- |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Florence-2 inference | Apache-2.0 |
| [Accelerate](https://github.com/huggingface/accelerate) | Device placement | Apache-2.0 |
| [timm](https://github.com/huggingface/pytorch-image-models) | Vision backbones | Apache-2.0 |
| [einops](https://github.com/arogozhnikov/einops) | Tensor reshaping | MIT |
| [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | Action/caption verification | MIT |
| [Ollama](https://github.com/ollama/ollama) | Local LLM/VLM serving | MIT |
| [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | Optional text phrasing | Apache-2.0 |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | Optional captioning/event verification | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | Frame extraction for VFM captions | LGPL-2.1+ / GPL-2.0+ (build-dependent) |

## Dashboard and UI

| Project | Used for | Licence |
| --- | --- | --- |
| [NiceGUI](https://github.com/zauberzeug/nicegui) | Python-driven web UI | MIT |
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP/JSON API | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI server | BSD-3-Clause |
| [Apache ECharts](https://github.com/apache/echarts) | Operator graph/timeline charts | Apache-2.0 |
| [vis-network](https://github.com/visjs/vis-network) | Zine dashboard memory graph | Apache-2.0 / MIT |

## Development tooling

| Project | Used for | Licence |
| --- | --- | --- |
| [pytest](https://github.com/pytest-dev/pytest) | Test suite | MIT |
| [Ruff](https://github.com/astral-sh/ruff) | Lint/format | MIT |
| [uv](https://github.com/astral-sh/uv) | Environment and lockfile | Apache-2.0 OR MIT |

## Datasets and media

| Asset | Used for | Licence |
| --- | --- | --- |
| [HoloAssist](https://holoassist.github.io/) | Real HoloLens 2 egocentric demo clip and action labels | CDLA-Permissive-2.0 |
| [Meta Project Aria Gen 2](https://www.projectaria.com/datasets/) | Real Aria cook/play/walk research clips | CC BY-NC 4.0 |

The Aria clips are locally trimmed research windows. They are not redistributed
as a dataset; the full source data remains with Meta. Check the dataset terms
before any commercial use or redistribution.

## Model weights

Model weights are not vendored in this repository. `scripts/fetch_assets.py`
downloads the optional weights, which carry their own terms:

- YOLO-World / Ultralytics weights: AGPL-3.0
- Florence-2 weights: MIT
- OpenCLIP weights: MIT (check the specific checkpoint card)
- Qwen2.5 / Qwen2.5-VL weights: Apache-2.0

The MIT licence in [`LICENSE`](LICENSE) covers Chronicle's source code only; it
does not relicense third-party code, weights or datasets.
