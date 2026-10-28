#!/usr/bin/env python3
"""Prepare and verify local model assets for offline Chronicle evaluation.

Two modes:

    # One-time, while online
    python scripts/fetch_models.py --download

    # Every later run, no network
    python scripts/fetch_models.py --check
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval_pipeline.py --evaluate

The download mode writes exactly the paths used by ``config.yaml``:

    models/yolov8l-worldv2.pt   # default detector
    models/yolov8s-worldv2.pt   # efficiency ablation / optional ensemble
    models/hand_landmarker.task
    models/Florence-2-base/
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = REPO_ROOT / "models"

ASSETS: dict[str, str] = {
    "yolov8s-worldv2.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        "yolov8s-worldv2.pt"
    ),
    "yolov8l-worldv2.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
        "yolov8l-worldv2.pt"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}
FLORENCE_REPO = "microsoft/Florence-2-base"
FLORENCE_DIRNAME = "Florence-2-base"
FLORENCE_REQUIRED = ("config.json", "preprocessor_config.json")
FLORENCE_WEIGHTS = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _download(url: str, destination: Path, *, chunk_size: int = 1 << 20) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "chronicle-models/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total and written % (10 * chunk_size) < chunk_size:
                    print(f"\r  {written / 1e6:7.1f} / {total / 1e6:7.1f} MB", end="", flush=True)
    if total and written != total:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download: {written} of {total} bytes")
    temporary.replace(destination)
    print(f"\r  saved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def download_florence(models_dir: Path, *, force: bool = False) -> Path:
    destination = models_dir / FLORENCE_DIRNAME
    if destination.is_dir() and not force:
        present = sum(1 for _ in destination.rglob("*") if _.is_file())
        if present:
            print(f"skip {destination} (already present; use --force to replace)")
            return destination
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Florence download needs huggingface_hub: "
            "pip install 'huggingface_hub[hf_transfer]'"
        ) from exc
    print(f"downloading {FLORENCE_REPO} -> {destination}")
    snapshot_download(
        repo_id=FLORENCE_REPO,
        local_dir=str(destination),
    )
    return destination


def check_assets(models_dir: Path, *, only: Sequence[str] | None = None) -> list[str]:
    """Return a list of human-readable problems; empty means ready."""

    selected = set(only or ("yolo", "hands", "florence"))
    problems: list[str] = []
    if "yolo" in selected:
        for name in ("yolov8s-worldv2.pt", "yolov8l-worldv2.pt"):
            path = models_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing or empty: {path}")
    if "hands" in selected:
        path = models_dir / "hand_landmarker.task"
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {path}")
    if "florence" in selected:
        florence = models_dir / FLORENCE_DIRNAME
        if not florence.is_dir():
            problems.append(f"missing directory: {florence}")
        else:
            for name in FLORENCE_REQUIRED:
                if not (florence / name).is_file():
                    problems.append(f"missing: {florence / name}")
            if not any((florence / name).is_file() for name in FLORENCE_WEIGHTS):
                problems.append(
                    f"missing Florence weights in {florence} "
                    f"(expected one of {', '.join(FLORENCE_WEIGHTS)})"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--download", action="store_true", help="fetch missing assets (network required)")
    parser.add_argument("--check", action="store_true", help="verify local assets only (no network)")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument(
        "--only",
        choices=("all", "yolo", "hands", "florence"),
        default="all",
        help="restrict download/check to one asset group",
    )
    args = parser.parse_args(argv)
    models = args.models_dir.expanduser().resolve()
    models.mkdir(parents=True, exist_ok=True)

    selected = {
        "all": ["yolo", "hands", "florence"],
        "yolo": ["yolo"],
        "hands": ["hands"],
        "florence": ["florence"],
    }[args.only]

    if args.download:
        if "yolo" in selected:
            for name in ("yolov8s-worldv2.pt", "yolov8l-worldv2.pt"):
                path = models / name
                if path.exists() and not args.force:
                    print(f"skip {path} (already present)")
                else:
                    print(f"fetching {name} -> {path}")
                    _download(ASSETS[name], path)
        if "hands" in selected:
            name = "hand_landmarker.task"
            path = models / name
            if path.exists() and not args.force:
                print(f"skip {path} (already present)")
            else:
                print(f"fetching {name} -> {path}")
                _download(ASSETS[name], path)
        if "florence" in selected:
            try:
                download_florence(models, force=args.force)
            except Exception as exc:
                print(f"Florence download failed: {exc}", file=sys.stderr)
                return 1

    problems = check_assets(models, only=selected)
    if problems:
        print("Local assets are NOT ready:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRun once with network access:\n"
            "  python scripts/fetch_models.py --download\n"
            "Then verify offline:\n"
            "  python scripts/fetch_models.py --check",
            file=sys.stderr,
        )
        return 1
    print(f"All model assets present under {models}")
    print(f"  Florence: {models / FLORENCE_DIRNAME}")
    print(f"  YOLO-S:   {models / 'yolov8s-worldv2.pt'}")
    print(f"  YOLO-L:   {models / 'yolov8l-worldv2.pt'}")
    print(f"  Hands:    {models / 'hand_landmarker.task'}")
    print("\nFor a strictly offline run:")
    print("  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    return 0


if __name__ == "__main__":
    if os.getenv("HF_HUB_OFFLINE") == "1":
        print("HF_HUB_OFFLINE=1: --download will not work; use --check.", file=sys.stderr)
    raise SystemExit(main())
