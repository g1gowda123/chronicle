"""Environment diagnostics for Chronicle.

``chronicle doctor`` is deliberately dependency-light: it can run before the
perception or UI extras are installed and tells the user which layer is missing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from chronicle.demo_data import demo_db_path

# module name -> distribution name (None when the import name is enough)
CORE_PACKAGES: dict[str, str | None] = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "shapely": "shapely",
    "networkx": "networkx",
    "PIL": "pillow",
}
OPTIONAL_PACKAGES: dict[str, str | None] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "mediapipe": "mediapipe",
    "open_clip": "open-clip-torch",
    "transformers": "transformers",
    "nicegui": "nicegui",
    "fastapi": "fastapi",
}


def _version(module: str, distribution: str | None) -> str | None:
    if importlib.util.find_spec(module) is None:
        return None
    names = [distribution, module] if distribution else [module]
    for name in names:
        if not name:
            continue
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        imported = __import__(module)
        value = getattr(imported, "__version__", None)
        return str(value) if value else "installed"
    except Exception:
        return "installed"


def _package_report(packages: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for module, distribution in packages.items():
        version = _version(module, distribution)
        report[module] = {"installed": version is not None, "version": version}
    return report


def _model_weights() -> list[str]:
    roots = []
    configured = os.environ.get("CHRONICLE_MODEL_DIR")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(
        [
            Path.cwd() / "models",
            Path.cwd() / "data" / "models",
            Path.home() / ".cache" / "chronicle" / "models",
        ]
    )
    found: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in {".pt", ".pth", ".safetensors", ".bin", ".onnx"}:
                found.append(str(path))
    return found


def _cuda_report() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device": torch.cuda.get_device_name(0) if available else None,
            "torch": str(torch.__version__),
        }
    except Exception as exc:  # pragma: no cover - platform/driver dependent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def collect_report() -> dict[str, Any]:
    core = _package_report(CORE_PACKAGES)
    optional = _package_report(OPTIONAL_PACKAGES)
    try:
        memory = str(demo_db_path())
        memory_ok = True
    except FileNotFoundError as exc:
        memory = str(exc)
        memory_ok = False
    core_ok = all(item["installed"] for item in core.values())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core": core,
        "optional": optional,
        "demo_memory": {"path": memory, "exists": memory_ok},
        "model_weights": _model_weights(),
        "cuda": _cuda_report(),
        "ok": core_ok and memory_ok,
    }


def _line(name: str, item: dict[str, Any], *, optional: bool = False) -> str:
    if item["installed"]:
        return f"  [ok] {name} {item['version']}"
    suffix = " (optional)" if optional else " (required)"
    return f"  [missing] {name}{suffix}"


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "Chronicle doctor",
        f"  python: {report['python']}",
        f"  platform: {report['platform']}",
        "",
        "Core dependencies",
    ]
    lines.extend(_line(name, item) for name, item in report["core"].items())
    lines.extend(["", "Optional feature dependencies"])
    lines.extend(
        _line(name, item, optional=True) for name, item in report["optional"].items()
    )
    memory = report["demo_memory"]
    lines.extend(
        [
            "",
            "Demo memory",
            f"  [{'ok' if memory['exists'] else 'missing'}] {memory['path']}",
            "",
            "Model weights",
        ]
    )
    if report["model_weights"]:
        lines.extend(f"  [ok] {path}" for path in report["model_weights"])
    else:
        lines.append(
            "  [--] none found; the perception extra downloads weights on first run"
        )
    cuda = report["cuda"]
    if cuda.get("available"):
        lines.append(f"  [ok] CUDA: {cuda.get('device')}")
    else:
        lines.append(f"  [--] CUDA: {cuda.get('reason', 'not available')}")
    lines.extend(
        [
            "",
            "Status: ready for model-free use"
            if report["ok"]
            else "Status: core installation is incomplete",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the Chronicle installation.")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)
    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
