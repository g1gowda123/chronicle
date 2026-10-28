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


def _demo_memory_report() -> dict[str, Any]:
    """Open the bundled memory and prove that it can answer a graph query."""

    try:
        path = demo_db_path()
    except FileNotFoundError as exc:
        return {
            "path": str(exc),
            "exists": False,
            "query_ok": False,
            "answer": None,
        }

    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "query_ok": False,
        "answer": None,
    }
    if not report["exists"]:
        return report

    try:
        from chronicle.graph.memory import ChronicleMemory

        with ChronicleMemory(path, read_only=True) as memory:
            result = memory.query("where is my cup?")
            report["answer"] = result.fallback_answer() if result is not None else None
            report["query_ok"] = bool(result is not None and report["answer"])
    except Exception as exc:  # pragma: no cover - corrupt/partial package data
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def collect_report() -> dict[str, Any]:
    core = _package_report(CORE_PACKAGES)
    optional = _package_report(OPTIONAL_PACKAGES)
    memory = _demo_memory_report()
    core_ok = all(item["installed"] for item in core.values())
    memory_ok = bool(memory["exists"] and memory["query_ok"])
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core": core,
        "optional": optional,
        "demo_memory": memory,
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
    memory_state = "ok" if memory["exists"] else "missing"
    query_state = "ok" if memory.get("query_ok") else "failed"
    lines.extend(
        [
            "",
            "Demo memory",
            f"  [{memory_state}] {memory['path']}",
            f"  [{query_state}] query engine",
        ]
    )
    if memory.get("answer"):
        lines.append(f"       {memory['answer']}")
    if memory.get("error"):
        lines.append(f"       {memory['error']}")
    lines.extend(["", "Model weights"])
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
    if report["ok"]:
        status = "Status: ready for model-free use"
    elif all(item["installed"] for item in report["core"].values()):
        status = "Status: demo memory is not usable"
    else:
        status = "Status: core installation is incomplete"
    lines.extend(["", status])
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
