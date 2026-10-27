"""Video-free memory (VFM): compiled, queryable memory from a Chronicle run.

The package is intentionally dependency-light.  Importing ``chronicle.vfm``
loads only the stdlib, so a vault can be queried on a machine with no vision
stack, no UI framework and no model weights.

Public entry points:

* :mod:`chronicle.vfm.query` — deterministic query path (stdlib only).
* :mod:`chronicle.vfm.compile` — source DB -> packed SQLite vault.
* :mod:`chronicle.vfm.memory_pack` — LLM-ready, timestamped memory pack.
* :mod:`chronicle.vfm.service` — optional job manager used by the demo UI.
  Import it explicitly; it is not needed for research.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DOWNLOAD_NAMES",
    "VFMJob",
    "VFMJobManager",
    "artifact",
    "artifacts",
    "compile_memory",
    "load_status",
    "memory_job_dir",
]

#: Lazily re-export the optional service layer so `import chronicle.vfm` never
#: pulls in the dashboard job manager or its report/zip helpers.
_SERVICE_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name not in _SERVICE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service

    value = getattr(service, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
