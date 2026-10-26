"""Video-free memory (VFM): compiled, queryable memory from a Chronicle run.

Stable package for the main dashboard integration.  The modules are copied
from ``experiments/video-free-memory/prototype`` so the dashboard never depends
on the experiment directory or on ``sys.path`` tricks.

Public entry points:

* :class:`~chronicle.vfm.service.VFMJobManager` — background compile/ask service.
* :func:`~chronicle.vfm.service.compile_memory` — synchronous compile.
* :mod:`chronicle.vfm.query` — deterministic query path (stdlib only).
* :mod:`chronicle.vfm.nlq` — optional Qwen phrasing over the memory pack.
"""

from __future__ import annotations

from .service import (
    DOWNLOAD_NAMES,
    VFMJob,
    VFMJobManager,
    artifact,
    artifacts,
    compile_memory,
    load_status,
    memory_job_dir,
)

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
