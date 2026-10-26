"""NiceGUI dashboard package.

``create_app`` is imported lazily so pure helpers such as
``chronicle.ui.graph_view`` can be used (and tested) without the optional
``ui`` extra installed.
"""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    from chronicle.ui.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
