"""Allow ``python -m chronicle`` as an alias for the ``chronicle`` CLI."""

from __future__ import annotations

import sys

from chronicle.cli import main

if __name__ == "__main__":
    sys.exit(main())
