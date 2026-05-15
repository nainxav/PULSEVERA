"""Pytest bootstrap: make ``src`` importable when tests run from any cwd."""

from __future__ import annotations

import sys
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))
