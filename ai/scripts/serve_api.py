"""Boot the FastAPI service from a single command.

Run with:
    python -m ai.scripts.serve_api
or:
    cd ai && uvicorn src.api:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

_AI_DIR = Path(__file__).resolve().parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))


def main() -> None:
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(_AI_DIR))


if __name__ == "__main__":
    main()
