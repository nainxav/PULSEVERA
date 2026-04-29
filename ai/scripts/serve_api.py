"""Boot the FastAPI service from a single command.

Run with:
    python -m ai.scripts.serve_api
or:
    cd ai && uvicorn src.api:app --reload
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
