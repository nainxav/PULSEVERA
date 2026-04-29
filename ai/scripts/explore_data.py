"""Week 1 helper: explore the dataset from the modelling perspective.

Prints a quick summary (rows, target distribution, missing values) and saves
a JSON snapshot to ``ai/reports/metrics/dataset_summary.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from src import config
from src.preprocessing import basic_clean, load_raw_dataset, summarise


def main() -> None:
    config.ensure_directories()

    frame = basic_clean(load_raw_dataset())
    summary = summarise(frame)

    print("Dataset summary")
    print("---------------")
    print(f"Rows:           {summary.n_rows}")
    print(f"Features:       {summary.n_features}")
    print(f"Target dist.:   {summary.target_distribution}")
    print(f"Missing values: {summary.missing_values}")

    output_path = config.METRICS_DIR / "dataset_summary.json"
    output_path.write_text(json.dumps(asdict(summary), indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
