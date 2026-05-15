"""Tiny CLI wrapper around :mod:`src.inference` for quick smoke-testing."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from src.inference import predict


SAMPLE_PROFILE = {
    "Gender": "Male",
    "age": 58,
    "education": "graduate",
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 0,
    "prevalentStroke": "no",
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 245,
    "sysBP": 152,
    "diaBP": 95,
    "BMI": 29.4,
    "heartRate": 84,
    "glucose": 105,
}


def main() -> None:
    [result] = predict(SAMPLE_PROFILE, explain=True)
    payload = asdict(result)
    payload["top_risk_factors"] = [asdict(f) for f in result.top_risk_factors]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
