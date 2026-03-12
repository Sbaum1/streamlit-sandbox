# ============================================================
# FILE: streamlit_sandbox/certification/run_certification.py
# ROLE: CANONICAL v1 DETERMINISM & HASH CERTIFICATION HARNESS
# STANDARD: FORTUNE 100 / ZERO DRIFT / BASELINE-ALIGNED
# EXECUTION:
#   python -m streamlit_sandbox.certification.run_certification
# ============================================================

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from dataclasses import is_dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from streamlit_sandbox.execution.engine import run_all_models


# ------------------------------------------------------------
# CERTIFICATION CONSTANTS
# ------------------------------------------------------------

SEED = 42
HORIZON = 12
CONFIDENCE_LEVEL = 0.95
FLOAT_PRECISION = 10


# ------------------------------------------------------------
# SEED ENFORCEMENT
# ------------------------------------------------------------

def _enforce_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


# ------------------------------------------------------------
# DETERMINISTIC NORMALIZATION (BASELINE-ALIGNED)
# ------------------------------------------------------------

def _normalize_float(value: float) -> float:
    return round(float(value), FLOAT_PRECISION)


def _normalize_dataframe(df: pd.DataFrame) -> Any:
    df_copy = df.copy()

    # Normalize floats only (preserve column order exactly as produced)
    for col in df_copy.columns:
        if pd.api.types.is_float_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].apply(_normalize_float)

    # Use pandas JSON representation to preserve datetime formatting alignment
    json_str = df_copy.to_json(orient="split", date_format="iso")
    return json.loads(json_str)


def _normalize_object(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        return _normalize_float(obj)
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, list):
        return [_normalize_object(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _normalize_object(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, pd.DataFrame):
        return _normalize_dataframe(obj)
    if is_dataclass(obj):
        return _normalize_object(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _normalize_object(vars(obj))
    return str(obj)


def _deterministic_json(obj: Any) -> str:
    normalized = _normalize_object(obj)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _compute_hash_from_object(obj: Any) -> str:
    payload = _deterministic_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# CERTIFICATION EXECUTION
# ------------------------------------------------------------

def main():
    _enforce_seed()

    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "datasets"
    baseline_dir = base_dir / "baseline"

    overall_pass = True

    for dataset_path in sorted(dataset_dir.glob("*.csv")):
        dataset_name = dataset_path.stem
        print(f"DATASET: {dataset_name}")

        df = pd.read_csv(dataset_path)

        hashes = []

        for i in range(10):
            _enforce_seed()
            result = run_all_models(df, HORIZON, CONFIDENCE_LEVEL)
            h = _compute_hash_from_object(result)
            hashes.append(h)
            print(f"HASH_{i+1}: {h}")

        unique_hash_count = len(set(hashes))
        print(f"UNIQUE_HASH_COUNT: {unique_hash_count}")

        canonical_hash = hashes[0]

        baseline_file = baseline_dir / f"{dataset_name}.json"

        if not baseline_file.exists():
            print("BASELINE_HASH: MISSING")
            print(f"CANONICAL_HASH: {canonical_hash}")
            print("RESULT: FAIL")
            overall_pass = False
            continue

        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)

        baseline_hash = _compute_hash_from_object(baseline_data)

        print(f"BASELINE_HASH: {baseline_hash}")
        print(f"CANONICAL_HASH: {canonical_hash}")

        if unique_hash_count == 1 and baseline_hash == canonical_hash:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
            overall_pass = False

    if overall_pass:
        print("FINAL_VERDICT: CERTIFIED")
    else:
        print("FINAL_VERDICT: NOT CERTIFIED")


if __name__ == "__main__":
    main()