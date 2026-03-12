# ============================================================
# FILE: streamlit_sandbox/certification/run_certification.py
# ROLE: CANONICAL v1 DETERMINISM & HASH CERTIFICATION HARNESS
# STANDARD: FORTUNE 100 / ZERO DRIFT / HASH PARITY ENFORCED
# ============================================================

# ------------------------------------------------------------
# LOGGING SUPPRESSION (GLOBAL)
# ------------------------------------------------------------

import logging
logging.disable(logging.CRITICAL)

# ------------------------------------------------------------
# GLOBAL SEED INJECTION (EXECUTES ONCE)
# ------------------------------------------------------------

import random
import numpy as np

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ------------------------------------------------------------
# STANDARD IMPORTS
# ------------------------------------------------------------

import json
import hashlib
import pandas as pd

from streamlit_sandbox.execution.engine import run_all_models, MODEL_REGISTRY
from streamlit_sandbox.certification.performance_suite import (
    run_performance_suite,
)
from streamlit_sandbox.certification.dataset_factory import seasonal_dataset

# ------------------------------------------------------------
# CANONICAL NORMALIZATION + HASHING (SERIALIZATION SAFE)
# ------------------------------------------------------------

def _hash_canonical_output(df: pd.DataFrame) -> str:

    df = df.copy()

    if "date" not in df.columns:
        raise RuntimeError("Canonical forecast missing 'date' column.")

    # ISO-8601 Date (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Sort rows by date ascending
    df = df.sort_values("date").reset_index(drop=True)

    # Sort columns alphabetically
    df = df[sorted(df.columns)]

    # Float normalization (10 decimal places, Python float)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype("float64")
                .round(10)
                .apply(lambda x: float(x) if pd.notna(x) else None)
            )
        else:
            df[col] = df[col].where(pd.notna(df[col]), None)

    records = df.to_dict(orient="records")

    payload = json.dumps(
        records,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ------------------------------------------------------------
# CERTIFICATION EXECUTION
# ------------------------------------------------------------

def run_certification():

    run_signature = {
        "certification_mode": True,
        "global_seed": GLOBAL_SEED,
    }

    df = seasonal_dataset()
    horizon = 6
    confidence = 0.95

    hash_list = []

    for _ in range(10):

        results = run_all_models(
            df=df,
            horizon=horizon,
            confidence_level=confidence,
        )

        canonical = results.get("Primary Ensemble")
        if canonical is None or canonical.get("forecast_df") is None:
            return {
                "STRUCTURAL": "REJECTED",
                "FINAL_VERDICT": "REJECTED",
                "run_signature": run_signature,
            }

        forecast_df = canonical["forecast_df"]
        hash_value = _hash_canonical_output(forecast_df)
        hash_list.append(hash_value)

    unique_hash_count = len(set(hash_list))

    if unique_hash_count != 1:
        return {
            "STRUCTURAL": "REJECTED",
            "FINAL_VERDICT": "REJECTED",
            "run_signature": run_signature,
        }

    run_performance_suite()

    return {
        "STRUCTURAL": "CERTIFIED",
        "FINAL_VERDICT": "CERTIFIED",
        "run_signature": run_signature,
    }


if __name__ == "__main__":
    result = run_certification()
    print(result)