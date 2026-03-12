# FILE: intake/validator.py
# ROLE: DATA VALIDATION & INTEGRITY ENFORCEMENT
# STATUS: CANONICAL / EXECUTIVE-GRADE
# ==================================================

import pandas as pd


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates incoming dataset with full transparency.

    Governance:
    - No silent row drops
    - Explicit error reporting
    - Deterministic behavior
    """

    df = df.copy()
    df.columns = ["date", "value"]

    # Preserve original index for audit
    df["_row"] = df.index + 1

    # Parse with coercion
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Identify invalid rows
    invalid = df[df["date"].isna() | df["value"].isna()]

    if not invalid.empty:
        bad_rows = invalid["_row"].tolist()
        raise ValueError(
            f"Invalid data detected in rows {bad_rows}. "
            f"Each row must contain a valid date and numeric value. "
            f"No data has been discarded."
        )

    # Remove helper column
    df = df.drop(columns=["_row"])

    # Sort and enforce uniqueness
    df = df.sort_values("date")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Each date must be unique.")

    return df

