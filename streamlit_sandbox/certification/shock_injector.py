# ==================================================
# FILE: streamlit_sandbox/certification/shock_injector.py
# ROLE: DETERMINISTIC SHOCK INJECTOR
# STANDARD: NO RANDOMNESS / PURE TRANSFORMATION
# ==================================================

from __future__ import annotations

import pandas as pd


def inject_spike(
    df: pd.DataFrame,
    magnitude: float = 0.40,
    duration: int = 3,
) -> pd.DataFrame:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'value'.")

    df = df.copy().sort_values("date").reset_index(drop=True)

    midpoint = len(df) // 2
    end = midpoint + duration

    df.loc[midpoint:end - 1, "value"] *= (1 + magnitude)

    return df