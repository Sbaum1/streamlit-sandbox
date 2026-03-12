# FILE: models/_utils.py
# ROLE: DATE NORMALIZATION (NON-DESTRUCTIVE)
# STATUS: LOCKED / ENGINE-SAFE
#
# GOVERNANCE:
# - Never inject missing periods
# - Never alter values
# - Never force frequency
# - Only sort + coerce datetime safely
# ==================================================

import pandas as pd


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes date handling without mutating the time series.

    Guarantees:
    - 'date' column is datetime64
    - Data is sorted chronologically
    - No asfreq()
    - No gap-filling
    - No value modification
    """

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'value' columns.")

    out = df.copy()

    # Coerce dates safely
    out["date"] = pd.to_datetime(out["date"], errors="raise")

    # Sort chronologically
    out = out.sort_values("date").reset_index(drop=True)

    return out

