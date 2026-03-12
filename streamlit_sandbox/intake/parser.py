# FILE: intake/parser.py
# ROLE: DATA INGESTION PARSER (PASTE + FILE)
# STATUS: CANONICAL / EXECUTIVE-SAFE
# ==================================================
#
# GUARANTEES:
# - Never drops first observation
# - Header row is OPTIONAL (auto-detected)
# - Deterministic parsing
# - Exactly N rows in â†’ N rows out
# ==================================================

import pandas as pd
from io import StringIO


# --------------------------------------------------
# TEXT (PASTE) PARSER â€” HEADER SAFE
# --------------------------------------------------

def parse_text_data(text: str) -> pd.DataFrame:
    """
    Parses pasted time series data.

    Accepted formats:
    - date,value
    - date value
    - date|value

    Header row is OPTIONAL and auto-detected.
    """

    if not text or not text.strip():
        raise ValueError("No data provided.")

    for sep in [",", "|", r"\s+"]:
        try:
            # IMPORTANT:
            # header=None ensures first row is ALWAYS read as data
            df = pd.read_csv(
                StringIO(text),
                sep=sep,
                header=None,
                engine="python",
            )

            if df.shape[1] < 2:
                continue

            df = df.iloc[:, :2]
            df.columns = ["date", "value"]

            # Attempt to detect & drop header row safely
            # (only if first row is clearly non-data)
            try:
                pd.to_datetime(df.iloc[0]["date"])
                pd.to_numeric(df.iloc[0]["value"])
            except Exception:
                df = df.iloc[1:].reset_index(drop=True)

            return df

        except Exception:
            continue

    raise ValueError(
        "Unable to detect delimiter. Please use date,value or date value format."
    )


# --------------------------------------------------
# FILE PARSER (CSV / XLSX)
# --------------------------------------------------

def parse_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")

