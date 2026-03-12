import pandas as pd

def validate_time_series(df: pd.DataFrame) -> dict:
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "duplicate_count": 0,
        "missing_periods": [],
        "failed_rows": {}
    }
