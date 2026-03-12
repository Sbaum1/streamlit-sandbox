import hashlib
import pandas as pd

def fingerprint_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

