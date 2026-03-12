import hashlib
import pandas as pd

def hash_dataframe(df: pd.DataFrame) -> str:
    data_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()[:12]
