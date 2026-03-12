import numpy as np
import pandas as pd

def classify_regime(actual: pd.Series) -> str:
    """
    Conservative shock detector based on variance jump.
    """
    if len(actual) < 12:
        return "unknown"

    early = actual.iloc[: len(actual)//2]
    late = actual.iloc[len(actual)//2 :]

    if late.var() > early.var() * 1.5:
        return "shock"

    return "stable"

