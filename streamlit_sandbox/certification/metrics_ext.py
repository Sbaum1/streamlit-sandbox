import numpy as np
import pandas as pd

def smape(actual: pd.Series, forecast: pd.Series) -> float:
    denom = (actual.abs() + forecast.abs())
    denom = denom.replace(0, np.nan)
    return float((2 * (forecast - actual).abs() / denom).mean() * 100)

def beat_naive(
    model_mae: float,
    naive_mae: float,
) -> bool:
    return model_mae < naive_mae

