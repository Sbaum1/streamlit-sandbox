import pandas as pd
from typing import Iterator, Tuple

def rolling_windows(
    df: pd.DataFrame,
    min_train: int,
    horizon: int,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward generator.
    Yields (train_df, test_df).
    """
    df = df.sort_values("date").reset_index(drop=True)

    for end in range(min_train, len(df) - horizon + 1):
        train = df.iloc[:end].copy()
        test = df.iloc[end:end + horizon].copy()
        yield train, test

