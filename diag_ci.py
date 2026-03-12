import pandas as pd
import numpy as np
from sentinel_engine import run_all_models

df = pd.read_csv('data/input.csv')
df['date'] = pd.to_datetime(df['date'])
results = run_all_models(df=df, horizon=12, confidence_level=0.90)

members = ['BSTS', 'ETS', 'Prophet', 'SARIMA', 'STL+ETS', 'TBATS', 'Theta']

for name in members:
    r = results.get(name, {})
    fdf = r.get('forecast_df')
    if fdf is None:
        print(f"{name}: no forecast_df")
        continue
    future = fdf[fdf['actual'].isna()]
    ci_low  = future['ci_low']
    ci_high = future['ci_high']
    print(f"{name}:")
    print(f"  ci_low  dtype={ci_low.dtype}  sample={ci_low.iloc[0]}")
    print(f"  ci_high dtype={ci_high.dtype}  sample={ci_high.iloc[0]}")