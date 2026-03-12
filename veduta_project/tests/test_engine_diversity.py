# ==================================================
# FILE: test_engine_diversity.py
# VERSION: 1.0.0
# ROLE: PHASE 3E — STAGE 4 DATASET DIVERSITY CERTIFICATION
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
# PURPOSE:
#   Certifies the engine against three dataset types.
#   Must achieve Elite ensemble MASE (< 0.70) on
#   at least 2 of 3 types to pass.
#
# DATASET TYPES:
#   Type 1 — Smooth trend (synthetic)
#   Type 2 — Volatile / intermittent (synthetic)
#   Type 3 — Regime change (data/input.csv or synthetic fallback)
#
# USAGE: (.venv) python test_engine_diversity.py
# ==================================================

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import pandas as pd
import numpy as np
from pathlib import Path

HORIZON         = 12
CONFIDENCE      = 0.90
SEASONAL_PERIOD = 12
MASE_ELITE      = 0.70
REQUIRED_PASSES = 2
SEED            = 42

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

print()
print("=" * 70)
print("  SENTINEL ENGINE v2.0.0 — STAGE 4 DATASET DIVERSITY (Phase 3E)")
print("=" * 70)

from sentinel_engine import run_all_models


def seasonal_naive_scale(train_vals, m):
    n = len(train_vals)
    errors = np.abs(train_vals[m:] - train_vals[:n - m])
    return float(np.mean(errors)) if len(errors) > 0 else 1.0


def ensemble_mase(results, actuals, scale):
    r   = results.get("Primary Ensemble", {})
    fdf = r.get("forecast_df")
    if fdf is None or fdf.empty:
        return None
    future = fdf[fdf["actual"].isna()]
    if future.empty:
        future = fdf.copy()
    fc = future["forecast"].values.astype(float)
    n  = min(len(actuals), len(fc))
    if n == 0:
        return None
    mae = np.mean(np.abs(actuals[:n] - fc[:n]))
    return mae / scale if scale > 0 else None


rng = np.random.default_rng(SEED)

# ── Type 1: Smooth trend ──────────────────────────────────────────────────────
print("\n[1] Type 1 — Smooth Trend")
n1     = 84
dates1 = pd.date_range("2017-01", periods=n1, freq="MS")
t1     = np.arange(n1, dtype=float)
vals1  = 200 + t1*1.5 + 30*np.sin(2*np.pi*t1/12) + rng.normal(0, 5, n1)
df1    = pd.DataFrame({"date": dates1, "value": vals1})
train1, act1 = df1.iloc[:-HORIZON], vals1[-HORIZON:]
scale1 = seasonal_naive_scale(train1["value"].values, SEASONAL_PERIOD)
mase1  = type1_pass = None

try:
    res1  = run_all_models(df=train1, horizon=HORIZON, confidence_level=CONFIDENCE)
    mase1 = ensemble_mase(res1, act1, scale1)
    type1_pass = mase1 is not None and mase1 < MASE_ELITE
    tier  = "Elite" if type1_pass else "Not Elite"
    icon  = PASS if type1_pass else FAIL
    print(f"  {icon}  Ensemble MASE: {mase1:.4f}  ({tier})")
except Exception as e:
    print(f"  {FAIL}  Engine failed: {e}")
    type1_pass = False

# ── Type 2: Volatile / intermittent ──────────────────────────────────────────
print("\n[2] Type 2 — Volatile / Intermittent")
n2        = 84
dates2    = pd.date_range("2017-01", periods=n2, freq="MS")
nz_count  = int(n2 * 0.65)
nz_vals   = rng.integers(5, 150, size=nz_count).astype(float)
z_vals    = np.zeros(n2 - nz_count)
vals2     = rng.permutation(np.concatenate([nz_vals, z_vals]))
zero_pct2 = float((vals2 == 0).mean())
df2       = pd.DataFrame({"date": dates2, "value": vals2})
train2, act2 = df2.iloc[:-HORIZON], vals2[-HORIZON:]
nz_train2 = train2["value"].values
nz_train2 = nz_train2[nz_train2 > 0]
scale2    = float(np.mean(nz_train2)) if len(nz_train2) > 0 else 1.0
mase2     = type2_pass = None

print(f"    Zero periods: {zero_pct2:.1%}")

try:
    res2  = run_all_models(df=train2, horizon=HORIZON, confidence_level=CONFIDENCE)
    mase2 = ensemble_mase(res2, act2, scale2)
    type2_pass = mase2 is not None and mase2 < MASE_ELITE
    tier       = "Elite" if type2_pass else "Not Elite"
    icon       = PASS if type2_pass else FAIL
    routing    = res2.get("Primary Ensemble", {}).get("metadata", {}).get("routing", "standard")
    print(f"  {icon}  Ensemble MASE: {mase2:.4f}  ({tier})  routing={routing}")
except Exception as e:
    print(f"  {FAIL}  Engine failed: {e}")
    type2_pass = False

# ── Type 3: Regime change ─────────────────────────────────────────────────────
print("\n[3] Type 3 — Regime Change")
mase3 = type3_pass = None
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "input.csv"

def run_type3(df3):
    global mase3, type3_pass
    df3 = df3.sort_values("date").reset_index(drop=True)
    train3, act3 = df3.iloc[:-HORIZON], df3["value"].values[-HORIZON:]
    scale3 = seasonal_naive_scale(train3["value"].values, SEASONAL_PERIOD)
    res3   = run_all_models(df=train3, horizon=HORIZON, confidence_level=CONFIDENCE)
    mase3  = ensemble_mase(res3, act3, scale3)
    type3_pass = mase3 is not None and mase3 < MASE_ELITE
    tier   = "Elite" if type3_pass else "Not Elite"
    icon   = PASS if type3_pass else FAIL
    print(f"  {icon}  Ensemble MASE: {mase3:.4f}  ({tier})")
    # Prophet flag check
    pr = res3.get("Prophet", {})
    if pr.get("status") == "success":
        pfdf = pr.get("forecast_df")
        if pfdf is not None:
            pf = pfdf[pfdf["actual"].isna()]["forecast"].values.astype(float)
            n  = min(len(act3), len(pf))
            pm = np.mean(np.abs(act3[:n] - pf[:n])) / scale3 if scale3 > 0 else None
            if pm:
                flag = "FLAGGED" if pm >= 1.0 else "OK"
                print(f"       Prophet MASE: {pm:.4f}  ({flag})")

try:
    if DATA_FILE.exists():
        df3 = pd.read_csv(DATA_FILE)
        df3["date"]  = pd.to_datetime(df3["date"])
        df3["value"] = pd.to_numeric(df3["value"])
        print(f"    Source: data/input.csv  ({len(df3)} obs)")
        run_type3(df3)
    else:
        print(f"  {WARN} data/input.csv not found — synthetic regime-change fallback")
        n3     = 72
        dates3 = pd.date_range("2018-01", periods=n3, freq="MS")
        t3     = np.arange(n3, dtype=float)
        regime = np.where(t3 < 24, 1.0, 0.55)
        vals3  = 500*regime + t3*0.8 + 40*np.sin(2*np.pi*t3/12) + rng.normal(0, 15, n3)
        df3    = pd.DataFrame({"date": dates3, "value": vals3})
        run_type3(df3)
except Exception as e:
    print(f"  {FAIL}  Dataset 3 failed: {e}")
    type3_pass = False

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  STAGE 4 SUMMARY")
print()

types = [
    ("Type 1 — Smooth trend",         mase1, type1_pass),
    ("Type 2 — Volatile/intermittent", mase2, type2_pass),
    ("Type 3 — Regime change",         mase3, type3_pass),
]

passes = sum(1 for _, _, p in types if p)
for name, mase, passed in types:
    ms   = f"{mase:.4f}" if mase is not None else "  N/A "
    tier = "Elite    " if (mase is not None and mase < MASE_ELITE) else "Not Elite"
    icon = PASS if passed else (WARN if mase is None else FAIL)
    print(f"  {icon}  {name:<38} MASE={ms}  {tier}")

print()
print(f"  Elite on {passes}/3  (required: {REQUIRED_PASSES})")

if passes >= REQUIRED_PASSES:
    print()
    print(f"  {PASS} STAGE 4 DATASET DIVERSITY CERTIFICATION PASSED")
    print("  Ready for Stage 5 — Tier Config Certification.")
else:
    print()
    print(f"  {FAIL} STAGE 4 DATASET DIVERSITY CERTIFICATION FAILED")
    sys.exit(1)

print("=" * 70)
print()
