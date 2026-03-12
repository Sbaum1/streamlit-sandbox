# ==================================================
# FILE: test_engine_smoke.py
# VERSION: 3.0.0
# ROLE: PHASE 3E — STAGE 1 SMOKE TEST
# ENGINE: Sentinel Engine v2.0.0
# UPDATED: PHASE 3C/3D — 7 NEW MODELS
# ==================================================
# USAGE: (.venv) python test_engine_smoke.py
# PASS CRITERIA:
#   - All non-diagnostic models return success or
#     graceful exclusion (no silent hangs or crashes)
#   - Primary Ensemble returns complete forecast
#   - No non-finite values in any forecast output
#   - Phase 3D governance metadata present
# ==================================================

import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE  = Path("data/input.csv")
HORIZON    = 12
CONFIDENCE = 0.90

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

print()
print("=" * 70)
print("  SENTINEL ENGINE v2.0.0 — STAGE 1 SMOKE TEST  (Phase 3E v3.0.0)")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Data Loading")

df = None
try:
    df = pd.read_csv(DATA_FILE)
    assert "date"  in df.columns
    assert "value" in df.columns
    assert len(df) >= 24
    assert not df.isnull().any().any()
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    print(f"  {PASS}  input.csv loaded — {len(df)} observations, "
          f"{df['date'].min().strftime('%Y-%m')} → "
          f"{df['date'].max().strftime('%Y-%m')}")
    results.append(("Data load", True, None))
except Exception as e:
    print(f"  {FAIL}  Data load failed: {e}")
    sys.exit(1)

# ── Engine execution ──────────────────────────────────────────────────────────
print("\n[2] Engine Execution")

engine_results = None
try:
    from sentinel_engine import run_all_models
    engine_results = run_all_models(
        df=df, horizon=HORIZON, confidence_level=CONFIDENCE,
    )
    print(f"  {PASS}  run_all_models() completed without exception")
    results.append(("run_all_models executes", True, None))
except Exception as e:
    print(f"  {FAIL}  run_all_models() raised: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Engine metadata ───────────────────────────────────────────────────────────
print("\n[3] Engine Metadata")
try:
    meta = engine_results.get("_engine")
    assert meta is not None
    assert meta["engine_version"] == "2.0.0"
    assert meta["horizon"] == HORIZON
    print(f"  {PASS}  Engine metadata correct")
    print(f"       Version: {meta['engine_version']}  |  "
          f"Attempted: {meta['models_attempted']}  |  "
          f"Succeeded: {meta['models_succeeded']}")
    results.append(("Engine metadata", True, None))
except Exception as e:
    print(f"  {FAIL}  Engine metadata: {e}")
    results.append(("Engine metadata", False, str(e)))

# ── Per-model results ─────────────────────────────────────────────────────────
print("\n[4] Per-Model Results")

# VAR and Croston_SBA expected to be gracefully excluded on single-series input
EXPECTED_MODELS = [
    "Primary Ensemble",
    "ARIMA", "BSTS", "ETS", "Naive",
    "Prophet", "SARIMA", "SARIMAX",
    "STL+ETS", "TBATS", "Theta", "X-13",
    "HW_Damped", "DHR", "NNETAR", "LightGBM", "GARCH",
]
GRACEFUL_EXCLUSION = {"VAR", "Croston_SBA"}

for model_name in EXPECTED_MODELS:
    if model_name not in engine_results:
        print(f"  {FAIL}  {model_name:<22} — MISSING from results")
        results.append((f"{model_name} present", False, "Missing"))
        continue

    result = engine_results[model_name]
    status = result.get("status", "unknown")

    if status == "failed":
        err = result.get("error", "unknown error")
        print(f"  {WARN} {model_name:<22} — FAILED: {str(err)[:60]}")
        results.append((f"{model_name} status", False, str(err)))
        continue

    if status != "success":
        print(f"  {FAIL}  {model_name:<22} — Unknown status: {status}")
        results.append((f"{model_name} status", False, status))
        continue

    fdf = result.get("forecast_df")
    if fdf is None or fdf.empty:
        print(f"  {FAIL}  {model_name:<22} — Empty forecast_df")
        results.append((f"{model_name} forecast_df", False, "Empty"))
        continue

    forecast_vals = fdf["forecast"].dropna().values
    if len(forecast_vals) > 0 and not np.isfinite(forecast_vals).all():
        print(f"  {FAIL}  {model_name:<22} — Non-finite forecast values")
        results.append((f"{model_name} finite", False, "Non-finite"))
        continue

    future_rows = fdf[fdf["actual"].isna()]
    print(f"  {PASS}  {model_name:<22} — {len(fdf)} total rows, "
          f"{len(future_rows)} future periods")
    results.append((f"{model_name} success", True, None))

print()
for model_name in sorted(GRACEFUL_EXCLUSION):
    status = engine_results.get(model_name, {}).get("status", "not in results")
    print(f"  {WARN} {model_name:<22} — status={status} "
          f"(graceful exclusion expected on single-series input)")

# ── Primary Ensemble deep check ───────────────────────────────────────────────
print("\n[5] Primary Ensemble Deep Check")
try:
    r   = engine_results.get("Primary Ensemble", {})
    fdf = r.get("forecast_df")
    assert fdf is not None and not fdf.empty
    for col in ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high"]:
        assert col in fdf.columns, f"Missing column: {col}"
    future = fdf[fdf["actual"].isna()]
    assert len(future) == HORIZON, f"Expected {HORIZON} future rows, got {len(future)}"
    assert np.isfinite(future["forecast"].values).all()
    assert (future["ci_high"] >= future["ci_low"]).all()
    meta  = r.get("metadata", {})
    valid = meta.get("component_count_valid", 0)
    assert valid >= 2, f"Quorum not met: {valid}"
    spread = (future["ci_high"] - future["ci_low"]).mean()
    assert spread > 0
    print(f"  {PASS}  Ensemble forecast complete — {valid} members, avg CI width {spread:.2f}")
    print(f"       Method       : {meta.get('aggregation_method', 'unknown')}")
    print(f"       Diversity cap: {meta.get('diversity_cap_applied')}")
    print(f"       Family wts   : {meta.get('family_weights', {})}")
    results.append(("Ensemble deep check", True, None))
except Exception as e:
    print(f"  {FAIL}  Ensemble deep check: {e}")
    results.append(("Ensemble deep check", False, str(e)))

# ── Phase 3D governance checks ────────────────────────────────────────────────
print("\n[6] Phase 3D Governance Metadata")
try:
    meta = engine_results.get("Primary Ensemble", {}).get("metadata", {})
    assert "diversity_cap_applied"  in meta
    assert "excluded_mase_threshold" in meta
    assert "family_weights"          in meta
    assert "intermittent_routing"    in meta
    print(f"  {PASS}  All Phase 3D governance keys present")
    print(f"       ARIMA cap applied  : {meta.get('diversity_cap_applied')}")
    print(f"       MASE auto-excluded : {meta.get('excluded_mase_threshold', [])}")
    print(f"       Intermittent route : {meta.get('intermittent_routing')}")
    results.append(("Phase 3D metadata", True, None))
except Exception as e:
    print(f"  {FAIL}  Phase 3D metadata: {e}")
    results.append(("Phase 3D metadata", False, str(e)))

# ── Failure log ───────────────────────────────────────────────────────────────
failures = engine_results.get("_failures", [])
print(f"\n[7] Failure Log")
if failures:
    print(f"  {len(failures)} model(s) failed:")
    for f in failures:
        print(f"  {WARN} {f['model_name']:<22} — "
              f"{f['error_type']}: {f['error_message'][:80]}")
else:
    print(f"  {PASS}  No model failures recorded")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")

if failed == 0:
    print()
    print(f"  {PASS} SMOKE TEST PASSED  (Phase 3E Stage 1 v3.0.0)")
    print("  Ready for Stage 2 — SHA-256 Reproducibility.")
else:
    print()
    print(f"  {FAIL} SMOKE TEST FAILURES DETECTED")
    for label, ok, err in results:
        if not ok:
            print(f"    - {label}: {err}")
    sys.exit(1)

print("=" * 70)
print()
