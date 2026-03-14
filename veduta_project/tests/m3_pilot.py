# ==================================================
# FILE: veduta_project/tests/m3_pilot.py
# VERSION: 1.0.0
# ROLE: M3 PILOT RUN — 30 SERIES
# ENGINE: Sentinel Engine v2.1.0
# ==================================================
#
# PURPOSE:
#   Runs the VEDUTA engine against 30 M3 monthly series
#   as a pre-flight check before the full 1,428-series run.
#
#   The 30 series are drawn stratified by length bucket:
#     Short  (n < 60)  : 10 series
#     Medium (60-99)   : 10 series
#     Long   (n >= 100): 10 series
#
#   This ensures the pilot exercises the engine across
#   the full range of series lengths present in M3 monthly.
#
# OUTPUT:
#   tests/m3_results/pilot_results.json
#   Contains per-series MASE, sMAPE, and ensemble metadata.
#   Summary statistics printed to console.
#
# USAGE:
#   cd veduta_project
#   python tests/m3_pilot.py
#
# PASS CRITERIA:
#   Median MASE across 30 series <= 1.0
#   (Beats seasonal naïve on more than half the series)
#   This is the minimum bar before full M3 run.
#   Target for full run: median MASE <= 0.70.
#
# GOVERNANCE:
#   - Engine called via run_primary_ensemble only
#   - No per-series tuning — engine runs blind
#   - Forecast horizon = 18 (M3 standard)
#   - Results written atomically (temp file then rename)
# ==================================================

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.m3_loader import load_m3_monthly, compute_mase, compute_smape
from sentinel_engine.ensemble import run_primary_ensemble

# ── Config ───────────────────────────────────────────────────────────────────
PILOT_N          = 30
CONFIDENCE_LEVEL = 0.80
HORIZON          = 18
OUTPUT_DIR       = SCRIPT_DIR / "m3_results"
OUTPUT_FILE      = OUTPUT_DIR / "pilot_results.json"
PASS_MASE        = 1.0   # median MASE target for pilot


def _stratified_sample(series_list: List[Dict], n: int = 30) -> List[Dict]:
    """
    Draw n series stratified by length bucket.
    Short <60, Medium 60-99, Long >=100.
    """
    short  = [s for s in series_list if s["n_train"] < 60]
    medium = [s for s in series_list if 60 <= s["n_train"] < 100]
    long_  = [s for s in series_list if s["n_train"] >= 100]

    per_bucket = n // 3
    remainder  = n - per_bucket * 3

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    sampled = []
    for bucket, extra in zip([short, medium, long_], [remainder, 0, 0]):
        k = min(per_bucket + extra, len(bucket))
        idx = rng.choice(len(bucket), size=k, replace=False)
        sampled.extend([bucket[i] for i in sorted(idx)])

    # If any bucket was too small, fill from others
    if len(sampled) < n:
        remaining = [s for s in series_list if s not in sampled]
        fill = rng.choice(len(remaining), size=n - len(sampled), replace=False)
        sampled.extend([remaining[i] for i in fill])

    return sampled[:n]


def run_pilot(verbose: bool = True) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load M3 ──────────────────────────────────────────────────────────────
    all_series = load_m3_monthly(verbose=verbose)
    pilot      = _stratified_sample(all_series, PILOT_N)

    if verbose:
        buckets = {"short": 0, "medium": 0, "long": 0}
        for s in pilot:
            n = s["n_train"]
            if n < 60:         buckets["short"]  += 1
            elif n < 100:      buckets["medium"] += 1
            else:              buckets["long"]   += 1
        print(f"\nPilot sample: {len(pilot)} series")
        print(f"  Short (<60):    {buckets['short']}")
        print(f"  Medium (60-99): {buckets['medium']}")
        print(f"  Long (>=100):   {buckets['long']}")
        print(f"\nRunning engine (horizon={HORIZON}, CI={CONFIDENCE_LEVEL})...\n")

    # ── Run engine ───────────────────────────────────────────────────────────
    series_results = []
    mase_values    = []

    for i, series in enumerate(pilot, 1):
        sid     = series["id"]
        df      = series["df"]
        actuals = series["actuals"][:HORIZON]
        horizon = min(HORIZON, len(actuals))

        t0 = time.time()
        try:
            result = run_primary_ensemble(
                df               = df,
                horizon          = horizon,
                confidence_level = CONFIDENCE_LEVEL,
            )

            # Extract forecast values
            fc_df    = result.forecast_df.copy()
            fc_df["date"] = pd.to_datetime(fc_df["date"])
            last_train    = df["date"].max()
            future        = fc_df[fc_df["date"] > last_train].reset_index(drop=True)

            if len(future) < horizon:
                raise RuntimeError(f"Forecast too short: {len(future)} < {horizon}")

            forecast = future["forecast"].astype("float64").values[:horizon]

            if not np.isfinite(forecast).all():
                raise RuntimeError("Non-finite forecast values")

            y_train = df["value"].astype("float64").values
            mase    = compute_mase(actuals, forecast, y_train)
            smape   = compute_smape(actuals, forecast)
            elapsed = round(time.time() - t0, 2)

            status = "PASS" if mase < 2.0 else "WARN"
            mase_values.append(mase)

            entry = {
                "id":              sid,
                "status":          status,
                "mase":            round(float(mase), 4),
                "smape":           round(float(smape), 4),
                "n_train":         series["n_train"],
                "horizon":         horizon,
                "duration_s":      elapsed,
                "aggregation":     result.metadata.get("aggregation_method", "unknown"),
                "stacker_active":  result.metadata.get("stacker_active", False),
                "models_used":     result.metadata.get("component_count_valid", None),
                "error":           None,
            }

            if verbose:
                flag = "✓" if mase < 1.0 else ("~" if mase < 2.0 else "✗")
                print(f"  [{i:2d}/{len(pilot)}] {sid:6s}  "
                      f"MASE={mase:.4f} {flag}  "
                      f"sMAPE={smape:.4f}  "
                      f"n={series['n_train']:3d}  "
                      f"{elapsed:.1f}s")

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            entry   = {
                "id":         sid,
                "status":     "CRASH",
                "mase":       None,
                "smape":      None,
                "n_train":    series["n_train"],
                "horizon":    horizon,
                "duration_s": elapsed,
                "error":      str(e),
            }
            if verbose:
                print(f"  [{i:2d}/{len(pilot)}] {sid:6s}  CRASH  {e}")

        series_results.append(entry)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid_mase = [r["mase"] for r in series_results if r["mase"] is not None]
    n_pass     = sum(1 for r in series_results if r["status"] == "PASS")
    n_warn     = sum(1 for r in series_results if r["status"] == "WARN")
    n_crash    = sum(1 for r in series_results if r["status"] == "CRASH")

    median_mase = float(np.median(valid_mase)) if valid_mase else None
    mean_mase   = float(np.mean(valid_mase))   if valid_mase else None
    pilot_pass  = median_mase is not None and median_mase <= PASS_MASE

    summary = {
        "pilot_series":     len(pilot),
        "valid_series":     len(valid_mase),
        "n_pass":           n_pass,
        "n_warn":           n_warn,
        "n_crash":          n_crash,
        "median_mase":      round(median_mase, 4) if median_mase else None,
        "mean_mase":        round(mean_mase,   4) if mean_mase   else None,
        "pilot_passed":     pilot_pass,
        "pass_threshold":   PASS_MASE,
        "target_full_run":  0.70,
    }

    output = {
        "summary":  summary,
        "results":  series_results,
        "config":   {
            "horizon":          HORIZON,
            "confidence_level": CONFIDENCE_LEVEL,
            "engine_version":   "2.1.0",
            "sampling_seed":    42,
        },
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"PILOT RESULTS")
        print(f"{'='*55}")
        print(f"  Series evaluated : {len(valid_mase)} / {len(pilot)}")
        print(f"  Median MASE      : {median_mase:.4f}" if median_mase else "  Median MASE      : N/A")
        print(f"  Mean MASE        : {mean_mase:.4f}"   if mean_mase   else "  Mean MASE        : N/A")
        print(f"  PASS (MASE<2.0)  : {n_pass}")
        print(f"  WARN (MASE>=2.0) : {n_warn}")
        print(f"  CRASH            : {n_crash}")
        print(f"  Pilot gate       : {'PASSED ✓' if pilot_pass else 'FAILED ✗'}  "
              f"(threshold: median MASE <= {PASS_MASE})")
        print(f"  Full run target  : median MASE <= 0.70")
        print(f"{'='*55}\n")

    # ── Write results atomically ──────────────────────────────────────────────
    tmp = OUTPUT_DIR / "_pilot_tmp.json"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    tmp.rename(OUTPUT_FILE)
    if verbose:
        print(f"Results written to: {OUTPUT_FILE}")

    return output


if __name__ == "__main__":
    run_pilot(verbose=True)
