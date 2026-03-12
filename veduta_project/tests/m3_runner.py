# ==================================================
# FILE: veduta_project/tests/m3_runner.py
# VERSION: 1.0.0
# ROLE: FULL M3 BENCHMARK RUN — 1,428 MONTHLY SERIES
# ENGINE: Sentinel Engine v2.1.0
# ==================================================
#
# PURPOSE:
#   Runs the VEDUTA engine against all 1,428 M3 monthly
#   series and produces a certified leaderboard-comparable
#   results file.
#
# USAGE:
#   cd veduta_project
#   python tests/m3_runner.py
#
#   Options:
#     --resume          Resume from last checkpoint (skip completed series)
#     --max N           Run only first N series (for testing)
#     --verbose         Print per-series results
#
# OUTPUT:
#   tests/m3_results/m3_full_results.json   — complete per-series results
#   tests/m3_results/m3_summary.json        — leaderboard comparison table
#   tests/m3_results/checkpoint.json        — progress checkpoint (auto-saved)
#
# LEADERBOARD TARGETS (M3 monthly, MASE):
#   Seasonal Naïve baseline : 1.000
#   Theta (M3 winner 2000)  : ~0.850
#   Forecast Pro            : ~0.830
#   Modern ensemble target  : ~0.700
#   VEDUTA target           : <= 0.700
#
# GOVERNANCE:
#   - Engine called via run_primary_ensemble only — no per-series tuning
#   - Hold-out actuals never touched until after forecast is generated
#   - Checkpoint saved every 50 series — safe to interrupt and resume
#   - All results written atomically
#   - SHA-256 of results file written to m3_summary.json for auditability
# ==================================================

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.m3_loader import load_m3_monthly, compute_mase, compute_smape
from sentinel_engine.ensemble import run_primary_ensemble

# ── Config ───────────────────────────────────────────────────────────────────
CONFIDENCE_LEVEL  = 0.80
HORIZON           = 18
CHECKPOINT_EVERY  = 50
OUTPUT_DIR        = SCRIPT_DIR / "m3_results"
RESULTS_FILE      = OUTPUT_DIR / "m3_full_results.json"
SUMMARY_FILE      = OUTPUT_DIR / "m3_summary.json"
CHECKPOINT_FILE   = OUTPUT_DIR / "checkpoint.json"

# Published leaderboard benchmarks (median MASE, M3 monthly)
LEADERBOARD = {
    "Seasonal Naïve":    1.000,
    "Theta (M3 winner)": 0.854,
    "ForecastPro":       0.828,
    "AutoARIMA":         0.893,
    "ETS (auto)":        0.861,
    "Modern ensemble":   0.700,
}


def _load_checkpoint() -> Dict[str, Any]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_ids": [], "results": []}


def _save_checkpoint(completed_ids: List[str], results: List[Dict]) -> None:
    tmp = OUTPUT_DIR / "_checkpoint_tmp.json"
    with open(tmp, "w") as f:
        json.dump({"completed_ids": completed_ids, "results": results}, f)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    tmp.rename(CHECKPOINT_FILE)


def _build_summary(results: List[Dict]) -> Dict[str, Any]:
    """Build leaderboard comparison summary from results list."""
    valid     = [r for r in results if r.get("mase") is not None]
    mase_vals = [r["mase"] for r in valid]
    smape_vals = [r["smape"] for r in valid if r.get("smape") is not None]

    if not mase_vals:
        return {"error": "No valid results"}

    median_mase = float(np.median(mase_vals))
    mean_mase   = float(np.mean(mase_vals))
    p25_mase    = float(np.percentile(mase_vals, 25))
    p75_mase    = float(np.percentile(mase_vals, 75))
    median_smape = float(np.median(smape_vals)) if smape_vals else None

    # Beats-naïve rate
    beats_naive = sum(1 for m in mase_vals if m < 1.0) / len(mase_vals)

    # Leaderboard position
    leaderboard_row = {"VEDUTA v2.1.0": round(median_mase, 4)}
    leaderboard_row.update({k: v for k, v in LEADERBOARD.items()})
    ranked = sorted(leaderboard_row.items(), key=lambda x: x[1])
    veduta_rank = next(i+1 for i,(k,v) in enumerate(ranked) if k == "VEDUTA v2.1.0")

    return {
        "series_total":      len(results),
        "series_valid":      len(valid),
        "series_crashed":    len(results) - len(valid),
        "median_mase":       round(median_mase, 4),
        "mean_mase":         round(mean_mase, 4),
        "p25_mase":          round(p25_mase, 4),
        "p75_mase":          round(p75_mase, 4),
        "median_smape":      round(median_smape, 4) if median_smape else None,
        "beats_naive_pct":   round(beats_naive * 100, 1),
        "leaderboard":       dict(ranked),
        "veduta_rank":       f"{veduta_rank} of {len(ranked)}",
        "target_met":        median_mase <= 0.70,
        "target_mase":       0.70,
    }


def run_full(
    max_series: Optional[int] = None,
    resume:     bool          = False,
    verbose:    bool          = True,
) -> Dict[str, Any]:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or resume ────────────────────────────────────────────────────────
    checkpoint      = _load_checkpoint() if resume else {"completed_ids": [], "results": []}
    completed_ids   = set(checkpoint["completed_ids"])
    results         = checkpoint["results"]

    all_series = load_m3_monthly(verbose=verbose)
    if max_series:
        all_series = all_series[:max_series]

    pending = [s for s in all_series if s["id"] not in completed_ids]

    if verbose:
        print(f"\nM3 Full Run — VEDUTA Engine v2.1.0")
        print(f"  Total series     : {len(all_series)}")
        print(f"  Already done     : {len(completed_ids)}")
        print(f"  To run           : {len(pending)}")
        print(f"  Horizon          : {HORIZON}")
        print(f"  Confidence level : {CONFIDENCE_LEVEL}")
        print(f"  Checkpoint every : {CHECKPOINT_EVERY} series")
        print()

    t_start   = time.time()
    completed = list(completed_ids)

    for i, series in enumerate(pending, 1):
        sid     = series["id"]
        df      = series["df"]
        actuals = series["actuals"][:HORIZON]
        horizon = min(HORIZON, len(actuals))
        t0      = time.time()

        try:
            result = run_primary_ensemble(
                df               = df,
                horizon          = horizon,
                confidence_level = CONFIDENCE_LEVEL,
            )

            fc_df = result.forecast_df.copy()
            fc_df["date"] = pd.to_datetime(fc_df["date"])
            last_train    = df["date"].max()
            future        = fc_df[fc_df["date"] > last_train].reset_index(drop=True)

            if len(future) < horizon:
                raise RuntimeError(f"Short forecast: {len(future)}")

            forecast = future["forecast"].astype("float64").values[:horizon]
            if not np.isfinite(forecast).all():
                raise RuntimeError("Non-finite forecast")

            y_train = df["value"].astype("float64").values
            mase    = compute_mase(actuals, forecast, y_train)
            smape   = compute_smape(actuals, forecast)
            elapsed = round(time.time() - t0, 2)

            entry = {
                "id":             sid,
                "status":         "PASS" if mase < 2.0 else "WARN",
                "mase":           round(float(mase), 6),
                "smape":          round(float(smape), 6),
                "n_train":        series["n_train"],
                "duration_s":     elapsed,
                "stacker_active": result.metadata.get("stacker_active", False),
                "error":          None,
            }

            if verbose:
                total_done = len(completed_ids) + i
                pct        = total_done / len(all_series) * 100
                flag       = "✓" if mase < 1.0 else ("~" if mase < 2.0 else "✗")
                print(f"  [{total_done:4d}/{len(all_series)}] {sid:6s}  "
                      f"MASE={mase:.4f} {flag}  {elapsed:.1f}s  ({pct:.1f}%)")

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            entry   = {
                "id":       sid,
                "status":   "CRASH",
                "mase":     None,
                "smape":    None,
                "n_train":  series["n_train"],
                "duration_s": elapsed,
                "error":    str(e),
            }
            if verbose:
                print(f"  [{len(completed_ids)+i:4d}/{len(all_series)}] "
                      f"{sid:6s}  CRASH  {str(e)[:60]}")

        results.append(entry)
        completed.append(sid)

        # Checkpoint
        if i % CHECKPOINT_EVERY == 0:
            _save_checkpoint(completed, results)
            summary_so_far = _build_summary(results)
            if verbose:
                done_so_far = [r["mase"] for r in results if r["mase"] is not None]
                if done_so_far:
                    print(f"\n  -- Checkpoint {i}/{len(pending)} --  "
                          f"Running median MASE: {np.median(done_so_far):.4f}\n")

    # ── Final results ─────────────────────────────────────────────────────────
    summary = _build_summary(results)
    elapsed_total = round(time.time() - t_start, 1)

    # SHA-256 for certification
    results_str  = json.dumps(results, sort_keys=True, default=str)
    results_hash = hashlib.sha256(results_str.encode()).hexdigest()

    full_output = {
        "results":         results,
        "results_sha256":  results_hash,
    }
    summary_output = {
        "summary":        summary,
        "results_sha256": results_hash,
        "elapsed_s":      elapsed_total,
        "engine":         "VEDUTA v2.1.0",
    }

    # Atomic write — unlink first for Windows compatibility
    for path, data in [(RESULTS_FILE, full_output), (SUMMARY_FILE, summary_output)]:
        tmp = OUTPUT_DIR / f"_{path.stem}_tmp.json"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        if path.exists():
            path.unlink()
        tmp.rename(path)

    # Clean up checkpoint on success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    if verbose:
        print(f"\n{'='*60}")
        print(f"M3 FULL RUN COMPLETE")
        print(f"{'='*60}")
        print(f"  Series valid     : {summary['series_valid']} / {summary['series_total']}")
        print(f"  Median MASE      : {summary['median_mase']}")
        print(f"  Mean MASE        : {summary['mean_mase']}")
        print(f"  Beats naïve      : {summary['beats_naive_pct']}%")
        print(f"  Target met       : {'YES ✓' if summary['target_met'] else 'NO ✗'} (target <= 0.70)")
        print(f"\n  LEADERBOARD:")
        for name, score in summary["leaderboard"].items():
            marker = "  ← VEDUTA" if "VEDUTA" in name else ""
            print(f"    {name:<25s} {score:.4f}{marker}")
        print(f"\n  VEDUTA rank      : {summary['veduta_rank']}")
        print(f"  SHA-256          : {results_hash[:16]}...")
        print(f"  Elapsed          : {elapsed_total}s")
        print(f"\n  Results : {RESULTS_FILE}")
        print(f"  Summary : {SUMMARY_FILE}")
        print(f"{'='*60}\n")

    return summary_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VEDUTA M3 Full Benchmark")
    parser.add_argument("--resume",  action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max",     type=int,  default=None, help="Max series to run")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    run_full(max_series=args.max, resume=args.resume, verbose=args.verbose)
