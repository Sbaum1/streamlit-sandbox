# ==================================================
# FILE: veduta_project/tests/smoke_test_hostile.py
# ROLE: HOSTILE SERIES SMOKE TEST RUNNER
# PURPOSE: Run all 6 hostile series through the engine
#          and report pass/fail, MASE, and failure modes
#          before committing to the full M3 simulation.
# USAGE:
#   1. python data/hostile_series.py   (generate test data)
#   2. python tests/smoke_test_hostile.py
# GOVERNANCE: Read-only relative to sentinel_engine/.
#             No engine files are modified.
# ==================================================

from __future__ import annotations

import os
import sys
import time
import json
import traceback
import pandas as pd
import numpy as np
from typing import Any, Dict

# ── Path setup — allows running from repo root ────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sentinel_engine.backtest import run_backtest
from sentinel_engine.ensemble import run_primary_ensemble

# ── Config ────────────────────────────────────────────────────────────
HOSTILE_DIR  = os.path.join(ROOT, "data", "hostile")
RESULTS_DIR  = os.path.join(ROOT, "data", "smoke_results")
HORIZON      = 12    # months ahead
CONF_LEVEL   = 0.80  # confidence interval level
MIN_OBS      = 36    # must match sentinel_engine floor

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Series metadata ───────────────────────────────────────────────────
SERIES_META = {
    "01_flat.csv":             "Flat / near-zero variance",
    "02_structural_break.csv": "Structural break at month 48",
    "03_short.csv":            "Short series (48 obs)",
    "04_high_volatility.csv":  "High volatility / noisy",
    "05_intermittent.csv":     "Intermittent demand (40% zeros)",
    "06_trend_reversal.csv":   "Trend reversal at month 48",
}

# ── Thresholds ────────────────────────────────────────────────────────
MASE_WARN    = 1.50
MASE_FAIL    = 3.00
CRASH_BUDGET = 2


# ── Model runner wrapper ──────────────────────────────────────────────
# run_backtest expects: model_runner(df, horizon, confidence_level) -> Any
# We wrap run_primary_ensemble to match that signature.

def model_runner(df: pd.DataFrame, horizon: int, confidence_level: float) -> Any:
    return run_primary_ensemble(
        df=df,
        horizon=horizon,
        confidence_level=confidence_level,
        active_tier="enterprise",
    )


# ── Helpers ───────────────────────────────────────────────────────────

def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def assess_mase(mase: float) -> tuple:
    if mase <= MASE_WARN:
        return "PASS", "✅ "
    elif mase <= MASE_FAIL:
        return "WARN", "⚠️ "
    else:
        return "FAIL", "❌"


def run_single_series(csv_file: str) -> dict:
    path        = os.path.join(HOSTILE_DIR, csv_file)
    description = SERIES_META.get(csv_file, csv_file)
    result: Dict[str, Any] = {
        "file":        csv_file,
        "description": description,
        "status":      "UNKNOWN",
        "mase":        None,
        "duration_s":  None,
        "error":       None,
        "metrics":     {},
    }

    try:
        df = load_series(path)

        if len(df) < MIN_OBS:
            result["status"] = "SKIP"
            result["error"]  = f"Only {len(df)} obs — below MIN_OBS={MIN_OBS}"
            return result

        t0  = time.time()
        out = run_backtest(
            df=df,
            model_runner=model_runner,
            horizon=HORIZON,
            confidence_level=CONF_LEVEL,
        )
        elapsed = round(time.time() - t0, 2)
        result["duration_s"] = elapsed

        # Extract MASE — try common key names defensively
        mase_val = (
            out.get("mase")
            or out.get("ensemble_mase")
            or out.get("primary_mase")
            or out.get("mean_mase")
        )

        # Capture all numeric metrics
        result["metrics"] = {
            k: round(float(v), 4)
            for k, v in out.items()
            if isinstance(v, (int, float)) and not np.isnan(float(v))
        }

        if mase_val is not None and not np.isnan(float(mase_val)):
            result["mase"]   = round(float(mase_val), 4)
            status, _        = assess_mase(result["mase"])
            result["status"] = status
        else:
            result["status"] = "WARN"
            result["error"]  = f"MASE not found or NaN. Keys: {list(out.keys())}"

    except Exception as e:
        result["status"] = "CRASH"
        result["error"]  = str(e)
        result["tb"]     = traceback.format_exc()

    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════════════════")
    print("  VEDUTA — Hostile Series Smoke Test")
    print("  sentinel_engine/ is READ-ONLY during this test")
    print("══════════════════════════════════════════════════════════\n")

    if not os.path.isdir(HOSTILE_DIR):
        print("ERROR: data/hostile/ not found.")
        print("Run: python data/hostile_series.py first.\n")
        sys.exit(1)

    csv_files = sorted(f for f in os.listdir(HOSTILE_DIR) if f.endswith(".csv"))
    if not csv_files:
        print("ERROR: No CSV files in data/hostile/")
        sys.exit(1)

    results     = []
    crash_count = 0

    for csv_file in csv_files:
        desc = SERIES_META.get(csv_file, csv_file)
        print(f"  Running: {desc} ...", end=" ", flush=True)

        result = run_single_series(csv_file)
        results.append(result)

        if result["status"] == "SKIP":
            print(f"SKIP — {result['error']}")
        elif result["status"] == "CRASH":
            crash_count += 1
            print(f"CRASH — {result['error']}")
        else:
            mase_str = f"MASE={result['mase']:.4f}" if result["mase"] is not None else "MASE=N/A"
            dur_str  = f"{result['duration_s']}s"
            _, icon  = assess_mase(result["mase"]) if result["mase"] else ("—", "??")
            print(f"{icon} {result['status']:4s}  {mase_str}  ({dur_str})")

        if crash_count >= CRASH_BUDGET:
            print(f"\n  Crash budget ({CRASH_BUDGET}) exceeded. Aborting.\n")
            break

    # ── Summary ───────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════")
    print("  RESULTS SUMMARY")
    print("══════════════════════════════════════════════════════════")
    print(f"  {'Description':<38} {'Status':<6} {'MASE':<10} {'Time'}")
    print(f"  {'-'*38} {'-'*6} {'-'*10} {'-'*7}")

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "CRASH": 0}

    for r in results:
        mase_disp = f"{r['mase']:.4f}" if r["mase"] is not None else "N/A"
        dur_disp  = f"{r['duration_s']}s" if r["duration_s"] else "—"
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"  {r['description']:<38} {r['status']:<6} {mase_disp:<10} {dur_disp}")

    print(f"\n  Pass:{counts['PASS']}  Warn:{counts['WARN']}  "
          f"Fail:{counts['FAIL']}  Skip:{counts['SKIP']}  Crash:{counts['CRASH']}")

    # ── M3 readiness verdict ──────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════")
    print("  M3 READINESS VERDICT")
    print("══════════════════════════════════════════════════════════")

    if counts["CRASH"] == 0 and counts["FAIL"] == 0:
        print("  READY FOR M3 PILOT (30-series batch)")
        if counts["WARN"] > 0:
            print(f"  {counts['WARN']} warning(s) — review JSON before full run.")
    elif counts["CRASH"] == 0 and counts["FAIL"] <= 1:
        print("  CONDITIONALLY READY — review failures before M3 pilot.")
    else:
        print("  NOT READY — fix crashes/failures before proceeding.")
        print("  Upload smoke_test_results.json to Claude for diagnosis.")

    # ── Save JSON ─────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "smoke_test_results.json")
    clean    = [{k: v for k, v in r.items() if k != "tb"} for r in results]

    with open(out_path, "w") as f:
        json.dump({"results": clean, "summary": counts}, f, indent=2)

    print(f"\n  Full results → {out_path}")
    print("  Upload that file to Claude for diagnosis if needed.")
    print("══════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
