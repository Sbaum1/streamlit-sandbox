# ==================================================
# FILE: m3_official_metrics.py
# VERSION: 1.1.0
# ROLE: OFFICIAL M3 METRICS COMPUTATION
# ENGINE: Sentinel Engine v2.2.0
# UPDATED: Phase 4 -- rebuilt from session record
# ==================================================
#
# PURPOSE:
#   Computes all 5 official M3 competition metrics from
#   m3_full_results.json produced by m3_runner.py.
#
# M3 OFFICIAL METRICS:
#   Metric 1 -- Mean MASE         (Hyndman & Koehler 2006)
#   Metric 2 -- Median MASE       (primary certification metric)
#   Metric 3 -- Mean sMAPE        (official 2000 M3 metric)
#   Metric 4 -- Median sMAPE
#   Metric 5 -- % Better than Naive (MASE < 1.0)
#
# USAGE:
#   python m3_official_metrics.py
#   python m3_official_metrics.py --results path/to/m3_full_results.json
#
# OUTPUT:
#   Prints metrics to console.
#   Writes tests/m3_results/m3_official_metrics.json
# ==================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "tests" / "m3_results"
DEFAULT_INPUT  = RESULTS_DIR / "m3_full_results.json"
OUTPUT_FILE    = RESULTS_DIR / "m3_official_metrics.json"

# --------------------------------------------------
# LEADERBOARD BENCHMARKS (M3 Monthly, median MASE)
# --------------------------------------------------

MASE_LEADERBOARD = {
    "Modern Ensemble (published)": 0.700,
    "ForecastPro (commercial)":    0.828,
    "Theta -- M3 Winner (2000)":   0.854,
    "ETS Auto (R forecast)":       0.861,
    "AutoARIMA":                   0.893,
    "Seasonal Naive (baseline)":   1.000,
}

SMAPE_LEADERBOARD = {
    "Theta -- M3 Winner (2000)":   13.84,
    "ForecastPro (commercial)":    14.03,
    "Dampen (ES)":                 14.16,
    "Comb S-H-D":                  14.27,
    "AutoBox1":                    15.19,
    "B-J Auto":                    15.29,
    "Naive2 (baseline)":           17.08,
}


# --------------------------------------------------
# COMPUTATION
# --------------------------------------------------

def compute_metrics(results_path: Path) -> dict:
    """
    Load m3_full_results.json and compute all 5 official M3 metrics.
    Returns metrics dict. Raises on bad input.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    # m3_full_results.json wraps results in a top-level key
    if isinstance(data, dict) and "results" in data:
        results_sha256 = data.get("results_sha256", "not recorded")
        results        = data["results"]
    else:
        # Flat list fallback
        results_sha256 = "not recorded"
        results        = data

    if not results:
        raise ValueError("No results found in file.")

    # Extract valid series only (no crashes)
    mase_vals  = []
    smape_vals = []

    for r in results:
        if r.get("mase") is not None and np.isfinite(r["mase"]):
            mase_vals.append(float(r["mase"]))
        if r.get("smape") is not None and np.isfinite(r["smape"]):
            smape_vals.append(float(r["smape"]) * 100)  # convert to %

    n_total   = len(results)
    n_valid   = len(mase_vals)
    n_crashed = n_total - n_valid

    if n_valid == 0:
        raise ValueError("No valid MASE values found in results.")

    # -- Metric 1: Mean MASE
    mean_mase   = float(np.mean(mase_vals))

    # -- Metric 2: Median MASE (primary certification metric)
    median_mase = float(np.median(mase_vals))

    # -- Metric 3: Mean sMAPE
    mean_smape   = float(np.mean(smape_vals))  if smape_vals else None

    # -- Metric 4: Median sMAPE
    median_smape = float(np.median(smape_vals)) if smape_vals else None

    # -- Metric 5: % Better than Naive
    pct_beat_naive = float(sum(1 for m in mase_vals if m < 1.0) / n_valid * 100)

    # -- Leaderboard ranks
    mase_board = {"VEDUTA v2.2.0": round(median_mase, 4)}
    mase_board.update(MASE_LEADERBOARD)
    mase_ranked = sorted(mase_board.items(), key=lambda x: x[1])
    mase_rank   = next(i + 1 for i, (k, _) in enumerate(mase_ranked)
                       if k == "VEDUTA v2.2.0")

    smape_rank = None
    if mean_smape is not None:
        smape_board = {"VEDUTA v2.2.0": round(mean_smape, 2)}
        smape_board.update(SMAPE_LEADERBOARD)
        smape_ranked = sorted(smape_board.items(), key=lambda x: x[1])
        smape_rank   = next(i + 1 for i, (k, _) in enumerate(smape_ranked)
                            if k == "VEDUTA v2.2.0")

    # -- Recompute SHA-256 from results for cross-check
    results_str  = json.dumps(results, sort_keys=True, default=str)
    computed_sha = hashlib.sha256(results_str.encode()).hexdigest()

    return {
        "series_total":       n_total,
        "series_valid":       n_valid,
        "series_crashed":     n_crashed,
        # Official metrics
        "metric_1_mean_mase":    round(mean_mase,    4),
        "metric_2_median_mase":  round(median_mase,  4),
        "metric_3_mean_smape":   round(mean_smape,   2) if mean_smape   is not None else None,
        "metric_4_median_smape": round(median_smape, 2) if median_smape is not None else None,
        "metric_5_pct_beat_naive": round(pct_beat_naive, 1),
        # Leaderboard
        "mase_rank":          mase_rank,
        "mase_leaderboard":   dict(mase_ranked),
        "smape_rank":         smape_rank,
        # Certification
        "results_sha256_recorded": results_sha256,
        "results_sha256_computed": computed_sha,
        "sha256_match":            results_sha256 == computed_sha,
    }


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

def print_report(m: dict) -> None:
    W = 56
    print("\n" + "=" * W)
    print("  VEDUTA v2.2.0 -- Official M3 Metrics")
    print("=" * W)
    print(f"  Series evaluated : {m['series_valid']:,} / {m['series_total']:,}")
    if m["series_crashed"]:
        print(f"  !! CRASHES       : {m['series_crashed']}")
    print()
    print("  OFFICIAL METRICS")
    print(f"  Metric 1 -- Mean MASE         : {m['metric_1_mean_mase']:.4f}")
    print(f"  Metric 2 -- Median MASE       : {m['metric_2_median_mase']:.4f}  "
          f"<-- PRIMARY CERT METRIC  Rank #{m['mase_rank']}")
    if m["metric_3_mean_smape"] is not None:
        rank_str = f"  Rank #{m['smape_rank']}" if m["smape_rank"] else ""
        print(f"  Metric 3 -- Mean sMAPE        : {m['metric_3_mean_smape']:.2f}%{rank_str}")
    if m["metric_4_median_smape"] is not None:
        print(f"  Metric 4 -- Median sMAPE      : {m['metric_4_median_smape']:.2f}%")
    print(f"  Metric 5 -- % Beat Naive      : {m['metric_5_pct_beat_naive']:.1f}%")

    print()
    print("  MASE LEADERBOARD (Median MASE, M3 Monthly)")
    for name, score in m["mase_leaderboard"].items():
        marker = "  <-- VEDUTA v2.2.0" if name == "VEDUTA v2.2.0" else ""
        print(f"    {name:<35} {score:.4f}{marker}")

    print()
    print("  SHA-256 CERTIFICATION")
    sha_ok = m["sha256_match"]
    print(f"  Recorded : {m['results_sha256_recorded']}")
    print(f"  Computed : {m['results_sha256_computed']}")
    print(f"  Match    : {'YES -- results file is unmodified' if sha_ok else 'NO -- FILE MAY HAVE BEEN ALTERED'}")
    print("=" * W + "\n")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute official M3 metrics from m3_full_results.json"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to m3_full_results.json (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()

    try:
        metrics = compute_metrics(args.results)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run m3_runner.py first to generate results.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR computing metrics: {e}\n")
        sys.exit(1)

    print_report(metrics)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Written -> {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
