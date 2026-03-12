#!/usr/bin/env python3
# ==================================================
# FILE: tests/run_tests.py
# ROLE: TEST RUNNER — writes full results to certification JSON after every run
# USAGE:
#   python tests/run_tests.py           — all 6 stages
#   python tests/run_tests.py --stage 2 — single stage
# ==================================================

import sys
import json
import subprocess
import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CERT_FILE = ROOT / "data" / "veduta_certification_v1.0.json"
STAGE2_RESULTS = ROOT / "data" / "stage2_results.json"

STAGES = {
    1: ("Smoke Test",        "tests/test_engine_smoke.py"),
    2: ("MASE & sMAPE",      "tests/test_engine_mase.py"),
    3: ("Reproducibility",   "tests/test_engine_reproducibility.py"),
    4: ("Backtest",          "tests/test_backtest_verification.py"),
    5: ("Dataset Diversity", "tests/test_engine_diversity.py"),
    6: ("Tier Config",       "tests/test_engine_tier_config.py"),
}

PASS_ICON = "✅"
FAIL_ICON = "❌"


def run_stage(num, name, path):
    print(f"\n{'='*60}")
    print(f"  STAGE {num} — {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(ROOT / path)],
        cwd=str(ROOT),
    )
    return result.returncode == 0


def load_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def save_cert(cert):
    CERT_FILE.write_text(
        json.dumps(cert, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def update_cert(stage_results, all_pass):
    cert = load_json(CERT_FILE)

    # Build stage result map for this run
    stage_map = {}
    for num, passed in stage_results.items():
        name, _ = STAGES[num]
        key = f"stage_{num}_{name.lower().replace(' ', '_').replace('&', 'and')}"
        stage_map[key] = {
            "result":      "PASS" if passed else "FAIL",
            "description": name,
        }

    last_run = {
        "timestamp":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_stages_pass": all_pass,
        "stages_run":      sorted(stage_results.keys()),
        "stage_results":   stage_map,
        "note": "Written automatically by tests/run_tests.py",
    }

    # If Stage 2 ran, merge full model scores (MASE + sMAPE) into results block
    if 2 in stage_results and stage_results[2] and STAGE2_RESULTS.exists():
        s2 = load_json(STAGE2_RESULTS)
        cert["results"] = {
            "gate_summary":  s2.get("gate_summary", cert.get("results", {}).get("gate_summary")),
            "median_mase":   s2.get("median_mase"),
            "mase_scale":    s2.get("mase_scale"),
            "naive_rmse":    s2.get("naive_rmse"),
            "best_mase":     min(
                (v["mase"] for v in s2["models"].values()
                 if v["mase"] is not None and v["gate"] != "FLAGGED"),
                default=None
            ),
            "best_model":    min(
                ((v["mase"], k) for k, v in s2["models"].items()
                 if v["mase"] is not None and v["gate"] != "FLAGGED"),
                default=(None, None)
            )[1],
            "models": s2["models"],
        }
        last_run["model_scores_updated"] = True
        # Clean up temp file
        STAGE2_RESULTS.unlink()

    cert["last_run"] = last_run

    # Update certified flag only on full run
    if set(stage_results.keys()) == set(STAGES.keys()):
        cert["certified"] = all_pass
        cert["last_full_run"] = last_run["timestamp"]

    save_cert(cert)
    print(f"\n  📄 Certification record updated → {CERT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    target = None
    if "--stage" in sys.argv:
        idx = sys.argv.index("--stage")
        try:
            target = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Usage: python tests/run_tests.py [--stage N]")
            sys.exit(1)

    to_run = {target: STAGES[target]} if target else STAGES

    print("\n" + "=" * 60)
    print("  VEDUTA — FORESIGHT ENGINE TEST SUITE")
    print("  Metrics: MASE (Hyndman & Koehler, 2006) + sMAPE (Makridakis, 1993)")
    print("=" * 60)

    stage_results = {}
    for num, (name, path) in to_run.items():
        stage_results[num] = run_stage(num, name, path)

    all_pass = all(stage_results.values())

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for num, (name, _) in to_run.items():
        icon = PASS_ICON if stage_results[num] else FAIL_ICON
        print(f"  {icon}  Stage {num} — {name}")
    print()
    if all_pass:
        print(f"  {PASS_ICON} ALL STAGES PASSED")
    else:
        print(f"  {FAIL_ICON} FAILURES DETECTED — review output above")
    print("=" * 60 + "\n")

    update_cert(stage_results, all_pass)

    sys.exit(0 if all_pass else 1)
