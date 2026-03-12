# ==================================================
# FILE: test_engine_reproducibility.py
# VERSION: 3.0.0
# ROLE: PHASE 3E — STAGE 2 SHA-256 REPRODUCIBILITY
# ENGINE: Sentinel Engine v2.0.0
# UPDATED: PHASE 3C — dynamic model count
# ==================================================

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import json
import pandas as pd
import numpy as np
from pathlib import Path

from sentinel_engine import run_all_models, generate_certificates
from sentinel_engine.certifier import hash_dataframe

DATA_FILE  = Path(__file__).resolve().parent.parent / "data" / "input.csv"
CERT_FILE  = Path("cert_artifacts/certified_hashes.json")
HORIZON    = 12
CONFIDENCE = 0.90

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

print()
print("=" * 65)
print("  SENTINEL ENGINE v2.0.0 — STAGE 2 SHA-256 REPRODUCIBILITY  (Phase 3E)")
print("=" * 65)

print("\n[1] Loading Data")
try:
    df = pd.read_csv(DATA_FILE)
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    print(f"  {PASS}  input.csv — {len(df)} observations")
except Exception as e:
    print(f"  {FAIL}  Failed to load data: {e}")
    sys.exit(1)

print("\n[2] Run 1 — Generating Forecasts")
try:
    results_1 = run_all_models(df=df, horizon=HORIZON, confidence_level=CONFIDENCE)
    n1 = results_1["_engine"]["models_succeeded"]
    na = results_1["_engine"]["models_attempted"]
    print(f"  {PASS}  Run 1 complete — {n1}/{na} models succeeded")
except Exception as e:
    print(f"  {FAIL}  Run 1 failed: {e}")
    sys.exit(1)

print("\n[3] Generating Golden Hashes")
CERT_FILE.parent.mkdir(parents=True, exist_ok=True)
try:
    golden = generate_certificates(results_1, str(CERT_FILE))
    print(f"  {PASS}  Golden hashes written to {CERT_FILE}")
    print(f"         {len(golden)} model(s) hashed")
except Exception as e:
    print(f"  {FAIL}  Hash generation failed: {e}")
    sys.exit(1)

print("\n[4] Run 2 — Reproducibility Check")
try:
    results_2 = run_all_models(df=df, horizon=HORIZON, confidence_level=CONFIDENCE)
    n2 = results_2["_engine"]["models_succeeded"]
    print(f"  {PASS}  Run 2 complete — {n2}/{na} models succeeded")
except Exception as e:
    print(f"  {FAIL}  Run 2 failed: {e}")
    sys.exit(1)

print("\n[5] SHA-256 Hash Comparison (Run 1 vs Run 2)")

all_match = True
hash_results = []

production_models = [
    name for name, result in results_1.items()
    if not name.startswith("_")
    and isinstance(result, dict)
    and result.get("status") == "success"
    and not result.get("diagnostic_only", False)
]

for name in sorted(production_models):
    r1 = results_1.get(name, {})
    r2 = results_2.get(name, {})

    fdf1 = r1.get("forecast_df")
    fdf2 = r2.get("forecast_df")

    if fdf1 is None or fdf2 is None:
        print(f"  {WARN} {name:22s} — forecast_df missing in one run")
        hash_results.append((name, False))
        all_match = False
        continue

    future1 = fdf1[fdf1["actual"].isna()]
    future2 = fdf2[fdf2["actual"].isna()]
    if future1.empty: future1 = fdf1
    if future2.empty: future2 = fdf2

    try:
        hash1 = hash_dataframe(future1)
        hash2 = hash_dataframe(future2)
    except Exception as e:
        print(f"  {FAIL}  {name:22s} — hash error: {e}")
        hash_results.append((name, False))
        all_match = False
        continue

    match = hash1 == hash2
    icon  = PASS if match else FAIL
    print(f"  {icon}  {name:22s}  {hash1[:16]}...")

    if not match:
        all_match = False
        print(f"       Run 1: {hash1}")
        print(f"       Run 2: {hash2}")

    hash_results.append((name, match))

print("\n[6] Verify Run 2 Against Golden Hashes")

golden_data   = json.loads(CERT_FILE.read_text())
golden_hashes = {
    name: entry["sha256"]
    for name, entry in golden_data.get("models", {}).items()
}

cert_match = True
for name in sorted(production_models):
    r2   = results_2.get(name, {})
    fdf2 = r2.get("forecast_df")
    if fdf2 is None:
        continue
    future2 = fdf2[fdf2["actual"].isna()]
    if future2.empty: future2 = fdf2
    try:
        hash2    = hash_dataframe(future2)
        expected = golden_hashes.get(name)
        if expected is None:
            print(f"  {WARN} {name:22s} — no golden hash on file")
            continue
        match = hash2 == expected
        icon  = PASS if match else FAIL
        print(f"  {icon}  {name:22s}")
        if not match:
            cert_match = False
    except Exception as e:
        print(f"  {FAIL}  {name:22s} — {e}")
        cert_match = False

print()
print("=" * 65)
passed = sum(1 for _, ok in hash_results if ok)
failed = sum(1 for _, ok in hash_results if not ok)
total  = len(hash_results)

print(f"  Run 1 vs Run 2 matches : {passed}/{total}")
print(f"  Golden file verified   : {'✅ PASS' if cert_match else '❌ FAIL'}")

if all_match and cert_match:
    print()
    print(f"  {PASS} SHA-256 REPRODUCIBILITY CERTIFIED")
    print(f"  Golden hashes written to {CERT_FILE}")
    print("  Ready for Stage 3 — MASE Certification.")
else:
    print()
    print(f"  {FAIL} REPRODUCIBILITY FAILURES DETECTED")
    if not all_match:
        print("  Non-deterministic models:")
        for name, ok in hash_results:
            if not ok:
                print(f"    - {name}")
    sys.exit(1)

print("=" * 65)
print()
