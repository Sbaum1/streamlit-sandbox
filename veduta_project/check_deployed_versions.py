# ==================================================
# FILE: check_deployed_versions.py
# PURPOSE: Confirm which model file versions are
#          actually deployed in sentinel_engine/
# USAGE: cd veduta_project && python check_deployed_versions.py
# ==================================================

import os, sys, re

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

files = {
    "sentinel_engine/backtest.py":        ("2.3.0", "mean_abs_train * 0.01"),
    "sentinel_engine/ensemble.py":        ("4.0.0", "_ridge_stacker_weights"),
    "sentinel_engine/models/ets.py":      ("3.0.0", "AIC grid search"),
    "sentinel_engine/models/sarima.py":   ("3.0.0", "auto_arima"),
    "sentinel_engine/models/theta.py":    ("3.0.0", "STL seasonal decomposition"),
    "sentinel_engine/models/prophet.py":  ("4.0.0", "m1_changepoint_optimised"),
    "sentinel_engine/models/bsts.py":     ("3.0.0", "LEVEL_VAR_SCALES"),
    "sentinel_engine/models/hw_damped.py":("3.0.0", "selected_seasonal_mode"),
    "sentinel_engine/models/croston.py":  ("2.1.0", "Bootstrap CI"),
}

print("\nDeployed version check:")
print(f"  {'File':<45} {'Expected':>8}  {'Found':>8}  {'Key marker'}")
print(f"  {'-'*45} {'-'*8}  {'-'*8}  {'-'*20}")

all_ok = True
for path, (expected_ver, marker) in files.items():
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"  {path:<45} {'???':>8}  {'MISSING':>8}  —")
        all_ok = False
        continue
    with open(full) as f:
        src = f.read()
    m = re.search(r'VERSION:\s*([\d.]+)', src)
    found_ver = m.group(1) if m else "unknown"
    has_marker = marker in src
    status = "OK" if (found_ver == expected_ver and has_marker) else "STALE"
    if status == "STALE": all_ok = False
    print(f"  {path:<45} {expected_ver:>8}  {found_ver:>8}  "
          f"{'✓' if has_marker else '✗ MARKER MISSING'}")

print(f"\n  {'ALL FILES CURRENT — ready to re-run smoke test' if all_ok else 'STALE FILES DETECTED — deploy updated files first'}\n")
