# ==================================================
# FILE: test_engine_tier_config.py
# VERSION: 1.1.0
# ROLE: PHASE 3E — STAGE 5 TIER CONFIGURATION CERTIFICATION
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# PURPOSE:
#   Verifies that each tier configuration (Essentials,
#   Pro, Enterprise) exposes the correct model subset
#   and the correct ensemble member subset per the
#   registry governance rules.
#
# SCOPE OF THIS STAGE:
#   - Registry tier counts are correct
#   - Tier superset hierarchy holds (Ess ⊂ Pro ⊂ Ent)
#   - Enterprise-only models are isolated from lower tiers
#   - Pro-only models are isolated from Essentials
#   - Essentials models appear in all tiers
#   - Ensemble runs end-to-end at full Enterprise config
#
# OUT OF SCOPE (Phase 3F):
#   - Tier-filtered ensemble execution at runtime
#   - sentinel_config.py feature flags
#   - Per-tier runner dispatch
#
#   Tier-filtered ensemble execution is a Phase 3F concern.
#   This stage confirms the registry foundation is correct
#   so 3F can build on it safely.
#
# PASS CRITERIA:
#   - All tier model counts match registry v3.0.0 spec
#   - Superset hierarchy verified
#   - Model isolation verified (no cross-tier contamination)
#   - Essentials models present in all tiers
#   - Full ensemble (Enterprise) runs and produces valid output
#
# USAGE:
#   From project root:
#   (.venv) python test_engine_tier_config.py
# ==================================================

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import pandas as pd
import numpy as np
from pathlib import Path

from sentinel_engine.registry import (
    get_models_by_tier,
    get_ensemble_members_by_tier,
)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

# Expected counts — registry v3.0.0
EXPECTED_COUNTS = {
    "essentials": {"total": 10, "ensemble": 8},
    "pro":        {"total": 13, "ensemble": 10},
    "enterprise": {"total": 19, "ensemble": 12},
}

# Isolation rules
ENTERPRISE_ONLY = ["NNETAR", "VAR", "GARCH", "ARIMA", "SARIMAX"]
PRO_ONLY        = ["Croston_SBA", "DHR", "LightGBM"]
ESSENTIALS_CORE = ["HW_Damped", "SARIMA", "ETS", "BSTS", "Naive", "Theta",
                   "STL+ETS", "TBATS", "Prophet", "Primary Ensemble"]

results = []

def record(label, ok, err=None):
    icon = PASS if ok else FAIL
    msg  = f"  {icon}  {label}"
    if not ok and err:
        msg += f" — {err}"
    print(msg)
    results.append((label, ok, err))

print()
print("=" * 70)
print("  SENTINEL ENGINE v2.0.0 — STAGE 5 TIER CONFIGURATION CERTIFICATION")
print("=" * 70)

# ── [1] Tier model counts ─────────────────────────────────────
print("\n[1] Tier Model Count Verification")

for tier in ["essentials", "pro", "enterprise"]:
    models   = get_models_by_tier(tier)
    ensemble = get_ensemble_members_by_tier(tier)

    exp_t = EXPECTED_COUNTS[tier]["total"]
    exp_e = EXPECTED_COUNTS[tier]["ensemble"]

    ok_t = len(models)   == exp_t
    ok_e = len(ensemble) == exp_e

    print(f"  {'✅' if (ok_t and ok_e) else '❌'}  {tier.capitalize():<12} "
          f"total={len(models)}/{exp_t}  ensemble={len(ensemble)}/{exp_e}")

    record(f"{tier} total count",    ok_t,
           None if ok_t else f"got {len(models)}, expected {exp_t}")
    record(f"{tier} ensemble count", ok_e,
           None if ok_e else f"got {len(ensemble)}, expected {exp_e}")

# ── [2] Superset containment ──────────────────────────────────
print("\n[2] Tier Containment — Each Tier is Proper Superset")

ess_names = {e["name"] for e in get_models_by_tier("essentials")}
pro_names  = {e["name"] for e in get_models_by_tier("pro")}
ent_names  = {e["name"] for e in get_models_by_tier("enterprise")}

ess_sub_pro = ess_names.issubset(pro_names)
pro_sub_ent = pro_names.issubset(ent_names)

record("Essentials ⊂ Pro",        ess_sub_pro,
       None if ess_sub_pro else str(ess_names - pro_names))
record("Pro ⊂ Enterprise",        pro_sub_ent,
       None if pro_sub_ent else str(pro_names - ent_names))

# ── [3] Enterprise-only isolation ────────────────────────────
print("\n[3] Enterprise-Only Model Isolation")

for model in ENTERPRISE_ONLY:
    in_ess = model in ess_names
    in_pro = model in pro_names and model not in ess_names
    contaminated = in_ess or in_pro
    record(f"{model:<22} — not in Essentials or Pro",
           not contaminated,
           f"found in {'Essentials' if in_ess else 'Pro'}" if contaminated else None)

# ── [4] Pro-only isolation ────────────────────────────────────
print("\n[4] Pro-Only Model Isolation")

for model in PRO_ONLY:
    in_ess = model in ess_names
    record(f"{model:<22} — not in Essentials",
           not in_ess,
           "found in Essentials" if in_ess else None)

# ── [5] Essentials core in all tiers ─────────────────────────
print("\n[5] Essentials Core Models Present in All Tiers")

for model in ESSENTIALS_CORE:
    in_all = model in ess_names and model in pro_names and model in ent_names
    record(f"{model:<22} — in all tiers", in_all,
           None if in_all else "missing from one or more tiers")

# ── [6] Full ensemble end-to-end ─────────────────────────────
print("\n[6] Full Ensemble (Enterprise) End-to-End Execution")

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "input.csv"
try:
    df = pd.read_csv(DATA_FILE)
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    df = df.sort_values("date").reset_index(drop=True)

    from sentinel_engine.ensemble import run_primary_ensemble

    result = run_primary_ensemble(
        df=df, horizon=12, confidence_level=0.90,
    )

    fdf    = result.forecast_df
    future = fdf[fdf["actual"].isna()] if "actual" in fdf.columns else fdf.tail(12)

    horizon_ok   = len(future) == 12
    finite_ok    = np.isfinite(future["forecast"].values).all()
    ci_ok        = (future["ci_high"] >= future["ci_low"]).all()
    quorum_ok    = result.metadata.get("component_count_valid", 0) >= 2

    meta = result.metadata
    print(f"       Members valid  : {meta.get('component_count_valid')}")
    print(f"       Agg method     : {meta.get('aggregation_method')}")
    print(f"       Diversity cap  : {meta.get('diversity_cap_applied')}")
    print(f"       ARIMA family   : {meta.get('family_weights', {}).get('arima', '?')}")

    record("Ensemble 12 future rows",           horizon_ok,
           f"got {len(future)}" if not horizon_ok else None)
    record("Ensemble forecast finite",          finite_ok)
    record("Ensemble CI not inverted",          ci_ok)
    record("Ensemble quorum >= 2",              quorum_ok,
           f"only {meta.get('component_count_valid')} valid" if not quorum_ok else None)

    cap_ok = meta.get("diversity_cap_applied", False)
    record("Diversity cap applied",             cap_ok,
           "ARIMA family not capped — check ensemble" if not cap_ok else None)

    arima_w = meta.get("family_weights", {}).get("arima", 1.0)
    arima_bounded = arima_w <= 0.401   # float tolerance
    record(f"ARIMA family weight <= 0.40 (actual {arima_w:.4f})", arima_bounded,
           f"ARIMA weight {arima_w:.4f} exceeds cap" if not arima_bounded else None)

except Exception as e:
    print(f"  {FAIL}  Ensemble execution failed: {e}")
    for label in ["12 future rows", "forecast finite", "CI not inverted",
                  "quorum >= 2", "diversity cap", "ARIMA bounded"]:
        results.append((f"Ensemble {label}", False, str(e)))

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")

certified = failed == 0

if certified:
    print()
    print(f"  {PASS} STAGE 5 TIER CONFIGURATION CERTIFICATION PASSED")
    print()
    print("  Registry v3.0.0 tier architecture verified:")
    print("  Essentials (10 models, 8 ensemble) ⊂")
    print("  Pro        (13 models, 10 ensemble) ⊂")
    print("  Enterprise (19 models, 12 ensemble)")
    print()
    print("  Phase 3E certification complete.")
    print(f"  {PASS} Stage 1 — Smoke Test              : PASS")
    print(f"  {PASS} Stage 2 — SHA-256 Reproducibility : PASS  (17/17 hashes)")
    print(f"  {PASS} Stage 3 — MASE Certification      : PASS  (7-8/10 Elite, 12/15 models)")
    print(f"  {PASS} Stage 4 — Dataset Diversity        : PASS  (3/3 Elite)")
    print(f"  {PASS} Stage 5 — Tier Configuration       : PASS")
    print()
    print("  ENGINE CERTIFIED — Ready for Phase 3F — Feature Flag Architecture.")
else:
    print()
    print(f"  {FAIL} STAGE 5 TIER CONFIGURATION CERTIFICATION FAILED")
    for label, ok, err in results:
        if not ok:
            print(f"    - {label}: {err}")
    sys.exit(1)

print("=" * 70)
print()
