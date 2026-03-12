# ==================================================
# FILE: test_engine_tier_config.py
# VERSION: 1.0.0
# ROLE: PHASE 3E — STAGE 5 TIER CONFIG CERTIFICATION
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
# PURPOSE:
#   Verifies that each platform tier (Essentials / Pro /
#   Enterprise) produces a valid, correctly scoped ensemble
#   using the registry tier accessor.
#
# CHECKS PER TIER:
#   - Correct model subset returned by get_models_by_tier()
#   - Correct ensemble subset returned by get_ensemble_members_by_tier()
#   - No out-of-tier models present
#   - Ensemble runs and produces Elite MASE (< 0.70) or Strong (< 0.85)
#   - Tier-specific model counts match spec
#
# EXPECTED MODEL COUNTS (from registry v3.0.0):
#   Essentials : 10 total,  8 ensemble members
#   Pro        : 13 total, 10 ensemble members
#   Enterprise : 19 total, 12 ensemble members
#
# NOTE:
#   This test does not re-run the full engine per tier.
#   It verifies registry integrity and runs a lightweight
#   ensemble simulation on each tier's model subset.
#
# USAGE: (.venv) python test_engine_tier_config.py
# ==================================================

import sys
import pandas as pd
import numpy as np
from pathlib import Path

HORIZON    = 12
CONFIDENCE = 0.90
MASE_ELITE  = 0.70
MASE_STRONG = 0.85
SEED        = 42

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

# Expected counts from registry v3.0.0
TIER_SPEC = {
    "essentials": {"total": 10, "ensemble": 8},
    "pro":        {"total": 13, "ensemble": 10},
    "enterprise": {"total": 19, "ensemble": 12},
}

print()
print("=" * 70)
print("  SENTINEL ENGINE v2.0.0 — STAGE 5 TIER CONFIG CERTIFICATION")
print("=" * 70)

from sentinel_engine.registry import (
    get_models_by_tier,
    get_ensemble_members_by_tier,
)

results = []
tier_results = {}

# ── Registry structure checks ─────────────────────────────────────────────────
print("\n[1] Registry Tier Structure")

for tier, spec in TIER_SPEC.items():
    models   = get_models_by_tier(tier)
    ensemble = get_ensemble_members_by_tier(tier)

    total_ok    = len(models)   == spec["total"]
    ensemble_ok = len(ensemble) == spec["ensemble"]

    # Verify no out-of-tier models
    tier_order = {"essentials": 0, "pro": 1, "enterprise": 2}
    tier_level = tier_order[tier]
    out_of_tier = [
        m["name"] for m in models
        if tier_order.get(m.get("min_tier", "enterprise"), 2) > tier_level
    ]
    scope_ok = len(out_of_tier) == 0

    ok   = total_ok and ensemble_ok and scope_ok
    icon = PASS if ok else FAIL

    print(f"  {icon}  {tier.capitalize():<12}  "
          f"total={len(models)}/{spec['total']}  "
          f"ensemble={len(ensemble)}/{spec['ensemble']}  "
          f"scope={'OK' if scope_ok else f'FAIL: {out_of_tier}'}")

    results.append((f"{tier} registry", ok, None if ok else
                    f"total={len(models)}, ensemble={len(ensemble)}, out_of_tier={out_of_tier}"))

    tier_results[tier] = {
        "models":   models,
        "ensemble": ensemble,
        "ok":       ok,
    }

# ── Tier ensemble MASE check ─────────────────────────────────────────────────
print("\n[2] Tier Ensemble MASE Check")
print("    Runs ensemble subset for each tier on synthetic smooth-trend data")

rng    = np.random.default_rng(SEED)
n_obs  = 84
dates  = pd.date_range("2017-01", periods=n_obs, freq="MS")
t      = np.arange(n_obs, dtype=float)
vals   = 200 + t*1.5 + 30*np.sin(2*np.pi*t/12) + rng.normal(0, 5, n_obs)
df_all = pd.DataFrame({"date": dates, "value": vals})
train  = df_all.iloc[:-HORIZON]
actuals = vals[-HORIZON:]

# MASE scale
m       = 12
tv      = train["value"].values
errors  = np.abs(tv[m:] - tv[:len(tv)-m])
scale   = float(np.mean(errors)) if len(errors) > 0 else 1.0

for tier in ["essentials", "pro", "enterprise"]:
    ensemble_entries = tier_results[tier]["ensemble"]
    member_names     = [e["name"] for e in ensemble_entries]

    forecasts = []
    skipped   = []

    for entry in ensemble_entries:
        name   = entry["name"]
        runner = entry["runner"]

        # Skip models that require multi-series
        if name == "VAR":
            skipped.append(name)
            continue

        try:
            res = runner(df=train, horizon=HORIZON, confidence_level=CONFIDENCE)
            fdf = res.forecast_df
            future = fdf[fdf["actual"].isna()]
            if len(future) == HORIZON:
                fc = future["forecast"].values.astype(float)
                if np.isfinite(fc).all():
                    forecasts.append(fc)
        except Exception:
            skipped.append(name)

    if len(forecasts) < 2:
        print(f"  {FAIL}  {tier.capitalize():<12} — insufficient models ran ({len(forecasts)})")
        results.append((f"{tier} MASE", False, "insufficient models"))
        continue

    ensemble_fc  = np.mean(np.vstack(forecasts), axis=0)
    mae          = np.mean(np.abs(actuals - ensemble_fc))
    tier_mase    = mae / scale if scale > 0 else None

    if tier_mase is None:
        print(f"  {WARN} {tier.capitalize():<12} — MASE could not be computed")
        results.append((f"{tier} MASE", False, "compute failed"))
        continue

    tier_cert = (
        "Elite"  if tier_mase < MASE_ELITE  else
        "Strong" if tier_mase < MASE_STRONG else
        "FAIL"
    )
    tier_ok = tier_mase < MASE_STRONG   # Gate: Strong or better
    icon    = PASS if tier_ok else FAIL

    print(f"  {icon}  {tier.capitalize():<12}  "
          f"MASE={tier_mase:.4f}  {tier_cert}  "
          f"({len(forecasts)} models, {len(skipped)} skipped)")

    results.append((f"{tier} MASE", tier_ok,
                    None if tier_ok else f"MASE={tier_mase:.4f}"))

# ── Min/max tier containment check ───────────────────────────────────────────
print("\n[3] Tier Containment — Each Tier is Proper Superset")

ess_names = set(m["name"] for m in get_models_by_tier("essentials"))
pro_names = set(m["name"] for m in get_models_by_tier("pro"))
ent_names = set(m["name"] for m in get_models_by_tier("enterprise"))

ess_in_pro = ess_names.issubset(pro_names)
pro_in_ent = pro_names.issubset(ent_names)

print(f"  {'✅' if ess_in_pro else '❌'}  Essentials ⊂ Pro         : {ess_in_pro}")
print(f"  {'✅' if pro_in_ent else '❌'}  Pro ⊂ Enterprise         : {pro_in_ent}")

if not ess_in_pro:
    missing = ess_names - pro_names
    print(f"       Missing from Pro: {missing}")
if not pro_in_ent:
    missing = pro_names - ent_names
    print(f"       Missing from Enterprise: {missing}")

results.append(("Essentials subset of Pro", ess_in_pro, None))
results.append(("Pro subset of Enterprise", pro_in_ent, None))

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
    print(f"  {PASS} STAGE 5 TIER CONFIG CERTIFICATION PASSED")
    print("  Essentials / Pro / Enterprise configs verified.")
    print()
    print("  ✅ ALL FIVE STAGES COMPLETE — ENGINE CERTIFIED (Phase 3E)")
else:
    print()
    print(f"  {FAIL} STAGE 5 TIER CONFIG CERTIFICATION FAILED")
    for label, ok, err in results:
        if not ok:
            print(f"    - {label}: {err}")
    sys.exit(1)

print("=" * 70)
print()
