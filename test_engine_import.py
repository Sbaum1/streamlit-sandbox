# ==================================================
# FILE: test_engine_import.py
# VERSION: 2.0.0
# ROLE: PHASE 3 — IMPORT CHAIN VERIFICATION
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
# PURPOSE:
#   Verifies that the sentinel_engine package loads
#   cleanly and all public symbols are accessible
#   before any certification tests are run.
#
# USAGE:
#   From project root:
#   (.venv) python test_engine_import.py
#
# EXPECTED OUTPUT:
#   All lines should show ✅
#   Any ❌ indicates an import or wiring issue
#   that must be resolved before Phase 3 continues
# ==================================================

import sys
import traceback

PASS = "✅"
FAIL = "❌"
results = []


def check(label: str, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        results.append((label, True))
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"       {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append((label, False))


print()
print("=" * 60)
print("  SENTINEL ENGINE v2.0.0 — IMPORT CHAIN VERIFICATION")
print("=" * 60)

# ── Engine package ───────────────────────────────────────────
print("\n[1] Engine Package")

check("sentinel_engine imports",
      lambda: __import__("sentinel_engine"))

check("ENGINE_VERSION accessible",
      lambda: __import__("sentinel_engine", fromlist=["ENGINE_VERSION"]))

check("ForecastResult accessible",
      lambda: __import__("sentinel_engine", fromlist=["ForecastResult"]))

check("run_all_models accessible",
      lambda: __import__("sentinel_engine", fromlist=["run_all_models"]))

check("apply_stress accessible",
      lambda: __import__("sentinel_engine", fromlist=["apply_stress"]))

check("get_model_registry accessible",
      lambda: __import__("sentinel_engine", fromlist=["get_model_registry"]))

check("certify accessible",
      lambda: __import__("sentinel_engine", fromlist=["certify"]))

check("verify_certificates accessible",
      lambda: __import__("sentinel_engine", fromlist=["verify_certificates"]))

check("save_report accessible",
      lambda: __import__("sentinel_engine", fromlist=["save_report"]))

# ── Submodules ───────────────────────────────────────────────
print("\n[2] Submodules")

check("sentinel_engine.contracts",
      lambda: __import__("sentinel_engine.contracts"))

check("sentinel_engine.registry",
      lambda: __import__("sentinel_engine.registry"))

check("sentinel_engine.ensemble",
      lambda: __import__("sentinel_engine.ensemble"))

check("sentinel_engine.runner",
      lambda: __import__("sentinel_engine.runner"))

check("sentinel_engine.certifier",
      lambda: __import__("sentinel_engine.certifier"))

# ── Model files ──────────────────────────────────────────────
print("\n[3] Model Files")

models = [
    "arima", "bsts", "ets", "naive", "prophet",
    "sarima", "sarimax", "stl_ets", "tbats", "theta", "x13"
]

for model in models:
    check(f"sentinel_engine.models.{model}",
          lambda m=model: __import__(f"sentinel_engine.models.{m}"))

# ── Registry integrity ───────────────────────────────────────
print("\n[4] Registry Integrity")

def check_registry():
    from sentinel_engine import get_model_registry
    registry = get_model_registry()
    assert len(registry) > 0, "Registry is empty"
    for entry in registry:
        assert "name"   in entry, f"Missing 'name' in entry: {entry}"
        assert "runner" in entry, f"Missing 'runner' in entry: {entry}"
        assert callable(entry["runner"]), f"Runner not callable: {entry['name']}"

check("Registry loads and all entries valid", check_registry)

def check_ensemble_members():
    from sentinel_engine import get_ensemble_members
    members = get_ensemble_members()
    assert len(members) >= 2, f"Too few ensemble members: {len(members)}"
    names = [m["name"] for m in members]
    print(f"       Ensemble members ({len(members)}): {', '.join(names)}")

check("Ensemble members accessible", check_ensemble_members)

def check_version():
    from sentinel_engine import ENGINE_VERSION
    assert ENGINE_VERSION == "2.0.0", f"Unexpected version: {ENGINE_VERSION}"
    print(f"       Engine version: {ENGINE_VERSION}")

check("Engine version is 2.0.0", check_version)

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)

print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")

if failed == 0:
    print()
    print("  ✅ ENGINE IMPORT CHAIN VERIFIED")
    print("  Ready to proceed with Phase 3 certification tests.")
else:
    print()
    print("  ❌ IMPORT ERRORS DETECTED")
    print("  Resolve all failures before proceeding.")
    print()
    print("  Failed checks:")
    for label, ok in results:
        if not ok:
            print(f"    - {label}")
    sys.exit(1)

print("=" * 60)
print()