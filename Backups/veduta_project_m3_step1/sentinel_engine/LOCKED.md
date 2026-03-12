# SENTINEL ENGINE — LOCKED

**Status: CERTIFIED & LOCKED**  
**Certification date: 2026-03-07**  
**Certification record: `data/veduta_certification_v1.0.json`**

---

## Governance Rule

**No file in this directory may be modified without a full re-certification.**

This means:
- Any change to any `.py` file in `sentinel_engine/` or `sentinel_engine/models/` — however small — invalidates the current certification
- A full 6-stage test run (`python tests/run_tests.py`) must pass before the change ships
- The certification record must be updated with a new version number, date, and the new SHA-256 hashes
- The `fixes_applied` block in the certification JSON must document what changed and why

## What is locked

All 29 Python files in this directory and `sentinel_engine/models/`.  
SHA-256 hashes for every file are recorded in `data/veduta_certification_v1.0.json` under `locked_engine_files`.

## How to verify integrity

```powershell
python tests/run_tests.py --stage 3
```

Stage 3 (Reproducibility) verifies that engine outputs are deterministic.  
To verify file hashes directly:

```python
import hashlib, json, glob
cert = json.load(open("data/veduta_certification_v1.0.json"))
locked = cert["locked_engine_files"]
for path, expected in locked.items():
    actual = hashlib.sha256(open(path,"rb").read()).hexdigest()
    status = "✅" if actual == expected else "❌ MODIFIED"
    print(f"{status}  {path}")
```

## Permitted changes (without re-certification)

- Files outside `sentinel_engine/` — UI, sidebar, tabs, utils, tests
- `data/` — test datasets and certification records
- `requirements.txt`
- This `LOCKED.md` file

## Version history

| Version | Date       | Change                          | Certified |
|---------|------------|---------------------------------|-----------|
| v1.0    | 2026-03-07 | Initial certification           | ✅ PASS    |
