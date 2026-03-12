# ============================================================
# FILE: afe_audit.py
# ROLE: AFE AUDIT & TRACEABILITY UTILITIES
# STATUS: CANONICAL / GOVERNANCE-LOCKED
# ============================================================

import hashlib
import json
from typing import Any


def compute_dataset_hash(dataset: Any) -> str:
    """
    Computes a deterministic hash for a committed dataset.

    This function MUST:
    - Be deterministic
    - Be stable across identical inputs
    - Avoid reliance on object memory addresses
    """

    try:
        payload = json.dumps(dataset, sort_keys=True, default=str)
    except Exception as exc:
        raise ValueError("Dataset hashing failed.") from exc

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
