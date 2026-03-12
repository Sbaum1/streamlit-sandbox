# ============================================================
# FILE: x13_artifacts.py
# ROLE: X-13 CENSUS ARTIFACT STORAGE (EVIDENCE VAULT)
# ============================================================

import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict


ARTIFACT_ROOT = Path("artifacts/x13")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store_census_artifacts(
    raw_outputs: Dict[str, bytes],
    dataset_hash: str,
    spec_hash: str,
    census_version: str,
) -> str:
    """
    Stores raw Census Bureau X-13 outputs verbatim.
    Returns immutable artifact_id.
    """

    artifact_id = f"x13-{uuid.uuid4().hex}"
    artifact_dir = ARTIFACT_ROOT / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "artifact_id": artifact_id,
        "dataset_hash": dataset_hash,
        "spec_hash": spec_hash,
        "census_version": census_version,
        "created_at": datetime.utcnow().isoformat(),
        "files": {},
    }

    for name, content in raw_outputs.items():
        file_path = artifact_dir / name
        file_path.write_bytes(content)
        metadata["files"][name] = {
            "sha256": _hash_bytes(content),
            "size": len(content),
        }

    (artifact_dir / "metadata.json").write_text(
        str(metadata),
        encoding="utf-8",
    )

    return artifact_id
