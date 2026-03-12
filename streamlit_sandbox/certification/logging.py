import json
from pathlib import Path
from datetime import datetime, timezone


def append_log(record: dict, log_path: str):
    """
    Append-only JSONL logger for certification events.

    Guarantees:
    - No overwrites
    - Deterministic ordering
    - File-system safe
    """

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        **record,
        "_logged_at": datetime.now(timezone.utc).isoformat(),
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
