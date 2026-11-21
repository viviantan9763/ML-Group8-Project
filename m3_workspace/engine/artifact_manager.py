"""Local run records and atomic JSON writes."""

import json
import os
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "m3_workspace" / "runs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reserve(path, value):
    """Create a shared lock exclusively across Streamlit sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)


def history(root=HISTORY):
    return sorted(
        (read_json(path) for path in root.glob("*/*/record.json")),
        key=lambda row: row["started_at"], reverse=True,
    )


def run_folder(record, root=HISTORY):
    return root / record["config"]["dataset"] / record["id"]
