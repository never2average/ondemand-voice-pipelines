from __future__ import annotations

import json
import time
from pathlib import Path


def ensure_state_dir(state_dir: str | Path) -> Path:
    resolved_state_dir = Path(state_dir)
    resolved_state_dir.mkdir(parents=True, exist_ok=True)
    return resolved_state_dir


def pipeline_id_path(state_dir: str | Path) -> Path:
    return ensure_state_dir(state_dir) / "pipeline_id.txt"


def done_path(state_dir: str | Path) -> Path:
    return ensure_state_dir(state_dir) / "done.json"


def write_pipeline_id(state_dir: str | Path, pipeline_id: str) -> None:
    pipeline_id_path(state_dir).write_text(f"{pipeline_id}\n", encoding="utf-8")


def read_pipeline_id(state_dir: str | Path) -> str | None:
    path = pipeline_id_path(state_dir)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def wait_for_pipeline_id(state_dir: str | Path, timeout_seconds: float) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        pipeline_id = read_pipeline_id(state_dir)
        if pipeline_id:
            return pipeline_id
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for pipeline_id.txt to appear.")


def write_done(state_dir: str | Path, payload: dict) -> None:
    done_path(state_dir).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

