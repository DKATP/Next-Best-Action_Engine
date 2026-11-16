"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load engine.yaml (defaults to config/engine.yaml at the repo root)."""
    path = Path(path) if path else ROOT / "config" / "engine.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
