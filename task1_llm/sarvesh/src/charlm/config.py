"""YAML config loading with optional `base:` inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "base" in raw:
        base_path = (path.parent / raw["base"]).resolve()
        base = load_config(base_path)
        return deep_merge(base, raw)
    return raw


def resolve_paths(cfg: dict, root: str | Path) -> dict[str, Path]:
    """Turn config path strings into absolute Paths under `root` (usually src/)."""
    root = Path(root).resolve()
    paths = {}
    for key, value in cfg["paths"].items():
        p = Path(value)
        paths[key] = (root / p).resolve() if not p.is_absolute() else p
    for key in ("data_dir", "processed_dir", "checkpoint_dir", "output_dir", "logs_dir"):
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)
    if "metrics_csv" in paths:
        paths["metrics_csv"].parent.mkdir(parents=True, exist_ok=True)
    return paths
