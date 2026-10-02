"""Config loading and path resolution (no hard-coded personal paths)."""
from pathlib import Path

import yaml

# task3/ root, found relative to this file (naman/src/config.py -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    """Read a YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require(cfg_section, key):
    """Return cfg_section[key]; raise if it is still null (an undecided value)."""
    val = cfg_section.get(key)
    if val is None:
        raise ValueError(f"Config value '{key}' is null. Fill it in the YAML before running.")
    return val


def resolve(rel_path):
    """Turn a config path (relative to task3/) into an absolute Path."""
    return (ROOT / rel_path).resolve()
