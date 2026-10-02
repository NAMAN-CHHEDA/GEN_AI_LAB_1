"""Reproducibility manifest: versions, seed, config, checkpoint -> result table."""
import json
from pathlib import Path

from config import resolve
from logging_utils import library_versions, timestamp


def write_run_manifest(cfg, split_manifest_path):
    """Write reproducibility/manifests/run_<name>.json. The checkpoint->result table
    starts empty and is filled in by hand/eval after results exist."""
    out = resolve(cfg["paths"]["manifest_dir"]) / f"run_{cfg['run_name']}.json"
    manifest = {
        "run_name": cfg["run_name"],
        "created_utc": timestamp(),
        "seed": cfg.get("seed"),
        "split_seed": cfg["data"]["split_seed"],
        "split_manifest": str(Path(split_manifest_path).name),
        "libraries": library_versions(),
        "config": cfg,
        "checkpoint_results": [],  # rows like {"checkpoint": "...", "result": "..."}
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out
