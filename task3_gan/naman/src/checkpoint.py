"""Checkpoint save / load / resume, including RNG state and an optional backup folder."""
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, models, optimizers, epoch, config, backup_dir=None):
    """Save everything needed to resume. models/optimizers are dicts name -> object
    (e.g. {'G':..., 'F':..., 'D_A':..., 'D_B':...}).

    Written to a temp file then renamed, so a crash cannot leave a half-written checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "config": config,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "rng": {  # so a resumed run continues the same random stream
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    if backup_dir:  # copy to a backup location (e.g. a synced/off-machine folder)
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, Path(backup_dir) / path.name)
    return path


def load_checkpoint(path, models, optimizers=None, map_location="cpu", restore_rng=True):
    """Load weights (and optionally optimizers + RNG) in place. Returns (epoch, config)."""
    # weights_only=False because the file holds RNG tuples / config dicts we saved ourselves.
    state = torch.load(path, map_location=map_location, weights_only=False)
    for k, m in models.items():
        m.load_state_dict(state["models"][k])
    if optimizers:
        for k, o in optimizers.items():
            o.load_state_dict(state["optimizers"][k])
    if restore_rng:
        r = state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"].cpu())
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
    return state["epoch"], state["config"]


def latest_checkpoint(folder, run_name):
    """Path of the highest-epoch checkpoint named '<run_name>_epoch###.pt', or None."""
    files = sorted(Path(folder).glob(f"{run_name}_epoch*.pt"))
    return files[-1] if files else None


def checkpoint_name(run_name, epoch):
    """Zero-padded so sorting by name sorts by epoch."""
    return f"{run_name}_epoch{epoch:03d}.pt"
