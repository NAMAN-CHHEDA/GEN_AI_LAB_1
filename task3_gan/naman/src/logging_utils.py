"""Append-only raw logger plus small environment helpers."""
import csv
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


class AppendOnlyLogger:
    """CSV logger. Opens in append mode, so earlier rows are never rewritten.

    Header is written only when the file is new. If the file already exists
    (e.g. on resume), its header must match `fields`, otherwise we refuse
    rather than edit the old log.
    """

    def __init__(self, path, fields):
        self.path, self.fields = Path(path), list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            if header != self.fields:
                raise ValueError(f"{self.path} has a different header; start a new log file.")
        else:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.fields)

    def log(self, row):
        """Append one row (dict). Keys must be exactly the declared fields."""
        if set(row) != set(self.fields):
            raise ValueError(f"Log keys mismatch. Missing: {set(self.fields)-set(row)}, extra: {set(row)-set(self.fields)}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:  # reopen+close = flushed to disk
            csv.writer(f).writerow([row[k] for k in self.fields])


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed):
    """Seed python, numpy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def peak_memory_mb():
    """Peak allocated GPU memory in MB (0 on CPU)."""
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def library_versions():
    """Versions of the libraries used, for the reproducibility manifest."""
    import torchvision, PIL, yaml
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "pyyaml": yaml.__version__,
        "cuda": torch.version.cuda, "gpu": gpu_name(),
    }
