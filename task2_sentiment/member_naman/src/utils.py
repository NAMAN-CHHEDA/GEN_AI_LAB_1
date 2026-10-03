"""Small helper functions shared by the whole project."""

import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def project_root():
    """The member_naman folder (this file lives in member_naman/src)."""
    return Path(__file__).resolve().parent.parent


def load_config(name, tag=None):
    """Read configs/<name>.yaml and turn the relative `paths` into absolute Paths.

    The absolute paths only exist in memory; the yaml file is never changed.
    With a tag, run_name becomes "<run_name>_<tag>" and checkpoint_dir / output_dir get the same
    suffix (processed_dir is shared, because the data does not depend on the run).
    """
    root = project_root()
    with open(root / "configs" / f"{name}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if tag:
        cfg["run_name"] = f"{cfg['run_name']}_{tag}"
        for key in ("checkpoint_dir", "output_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}_{tag}"
    cfg["paths"] = {key: root / rel for key, rel in cfg["paths"].items()}
    return cfg


def set_seed(seed):
    """Make python, numpy and torch (CPU and GPU) random numbers repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # does nothing if there is no GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cpu_name():
    """Best-effort readable CPU name on Linux, Windows or macOS."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def get_device(prefer="auto"):
    """Return (torch.device, readable name). auto = cuda, then mps, then cpu; "cpu"/"cuda" force one."""
    if prefer == "cpu":
        return torch.device("cpu"), _cpu_name()
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"
    return torch.device("cpu"), _cpu_name()


class Logger:
    """Prints lines and appends them to a log file (existing lines are never touched)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message=""):
        print(message)
        with open(self.path, "a", encoding="utf-8") as f:  # "a" = append only
            f.write(str(message) + "\n")


def _to_plain(obj):
    """Convert Paths (and nested dicts) to strings relative to member_naman, for JSON logging."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return obj.relative_to(project_root()).as_posix()
    return obj


def log_config(logger, cfg):
    """Write the full config, torch version, device name and date to the log."""
    _, device_name = get_device()
    logger.log(f"date: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"torch: {torch.__version__}")
    logger.log(f"device: {device_name}")
    logger.log("config:")
    logger.log(json.dumps(_to_plain(cfg), indent=2))


def count_parameters(model):
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path):
    """Write obj to a JSON file, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
