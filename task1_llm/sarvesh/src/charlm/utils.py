"""Seeding and device helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(cfg: dict, device: torch.device) -> bool:
    return bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
