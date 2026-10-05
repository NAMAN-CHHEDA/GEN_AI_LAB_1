"""Character-level GPT package for Task 1 (Sarvesh)."""

from .config import load_config, deep_merge
from .model import CharGPT, count_parameters

__all__ = ["load_config", "deep_merge", "CharGPT", "count_parameters"]
