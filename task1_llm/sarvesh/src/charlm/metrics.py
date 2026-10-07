"""Generation diversity metrics and evaluation report helpers."""

from __future__ import annotations

import csv
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from .data import decode_ids
from .model import CharGPT
from .train import evaluate


_WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z'\s]")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(tokenize_words(text), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repeated_4gram_rate(texts: Iterable[str]) -> float:
    rates = []
    for text in texts:
        grams = ngrams(tokenize_words(text), 4)
        if not grams:
            rates.append(0.0)
            continue
        counts = Counter(grams)
        repeated = sum(c for c in counts.values() if c > 1)
        rates.append(repeated / len(grams))
    return sum(rates) / max(1, len(rates))


@torch.no_grad()
def generation_speed(
    model: CharGPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    model.eval()
    prompt_ids = prompt_ids.to(device)
    t0 = time.perf_counter()
    _ = model.generate(prompt_ids, max_new_tokens=max_new_tokens, greedy=True)
    dt = max(1e-8, time.perf_counter() - t0)
    return max_new_tokens / dt


def write_metrics_report(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_full_metrics(
    model: CharGPT,
    train_loader,
    val_loader,
    cfg: dict,
    device: torch.device,
    use_amp: bool,
    summary: dict,
    generated: dict[str, list[str]],
    gen_tokens_per_sec: dict[str, float],
) -> list[dict]:
    train_stats = evaluate(
        model,
        train_loader,
        device,
        use_amp,
        max_batches=cfg["eval"].get("train_eval_batches"),
    )
    val_stats = evaluate(model, val_loader, device, use_amp)

    gap = val_stats["loss"] - train_stats["loss"]
    rows = [
        {"metric": "train_loss_eval_mode", "setting": "all", "value": train_stats["loss"]},
        {"metric": "val_loss", "setting": "all", "value": val_stats["loss"]},
        {"metric": "val_perplexity", "setting": "all", "value": val_stats["perplexity"]},
        {"metric": "val_bpc", "setting": "all", "value": val_stats["bpc"]},
        {"metric": "generalization_gap", "setting": "all", "value": gap},
        {"metric": "train_top1_accuracy", "setting": "all", "value": train_stats["top1_accuracy"]},
        {"metric": "val_top1_accuracy", "setting": "all", "value": val_stats["top1_accuracy"]},
        {"metric": "param_count", "setting": "all", "value": summary["param_count"]},
        {"metric": "train_tokens_per_sec", "setting": "all", "value": summary["train_tokens_per_sec_mean"]},
        {"metric": "peak_memory_mb", "setting": "all", "value": summary["peak_memory_mb"]},
        {"metric": "total_train_time_sec", "setting": "all", "value": summary["total_train_time_sec"]},
        {"metric": "nonfinite_steps", "setting": "all", "value": summary["nonfinite_steps"]},
        {"metric": "max_grad_norm", "setting": "all", "value": summary["max_grad_norm"]},
        {"metric": "best_epoch", "setting": "all", "value": summary["best_epoch"]},
        {"metric": "epochs_completed", "setting": "all", "value": summary["epochs_completed"]},
        {"metric": "vocab_size", "setting": "all", "value": model.vocab_size},
    ]

    for setting, texts in generated.items():
        rows.append({"metric": "distinct_1", "setting": setting, "value": distinct_n(texts, 1)})
        rows.append({"metric": "distinct_2", "setting": setting, "value": distinct_n(texts, 2)})
        rows.append({"metric": "distinct_3", "setting": setting, "value": distinct_n(texts, 3)})
        rows.append(
            {"metric": "repeated_4gram_rate", "setting": setting, "value": repeated_4gram_rate(texts)}
        )
        rows.append(
            {
                "metric": "generation_tokens_per_sec",
                "setting": setting,
                "value": gen_tokens_per_sec.get(setting, float("nan")),
            }
        )
    return rows


def encode_prompt(text: str, char_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    ids = [char_to_idx[c] for c in text if c in char_to_idx]
    if not ids:
        raise ValueError(f"Prompt produced no known characters: {text!r}")
    return torch.tensor([ids], dtype=torch.long, device=device)
