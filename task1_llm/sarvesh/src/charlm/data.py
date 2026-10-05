"""TinyStories loading, character vocab, story-shuffle split, sliding windows."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def ensure_raw_file(data_dir: Path, raw_file: str, url: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / raw_file
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading TinyStories to {path} ...")
    urllib.request.urlretrieve(url, path)
    return path


def split_stories(raw: str, delimiter: str) -> list[str]:
    parts = raw.split(delimiter)
    stories = [p.strip() for p in parts if p.strip()]
    return stories


def story_shuffle_split(
    stories: list[str],
    joiner: str,
    n_train_chars: int,
    n_val_chars: int,
    seed: int,
) -> tuple[str, str, dict[str, Any]]:
    """
    Assign whole stories to train or validation via a seeded shuffle, then
    concatenate until each side reaches its character budget.

    This is intentionally different from a single contiguous chunk cut.
    """
    if not stories:
        raise ValueError("No stories found in the raw file")

    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(stories))]

    def fill(budget: int, candidate_ids: list[int]) -> tuple[str, list[int]]:
        parts: list[str] = []
        used: list[int] = []
        length = 0
        for idx in candidate_ids:
            story = stories[idx]
            add = len(story) if not parts else len(joiner) + len(story)
            # Take at least one story; afterwards stop once we would overshoot badly
            # after already meeting the budget, or keep going until budget is met.
            if parts and length >= budget:
                break
            if parts and length + add > budget and length > 0:
                # One final story so we can trim exactly to budget
                parts.append(story)
                used.append(idx)
                length += add
                break
            parts.append(story)
            used.append(idx)
            length += add
        text = joiner.join(parts)[:budget]
        return text, used

    train_text, train_story_ids = fill(n_train_chars, order)
    train_set = set(train_story_ids)
    remaining = [i for i in order if i not in train_set]
    val_text, val_story_ids = fill(n_val_chars, remaining)

    if len(train_text) < n_train_chars or len(val_text) < n_val_chars:
        raise ValueError(
            f"Raw file too small for budgets train={n_train_chars}, val={n_val_chars} "
            f"(got train={len(train_text)}, val={len(val_text)})."
        )

    info = {
        "split_strategy": "story_shuffle",
        "seed": seed,
        "n_stories_in_raw_file": len(stories),
        "n_train_stories": len(train_story_ids),
        "n_val_stories": len(val_story_ids),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "first_train_story_index": train_story_ids[0] if train_story_ids else None,
        "first_val_story_index": val_story_ids[0] if val_story_ids else None,
        "train_val_story_overlap": len(train_set.intersection(val_story_ids)),
    }
    if info["train_val_story_overlap"] != 0:
        raise RuntimeError("train/val story pools overlapped — bug in split")
    return train_text, val_text, info


def build_char_vocab(texts: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted({c for text in texts for c in text})
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def encode_text(text: str, char_to_idx: dict[str, int]) -> np.ndarray:
    missing = sorted({c for c in text if c not in char_to_idx})
    if missing:
        raise ValueError(f"Characters missing from vocab: {missing[:20]}")
    return np.fromiter((char_to_idx[c] for c in text), dtype=np.int64, count=len(text))


def decode_ids(ids: list[int] | np.ndarray, idx_to_char: dict[int, str]) -> str:
    return "".join(idx_to_char[int(i)] for i in ids)


class CharWindowDataset(Dataset):
    """Fixed-length (x, y) pairs for next-character prediction with optional stride."""

    def __init__(self, ids: np.ndarray, block_size: int, stride: int):
        if len(ids) < block_size + 1:
            raise ValueError(
                f"Need at least block_size+1={block_size + 1} characters, got {len(ids)}"
            )
        self.ids = torch.from_numpy(ids.astype(np.int64))
        self.block_size = block_size
        self.stride = max(1, int(stride))
        last_start = len(ids) - (block_size + 1)
        self.starts = list(range(0, last_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        s = self.starts[index]
        chunk = self.ids[s : s + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def chars_for_sequences(n_sequences: int, block_size: int, stride: int) -> int:
    """
    Approximate characters needed so that sliding windows yield ~n_sequences.
    With stride S and block B: n ≈ 1 + (L - B - 1) / S  =>  L ≈ n*S + B
    """
    return int(n_sequences) * int(stride) + int(block_size)


def prepare_datasets(cfg: dict, paths: dict[str, Path], seed: int) -> dict[str, Any]:
    dcfg = cfg["data"]
    raw_path = ensure_raw_file(paths["data_dir"], dcfg["raw_file"], dcfg["raw_url"])
    raw_text = raw_path.read_text(encoding="utf-8")
    stories = split_stories(raw_text, dcfg["story_delimiter"])

    block = int(dcfg["block_size"])
    train_stride = int(dcfg.get("train_stride", block))
    val_stride = int(dcfg.get("val_stride", block))
    n_train_chars = chars_for_sequences(dcfg["train_sequences"], block, train_stride)
    # Non-overlapping val windows need +1 target char beyond last window.
    n_val_chars = int(dcfg["val_sequences"]) * block + 1

    if dcfg.get("split_strategy", "story_shuffle") != "story_shuffle":
        raise ValueError("Only split_strategy='story_shuffle' is implemented in this project")

    train_text, val_text, split_info = story_shuffle_split(
        stories,
        dcfg["story_joiner"],
        n_train_chars,
        n_val_chars,
        seed,
    )
    split_info.update(
        {
            "block_size": block,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "train_sequences_target": dcfg["train_sequences"],
            "val_sequences_target": dcfg["val_sequences"],
            "n_train_chars_budget": n_train_chars,
            "n_val_chars_budget": n_val_chars,
        }
    )

    chars, char_to_idx, idx_to_char = build_char_vocab([train_text, val_text])
    train_ids = encode_text(train_text, char_to_idx)
    val_ids = encode_text(val_text, char_to_idx)

    processed = paths["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    np.save(processed / "train_ids.npy", train_ids)
    np.save(processed / "val_ids.npy", val_ids)
    with open(processed / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"chars": chars, "char_to_idx": char_to_idx}, f, ensure_ascii=False, indent=1)
    with open(processed / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=1)

    train_ds = CharWindowDataset(train_ids, block, train_stride)
    val_ds = CharWindowDataset(val_ids, block, val_stride)

    return {
        "train_text": train_text,
        "val_text": val_text,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "chars": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(chars),
        "split_info": split_info,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "block_size": block,
    }


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    train_batch: int,
    eval_batch: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        drop_last=True,
        generator=g,
        worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
