"""Batching: turns the processed arrays into batches for the bag models and the padded models.

Nothing here is a torch Dataset: a batch is built from a list of review positions (indices),
and every batch carries those positions under "index" so results can be put back in order.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import bigram_rows_for_split, truncate_head_tail
from models import BOS_ID, INPUT_KINDS
from utils import load_json


class SplitArrays:
    """One split read from arrays.npz: flat ids, CSR offsets, labels and review lengths."""

    def __init__(self, ids, offsets, labels):
        self.ids = ids
        self.offsets = offsets
        self.labels = labels
        self.n = len(labels)
        self.lengths = np.diff(offsets)  # cleaned token count of every review


def load_split(cfg, split):
    """Read only the arrays of one split ("train", "val" or "test") from the processed folder."""
    with np.load(cfg["paths"]["processed_dir"] / "arrays.npz") as npz:  # npz loads keys lazily
        return SplitArrays(npz[f"{split}_ids"], npz[f"{split}_offsets"], npz[f"{split}_labels"])


def get_vocab_size(cfg):
    return len(load_json(cfg["paths"]["processed_dir"] / "vocab.json"))


# ----------------------------------------------------------------------------- batch builders

class BagBatches:
    """Builds EmbeddingBag batches: per review its unigram ids then its hashed-bigram rows. No truncation."""

    def __init__(self, split, vocab_size, num_buckets):
        self.split = split
        self.lengths = split.lengths
        self.bigrams = bigram_rows_for_split(split.ids, split.offsets, vocab_size, num_buckets, BOS_ID)

    def make(self, indices):
        offsets, pieces, sizes = self.split.offsets, [], []
        for i in indices:
            start, end = offsets[i], offsets[i + 1]
            pieces.append(self.split.ids[start:end])
            pieces.append(self.bigrams[start:end])
            sizes.append(2 * (end - start))
        flat_ids = torch.from_numpy(np.concatenate(pieces).astype(np.int64))
        bag_offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64))
        return {"flat_ids": flat_ids, "offsets": bag_offsets}


class PaddedBatches:
    """Builds [B, L] id batches padded with 0. Long reviews are cut once (head + tail) when this is created."""

    def __init__(self, split, max_len, head_fraction):
        offsets = split.offsets
        self.reviews = [truncate_head_tail(split.ids[offsets[i]:offsets[i + 1]], max_len, head_fraction)
                        for i in range(split.n)]
        self.lengths = np.array([len(r) for r in self.reviews], dtype=np.int64)

    def make(self, indices):
        lengths = self.lengths[indices]
        ids = np.zeros((len(indices), int(lengths.max())), dtype=np.int64)
        for row, i in enumerate(indices):
            ids[row, :lengths[row]] = self.reviews[i]
        return {"ids": torch.from_numpy(ids), "lengths": torch.from_numpy(lengths)}


# ----------------------------------------------------------------------------- samplers

class ShuffledBatchSampler:
    """Training batches in random order (new shuffle every epoch). Used for the bag models."""

    def __init__(self, indices, batch_size, seed):
        self.indices, self.batch_size, self.seed, self.epoch = np.asarray(indices), batch_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return -(-len(self.indices) // self.batch_size)

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.indices)
        for start in range(0, len(order), self.batch_size):
            yield order[start:start + self.batch_size]


class LengthBucketedSampler:
    """Training batches of similar length (less padding): shuffle, cut into chunks of 50 batches,
    sort each chunk by length, split into batches, then shuffle the batch order."""

    def __init__(self, indices, lengths, batch_size, seed, chunk_batches=50):
        self.indices, self.lengths = np.asarray(indices), lengths
        self.batch_size, self.seed, self.chunk_batches, self.epoch = batch_size, seed, chunk_batches, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return -(-len(self.indices) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(self.indices)
        chunk = self.batch_size * self.chunk_batches
        batches = []
        for start in range(0, len(order), chunk):
            part = order[start:start + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i:i + self.batch_size] for i in range(0, len(part), self.batch_size)]
        for b in rng.permutation(len(batches)):
            yield batches[b]


class EvalBatchSampler:
    """Deterministic batches for val/test: sorted by length (padded models) or in order (bag models)."""

    def __init__(self, n, batch_size, lengths=None):
        self.order = np.arange(n) if lengths is None else np.argsort(lengths, kind="stable")
        self.batch_size = batch_size

    def __len__(self):
        return -(-len(self.order) // self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.order), self.batch_size):
            yield self.order[start:start + self.batch_size]


# ----------------------------------------------------------------------------- loader

class BatchLoader:
    """Iterable of batches (dicts of tensors). Single process (num_workers=0, as needed on Windows)."""

    def __init__(self, builder, labels, sampler, pin_memory):
        self.builder, self.labels, self.sampler, self.pin_memory = builder, labels, sampler, pin_memory
        self.n = len(labels)  # size of the whole split (not just the training subset)

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for indices in self.sampler:
            batch = self.builder.make(indices)
            batch["labels"] = torch.from_numpy(self.labels[indices].astype(np.float32))
            if self.pin_memory:
                batch = {k: v.pin_memory() for k, v in batch.items()}
            batch["index"] = indices  # positions inside the split, as a numpy array
            yield batch


def get_loader(cfg, model_name, split, train, max_train=None, seed=None):
    """Loader for one split. max_train (train only) keeps a seeded random subset, for speed tests."""
    seed = cfg["seed"] if seed is None else seed
    data = load_split(cfg, split)
    kind = INPUT_KINDS[model_name]
    if kind == "bag":
        builder = BagBatches(data, get_vocab_size(cfg), cfg["models"][model_name]["bigram_buckets"])
    else:
        builder = PaddedBatches(data, cfg["data"]["max_len"], cfg["data"]["head_fraction"])

    batch_size = cfg["train"]["batch_size"]
    if train:
        pool = np.arange(data.n)
        if max_train is not None and max_train < data.n:
            pool = np.sort(np.random.default_rng(seed).choice(data.n, size=max_train, replace=False))
        if kind == "bag":
            sampler = ShuffledBatchSampler(pool, batch_size, seed)
        else:
            sampler = LengthBucketedSampler(pool, builder.lengths, batch_size, seed)
    else:
        sampler = EvalBatchSampler(data.n, batch_size * 2, None if kind == "bag" else builder.lengths)
    return BatchLoader(builder, data.labels, sampler, pin_memory=torch.cuda.is_available())
