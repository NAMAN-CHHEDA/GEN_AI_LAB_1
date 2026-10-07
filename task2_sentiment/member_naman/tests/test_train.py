"""Plain-assert tests for dataset.py, engine.py and train.py (CPU, local config). Run:  python tests/test_train.py"""

import io
import contextlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataset
import train
from dataset import LengthBucketedSampler, get_loader, get_vocab_size, load_split
from engine import EMA, evaluate_loader, forward_batch, lr_factor
from features import bag_ids_for_review, bigram_rows_for_split
from models import BOS_ID, build_model
from utils import load_config

LOCAL = load_config("local")
VOCAB = get_vocab_size(LOCAL)


def quiet(fn, *args):
    """Run fn while hiding its printed output (training prints a lot)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def test_bigram_rows_for_split_matches_per_review():
    split = load_split(LOCAL, "train")
    buckets = LOCAL["models"]["ngram_bag"]["bigram_buckets"]
    rows = bigram_rows_for_split(split.ids, split.offsets, VOCAB, buckets, BOS_ID)
    small_chunks = bigram_rows_for_split(split.ids, split.offsets, VOCAB, buckets, BOS_ID, chunk_tokens=997)
    assert (rows == small_chunks).all()  # chunk borders must not change anything
    for i in range(200):
        start, end = split.offsets[i], split.offsets[i + 1]
        expected = bag_ids_for_review(split.ids[start:end].astype(np.int64), VOCAB, buckets, BOS_ID)[end - start:]
        assert (rows[start:end] == expected).all(), i


def test_length_bucketed_sampler():
    split = load_split(LOCAL, "train")
    lengths, batch_size = split.lengths, 64
    sampler = LengthBucketedSampler(np.arange(split.n), lengths, batch_size, seed=42)
    sampler.set_epoch(1)
    first = list(sampler)
    assert sorted(np.concatenate(first).tolist()) == list(range(split.n))  # every index exactly once
    sampler.set_epoch(2)
    second = list(sampler)
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))  # different order each epoch
    bucket_spread = np.mean([lengths[b].max() - lengths[b].min() for b in first[:-1]])
    order = np.random.default_rng(0).permutation(split.n)
    random_batches = [order[i:i + batch_size] for i in range(0, split.n - batch_size, batch_size)]
    random_spread = np.mean([lengths[b].max() - lengths[b].min() for b in random_batches])
    assert bucket_spread < 0.5 * random_spread, (bucket_spread, random_spread)


def test_evaluation_order_is_restored():
    for name in ("transformer", "ngram_bag"):
        torch.manual_seed(0)
        model = build_model(name, LOCAL, VOCAB).eval()
        loader = get_loader(LOCAL, name, "val", train=False)
        result = evaluate_loader(model, loader, loader.n, torch.device("cpu"), None)
        for i in range(20):
            batch = loader.builder.make(np.array([i]))
            with torch.no_grad():
                alone = forward_batch(model, batch, torch.device("cpu"))
            assert abs(torch.sigmoid(alone)[0].item() - result["probs"][i]) < 1e-4, (name, i)
        assert (result["labels"] == loader.labels).all()  # labels also in the original order


def test_lr_schedule():
    total, warmup = 1000, 0.05
    assert lr_factor(0, total, warmup) < 0.05
    assert abs(lr_factor(49, total, warmup) - 1.0) < 1e-9  # last warm-up step reaches the peak
    assert lr_factor(total - 1, total, warmup) < 1e-3
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warmup))
    lrs = []
    for _ in range(total):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert lrs[0] < 0.05 and max(lrs) == lrs[49] and lrs[-1] < 1e-3


def test_ema():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA(model, 0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)  # t = 0: decay = min(0.9, 1/10) = 0.1  ->  0.1 * 1 + 0.9 * 3
    assert abs(ema.shadow["weight"].item() - (0.1 * 1.0 + 0.9 * 3.0)) < 1e-6
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)  # t = 1: decay = min(0.9, 2/11) = 2/11
    expected = (2 / 11) * 2.8 + (9 / 11) * 5.0
    assert abs(ema.shadow["weight"].item() - expected) < 1e-6
    before = model.weight.detach().clone()
    with ema.swapped_in(model):
        assert abs(model.weight.item() - expected) < 1e-6
    assert torch.equal(model.weight.detach(), before)


def test_end_to_end_and_resume():
    cfg = load_config("local", "unittest")
    shutil.rmtree(cfg["paths"]["checkpoint_dir"], ignore_errors=True)
    shutil.rmtree(cfg["paths"]["output_dir"], ignore_errors=True)
    try:
        common = ["--config", "local", "--model", "ngram_bag", "--max-train", "512", "--tag", "unittest",
                  "--device", "cpu"]
        assert quiet(train.main, common + ["--epochs", "2"]) == 0
        out = cfg["paths"]["output_dir"] / "ngram_bag"
        history = pd.read_csv(out / "history.csv")
        assert len(history) == 2 and (cfg["paths"]["checkpoint_dir"] / "ngram_bag" / "best.pt").exists()
        assert (out / "summary.json").exists() and (out / "loss_curves.png").exists()
        # finished runs are skipped
        assert quiet(train.main, common + ["--epochs", "2"]) == 0
        assert len(pd.read_csv(out / "history.csv")) == 2
        # resume with one more epoch: continues from epoch 2
        assert quiet(train.main, common + ["--epochs", "3", "--resume"]) == 0
        resumed = pd.read_csv(out / "history.csv")
        assert resumed["epoch"].tolist() == [1, 2, 3], resumed["epoch"].tolist()
        assert np.allclose(resumed["train_loss"][:2], history["train_loss"])  # old epochs were not redone
    finally:
        shutil.rmtree(cfg["paths"]["checkpoint_dir"], ignore_errors=True)
        shutil.rmtree(cfg["paths"]["output_dir"], ignore_errors=True)


def test_training_never_loads_test_split():
    source = (Path(train.__file__)).read_text(encoding="utf-8")
    assert '"test"' not in source and "'test'" not in source
    cfg = load_config("local", "unittest2")
    original = dataset.load_split
    requested = []

    def guarded(cfg_, split):
        requested.append(split)
        if split == "test":
            raise AssertionError("training tried to load the test split")
        return original(cfg_, split)

    dataset.load_split = guarded
    try:
        quiet(train.main, ["--config", "local", "--model", "transformer", "--max-train", "128", "--epochs", "1",
                           "--tag", "unittest2", "--device", "cpu"])
    finally:
        dataset.load_split = original
        shutil.rmtree(cfg["paths"]["checkpoint_dir"], ignore_errors=True)
        shutil.rmtree(cfg["paths"]["output_dir"], ignore_errors=True)
    assert sorted(set(requested)) == ["train", "val"], requested


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {name} -> {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
