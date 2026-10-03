"""Plain-assert tests for features.py and models.py. Run:  python tests/test_models.py"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import bag_ids_for_review, bigram_buckets_padded, hash_bigram, truncate_head_tail
from models import (BOS_ID, PAD_ID, UNK_ID, BiGRUClassifier, NGramBag, TransformerClassifier, build_model,
                    parameter_breakdown)
from utils import load_config

MODEL_NAMES = ["ngram_bag", "transformer", "bigru"]
TEST_VOCAB = 1000
LOCAL = load_config("local")
GPU = load_config("gpu")
MEMORISATION_STEPS = {}
PARAM_TABLE = {}


# ---------------------------------------------------------------- helpers

def make_model(name, seed=0):
    torch.manual_seed(seed)
    return build_model(name, LOCAL, TEST_VOCAB).eval()


def random_reviews(lengths, seed=0):
    """List of numpy id arrays (ids >= 3, so no special tokens inside)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(3, TEST_VOCAB, size=n).astype(np.int64) for n in lengths]


def model_inputs(model, reviews, extra_pad=0):
    """Arguments for model(...) built from a list of id arrays."""
    if model.input_kind == "bag":
        buckets = LOCAL["models"]["ngram_bag"]["bigram_buckets"]
        bags = [bag_ids_for_review(r, TEST_VOCAB, buckets, BOS_ID) for r in reviews]
        flat = torch.from_numpy(np.concatenate(bags))
        offsets = torch.tensor(np.concatenate([[0], np.cumsum([len(b) for b in bags])[:-1]]), dtype=torch.long)
        return (flat, offsets)
    lengths = torch.tensor([len(r) for r in reviews])
    ids = torch.zeros(len(reviews), int(lengths.max()) + extra_pad, dtype=torch.long)
    for i, r in enumerate(reviews):
        ids[i, :len(r)] = torch.from_numpy(r)
    return (ids, lengths)


# ---------------------------------------------------------------- features

def test_hash_deterministic_in_range_and_matches_reference():
    prev = torch.randint(0, 50000, (1000,))
    cur = torch.randint(0, 50000, (1000,))
    a, b = hash_bigram(prev, cur, 12345), hash_bigram(prev, cur, 12345)
    assert (a == b).all() and a.min() >= 0 and a.max() < 12345

    def wrap(x):  # python int -> signed 64-bit, like torch int64 overflow
        x &= (1 << 64) - 1
        return x - (1 << 64) if x >= (1 << 63) else x

    for p, c in [(0, 0), (1, 2), (49999, 3), (123, 45678)]:
        h = wrap(wrap(p * 0x2545F4914F6CDD1D) + wrap(c * 0x1B873593CC9E2D51))
        expected = ((h >> 32) & 0x7FFFFFFF) % 1000
        assert hash_bigram(torch.tensor([p]), torch.tensor([c]), 1000).item() == expected


def test_bigram_buckets_padded():
    ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]])
    out = bigram_buckets_padded(ids, BOS_ID, 1000)
    assert out.shape == ids.shape
    assert ((out == 0) == (ids == 0)).all()
    real = out[ids != 0]
    assert real.min() >= 1 and real.max() <= 1000
    # first position pairs BOS with the first word
    assert out[0, 0].item() == 1 + hash_bigram(torch.tensor(BOS_ID), torch.tensor(5), 999).item()


def test_hash_uniformity():
    grid = torch.arange(300)
    prev, cur = torch.meshgrid(grid, grid, indexing="ij")
    counts = torch.bincount(hash_bigram(prev.reshape(-1), cur.reshape(-1), 50000), minlength=50000)
    assert counts.max().item() < 15, counts.max().item()


def test_hash_has_no_linear_structure():
    grid = torch.arange(300)
    prev, cur = torch.meshgrid(grid, grid, indexing="ij")
    prev, cur = prev.reshape(-1), cur.reshape(-1)
    a = hash_bigram(prev + 2, cur, 50000)
    b = hash_bigram(prev, cur + 6, 50000)
    assert (a == b).float().mean().item() < 0.01


def test_truncate_head_tail():
    ids = np.arange(100)
    assert (truncate_head_tail(ids, 100, 0.25) == ids).all()
    assert (truncate_head_tail(ids, 200, 0.25) == ids).all()
    out = truncate_head_tail(ids, 40, 0.25)
    assert len(out) == 40
    assert (out[:10] == np.arange(10)).all() and (out[10:] == np.arange(70, 100)).all()
    assert len(truncate_head_tail(ids, 40, 0.0)) == 40  # no head: whole budget goes to the tail


def test_bag_ids_for_review():
    ids = np.array([10, 20, 30, 40])
    out = bag_ids_for_review(ids, vocab_size=100, num_buckets=500, bos_id=BOS_ID)
    assert len(out) == 8 and out.dtype == np.int64
    assert (out[:4] == ids).all()
    assert out[4:].min() >= 100 and out[4:].max() < 600
    expected_first = 100 + hash_bigram(torch.tensor(BOS_ID), torch.tensor(10), 500).item()
    assert out[4] == expected_first


# ---------------------------------------------------------------- models

def test_ngram_bag_matches_manual_computation():
    torch.manual_seed(0)
    cfg_model = dict(LOCAL["models"]["ngram_bag"], dropout=0.0, bigram_buckets=20)
    model = NGramBag(30, cfg_model).eval()
    flat = torch.tensor([1, 2, 3, 31, 35, 40, 7, 8])
    offsets = torch.tensor([0, 3, 6])  # three bags: [1,2,3] [31,35,40] [7,8]
    logits = model(flat, offsets)
    table = model.bag.weight
    for i, rows in enumerate([[1, 2, 3], [31, 35, 40], [7, 8]]):
        manual = model.out(table[rows].mean(dim=0))
        assert torch.allclose(logits[i], manual.squeeze(), atol=1e-6)


def test_output_shapes():
    reviews = random_reviews([5, 17, 30, 3, 64])
    for name in MODEL_NAMES:
        model = make_model(name)
        out = model(*model_inputs(model, reviews))
        assert out.shape == (5,), (name, out.shape)


def test_padding_invariance():
    reviews = random_reviews([5, 17, 30, 3])
    for name in ("transformer", "bigru"):
        model = make_model(name)
        with torch.no_grad():
            base = model(*model_inputs(model, reviews))
            padded = model(*model_inputs(model, reviews, extra_pad=30 if name == "bigru" else 20))
        assert torch.allclose(base, padded, atol=1e-5), (name, (base - padded).abs().max().item())


def test_batch_invariance():
    reviews = random_reviews([5, 17, 30, 3, 22])
    for name in MODEL_NAMES:
        model = make_model(name)
        with torch.no_grad():
            batch = model(*model_inputs(model, reviews))
            for i, review in enumerate(reviews):
                alone = model(*model_inputs(model, [review]))
                assert torch.allclose(alone[0], batch[i], atol=1e-5), (name, i, (alone[0] - batch[i]).abs().item())


def test_bigru_attention():
    model = make_model("bigru")
    reviews = random_reviews([5, 17, 30, 3])
    ids, lengths = model_inputs(model, reviews)
    with torch.no_grad():
        logits, attn = model(ids, lengths, return_attention=True)
    assert attn.shape == ids.shape
    assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (attn[ids == PAD_ID] == 0).all()


def test_gradient_flow():
    reviews = random_reviews([5, 17, 30, 3])
    for name in MODEL_NAMES:
        torch.manual_seed(0)
        model = build_model(name, LOCAL, TEST_VOCAB).train()
        out = model(*model_inputs(model, reviews))
        out.sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, (name, missing)


def test_memorise_random_labels():
    lrs = {"ngram_bag": 2e-2, "transformer": 2e-3, "bigru": 5e-3}
    rng = np.random.default_rng(1)
    reviews = random_reviews(rng.integers(5, 21, size=32).tolist(), seed=1)
    labels = torch.tensor(rng.integers(0, 2, size=32), dtype=torch.float32)
    for name in MODEL_NAMES:
        model = make_model(name, seed=3)  # eval(): dropout is off, so this measures capacity only
        inputs = model_inputs(model, reviews)
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs[name])
        steps, loss = 0, None
        for steps in range(1, 301):
            loss = nn.functional.binary_cross_entropy_with_logits(model(*inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < 0.1:
                break
        MEMORISATION_STEPS[name] = steps
        print(f"      {name}: loss {loss.item():.4f} after {steps} steps")
        assert loss.item() < 0.1, (name, loss.item())


def test_parameter_counts_gpu_size():
    for name in MODEL_NAMES:
        model = build_model(name, GPU, 50000)
        total, emb, rest = parameter_breakdown(model)
        PARAM_TABLE[name] = (total, emb, rest)
        print(f"      {name}: total {total:,}  embeddings {emb:,}  rest {rest:,}")
    assert 60e6 < PARAM_TABLE["ngram_bag"][0] < 70e6
    assert 6e6 < PARAM_TABLE["transformer"][0] < 8e6
    assert 35e6 < PARAM_TABLE["bigru"][0] < 42e6


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
    if "--arch" in sys.argv:
        for name in MODEL_NAMES:
            print(f"\n===== {name} (GPU size) =====")
            print(build_model(name, GPU, 50000))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
