"""Pre-training sanity checks: shapes, init loss, causal mask, overfit batch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import CharGPT, build_model
from .utils import set_seed


def test_shapes(model: CharGPT, vocab_size: int, block_size: int, device: torch.device) -> bool:
    B, T = 2, min(16, block_size)
    x = torch.randint(0, vocab_size, (B, T), device=device)
    logits, loss = model(x, x)
    ok = logits.shape == (B, T, vocab_size) and loss is not None and loss.ndim == 0
    print(f"shapes: logits={tuple(logits.shape)} loss={tuple(loss.shape) if loss is not None else None} -> {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def test_initial_loss(model: CharGPT, vocab_size: int, block_size: int, device: torch.device, tol: float = 0.35) -> bool:
    B, T = 8, min(32, block_size)
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)
    _, loss = model(x, y)
    expected = math.log(vocab_size)
    diff = abs(float(loss.item()) - expected)
    ok = diff < tol
    print(f"initial loss: {loss.item():.4f} vs ln(V)={expected:.4f} |diff|={diff:.4f} -> {'PASS' if ok else 'FAIL'}")
    return bool(ok)


@torch.no_grad()
def test_causal_mask(model: CharGPT, vocab_size: int, block_size: int, device: torch.device) -> bool:
    model.eval()
    T = min(48, block_size)
    x = torch.randint(0, vocab_size, (1, T), device=device)
    logits_a, _ = model(x)

    t = T // 2
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % vocab_size
    logits_b, _ = model(x2)

    before = (logits_a[0, :t] - logits_b[0, :t]).abs().max().item()
    after = (logits_a[0, t:] - logits_b[0, t:]).abs().max().item()
    ok = before < 1e-5 and after > 1e-5
    print(f"causal mask: max_delta before={before:.2e} after={after:.2e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError("Causal mask appears to leak future tokens")
    return True


def test_overfit_batch(
    cfg: dict,
    vocab_size: int,
    device: torch.device,
    steps: int = 200,
    lr: float = 3e-3,
    pass_loss: float = 0.15,
) -> bool:
    set_seed(cfg["seed"] + 7)
    model = build_model(cfg, vocab_size).to(device)
    # Turn off dropout for memorization check
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    B = 4
    T = min(32, cfg["data"]["block_size"])
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  overfit step {step + 1}: loss={last:.4f}")
    ok = last is not None and last < pass_loss
    print(f"overfit one batch: final_loss={last:.4f} (need < {pass_loss}) -> {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_sanity_suite(cfg: dict, vocab_size: int, device: torch.device) -> dict[str, bool]:
    set_seed(cfg["seed"] + 1)
    model = build_model(cfg, vocab_size).to(device)
    block = cfg["data"]["block_size"]
    results = {
        "shapes": test_shapes(model, vocab_size, block, device),
        "initial_loss": test_initial_loss(model, vocab_size, block, device),
        "causal_mask": test_causal_mask(model, vocab_size, block, device),
        "overfit_one_batch": test_overfit_batch(cfg, vocab_size, device),
    }
    print("summary:", results)
    return results
