"""Losses, DiffAugment (discriminator inputs only) and EMA update."""
import torch
import torch.nn.functional as F


def lsgan_loss(pred, target_is_real):
    """Least-squares GAN loss: MSE to 1 (real) or 0 (fake)."""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def cycle_l1(recon, real):
    return F.l1_loss(recon, real)


def identity_l1(same, real):
    return F.l1_loss(same, real)


# --- DiffAugment (Zhao et al. 2020): differentiable, so gradients reach the generator ---

def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    s = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - mean) * s + mean


def _rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    s = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - mean) * s + mean


def _rand_translation(x, ratio=0.125):
    """Per-sample integer shift up to ratio*size, zero-padded."""
    n, _, h, w = x.shape
    sx, sy = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, (n, 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (n, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(n, device=x.device),
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


_POLICIES = {
    "color": [_rand_brightness, _rand_saturation, _rand_contrast],
    "translation": [_rand_translation],
}


def diff_augment(x, policy="color,translation"):
    """Apply DiffAugment to a batch. Use on discriminator inputs only (real and fake alike)."""
    if not policy:
        return x
    for name in policy.split(","):
        if name not in _POLICIES:
            raise ValueError(f"Unknown DiffAugment policy '{name}'. Choose from {list(_POLICIES)}.")
        for fn in _POLICIES[name]:
            x = fn(x)
    return x.contiguous()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """ema = decay * ema + (1 - decay) * model, for parameters; buffers are copied."""
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema_model.buffers(), model.buffers()):
        be.copy_(b)
