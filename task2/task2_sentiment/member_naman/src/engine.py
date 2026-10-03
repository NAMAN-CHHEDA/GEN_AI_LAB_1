"""Training engine pieces: forward dispatch, evaluation, learning-rate schedule and EMA."""

import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def forward_batch(model, batch, device):
    """Run the model on one batch (whatever kind it needs) and return logits [B]."""
    if model.input_kind == "bag":
        return model(batch["flat_ids"].to(device, non_blocking=True), batch["offsets"].to(device, non_blocking=True))
    return model(batch["ids"].to(device, non_blocking=True), batch["lengths"].to(device, non_blocking=True))


@torch.no_grad()
def evaluate_loader(model, loader, n, device, amp_dtype):
    """Evaluate on a whole split. Batches may come in any order; results are put back in the original order."""
    model.eval()
    logits_all = np.zeros(n, dtype=np.float32)
    labels_all = np.zeros(n, dtype=np.float32)
    seen = np.zeros(n, dtype=np.int64)
    use_amp = amp_dtype is not None
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = forward_batch(model, batch, device)
        index = batch["index"]
        logits_all[index] = logits.float().cpu().numpy()
        labels_all[index] = batch["labels"].numpy()
        seen[index] += 1
    assert (seen == 1).all(), "every review must be evaluated exactly once"
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits_all), torch.from_numpy(labels_all)).item()
    predictions = (logits_all > 0).astype(np.int64)
    truth = labels_all.astype(np.int64)
    return {
        "loss": loss,
        "accuracy": float((predictions == truth).mean()),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "probs": (1.0 / (1.0 + np.exp(-logits_all.astype(np.float64)))).astype(np.float32),
        "logits": logits_all,
        "labels": labels_all,
    }


def lr_factor(step, total_steps, warmup_ratio):
    """Multiplier on the peak lr: linear warm-up, then cosine down to 0."""
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class EMA:
    """Exponential moving average of the weights; the averaged weights usually evaluate a bit better.

    The decay warms up: decay_t = min(ema_decay, (1 + t) / (10 + t)), t = number of updates so far.
    """

    def __init__(self, model, ema_decay):
        self.ema_decay = ema_decay
        self.updates = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        t = self.updates
        decay = min(self.ema_decay, (1 + t) / (10 + t))
        for name, p in model.named_parameters():
            self.shadow[name].mul_(decay).add_(p.detach(), alpha=1 - decay)
        self.updates += 1

    @contextlib.contextmanager
    def swapped_in(self, model):
        """Inside the with-block the model holds the EMA weights; the original weights come back afterwards."""
        backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        self._copy_into(model, self.shadow)
        try:
            yield
        finally:
            self._copy_into(model, backup)

    @staticmethod
    @torch.no_grad()
    def _copy_into(model, weights):
        for name, p in model.named_parameters():
            p.copy_(weights[name])

    def state_dict(self):
        return {"updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.updates = state["updates"]
        self.shadow = state["shadow"]
