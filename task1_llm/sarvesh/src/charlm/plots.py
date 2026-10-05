"""Loss-curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _smooth(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    window = max(1, int(window))
    x = np.asarray(values, dtype=np.float64)
    if window == 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_loss_curves(history: dict, output_path: Path, smooth_window: int = 25) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    step_loss = history.get("step_loss", [])
    if step_loss:
        smooth = _smooth(step_loss, smooth_window)
        axes[0].plot(range(len(step_loss)), step_loss, alpha=0.25, color="C0", label="step loss")
        offset = len(step_loss) - len(smooth)
        axes[0].plot(
            range(offset, offset + len(smooth)),
            smooth,
            color="C0",
            label=f"MA-{smooth_window}",
        )
    axes[0].set_title("Training loss (per step)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()

    epochs = list(range(1, len(history.get("epoch_val_loss", [])) + 1))
    if epochs:
        axes[1].plot(epochs, history["epoch_train_loss"], marker="o", label="train (running)")
        axes[1].plot(epochs, history["epoch_val_loss"], marker="o", label="val")
    axes[1].set_title("Epoch train / val loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
