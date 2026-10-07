"""Training loop, LR schedule, evaluation, checkpointing."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import CharGPT, count_parameters
from .utils import amp_enabled, set_seed


def lr_warmup_cosine(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warm-up, then cosine decay down to an absolute min_lr."""
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(1, warmup_steps))
    remain = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(remain))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model: CharGPT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with amp_ctx:
            logits, loss = model(x, y)
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_tokens += n
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())

    mean_loss = total_loss / max(1, total_tokens)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "bpc": mean_loss / math.log(2.0),
        "top1_accuracy": correct / max(1, total_tokens),
        "tokens": float(total_tokens),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_model(
    model: CharGPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    paths: dict[str, Path],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    tcfg = cfg["train"]
    run_name = cfg["run_name"]
    use_amp = amp_enabled(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
    )

    epochs = int(tcfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(tcfg["warmup_steps"])
    tokens_per_step = tcfg["batch_size"] * cfg["data"]["block_size"]

    history = {
        "step_loss": [],
        "step_lr": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_ppl": [],
        "epoch_val_bpc": [],
        "epoch_val_acc": [],
    }

    best_val = float("inf")
    best_epoch = -1
    nonfinite_steps = 0
    max_grad_norm_seen = 0.0
    tokens_seen = 0
    tokens_per_sec_samples: list[float] = []

    ckpt_dir = paths["checkpoint_dir"]
    logs_dir = paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = logs_dir / f"{run_name}_steps.csv"
    epoch_log_path = logs_dir / f"{run_name}_epochs.csv"

    start_epoch = 0
    global_step = 0
    resume_path = ckpt_dir / f"{run_name}_resume.pt"
    if tcfg.get("resume") and resume_path.exists():
        blob = torch.load(resume_path, map_location=device)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        if use_amp and blob.get("scaler") is not None:
            scaler.load_state_dict(blob["scaler"])
        start_epoch = int(blob["epoch"]) + 1
        global_step = int(blob["global_step"])
        best_val = float(blob.get("best_val", best_val))
        best_epoch = int(blob.get("best_epoch", best_epoch))
        history = blob.get("history", history)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    with open(step_log_path, "w", newline="", encoding="utf-8") as sf, open(
        epoch_log_path, "w", newline="", encoding="utf-8"
    ) as ef:
        step_writer = csv.DictWriter(
            sf,
            fieldnames=["step", "epoch", "loss", "lr", "grad_norm", "tokens_per_sec", "nonfinite"],
        )
        epoch_writer = csv.DictWriter(
            ef,
            fieldnames=[
                "epoch",
                "train_loss_running",
                "val_loss",
                "val_ppl",
                "val_bpc",
                "val_top1",
                "lr",
                "seconds",
            ],
        )
        step_writer.writeheader()
        epoch_writer.writeheader()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_t0 = time.perf_counter()
        model.to(device)

        for epoch in range(start_epoch, epochs):
            model.train()
            set_seed(seed + epoch * 1009)
            running = 0.0
            n_batches = 0
            epoch_t0 = time.perf_counter()
            pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

            for x, y in pbar:
                step_t0 = time.perf_counter()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                lr = lr_warmup_cosine(
                    global_step,
                    total_steps,
                    warmup_steps,
                    tcfg["lr"],
                    tcfg["min_lr"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    _, loss = model(x, y)

                nonfinite = 0
                if not torch.isfinite(loss):
                    nonfinite = 1
                    nonfinite_steps += 1
                    global_step += 1
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                if not torch.isfinite(grad_norm):
                    nonfinite = 1
                    nonfinite_steps += 1
                else:
                    max_grad_norm_seen = max(max_grad_norm_seen, float(grad_norm))

                scaler.step(optimizer)
                scaler.update()

                dt = max(1e-8, time.perf_counter() - step_t0)
                tps = tokens_per_step / dt
                tokens_per_sec_samples.append(tps)
                tokens_seen += tokens_per_step

                loss_f = float(loss.item())
                running += loss_f
                n_batches += 1
                history["step_loss"].append(loss_f)
                history["step_lr"].append(lr)
                history["step_grad_norm"].append(float(grad_norm) if torch.isfinite(grad_norm) else float("nan"))

                step_writer.writerow(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss_f,
                        "lr": lr,
                        "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else "inf",
                        "tokens_per_sec": tps,
                        "nonfinite": nonfinite,
                    }
                )
                if global_step % int(tcfg["log_every"]) == 0:
                    pbar.set_postfix(loss=f"{loss_f:.3f}", lr=f"{lr:.2e}", tps=f"{tps:.0f}")

                global_step += 1

            train_loss_running = running / max(1, n_batches)
            val_stats = evaluate(model, val_loader, device, use_amp)
            history["epoch_train_loss"].append(train_loss_running)
            history["epoch_val_loss"].append(val_stats["loss"])
            history["epoch_val_ppl"].append(val_stats["perplexity"])
            history["epoch_val_bpc"].append(val_stats["bpc"])
            history["epoch_val_acc"].append(val_stats["top1_accuracy"])

            epoch_sec = time.perf_counter() - epoch_t0
            epoch_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss_running": train_loss_running,
                    "val_loss": val_stats["loss"],
                    "val_ppl": val_stats["perplexity"],
                    "val_bpc": val_stats["bpc"],
                    "val_top1": val_stats["top1_accuracy"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": epoch_sec,
                }
            )
            ef.flush()
            sf.flush()

            print(
                f"epoch {epoch + 1}/{epochs} | train_loss={train_loss_running:.4f} | "
                f"val_loss={val_stats['loss']:.4f} | ppl={val_stats['perplexity']:.3f} | "
                f"bpc={val_stats['bpc']:.3f} | acc={val_stats['top1_accuracy']:.3f} | "
                f"{epoch_sec:.1f}s"
            )

            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "cfg": cfg,
                "vocab_size": model.vocab_size,
            }
            save_checkpoint(resume_path, payload)
            save_checkpoint(ckpt_dir / f"{run_name}_final.pt", payload)

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                best_epoch = epoch + 1
                payload["best_val"] = best_val
                payload["best_epoch"] = best_epoch
                save_checkpoint(ckpt_dir / f"{run_name}_best.pt", payload)

        total_train_time = time.perf_counter() - train_t0

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    summary = {
        "run_name": run_name,
        "param_count": count_parameters(model),
        "epochs_completed": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_val_loss": history["epoch_val_loss"][-1] if history["epoch_val_loss"] else None,
        "train_tokens_per_sec_mean": sum(tokens_per_sec_samples) / max(1, len(tokens_per_sec_samples)),
        "tokens_seen": tokens_seen,
        "total_train_time_sec": total_train_time,
        "peak_memory_mb": peak_mem,
        "nonfinite_steps": nonfinite_steps,
        "max_grad_norm": max_grad_norm_seen,
        "device": str(device),
        "fp16_autocast": use_amp,
        "history": history,
    }

    with open(logs_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        dump = {k: v for k, v in summary.items() if k != "history"}
        json.dump(dump, f, indent=2)

    return summary
