"""Train one model on the train split, select the best epoch on the VALIDATION split (never the test split).

Run:  python src/train.py --config local --model ngram_bag
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_loader, get_vocab_size
from engine import EMA, evaluate_loader, forward_batch, lr_factor
from models import build_model
from utils import Logger, _to_plain, count_parameters, get_device, load_config, log_config, project_root, save_json, set_seed


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train one model")
    p.add_argument("--config", default="local")
    p.add_argument("--model", required=True, choices=["ngram_bag", "transformer", "bigru"])
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--max-train", type=int, default=None, help="use a random subset of the training reviews")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tag", default=None, help="adds _<tag> to run_name, so the outputs go to separate folders")
    p.add_argument("--resume", action="store_true", help="continue from last.pt (fresh start if there is none)")
    p.add_argument("--force", action="store_true", help="delete this model's checkpoints/outputs and start again")
    p.add_argument("--no-ema", action="store_true")
    return p.parse_args(argv)


def log_file_path(cfg, model_name, out_dir):
    """The full gpu run logs to the committed reproducibility folder; every other run logs next to its outputs."""
    if cfg["run_name"] == "gpu":
        return project_root().parent.parent / "reproducibility" / "raw_logs" / f"naman_task2_gpu_{model_name}_train.log"
    return out_dir / "train_log.txt"


def make_optimizer(model, lr, weight_decay):
    """AdamW with two groups: weights get weight decay; biases, LayerNorm weights and embedding tables do not."""
    embedding_params = {id(p) for m in model.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))
                        for p in m.parameters()}
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim < 2 or id(p) in embedding_params else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def atomic_save(obj, path):
    """Write to a temporary file first, then rename, so a crash never leaves a half-written checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_loss_curves(history, path):
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(df["epoch"], df["train_loss"], marker="o", label="train")
    ax1.plot(df["epoch"], df["val_loss"], marker="o", label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(df["epoch"], df["val_accuracy"], marker="o", color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.tag)
    name = args.model
    epochs = args.epochs or cfg["train"]["epochs"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"] / name
    out_dir = cfg["paths"]["output_dir"] / name
    log_path = log_file_path(cfg, name, out_dir)
    summary_path, last_path, best_path = out_dir / "summary.json", ckpt_dir / "last.pt", ckpt_dir / "best.pt"

    # ---- already finished? / force restart
    if args.force:
        if log_path.exists() and log_path.is_relative_to(out_dir):
            pass  # the log lives in out_dir and is removed with it
        elif log_path.exists():
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== restarted {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for folder in (ckpt_dir, out_dir):
            shutil.rmtree(folder, ignore_errors=True)
    elif summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        can_continue = (args.resume and last_path.exists() and not previous["stopped_early"]
                        and previous["epochs_run"] < epochs)
        if not can_continue:
            print(f"{name} ({cfg['run_name']}): already finished (use --force to retrain)")
            return 0
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_path)
    log = logger.log
    log(f"\n##### train {name} | run {cfg['run_name']} | {time.strftime('%Y-%m-%d %H:%M:%S')} #####")
    log_config(logger, cfg)

    # ---- setup
    set_seed(cfg["seed"])
    device, device_name = get_device(args.device)
    log(f"training device: {device} ({device_name})")
    vocab_size = get_vocab_size(cfg)
    model = build_model(name, cfg, vocab_size).to(device)
    n_params = count_parameters(model)
    log(f"parameters: {n_params:,}   vocab_size: {vocab_size}   epochs: {epochs}")

    tcfg = cfg["train"]
    train_loader = get_loader(cfg, name, "train", train=True, max_train=args.max_train)
    val_loader = get_loader(cfg, name, "val", train=False)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    log(f"train reviews: {train_loader.sampler.indices.size if hasattr(train_loader.sampler, 'indices') else '?'}, "
        f"val reviews: {val_loader.n}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}")

    weight_decay = cfg["models"][name].get("weight_decay", tcfg["weight_decay"])
    optimizer = make_optimizer(model, cfg["models"][name]["lr"], weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, tcfg["lr_warmup_ratio"]))

    amp_dtype, scaler = None, None
    if tcfg["amp"] and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16  # bfloat16 does not need loss scaling
        else:
            amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")
    log(f"mixed precision: {amp_dtype}")
    ema = EMA(model, tcfg["ema_decay"]) if tcfg["ema_decay"] is not None and not args.no_ema else None
    log(f"EMA: {'decay ' + str(tcfg['ema_decay']) if ema else 'off'}")

    # ---- resume
    state = {"epoch": 0, "step": 0, "best_loss": math.inf, "bad_epochs": 0, "history": [], "nan_losses": 0,
             "stopped_early": False}
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        if ema and ckpt["ema"]:
            ema.load_state_dict(ckpt["ema"])
        state = ckpt["state"]
        log(f"resumed from {last_path.name}: {state['epoch']} epochs already done")
    elif args.resume:
        log("--resume given but there is no last.pt: starting fresh")

    # ---- training
    peak_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    while state["epoch"] < epochs and not state["stopped_early"]:
        epoch = state["epoch"] + 1
        train_loader.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        loss_sum, seen, window_start, window_examples = 0.0, 0, time.time(), 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = forward_batch(model, batch, device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"].to(device))
            if not torch.isfinite(loss):  # skip the update, but keep the schedule moving
                state["nan_losses"] += 1
                scheduler.step()
                state["step"] += 1
                continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)
            state["step"] += 1
            n = len(batch["index"])
            loss_sum += loss.item() * n
            seen += n
            window_examples += n
            if state["step"] % 100 == 0:
                now = time.time()
                log(f"step {state['step']}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}  "
                    f"grad_norm {float(grad_norm):.3f}  {window_examples / (now - window_start):.0f} examples/s")
                window_start, window_examples = now, 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - epoch_start

        # validation (with the EMA weights when EMA is on)
        def validate():
            return evaluate_loader(model, val_loader, val_loader.n, device, amp_dtype)

        if ema:
            with ema.swapped_in(model):
                val = validate()
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()} \
                    if val["loss"] < state["best_loss"] else None
        else:
            val = validate()
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()} \
                if val["loss"] < state["best_loss"] else None
        epoch_time = time.time() - epoch_start
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
        else:
            peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / 2**20)  # approximate

        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
               "lr": scheduler.get_last_lr()[0], "train_time_s": train_time, "epoch_time_s": epoch_time,
               "train_examples_per_sec": seen / train_time, "peak_memory_mb": peak_memory_mb,
               "nan_losses": state["nan_losses"]}
        state["history"].append(row)
        state["epoch"] = epoch
        log(f"epoch {epoch}/{epochs}  train_loss {row['train_loss']:.4f}  val_loss {row['val_loss']:.4f}  "
            f"val_acc {row['val_accuracy']:.4f}  val_macro_f1 {row['val_macro_f1']:.4f}  "
            f"time {epoch_time:.1f}s  peak_mem {peak_memory_mb:.0f}MB  nan {state['nan_losses']}")
        if ema:
            log(f"  (validation used EMA weights, {ema.updates} updates)")

        if best_weights is not None:
            state["best_loss"], state["bad_epochs"] = val["loss"], 0
            atomic_save({"model": best_weights, "epoch": epoch, "ema": ema is not None,
                         "val": {k: val[k] for k in ("loss", "accuracy", "macro_f1")}, "model_name": name,
                         "config": _to_plain(cfg), "vocab_size": vocab_size, "max_len": cfg["data"]["max_len"]},
                        best_path)
            log(f"  new best (val_loss {val['loss']:.4f}) -> {best_path.name}")
        else:
            state["bad_epochs"] += 1
            if state["bad_epochs"] >= tcfg["patience"]:
                state["stopped_early"] = True
                log(f"early stopping: no improvement for {state['bad_epochs']} epochs")

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                     "ema": ema.state_dict() if ema else None, "state": state}, last_path)
        pd.DataFrame(state["history"]).to_csv(out_dir / "history.csv", index=False)

    # ---- summary
    history = pd.DataFrame(state["history"])
    best_row = history.loc[history["val_loss"].idxmin()]
    train_examples = history["train_examples_per_sec"] * history["train_time_s"]
    summary = {
        "model": name, "run_name": cfg["run_name"], "device": device_name, "torch_version": torch.__version__,
        "parameters": n_params, "epochs_run": int(len(history)), "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]), "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_macro_f1": float(best_row["val_macro_f1"]), "stopped_early": state["stopped_early"],
        "total_training_time_s": float(history["epoch_time_s"].sum()),
        "train_examples_per_sec": float(train_examples.sum() / history["train_time_s"].sum()),
        "peak_memory_mb": float(history["peak_memory_mb"].max()),
        "peak_memory_note": "CUDA max_memory_allocated" if device.type == "cuda" else "process RSS at epoch ends (approximate)",
        "nan_losses": state["nan_losses"], "seed": cfg["seed"], "ema_used": ema is not None,
        "max_train": args.max_train, "config": _to_plain(cfg),
    }
    save_json(summary, summary_path)
    save_loss_curves(state["history"], out_dir / "loss_curves.png")
    log(f"finished: best epoch {summary['best_epoch']}, val_loss {summary['best_val_loss']:.4f}, "
        f"val_acc {summary['best_val_accuracy']:.4f}, total time {summary['total_training_time_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
