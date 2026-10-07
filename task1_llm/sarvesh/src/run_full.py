"""Full Task 1 training run (uses config.yaml by default)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

# Change this if your course assigned a different seed.
SEED = 3141


def main() -> None:
    from charlm.config import load_config, resolve_paths
    from charlm.data import make_loaders, prepare_datasets
    from charlm.generate import run_generation_suite
    from charlm.metrics import collect_full_metrics, write_metrics_report
    from charlm.model import build_model, count_parameters
    from charlm.plots import plot_loss_curves
    from charlm.sanity import run_sanity_suite
    from charlm.train import train_model
    from charlm.utils import amp_enabled, pick_device, set_seed
    import torch

    config_name = os.environ.get("TASK1_CONFIG", "config.yaml")
    cfg = load_config(SRC / config_name)
    cfg["seed"] = SEED
    paths = resolve_paths(cfg, SRC)
    device = pick_device(cfg.get("device", "auto"))
    use_amp = amp_enabled(cfg, device)
    set_seed(SEED)

    print("config:", config_name)
    print("device:", device, "| amp:", use_amp, "| torch:", torch.__version__)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    data = prepare_datasets(cfg, paths, SEED)
    train_loader, val_loader = make_loaders(
        data["train_ds"],
        data["val_ds"],
        cfg["train"]["batch_size"],
        cfg["eval"]["batch_size"],
        SEED,
    )
    print("vocab:", data["vocab_size"])
    print("train windows:", len(data["train_ds"]), "| val windows:", len(data["val_ds"]))
    print(json.dumps(data["split_info"], indent=2))

    sanity = run_sanity_suite(cfg, data["vocab_size"], device)
    assert sanity["causal_mask"], "causal mask failed"

    set_seed(SEED)
    model = build_model(cfg, data["vocab_size"])
    print("parameters:", f"{count_parameters(model):,}")

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        paths=paths,
        device=device,
        seed=SEED,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))

    curve_path = paths["output_dir"] / f"{cfg['run_name']}_loss_curves.png"
    plot_loss_curves(summary["history"], curve_path, cfg["plots"]["smooth_window"])
    print("loss curves ->", curve_path)

    # Reload best checkpoint for eval / generation
    ckpt_name = cfg["eval"].get("checkpoint", "best")
    ckpt_path = paths["checkpoint_dir"] / f"{cfg['run_name']}_{ckpt_name}.pt"
    blob = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()
    print("loaded:", ckpt_path, "| best_epoch:", blob.get("best_epoch"))

    generated, speeds, _ = run_generation_suite(
        model=model,
        cfg=cfg,
        char_to_idx=data["char_to_idx"],
        idx_to_char=data["idx_to_char"],
        device=device,
        seed=SEED,
        output_dir=paths["output_dir"],
    )
    rows = collect_full_metrics(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        use_amp=use_amp,
        summary=summary,
        generated=generated,
        gen_tokens_per_sec=speeds,
    )
    write_metrics_report(paths["metrics_csv"], rows)
    print("metrics ->", paths["metrics_csv"])
    print("\n======== greedy sample ========")
    print(generated["greedy"][0][:600])
    print("\nDONE. Fill results.md and failure_analysis.md from outputs/.")


if __name__ == "__main__":
    main()
