"""End-to-end smoke run for Task 1 (Sarvesh)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

SEED = 3141

from charlm.config import load_config, resolve_paths
from charlm.data import make_loaders, prepare_datasets
from charlm.generate import run_generation_suite
from charlm.metrics import collect_full_metrics, write_metrics_report
from charlm.model import build_model, count_parameters
from charlm.plots import plot_loss_curves
from charlm.sanity import run_sanity_suite
from charlm.train import train_model
from charlm.utils import amp_enabled, pick_device, set_seed


def main() -> None:
    cfg = load_config(SRC / "config_smoke.yaml")
    cfg["seed"] = SEED
    paths = resolve_paths(cfg, SRC)
    device = pick_device(cfg.get("device", "auto"))
    use_amp = amp_enabled(cfg, device)
    set_seed(SEED)
    print("device", device)

    data = prepare_datasets(cfg, paths, SEED)
    train_loader, val_loader = make_loaders(
        data["train_ds"],
        data["val_ds"],
        cfg["train"]["batch_size"],
        cfg["eval"]["batch_size"],
        SEED,
    )
    print("vocab", data["vocab_size"], "train", len(data["train_ds"]), "val", len(data["val_ds"]))
    print(json.dumps(data["split_info"], indent=2))

    sanity = run_sanity_suite(cfg, data["vocab_size"], device)
    assert all(sanity.values()), sanity

    set_seed(SEED)
    model = build_model(cfg, data["vocab_size"])
    print("params", count_parameters(model))
    summary = train_model(model, train_loader, val_loader, cfg, paths, device, SEED)
    plot_loss_curves(
        summary["history"],
        paths["output_dir"] / f"{cfg['run_name']}_loss_curves.png",
        5,
    )
    generated, speeds, _ = run_generation_suite(
        model,
        cfg,
        data["char_to_idx"],
        data["idx_to_char"],
        device,
        SEED,
        paths["output_dir"],
    )
    rows = collect_full_metrics(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        use_amp,
        summary,
        generated,
        speeds,
    )
    write_metrics_report(paths["metrics_csv"], rows)
    print("OK smoke complete")
    print("sample:", generated["greedy"][0][:200])
    print("metrics rows", len(rows))


if __name__ == "__main__":
    main()
