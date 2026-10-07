"""Resume post-training: generation + full metrics from best checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

SEED = 3141


def main() -> None:
    import torch
    from charlm.config import load_config, resolve_paths
    from charlm.data import make_loaders, prepare_datasets
    from charlm.generate import run_generation_suite
    from charlm.metrics import collect_full_metrics, write_metrics_report
    from charlm.model import build_model
    from charlm.utils import amp_enabled, pick_device, set_seed

    cfg = load_config(SRC / "config.yaml")
    cfg["seed"] = SEED
    paths = resolve_paths(cfg, SRC)
    device = pick_device(cfg.get("device", "auto"))
    use_amp = amp_enabled(cfg, device)
    set_seed(SEED)

    data = prepare_datasets(cfg, paths, SEED)
    train_loader, val_loader = make_loaders(
        data["train_ds"],
        data["val_ds"],
        cfg["train"]["batch_size"],
        cfg["eval"]["batch_size"],
        SEED,
    )

    ckpt_path = paths["checkpoint_dir"] / f"{cfg['run_name']}_best.pt"
    blob = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg, data["vocab_size"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    print("loaded:", ckpt_path, "| best_epoch:", blob.get("best_epoch"))

    summary_path = paths["logs_dir"] / f"{cfg['run_name']}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

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
    for name, texts in generated.items():
        print(f"\n======== {name} ========")
        print(texts[0][:450])
    print("\nDONE post-train eval/generation")


if __name__ == "__main__":
    main()
