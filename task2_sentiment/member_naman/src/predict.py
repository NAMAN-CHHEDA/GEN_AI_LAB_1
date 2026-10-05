"""Run the saved best checkpoint of a model on the TEST split (once) and save its predictions.

Run:  python src/predict.py --config local --model all --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_loader
from engine import evaluate_loader
from models import build_model
from utils import Logger, get_device, load_config, save_json

MODEL_NAMES = ["ngram_bag", "transformer", "bigru"]


def checkpoint_path(cfg, model_name):
    return cfg["paths"]["checkpoint_dir"] / model_name / "best.pt"


def load_best_model(cfg, model_name, device):
    """Build the model and load the weights of its best epoch (refuses if there is no checkpoint)."""
    path = checkpoint_path(cfg, model_name)
    if not path.exists():
        raise SystemExit(f"No checkpoint for '{model_name}' in run '{cfg['run_name']}': "
                         f"train it first with  python src/train.py --config {cfg['run_name']} --model {model_name}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert ckpt["max_len"] == cfg["data"]["max_len"], "checkpoint was trained with a different max_len"
    model = build_model(model_name, cfg, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def read_test_raw(cfg):
    """test_raw.csv: orig_index, text, label, raw_words (in the original test order)."""
    return pd.read_csv(cfg["paths"]["processed_dir"] / "test_raw.csv", dtype={"text": str}, keep_default_na=False)


def predict_model(cfg, model_name, device, device_name, logger):
    """Predict the test split with one model and write test_predictions.csv and inference.json."""
    model, ckpt = load_best_model(cfg, model_name, device)
    loader = get_loader(cfg, model_name, "test", train=False)
    start = time.time()
    result = evaluate_loader(model, loader, loader.n, device, None)  # full float32 for the final numbers
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start

    test_raw = read_test_raw(cfg)
    assert len(test_raw) == loader.n, "test_raw.csv and the test arrays have different sizes"
    assert (test_raw["label"].to_numpy() == result["labels"].astype(int)).all(), "test labels do not match test_raw.csv"
    probs = result["probs"]
    table = pd.DataFrame({"orig_index": test_raw["orig_index"], "true_label": test_raw["label"],
                          "predicted_label": (probs >= 0.5).astype(int), "positive_probability": probs})
    out_dir = cfg["paths"]["output_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "test_predictions.csv", index=False)
    save_json({"model": model_name, "device": device_name, "n_reviews": loader.n, "seconds": seconds,
               "inference_examples_per_sec": loader.n / seconds, "checkpoint_epoch": ckpt["epoch"]},
              out_dir / "inference.json")
    accuracy = float((table["predicted_label"] == table["true_label"]).mean())
    logger.log(f"{model_name}: {loader.n} test reviews in {seconds:.1f}s ({loader.n / seconds:.0f}/s), "
               f"accuracy {accuracy:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict the test split with the best checkpoint")
    parser.add_argument("--config", default="local")
    parser.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    names = MODEL_NAMES if args.model == "all" else [args.model]
    missing = [n for n in names if not checkpoint_path(cfg, n).exists()]
    if missing:
        raise SystemExit(f"No checkpoint for {missing} in run '{cfg['run_name']}': train them first")
    device, device_name = get_device(args.device)
    logger = Logger(cfg["paths"]["output_dir"] / "predict.log")
    logger.log(f"predict | run {cfg['run_name']} | device {device} ({device_name})")
    for name in names:
        predict_model(cfg, name, device, device_name, logger)


if __name__ == "__main__":
    main()
