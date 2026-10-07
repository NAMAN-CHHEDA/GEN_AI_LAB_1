"""Functions used by the notebook cells. They only LOAD saved artifacts and show them: nothing is trained.

If an artifact does not exist yet, the function prints a "not available yet" message instead of failing.
"""

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model, parameter_breakdown
from predict import MODEL_NAMES
from utils import get_device, load_config, load_json, project_root


def missing(what):
    print(f"Not available yet: {what}")


def show_image(path, title=None):
    path = Path(path)
    if path.exists():
        if title:
            display(Markdown(f"**{title}**"))
        display(Image(filename=str(path)))
    else:
        missing(path.name)


def setup(config_name):
    """Load the config and print which run and device this notebook is about."""
    cfg = load_config(config_name)
    device, device_name = get_device()
    print(f"run: {cfg['run_name']}   config: {config_name}.yaml   seed: {cfg['seed']}")
    print(f"device on this machine: {device} ({device_name})")
    print(f"models: {', '.join(MODEL_NAMES)}")
    return cfg


def show_data_summary(cfg):
    path = cfg["paths"]["processed_dir"] / "split_info.json"
    if not path.exists():
        return missing("split_info.json (run data.py)")
    info = load_json(path)
    rows = [{"split": s, "reviews": info["counts"][s], "negative": info["class_counts"][s]["label_0"],
             "positive": info["class_counts"][s]["label_1"],
             "OOV tokens %": info["oov_token_percent"][s], "empty after cleaning": info["empty_after_cleaning"][s]}
            for s in ("train", "val", "test")]
    display(pd.DataFrame(rows))
    print(f"vocabulary size: {info['vocab_size']}   cleaned mean tokens: {info['mean_tokens_per_review']}")
    print(f"rows dropped as invalid: {info['dropped_rows']}")
    print(f"reviews with literal escape sequences: train {info['reviews_with_escapes']['train']['any']:,}, "
          f"test {info['reviews_with_escapes']['test']['any']:,}")


def show_eda(cfg):
    eda_dir = cfg["paths"]["output_dir"] / "eda"
    for name in ("class_balance", "length_raw_words", "length_tokens"):
        show_image(eda_dir / f"{name}.png", name)
    examples = cfg["paths"]["processed_dir"] / "examples.txt"
    if examples.exists():
        print("Before / after cleaning (first training review):")
        print(examples.read_text(encoding="utf-8").split("\n\n")[0])
    else:
        missing("examples.txt")


def show_architectures(cfg):
    vocab_path = cfg["paths"]["processed_dir"] / "vocab.json"
    if not vocab_path.exists():
        return missing("vocab.json")
    vocab_size = len(load_json(vocab_path))
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, cfg, vocab_size)
        total, embeddings, rest = parameter_breakdown(model)
        print(f"===== {name} =====")
        print(model)
        rows.append({"model": name, "total": total, "embeddings": embeddings, "other": rest})
    display(pd.DataFrame(rows))


def show_training(cfg):
    out = cfg["paths"]["output_dir"]
    rows = []
    for name in MODEL_NAMES:
        summary_path = out / name / "summary.json"
        if not summary_path.exists():
            missing(f"{name}/summary.json")
            continue
        s = load_json(summary_path)
        rows.append({"model": name, "epochs run": s["epochs_run"], "best epoch": s["best_epoch"],
                     "best val loss": round(s["best_val_loss"], 4), "best val acc": round(s["best_val_accuracy"], 4),
                     "train time (s)": round(s["total_training_time_s"]), "examples/s": round(s["train_examples_per_sec"]),
                     "peak memory (MB)": round(s["peak_memory_mb"]), "device": s["device"]})
        show_image(out / name / "loss_curves.png", f"{name}: loss curves")
    if rows:
        display(pd.DataFrame(rows))


def show_test_metrics(cfg):
    out = cfg["paths"]["output_dir"]
    table = out / "metrics_table.md"
    if not table.exists():
        return missing("metrics_table.md (run predict.py and evaluate.py)")
    display(Markdown(table.read_text(encoding="utf-8")))
    for name in ("roc_curves", "pr_curves", "reliability_diagrams", "macro_f1_bar", "loss_curves_overlay"):
        show_image(out / "eval" / f"{name}.png", name)
    for model in MODEL_NAMES:
        show_image(out / "eval" / f"confusion_matrix_{model}.png")


def show_mcnemar_and_slices(cfg):
    path = cfg["paths"]["output_dir"] / "metrics_report.csv"
    if not path.exists():
        return missing("metrics_report.csv")
    report = pd.read_csv(path)
    mcnemar = report[report["role"] == "experimental"][["model", "mcnemar_b", "mcnemar_c", "mcnemar_p_exact",
                                                       "mcnemar_chi2", "mcnemar_p_chi2"]]
    print("McNemar test against the baseline (b: baseline right and model wrong; c: baseline wrong and model right)")
    display(mcnemar)
    slices = ["short", "medium", "long", "has_negation", "has_contrast"]
    rows = [{"model": r["model"], "slice": s, "n": int(r[f"slice_{s}_n"]), "macro-F1": round(r[f"slice_{s}_macro_f1"], 4),
             "error rate": round(r[f"slice_{s}_error_rate"], 4)} for _, r in report.iterrows() for s in slices]
    display(pd.DataFrame(rows))
    show_image(cfg["paths"]["output_dir"] / "eval" / "slice_error_rates.png")


def show_error_review(cfg):
    path = cfg["paths"]["output_dir"] / "error_review" / "candidates.csv"
    if not path.exists():
        return missing("error_review/candidates.csv (run error_review.py)")
    candidates = pd.read_csv(path)
    candidates["text"] = candidates["text"].str.slice(0, 200)
    display(candidates[["rank", "bucket", "orig_index", "true_label", "predicted_label", "positive_probability",
                        "raw_words", "truncated_at_max_len", "text"]])


def show_hardware(cfg):
    _, device_name = get_device()
    print(f"torch {torch.__version__}, numpy {np.__version__}, pandas {pd.__version__}, Python {platform.python_version()}")
    print(f"platform: {platform.system()} {platform.release()}")
    print(f"device: {device_name}   CUDA available: {torch.cuda.is_available()}")
    print(f"seed: {cfg['seed']}")
    for name in ("local", "gpu"):
        digest = hashlib.sha256((project_root() / "configs" / f"{name}.yaml").read_bytes()).hexdigest()[:12]
        print(f"config {name}.yaml sha256: {digest}")
    for model in MODEL_NAMES:
        path = cfg["paths"]["output_dir"] / model / "summary.json"
        if path.exists():
            s = load_json(path)
            print(f"{model}: trained on {s['device']} with torch {s['torch_version']}")
