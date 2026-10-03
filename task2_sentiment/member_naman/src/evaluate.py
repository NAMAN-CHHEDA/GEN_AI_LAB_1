"""Compute every test metric for all models that have predictions, and make the comparison plots.

Run:  python src/evaluate.py --config local        (predict.py must have been run first)

Baseline = ngram_bag; transformer and bigru are the experimental models. Metric definitions live in
metrics.py (micro-F1 equals accuracy for single-label binary classification; ECE uses 15 equal-width
bins on the positive-class probability). McNemar: b = baseline right and model wrong, c = baseline
wrong and model right.
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
from predict import MODEL_NAMES, read_test_raw
from slices import make_slices
from utils import Logger, load_config, load_json, project_root

BASELINE = "ngram_bag"
COLORS = {"ngram_bag": "#0072B2", "transformer": "#E69F00", "bigru": "#009E73"}
SLICE_NAMES = ["short", "medium", "long", "has_negation", "has_contrast"]

# Every column the metrics report must contain.
REQUIRED_COLUMNS = (
    ["model", "role", "n_test", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
     "precision_macro", "recall_macro", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high",
     "precision_micro", "recall_micro", "f1_micro", "precision_weighted", "recall_weighted", "f1_weighted",
     "tn", "fp", "fn", "tp", "roc_auc", "pr_auc", "mcc", "mcc_ci_low", "mcc_ci_high", "brier", "ece",
     "mcnemar_b", "mcnemar_c", "mcnemar_p_exact", "mcnemar_chi2", "mcnemar_p_chi2"]
    + [f"slice_{s}_{what}" for s in SLICE_NAMES for what in ("n", "macro_f1", "error_rate")]
    + ["parameters", "training_time_s", "train_examples_per_sec", "peak_memory_mb", "inference_examples_per_sec",
       "train_device", "inference_device", "best_epoch", "epochs_run"]
)


def load_model_results(cfg):
    """{model: (predictions DataFrame, summary dict, inference dict)} for models that have predictions."""
    results = {}
    for name in MODEL_NAMES:
        folder = cfg["paths"]["output_dir"] / name
        if (folder / "test_predictions.csv").exists():
            results[name] = (pd.read_csv(folder / "test_predictions.csv"), load_json(folder / "summary.json"),
                             load_json(folder / "inference.json"))
    return results


def build_report_row(name, predictions, summary, inference, masks, baseline_correct):
    """One row of the metrics report."""
    y, pred, prob = (predictions[c].to_numpy() for c in ("true_label", "predicted_label", "positive_probability"))
    row = {"model": name, "role": "baseline" if name == BASELINE else "experimental", "n_test": len(y)}
    row.update(metrics.core_metrics(y, pred, prob))
    for metric, key in (("accuracy", "accuracy"), ("macro_f1", "f1_macro"), ("mcc", "mcc")):
        low, high = metrics.bootstrap_ci(y, pred)[metric]
        row[f"{key}_ci_low"], row[f"{key}_ci_high"] = low, high

    if name == BASELINE:
        test = {"b": np.nan, "c": np.nan, "p_exact": np.nan, "chi2": np.nan, "p_chi2": np.nan}
    else:
        test = metrics.mcnemar(baseline_correct, pred == y)
    row.update({"mcnemar_b": test["b"], "mcnemar_c": test["c"], "mcnemar_p_exact": test["p_exact"],
                "mcnemar_chi2": test["chi2"], "mcnemar_p_chi2": test["p_chi2"]})

    for slice_name in SLICE_NAMES:
        mask = masks[slice_name]
        counts = metrics.confusion_counts(y[mask], pred[mask])
        row[f"slice_{slice_name}_n"] = int(mask.sum())
        row[f"slice_{slice_name}_macro_f1"] = float(metrics.macro_f1_from_counts(*counts))
        row[f"slice_{slice_name}_error_rate"] = float(1 - metrics.accuracy_from_counts(*counts))

    row.update({"parameters": summary["parameters"], "training_time_s": summary["total_training_time_s"],
                "train_examples_per_sec": summary["train_examples_per_sec"], "peak_memory_mb": summary["peak_memory_mb"],
                "inference_examples_per_sec": inference["inference_examples_per_sec"],
                "train_device": summary["device"], "inference_device": inference["device"],
                "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"]})
    return row


def markdown_table(report):
    """Readable markdown table of the main columns."""
    def with_ci(r, key):
        return f"{r[key]:.4f} [{r[key + '_ci_low']:.4f}, {r[key + '_ci_high']:.4f}]"

    header = ["model", "accuracy [95% CI]", "macro-F1 [95% CI]", "ROC-AUC", "PR-AUC", "MCC [95% CI]", "Brier", "ECE",
              "McNemar p (vs baseline)", "params", "train time (s)", "inference ex/s"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, r in report.iterrows():
        p = "baseline" if r["role"] == "baseline" else f"{r['mcnemar_p_exact']:.3g} (b={int(r['mcnemar_b'])}, c={int(r['mcnemar_c'])})"
        cells = [r["model"], with_ci(r, "accuracy"), with_ci(r, "f1_macro"), f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}",
                 with_ci(r, "mcc"), f"{r['brier']:.4f}", f"{r['ece']:.4f}", p, f"{int(r['parameters']):,}",
                 f"{r['training_time_s']:.0f}", f"{r['inference_examples_per_sec']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- plots

def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(results, eval_dir):
    """ROC and precision-recall curves, all models on one axes each."""
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, (pred, _, _) in results.items():
            y, prob = pred["true_label"], pred["positive_probability"]
            if kind == "roc":
                x_values, y_values, _ = roc_curve(y, prob)
                label = f"{name} (AUC {metrics.roc_auc(y, prob):.4f})"
            else:
                y_values, x_values, _ = precision_recall_curve(y, prob)
                label = f"{name} (AP {metrics.pr_auc(y, prob):.4f})"
            ax.plot(x_values, y_values, color=COLORS[name], label=label, linewidth=1.8)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
            ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)")
        else:
            ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curves (test set)")
        ax.legend(loc="lower right")
        finish(fig, eval_dir / ("roc_curves.png" if kind == "roc" else "pr_curves.png"))


def plot_reliability(results, eval_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4), squeeze=False)
    for ax, (name, (pred, _, _)) in zip(axes[0], results.items()):
        counts, mean_prob, frac_positive = metrics.reliability_bins(pred["true_label"], pred["positive_probability"])
        used = counts > 0
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="perfectly calibrated")
        ax.plot(mean_prob[used], frac_positive[used], color=COLORS[name], marker="o", linewidth=1.5, label=name)
        ax.set(xlabel="mean predicted probability", ylabel="fraction of positive reviews",
               title=f"{name}: ECE {metrics.expected_calibration_error(pred['true_label'], pred['positive_probability']):.4f}",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    finish(fig, eval_dir / "reliability_diagrams.png")


def plot_confusion_matrices(report, eval_dir):
    for _, r in report.iterrows():
        matrix = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
               xlabel="predicted", ylabel="true", title=f"{r['model']}: confusion matrix")
        finish(fig, eval_dir / f"confusion_matrix_{r['model']}.png")


def plot_macro_f1(report, eval_dir):
    values = report["f1_macro"].to_numpy()
    low, high = report["f1_macro_ci_low"].to_numpy(), report["f1_macro_ci_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(report["model"], values, yerr=[values - low, high - values], capsize=6,
           color=[COLORS[m] for m in report["model"]])
    ax.set_ylim(max(0.0, np.floor((low.min() - 0.02) * 100) / 100), min(1.0, high.max() + 0.01))
    ax.set(ylabel="macro-F1 (axis does not start at 0)", title="Macro-F1 on the test set (95% bootstrap CI)")
    finish(fig, eval_dir / "macro_f1_bar.png")


def plot_slice_errors(report, eval_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(report)
    for k, (_, r) in enumerate(report.iterrows()):
        errors = [100 * r[f"slice_{s}_error_rate"] for s in SLICE_NAMES]
        ax.bar(np.arange(len(SLICE_NAMES)) + k * width, errors, width, color=COLORS[r["model"]], label=r["model"])
    sizes = [int(report.iloc[0][f"slice_{s}_n"]) for s in SLICE_NAMES]
    ax.set_xticks(np.arange(len(SLICE_NAMES)) + width * (len(report) - 1) / 2)
    ax.set_xticklabels([f"{s}\n(n={n:,})" for s, n in zip(SLICE_NAMES, sizes)])
    ax.set(ylabel="error rate (%)", title="Error rate per test slice")
    ax.legend()
    finish(fig, eval_dir / "slice_error_rates.png")


def plot_loss_overlay(cfg, results, eval_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in results:
        history = pd.read_csv(cfg["paths"]["output_dir"] / name / "history.csv")
        ax.plot(history["epoch"], history["train_loss"], color=COLORS[name], linestyle="--", marker="o", label=f"{name} train")
        ax.plot(history["epoch"], history["val_loss"], color=COLORS[name], linestyle="-", marker="o", label=f"{name} validation")
    ax.set(xlabel="epoch", ylabel="loss", title="Training and validation loss")
    ax.legend(fontsize=8)
    finish(fig, eval_dir / "loss_curves_overlay.png")


# ----------------------------------------------------------------------------- main

def run_evaluation(cfg, logger):
    """Build the report, write all files, and return the report DataFrame."""
    results = load_model_results(cfg)
    if BASELINE not in results:
        raise SystemExit(f"No test predictions for the baseline '{BASELINE}': run predict.py first")
    test_raw = read_test_raw(cfg)
    masks = make_slices(test_raw["text"].tolist(), test_raw["raw_words"].to_numpy())
    for name, (pred, _, _) in results.items():  # every file must be about the same reviews in the same order
        assert (pred["orig_index"].to_numpy() == test_raw["orig_index"].to_numpy()).all(), name
        assert (pred["true_label"].to_numpy() == test_raw["label"].to_numpy()).all(), name

    base = results[BASELINE][0]
    baseline_correct = (base["predicted_label"] == base["true_label"]).to_numpy()
    rows = [build_report_row(name, pred, summary, inference, masks, baseline_correct)
            for name, (pred, summary, inference) in results.items()]
    report = pd.DataFrame(rows)[REQUIRED_COLUMNS]

    out_dir = cfg["paths"]["output_dir"]
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "metrics_report.csv", index=False)
    table = markdown_table(report)
    (out_dir / "metrics_table.md").write_text(table, encoding="utf-8")
    if cfg["run_name"] == "gpu":  # only the full run is copied to the member folder
        shutil.copyfile(out_dir / "metrics_report.csv", project_root() / "metrics_report.csv")

    plot_curves(results, eval_dir)
    plot_reliability(results, eval_dir)
    plot_confusion_matrices(report, eval_dir)
    plot_macro_f1(report, eval_dir)
    plot_slice_errors(report, eval_dir)
    plot_loss_overlay(cfg, results, eval_dir)
    logger.log(table)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate all models on the test split")
    parser.add_argument("--config", default="local")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="accepted for symmetry with predict.py; evaluate.py reads saved predictions and runs no model")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logger = Logger(cfg["paths"]["output_dir"] / "evaluate.log")
    logger.log(f"evaluate | run {cfg['run_name']}")
    run_evaluation(cfg, logger)


if __name__ == "__main__":
    main()
