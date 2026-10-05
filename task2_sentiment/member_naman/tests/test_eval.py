"""Plain-assert tests for metrics.py, predict.py, evaluate.py and compare_external.py (local config, CPU).

Run:  python tests/test_eval.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as skm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import compare_external
import evaluate
import metrics
import predict
from dataset import get_loader
from engine import forward_batch
from slices import make_slices
from utils import load_config, project_root

LOCAL = load_config("local")

# Written out explicitly (not imported from evaluate.py) so the test checks the real contract.
EXPECTED_COLUMNS = [
    "model", "role", "n_test",
    "accuracy", "accuracy_ci_low", "accuracy_ci_high",
    "precision_macro", "recall_macro", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high",
    "precision_micro", "recall_micro", "f1_micro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "tn", "fp", "fn", "tp", "roc_auc", "pr_auc",
    "mcc", "mcc_ci_low", "mcc_ci_high", "brier", "ece",
    "mcnemar_b", "mcnemar_c", "mcnemar_p_exact", "mcnemar_chi2", "mcnemar_p_chi2",
    "slice_short_n", "slice_short_macro_f1", "slice_short_error_rate",
    "slice_medium_n", "slice_medium_macro_f1", "slice_medium_error_rate",
    "slice_long_n", "slice_long_macro_f1", "slice_long_error_rate",
    "slice_has_negation_n", "slice_has_negation_macro_f1", "slice_has_negation_error_rate",
    "slice_has_contrast_n", "slice_has_contrast_macro_f1", "slice_has_contrast_error_rate",
    "parameters", "training_time_s", "train_examples_per_sec", "peak_memory_mb", "inference_examples_per_sec",
    "train_device", "inference_device", "best_epoch", "epochs_run",
]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def random_predictions(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    prob = np.clip(0.6 * y + 0.2 + rng.normal(0, 0.25, n), 0.001, 0.999)  # informative but noisy
    pred = (prob >= 0.5).astype(int)
    flip = rng.random(n) < 0.1  # extra random mistakes
    return y, np.where(flip, 1 - pred, pred), prob


def test_metrics_match_sklearn():
    y, pred, prob = random_predictions(5000, 0)
    result = metrics.core_metrics(y, pred, prob)
    assert abs(result["accuracy"] - skm.accuracy_score(y, pred)) < 1e-12
    for average in ("macro", "micro", "weighted"):
        p, r, f, _ = skm.precision_recall_fscore_support(y, pred, average=average, zero_division=0)
        assert abs(result[f"precision_{average}"] - p) < 1e-12, average
        assert abs(result[f"recall_{average}"] - r) < 1e-12, average
        assert abs(result[f"f1_{average}"] - f) < 1e-12, average
    assert abs(result["f1_micro"] - result["accuracy"]) < 1e-12
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (tn, fp, fn, tp)
    assert abs(result["mcc"] - skm.matthews_corrcoef(y, pred)) < 1e-12
    assert abs(result["roc_auc"] - skm.roc_auc_score(y, prob)) < 1e-12
    assert abs(result["pr_auc"] - skm.average_precision_score(y, prob)) < 1e-12
    assert abs(result["brier"] - skm.brier_score_loss(y, prob)) < 1e-12
    # ECE against a slow, obviously-correct loop over bins (lo, hi], first bin includes 0
    edges, expected = np.linspace(0, 1, 16), 0.0
    for k in range(15):
        in_bin = (prob > edges[k]) & (prob <= edges[k + 1]) if k > 0 else (prob >= 0) & (prob <= edges[1])
        if in_bin.any():
            expected += in_bin.mean() * abs(y[in_bin].mean() - prob[in_bin].mean())
    assert abs(result["ece"] - expected) < 1e-9


def test_bootstrap_ci():
    widths = {}
    for n in (500, 20000):
        y, pred, _ = random_predictions(n, 1)
        ci = metrics.bootstrap_ci(y, pred)
        point = {"accuracy": skm.accuracy_score(y, pred), "macro_f1": skm.f1_score(y, pred, average="macro"),
                 "mcc": skm.matthews_corrcoef(y, pred)}
        for name, (low, high) in ci.items():
            assert low <= point[name] <= high, (n, name, low, point[name], high)
        widths[n] = ci["accuracy"][1] - ci["accuracy"][0]
    assert widths[20000] < widths[500], widths


def test_mcnemar():
    a = np.array([True] * 10 + [True] * 20 + [False] * 5)
    b = np.array([False] * 10 + [True] * 20 + [False] * 5)
    result = metrics.mcnemar(a, b)
    assert (result["b"], result["c"]) == (10, 0)
    assert abs(result["p_exact"] - 2 * 0.5 ** 10) < 1e-12
    assert abs(result["chi2"] - (9 ** 2) / 10) < 1e-12  # (|10 - 0| - 1)^2 / 10
    same = metrics.mcnemar(np.array([True, False, True, False]), np.array([False, True, True, False]))
    assert (same["b"], same["c"]) == (1, 1) and abs(same["p_exact"] - 1.0) < 1e-12


def test_ece():
    rng = np.random.default_rng(0)
    prob = rng.random(50000)
    y = (rng.random(50000) < prob).astype(int)  # calibrated by construction
    assert metrics.expected_calibration_error(y, prob) < 0.02
    y = np.array([0, 1] * 5000)
    assert abs(metrics.expected_calibration_error(y, np.full(10000, 0.9)) - 0.4) < 1e-9


def test_slices():
    texts = ["It was wasn't good at all", "never again!", "Great place", "good but pricey", "However it works"]
    words = np.array([5, 40, 60, 250, 70])
    masks = make_slices(texts, words)
    assert set(masks) == {"short", "medium", "long", "has_negation", "has_contrast"}
    for mask in masks.values():
        assert mask.dtype == bool and len(mask) == len(texts)
    assert masks["has_negation"].tolist() == [True, True, False, False, False]
    assert masks["has_contrast"].tolist() == [False, False, False, True, True]
    assert masks["short"].tolist() == [True, True, False, False, False] and masks["long"].tolist()[3]
    real = pd.read_csv(LOCAL["paths"]["processed_dir"] / "test_raw.csv", dtype={"text": str}, keep_default_na=False)
    real_masks = make_slices(real["text"].tolist(), real["raw_words"].to_numpy())
    assert all(len(m) == len(real) and m.dtype == bool for m in real_masks.values())


def make_all_predictions():
    quiet(predict.main, ["--config", "local", "--model", "all", "--device", "cpu"])


def test_predict_order_matches_batch_of_one():
    make_all_predictions()
    for name in ("ngram_bag", "bigru"):
        table = pd.read_csv(LOCAL["paths"]["output_dir"] / name / "test_predictions.csv")
        assert list(table.columns) == ["orig_index", "true_label", "predicted_label", "positive_probability"]
        raw = predict.read_test_raw(LOCAL)
        assert (table["orig_index"].to_numpy() == raw["orig_index"].to_numpy()).all()
        model, _ = predict.load_best_model(LOCAL, name, torch.device("cpu"))
        loader = get_loader(LOCAL, name, "test", train=False)
        for i in range(20):
            with torch.no_grad():
                logit = forward_batch(model, loader.builder.make(np.array([i])), torch.device("cpu"))
            assert abs(torch.sigmoid(logit)[0].item() - table["positive_probability"][i]) < 1e-4, (name, i)


def test_evaluate_writes_report_and_not_to_root():
    make_all_predictions()
    root = project_root()
    before = sorted((p.name, p.stat().st_mtime) for p in root.iterdir())
    quiet(evaluate.main, ["--config", "local"])
    assert sorted((p.name, p.stat().st_mtime) for p in root.iterdir()) == before  # nothing new in member_naman/
    report = pd.read_csv(LOCAL["paths"]["output_dir"] / "metrics_report.csv")
    missing = [c for c in EXPECTED_COLUMNS if c not in report.columns]
    assert not missing, missing
    assert report["model"].tolist() == ["ngram_bag", "transformer", "bigru"]
    assert (LOCAL["paths"]["output_dir"] / "metrics_table.md").exists()
    for plot in ("roc_curves", "pr_curves", "reliability_diagrams", "macro_f1_bar", "slice_error_rates",
                 "loss_curves_overlay", "confusion_matrix_bigru"):
        assert (LOCAL["paths"]["output_dir"] / "eval" / f"{plot}.png").exists(), plot


def test_compare_external_with_itself():
    make_all_predictions()
    ours = pd.read_csv(LOCAL["paths"]["output_dir"] / "ngram_bag" / "test_predictions.csv")
    result = compare_external.compare_predictions(ours, ours.copy(), "ngram_bag", "self")
    assert result["accuracy_difference_ours_minus_other"] == 0.0
    assert result["mcnemar_b_ours_right_other_wrong"] == 0 and result["mcnemar_c_ours_wrong_other_right"] == 0
    assert result["mcnemar_p_exact"] == 1.0
    assert result["accuracy_difference_ci95"] == [0.0, 0.0]
    for broken in (ours.iloc[:-1], ours.assign(true_label=1 - ours["true_label"])):  # wrong size / wrong labels
        try:
            compare_external.compare_predictions(ours, broken, "ngram_bag", "bad")
            raise AssertionError("a mismatching file should have been rejected")
        except ValueError:
            pass


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {name} -> {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
