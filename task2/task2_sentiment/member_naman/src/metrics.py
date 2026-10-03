"""Evaluation metrics written by hand from confusion-matrix counts (plus a few scikit-learn curves).

Binary task: class 1 = positive review, class 0 = negative. Counts are tn, fp, fn, tp
(true class first, predicted class second). Most functions also work on numpy arrays of counts,
which is how the bootstrap computes 1000 resamples at once.
"""

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion_counts(y_true, y_pred):
    """(tn, fp, fn, tp) as python ints."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def safe_div(a, b):
    """a / b, with 0 where b is 0 (same as scikit-learn's zero_division=0)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), 0.0)


def accuracy_from_counts(tn, fp, fn, tp):
    return safe_div(tn + tp, tn + fp + fn + tp)


def mcc_from_counts(tn, fp, fn, tp):
    """Matthews correlation coefficient (0 when a row or column of the confusion matrix is empty)."""
    tn, fp, fn, tp = (np.asarray(v, dtype=np.float64) for v in (tn, fp, fn, tp))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_div(tp * tn - fp * fn, denominator)


def precision_recall_f1(tn, fp, fn, tp):
    """Precision, recall and F1 averaged three ways, as a dict of dicts.

    macro    = plain mean of the two classes,
    weighted = mean weighted by the number of true reviews in each class,
    micro    = pooled counts. For single-label binary classification micro precision = micro recall
               = micro F1 = accuracy, so the three micro numbers are always identical.
    """
    tn, fp, fn, tp = (np.asarray(v, dtype=np.float64) for v in (tn, fp, fn, tp))
    p1, r1 = safe_div(tp, tp + fp), safe_div(tp, tp + fn)  # positive class
    p0, r0 = safe_div(tn, tn + fn), safe_div(tn, tn + fp)  # negative class
    f1_pos, f0_neg = safe_div(2 * p1 * r1, p1 + r1), safe_div(2 * p0 * r0, p0 + r0)
    n1, n0 = tp + fn, tn + fp
    weight1, weight0 = safe_div(n1, n0 + n1), safe_div(n0, n0 + n1)
    accuracy = accuracy_from_counts(tn, fp, fn, tp)
    return {
        "macro": {"precision": (p0 + p1) / 2, "recall": (r0 + r1) / 2, "f1": (f0_neg + f1_pos) / 2},
        "micro": {"precision": accuracy, "recall": accuracy, "f1": accuracy},
        "weighted": {"precision": weight0 * p0 + weight1 * p1, "recall": weight0 * r0 + weight1 * r1,
                     "f1": weight0 * f0_neg + weight1 * f1_pos},
    }


def macro_f1_from_counts(tn, fp, fn, tp):
    return precision_recall_f1(tn, fp, fn, tp)["macro"]["f1"]


def brier_score(y_true, prob):
    """Mean squared difference between the positive-class probability and the 0/1 label."""
    return float(np.mean((np.asarray(prob, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2))


def reliability_bins(y_true, prob, n_bins=15):
    """Per-bin (count, mean predicted probability, fraction of real positives).

    15 equal-width bins on the positive-class probability. Bins are right-closed, except the first
    which also contains 0: [0, 1/15], (1/15, 2/15], ..., (14/15, 1].
    """
    y_true, prob = np.asarray(y_true, dtype=np.float64), np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.searchsorted(edges, prob, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins)
    sum_prob = np.bincount(bin_index, weights=prob, minlength=n_bins)
    sum_true = np.bincount(bin_index, weights=y_true, minlength=n_bins)
    return counts, safe_div(sum_prob, counts), safe_div(sum_true, counts)


def expected_calibration_error(y_true, prob, n_bins=15):
    """ECE = sum over bins of (bin size / n) * |fraction of positives - mean predicted probability|."""
    counts, mean_prob, frac_positive = reliability_bins(y_true, prob, n_bins)
    return float(np.sum(counts / counts.sum() * np.abs(frac_positive - mean_prob)))


def roc_auc(y_true, prob):
    return float(roc_auc_score(y_true, prob))


def pr_auc(y_true, prob):
    """Area under the precision-recall curve as average precision."""
    return float(average_precision_score(y_true, prob))


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    """95% percentile bootstrap intervals for accuracy, macro-F1 and MCC.

    Resampling n (true, pred) pairs with replacement gives confusion-matrix counts that follow a
    multinomial distribution with the observed cell frequencies, so the counts are drawn directly
    (same distribution, much faster than re-sampling 38,000 rows 1000 times).
    Returns {metric: (low, high)}.
    """
    counts = np.array(confusion_counts(y_true, y_pred), dtype=np.float64)
    n = int(counts.sum())
    rng = np.random.default_rng(seed)
    boot = rng.multinomial(n, counts / n, size=n_boot).T  # rows: tn, fp, fn, tp
    tn, fp, fn, tp = boot
    samples = {"accuracy": accuracy_from_counts(tn, fp, fn, tp), "macro_f1": macro_f1_from_counts(tn, fp, fn, tp),
               "mcc": mcc_from_counts(tn, fp, fn, tp)}
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {name: (float(np.percentile(v, low)), float(np.percentile(v, high))) for name, v in samples.items()}


def mcnemar(correct_a, correct_b):
    """Paired McNemar test on the same reviews.

    b = A right and B wrong, c = A wrong and B right. Exact two-sided binomial p-value on min(b, c)
    out of b + c (p = 1 when b + c = 0), and the continuity-corrected chi-square statistic
    (|b - c| - 1)^2 / (b + c) with its p-value (1 degree of freedom).
    """
    correct_a, correct_b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_exact": 1.0, "chi2": 0.0, "p_chi2": 1.0}
    p_exact = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    statistic = max(abs(b - c) - 1, 0) ** 2 / n
    return {"b": b, "c": c, "p_exact": p_exact, "chi2": float(statistic), "p_chi2": float(chi2.sf(statistic, 1))}


def paired_bootstrap_accuracy_difference(correct_a, correct_b, n_boot=1000, seed=42):
    """95% percentile CI of accuracy(A) - accuracy(B) on the same reviews.

    Each review is one of four kinds: both right, only A right, only B right, both wrong.
    A resample of n reviews has multinomial counts of these kinds; the difference is
    (only A - only B) / n.
    """
    correct_a, correct_b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    n = len(correct_a)
    kinds = np.array([(correct_a & correct_b).sum(), (correct_a & ~correct_b).sum(),
                      (~correct_a & correct_b).sum(), (~correct_a & ~correct_b).sum()], dtype=np.float64)
    boot = np.random.default_rng(seed).multinomial(n, kinds / n, size=n_boot)
    diff = (boot[:, 1] - boot[:, 2]) / n
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def core_metrics(y_true, y_pred, prob):
    """Everything computed directly from labels, predictions and probabilities (no intervals or tests)."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    prf = precision_recall_f1(tn, fp, fn, tp)
    result = {"accuracy": float(accuracy_from_counts(tn, fp, fn, tp)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
              "roc_auc": roc_auc(y_true, prob), "pr_auc": pr_auc(y_true, prob),
              "mcc": float(mcc_from_counts(tn, fp, fn, tp)), "brier": brier_score(y_true, prob),
              "ece": expected_calibration_error(y_true, prob)}
    for average in ("macro", "micro", "weighted"):
        for name in ("precision", "recall", "f1"):
            result[f"{name}_{average}"] = float(prf[average][name])
    return result
