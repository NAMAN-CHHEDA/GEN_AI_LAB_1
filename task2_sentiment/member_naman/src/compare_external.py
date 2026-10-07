"""Compare one of our models with predictions made elsewhere (for example a teammate's model).

Run:  python src/compare_external.py --config gpu --model bigru --other <csv> --other-name teammate_lstm

The other csv needs the columns true_label, predicted_label, positive_probability for the same
test reviews in the original Hugging Face order. The csv path is only read from the command line
and is never written into the output.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
from predict import MODEL_NAMES
from utils import load_config, save_json

REQUIRED_OTHER_COLUMNS = ["true_label", "predicted_label", "positive_probability"]


def compare_predictions(ours, other, our_name, other_name):
    """Paired comparison of two prediction tables on the same reviews. Raises ValueError on a mismatch."""
    missing = [c for c in REQUIRED_OTHER_COLUMNS if c not in other.columns]
    if missing:
        raise ValueError(f"the other csv is missing columns {missing}")
    if len(other) != len(ours):
        raise ValueError(f"the other csv has {len(other)} rows but our test set has {len(ours)}. "
                         "Both must cover the same test reviews (the local config uses a 2,000-review subset: "
                         "use --config gpu for the full 38,000)")
    if not (other["true_label"].to_numpy() == ours["true_label"].to_numpy()).all():
        raise ValueError("true labels differ, so the rows are not the same reviews in the same order")

    y = ours["true_label"].to_numpy()
    correct_ours = ours["predicted_label"].to_numpy() == y
    correct_other = other["predicted_label"].to_numpy() == y
    test = metrics.mcnemar(correct_ours, correct_other)
    low, high = metrics.paired_bootstrap_accuracy_difference(correct_ours, correct_other)
    return {
        "our_model": our_name, "other_name": other_name, "n_test": int(len(y)),
        "accuracy_ours": float(correct_ours.mean()), "accuracy_other": float(correct_other.mean()),
        "accuracy_difference_ours_minus_other": float(correct_ours.mean() - correct_other.mean()),
        "accuracy_difference_ci95": [low, high],
        "mcnemar_b_ours_right_other_wrong": test["b"], "mcnemar_c_ours_wrong_other_right": test["c"],
        "mcnemar_p_exact": test["p_exact"],
        "bootstrap": {"resamples": 1000, "seed": 42, "method": "percentile, paired"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare one of our models with external predictions")
    parser.add_argument("--config", default="gpu")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--other", required=True, help="csv with true_label, predicted_label, positive_probability")
    parser.add_argument("--other-name", required=True, help="short label, used in the output file name")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ours_path = cfg["paths"]["output_dir"] / args.model / "test_predictions.csv"
    if not ours_path.exists():
        raise SystemExit(f"No predictions for '{args.model}' in run '{cfg['run_name']}': run predict.py first")
    try:
        result = compare_predictions(pd.read_csv(ours_path), pd.read_csv(args.other), args.model, args.other_name)
    except ValueError as error:
        raise SystemExit(f"Cannot compare: {error}")

    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", args.other_name)
    save_json(result, cfg["paths"]["output_dir"] / f"vs_{safe_name}.json")
    print(f"{args.model} accuracy {result['accuracy_ours']:.4f} vs {args.other_name} {result['accuracy_other']:.4f}; "
          f"difference {result['accuracy_difference_ours_minus_other']:+.4f}, 95% CI "
          f"[{result['accuracy_difference_ci95'][0]:+.4f}, {result['accuracy_difference_ci95'][1]:+.4f}]; "
          f"McNemar b={result['mcnemar_b_ours_right_other_wrong']}, c={result['mcnemar_c_ours_wrong_other_right']}, "
          f"exact p={result['mcnemar_p_exact']:.4g}")


if __name__ == "__main__":
    main()
