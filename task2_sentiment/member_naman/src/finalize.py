"""One command to run AFTER the full training has finished: predictions, evaluation, reports, manifest, checks.

Run:  python src/finalize.py --config gpu [--other PATH --other-name LABEL] [--device auto|cpu|cuda]
(--other / --other-name can be repeated, once per external prediction file.)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import MODEL_NAMES
from run_steps import print_table, run_script
from utils import load_config

MANUAL_ITEMS = [
    "Fill the 'Error type' and 'Testable fix' columns of failure_analysis.md",
    'Write the "TODO (my words)" sections of results.md',
    "Commit and push (reproducibility/raw_logs, manifests, metrics_report.csv, failure_analysis.md, results.md)",
]


def missing_checkpoints(cfg):
    """Names of the models whose best.pt does not exist yet."""
    return [m for m in MODEL_NAMES if not (cfg["paths"]["checkpoint_dir"] / m / "best.pt").exists()]


def best_model_by_test_accuracy(cfg):
    report = pd.read_csv(cfg["paths"]["output_dir"] / "metrics_report.csv")
    return report.loc[report["accuracy"].idxmax(), "model"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finish the project after the full training")
    parser.add_argument("--config", default="gpu")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--other", action="append", default=[], help="csv with a teammate's predictions (repeatable)")
    parser.add_argument("--other-name", action="append", default=[], help="label for each --other (repeatable)")
    args = parser.parse_args(argv)
    if len(args.other) != len(args.other_name):
        raise SystemExit("--other and --other-name must be given the same number of times")

    cfg = load_config(args.config)
    missing = missing_checkpoints(cfg)
    if missing:
        print(f"Cannot finalize run '{cfg['run_name']}': best.pt is missing for {', '.join(missing)}.")
        print("Wait for the training to finish (run_full_training.ps1) and try again.")
        sys.exit(1)

    config = ["--config", args.config]
    results = [run_script("predict (test split, all models)", "predict.py", *config, "--model", "all", "--device", args.device),
               run_script("evaluate", "evaluate.py", *config)]
    if all(r["ok"] for r in results):
        best = best_model_by_test_accuracy(cfg)
        for path, label in zip(args.other, args.other_name):
            results.append(run_script(f"compare with {label} (model {best})", "compare_external.py", *config,
                                      "--model", best, "--other", path, "--other-name", label))
        steps = [("error review", "error_review.py", config), ("results.md", "make_results_md.py", config),
                 ("build notebook", "build_notebook.py", []), ("run notebook", "run_notebook.py", config),
                 ("manifest", "make_manifest.py", config), ("repo check", "check_repo.py", [])]
        for label, script, script_args in steps:
            results.append(run_script(label, script, *script_args))
            if not results[-1]["ok"]:
                break
    print_table(results)
    print("\nStill for you to do by hand:")
    for item in MANUAL_ITEMS:
        print(f"  [ ] {item}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
