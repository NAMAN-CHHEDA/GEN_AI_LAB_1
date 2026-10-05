"""The reproducibility command: runs the whole pipeline on the tiny local config, on CPU, in a few minutes.

Run from the repository root:   python task2_sentiment/member_naman/src/smoke_test.py --device cpu --fresh
It never writes the full-run deliverables (metrics_report.csv, failure_analysis.md, results.md in member_naman,
outputs/gpu, checkpoints/gpu, reproducibility/raw_logs).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data import cache_is_valid
from evaluate import REQUIRED_COLUMNS
from predict import MODEL_NAMES
from run_steps import SRC_DIR, TESTS_DIR, print_table, run_script
from utils import load_config, project_root

PROTECTED_FILES = ["metrics_report.csv", "failure_analysis.md", "results.md"]  # in member_naman: gpu deliverables
EVAL_PLOTS = ["roc_curves", "pr_curves", "reliability_diagrams", "macro_f1_bar", "slice_error_rates",
              "loss_curves_overlay"] + [f"confusion_matrix_{m}" for m in MODEL_NAMES]


def snapshot_protected():
    """(exists, modification time) of the full-run deliverables, to prove the smoke test did not touch them."""
    state = {}
    for name in PROTECTED_FILES:
        path = project_root() / name
        state[name] = (path.exists(), path.stat().st_mtime if path.exists() else None)
    return state


def deliverable_problems(cfg, skip_notebook):
    """Things that are missing or empty for the local run (empty list = all good)."""
    out, root = cfg["paths"]["output_dir"], project_root()
    required = [out / "metrics_report.csv", out / "metrics_table.md", out / "error_review" / "failure_analysis.md",
                out / "error_review" / "candidates.csv", out / "results.md",
                root.parent.parent / "reproducibility" / "manifests" / f"naman_task2_{cfg['run_name']}_manifest.json"]
    required += [out / m / "test_predictions.csv" for m in MODEL_NAMES]
    required += [out / "eval" / f"{p}.png" for p in EVAL_PLOTS]
    if not skip_notebook:
        required.append(SRC_DIR / "task2_sentiment.ipynb")
    problems = [f"missing or empty: {p.relative_to(root.parent.parent).as_posix()}"
                for p in required if not p.exists() or p.stat().st_size == 0]
    report_path = out / "metrics_report.csv"
    if report_path.exists():
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in pd.read_csv(report_path).columns]
        if missing_columns:
            problems.append(f"metrics_report.csv lacks columns: {missing_columns}")
    return problems


def gpu_warnings():
    """Warnings (not failures) about the full-run artifacts."""
    gpu = load_config("gpu")
    notes = []
    notes.append("gpu data cache: " + ("present" if cache_is_valid(gpu) else "MISSING (run python src/data.py --config gpu)"))
    for m in MODEL_NAMES:
        best = gpu["paths"]["checkpoint_dir"] / m / "best.pt"
        notes.append(f"gpu checkpoint {m}: " + (f"present ({best.stat().st_size / 2**20:.0f} MB)" if best.exists() else "not there yet"))
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description="End-to-end smoke test on the local config")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="rebuild the local processed data from scratch")
    args = parser.parse_args(argv)

    cfg = load_config("local")
    protected_before = snapshot_protected()
    local, device = ["--config", "local"], ["--device", args.device]
    force_data = ["--force"] if args.fresh or not cache_is_valid(cfg) else []

    steps = [("1 check_setup", "check_setup.py", [])]
    # tests that need no trained local run come first; the others run once the local run exists
    early = ["test_data.py", "test_models.py", "test_repo_tools.py"]
    late = [p.name for p in sorted(TESTS_DIR.glob("test_*.py")) if p.name not in early]
    steps += [(f"2 unit tests: {name[:-3]}", name, []) for name in early]
    steps += [("3 check_repo", "check_repo.py", []), ("4 data build (local)", "data.py", local + force_data),
              ("5 eda (local)", "eda.py", local)]
    steps += [(f"6 train {m}", "train.py", local + ["--model", m, "--force"] + device) for m in MODEL_NAMES]
    steps += [("7 predict (all models)", "predict.py", local + ["--model", "all"] + device),
              ("7 evaluate", "evaluate.py", local),
              ("8 error_review", "error_review.py", local), ("8 make_results_md", "make_results_md.py", local)]
    if not args.skip_notebook:
        steps += [("9 build_notebook", "build_notebook.py", []), ("9 run_notebook", "run_notebook.py", local)]
    steps += [("10 make_manifest", "make_manifest.py", local)]
    steps += [(f"2b unit tests: {name[:-3]}", name, []) for name in late]
    steps += [("3b check_repo (with outputs)", "check_repo.py", [])]

    results, failed_at = [], None
    for number, (label, script, script_args) in enumerate(steps, start=1):
        results.append(run_script(label, script, *script_args))
        if not results[-1]["ok"]:
            failed_at = number
            break

    if failed_at is None:  # step 11: deliverables, and proof that the full-run files were left alone
        start = time.time()
        problems = deliverable_problems(cfg, args.skip_notebook)
        if snapshot_protected() != protected_before:
            problems.append("a full-run deliverable in member_naman was changed by the smoke test")
        for problem in problems:
            print(f"    {problem}")
        results.append({"label": "11 deliverable check", "seconds": time.time() - start, "ok": not problems})
        if problems:
            failed_at = len(results)

    print_table(results)
    print("\nFull-run artifacts (warnings only):")
    for note in gpu_warnings():
        print(f"  {note}")
    if failed_at is None:
        print("\nSMOKE TEST PASSED")
    else:
        print(f"\nSMOKE TEST FAILED at step {failed_at}")
    sys.exit(0 if failed_at is None else 1)


if __name__ == "__main__":
    main()
