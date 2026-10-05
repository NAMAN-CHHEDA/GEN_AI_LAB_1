"""Write the reproducibility manifest and a pip freeze for one run.

Run:  python src/make_manifest.py --config local
Outputs: reproducibility/manifests/naman_task2_<run>_manifest.json and ..._requirements_freeze.txt.
Files in reproducibility/raw_logs are only referenced, never modified.
"""

import argparse
import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import MODEL_NAMES
from train import log_file_path
from utils import get_device, load_config, load_json, project_root, save_json

KEY_PACKAGES = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "datasets", "nltk", "matplotlib",
                "tqdm", "psutil", "nbformat", "nbclient", "nbconvert", "ipykernel"]
MISSING = "missing"


def sha256_of(path):
    """Hex sha256 of a file (read in chunks, so big checkpoints are fine)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root():
    return project_root().parent.parent


def relative(path):
    """Path relative to the repository root, with forward slashes (no personal paths in the manifest)."""
    return Path(path).relative_to(repo_root()).as_posix()


def git_commit():
    """(commit hash or None, note)."""
    if shutil.which("git") is None:
        return None, "git is not installed on this machine"
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True)
    if done.returncode != 0:
        return None, "not a git repository (or no commit yet)"
    return done.stdout.strip(), ""


def package_versions():
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def pip_freeze():
    """pip freeze output, with local-path installs reduced to the package name (no paths in the file)."""
    done = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    lines = []
    for line in done.stdout.splitlines():
        lines.append(line.split(" @ ")[0] + "  # installed from a local path (path removed)" if " @ " in line else line)
    return "\n".join(lines) + "\n"


def artifact(path):
    """Relative path of an existing file, or the word 'missing'."""
    return relative(path) if Path(path).exists() else MISSING


def model_entry(cfg, name, report):
    """Everything that connects one checkpoint to its logs and results."""
    ckpt = cfg["paths"]["checkpoint_dir"] / name / "best.pt"
    out_dir = cfg["paths"]["output_dir"] / name
    summary_path = out_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else None
    in_report = report is not None and name in set(report["model"])
    return {
        "checkpoint": artifact(ckpt),
        "checkpoint_sha256": sha256_of(ckpt) if ckpt.exists() else MISSING,
        "best_epoch": summary["best_epoch"] if summary else MISSING,
        "best_val_loss": summary["best_val_loss"] if summary else MISSING,
        "train_log": artifact(log_file_path(cfg, name, out_dir)),
        "summary_json": artifact(summary_path),
        "predictions_csv": artifact(out_dir / "test_predictions.csv"),
        "metrics_report_row": name if in_report else MISSING,
    }


def build_manifest(cfg, config_name):
    out_dir = cfg["paths"]["output_dir"]
    config_file = project_root() / "configs" / f"{config_name}.yaml"
    info_path = cfg["paths"]["processed_dir"] / "split_info.json"
    info = load_json(info_path) if info_path.exists() else None
    report_path = out_dir / "metrics_report.csv"
    report = pd.read_csv(report_path) if report_path.exists() else None
    commit, note = git_commit()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "run_name": cfg["run_name"], "seed": cfg["seed"],
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": package_versions(), "os": platform.platform(),
        "cpu": get_device("cpu")[1], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit, "git_note": note,
        "config_sha256": {f"configs/{config_name}.yaml": sha256_of(config_file)},
        "data_cache_key": info["cache_key"] if info else MISSING,
        "dataset": cfg["data"]["dataset"], "dataset_sizes": info["counts"] if info else MISSING,
        "checkpoint_to_result": {name: model_entry(cfg, name, report) for name in MODEL_NAMES},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the reproducibility manifest")
    parser.add_argument("--config", default="local")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    folder = repo_root() / "reproducibility" / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / f"naman_task2_{cfg['run_name']}_manifest.json"
    freeze_path = folder / f"naman_task2_{cfg['run_name']}_requirements_freeze.txt"
    save_json(build_manifest(cfg, args.config), manifest_path)
    freeze_path.write_text(pip_freeze(), encoding="utf-8")
    print(f"wrote {relative(manifest_path)} and {relative(freeze_path)}")


if __name__ == "__main__":
    main()
