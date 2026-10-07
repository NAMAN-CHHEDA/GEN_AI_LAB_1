"""CycleGAN training: Monet (A) <-> Photo (B), unpaired, one random image per domain per step.

Usage (from the naman/ folder):
    python train.py --config configs/run02.yaml             start a new run (refuses if its folders already have files)
    python train.py --config configs/run02.yaml --resume   continue from <checkpoint_dir>/last.pt
    python train.py --config configs/run01.yaml --smoke    tiny test run (64px, 20 images per domain, 300 steps)
Ctrl+C once = finish the current step, save last.pt + ema_step<N>.pt, print the resume command and exit 0.
Ctrl+C twice = exit immediately.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # grader_eval.py / evaluate_local.py live next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from config import load_config, require, resolve  # noqa: E402
from data import (ImageFolderDataset, build_manifest, build_transform, list_images,  # noqa: E402
                  make_dataloaders)
from logging_utils import AppendOnlyLogger, peak_memory_mb, seed_everything  # noqa: E402
from losses_aug import cycle_l1, diff_augment, identity_l1, lsgan_loss, update_ema  # noqa: E402
from manifest import write_run_manifest  # noqa: E402
from models import ImagePool, PatchDiscriminator, ResnetGenerator, count_params, init_weights  # noqa: E402

LOG_FIELDS = ["step", "g_adv", "cycle", "identity", "d_a", "d_b", "grad_g", "grad_d_a",
              "grad_d_b", "lr", "nan_count", "images_per_sec", "seconds"]
FID_FIELDS = ["step", "fid_a2b", "fid_b2a", "fid_avg", "mifid_avg", "seconds"]
SMOKE_IMAGES = 20

# Config keys that may be changed when resuming (everything else, and every model.* key, must stay as it was).
ALLOWED_RESUME_CHANGES = {"train.lambda_cycle", "train.lambda_identity", "train.diffaug", "train.lr", "train.betas",
                          "train.decay_start", "train.total_steps", "train.fid_every", "train.ckpt_every",
                          "train.log_every", "data.num_workers"}
CFG_DEFAULTS = {"model.upsample": "convtranspose", "train.fid_every": 0}  # keys older configs (run01) do not have


# ---- small adapters so extra state rides inside checkpoint.py's save/load (models/optimizers dicts)

class PoolState:
    """Exposes an ImagePool through state_dict / load_state_dict."""

    def __init__(self, pool, device):
        self.pool, self.device = pool, device

    def state_dict(self):
        return {"images": [im.cpu() for im in self.pool.images]}

    def load_state_dict(self, s):
        self.pool.images = [im.to(self.device) for im in s["images"]]


class MetaState:
    """Run bookkeeping that must survive a resume: history rows, nan_count, elapsed seconds."""

    def __init__(self):
        self.data = {"history": [], "nan_count": 0, "elapsed": 0.0}

    def state_dict(self):
        return self.data

    def load_state_dict(self, s):
        self.data = s


def apply_smoke(cfg):
    cfg["run_name"] += "_smoke"
    cfg["data"].update(image_size=64, load_size=72, num_workers=0)
    cfg["train"].update(total_steps=300, decay_start=150, ckpt_every=100, log_every=20)
    if cfg["train"].get("fid_every"):  # exercise the in-training grader FID once, at the end of the smoke run
        cfg["train"]["fid_every"] = 300


BENCH_WARMUP = 10  # steps excluded from the --bench speed measurement


def apply_bench(cfg):
    """Speed benchmark: real config (256px, full data, config num_workers) but only 100 steps."""
    cfg["run_name"] += "_bench"
    cfg["train"].update(total_steps=100, ckpt_every=100, log_every=10)  # one checkpoint, at the end
    if cfg["train"].get("fid_every"):
        cfg["train"]["fid_every"] = 0  # a speed test must not include scoring time


def infinite(loader):
    """Endless stream of batches; each domain reshuffles independently."""
    while True:
        yield from loader


def grad_norm(params):
    """Total L2 grad norm (no clipping)."""
    return torch.nn.utils.clip_grad_norm_(params, float("inf"))


def set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


def finite(*tensors):
    return bool(torch.isfinite(torch.stack([t.detach().float() for t in tensors])).all())


def lr_lambda(decay_start, total):
    def f(step):
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(1, total - decay_start))
    return f


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---- one run per output folder: never overwrite or resume another run's files

RUN_MARKER = "run_name.txt"  # written into a fresh checkpoint / log folder


def run_names_in(folder):
    """(has_files, run names) for the files directly inside `folder`.

    Evidence: run_name.txt (written by this script), run_summary.json, or the config stored in an
    ema_step*.pt checkpoint. A folder with files but no evidence returns an empty set (unknown origin).
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False, set()
    files = [p for p in folder.iterdir() if p.is_file() and p.name != ".gitkeep"]
    if not files:
        return False, set()
    names = set()
    if (folder / RUN_MARKER).exists():
        names.add((folder / RUN_MARKER).read_text(encoding="utf-8").strip())
    if (folder / "run_summary.json").exists():
        with open(folder / "run_summary.json", encoding="utf-8") as f:
            names.add(json.load(f).get("run_name"))
    ema = latest_ema(folder)
    if ema is not None:
        names.add(torch.load(ema, map_location="cpu", weights_only=True)["config"]["run_name"])
    return True, names - {None}


def latest_ema(folder):
    """Path of the highest-step ema_step<N>.pt in `folder`, or None."""
    found = [(int(p.stem.replace("ema_step", "")), p) for p in Path(folder).glob("ema_step*.pt")]
    return max(found)[1] if found else None


def guard_run_dirs(run_name, folders):
    """Exit if a target folder already holds files from a different (or unidentifiable) run.
    Returns {folder: has_files} so the caller can also refuse to start over existing files."""
    existing = {}
    for folder in folders:
        has_files, names = run_names_in(folder)
        if has_files and names != {run_name}:
            found = ", ".join(sorted(names)) if names else "an unknown run (no run_name.txt, run_summary.json or ema checkpoint)"
            sys.exit(f"ERROR: {folder} already contains files from {found}, but this run is '{run_name}'. "
                     "Refusing to start so nothing is overwritten or resumed from another run. "
                     "Point checkpoint_dir / log_dir in the config at new folders.")
        existing[Path(folder)] = has_files
    return existing


def write_run_marker(folder, run_name):
    """Tag a fresh folder with its run name. Folders that already have files (older runs) are left untouched."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not run_names_in(folder)[0]:
        (folder / RUN_MARKER).write_text(run_name + "\n", encoding="utf-8")


# ---- resume: compare the current config with the saved one, record what changed

def flatten_cfg(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_cfg(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def diff_configs(old_cfg, new_cfg):
    """{key: (old, new)} for every flattened key whose value differs (missing keys get CFG_DEFAULTS)."""
    a, b = flatten_cfg(old_cfg), flatten_cfg(new_cfg)
    for k, v in CFG_DEFAULTS.items():
        a.setdefault(k, v)
        b.setdefault(k, v)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def split_resume_changes(changes):
    """(model_changes, other_disallowed_changes): both must be empty to resume."""
    model = {k: v for k, v in changes.items() if k.startswith("model.")}
    other = {k: v for k, v in changes.items() if not k.startswith("model.") and k not in ALLOWED_RESUME_CHANGES}
    return model, other


def saved_run_config(manifest_dir, run_name, ckpt_dir):
    """(config, where) the run was started with: the run manifest, else the config stored in the newest EMA file."""
    p = Path(manifest_dir) / f"run_{run_name}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)["config"], f"run manifest {p.name}"
    ema = latest_ema(ckpt_dir)
    if ema is not None:
        return torch.load(ema, map_location="cpu", weights_only=True)["config"], f"config stored in {ema.name}"
    return None, None


def code_fingerprint(config_path):
    """sha256 of the code + config files, and the git commit if git is available."""
    root = Path(__file__).resolve().parent
    names = ["train.py", "grader_eval.py", "gen_preds.py", "evaluate_local.py", "src/models.py", "src/losses_aug.py",
             "src/data.py", "src/manifest.py", "src/checkpoint.py", "src/config.py", "src/logging_utils.py"]
    files = {n: root / n for n in names}
    files[f"config:{Path(config_path).name}"] = Path(config_path)
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items() if p.is_file()}
    git = {"commit": None, "uncommitted_changes_here": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=15)
        if head.returncode == 0:
            git["commit"] = head.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=root, capture_output=True, text=True, timeout=15)
            git["uncommitted_changes_here"] = bool(st.stdout.strip())
    except Exception:  # git missing or not a repository: the file hashes still identify the code
        pass
    return {"git": git, "file_sha256": hashes}


def write_resume_manifest(cfg, config_path, manifest_dir, step, changes, source):
    out = Path(manifest_dir) / f"run_{cfg['run_name']}_resume_step{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"run_name": cfg["run_name"], "resumed_at_step": step, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "compared_with": source, "changed_keys": {k: {"old": o, "new": n} for k, (o, n) in changes.items()},
           "config": cfg, "code": code_fingerprint(config_path)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return out


def apply_current_hparams(pairs, lr, betas, lam, step):
    """After loading a checkpoint, make the optimizers and schedulers follow the CURRENT config (lr, betas and the
    LR curve are rebuilt from it); a scheduler's lambda is already the current one because it was built from it."""
    for opt, sch in pairs:
        sch.base_lrs = [lr] * len(opt.param_groups)
        for g in opt.param_groups:
            g["betas"], g["initial_lr"], g["lr"] = tuple(betas), lr, lr * lam(step)


# ---- saving, sample grids

def save_state(step, last_path, ckpt_dir, models, optimizers, cfg, backup_dir, ema_ab, ema_ba, meta, elapsed):
    """last.pt (everything needed to resume) and ema_step<step>.pt (EMA generators only); both written atomically."""
    meta.data["elapsed"] = elapsed
    save_checkpoint(last_path, models, optimizers, step, cfg, backup_dir)  # tmp file, then replace
    atomic_save({"step": step, "G_AB": ema_ab.state_dict(), "G_BA": ema_ba.state_dict(), "config": cfg},
                Path(ckpt_dir) / f"ema_step{step}.pt")


@torch.no_grad()
def save_samples(g_ab, g_ba, fixed_a, fixed_b, path):
    """Columns: photo, fake Monet, Monet, fake photo; one row per fixed image."""
    fake_a, fake_b = g_ba(fixed_b), g_ab(fixed_a)
    grid = torch.cat([fixed_b, fake_a, fixed_a, fake_b], dim=3)  # concat along width
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path, nrow=1, normalize=True, value_range=(-1, 1))


# ---- in-training grader FID (off unless train.fid_every > 0)

def emit(line, raw_log):
    """Console + raw log, flushed."""
    print(line, flush=True)
    with open(raw_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def rng_snapshot():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def rng_restore(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


class BestFid:
    """Best average grader FID so far (rebuilt from grader_fid.csv on resume)."""

    def __init__(self, csv_path=None):
        self.step, self.value = None, float("inf")
        if csv_path is not None and Path(csv_path).exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if float(row["fid_avg"]) < self.value:
                        self.step, self.value = int(row["step"]), float(row["fid_avg"])


def grader_fid_hook(step, ema_ab, ema_ba, cfg, device, raw_log, best, fid_logger, best_json, tmp_dir, n_images=None):
    """Score the EMA generators the way the grader does and log it. Never raises, never changes training state:
    weights are untouched (no_grad, no optimizer), the EMA train/eval flags are restored, and ALL random-number
    generators are restored (building a DataLoader iterator draws from the global torch RNG)."""
    t0 = time.perf_counter()
    rng = rng_snapshot()
    was_training = (ema_ab.training, ema_ba.training)
    tmp_dir = Path(tmp_dir)
    try:
        from evaluate_local import translate  # the image writer gen_preds.py uses (256x256 RGB JPG q95, original names)
        import grader_eval as ge              # grader functions, imported unchanged (its scipy shim applies on import)
        n = ge.N_EVAL if n_images is None else n_images
        folder_a, folder_b = resolve(cfg["paths"]["monet_dir"]), resolve(cfg["paths"]["photo_dir"])
        files_a, files_b = list_images(folder_a)[:n], list_images(folder_b)[:n]  # first n sorted files
        tf = build_transform(cfg["data"]["image_size"], [], train=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        ema_ab.eval()
        ema_ba.eval()
        translate(ema_ab, ema_ba, folder_a, files_a, tmp_dir / "pred_A2B", tf, device)
        translate(ema_ba, ema_ab, folder_b, files_b, tmp_dir / "pred_B2A", tf, device)
        # exactly the grader's flow (Part3_Evaluation_Script.ipynb): B2A = real Monet vs pred_B2A, A2B = real photo vs pred_A2B
        real_monet = ge.take_n(ge.list_images(str(folder_a)), n)
        real_photo = ge.take_n(ge.list_images(str(folder_b)), n)
        gen_a2b = ge.take_n(ge.list_images(str(tmp_dir / "pred_A2B")), n)
        gen_b2a = ge.take_n(ge.list_images(str(tmp_dir / "pred_B2A")), n)
        fid_b2a, mifid_b2a = ge.calculate_fid_mifid(real_monet, gen_b2a, batch_size=ge.BATCH_SIZE)
        fid_a2b, mifid_a2b = ge.calculate_fid_mifid(real_photo, gen_a2b, batch_size=ge.BATCH_SIZE)
        fid_avg, mifid_avg = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
        seconds = time.perf_counter() - t0
        new_best = fid_avg < best.value
        if new_best:
            best.step, best.value = step, fid_avg
            tmp = Path(str(best_json) + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"best_step": step, "best_fid_avg": fid_avg, "mifid_avg": mifid_avg,
                           "fid_a2b": fid_a2b, "fid_b2a": fid_b2a}, f, indent=2)
            os.replace(tmp, best_json)
        tail = f"best so far: step {best.step} avg {best.value:.2f}" + (" (NEW BEST)" if new_best else "")
        emit(f"[GRADER FID] step {step} | A2B {fid_a2b:.2f} | B2A {fid_b2a:.2f} | avg {fid_avg:.2f} | "
             f"MiFID {mifid_avg:.4f} | {tail}", raw_log)
        fid_logger.log({"step": step, "fid_a2b": fid_a2b, "fid_b2a": fid_b2a, "fid_avg": fid_avg,
                        "mifid_avg": mifid_avg, "seconds": seconds})
    except Exception as e:  # scoring must never stop training
        emit(f"[GRADER FID] failed at step {step}: {type(e).__name__}: {e}", raw_log)
    finally:
        ema_ab.train(was_training[0])
        ema_ba.train(was_training[1])
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rng_restore(rng)


# ---- graceful stop

class StopFlag:
    requested = False


def install_sigint(stop):
    """First Ctrl+C only sets a flag (the loop saves and exits at the top of the next step); the second exits at once."""
    def handler(signum, frame):
        if stop.requested:
            print("\nSecond Ctrl+C: exiting immediately, nothing more is saved.", flush=True)
            os._exit(130)
        stop.requested = True
        print("\nCtrl+C received: finishing the current step, then saving a checkpoint and stopping "
              "(press Ctrl+C again to quit immediately).", flush=True)
    signal.signal(signal.SIGINT, handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run01.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/last.pt (never done implicitly)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 64px, 20 imgs/domain, 300 steps")
    ap.add_argument("--bench", action="store_true", help="speed test: real config, 100 steps, output under bench/ folders")
    args = ap.parse_args()
    if args.smoke and args.bench:
        ap.error("--smoke and --bench are mutually exclusive")
    if args.bench and args.resume:
        ap.error("--bench always starts from step 0; it cannot be combined with --resume")
    mode = "smoke" if args.smoke else "bench" if args.bench else None

    cfg = load_config(args.config)
    if args.smoke:
        apply_smoke(cfg)
    if args.bench:
        apply_bench(cfg)
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    total, decay_start = require(t, "total_steps"), require(t, "decay_start")
    log_every, ckpt_every = require(t, "log_every"), require(t, "ckpt_every")
    lam_cyc, lam_id = require(t, "lambda_cycle"), require(t, "lambda_identity")
    policy, ema_decay = t["diffaug"], require(t, "ema_decay")
    fid_every = int(t.get("fid_every", 0) or 0)  # 0 / missing = in-training grader FID off
    if fid_every and fid_every % ckpt_every:
        sys.exit(f"ERROR: train.fid_every ({fid_every}) must be a multiple of train.ckpt_every ({ckpt_every}): "
                 "the grader FID runs right after a checkpoint is saved.")
    resume_cmd = f".venv\\Scripts\\python train.py --config {args.config} --resume" + (" --smoke" if args.smoke else "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    seed_everything(require(cfg, "seed"))

    # smoke / bench output never mixes with the real run
    ckpt_dir, log_dir = resolve(cfg["paths"]["checkpoint_dir"]), resolve(cfg["paths"]["log_dir"])
    sample_dir = resolve(cfg["paths"]["sample_dir"])
    if mode:
        ckpt_dir, log_dir, sample_dir = ckpt_dir / mode, log_dir / mode, sample_dir.parent / mode
    backup_dir = cfg["paths"].get("backup_dir")
    backup_dir = resolve(backup_dir) if backup_dir else None
    manifest_dir = resolve(cfg["paths"]["manifest_dir"])
    last_path = ckpt_dir / "last.pt"

    # ---- explicit start / resume: never resume implicitly, never touch another run's files
    existing = guard_run_dirs(cfg["run_name"], [ckpt_dir, log_dir])  # before anything is written
    if args.resume:
        if not last_path.exists():
            sys.exit(f"ERROR: --resume was given but {last_path} does not exist, so there is nothing to resume. "
                     "Start the run without --resume (the folders must then be empty).")
        old_cfg, source = saved_run_config(manifest_dir, cfg["run_name"], ckpt_dir)
        if old_cfg is None:
            sys.exit(f"ERROR: cannot resume: no run manifest ({manifest_dir / ('run_' + cfg['run_name'] + '.json')}) and no "
                     "ema_step checkpoint to compare the config with.")
        changes = diff_configs(old_cfg, cfg)
        model_bad, other_bad = split_resume_changes(changes)
        if model_bad:
            sys.exit("ERROR: refusing to resume BEFORE loading any weights: the model definition changed, so the saved "
                     "weights would not fit. Changed: " + "; ".join(f"{k}: {o} -> {n}" for k, (o, n) in model_bad.items()) +
                     f". Put them back to the values in {source}, or start a new run_name.")
        if other_bad:
            sys.exit("ERROR: refusing to resume: these config keys changed but may not be changed on resume: " +
                     "; ".join(f"{k}: {o} -> {n}" for k, (o, n) in other_bad.items()) +
                     ". Allowed: " + ", ".join(sorted(ALLOWED_RESUME_CHANGES)) + f". (compared with {source})")
    elif any(existing.values()):
        full = [str(k) for k, v in existing.items() if v]
        sys.exit("ERROR: this run's folder(s) already contain files: " + ", ".join(full) + ". Starting over would mix runs. "
                 f"To continue the run, add --resume ({resume_cmd}); to start fresh, choose a new run_name / folders "
                 "or remove these folders yourself.")
    write_run_marker(ckpt_dir, cfg["run_name"])
    write_run_marker(log_dir, cfg["run_name"])

    # data + manifests (the split is created once and then reused)
    manifest, split_path = build_manifest(cfg)
    run_manifest = manifest_dir / f"run_{cfg['run_name']}.json"
    if not run_manifest.exists():
        write_run_manifest(cfg, split_path)
    limit = SMOKE_IMAGES if args.smoke else None
    loader_a, loader_b = make_dataloaders(cfg, manifest, "train", limit=limit)
    stream_a, stream_b = infinite(loader_a), infinite(loader_b)
    holdout = "none" if manifest.get("holdout") == "none" else \
        f"A {len(manifest['A']['heldout'])} / B {len(manifest['B']['heldout'])} images held out"
    print(f"train images: A(Monet)={len(loader_a.dataset)} B(photo)={len(loader_b.dataset)} | holdout: {holdout}")

    # fixed images for the sample grids (deterministic transform, same images every time)
    tf_eval = build_transform(d["image_size"], [], train=False)
    fixed = []
    for dom in ("A", "B"):
        ds = ImageFolderDataset(resolve(manifest[dom]["dir"]), manifest[dom]["train"][:4], tf_eval)
        fixed.append(torch.stack([ds[i] for i in range(len(ds))]).to(device))
    fixed_a, fixed_b = fixed

    # models: G_AB Monet->photo, G_BA photo->Monet; D_A judges Monet, D_B judges photos
    up = m.get("upsample", "convtranspose")  # run01 predates this key
    g_ab = init_weights(ResnetGenerator(m["ngf"], m["n_blocks"], upsample=up)).to(device)
    g_ba = init_weights(ResnetGenerator(m["ngf"], m["n_blocks"], upsample=up)).to(device)
    d_a = init_weights(PatchDiscriminator(m["ndf"])).to(device)
    d_b = init_weights(PatchDiscriminator(m["ndf"])).to(device)
    n_up = sum(isinstance(x, torch.nn.ConvTranspose2d) for x in g_ab.modules())
    n_nn = sum(isinstance(x, torch.nn.Upsample) for x in g_ab.modules())
    print(f"architecture: n_blocks={m['n_blocks']} ngf={m['ngf']} ndf={m['ndf']} upsample={up} | upsampling: "
          + (f"{n_up} x ConvTranspose2d(k3,s2,p1,op1) + InstanceNorm + ReLU" if n_up
             else f"{n_nn} x Upsample(nearest) + ReflectionPad2d(1) + Conv2d(k3) + InstanceNorm + ReLU")
          + f" | lambda_cycle={lam_cyc} lambda_identity={lam_id}")
    print("parameters: " + ", ".join(f"{n}={count_params(net):,}" for n, net in
                                      (("G_AB", g_ab), ("G_BA", g_ba), ("D_A", d_a), ("D_B", d_b)))
          + f", total={sum(count_params(x) for x in (g_ab, g_ba, d_a, d_b)):,}")
    ema_ab, ema_ba = copy.deepcopy(g_ab).eval(), copy.deepcopy(g_ba).eval()
    for p in list(ema_ab.parameters()) + list(ema_ba.parameters()):
        p.requires_grad_(False)

    betas = tuple(t["betas"])
    g_params = list(g_ab.parameters()) + list(g_ba.parameters())
    opt_g = torch.optim.Adam(g_params, lr=t["lr"], betas=betas)
    opt_da = torch.optim.Adam(d_a.parameters(), lr=t["lr"], betas=betas)
    opt_db = torch.optim.Adam(d_b.parameters(), lr=t["lr"], betas=betas)
    lam = lr_lambda(decay_start, total)  # built from the CURRENT config, also when resuming
    sched = {k: torch.optim.lr_scheduler.LambdaLR(o, lam)
             for k, o in (("sched_g", opt_g), ("sched_da", opt_da), ("sched_db", opt_db))}

    pool_a, pool_b = ImagePool(t["pool_size"]), ImagePool(t["pool_size"])
    meta = MetaState()
    models = {"G_AB": g_ab, "G_BA": g_ba, "D_A": d_a, "D_B": d_b, "ema_G_AB": ema_ab, "ema_G_BA": ema_ba,
              "pool_A": PoolState(pool_a, device), "pool_B": PoolState(pool_b, device), "meta": meta}
    optimizers = {"opt_g": opt_g, "opt_da": opt_da, "opt_db": opt_db, **sched}

    raw_log = log_dir / "raw_train_log.txt"
    step = 0
    if args.resume:
        step, saved_cfg = load_checkpoint(last_path, models, optimizers, map_location=device)
        if saved_cfg.get("run_name") != cfg["run_name"]:  # never continue another run's training
            sys.exit(f"ERROR: {last_path} belongs to run '{saved_cfg.get('run_name')}', not '{cfg['run_name']}'.")
        if {"train.lr", "train.betas", "train.decay_start", "train.total_steps"} & set(changes):
            apply_current_hparams([(opt_g, sched["sched_g"]), (opt_da, sched["sched_da"]), (opt_db, sched["sched_db"])],
                                  t["lr"], betas, lam, step)
        print(f"RESUMED {cfg['run_name']} from step {step}", flush=True)
        if changes:
            emit(f"[RESUME] step {step} | config changes since {source}: " +
                 "; ".join(f"{k}: {o} -> {n}" for k, (o, n) in changes.items()), raw_log)
        else:
            emit(f"[RESUME] step {step} | config identical to {source}", raw_log)
        rm = write_resume_manifest(cfg, args.config, manifest_dir, step, changes, source)
        print(f"wrote {rm}", flush=True)
        if step >= total:
            print(f"Run already complete: step {step} >= total_steps {total}. Nothing to do "
                  "(raise train.total_steps in the config to train further).")
            return

    logger = AppendOnlyLogger(log_dir / "history.csv", LOG_FIELDS)  # opened in append mode: earlier rows are never rewritten
    fid_logger = AppendOnlyLogger(log_dir / "grader_fid.csv", FID_FIELDS) if fid_every else None
    best = BestFid(log_dir / "grader_fid.csv" if fid_every else None)
    best_json = log_dir / "best_fid.json"
    fid_tmp = sample_dir.parent / "_train_eval_tmp"

    # ---- start-up summary
    print("=" * 100)
    print(f"RUN {cfg['run_name']} | config {args.config} | seed {cfg['seed']} | device {device}" + (f" | mode {mode}" if mode else ""))
    print(f"steps: {step} -> {total} | LR {t['lr']} constant until {decay_start}, then linear to 0 at {total} | "
          f"log_every {log_every} | ckpt_every {ckpt_every} | grader FID every {fid_every if fid_every else 'off'}")
    print(f"losses: lambda_cycle {lam_cyc} | lambda_identity {lam_id} | diffaug {policy} | betas {list(betas)} | "
          f"pool {t['pool_size']} | ema {ema_decay}")
    print(f"model: n_blocks {m['n_blocks']} ngf {m['ngf']} ndf {m['ndf']} upsample {up} | data: image {d['image_size']} "
          f"batch {d['batch_size']} workers {d['num_workers']} | holdout: {holdout}")
    print(f"folders: checkpoints {ckpt_dir} | logs {log_dir} | samples {sample_dir}")
    print(f"Ctrl+C = save and stop after the current step | resume with: {resume_cmd}")
    print("=" * 100, flush=True)

    stop = StopFlag()
    install_sigint(stop)  # only now: Ctrl+C during start-up behaves normally

    acc, n_acc = {}, 0  # interval means of losses / grad norms (tensors, no per-step sync)
    t_last, step_last = time.perf_counter(), step
    elapsed0 = meta.data["elapsed"]
    t_start = time.perf_counter()
    last_saved_step = step if args.resume else None

    while step < total:
        if stop.requested:  # the previous step is complete here
            if step == 0:
                print("STOPPED at step 0 (no training was done, nothing saved).", flush=True)
            else:
                if last_saved_step != step:
                    save_state(step, last_path, ckpt_dir, models, optimizers, cfg, backup_dir, ema_ab, ema_ba, meta,
                               elapsed0 + (time.perf_counter() - t_start))
                emit(f"[STOP] step {step} | checkpoint saved", raw_log)  # history.csv rows are written (and closed) as they happen
                print(f"STOPPED at step {step}. Checkpoint saved. Resume with: {resume_cmd}", flush=True)
            sys.exit(0)
        step += 1
        real_a = next(stream_a).to(device, non_blocking=True)
        real_b = next(stream_b).to(device, non_blocking=True)

        # ---- generators
        set_requires_grad([d_a, d_b], False)
        opt_g.zero_grad(set_to_none=True)
        fake_b, fake_a = g_ab(real_a), g_ba(real_b)
        g_adv = lsgan_loss(d_b(diff_augment(fake_b, policy)), True) + \
            lsgan_loss(d_a(diff_augment(fake_a, policy)), True)
        cyc = cycle_l1(g_ba(fake_b), real_a) + cycle_l1(g_ab(fake_a), real_b)
        idt = identity_l1(g_ba(real_a), real_a) + identity_l1(g_ab(real_b), real_b)
        loss_g = g_adv + lam_cyc * cyc + lam_id * idt
        loss_g.backward()
        gn_g = grad_norm(g_params)
        ok = finite(loss_g, gn_g)
        if ok:
            opt_g.step()
            update_ema(ema_ab, g_ab, ema_decay)
            update_ema(ema_ba, g_ba, ema_decay)
        else:
            opt_g.zero_grad(set_to_none=True)
            meta.data["nan_count"] += 1

        # ---- discriminators (fakes come from the history pool)
        set_requires_grad([d_a, d_b], True)
        if ok:
            opt_da.zero_grad(set_to_none=True)
            opt_db.zero_grad(set_to_none=True)
            pf_a, pf_b = pool_a.query(fake_a), pool_b.query(fake_b)
            loss_da = 0.5 * (lsgan_loss(d_a(diff_augment(real_a, policy)), True) +
                             lsgan_loss(d_a(diff_augment(pf_a, policy)), False))
            loss_db = 0.5 * (lsgan_loss(d_b(diff_augment(real_b, policy)), True) +
                             lsgan_loss(d_b(diff_augment(pf_b, policy)), False))
            loss_da.backward()
            loss_db.backward()
            gn_da, gn_db = grad_norm(d_a.parameters()), grad_norm(d_b.parameters())
            if finite(loss_da, loss_db, gn_da, gn_db):
                opt_da.step()
                opt_db.step()
                stats = {"g_adv": g_adv, "cycle": cyc, "identity": idt, "d_a": loss_da, "d_b": loss_db,
                         "grad_g": gn_g, "grad_d_a": gn_da, "grad_d_b": gn_db}
                for k, v in stats.items():
                    acc[k] = acc.get(k, 0.0) + v.detach().float()
                n_acc += 1
            else:
                opt_da.zero_grad(set_to_none=True)
                opt_db.zero_grad(set_to_none=True)
                meta.data["nan_count"] += 1

        for s in sched.values():
            s.step()

        if args.bench and step == BENCH_WARMUP:  # start the timed window after warmup (cudnn autotune etc.)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_warm = time.perf_counter()
        if args.bench and step == total:  # stop the clock before the end-of-run checkpoint / samples
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

        # ---- logging
        if step % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            ips = 2 * (step - step_last) / max(now - t_last, 1e-9)  # one image per domain per step
            t_last, step_last = now, step
            seconds = elapsed0 + (now - t_start)
            row = {k: (acc[k] / n_acc).item() if n_acc else float("nan") for k in LOG_FIELDS[1:9]}
            row = {"step": step, **row, "lr": opt_g.param_groups[0]["lr"],
                   "nan_count": meta.data["nan_count"], "images_per_sec": ips, "seconds": seconds}
            acc, n_acc = {}, 0
            logger.log(row)
            meta.data["history"].append(row)
            line = " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
            print(line, flush=True)
            with open(raw_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        # ---- checkpoint + samples (+ grader FID)
        if step % ckpt_every == 0 or step == total:
            save_state(step, last_path, ckpt_dir, models, optimizers, cfg, backup_dir, ema_ab, ema_ba, meta,
                       elapsed0 + (time.perf_counter() - t_start))
            last_saved_step = step
            save_samples(g_ab, g_ba, fixed_a, fixed_b, sample_dir / f"step{step:06d}.png")
            save_samples(ema_ab, ema_ba, fixed_a, fixed_b, sample_dir / f"step{step:06d}_ema.png")
            print(f"saved checkpoint + samples at step {step}", flush=True)
            if fid_every and step % fid_every == 0:
                grader_fid_hook(step, ema_ab, ema_ba, cfg, device, raw_log, best, fid_logger, best_json, fid_tmp)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = elapsed0 + (time.perf_counter() - t_start)
    summary = {
        "run_name": cfg["run_name"], "device": str(device), "steps": step,
        "total_train_seconds": total_seconds,
        "peak_gpu_memory_mb": peak_memory_mb() if device.type == "cuda" else None,
        "nan_count": meta.data["nan_count"],
    }
    if fid_every:
        summary.update(best_grader_fid_step=best.step, best_grader_fid_avg=None if best.step is None else best.value)
    if args.bench:
        sps = (step - BENCH_WARMUP) / (t_end - t_warm)
        summary.update(steps_per_sec_after_warmup=sps, images_per_sec_after_warmup=2 * sps,
                       warmup_steps=BENCH_WARMUP)
    with open(log_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("run summary:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
