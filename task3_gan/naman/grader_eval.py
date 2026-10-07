"""Grader-faithful scorer.

The functions below are copied VERBATIM from Part3_Evaluation_Script.ipynb (list_images, take_n,
get_inception_model, INCEPTION_TF, load_batch, get_activations, frechet_distance, calculate_fid_mifid).
Only the command-line wrapper at the bottom is new. It reproduces the notebook's flow:
  B2A: real Monet  vs pred_B2A     A2B: real photo vs pred_A2B
  output: one row  ID=1, FID, MiFID = average of the two directions.

Usage (from the naman/ folder):
    python grader_eval.py --pred-a2b outputs/pred_A2B --pred-b2a outputs/pred_B2A --out-csv outputs/grader_submission.csv
Relative paths are looked up in the current folder first, then in the task3_gan folder.
"""
import argparse
import os, glob
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

import scipy.linalg
from scipy.spatial.distance import cosine

# ---- notebook cell 3 (constants; paths are supplied by the CLI below) ----
N_EVAL = 300

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- notebook cell 4 ----
def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths

def take_n(paths, n):
    if n is None:
        return paths
    return paths[:min(n, len(paths))]

# ---- notebook cell 5 ----
# Inception feature extractor
def get_inception_model():
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    return inception

INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def load_batch(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(INCEPTION_TF(img))
    return torch.stack(imgs, dim=0)

@torch.no_grad()
def get_activations(model, image_paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        batch_paths = image_paths[i:i+batch_size]
        x = load_batch(batch_paths).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)

# ---- notebook cell 6 ----
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

# ---- notebook cell 7 ----
def calculate_fid_mifid(real_paths, gen_paths, batch_size=32, subsample_to_match=True):
    # For fair comparison, match counts
    real_paths = sorted(real_paths)
    gen_paths  = sorted(gen_paths)

    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:n]
        gen_paths  = gen_paths[:n]

    model = get_inception_model()

    real_act = get_activations(model, real_paths, batch_size=batch_size)
    gen_act  = get_activations(model, gen_paths,  batch_size=batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0),  np.cov(gen_act,  rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    # mean cosine distance between feature vectors,
    # paired by index after subsampling/matching
    m = min(len(real_act), len(gen_act))
    cos_dists = [cosine(real_act[i], gen_act[i]) for i in range(m)]
    mifid = float(np.mean(cos_dists))

    return fid, mifid


# ---- thin CLI (not part of the notebook) ----
TASK3 = Path(__file__).resolve().parents[1]  # task3_gan/

# Compatibility shim, NOT part of the notebook: scipy >= 1.18 removed the `disp` argument of sqrtm, which
# frechet_distance (copied verbatim above) still passes. On such a scipy, accept `disp` and return what
# old scipy returned: the matrix for disp=True, (matrix, relative error) for disp=False. The square root
# itself is still scipy's own sqrtm. On an older scipy this block does nothing.
import inspect
if "disp" not in inspect.signature(scipy.linalg.sqrtm).parameters:
    _sqrtm_new = scipy.linalg.sqrtm

    def _sqrtm_compat(A, disp=True, blocksize=64):
        X = _sqrtm_new(A)
        if disp:
            return X
        return X, float(np.linalg.norm(X.dot(X) - A, "fro") / np.linalg.norm(A, "fro"))
    scipy.linalg.sqrtm = _sqrtm_compat


def find_dir(p):
    """Absolute path as given; a relative path is tried against the current folder, then task3_gan/."""
    p = Path(p)
    if p.is_absolute():
        return str(p)
    for base in (Path.cwd(), TASK3):
        if (base / p).is_dir():
            return str(base / p)
    return str(TASK3 / p)  # not found: the folder check below reports it


def main():
    ap = argparse.ArgumentParser(description="Grader-faithful FID / MiFID (copy of Part3_Evaluation_Script.ipynb)")
    ap.add_argument("--real-monet", default="data/monet_jpg")
    ap.add_argument("--real-photo", default="data/photo_jpg")
    ap.add_argument("--pred-a2b", required=True, help="generated photos (Monet -> Photo)")
    ap.add_argument("--pred-b2a", required=True, help="generated Monet (Photo -> Monet)")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()

    REAL_MONET, REAL_PHOTO = find_dir(args.real_monet), find_dir(args.real_photo)
    GEN_A2B, GEN_B2A = find_dir(args.pred_a2b), find_dir(args.pred_b2a)

    # ---- notebook cell 8 ----
    for d in [REAL_MONET, REAL_PHOTO, GEN_A2B, GEN_B2A]:
        assert os.path.isdir(d), f"Missing folder: {d}"

    real_monet = take_n(list_images(REAL_MONET), N_EVAL)
    real_photo = take_n(list_images(REAL_PHOTO), N_EVAL)
    gen_a2b    = take_n(list_images(GEN_A2B),    N_EVAL)  # Monet->Photo (generated photos)
    gen_b2a    = take_n(list_images(GEN_B2A),    N_EVAL)  # Photo->Monet (generated monet)

    print("\nCounts (after N_EVAL cap):")
    print("Real Monet:", len(real_monet), " | Gen Monet (B2A):", len(gen_b2a))
    print("Real Photo:", len(real_photo), " | Gen Photo (A2B):", len(gen_a2b))
    print("Used after min() matching: B2A n =", min(len(real_monet), len(gen_b2a)),
          "| A2B n =", min(len(real_photo), len(gen_a2b)))

    # ---- notebook cell 9 ----
    print("\n Evaluating Photo -> Monet (B2A) ")
    # Ground truth = real Monet; Generated = pred_B2A (generated Monet-like)
    fid_B2A, mifid_B2A = calculate_fid_mifid(real_monet, gen_b2a, batch_size=BATCH_SIZE)
    print(f"[Photo->Monet] FID={fid_B2A:.3f}  MiFID={mifid_B2A:.4f}")

    print("\n Evaluating Monet -> Photo (A2B) ")
    # Ground truth = real Photo; Generated = pred_A2B (generated Photo-like)
    fid_A2B, mifid_A2B = calculate_fid_mifid(real_photo, gen_a2b, batch_size=BATCH_SIZE)
    print(f"[Monet->Photo] FID={fid_A2B:.3f}  MiFID={mifid_A2B:.4f}")

    # ---- notebook cell 10 ----
    import pandas as pd

    sub_fid  = (fid_A2B + fid_B2A) / 2
    sub_mifid = (mifid_A2B + mifid_B2A) / 2

    submission = pd.DataFrame([{
        "ID": 1,
        "FID": float(sub_fid),
        "MiFID": float(sub_mifid)}])

    out = Path(args.out_csv)
    out = out if out.is_absolute() else Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(submission)
    print(f"wrote {out} | runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
