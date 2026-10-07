"""
Standalone Evaluation Script for CycleGAN Image Style Transfer (Task 3)
Calculates FID and MiFID between real and generated images in both directions:
  - Photo -> Monet (pred_B2A vs real monet)
  - Monet -> Photo (pred_A2B vs real photo)
Generates `submission.csv` in the format required for Kaggle.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

import scipy.linalg
from scipy.spatial.distance import cosine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN Style Transfer (FID & MiFID)")
    parser.add_argument("--base", type=str, default=None, help="Base directory containing dataset/ and outputs/")
    parser.add_argument("--real_monet", type=str, default=None, help="Path to real Monet images")
    parser.add_argument("--real_photo", type=str, default=None, help="Path to real Photo images")
    parser.add_argument("--gen_a2b", type=str, default=None, help="Path to generated Photo images (pred_A2B)")
    parser.add_argument("--gen_b2a", type=str, default=None, help="Path to generated Monet images (pred_B2A)")
    parser.add_argument("--n_eval", type=int, default=300, help="Number of images per domain to evaluate (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Inception v3")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Where to save submission.csv")
    return parser.parse_args()


def resolve_path(candidates, folder_name):
    for cand in candidates:
        if cand is None:
            continue
        p = Path(cand) / folder_name if folder_name else Path(cand)
        if p.is_dir():
            return str(p.resolve())
    return None


def find_directories(args):
    cwd = Path.cwd()
    root = Path(__file__).resolve().parent.parent

    base_candidates = [
        args.base,
        r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl",
        r"C:\Users\019129357\Downloads\lab1_dl",
        str(root),
        str(cwd),
        str(cwd.parent),
        "/kaggle/working",
    ]

    monet_candidates = [
        args.real_monet,
        r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\dataset\dataset\monet_jpg",
        r"C:\Users\019129357\Downloads\lab1_dl\dataset\dataset\monet_jpg",
        str(root / "dataset" / "dataset" / "monet_jpg"),
        str(root / "dataset" / "monet_jpg"),
        str(cwd / "dataset" / "dataset" / "monet_jpg"),
        str(cwd / "dataset" / "monet_jpg"),
        "/kaggle/input/gan-getting-started/monet_jpg",
    ]

    photo_candidates = [
        args.real_photo,
        r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\dataset\dataset\photo_jpg",
        r"C:\Users\019129357\Downloads\lab1_dl\dataset\dataset\photo_jpg",
        str(root / "dataset" / "dataset" / "photo_jpg"),
        str(root / "dataset" / "photo_jpg"),
        str(cwd / "dataset" / "dataset" / "photo_jpg"),
        str(cwd / "dataset" / "photo_jpg"),
        "/kaggle/input/gan-getting-started/photo_jpg",
    ]

    a2b_candidates = [
        args.gen_a2b,
        str(root / "outputs" / "pred_A2B"),
        str(root / "04_preds" / "pred_A2B"),
        str(cwd / "outputs" / "pred_A2B"),
        str(cwd / "04_preds" / "pred_A2B"),
        r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\outputs\pred_A2B",
    ]

    b2a_candidates = [
        args.gen_b2a,
        str(root / "outputs" / "pred_B2A"),
        str(root / "04_preds" / "pred_B2A"),
        str(cwd / "outputs" / "pred_B2A"),
        str(cwd / "04_preds" / "pred_B2A"),
        r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\outputs\pred_B2A",
    ]

    real_monet = resolve_path(monet_candidates, "")
    real_photo = resolve_path(photo_candidates, "")
    gen_a2b = resolve_path(a2b_candidates, "")
    gen_b2a = resolve_path(b2a_candidates, "")

    return real_monet, real_photo, gen_a2b, gen_b2a


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


def get_inception_model(device):
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
def get_activations(model, image_paths, device, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        batch_paths = image_paths[i:i + batch_size]
        x = load_batch(batch_paths).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    try:
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    except TypeError:
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        try:
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
            if isinstance(covmean, tuple):
                covmean = covmean[0]
        except TypeError:
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(model, real_paths, gen_paths, device, batch_size=32, subsample_to_match=True):
    real_paths = sorted(real_paths)
    gen_paths = sorted(gen_paths)

    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:n]
        gen_paths = gen_paths[:n]

    real_act = get_activations(model, real_paths, device, batch_size=batch_size)
    gen_act = get_activations(model, gen_paths, device, batch_size=batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    m = min(len(real_act), len(gen_act))
    cos_dists = [cosine(real_act[i], gen_act[i]) for i in range(m)]
    mifid = float(np.mean(cos_dists))

    return fid, mifid


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    real_monet, real_photo, gen_a2b, gen_b2a = find_directories(args)

    print("\n--- Folder Paths ---")
    print(f"REAL_MONET : {real_monet}")
    print(f"REAL_PHOTO : {real_photo}")
    print(f"GEN_A2B    : {gen_a2b}")
    print(f"GEN_B2A    : {gen_b2a}")

    missing = []
    for name, p in [("REAL_MONET", real_monet), ("REAL_PHOTO", real_photo),
                    ("GEN_A2B", gen_a2b), ("GEN_B2A", gen_b2a)]:
        if p is None or not os.path.isdir(p):
            missing.append(f"{name} ({p})")

    if missing:
        print(f"\nERROR: The following required directories were not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease supply paths via CLI flags or make sure dataset & output directories exist.")
        sys.exit(1)

    real_monet_imgs = take_n(list_images(real_monet), args.n_eval)
    real_photo_imgs = take_n(list_images(real_photo), args.n_eval)
    gen_a2b_imgs = take_n(list_images(gen_a2b), args.n_eval)
    gen_b2a_imgs = take_n(list_images(gen_b2a), args.n_eval)

    print(f"\nCounts (capped at {args.n_eval}):")
    print(f"  Real Monet: {len(real_monet_imgs):3d} | Gen Monet (B2A): {len(gen_b2a_imgs):3d}")
    print(f"  Real Photo: {len(real_photo_imgs):3d} | Gen Photo (A2B): {len(gen_a2b_imgs):3d}")

    print("\nLoading InceptionV3 model...")
    inception = get_inception_model(device)

    print("\n[1/2] Evaluating Photo -> Monet (pred_B2A vs Real Monet)...")
    fid_B2A, mifid_B2A = calculate_fid_mifid(
        inception, real_monet_imgs, gen_b2a_imgs, device, batch_size=args.batch_size
    )
    print(f"  -> FID_B2A: {fid_B2A:.4f} | MiFID_B2A: {mifid_B2A:.4f}")

    print("\n[2/2] Evaluating Monet -> Photo (pred_A2B vs Real Photo)...")
    fid_A2B, mifid_A2B = calculate_fid_mifid(
        inception, real_photo_imgs, gen_a2b_imgs, device, batch_size=args.batch_size
    )
    print(f"  -> FID_A2B: {fid_A2B:.4f} | MiFID_A2B: {mifid_A2B:.4f}")

    sub_fid = (fid_A2B + fid_B2A) / 2.0
    sub_mifid = (mifid_A2B + mifid_B2A) / 2.0

    submission = pd.DataFrame([{
        "ID": 1,
        "FID": float(sub_fid),
        "MiFID": float(sub_mifid)
    }])

    # Save to requested destination
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)

    # Also save to 01_submission/submission.csv if directory exists
    sub_dir = Path(__file__).resolve().parent.parent / "01_submission"
    if sub_dir.is_dir():
        submission.to_csv(sub_dir / "submission.csv", index=False)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE — SUBMISSION GENERATED")
    print("=" * 55)
    print(f"Photo -> Monet (B2A) : FID={fid_B2A:.3f}, MiFID={mifid_B2A:.4f}")
    print(f"Monet -> Photo (A2B) : FID={fid_A2B:.3f}, MiFID={mifid_A2B:.4f}")
    print("-" * 55)
    print(f"Average FID          : {sub_fid:.4f}")
    print(f"Average MiFID        : {sub_mifid:.4f}")
    print("-" * 55)
    print(f"Saved: {out_csv.resolve()}")
    if sub_dir.is_dir():
        print(f"Saved: {(sub_dir / 'submission.csv').resolve()}")
    print("=" * 55)
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
