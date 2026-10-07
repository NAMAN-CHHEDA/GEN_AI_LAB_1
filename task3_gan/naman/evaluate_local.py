"""Local evaluation of a CycleGAN EMA checkpoint (A = Monet, B = photo).

1. Translate ALL Monet images -> outputs/pred_A2B/ and ALL photos -> outputs/pred_B2A/ (RGB JPG q95).
2. Verify every input has a correctly sized RGB JPG prediction.
3. Metrics per direction, measured on the saved JPGs:
   FID, KID (mean/std), precision & recall (k=3), LPIPS (input vs translation),
   content cosine similarity (input vs translation, Inception features),
   cycle-reconstruction L1 (A->B->A and B->A->B, in [-1, 1] units).
   "Real" = all images of the target domain (A2B -> photos, B2A -> Monet).
4. Writes submission.csv and full_metrics_report.csv.

Usage (from the naman/ folder):
    python evaluate_local.py --ckpt checkpoints/ema_step25000.pt [--config configs/run01.yaml]
                             [--max-images N] [--smoke] [--out-dir DIR]

The pretrained Inception / AlexNet weights are used ONLY to measure; they never generate images.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

NAMAN = Path(__file__).resolve().parent
sys.path.insert(0, str(NAMAN / "src"))
sys.path.insert(0, str(NAMAN))

from config import load_config, resolve  # noqa: E402
from data import ImageFolderDataset, build_transform, list_images  # noqa: E402
from logging_utils import seed_everything  # noqa: E402
from models import ResnetGenerator  # noqa: E402
from train import apply_smoke  # noqa: E402  (reuse the exact smoke overrides)

BATCH = 16
KID_SUBSETS, KID_SUBSET_SIZE = 100, 1000
PRC_K = 3
MEMORIZATION_EPS = 0.1  # Kaggle MiFID threshold on the memorization distance


def to_uint8(x):
    """[-1, 1] float tensor -> uint8 [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def make_loader(folder, files, tf):
    # num_workers=0: simplest and safest on Windows; evaluation is not data-bound.
    return DataLoader(ImageFolderDataset(folder, files, tf), batch_size=BATCH, shuffle=False, num_workers=0)


# ---------------------------------------------------------------- step 1: translate

@torch.no_grad()
def translate(gen, rev_gen, folder, files, out_dir, tf, device):
    """Write gen(x) as JPG q95 for every file. Returns mean L1 of rev_gen(gen(x)) vs x."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.jpeg"))
    for p in stale:  # this folder holds only generated predictions; start from a clean slate
        p.unlink()
    if stale:
        print(f"  removed {len(stale)} old prediction files from {out_dir}")
    cycle_sum, i = 0.0, 0
    for x in make_loader(folder, files, tf):
        x = x.to(device)
        fake = gen(x)
        cycle_sum += (rev_gen(fake) - x).abs().flatten(1).mean(1).sum().item()
        arr = to_uint8(fake).permute(0, 2, 3, 1).cpu().numpy()
        for a in arr:
            Image.fromarray(a).save(out_dir / (Path(files[i]).stem + ".jpg"), "JPEG", quality=95)
            i += 1
    return cycle_sum / len(files)


# ---------------------------------------------------------------- step 2: verify

def verify(files, pred_dir, size, name):
    stems = [Path(f).stem for f in files]
    if len(set(stems)) != len(stems):
        raise RuntimeError(f"{name}: input file names collide after switching to .jpg")
    present = {p.name for p in pred_dir.iterdir() if p.is_file() and not p.name.startswith(".")}  # ignore .gitkeep
    expected = {s + ".jpg" for s in stems}
    if present != expected:
        raise RuntimeError(f"{name}: missing {sorted(expected - present)[:5]} extra {sorted(present - expected)[:5]}")
    for fn in sorted(expected):
        with Image.open(pred_dir / fn) as im:
            if im.format != "JPEG" or im.mode != "RGB" or im.size != (size, size):
                raise RuntimeError(f"{name}: {fn} is {im.format}/{im.mode}/{im.size}, want JPEG/RGB/{(size, size)}")
    print(f"  {name}: OK - {len(expected)} predictions, all RGB JPG {size}x{size}")


# ---------------------------------------------------------------- features + metrics

def load_inception(device):
    from torch_fidelity.utils import create_feature_extractor
    return create_feature_extractor("inception-v3-compat", ["2048"], cuda=device.type == "cuda")


@torch.no_grad()
def inception_feats(ext, x, device):
    return ext(to_uint8(x).to(device))[0].double()


def fid_from_feats(f1, f2):
    f1, f2 = f1.cpu().numpy(), f2.cpu().numpy()
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    # tr sqrt(s1 s2) = sum sqrt(eigenvalues of s1 s2) (real, >= 0 for PSD matrices)
    eig = np.linalg.eigvals(s1 @ s2).real.clip(min=0)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(eig).sum())


def kid_from_feats(f1, f2, seed):
    """Unbiased MMD^2 with cubic polynomial kernel over random subsets (same recipe as torch-fidelity)."""
    n = min(KID_SUBSET_SIZE, len(f1), len(f2))
    g = torch.Generator().manual_seed(seed)
    d = f1.shape[1]
    vals = []
    for _ in range(KID_SUBSETS):
        x = f1[torch.randperm(len(f1), generator=g)[:n].to(f1.device)]
        y = f2[torch.randperm(len(f2), generator=g)[:n].to(f2.device)]
        kxx, kyy, kxy = ((a @ b.T / d + 1) ** 3 for a, b in ((x, x), (y, y), (x, y)))
        m = n
        mmd = ((kxx.sum() - kxx.trace()) / (m * (m - 1)) + (kyy.sum() - kyy.trace()) / (m * (m - 1))
               - 2 * kxy.sum() / (m * m))
        vals.append(mmd.item())
    return float(np.mean(vals)), float(np.std(vals))


def precision_recall(fake, real, k=PRC_K):
    """Kynkaanniemi et al. 2019: k-NN manifolds. precision = fakes inside the real manifold,
    recall = reals inside the fake manifold."""
    def radii(f):  # distance to k-th nearest neighbour (index 0 is the point itself)
        return torch.cdist(f, f).kthvalue(k + 1, dim=1).values

    def inside(a, b, rb):  # fraction of a within any radius-rb ball around b
        return (torch.cdist(a, b) <= rb[None, :]).any(1).float().mean().item()
    return inside(fake, real, radii(real)), inside(real, fake, radii(fake))


def evaluate_direction(name, src_dir_info, pred_dir, real_dir_info, tf, ext, lp, device, seed):
    src_folder, src_files = src_dir_info
    real_folder, real_files = real_dir_info
    pred_files = [Path(f).stem + ".jpg" for f in src_files]
    f_src, f_fake, lpips_vals = [], [], []
    with torch.no_grad():
        for xs, xf in zip(make_loader(src_folder, src_files, tf), make_loader(pred_dir, pred_files, tf)):
            xs, xf = xs.to(device), xf.to(device)
            lpips_vals.append(lp(xs, xf).flatten().cpu())
            f_src.append(inception_feats(ext, xs, device))
            f_fake.append(inception_feats(ext, xf, device))
        f_real = [inception_feats(ext, x.to(device), device) for x in make_loader(real_folder, real_files, tf)]
    f_src, f_fake, f_real = (torch.cat(v) for v in (f_src, f_fake, f_real))

    fid = fid_from_feats(f_fake, f_real)
    kid_mean, kid_std = kid_from_feats(f_fake, f_real, seed)
    prec, rec = precision_recall(f_fake, f_real)
    content_cos = torch.nn.functional.cosine_similarity(f_src, f_fake, dim=1).mean().item()
    # MiFID, LOCAL APPROXIMATION: memorization distance d = mean over fakes of (1 - cosine similarity to the
    # nearest real image's Inception feature); MiFID = FID / d if d < 0.1 else FID (Kaggle's thresholding).
    # Kaggle computes its own MiFID on its own features/reference set, so numbers will differ.
    fn, rn = torch.nn.functional.normalize(f_fake, dim=1), torch.nn.functional.normalize(f_real, dim=1)
    mem_d = (1 - (fn @ rn.T).max(1).values).mean().item()
    mifid = fid / (mem_d if mem_d < MEMORIZATION_EPS else 1.0)
    return {
        "direction": name, "n_fake": len(f_fake), "n_real": len(f_real), "FID": fid, "MiFID": mifid,
        "KID_mean": kid_mean, "KID_std": kid_std, "precision": prec, "recall": rec,
        "LPIPS": torch.cat(lpips_vals).mean().item(), "content_cosine": content_cos,
        "memorization_distance": mem_d,
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ema_stepN.pt")
    ap.add_argument("--config", default="configs/run01.yaml")
    ap.add_argument("--max-images", type=int, default=None, help="use only the first N images per domain")
    ap.add_argument("--smoke", action="store_true", help="use the smoke settings and smoke output folders")
    ap.add_argument("--out-dir", default=None,
                    help="write pred_A2B/, pred_B2A/, submission.csv and full_metrics_report.csv here instead of "
                         "the default outputs (the real outputs/pred_* are then left untouched)")
    args = ap.parse_args()
    if "smoke" in Path(args.ckpt).as_posix().lower() and not args.smoke:
        sys.exit(f"ERROR: checkpoint path '{args.ckpt}' contains 'smoke' but --smoke was not passed. "
                 "Re-run with --smoke, or use a real checkpoint, so smoke weights never write into the real outputs/pred_* folders.")

    cfg = load_config(args.config)
    if args.smoke:
        apply_smoke(cfg)
    size = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    seed = cfg.get("seed") or 0
    seed_everything(seed)

    out_root = NAMAN / "outputs" / "smoke" if args.smoke else NAMAN / "outputs"
    csv_dir = out_root if args.smoke else NAMAN
    if args.out_dir:  # e.g. a candidate checkpoint: everything goes to one folder
        out_root = csv_dir = Path(args.out_dir).resolve()
    dir_a2b, dir_b2a = out_root / "pred_A2B", out_root / "pred_B2A"

    # ---- generators (EMA weights)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    mc = ckpt.get("config", cfg)["model"]  # architecture comes from the checkpoint's own config
    up = mc.get("upsample", "convtranspose")  # run01 checkpoints predate this key
    g_ab = ResnetGenerator(mc["ngf"], mc["n_blocks"], upsample=up).to(device).eval()
    g_ba = ResnetGenerator(mc["ngf"], mc["n_blocks"], upsample=up).to(device).eval()
    g_ab.load_state_dict(ckpt["G_AB"])
    g_ba.load_state_dict(ckpt["G_BA"])
    print(f"loaded EMA generators from {args.ckpt} (step {ckpt.get('step')}), image size {size}")

    folder_a, folder_b = resolve(cfg["paths"]["monet_dir"]), resolve(cfg["paths"]["photo_dir"])
    files_a = list_images(folder_a)[:args.max_images]
    files_b = list_images(folder_b)[:args.max_images]
    tf = build_transform(size, [], train=False)  # resize to (size, size) -> [-1, 1]
    print(f"images: Monet={len(files_a)} photos={len(files_b)}")

    # ---- 1. translate everything
    print("translating...")
    cyc_aba = translate(g_ab, g_ba, folder_a, files_a, dir_a2b, tf, device)
    cyc_bab = translate(g_ba, g_ab, folder_b, files_b, dir_b2a, tf, device)

    # ---- 2. verify
    print("verifying predictions...")
    verify(files_a, dir_a2b, size, "A2B")
    verify(files_b, dir_b2a, size, "B2A")

    # ---- 3. metrics
    import lpips
    ext = load_inception(device)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    print("pretrained weights used for MEASUREMENT ONLY (never for generating images):")
    print("  - Inception-v3: torch-fidelity 'inception-v3-compat' (pt_inception-2015-12-05, the standard "
          "FID Inception, ImageNet-trained) for FID / KID / precision-recall / content cosine / MiFID")
    print("  - AlexNet: torchvision ImageNet-pretrained AlexNet + LPIPS v0.1 linear heads for LPIPS")
    rows = []
    for name, src, pred_dir, real in (("A2B", (folder_a, files_a), dir_a2b, (folder_b, files_b)),
                                      ("B2A", (folder_b, files_b), dir_b2a, (folder_a, files_a))):
        print(f"measuring {name}...")
        rows.append(evaluate_direction(name, src, pred_dir, real, tf, ext, lp, device, seed))
    rows[0]["cycle_L1"], rows[1]["cycle_L1"] = cyc_aba, cyc_bab  # A->B->A, B->A->B

    # ---- 4. CSVs (overall = mean of the two directions; n_* are summed)
    num = [k for k in rows[0] if k not in ("direction", "n_fake", "n_real")]
    overall = {"direction": "overall", "n_fake": sum(r["n_fake"] for r in rows),
               "n_real": sum(r["n_real"] for r in rows), **{k: float(np.mean([r[k] for r in rows])) for k in num}}
    rows.append(overall)
    for r in rows:
        r["score"] = r["MiFID"]  # competition-style score = MiFID (lower is better)

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csv(csv_dir / "submission.csv", ["direction", "FID", "MiFID", "score"],
              [{k: r[k] for k in ("direction", "FID", "MiFID", "score")} for r in rows])
    full_fields = ["direction", "n_fake", "n_real", "KID_mean", "KID_std", "precision", "recall", "LPIPS",
                   "content_cosine", "cycle_L1", "memorization_distance"]
    write_csv(csv_dir / "full_metrics_report.csv", full_fields, [{k: r[k] for k in full_fields} for r in rows])

    print(f"\nwrote {csv_dir / 'submission.csv'} and {csv_dir / 'full_metrics_report.csv'}")
    for r in rows:
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
