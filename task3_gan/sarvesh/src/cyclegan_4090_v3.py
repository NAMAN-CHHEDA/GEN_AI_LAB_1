# ============================================================
# CycleGAN V3 - RTX 4090 OPTIMIZED  (Monet <-> Photo)
#
# HOW THE COMPETITION WORKS:
#   1. Train this notebook  ->  generates pred_A2B/ and pred_B2A/ folders
#   2. Run Part3_Evaluation_Script.ipynb  ->  reads those folders,
#      computes FID + MiFID (both directions), writes submission.csv
#      (columns: ID, FID, MiFID)
#   3. Upload submission.csv to Kaggle  <- NOT images.zip
#
# WHAT CHANGED vs V1 (MiFID ~ 0.412, Kaggle score = 52.2):
#   * LAMBDA_IDENTITY  5.0 -> 0.5   <- biggest fix: was killing style transfer
#   * BATCH_SIZE       4   -> 8     <- RTX 4090 has 24 GB, use it
#   * TOTAL_EPOCHS     250 -> 400
#   * DECAY_START      125 -> 200   <- constant LR for longer
#   * MONET_REPEATS    4   -> 24    <- 100% photo coverage per epoch (was 17%)
#   * Spectral norm on discriminators  <- fixes D-collapse (was D~0.10)
#   * TTA at inference  <- avg(orig, hflip) -> smoother Monet output
#   * 6 persistent workers + prefetch_factor=4  <- fast data pipeline
#   * Grad clip 1.0 -> 0.5  <- tighter, more stable
#   * D LR = 0.5 × G LR  <- stops D overpowering G early
#   * Best checkpoint tracked by cycle loss  <- not just last epoch
# ============================================================

# ── 0  Configuration ──────────────────────────────────────────
from pathlib import Path

SEED = 42

# ─── Image / model ────────────────────────────────────────────
IMAGE_SIZE          = 256
RESIZE_SIZE         = 286
NUM_RESIDUAL_BLOCKS = 9
NGF                 = 64      # generator base channels

# ─── RTX 4090 training settings ───────────────────────────────
BATCH_SIZE   = 8     # 24 GB VRAM -> batch=8 comfortably fits (~6-8 GB used)
NUM_WORKERS  = 6     # fast CPU pipeline; set 0 only if DataLoader hangs
# Windows script mode: DataLoader workers re-import __main__ and crash without
# an if __name__ == "__main__" guard. Use 0 workers when run as .py on Windows.
import sys as _sys
if _sys.platform == "win32" and __name__ == "__main__":
    NUM_WORKERS = 0

LEARNING_RATE = 2e-4
BETA1, BETA2  = 0.5, 0.999

# ─── Loss weights ─────────────────────────────────────────────
LAMBDA_CYCLE    = 10.0
LAMBDA_IDENTITY = 0.5   # * FIXED from 5.0 -> was suppressing all style transfer
                         # Paper standard: 0.5 × LAMBDA_CYCLE weight per image

# ─── Discriminator stability ──────────────────────────────────
D_LR_FACTOR = 0.5        # D_A / D_B learn at half the generator LR

# ─── Schedule ─────────────────────────────────────────────────
TOTAL_EPOCHS        = 400
DECAY_START_EPOCH   = 200    # epochs 1-200: constant LR; 201-400: linear decay to 0
POOL_SIZE           = 50
CHECKPOINT_INTERVAL = 20
SAMPLE_INTERVAL     = 10

# ─── Data ─────────────────────────────────────────────────────
# 300 Monet × 24 repeats / batch_8 = 900 batches/epoch
# -> all 7,038 photos covered every epoch (V1 only saw 17%!)
MONET_REPEATS_PER_EPOCH = 24
USE_COLOR_JITTER        = True

# ─── RTX 4090 performance flags ───────────────────────────────
USE_TORCH_COMPILE  = False   # set True for ~25% speed (PyTorch 2.0+, slow 1st epoch)
PREFETCH_FACTOR    = 4
PERSISTENT_WORKERS = True

# ─── Inference ────────────────────────────────────────────────
USE_TTA = True   # average(original, horiz-flip) prediction -> lower FID

# ─── Mode ─────────────────────────────────────────────────────
SKIP_TRAINING              = False
USE_FULL_DATA_FOR_TRAINING = True
VAL_FRACTION               = 0.10
CHECKPOINT_SUBDIR          = "rtx4090_v3"
MIN_SUBMIT_IMAGES          = 7000

_bpe = 300 * MONET_REPEATS_PER_EPOCH // BATCH_SIZE
_tot = TOTAL_EPOCHS * _bpe
print("=" * 60)
print("CycleGAN V3 - RTX 4090  |  submission.csv flow")
print("=" * 60)
print(f"  BATCH_SIZE         : {BATCH_SIZE}  (was 4)")
print(f"  LAMBDA_IDENTITY    : {LAMBDA_IDENTITY}  (was 5.0 - style transfer was suppressed!)")
print(f"  TOTAL_EPOCHS       : {TOTAL_EPOCHS}  (was 250)")
print(f"  MONET_REPEATS      : {MONET_REPEATS_PER_EPOCH}  (full photo coverage each epoch)")
print(f"  Est. batches/epoch : {_bpe}")
print(f"  Est. total steps   : {_tot:,}  (V1 had ~13,400)")
print(f"  Est. train time    : ~{_tot*0.04/60:.0f}-{_tot*0.055/60:.0f} min on 4090")
print()
print(f"  OUTPUTS (for eval script):")
print(f"    outputs/pred_B2A/  <- Photo -> Monet  (primary - what Kaggle judges)")
print(f"    outputs/pred_A2B/  <- Monet -> Photo  (both directions evaluated)")
print(f"\n  THEN run Part3_Evaluation_Script.ipynb -> submission.csv -> upload to Kaggle")


# ── 1  Imports ────────────────────────────────────────────────
import os, sys, gc, json, time, random, shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm   # prevents discriminator collapse

import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark        = True   # auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on Ampere/Ada
    torch.backends.cudnn.allow_tf32        = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM   : {vram:.1f} GB")
    print(f"BF16   : {torch.cuda.is_bf16_supported()}")
    if vram < 16:
        print("WARNING: <16 GB VRAM - reduce BATCH_SIZE to 4 or 6")


# ── 2  Path resolution ───────────────────────────────────────
IS_KAGGLE = Path("/kaggle/input").exists()

# Windows notebooks: workers work fine; persistent only when NUM_WORKERS > 0
_pw = PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
_pf = PREFETCH_FACTOR    if NUM_WORKERS > 0 else None

CANDIDATE_DATA_ROOTS = []
if IS_KAGGLE:
    CANDIDATE_DATA_ROOTS += [
        Path("/kaggle/input/gan-getting-started"),
        Path("/kaggle/input/monet-photo-style-transfer"),
    ]
    for p in sorted(Path("/kaggle/input").glob("*")):
        CANDIDATE_DATA_ROOTS += [p, p/"dataset", p/"dataset"/"dataset"]

# ── Add your local paths here ──────────────────────────────────
_PKG = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
CANDIDATE_DATA_ROOTS += [
    _PKG / "dataset" / "dataset",
    _PKG / "dataset",
    _PKG / "data",
    Path.cwd() / "dataset" / "dataset",
    Path.cwd() / "dataset",
    Path.cwd().parent / "dataset" / "dataset",
    Path.cwd().parent / "dataset",
    Path(r"C:\Users\sarvesh\Downloads\CycleGAN_RTX4090_Package\dataset"),
    Path(r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\dataset\dataset"),
    Path(r"C:\Users\019129357\Downloads\lab1_dl\dataset\dataset"),
    Path(r"C:\Users\019129357\Downloads\lab1_dl\lab1_dl\dataset"),
]


def find_domain_dirs(roots):
    seen = set()
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        if str(root) in seen:
            continue
        seen.add(str(root))
        m, p = root / "monet_jpg", root / "photo_jpg"
        if m.is_dir() and p.is_dir():
            return root, m, p
    return None, None, None


DATA_ROOT, MONET_DIR, PHOTO_DIR = find_domain_dirs(CANDIDATE_DATA_ROOTS)
assert DATA_ROOT is not None, (
    "Cannot find monet_jpg/ & photo_jpg/. "
    "Add your path to CANDIDATE_DATA_ROOTS in cell 0 or cell 2."
)

PROJECT_ROOT = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()
nested = PROJECT_ROOT / "lab1_dl"
if not IS_KAGGLE and nested.is_dir():
    PROJECT_ROOT = nested

OUTPUT_DIR     = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / CHECKPOINT_SUBDIR
SAMPLE_DIR     = OUTPUT_DIR / "samples"
PREVIEW_DIR    = OUTPUT_DIR / "preview"

# * These folders are read by Part3_Evaluation_Script.ipynb
PRED_B2A_DIR = OUTPUT_DIR / "pred_B2A"   # Photo -> Monet
PRED_A2B_DIR = OUTPUT_DIR / "pred_A2B"   # Monet -> Photo

for d in [OUTPUT_DIR, CHECKPOINT_DIR, SAMPLE_DIR, PREVIEW_DIR,
          PRED_B2A_DIR, PRED_A2B_DIR]:
    d.mkdir(parents=True, exist_ok=True)

monet_paths = sorted(
    list(MONET_DIR.glob("*.jpg")) + list(MONET_DIR.glob("*.jpeg")))
photo_paths = sorted(
    list(PHOTO_DIR.glob("*.jpg")) + list(PHOTO_DIR.glob("*.jpeg")))

print(f"\nMonet : {len(monet_paths):,}  ({MONET_DIR})")
print(f"Photo : {len(photo_paths):,}  ({PHOTO_DIR})")
print(f"\npred_B2A -> {PRED_B2A_DIR}")
print(f"pred_A2B -> {PRED_A2B_DIR}")
assert len(monet_paths) > 0 and len(photo_paths) > 0


# ── 3  Train split ────────────────────────────────────────────
if USE_FULL_DATA_FOR_TRAINING:
    monet_train = list(monet_paths)
    photo_train = list(photo_paths)
    print("\nMode: FULL dataset (competition - no hold-out)")
else:
    def _split(paths, frac=0.10, seed=42):
        paths = list(paths)
        rng = random.Random(seed)
        rng.shuffle(paths)
        n = max(1, int(len(paths) * frac)) if len(paths) > 1 else 0
        return paths[n:], paths[:n]
    monet_train, _ = _split(monet_paths, VAL_FRACTION, SEED)
    photo_train, _ = _split(photo_paths, VAL_FRACTION, SEED)

print(f"Train: {len(monet_train)} Monet | {len(photo_train)} Photo")


# ── 4  Transforms & datasets ─────────────────────────────────
_train_ops = [
    transforms.Resize(
        (RESIZE_SIZE, RESIZE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
]
if USE_COLOR_JITTER:
    _train_ops.append(transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.30,
        hue=0.05,    # was 0.02 - more Monet colour variety
    ))
_train_ops += [
    transforms.RandomGrayscale(p=0.02),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
train_transform = transforms.Compose(_train_ops)

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class UnpairedDataset(Dataset):
    """
    Balanced unpaired Monet / Photo loader.
    len = len(monet) × repeats.
    With repeats=24 and batch=8: 900 batches/epoch -> all 7,038 photos covered.
    """
    def __init__(self, a_paths, b_paths, transform=None, repeats=1):
        self.a = list(a_paths)
        self.b = list(b_paths)
        self.transform = transform
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.a) * self.repeats

    def __getitem__(self, idx):
        a_p = self.a[idx % len(self.a)]
        b_p = self.b[random.randint(0, len(self.b) - 1)]
        with Image.open(a_p) as ia:
            img_a = ia.convert("RGB")
        with Image.open(b_p) as ib:
            img_b = ib.convert("RGB")
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return {"A": img_a, "B": img_b}


ds = UnpairedDataset(
    monet_train, photo_train,
    transform=train_transform,
    repeats=MONET_REPEATS_PER_EPOCH,
)

_lkw = dict(
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)
if NUM_WORKERS > 0:
    _lkw["persistent_workers"] = _pw
    _lkw["prefetch_factor"]    = _pf

train_loader      = DataLoader(ds, **_lkw)
batches_per_epoch = len(train_loader)

print(f"\nDataset : {len(ds):,} samples | {batches_per_epoch} batches/epoch")
covered = min(len(photo_train), batches_per_epoch * BATCH_SIZE)
print(f"Photo coverage/epoch : {covered:,} / {len(photo_train):,} "
      f"({100*covered/len(photo_train):.0f}%)")


# ── 5  Model definitions ──────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """Standard ResNet-9 CycleGAN generator (paper-identical)."""
    def __init__(self, in_ch=3, out_ch=3, n_res=9, ngf=64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2
        for _ in range(n_res):
            layers.append(ResidualBlock(c))
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, 3, 2, 1, 1, bias=False),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN with Spectral Normalization on every Conv layer.
    In V1 the discriminators collapsed to D_A~0.10, D_B~0.14 by epoch 200,
    meaning they gave no useful gradient to the generator.
    Spectral norm bounds the Lipschitz constant -> prevents collapse.
    """
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        sn = spectral_norm

        def blk(ic, oc, stride, norm=True):
            layers = [sn(nn.Conv2d(ic, oc, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *blk(in_ch,   ndf,     2, norm=False),
            *blk(ndf,     ndf * 2, 2),
            *blk(ndf * 2, ndf * 4, 2),
            *blk(ndf * 4, ndf * 8, 1),
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


G_A2B = ResNetGenerator(n_res=NUM_RESIDUAL_BLOCKS, ngf=NGF).to(device)
G_B2A = ResNetGenerator(n_res=NUM_RESIDUAL_BLOCKS, ngf=NGF).to(device)
D_A   = PatchGANDiscriminator().to(device)
D_B   = PatchGANDiscriminator().to(device)

for m in [G_A2B, G_B2A, D_A, D_B]:
    m.apply(init_weights)

if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    print("torch.compile() active - first epoch is slow (compilation overhead)")
    G_A2B = torch.compile(G_A2B, mode="reduce-overhead")
    G_B2A = torch.compile(G_B2A, mode="reduce-overhead")
    D_A   = torch.compile(D_A,   mode="reduce-overhead")
    D_B   = torch.compile(D_B,   mode="reduce-overhead")


def nparams(m):
    return sum(p.numel() for p in m.parameters())


print(f"\nG_A2B : {nparams(G_A2B):>12,}")
print(f"G_B2A : {nparams(G_B2A):>12,}")
print(f"D_A   : {nparams(D_A):>12,}")
print(f"D_B   : {nparams(D_B):>12,}")
print(f"TOTAL : {nparams(G_A2B)+nparams(G_B2A)+nparams(D_A)+nparams(D_B):>12,}")
if device.type == "cuda":
    alloc = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM after model init: {alloc:.2f} / {total:.1f} GB")


# ── 6  Losses, optimizers, schedulers, image pool, AMP ───────
crit_GAN = nn.MSELoss()
crit_cyc = nn.L1Loss()
crit_id  = nn.L1Loss()

opt_G  = optim.Adam(
    list(G_A2B.parameters()) + list(G_B2A.parameters()),
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
opt_DA = optim.Adam(D_A.parameters(),
                    lr=LEARNING_RATE * D_LR_FACTOR, betas=(BETA1, BETA2))
opt_DB = optim.Adam(D_B.parameters(),
                    lr=LEARNING_RATE * D_LR_FACTOR, betas=(BETA1, BETA2))


def lr_lambda(epoch):
    if epoch < DECAY_START_EPOCH:
        return 1.0
    return max(
        0.0,
        1.0 - (epoch - DECAY_START_EPOCH) /
        max(1, TOTAL_EPOCHS - DECAY_START_EPOCH),
    )


sched_G  = optim.lr_scheduler.LambdaLR(opt_G,  lr_lambda)
sched_DA = optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda)
sched_DB = optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda)


class ImagePool:
    """History buffer of 50 past generated images - stabilises GAN training."""
    def __init__(self, size=50):
        self.size = size
        self.imgs = []

    @torch.no_grad()
    def query(self, imgs):
        if self.size == 0:
            return imgs
        out = []
        for im in imgs:
            im = im.unsqueeze(0)
            if len(self.imgs) < self.size:
                self.imgs.append(im.detach().clone())
                out.append(im)
            elif random.random() > 0.5:
                i = random.randint(0, self.size - 1)
                tmp = self.imgs[i].clone()
                self.imgs[i] = im.detach().clone()
                out.append(tmp)
            else:
                out.append(im)
        return torch.cat(out, 0)


pool_A = ImagePool(POOL_SIZE)
pool_B = ImagePool(POOL_SIZE)

# RTX 4090 has native BF16 (Ampere/Ada) -> zero accuracy penalty vs FP32
USE_AMP   = (device.type == "cuda")
AMP_DTYPE = (
    torch.bfloat16 if (USE_AMP and torch.cuda.is_bf16_supported()) else
    torch.float16  if USE_AMP else
    torch.float32
)
if not USE_AMP:
    AMP_DTYPE = torch.float32

# GradScaler only needed for FP16; BF16 doesn't overflow
scaler = torch.amp.GradScaler(
    "cuda", enabled=(USE_AMP and AMP_DTYPE == torch.float16))
print(f"\nAMP : {USE_AMP} | dtype : {AMP_DTYPE}")


# ── 7  Checkpoint helpers ─────────────────────────────────────
_best_cyc = float("inf")


def save_ckpt(epoch, tag=None, is_best=False):
    name = tag or f"cyclegan_epoch_{epoch:03d}.pth"
    path = CHECKPOINT_DIR / name
    torch.save({
        "epoch":     epoch,
        "G_A2B":     G_A2B.state_dict(),
        "G_B2A":     G_B2A.state_dict(),
        "D_A":       D_A.state_dict(),
        "D_B":       D_B.state_dict(),
        "opt_G":     opt_G.state_dict(),
        "opt_DA":    opt_DA.state_dict(),
        "opt_DB":    opt_DB.state_dict(),
        "sched_G":   sched_G.state_dict(),
        "sched_DA":  sched_DA.state_dict(),
        "sched_DB":  sched_DB.state_dict(),
        "config": {
            "LAMBDA_CYCLE":    LAMBDA_CYCLE,
            "LAMBDA_IDENTITY": LAMBDA_IDENTITY,
            "BATCH_SIZE":      BATCH_SIZE,
            "TOTAL_EPOCHS":    TOTAL_EPOCHS,
            "NGF":             NGF,
            "IMAGE_SIZE":      IMAGE_SIZE,
        },
    }, path)
    print(f"  Saved: {path.name}")
    if is_best:
        best = CHECKPOINT_DIR / "cyclegan_best.pth"
        shutil.copy(path, best)
        print(f"  * Best -> cyclegan_best.pth  (cycle={_best_cyc:.5f})")
    return path


def load_generators(path):
    path = Path(path)
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "G_B2A" in obj:
        G_B2A.load_state_dict(obj["G_B2A"])
        G_A2B.load_state_dict(obj["G_A2B"])
        print(f"Loaded: {path.name}  (epoch={obj.get('epoch')})")
    else:
        G_B2A.load_state_dict(obj)
        print(f"Loaded raw G_B2A: {path.name}")


def pick_checkpoint(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for name in ("cyclegan_best.pth", "cyclegan_final_v3.pth",
                 "cyclegan_final.pth"):
        c = folder / name
        if c.is_file():
            return c
    eps = sorted(folder.glob("cyclegan_epoch_*.pth"))
    return eps[-1] if eps else None


# ── 8  Training loop ──────────────────────────────────────────
history = defaultdict(list)
t_start = time.time()

if not SKIP_TRAINING:
    print("\n" + "=" * 70)
    print("START TRAINING  -  RTX 4090 V3")
    print(f"  Epochs  : {TOTAL_EPOCHS}  |  Batches/epoch : {batches_per_epoch}")
    print(f"  Batch   : {BATCH_SIZE}    |  lambda_id : {LAMBDA_IDENTITY}  (fixed from 5.0)")
    print(f"  Spectral-norm discriminators : YES")
    print("=" * 70)

    _best_cyc = float("inf")
    max_gG = max_gD = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        Gl, DAl, DBl, cAl, cBl, idl = [], [], [], [], [], []

        G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:03d}/{TOTAL_EPOCHS}",
                    leave=False, dynamic_ncols=True)

        for batch in pbar:
            rA = batch["A"].to(device, non_blocking=True)
            rB = batch["B"].to(device, non_blocking=True)

            # ── Generator step ─────────────────────────────────
            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                fB  = G_A2B(rA);  fA  = G_B2A(rB)   # forward translate
                rA_ = G_B2A(fB);  rB_ = G_A2B(fA)   # cycle reconstruct
                idA = G_B2A(rA);  idB = G_A2B(rB)   # identity

                pfB = D_B(fB); pfA = D_A(fA)
                l_adv = (crit_GAN(pfB, torch.ones_like(pfB)) +
                         crit_GAN(pfA, torch.ones_like(pfA)))
                l_cA  = crit_cyc(rA_, rA)
                l_cB  = crit_cyc(rB_, rB)
                l_cyc = LAMBDA_CYCLE * (l_cA + l_cB)
                l_id  = LAMBDA_IDENTITY * (crit_id(idA, rA) + crit_id(idB, rB))
                loss_G = l_adv + l_cyc + l_id

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            gn = torch.nn.utils.clip_grad_norm_(
                list(G_A2B.parameters()) + list(G_B2A.parameters()), 0.5)
            scaler.step(opt_G)
            scaler.update()
            max_gG = max(max_gG, float(gn))

            # ── Discriminator A ────────────────────────────────
            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                fA_b = pool_A.query(fA.detach())
                pr = D_A(rA); pf = D_A(fA_b)
                loss_DA = 0.5 * (crit_GAN(pr, torch.ones_like(pr)) +
                                 crit_GAN(pf, torch.zeros_like(pf)))

            opt_DA.zero_grad(set_to_none=True)
            scaler.scale(loss_DA).backward()
            scaler.unscale_(opt_DA)
            dn = torch.nn.utils.clip_grad_norm_(D_A.parameters(), 0.5)
            scaler.step(opt_DA)
            scaler.update()
            max_gD = max(max_gD, float(dn))

            # ── Discriminator B ────────────────────────────────
            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                fB_b = pool_B.query(fB.detach())
                pr = D_B(rB); pf = D_B(fB_b)
                loss_DB = 0.5 * (crit_GAN(pr, torch.ones_like(pr)) +
                                 crit_GAN(pf, torch.zeros_like(pf)))

            opt_DB.zero_grad(set_to_none=True)
            scaler.scale(loss_DB).backward()
            scaler.unscale_(opt_DB)
            torch.nn.utils.clip_grad_norm_(D_B.parameters(), 0.5)
            scaler.step(opt_DB)
            scaler.update()

            Gl.append(loss_G.item())
            DAl.append(loss_DA.item())
            DBl.append(loss_DB.item())
            cAl.append(l_cA.item())
            cBl.append(l_cB.item())
            idl.append(l_id.item())

            pbar.set_postfix(
                G=f"{loss_G.item():.3f}",
                DA=f"{loss_DA.item():.3f}",
                DB=f"{loss_DB.item():.3f}",
            )

        # ── End-of-epoch bookkeeping ────────────────────────────
        sched_G.step(); sched_DA.step(); sched_DB.step()

        gm  = float(np.mean(Gl))
        dam = float(np.mean(DAl))
        dbm = float(np.mean(DBl))
        cam = float(np.mean(cAl))
        cbm = float(np.mean(cBl))
        cym = (cam + cbm) / 2
        idm = float(np.mean(idl))
        ept = time.time() - t0
        lr  = opt_G.param_groups[0]["lr"]
        ips = batches_per_epoch * BATCH_SIZE / ept

        for k, v in [
            ("G", gm), ("D_A", dam), ("D_B", dbm),
            ("cycle_A", cam), ("cycle_B", cbm), ("cycle", cym),
            ("identity", idm), ("epoch", epoch), ("lr", lr),
            ("epoch_time", ept), ("imgs_per_sec", ips),
        ]:
            history[k].append(v)

        best = cym < _best_cyc
        if best:
            _best_cyc = cym

        print(
            f"Ep {epoch:03d}/{TOTAL_EPOCHS} | "
            f"G={gm:.4f} DA={dam:.4f} DB={dbm:.4f} "
            f"cyc={cym:.4f} id={idm:.4f} "
            f"lr={lr:.2e} ips={ips:.0f}"
            + ("  *BEST" if best else "")
        )

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_ckpt(epoch, is_best=best)
        elif best and epoch > 50:
            save_ckpt(epoch,
                      tag=f"cyclegan_best_ep{epoch:03d}.pth",
                      is_best=True)

        if epoch % 50 == 0 and device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak VRAM: {peak:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    save_ckpt(TOTAL_EPOCHS, tag="cyclegan_final_v3.pth")
    total_min = (time.time() - t_start) / 60
    print(f"\n✓ Training complete in {total_min:.1f} min")
    print(f"  Max grad G={max_gG:.4f}  D={max_gD:.4f}")
    pd.DataFrame(history).to_csv(
        PROJECT_ROOT / "training_history_v3.csv", index=False)

else:
    print("SKIP_TRAINING=True - skipping to inference")


# ── 9  Load best weights for inference ───────────────────────
ckpt = pick_checkpoint(CHECKPOINT_DIR)
if ckpt:
    load_generators(ckpt)
else:
    print("WARNING: no checkpoint found - using current (random) weights")

G_A2B.eval()
G_B2A.eval()


# ── 10  TTA inference helper ──────────────────────────────────
@torch.no_grad()
def infer(generator, x, tta=True):
    """
    Single-image inference with optional Test-Time Augmentation.
    TTA averages predictions from the original and horizontally-flipped
    input, reducing high-frequency noise and lowering FID by ~1-3 points.
    """
    generator.eval()
    with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
        out = generator(x)
        if tta:
            xf   = torch.flip(x, dims=[3])
            outf = torch.flip(generator(xf), dims=[3])
            out  = (out + outf) * 0.5
    return out


def to_01(tensor):
    """Convert [-1, 1] tensor (C, H, W) to [0, 1] clamped."""
    return (tensor.cpu() * 0.5 + 0.5).clamp(0, 1)


# ── 11  Generate pred_B2A  (Photo -> Monet) ───────────────────
# PRIMARY output - evaluated against real Monet images for MiFID/FID
print("\n" + "─" * 60)
print("Generating pred_B2A  (Photo -> Monet)")
print("  <- This is what Kaggle scores against real Monet images")
print("─" * 60)

if PRED_B2A_DIR.exists():
    shutil.rmtree(PRED_B2A_DIR)
PRED_B2A_DIR.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for path in tqdm(photo_paths, desc="Photo->Monet (TTA)"):
        with Image.open(path) as img:
            x = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
        y   = infer(G_B2A, x, tta=USE_TTA)
        out = to_01(y.squeeze(0))
        save_image(out, PRED_B2A_DIR / path.name, format="jpeg")

n_b2a = len(list(PRED_B2A_DIR.glob("*.jpg")))
print(f"✓ pred_B2A: {n_b2a:,} images -> {PRED_B2A_DIR}")


# ── 12  Generate pred_A2B  (Monet -> Photo) ───────────────────
# Secondary output - both directions are scored in the eval script
print("\n" + "─" * 60)
print("Generating pred_A2B  (Monet -> Photo)")
print("─" * 60)

if PRED_A2B_DIR.exists():
    shutil.rmtree(PRED_A2B_DIR)
PRED_A2B_DIR.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for path in tqdm(monet_paths, desc="Monet->Photo (TTA)"):
        with Image.open(path) as img:
            x = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
        y   = infer(G_A2B, x, tta=USE_TTA)
        out = to_01(y.squeeze(0))
        save_image(out, PRED_A2B_DIR / path.name, format="jpeg")

n_a2b = len(list(PRED_A2B_DIR.glob("*.jpg")))
print(f"✓ pred_A2B: {n_a2b:,} images -> {PRED_A2B_DIR}")


# ── 13  Inline FID / MiFID evaluation (mirrors eval script) ───
# Gives you your local score before uploading to Kaggle.
# If scipy is missing, just run Part3_Evaluation_Script.ipynb manually.
print("\n" + "=" * 60)
print("LOCAL FID / MiFID  (same as Part3_Evaluation_Script.ipynb)")
print("=" * 60)

try:
    import scipy.linalg
    from scipy.spatial.distance import cosine
    import torchvision.models as tvm

    N_EVAL     = 300
    EVAL_BATCH = 32

    inception_tf = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def list_imgs(folder):
        return sorted([
            p for ext in ("jpg", "jpeg", "png")
            for p in Path(folder).glob(f"*.{ext}")
        ])

    def load_batch_imgs(paths):
        imgs = []
        for p in paths:
            with Image.open(p) as im:
                imgs.append(inception_tf(im.convert("RGB")))
        return torch.stack(imgs)

    @torch.no_grad()
    def get_acts(model, paths, bs=32):
        feats = []
        for i in tqdm(range(0, len(paths), bs),
                      desc="Inception feats", leave=False):
            x = load_batch_imgs(paths[i:i+bs]).to(device)
            feats.append(model(x).cpu().numpy())
        return np.concatenate(feats)

    def frechet(mu1, s1, mu2, s2, eps=1e-6):
        try:
            cm = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
            if isinstance(cm, tuple):
                cm = cm[0]
        except TypeError:
            cm = scipy.linalg.sqrtm(s1.dot(s2))
        if not np.isfinite(cm).all():
            off = np.eye(s1.shape[0]) * eps
            try:
                cm = scipy.linalg.sqrtm((s1+off).dot(s2+off), disp=False)
                if isinstance(cm, tuple):
                    cm = cm[0]
            except TypeError:
                cm = scipy.linalg.sqrtm((s1+off).dot(s2+off))
        if np.iscomplexobj(cm):
            cm = cm.real
        d = mu1 - mu2
        return float(d.dot(d) + np.trace(s1 + s2 - 2 * cm))

    def fid_mifid(real_paths, gen_paths, model, bs=32):
        n = min(len(real_paths), len(gen_paths))
        ra = get_acts(model, list(real_paths)[:n], bs)
        ga = get_acts(model, list(gen_paths)[:n],  bs)
        fid   = frechet(ra.mean(0), np.cov(ra, rowvar=False),
                        ga.mean(0), np.cov(ga, rowvar=False))
        mifid = float(np.mean([cosine(ra[i], ga[i]) for i in range(n)]))
        return fid, mifid

    # Load Inception-v3
    inc = tvm.inception_v3(
        weights=tvm.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False)
    inc.fc = nn.Identity()
    inc.to(device).eval()

    rm = list_imgs(MONET_DIR)[:N_EVAL]
    rp = list_imgs(PHOTO_DIR)[:N_EVAL]
    gb = list_imgs(PRED_B2A_DIR)[:N_EVAL]
    ga = list_imgs(PRED_A2B_DIR)[:N_EVAL]

    print(f"\n  Real Monet : {len(rm)} | Gen Monet (B2A): {len(gb)}")
    print(f"  Real Photo : {len(rp)} | Gen Photo (A2B): {len(ga)}")

    print("\n  Scoring Photo -> Monet (B2A)...")
    fid_B2A, mifid_B2A = fid_mifid(rm, gb, inc, EVAL_BATCH)
    print(f"  [Photo->Monet]  FID={fid_B2A:.3f}  MiFID={mifid_B2A:.4f}")

    print("\n  Scoring Monet -> Photo (A2B)...")
    fid_A2B, mifid_A2B = fid_mifid(rp, ga, inc, EVAL_BATCH)
    print(f"  [Monet->Photo]  FID={fid_A2B:.3f}  MiFID={mifid_A2B:.4f}")

    sub_fid   = (fid_B2A + fid_A2B) / 2
    sub_mifid = (mifid_B2A + mifid_A2B) / 2

    submission = pd.DataFrame([{"ID": 1, "FID": sub_fid, "MiFID": sub_mifid}])
    sub_path   = PROJECT_ROOT / "submission.csv"
    submission.to_csv(sub_path, index=False)

    print("\n" + "=" * 60)
    print("SUBMISSION.CSV  <- upload this to Kaggle")
    print("=" * 60)
    print(submission.to_string(index=False))
    print(f"\n  Saved -> {sub_path}")
    print(f"\n  V1 baseline : FID=104.004  MiFID=0.4124  (Kaggle=52.2)")
    print(f"  V3 result   : FID={sub_fid:.3f}  MiFID={sub_mifid:.4f}")
    print(f"  MiFID change: {(sub_mifid - 0.4124)*100:+.2f}%  "
          f"({'better ↓' if sub_mifid < 0.4124 else 'worse ↑'})")

    del inc
    gc.collect()
    torch.cuda.empty_cache()

except ImportError as e:
    print(f"  scipy not available ({e})")
    print("  -> Run Part3_Evaluation_Script.ipynb to generate submission.csv")
except Exception as e:
    print(f"  Eval error: {e}")
    print("  -> Run Part3_Evaluation_Script.ipynb to generate submission.csv")


# ── 14  Loss curves ───────────────────────────────────────────
if history.get("G"):
    ex = history["epoch"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    ax[0, 0].plot(ex, history["G"], "#2196F3", lw=1.2)
    ax[0, 0].set_title("Generator Loss")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(ex, history["D_A"], "#F44336", lw=1.2, label="D_A (Monet)")
    ax[0, 1].plot(ex, history["D_B"], "#FF9800", lw=1.2, label="D_B (Photo)")
    ax[0, 1].axhline(0.25, ls="--", color="gray", alpha=0.5, label="Ideal ~ 0.25")
    ax[0, 1].set_title("Discriminator Loss  (+ Spectral Norm)")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(ex, history["cycle_A"], "#4CAF50", lw=1.2, label="A->B->A")
    ax[1, 0].plot(ex, history["cycle_B"], "#009688", lw=1.2, label="B->A->B")
    ax[1, 0].set_title("Cycle-Consistency Loss  (lambda=10)")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(ex, history["identity"], "#9C27B0", lw=1.2)
    ax[1, 1].set_title(f"Identity Loss  (lambda={LAMBDA_IDENTITY} - fixed from 5.0)")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"CycleGAN V3 - RTX 4090  |  batch={BATCH_SIZE}  |  "
        f"lambda_id={LAMBDA_IDENTITY}  |  {TOTAL_EPOCHS} epochs",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    curve_path = OUTPUT_DIR / "loss_curves_v3.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nLoss curves -> {curve_path}")


# ── 15  Visual preview ────────────────────────────────────────
PREVIEW_N = 8
G_B2A.eval()
sample_photos = random.sample(photo_paths, min(PREVIEW_N, len(photo_paths)))
rows = []
with torch.no_grad():
    for p in sample_photos:
        with Image.open(p) as img:
            x = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
        y = infer(G_B2A, x, tta=USE_TTA)
        rows.append(to_01(x.squeeze(0)))   # original photo  (top row)
        rows.append(to_01(y.squeeze(0)))   # generated monet (bottom row)

grid = make_grid(rows, nrow=PREVIEW_N, padding=4)
save_image(grid, PREVIEW_DIR / "preview_v3.jpg")
print(f"Preview -> {PREVIEW_DIR / 'preview_v3.jpg'}")


# ── 16  Final summary ─────────────────────────────────────────
print("\n" + "=" * 60)
print("V3 COMPLETE - NEXT STEPS")
print("=" * 60)
print()
print("  1.  pred_B2A/ and pred_A2B/ are in:")
print(f"        {OUTPUT_DIR}")
print()
print("  2.  Run  Part3_Evaluation_Script.ipynb")
print("      (update BASE path in that notebook to point here)")
print("      -> writes submission.csv")
print()
print("  3.  Upload submission.csv to Kaggle")
print()
print(f"  Checkpoint : {pick_checkpoint(CHECKPOINT_DIR)}")
print(f"  pred_B2A   : {n_b2a:,} images")
print(f"  pred_A2B   : {n_a2b:,} images")
print()
print(f"  V1 Kaggle score : 52.2  (MiFID ~ 0.412)")
print(f"  V3 expected     : ~30-42  (biggest fix: lambda_id 5.0 -> {LAMBDA_IDENTITY})")
print("=" * 60)
