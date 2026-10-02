"""Unpaired data loading, fixed held-out split and manifest.

Domain A = Monet, domain B = photo. The two domains are never indexed together:
each has its own dataset and its own DataLoader.
"""
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config import require, resolve

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    """Sorted image filenames in a folder (sorted so the split is deterministic)."""
    return sorted(p.name for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def _n_holdout(holdout, n_total):
    """holdout may be an int count or a float fraction in (0,1)."""
    n = round(holdout * n_total) if isinstance(holdout, float) and holdout < 1 else int(holdout)
    if not 0 < n < n_total:
        raise ValueError(f"Held-out size {n} invalid for {n_total} images.")
    return n


def make_split(files, holdout, seed):
    """Seeded shuffle of one domain's files -> {'train': [...], 'heldout': [...]}."""
    files = sorted(files)
    rng = random.Random(seed)  # local RNG: does not touch global random state
    rng.shuffle(files)
    n = _n_holdout(holdout, len(files))
    return {"heldout": sorted(files[:n]), "train": sorted(files[n:])}


def build_manifest(cfg):
    """Create the split for both domains, or load it if the manifest already exists.

    An existing manifest is never regenerated: the held-out set stays fixed.
    """
    d = cfg["data"]
    manifest_path = resolve(cfg["paths"]["manifest_dir"]) / f"split_{cfg['run_name']}.json"
    if manifest_path.exists():
        return load_manifest(manifest_path), manifest_path

    seed = require(d, "split_seed")
    files_a = list_images(resolve(cfg["paths"]["monet_dir"]))
    files_b = list_images(resolve(cfg["paths"]["photo_dir"]))
    manifest = {
        "split_seed": seed,
        "A": {"dir": cfg["paths"]["monet_dir"], **make_split(files_a, require(d, "holdout_a"), seed)},
        "B": {"dir": cfg["paths"]["photo_dir"], **make_split(files_b, require(d, "holdout_b"), seed)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest, manifest_path


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_transform(image_size, augment, load_size=None, train=True):
    """Resize -> (optional augmentation, train only) -> tensor in [-1, 1]."""
    ops = []
    if train and "random_crop" in augment:
        ops += [T.Resize(load_size), T.RandomCrop(image_size)]
    else:
        ops += [T.Resize((image_size, image_size))]
    if train and "hflip" in augment:
        ops.append(T.RandomHorizontalFlip())
    unknown = set(augment) - {"hflip", "random_crop"}
    if unknown:
        raise ValueError(f"Unknown augmentations: {unknown}")
    # ToTensor gives [0,1]; Normalize(0.5, 0.5) maps to [-1,1].
    ops += [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]
    return T.Compose(ops)


class ImageFolderDataset(Dataset):
    """One domain: a list of filenames in a directory -> normalized image tensors."""

    def __init__(self, folder, files, transform):
        self.folder, self.files, self.transform = Path(folder), files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.folder / self.files[i]).convert("RGB")
        return self.transform(img)


def make_dataloaders(cfg, manifest, subset="train", limit=None):
    """Return (loader_A, loader_B), independent. subset is 'train' or 'heldout'.

    limit: optional max images per domain (used by the smoke test).
    """
    d = cfg["data"]
    train = subset == "train"
    tf = build_transform(require(d, "image_size"), require(d, "augment"), d.get("load_size"), train)
    loaders = []
    for dom in ("A", "B"):
        files = manifest[dom][subset][:limit]
        ds = ImageFolderDataset(resolve(manifest[dom]["dir"]), files, tf)
        loaders.append(DataLoader(ds, batch_size=require(d, "batch_size"), shuffle=train,
                                  num_workers=require(d, "num_workers"), drop_last=train))
    return tuple(loaders)
