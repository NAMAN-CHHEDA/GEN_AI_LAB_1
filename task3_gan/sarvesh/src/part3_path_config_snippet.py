# ============================================================
# PART 3 — PATH CONFIG (repo-relative; no personal absolute paths)
# ============================================================
from pathlib import Path
import os
import torch

# Member folder = task3_gan/sarvesh
MEMBER = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().resolve()
if MEMBER.name == "src":
    MEMBER = MEMBER.parent
TASK = MEMBER.parent  # task3_gan/

REAL_MONET = TASK / "data" / "monet_jpg"   # shared raw Monet
REAL_PHOTO = TASK / "data" / "photo_jpg"   # shared raw Photo
GEN_A2B = MEMBER / "outputs" / "pred_A2B"  # Monet -> Photo
GEN_B2A = MEMBER / "outputs" / "pred_B2A"  # Photo -> Monet

N_EVAL = 300
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
for name, path in [
    ("REAL_MONET", REAL_MONET),
    ("REAL_PHOTO", REAL_PHOTO),
    ("GEN_A2B", GEN_A2B),
    ("GEN_B2A", GEN_B2A),
]:
    print(name + ":", path, "->", len(list(path.glob("*.jpg"))) if path.is_dir() else "MISSING")
