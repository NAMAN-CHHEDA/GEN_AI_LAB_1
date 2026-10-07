# ============================================================
# STEP 1 — Generate Monet -> Photo  (pred_A2B)
# Paste this into a NEW cell in cyclegan_final.ipynb (or run as script)
# and run it AFTER the model/checkpoint cells have been defined,
# OR use the standalone version below which is self-contained.
# ============================================================

from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import shutil

# -------------------- paths --------------------
PROJECT_ROOT = Path(r".")
MONET_DIR = PROJECT_ROOT / "dataset" / "dataset" / "monet_jpg"
PRED_A2B_DIR = PROJECT_ROOT / "outputs" / "pred_A2B"
CKPT = PROJECT_ROOT / "checkpoints" / "retrain_full_v1" / "cyclegan_final.pth"
# fallback if full ckpt missing:
CKPT_G = PROJECT_ROOT / "checkpoints" / "retrain_full_v1" / "G_A2B_final.pth"

IMAGE_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Monet dir:", MONET_DIR, "exists:", MONET_DIR.is_dir())
print("Checkpoint:", CKPT if CKPT.is_file() else CKPT_G)

# -------------------- model (same as training) --------------------
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(in_ch))
        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7, 1, 0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -------------------- load weights --------------------
G_A2B = ResNetGenerator(num_residual_blocks=9).to(device)

ckpt_path = CKPT if CKPT.is_file() else CKPT_G
assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

try:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
except TypeError:
    obj = torch.load(ckpt_path, map_location=device)

if isinstance(obj, dict) and "G_A2B" in obj:
    G_A2B.load_state_dict(obj["G_A2B"])
    print("Loaded G_A2B from full checkpoint, epoch=", obj.get("epoch"))
else:
    G_A2B.load_state_dict(obj)
    print("Loaded raw G_A2B_final.pth")

G_A2B.eval()

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# -------------------- generate --------------------
if PRED_A2B_DIR.exists():
    shutil.rmtree(PRED_A2B_DIR)
PRED_A2B_DIR.mkdir(parents=True, exist_ok=True)

monet_paths = sorted(list(MONET_DIR.glob("*.jpg")) + list(MONET_DIR.glob("*.jpeg")))
print(f"Generating Monet -> Photo for {len(monet_paths)} images...")
assert len(monet_paths) > 0, "No Monet images found"

with torch.no_grad():
    for path in tqdm(monet_paths, desc="Monet -> Photo"):
        with Image.open(path) as img:
            x = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
        y = G_A2B(x)
        out = (y.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        save_image(out, PRED_A2B_DIR / path.name)

n = len(list(PRED_A2B_DIR.glob("*.jpg")))
print("DONE. Saved:", n, "images to", PRED_A2B_DIR)
print("Next: run Part3_Evaluation_Script.ipynb with paths pointed to pred_A2B / pred_B2A")
