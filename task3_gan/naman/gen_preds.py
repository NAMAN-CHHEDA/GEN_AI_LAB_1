"""Write exactly the images the grader reads, and nothing else (no metrics).

Part3_Evaluation_Script.ipynb takes the first 300 sorted files of each prediction folder, so this
translates the first N sorted files of data/monet_jpg (A2B, Monet -> photo) and data/photo_jpg
(B2A, photo -> Monet) with the EMA generators of one checkpoint, as RGB JPG q95 at the config image
size, keeping the original file names. Reuses translate() / verify() from evaluate_local.py.

Usage (from the naman/ folder):
    python gen_preds.py --ckpt checkpoints/ema_step60000.pt --out-dir outputs/candidates/step60000 [--n-images 300]
Then score with:
    python grader_eval.py --pred-a2b <out-dir>/pred_A2B --pred-b2a <out-dir>/pred_B2A --out-csv <out-dir>/grader_submission.csv
"""
import argparse
import sys
import time
from pathlib import Path

import torch

NAMAN = Path(__file__).resolve().parent
sys.path.insert(0, str(NAMAN / "src"))
sys.path.insert(0, str(NAMAN))

from config import load_config, resolve  # noqa: E402
from data import build_transform, list_images  # noqa: E402
from evaluate_local import translate, verify  # noqa: E402  (same image-writing and checks as the full evaluation)
from models import ResnetGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ema_stepN.pt")
    ap.add_argument("--out-dir", required=True, help="folder that will receive pred_A2B/ and pred_B2A/")
    ap.add_argument("--n-images", type=int, default=300, help="first N sorted files per direction (grader reads 300)")
    ap.add_argument("--config", default="configs/run01.yaml",
                    help="the config the checkpoint was trained with (configs/run02.yaml for run02 checkpoints)")
    args = ap.parse_args()
    t0 = time.perf_counter()

    out = Path(args.out_dir).resolve()
    dir_a2b, dir_b2a = out / "pred_A2B", out / "pred_B2A"
    real_outputs = (NAMAN / "outputs").resolve()
    if out == real_outputs:
        sys.exit("ERROR: refusing to write into the real outputs/pred_* folders; choose another --out-dir.")
    for d in (dir_a2b, dir_b2a):  # never overwrite an existing candidate
        if d.is_dir() and any(d.glob("*.jp*g")):
            sys.exit(f"ERROR: {d} already contains images; not overwriting. Remove it yourself or pick another --out-dir.")

    cfg = load_config(args.config)
    size = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    ck_cfg = ckpt.get("config", cfg)  # the config the checkpoint was trained with
    if ck_cfg.get("run_name") != cfg["run_name"]:
        sys.exit(f"ERROR: {args.ckpt} was trained as '{ck_cfg.get('run_name')}' but --config is '{cfg['run_name']}' "
                 f"({args.config}). Pass the config of the run that produced the checkpoint.")
    mc = ck_cfg["model"]  # architecture from the training config, as in evaluate_local
    up = mc.get("upsample", "convtranspose")  # run01 checkpoints predate this key
    g_ab = ResnetGenerator(mc["ngf"], mc["n_blocks"], upsample=up).to(device).eval()
    g_ba = ResnetGenerator(mc["ngf"], mc["n_blocks"], upsample=up).to(device).eval()
    g_ab.load_state_dict(ckpt["G_AB"])
    g_ba.load_state_dict(ckpt["G_BA"])
    print(f"loaded EMA generators from {args.ckpt} (step {ckpt.get('step')}), image size {size}")

    folder_a, folder_b = resolve(cfg["paths"]["monet_dir"]), resolve(cfg["paths"]["photo_dir"])
    files_a = list_images(folder_a)[:args.n_images]  # list_images is sorted by file name
    files_b = list_images(folder_b)[:args.n_images]
    tf = build_transform(size, [], train=False)
    translate(g_ab, g_ba, folder_a, files_a, dir_a2b, tf, device)
    translate(g_ba, g_ab, folder_b, files_b, dir_b2a, tf, device)
    verify(files_a, dir_a2b, size, "A2B")
    verify(files_b, dir_b2a, size, "B2A")
    print(f"wrote {len(files_a)} + {len(files_b)} images under {out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
