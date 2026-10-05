"""Local evaluation helper for Sarvesh Task 3 CycleGAN.

Preferred full evaluation path: open `src/Part3_Evaluation_Script.ipynb`
after placing shared raw images in `task3_gan/data/monet_jpg` and `photo_jpg`.

This script only verifies that prediction folders and checkpoints exist.
"""

from pathlib import Path

MEMBER = Path(__file__).resolve().parent


def main() -> None:
    checks = {
        "checkpoints/cyclegan_final.pth": MEMBER / "checkpoints" / "cyclegan_final.pth",
        "checkpoints/G_A2B_final.pth": MEMBER / "checkpoints" / "G_A2B_final.pth",
        "checkpoints/G_B2A_final.pth": MEMBER / "checkpoints" / "G_B2A_final.pth",
        "outputs/pred_A2B": MEMBER / "outputs" / "pred_A2B",
        "outputs/pred_B2A": MEMBER / "outputs" / "pred_B2A",
        "submission.csv": MEMBER / "submission.csv",
        "full_metrics_report.csv": MEMBER / "full_metrics_report.csv",
    }
    ok = True
    for name, path in checks.items():
        if path.is_dir():
            n = len(list(path.glob("*.jpg")))
            print(f"OK  {name} ({n} jpg)")
            if n == 0:
                ok = False
        elif path.is_file():
            print(f"OK  {name} ({path.stat().st_size} bytes)")
        else:
            print(f"MISS {name}")
            ok = False
    print("PASS" if ok else "INCOMPLETE")
    print("For FID/KID/LPIPS recompute, run src/Part3_Evaluation_Script.ipynb with shared data/.")


if __name__ == "__main__":
    main()
