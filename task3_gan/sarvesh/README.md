# Task 3 — CycleGAN (Sarvesh)

Unpaired Monet ↔ Photo style transfer for the class Kaggle competition.

## Latest run (V3)

| Metric | Value |
|--------|------:|
| **FID** (avg both directions) | **94.494** |
| **MiFID** (avg both directions) | **0.411** |
| Training | 150 epochs, batch 12, λ_identity=0.5, spectral-norm D |
| Device | NVIDIA GeForce RTX 4090 |
| Wall time | ~12.6 hours (755.8 min) |
| **Kaggle rank** | **11** (recorded Tue Oct 6, 2026 ~10:44 AM) |
| **Kaggle score** | **47.4524** |

Upload **`submission.csv`** to the class Kaggle competition.

## Layout

```text
sarvesh/
├── src/                         # notebooks + helpers
│   ├── cyclegan_4090_v3.ipynb   # ★ main V3 training notebook
│   ├── cyclegan_4090_v3.py      # same as CLI script
│   ├── Part3_Evaluation_Script.ipynb
│   ├── cyclegan_final.ipynb     # earlier V1 notebook (kept)
│   └── cyclegan_task3.ipynb
├── checkpoints/
│   ├── G_A2B_final.pth          # Monet→Photo generator (committed)
│   ├── G_B2A_final.pth          # Photo→Monet generator (committed)
│   └── checkpoint_source_v3.json
├── outputs/
│   ├── pred_A2B/                # Monet → Photo (300)
│   ├── pred_B2A/                # Photo → Monet (7038)
│   ├── loss_curves_v3.png
│   ├── human_audit/             # 30-panel audit (from prior pass)
│   └── preview/ / metrics/
├── logs/
│   ├── training_history_v3.csv
│   └── training_summary_v3.log
├── submission.csv               # Kaggle CSV (FID / MiFID)
├── full_metrics_report.csv      # extended local metrics
├── evaluate_local.py
├── results.md
└── failure_analysis.md
```

## Smoke check

```bash
cd task3_gan/sarvesh
python evaluate_local.py
```

Shared raw images belong in `task3_gan/data/monet_jpg` and `task3_gan/data/photo_jpg` (not committed).

## GitHub size notes

- Full `cyclegan_best.pth` / `cyclegan_final_v3.pth` (~324 MB each) stay **local only** (over GitHub’s 100 MB limit).
- Committed weights are **generator-only** (`G_A2B_final.pth`, `G_B2A_final.pth`, ~43 MB each), extracted from the V3 best checkpoint.
