# Task 3 — CycleGAN (Sarvesh)

Unpaired Monet ↔ Photo style transfer for the class Kaggle competition.

## Layout

```text
sarvesh/
├── src/                      # notebooks + helper scripts
│   ├── cyclegan_final.ipynb  # main training / inference notebook
│   ├── cyclegan_task3.ipynb  # earlier full pipeline notebook
│   ├── Part3_Evaluation_Script.ipynb
│   ├── generate_pred_A2B.py
│   └── part3_path_config_snippet.py
├── checkpoints/              # final + generator weights
├── outputs/
│   ├── pred_A2B/             # Monet -> Photo (300)
│   ├── pred_B2A/             # Photo -> Monet (7038, Kaggle submit set)
│   ├── human_audit/          # 30 panels + rater_1/2 + comparison + agreement
│   ├── preview/
│   ├── metrics/
│   └── images.zip
├── logs/                     # epoch history + stability summary
├── data_processed/           # dataset split JSON
├── submission.csv            # Kaggle CSV (FID / MiFID)
├── full_metrics_report.csv   # full local metrics
├── metrics_report.csv
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
