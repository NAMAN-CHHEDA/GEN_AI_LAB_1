# Results — Sarvesh Task 3 (CycleGAN Monet ↔ Photo)

## Setup

- **Model:** CycleGAN with two generators (`G_A2B`, `G_B2A`) and two discriminators (`D_A`, `D_B`)
- **Domains:** Monet (A, 300 images) ↔ Photo (B, 7038 images)
- **Image size:** 256×256
- **Training:** 200 epochs completed (config target 250), 13,400 optimization steps
- **Losses:** adversarial + cycle-consistency + identity
- **Device:** CUDA (NVIDIA GeForce RTX 4090 on original training machine), peak memory ~4.89 GB
- **Parameters:** 28,275,336

## Checkpoints → results

| Checkpoint | Path | Used for |
|---|---|---|
| Final full state | `checkpoints/cyclegan_final.pth` (local only; >100MB, not on GitHub) | resume / full model state |
| Epoch 250 snapshot | `checkpoints/cyclegan_epoch_250.pth` (local only; >100MB, not on GitHub) | late-training snapshot |
| Generator A→B | `checkpoints/G_A2B_final.pth` (committed) | `outputs/pred_A2B/` (Monet→Photo) |
| Generator B→A | `checkpoints/G_B2A_final.pth` (committed) | `outputs/pred_B2A/` + Kaggle images (Photo→Monet) |

## Quantitative metrics (both directions)

From `full_metrics_report.csv` / `metrics_report.csv`:

| Metric | Monet → Photo | Photo → Monet |
|---|---:|---:|
| FID | 199.27 | 220.99 |
| KID | 0.0301 | 0.0291 |
| Generative precision | 0.633 | 0.533 |
| Generative recall | 0.533 | 0.600 |
| Cycle L1 | 0.146 | 0.113 |
| Translation LPIPS | 0.282 | 0.360 |
| Cycle LPIPS | 0.370 | 0.270 |
| Content cosine | 0.690 | 0.645 |

Training / systems metrics:
- Mean throughput ≈ **50.5 images/sec**
- Training time ≈ **36 minutes** (stability summary ~35.4 min)
- Non-finite loss/grad steps: **0**
- Final losses (approx): G 3.67, D_A 0.118, D_B 0.152, cycle A/B 0.077/0.098, identity A/B 0.081/0.096

Loss curves / history: `logs/final_training_history_epochs.csv`  
Preview grid: `outputs/preview/preview_grid.png`

## Kaggle submission

- `submission.csv` local Part3 values: **FID ≈ 104.004**, **MiFID ≈ 0.412**
- **Kaggle submission score (confirmed on leaderboard): `-52.2081`**
- Submit artifact: `outputs/images.zip` (Photo→Monet translations)
- **Still record:** public/private distinction (if shown) and exact **leaderboard rank** for the team report bonus table.

## Human audit (30 fixed samples, 2 raters)

Artifacts in `outputs/human_audit/`:
- `comparison_panels/` — 30 blinded sample images (`sample_01.jpg` … `sample_30.jpg`)
- `rater_1.csv`, `rater_2.csv` — style / content / artifact scores (1–5)
- `rater_comparison.csv` — side-by-side merge
- `inter_rater_agreement.json` — agreement summary

| Axis | Mean R1 | Mean R2 | Exact agree | Within ±1 | Cohen’s κ |
|---|---:|---:|---:|---:|---:|
| Style | 3.97 | 3.23 | 26.7% | 73.3% | -0.059 |
| Content | 4.10 | 4.17 | 50.0% | 90.0% | 0.185 |
| Artifacts | 4.33 | 3.30 | 26.7% | 56.7% | 0.073 |
| **Overall** | **4.13** | **3.57** | **34.4%** | — | — |

Content preservation shows the strongest agreement (50% exact, 90% within 1). Style and artifact axes disagree more often — Rater 1 scores higher on average — so report both means and κ rather than a single “quality” number.

## Observations

- Training was stable (no NaN losses/grads) with falling adversarial + cycle terms in the epoch log.
- Photo→Monet is the competition direction (7038 outputs); Monet→Photo is also generated for both-direction metrics.
- FID/KID remain high in absolute terms on the local 30-sample probe — report leaderboard score as the official competition metric.
