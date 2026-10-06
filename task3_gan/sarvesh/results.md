# Results — Sarvesh Task 3 (CycleGAN Monet ↔ Photo)

## Setup (V3 — current submission)

- **Model:** CycleGAN with two generators (`G_A2B`, `G_B2A`) and two discriminators (`D_A`, `D_B`, spectral norm)
- **Domains:** Monet (A, 300 images) ↔ Photo (B, 7038 images)
- **Image size:** 256×256
- **Training:** 150 epochs, batch size 12, λ_cycle=10, λ_identity=**0.5**, D LR = 0.5×G LR, TTA at inference
- **Losses:** adversarial + cycle-consistency + identity
- **Device:** NVIDIA GeForce RTX 4090, AMP bfloat16
- **Parameters:** 28,275,336
- **Wall time:** ~755.8 min (~12.6 h)

## Checkpoints → results

| Checkpoint | Path | Used for |
|---|---|---|
| Best full state | `cyclegan_best.pth` (local only; ~324MB, not on GitHub) | source of generator weights |
| Final full state | `cyclegan_final_v3.pth` (local only) | end-of-run snapshot |
| Generator A→B | `checkpoints/G_A2B_final.pth` (committed) | `outputs/pred_A2B/` |
| Generator B→A | `checkpoints/G_B2A_final.pth` (committed) | `outputs/pred_B2A/` + Kaggle direction |

## Kaggle submission (V3)

| | V1 (previous) | **V3 (current)** |
|---|---:|---:|
| FID (avg) | 104.004 | **94.494** |
| MiFID (avg) | 0.4124 | **0.4108** |
| Est. class score (≈ FID/2) | ~52.2 | **~47.2** |

Per-direction local eval (inline Part3-style, N=300):

| Direction | FID | MiFID |
|---|---:|---:|
| Photo → Monet | 95.309 | 0.4056 |
| Monet → Photo | 93.679 | 0.4159 |

- File: **`submission.csv`**
- Predictions: `outputs/pred_B2A/` (7038), `outputs/pred_A2B/` (300)
- Loss curves: `outputs/loss_curves_v3.png`
- Epoch history: `logs/training_history_v3.csv`

### Leaderboard (recorded)

| Field | Value |
|---|---|
| **Rank** | **11** |
| **Kaggle score** | **47.4524** |
| Recorded at | Tue Oct 6, 2026 ~10:44 AM |
| Local FID / MiFID | 94.494 / 0.4108 |

Update if public/private scores are shown separately on Kaggle.

## Extended metrics note

`full_metrics_report.csv` / `metrics_report.csv` still contain the **earlier V1 probe** (FID/KID/LPIPS/precision-recall on a 30-sample subset). Re-run the full metrics notebook on V3 preds if you need those updated for the written report. The **official class CSV numbers** for this submission are the V3 FID/MiFID above.

## Human audit (30 fixed samples, 2 raters)

Artifacts in `outputs/human_audit/` (from prior audit pass — panels may need refresh if required to match V3 preds):

| Axis | Mean R1 | Mean R2 | Exact agree | Within ±1 | Cohen’s κ |
|---|---:|---:|---:|---:|---:|
| Style | 3.97 | 3.23 | 26.7% | 73.3% | -0.059 |
| Content | 4.10 | 4.17 | 50.0% | 90.0% | 0.185 |
| Artifacts | 4.33 | 3.30 | 26.7% | 56.7% | 0.073 |
| **Overall** | **4.13** | **3.57** | **34.4%** | — | — |

## Observations

- V3 main fix vs V1: λ_identity **5.0 → 0.5** (style transfer was over-suppressed), plus spectral-norm D, larger batch, full photo coverage per epoch.
- Local FID improved ~9.5 points (104 → 94.5); MiFID only slightly better.
- Training completed 150/150 epochs with best checkpoint tracked by cycle loss.
