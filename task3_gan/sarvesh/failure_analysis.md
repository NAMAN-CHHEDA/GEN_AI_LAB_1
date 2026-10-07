# Task 3 — Failure / Quality Analysis (Sarvesh)

Evidence from `outputs/preview/`, `outputs/pred_A2B/`, `outputs/pred_B2A/`, and `full_metrics_report.csv`.

## Case 1: Style under-transfer / residual photo look (Photo → Monet)

**Observation:** Some Photo→Monet outputs keep strong photographic lighting/texture and only weakly adopt Monet brushwork.  
**Type:** incomplete style transfer  
**Related metric:** higher translation LPIPS on Photo→Monet (0.360) vs Monet→Photo (0.282) suggests larger perceptual change is needed but not always stylistically “Monet-like.”  
**Testable fix:** increase identity/cycle balance sweep or train more epochs with a lower identity weight so the generator can move further from the photo domain.

## Case 2: Content drift / geometry warping

**Observation:** In a subset of translations, object boundaries soften or warp (trees/buildings less coherent) while color palette shifts.  
**Type:** content preservation failure  
**Related metric:** content cosine ~0.64–0.69 (not collapsed, but imperfect).  
**Testable fix:** strengthen cycle-consistency weight or add an explicit content/perceptual feature-matching term and re-check content cosine + cycle L1.

## Case 3: Texture artifacts / blotches

**Observation:** Some outputs show blotchy color patches or checker-like texture, especially in sky/water regions.  
**Type:** generative artifact / discriminator instability residue  
**Related metric:** max generator/discriminator gradient norms were large during training (G ~157, D_A ~199) even though no NaNs occurred.  
**Testable fix:** stronger gradient clipping, TTUR, or spectral norm on discriminators; compare artifact rate on a fixed 30-image panel.

## Human audit notes

Completed 2-rater audit is in `outputs/human_audit/` (30 panels + `rater_1.csv` / `rater_2.csv` + `rater_comparison.csv`).

- Content scores align fairly well (κ ≈ 0.19; 90% within ±1).
- Style/artifact scores show low exact agreement and near-chance κ; Rater 2 is stricter and notes grid/streak artifacts that Rater 1 often marked higher.
- This matches Case 3 above: artifacts are visible to careful raters even when overall “looks reasonable.”
