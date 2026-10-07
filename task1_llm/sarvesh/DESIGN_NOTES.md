# Design notes — what I kept vs changed (vs the inspected reference)

This project was built after inspecting another complete Task 1 solution, then
**re-implementing from requirements** with a deliberately different architecture.
No code was copied from that notebook.

## What that solution did well (inspirations, not copies)

1. **Config-driven runs** — one YAML for every hyperparameter; easy ablations.
2. **Sanity checks before training** — shape probe, init loss ≈ ln(V), causal-mask leak test, overfit-one-batch.
3. **Story-aware leak prevention** — train/val must not share stories.
4. **Full metrics table** — every assignment metric logged to CSV.
5. **Reproducibility** — seeded RNG, raw step/epoch logs, checkpoints, manifests.
6. **Grounded failure analysis** — snippets + counts of phrases in the training data.
7. **fp16 GradScaler on CUDA** — large speedups when a GPU is available.

## What could be improved (and what we changed)

1. **Underfitting at dropout 0.2** — their ablation showed dropout 0 hurt less than expected; we use 0.1.
2. **Still improving at last epoch** — train a bit longer or stop on plateau; we default to 12 epochs with a clear best-ckpt.
3. **Greedy repetition loops** — add nucleus sampling (`top_p`) so decoding options are richer.
4. **Monolithic notebook** — logic lives in a `charlm/` package; the notebook orchestrates.
5. **Contiguous chunk split only** — we use seeded **story-shuffle** pools instead.
6. **Fused QKV + GELU + 128/4/4** — we use separate Q/K/V, ReLU FFN, 192/5/6, weight tying, emb scaling.

## Helped him most (do these yourself)

- Run sanity tests before the long train.
- Log every required metric automatically.
- Keep failure analysis honest and evidence-based.
- Backup checkpoints / outputs after each full run.
- Use a GPU + CUDA torch build if available (CPU torch is fine for smoke tests only).
