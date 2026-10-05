# Results — Sarvesh Task 1 (run `sarvesh_main`)

Numbers from `metrics_report.csv`, `logs/sarvesh_main_summary.json`, and
`data_processed/split_info.json`.

## Setup

- **Model:** character-level decoder-only Transformer (`CharGPT` in `charlm/model.py`)
- **Data:** TinyStoriesV2-GPT4-valid.txt (27,630 stories)
- **Split:** story-shuffle, seed = 3141 (9,946 train stories / 1,999 val stories, overlap = 0)
- **Split sizes:** 100,000 train windows / 10,000 val windows (block_size=160, train stride=80)
- **Vocab size:** 85
- **Parameters:** 2,267,904
- **Device / precision:** NVIDIA RTX 4090, fp16 autocast + GradScaler

## Architecture

| Item | Value |
|---|---|
| Block type | Pre-LN, separate Q/K/V, causal `triu` mask |
| d_model | 192 |
| Layers | 5 |
| Heads | 6 |
| d_ff | 768 |
| Activation | ReLU |
| Dropout | 0.1 |
| Block size | 160 |
| Tied embeddings | true |
| Emb scale | sqrt(d_model) |
| Epochs | 12 |
| Batch size | 48 |
| Optimizer | AdamW, betas (0.9, 0.999), wd 0.05, clip 1.0 |
| LR | peak 8e-4, warmup 800 steps, cosine → 8e-5 |

## Metrics

| Metric | Value |
|---|---|
| Training cross-entropy (eval mode) | 0.7040 |
| Validation cross-entropy | 0.7392 |
| Perplexity | 2.094 |
| Bits-per-character | 1.066 |
| Generalization gap (val − train) | 0.0352 |
| Top-1 next-character accuracy (val) | 0.767 |
| Top-1 next-character accuracy (train) | 0.776 |
| Parameter count | 2,267,904 |
| Train tokens/sec | ~573,183 |
| Generation tokens/sec (greedy) | ~421 |
| Peak GPU memory (MB) | 598.6 |
| Total training time (s) | 369.9 (~6.2 min) |
| Max grad norm | 3.20 |
| Nonfinite steps | 8 (fp16 GradScaler overflows; losses stayed finite) |
| Best epoch | 12 |
| Vocab size | 85 |

### Generation diversity (from `metrics_report.csv`)

| Setting | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate |
|---|---|---|---|---|
| greedy | 0.183 | 0.410 | 0.547 | 0.277 |
| temp_0.8 | 0.329 | 0.744 | 0.911 | 0.019 |
| temp_1.0 | 0.415 | 0.813 | 0.956 | 0.000 |
| temp_1.0_topp0.9 | 0.322 | 0.720 | 0.892 | 0.052 |
| temp_0.9_topk50 | 0.393 | 0.804 | 0.935 | 0.008 |

Loss curves: `outputs/sarvesh_main_loss_curves.png`

## Observations

- Validation loss fell every epoch (1.066 → 0.739); best checkpoint is epoch 12, so the model was still improving.
- Generalization gap is small (~0.035), so overfitting is not the main issue at this size/schedule.
- Greedy decoding is fluent but repetitive (repeated 4-gram rate 0.277); sampling sharply reduces repetition.
- Higher temperature improves diversity but introduces spelling/grammar breaks typical of character LMs.
- 8 nonfinite gradient steps are consistent with fp16 GradScaler overflow skips, not NaN losses.
