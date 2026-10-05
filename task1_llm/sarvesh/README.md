# Task 1: Character-level GPT on TinyStories (Sarvesh)

From-scratch decoder-only Transformer in PyTorch. Attention, causal masking, and the
feed-forward block are written by hand — no `nn.Transformer`, `nn.MultiheadAttention`,
or `scaled_dot_product_attention`.

## Layout

```text
task1_sarvesh/
  data/                         TinyStories raw text
  src/
    task1_gpt.ipynb             main notebook
    config.yaml                 full-run hyperparameters
    config_smoke.yaml           tiny pipeline check
    charlm/                     Python package (model, data, train, metrics, ...)
  data_processed/               vocab, ids, split info
  checkpoints/                  best / final / resume
  outputs/                      loss curves + generated samples
  logs/                         step/epoch CSVs + run summary
  metrics_report.csv            all required evaluation metrics
  results.md                    written after the full run
  failure_analysis.md           three generation failure cases
```

## How this differs from a typical fused-QKV / GELU baseline

| Choice | This project |
|---|---|
| Attention projections | Separate `W_q`, `W_k`, `W_v` |
| Causal mask | `torch.triu` future mask |
| FFN activation | ReLU (Vaswani et al.) |
| Embeddings | `sqrt(d_model)` scale + weight tying |
| Width / depth | d_model=192, 5 layers, 6 heads, d_ff=768 |
| Context | block_size=160, train stride=80 |
| Split | seeded story-shuffle (disjoint stories) |
| LR schedule | fixed warm-up steps + cosine → absolute `min_lr` |
| Decoding | greedy, temperature, top-k, **nucleus top-p** |

## Setup

```powershell
cd <REDACTED_PERSONAL_PATH>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If you have an NVIDIA GPU, install the matching CUDA torch build from pytorch.org
```

Set your personal seed in the first notebook cell (`SEED = ...`). If the course gave you a
seed, use that. The notebook overwrites `config.yaml`'s seed with the notebook value.

## Smoke test

```powershell
cd src
$env:TASK1_CONFIG = "config_smoke.yaml"
jupyter nbconvert --to notebook --execute task1_gpt.ipynb --output smoke_run.ipynb --ExecutePreprocessor.timeout=1800
```

## Full run

Open `src/task1_gpt.ipynb` and Run All (or execute without `TASK1_CONFIG` so it uses `config.yaml`).

Minimum requirement: **≥ 10 epochs**. Default config trains **12 epochs**.

After training:

1. Check `outputs/sarvesh_main_loss_curves.png` and `metrics_report.csv`
2. Fill `failure_analysis.md` from `outputs/sarvesh_main_samples/`
3. Fill `results.md` from the metrics CSV and logs
