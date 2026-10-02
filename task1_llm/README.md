# Task 1: Character-level GPT on TinyStories (Naman)

A decoder-only Transformer written by hand in PyTorch (only `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout` are used; the attention math is written out explicitly).
Everything runs from one notebook, driven by one config file.

## Layout

```text
task1_llm/
  data/                          shared raw TinyStories (downloaded on first run)
  naman/
    src/
      task1_gpt.ipynb            the notebook
      config.yaml                every hyperparameter and path
      config_smoke.yaml          tiny config for the smoke test
    data_processed/              encoded train/val ids, vocab, split info
    checkpoints/                 <run_name>_best.pt, _final.pt, _resume.pt
    outputs/                     loss curves, generated samples
    metrics_report.csv           every required metric
  reproducibility/
    manifests/                   library versions, config copy, run summary, checkpoint map
    raw_logs/                    unedited per-step and per-epoch training logs
```

`failure_analysis.md` and `results.md` are written by hand from the real outputs.

## Requirements

Python 3.9+, `torch`, `numpy`, `matplotlib`, `pyyaml`, plus `jupyter` / `nbconvert` to run
headless. Device selection is automatic (CPU, Colab T4 or a lab GPU). fp16 autocast with a
`GradScaler` is used only when CUDA is available; bf16 is never used.

## Smoke test (one command)

Run from `naman/src`. It trains 1 epoch of 4 batches on a tiny model, then runs the sanity
tests, evaluation and generation, and writes everything to `naman/smoke/`.

```bash
cd naman/src
TASK1_CONFIG=config_smoke.yaml jupyter nbconvert --to notebook --execute task1_gpt.ipynb --output smoke_run.ipynb --ExecutePreprocessor.timeout=1800
```

PowerShell equivalent:

```powershell
cd naman/src
$env:TASK1_CONFIG = "config_smoke.yaml"; jupyter nbconvert --to notebook --execute task1_gpt.ipynb --output smoke_run.ipynb --ExecutePreprocessor.timeout=1800
```

The first run downloads the raw TinyStories validation file into `data/`.

## Full run

Open `task1_gpt.ipynb` in Jupyter / Colab and run all cells. The default config is
`config.yaml`; point `TASK1_CONFIG` at another file to use a different one.

- Split unit: set `data.split_unit` to `chars` (default) or `sequences`.
- Resume: set `train.resume: true` to continue from `<run_name>_resume.pt`.
- Back up `checkpoints/`, `outputs/` and `reproducibility/raw_logs/` after every run.
