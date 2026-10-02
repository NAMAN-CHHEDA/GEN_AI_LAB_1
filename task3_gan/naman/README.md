# Task 3: CycleGAN (Monet <-> photo), naman

Domain A = Monet (`data/monet_jpg`), domain B = photo (`data/photo_jpg`). Unpaired.
Everything is trained from scratch. No pretrained generators or image models.

## Layout
```
task3/
  data/monet_jpg, data/photo_jpg      shared images
  reproducibility/manifests           split + run manifests
  reproducibility/raw_logs            append-only training logs (never edited)
  naman/
    configs/run01.yaml                all run settings
    src/                              config.py data.py logging_utils.py checkpoint.py manifest.py
    checkpoints/  outputs/pred_A2B  outputs/pred_B2A
    evaluate_local.py  failure_analysis.md  results.md
```

## Setup
```
pip install -r naman/requirements.txt
```
Settings left `null` in `naman/configs/run01.yaml` must be filled in before running.

## Data split
Held-out split is seeded (`data.split_seed: 8893`) and saved to
`reproducibility/manifests/split_run01.json`. An existing manifest is reused, never regenerated.

## Smoke test
TODO: one command, added once the training loop exists.

## Training / resume
TODO.

## Evaluation
TODO.

## Inference / Kaggle submission
TODO.

## Reproducibility
`reproducibility/manifests/run_<name>.json` records library versions, seeds, config and a
checkpoint -> result table.
