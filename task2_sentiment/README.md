<!-- TASK2 NAMAN START -->
# Task 2 (Naman): Yelp Polarity sentiment classification

**Contents:** [Approach](#approach) | [Environment setup](#environment-setup) | [Smoke test](#smoke-test-the-reproducibility-command) | [Full pipeline](#full-pipeline) | [Where results live](#where-results-live) | [Comparing with a teammate](#comparing-with-a-teammates-predictions) | [Reproducibility notes](#reproducibility-notes) | [Hardware disclosure](#hardware-disclosure) | [Gitignored files](#checkpoints-and-processed-data-are-not-committed)

All code is in `task2_sentiment/member_naman/`. Every path in the code is resolved relative to that folder.

## Approach

Binary sentiment classification on `fancyzhx/yelp_polarity` (560,000 train and 38,000 test reviews), with **no pretrained embeddings or language models**: every model is trained from scratch. Reviews are cleaned (escape decoding, URL/HTML removal, contraction expansion, a 37-word neutral stopword list that keeps negations and intensifiers, WordNet lemmatisation) and a vocabulary is built from training tokens only. The validation split (10% of train) selects the best epoch; the test split is used only by `predict.py`.

| Model | Idea |
|---|---|
| `ngram_bag` (baseline) | fastText-style bag of unigrams plus hashed bigrams, averaged by an `EmbeddingBag`, then one linear layer |
| `transformer` | small pre-norm Transformer encoder (2 layers, learned positions, word dropout) with mean+max pooling |
| `bigru` | bidirectional 2-layer GRU over word + hashed-bigram embeddings with max, mean and attention pooling |

## Environment setup

Python 3.12 on Windows/PowerShell (other platforms should work; the scripts use only relative paths).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# NVIDIA GPU: install the CUDA build of torch from the PyTorch index first (a plain pip install gives the CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r task2_sentiment/member_naman/requirements.txt
```

The same index-URL note is written at the top of `requirements.txt`. The dataset is downloaded from Hugging Face on first use and cached in `task2_sentiment/data/hf_cache` (gitignored); NLTK downloads the WordNet data on first use.

## Smoke test (the reproducibility command)

From the repository root:

```powershell
python task2_sentiment/member_naman/src/smoke_test.py --device cpu --fresh
```

It runs the whole pipeline on the tiny `local` config (5,000 train reviews, 1 epoch per model): setup check, unit tests, repo check, data build, EDA, training of the three models, test predictions, evaluation, error review, `results.md`, the notebook, the manifest, and a deliverable check. Options: `--skip-notebook`, `--device cuda`; without `--fresh` a valid data cache is reused.

Expected: one line per step with `PASS`, a step table, then the final line `SMOKE TEST PASSED` (or `SMOKE TEST FAILED at step N`, with the last lines of that step's output). Expected duration on a modern CPU with the dataset already cached: **about 3 minutes** (167 s measured; the first run also downloads the dataset). The smoke test only writes the `local` outputs and never touches the full-run deliverables.

The unit tests live in `task2_sentiment/member_naman/tests/` (plain asserts, no pytest). The smoke test runs all of them; to run one, from `task2_sentiment/member_naman/`: `python tests/test_data.py` (likewise `test_models`, `test_train`, `test_eval`, `test_reports`, `test_repo_tools`). `test_train`, `test_eval` and `test_reports` need the `local` run to exist first (the smoke test creates it).

## Full pipeline

Run these from `task2_sentiment/member_naman/`:

```powershell
python src/data.py --config gpu                 # build the full processed data (about 1 minute; reused if already valid)
.\run_full_training.ps1                         # trains ngram_bag, transformer, bigru in turn; re-run to resume
python src/finalize.py --config gpu             # after training: predictions, evaluation, reports, notebook, manifest, repo check
```

`finalize.py` refuses to run (and says which models are missing) until all three `best.pt` files exist. It accepts `--other PATH --other-name LABEL` (repeatable) to compare with other prediction files. Afterwards, fill in the two analysis columns of `failure_analysis.md`, write the `TODO (my words)` sections of `results.md`, then commit and push.

## Where results live

| What | Where |
|---|---|
| Final metrics table (full run) | `task2_sentiment/member_naman/metrics_report.csv` |
| Manual error analysis (full run) | `task2_sentiment/member_naman/failure_analysis.md` |
| Written results (full run) | `task2_sentiment/member_naman/results.md` |
| All plots, per-model predictions and summaries | `task2_sentiment/member_naman/outputs/<run>/` (`gpu` = full run, `local` = smoke test) |
| Evaluation plots | `outputs/<run>/eval/`, EDA in `outputs/<run>/eda/`, error review in `outputs/<run>/error_review/` |
| Executed notebook | `task2_sentiment/member_naman/src/task2_sentiment.ipynb` |
| Checkpoints (gitignored) | `task2_sentiment/member_naman/checkpoints/<run>/<model>/` |
| Raw training logs (committed, append-only) | `reproducibility/raw_logs/naman_task2_gpu_<model>_train.log` |
| Manifests and `pip freeze` | `reproducibility/manifests/naman_task2_<run>_manifest.json` and `..._requirements_freeze.txt` |

## Comparing with a teammate's predictions

The other file is a csv with the columns `true_label`, `predicted_label`, `positive_probability` for all 38,000 test reviews in the original Hugging Face order:

```powershell
python src/compare_external.py --config gpu --model bigru --other <path-to-csv> --other-name teammate_model
```

It checks the row count and the true labels, then reports both accuracies, a paired McNemar test (b, c, exact p) and a paired-bootstrap 95% interval of the accuracy difference, saved to `outputs/gpu/vs_<other-name>.json`. The path is only read from the command line and is never stored.

## Reproducibility notes

- One seed (`seed: 42` in the configs) drives python, numpy and torch; data splits use `numpy.random.default_rng(seed)`; cuDNN is set to deterministic with benchmarking off.
- Data processing, splits, vocabulary and the CPU metric computations are deterministic. The processed-data cache is keyed on the settings that affect preprocessing and a `CLEANING_VERSION`.
- **Not bitwise reproducible on GPU:** `torch.use_deterministic_algorithms` is not enabled (the CUDA backward of `EmbeddingBag` uses non-deterministic atomic additions), bfloat16 mixed precision is used on CUDA, and cuDNN RNN kernels are not guaranteed identical across runs. Expect small run-to-run differences in the last digits of the loss; rankings of clearly different models should not change. Bootstrap intervals (1000 resamples, seed 42) are repeatable.
- Every run logs the torch version, the config and the exact device name; `make_manifest.py` records versions, hashes of the config and checkpoints, and the data cache key.

## Hardware disclosure

The device name (CPU or GPU model) is logged at the start of every training run, stored in `summary.json` and the metrics report, and written to `results.md` and the manifest. Reported training times and throughputs are only comparable on the same hardware.

## Checkpoints and processed data are not committed

`checkpoints/*`, `data_processed/*`, `*.pt`, `*.npz`, `*.npy` and the Hugging Face cache are gitignored. Regenerate them with `python src/data.py --config gpu` and `.\run_full_training.ps1`; `python src/check_repo.py` verifies that no personal paths, secrets or unignored large binaries are in the repository.
<!-- TASK2 NAMAN END -->
