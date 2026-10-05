# DATA266 Lab 1 — Team Repo

LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer

## Team

| Member   | Folder name(s)                                      |
|----------|-----------------------------------------------------|
| Naman    | `task1_llm/naman`, `task2_sentiment/member_naman`, `task3_gan/naman` |
| Sarvesh  | `task1_llm/sarvesh`, `task2_sentiment/sarvesh`, `task3_gan/sarvesh` |

Each member owns their own architecture, hyperparameters, metrics, and `results.md`.

## Repo layout

```text
GEN_AI_LAB_1/
├── README.md
├── task1_llm/
│   ├── data/                 # shared raw TinyStories
│   ├── naman/
│   └── sarvesh/
├── task2_sentiment/
│   ├── data/                 # shared Yelp/IMDB polarity data
│   ├── member_naman/
│   └── sarvesh/
├── task3_gan/
│   ├── data/                 # shared Monet / Photo images (local; usually gitignored)
│   ├── naman/
│   └── sarvesh/
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
└── report/                   # combined team PDF goes here
```

Each member folder follows the lab template:

```text
member_name/
├── src/
├── data_processed/
├── checkpoints/
├── outputs/
├── metrics_report.csv
├── failure_analysis.md
└── results.md
```

## Smoke-test / reproduce a run

### Task 1 — Naman

```bash
cd task1_llm/naman/src
python -m pip install -r ../requirements.txt
# see task1_llm/README.md for the exact one-command smoke test
```

### Task 1 — Sarvesh

```bash
cd task1_llm/sarvesh
python -m pip install -r requirements.txt
python src/run_smoke.py
```

### Task 2 — Naman

```bash
cd task2_sentiment/member_naman
python -m pip install -r requirements.txt
python src/smoke_test.py
```

### Task 2 — Sarvesh

Open the notebooks under `task2_sentiment/sarvesh/src/` (`01_baseline_meanpool.ipynb`, etc.).

### Task 3 — Naman

See `task3_gan/naman/README.md`.

### Task 3 — Sarvesh

```bash
cd task3_gan/sarvesh
python evaluate_local.py
# Full metric recompute: open src/Part3_Evaluation_Script.ipynb
# (requires shared task3_gan/data/monet_jpg and photo_jpg)
```

## Notes

- Do not commit personal absolute paths, credentials, or API keys.
- Keep raw training logs unedited under `reproducibility/raw_logs/` (and each member’s own `logs/` where present).
- Shared datasets live under each task’s `data/`; per-member preprocessing stays in `data_processed/`.
- Final team report: `report/DATA266_Lab1_Report_Team_[Team Number].pdf`.
