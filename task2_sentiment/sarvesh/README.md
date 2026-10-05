# Task 2 — Yelp Polarity (Sarvesh)

Binary sentiment classification with embeddings learned from scratch (no pretrained LMs).

## Layout

```text
sarvesh/
├── src/                 # experiment notebooks
│   ├── 01_baseline_meanpool.ipynb
│   ├── 02_textcnn_experimental1.ipynb
│   └── 03_bilstm_experimental2.ipynb
├── checkpoints/         # best weights for all 3 models
├── outputs/             # metrics, predictions, CIs, slices, McNemar
├── logs/                # raw training logs
├── data_processed/      # runtime note (preprocessing is in-notebook)
├── metrics_report.csv
├── failure_analysis.md
├── results.md
└── requirements.txt
```

## Setup

```bash
cd task2_sentiment/sarvesh
python -m pip install -r requirements.txt
```

Open notebooks under `src/` in order (01 → 02 → 03). Dataset downloads via Hugging Face on first run into shared `task2_sentiment/data/`.
