# Task 1 Sarvesh — submission / Google Drive backup pack

Created for zip + Drive backup. Does **not** include .venv (reinstall with requirements.txt).

## Included
- Source: `src/` (notebook + `charlm` package + configs + run scripts)
- Reports: `results.md`, `failure_analysis.md`, `README.md`, `DESIGN_NOTES.md`
- Metrics: `metrics_report.csv`
- Curves & samples: `outputs/`
- Training logs: `logs/`
- Checkpoints: `checkpoints/sarvesh_main_best.pt`, `sarvesh_main_final.pt`
- Data: raw TinyStories + `data_processed/` (vocab, split, ids)

## How to re-run later
```powershell
cd task1_sarvesh_submission
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# GPU (recommended):
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe src\run_full.py
```

## Zip command
```powershell
Compress-Archive -Path <REDACTED_PERSONAL_PATH> -DestinationPath <REDACTED_PERSONAL_PATH> -Force
```
