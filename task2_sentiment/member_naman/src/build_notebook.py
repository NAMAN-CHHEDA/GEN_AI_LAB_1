"""Create src/task2_sentiment.ipynb (without outputs). The notebook trains nothing: it shows saved artifacts.

Run:  python src/build_notebook.py     then     python src/run_notebook.py --config local
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = Path(__file__).resolve().parent / "task2_sentiment.ipynb"

# (section header, code of the cell below it)
SECTIONS = [
    ("## 1. Setup and config",
     'import os, sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd()))\nimport notebook_helpers as nh\n'
     'cfg = nh.setup(os.environ.get("CONFIG", "local"))'),
    ("## 2. Data summary", "nh.show_data_summary(cfg)"),
    ("## 3. EDA figures and a cleaning example", "nh.show_eda(cfg)"),
    ("## 4. Model architectures", "nh.show_architectures(cfg)"),
    ("## 5. Training curves and training summary", "nh.show_training(cfg)"),
    ("## 6. Test metrics and evaluation plots", "nh.show_test_metrics(cfg)"),
    ("## 7. McNemar results and slice table", "nh.show_mcnemar_and_slices(cfg)"),
    ("## 8. Error review candidates", "nh.show_error_review(cfg)"),
    ("## 9. Hardware disclosure and reproducibility", "nh.show_hardware(cfg)"),
]


def build():
    """The notebook object: a title, then one markdown header and one code cell per section."""
    cells = [new_markdown_cell("# Task 2: Yelp Polarity sentiment classification\n\n"
                               "Loads saved artifacts only (no training). The run is chosen with the CONFIG environment variable.")]
    for header, code in SECTIONS:
        cells += [new_markdown_cell(header), new_code_cell(code)]
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main():
    nbformat.write(build(), NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH.name} with {len(SECTIONS)} sections")


if __name__ == "__main__":
    main()
