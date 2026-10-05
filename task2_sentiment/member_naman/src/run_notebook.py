"""Execute src/task2_sentiment.ipynb and save the outputs inside the notebook file.

Run:  python src/run_notebook.py --config local
"""

import argparse
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from build_notebook import NOTEBOOK_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the notebook and save its outputs")
    parser.add_argument("--config", default="local")
    args = parser.parse_args(argv)

    os.environ["CONFIG"] = args.config  # the notebook kernel inherits this
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(Path(NOTEBOOK_PATH).parent)}})
    start = time.time()
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    with_output = sum(1 for c in code_cells if c.outputs)
    errors = sum(1 for c in code_cells for o in c.outputs if o.output_type == "error")
    print(f"ran {len(code_cells)} code cells in {time.time() - start:.1f}s; {with_output} have saved output, {errors} errors")


if __name__ == "__main__":
    main()
