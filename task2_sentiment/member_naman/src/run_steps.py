"""Tiny helper shared by finalize.py and smoke_test.py: run scripts as subprocesses, time them, print a table."""

import os
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TESTS_DIR = SRC_DIR.parent / "tests"
TAIL_LINES = 30  # how much of a failed step's output is shown


def run_script(label, script, *script_args):
    """Run `python <script> <args>` in the member_naman folder (script is looked up in src/, then tests/).

    Returns a result dict; the output is shown only on failure.
    """
    path = SRC_DIR / script if (SRC_DIR / script).exists() else TESTS_DIR / script
    command = [sys.executable, str(path), *map(str, script_args)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    print(f">>> {label}: python {path.parent.name}/{script} {' '.join(map(str, script_args))}".rstrip(), flush=True)
    start = time.time()
    done = subprocess.run(command, cwd=SRC_DIR.parent, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=environment)
    seconds = time.time() - start
    ok = done.returncode == 0
    print(f"    {'PASS' if ok else 'FAIL'} ({seconds:.1f}s)", flush=True)
    if not ok:
        output = (done.stdout + done.stderr).strip().splitlines()
        print("    --- last lines of output ---")
        print("\n".join("    " + line for line in output[-TAIL_LINES:]))
    return {"label": label, "seconds": seconds, "ok": ok}


def print_table(results):
    """Step table: number, name, duration, PASS/FAIL, and the total time."""
    print("\n step | name" + " " * 44 + "| seconds | result")
    print("------+" + "-" * 49 + "+---------+-------")
    for number, r in enumerate(results, start=1):
        print(f" {number:>4} | {r['label']:<47} | {r['seconds']:>7.1f} | {'PASS' if r['ok'] else 'FAIL'}")
    print(f"total: {sum(r['seconds'] for r in results):.1f}s")
