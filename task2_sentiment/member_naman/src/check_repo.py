"""Repository hygiene check: no personal paths, no secrets, big binaries are git-ignored, folders exist.

Run:  python src/check_repo.py        (exit code 1 if any rule fails)
"""

import fnmatch
import getpass
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import project_root

SKIP_DIRS = {"data_processed", "checkpoints", "hf_cache", ".ipynb_checkpoints", "__pycache__", ".git", ".venv", "venv",
             "node_modules"}
SKIP_SUFFIXES = {".pt", ".npz", ".npy", ".png", ".jpg", ".pdf", ".gz", ".zip"}
BIG_BINARY_SUFFIXES = {".pt", ".npz", ".npy"}
MAX_FILE_BYTES = 20 * 2**20

SECRET_PATTERNS = {
    "api key": re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{8,}"),
    "token assignment": re.compile(r"(?i)\btoken\s*=\s*[\"'][^\"']{8,}[\"']"),
    "secret assignment": re.compile(r"(?i)\bsecret\s*=\s*[\"'][^\"']{4,}[\"']"),
    "password": re.compile(r"(?i)\bpassword\s*[=:]\s*[\"']?[^\s\"']{4,}"),
    "hf_ token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "sk- key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def personal_path_patterns():
    """Regexes for absolute personal paths, including the name of the user running this script."""
    patterns = {
        "windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"),
        "mac user path": re.compile(r"(?<![\w.])/Users/[^/\s\"']+/"),
        "linux home path": re.compile(r"(?<![\w.])/home/[^/\s\"']+/"),
    }
    names = {os.environ.get("USERNAME", ""), os.environ.get("USER", ""), Path.home().name}
    try:
        names.add(getpass.getuser())
    except Exception:
        pass
    for name in sorted(n for n in names if len(n) >= 3):
        patterns[f"user name '{name[0]}...'"] = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
    return patterns


def scan_text(text, patterns):
    """[(line number, rule name)] for every line of text that matches a pattern."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in patterns.items():
            if pattern.search(line):
                hits.append((number, rule))
    return hits


def notebook_text(raw_json):
    """Text of a notebook's sources and outputs (images are skipped), one cell per line block."""
    notebook = json.loads(raw_json)
    lines = []
    for number, cell in enumerate(notebook.get("cells", []), start=1):
        parts = ["".join(cell.get("source", []))]
        for output in cell.get("outputs", []):
            parts.append("".join(output.get("text", [])))
            for mime in ("text/plain", "text/html", "text/markdown"):
                parts.append("".join(output.get("data", {}).get(mime, [])))
            parts.append("\n".join(output.get("traceback", [])))
        lines.append(f"[cell {number}] " + " ".join(parts).replace("\n", " "))
    return "\n".join(lines)


def list_files(root):
    """Every file under root except the skipped folders."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            yield Path(folder) / name


def read_text_file(path):
    """File text, or None for binary / skipped / huge files."""
    if path.suffix.lower() in SKIP_SUFFIXES or path.stat().st_size > MAX_FILE_BYTES:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    text = data.decode("utf-8", errors="replace")
    return notebook_text(text) if path.suffix == ".ipynb" else text


def is_git_ignored(path, root):
    """Simplified .gitignore check (patterns, directory patterns, anchored patterns, ! negation)."""
    path = Path(path)
    ignored = False
    folders = [root] + [p for p in reversed(path.relative_to(root).parents) if str(p) != "."]
    for folder in folders:
        folder = root / folder if folder != root else root
        gitignore = folder / ".gitignore"
        if not gitignore.is_file():
            continue
        relative = path.relative_to(folder).as_posix()
        parts = relative.split("/")
        for raw in gitignore.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negate = line.startswith("!")
            pattern = line[1:] if negate else line
            directory_only = pattern.endswith("/")
            pattern = pattern.rstrip("/")
            if "/" in pattern:  # anchored to this .gitignore's folder
                pattern = pattern.lstrip("/")
                matched = any(fnmatch.fnmatch("/".join(parts[:i]), pattern) for i in range(1, len(parts) + 1))
            else:  # matches a file or folder name at any depth
                names = parts[:-1] if directory_only else parts
                matched = any(fnmatch.fnmatch(name, pattern) for name in names)
            if matched:
                ignored = not negate
    return ignored


def check(root):
    """Run all rules. Returns {rule: [offending 'file:line' strings]} (an empty list means PASS)."""
    patterns = {**personal_path_patterns()}
    results = {"no absolute personal paths": [], "no secrets": [], ".pt/.npz/.npy files are git-ignored": [],
               "reproducibility folders exist": []}
    for path in list_files(root):
        relative = path.relative_to(root).as_posix()
        if path == Path(__file__).resolve():
            continue
        text = read_text_file(path)
        if text is None:
            continue
        for number, rule in scan_text(text, patterns):
            results["no absolute personal paths"].append(f"{relative}:{number} ({rule})")
        for number, rule in scan_text(text, SECRET_PATTERNS):
            results["no secrets"].append(f"{relative}:{number} ({rule})")
    # big binaries are searched everywhere, also inside checkpoints/ and data_processed/
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "venv", "node_modules"}]
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() in BIG_BINARY_SUFFIXES and not is_git_ignored(path, root):
                results[".pt/.npz/.npy files are git-ignored"].append(path.relative_to(root).as_posix())
    for folder in ("reproducibility/raw_logs", "reproducibility/manifests"):
        if not (Path(root) / folder).is_dir():
            results["reproducibility folders exist"].append(folder)
    return results


def main():
    root = project_root().parent.parent  # the repository root (contains task2_sentiment/ and reproducibility/)
    results = check(root)
    failed = False
    for rule, offenders in results.items():
        print(f"{'FAIL' if offenders else 'PASS'}: {rule}")
        for offender in offenders[:50]:
            print(f"    {offender}")
        failed = failed or bool(offenders)
    print("REPO CHECK FAILED" if failed else "REPO CHECK PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
