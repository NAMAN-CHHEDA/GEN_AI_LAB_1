"""Plain-assert tests for check_repo.py, make_manifest.py and finalize.py helpers. Run:  python tests/test_repo_tools.py

Strings that look like personal paths or secrets are assembled from pieces here, so this file itself stays clean.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import check_repo
import finalize
import make_manifest
from utils import load_config

PATTERNS = {**check_repo.personal_path_patterns(), **check_repo.SECRET_PATTERNS}


def rules_found(text):
    return {rule for _, rule in check_repo.scan_text(text, PATTERNS)}


def test_personal_paths_are_detected():
    assert "windows user path" in rules_found("path = '" + "C:" + "\\Us" + "ers\\bob\\project'")
    assert "windows user path" in rules_found("d" + ":/Us" + "ers/bob/x")
    assert "mac user path" in rules_found("/Us" + "ers/bob/project")
    assert "linux home path" in rules_found("/ho" + "me/bob/project")
    me = check_repo.Path.home().name
    assert any(rule.startswith("user name") for rule in rules_found(f"owner: {me}"))
    assert not rules_found("outputs/local/ngram_bag/summary.json and src/utils.py")


def test_secrets_are_detected_and_normal_code_is_not():
    assert "api key" in rules_found("api" + "_key = 'abcd1234efgh'")
    assert "hf_ token" in rules_found("x = '" + "hf" + "_" + "a" * 30 + "'")
    assert "sk- key" in rules_found("key " + "sk" + "-" + "b" * 30)
    assert "AWS key" in rules_found("id " + "AKIA" + "A" * 16)
    assert "token assignment" in rules_found("tok" + "en = 'abcdefgh1234'")
    assert "password" in rules_found("pass" + "word: hunter22")
    for harmless in ("token_dropout: 0.1", "tokens = []", "for token in SPECIAL_TOKENS:", "vocab = {token: i}"):
        assert not rules_found(harmless), harmless


def test_gitignore_matching():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / ".gitignore").write_text("*.pt\n")
        (root / "m").mkdir()
        (root / "m" / ".gitignore").write_text("checkpoints/*\n!checkpoints/.gitkeep\ndata_processed/*\n")
        (root / "m" / "checkpoints" / "gpu").mkdir(parents=True)
        (root / "m" / "outputs").mkdir()
        assert check_repo.is_git_ignored(root / "m" / "outputs" / "a.pt", root)  # by *.pt
        assert check_repo.is_git_ignored(root / "m" / "checkpoints" / "gpu" / "x.npz", root)  # by checkpoints/*
        assert not check_repo.is_git_ignored(root / "m" / "checkpoints" / ".gitkeep", root)  # negated
        assert not check_repo.is_git_ignored(root / "m" / "outputs" / "x.npz", root)  # not covered


def make_repo(root, with_folders=True):
    if with_folders:
        (root / "reproducibility" / "raw_logs").mkdir(parents=True)
        (root / "reproducibility" / "manifests").mkdir(parents=True)


def test_check_repo_end_to_end():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        make_repo(root)
        (root / "ok.py").write_text("print('hello')\n")
        results = check_repo.check(root)
        assert all(not offenders for offenders in results.values()), results

        (root / "bad.py").write_text("p = '" + "C:" + "\\Us" + "ers\\bob'\n")
        (root / "bad.env").write_text("pass" + "word = hunter22\n")
        (root / "model.npz").write_bytes(b"\x00\x01")
        results = check_repo.check(root)
        assert any(o.startswith("bad.py:1") for o in results["no absolute personal paths"])
        assert any(o.startswith("bad.env:1") for o in results["no secrets"])
        assert results[".pt/.npz/.npy files are git-ignored"] == ["model.npz"]
        (root / ".gitignore").write_text("*.npz\n")
        assert check_repo.check(root)[".pt/.npz/.npy files are git-ignored"] == []

    with tempfile.TemporaryDirectory() as folder:
        make_repo(Path(folder), with_folders=False)
        assert len(check_repo.check(Path(folder))["reproducibility folders exist"]) == 2


def test_notebook_outputs_are_scanned_but_images_are_not():
    bad_path = "C:" + "\\Us" + "ers\\bob\\data"
    notebook = {"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": [
        {"output_type": "stream", "name": "stdout", "text": [f"loaded {bad_path}\n"]},
        {"output_type": "display_data", "data": {"image/png": "AKIA" + "A" * 16, "text/plain": ["<Image>"]}}]}]}
    text = check_repo.notebook_text(json.dumps(notebook))
    rules = {rule for _, rule in check_repo.scan_text(text, PATTERNS)}
    assert "windows user path" in rules and "AWS key" not in rules


def test_finalize_reports_missing_checkpoints():
    cfg = {"paths": {}}
    with tempfile.TemporaryDirectory() as folder:
        cfg["paths"]["checkpoint_dir"] = Path(folder)
        assert finalize.missing_checkpoints(cfg) == ["ngram_bag", "transformer", "bigru"]
        (Path(folder) / "bigru").mkdir()
        (Path(folder) / "bigru" / "best.pt").write_bytes(b"x")
        assert finalize.missing_checkpoints(cfg) == ["ngram_bag", "transformer"]


def test_manifest_lists_missing_artifacts_and_hashes_files():
    cfg = load_config("local", "manifesttest")  # a tag whose folders do not exist
    entry = make_manifest.model_entry(cfg, "bigru", None)
    assert set(entry.values()) == {make_manifest.MISSING}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "f.bin"
        path.write_bytes(b"abc")
        assert make_manifest.sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    commit, note = make_manifest.git_commit()
    assert commit is None or len(commit) == 40
    assert commit is not None or note


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {name} -> {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
