"""Plain-assert tests for error_review.py, make_results_md.py and build_notebook.py. Run:  python tests/test_reports.py"""

import re
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import build_notebook
import error_review
import make_results_md
from utils import load_config

LOCAL = load_config("local")


def synthetic_predictions(n=3000, seed=0, few_false_positives=False):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 2, n)
    prob = np.where(true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    if few_false_positives:
        prob = np.where(true == 0, np.minimum(prob, 0.45), prob)  # negatives are never predicted positive
        prob[np.where(true == 0)[0][:3]] = 0.9  # ... except three
    pred = (prob >= 0.5).astype(int)
    masks = {"short": rng.random(n) < 0.3, "long": rng.random(n) < 0.2, "tiny": np.arange(n) < 50}
    frame = pd.DataFrame({"orig_index": np.arange(n), "true_label": true, "predicted_label": pred,
                          "positive_probability": prob})
    return frame, masks


def bucket_rows(rows, bucket):
    return rows[rows["bucket"] == bucket]["row"].tolist()


def test_select_candidates_buckets():
    frame, masks = synthetic_predictions()
    rows, notes = error_review.select_candidates(frame, masks, seed=42)
    assert len(rows) == 20 and rows["row"].is_unique and not notes
    assert rows["bucket"].value_counts().to_dict() == {b: 5 for b in rows["bucket"].unique()} and rows["bucket"].nunique() == 4
    wrong = frame["true_label"] != frame["predicted_label"]

    fp = frame.loc[bucket_rows(rows, "confident false positive")]
    assert (fp["true_label"] == 0).all() and (fp["predicted_label"] == 1).all()
    all_fp = frame[wrong & (frame["true_label"] == 0)]["positive_probability"].sort_values(ascending=False)
    assert sorted(fp["positive_probability"]) == sorted(all_fp.iloc[:5])  # the 5 highest probabilities

    fn = frame.loc[bucket_rows(rows, "confident false negative")]
    assert (fn["true_label"] == 1).all() and (fn["predicted_label"] == 0).all()
    all_fn = frame[wrong & (frame["true_label"] == 1)]["positive_probability"].sort_values()
    assert sorted(fn["positive_probability"]) == sorted(all_fn.iloc[:5])  # the 5 lowest probabilities

    near_rows = bucket_rows(rows, "near-threshold error")
    assert wrong.iloc[near_rows].all()
    distance = (frame["positive_probability"] - 0.5).abs()
    used_before_near = set(bucket_rows(rows, "confident false positive") + bucket_rows(rows, "confident false negative") + near_rows)
    others = [i for i in np.where(wrong)[0] if i not in used_before_near]
    assert distance.iloc[near_rows].max() <= distance.iloc[others].min()  # nothing closer to 0.5 was left out

    slice_name, _, size = error_review.worst_slice(frame, masks)
    assert size >= 200 and slice_name != "tiny"  # slices under 200 reviews are ignored
    slice_rows = bucket_rows(rows, "slice-specific failure")
    assert wrong.iloc[slice_rows].all() and masks[slice_name][slice_rows].all()
    again, _ = error_review.select_candidates(frame, masks, seed=42)
    assert again["row"].tolist() == rows["row"].tolist()  # seeded: same choice twice


def test_select_candidates_shortage_is_reported():
    frame, masks = synthetic_predictions(few_false_positives=True)
    rows, notes = error_review.select_candidates(frame, masks, seed=1)
    assert len(rows) == 20 and rows["row"].is_unique
    assert any("confident false positive" in n for n in notes), notes


def table_rows(markdown):
    """The cells of every numbered table row."""
    result = []
    for line in markdown.splitlines():
        cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
        if len(cells) >= 10 and cells[0].strip().isdigit():
            result.append([c.strip() for c in cells])
    return result


def test_error_review_output_leaves_analysis_columns_empty():
    error_review.main(["--config", "local"])
    markdown = (LOCAL["paths"]["output_dir"] / "error_review" / "failure_analysis.md").read_text(encoding="utf-8")
    rows = table_rows(markdown)
    assert len(rows) == 20 and all(r[-1] == "" and r[-2] == "" for r in rows)
    assert not error_review.analysis_columns_filled(markdown)
    for word in ("sarcasm", "negation scope", "label noise"):
        assert word in markdown  # the reminder block
    candidates = pd.read_csv(LOCAL["paths"]["output_dir"] / "error_review" / "candidates.csv")
    assert len(candidates) == 20 and candidates["orig_index"].is_unique
    assert "\\n" not in " ".join(candidates["text"])  # escapes were decoded
    assert candidates["top_attention_words"].isna().all() == (error_review.best_model_by_validation(LOCAL)[0] != "bigru")


def test_filled_analysis_is_never_overwritten():
    frame, masks = synthetic_predictions()
    rows, notes = error_review.select_candidates(frame, masks, 42)
    candidates = pd.DataFrame({"rank": range(1, 21), "bucket": rows["bucket"], "orig_index": rows["row"],
                               "true_label": 0, "predicted_label": 1, "positive_probability": 0.9, "text": "some review",
                               "top_attention_words": ""})
    empty = error_review.render_markdown("m", "0.9", "x", candidates, notes, None)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "failure_analysis.md"
        assert error_review.write_protected(path, empty) == path  # nothing there yet: written
        filled = empty.replace("| 1 | confident false positive", "| 1 | confident false positive", 1)
        first_row = [l for l in filled.splitlines() if l.startswith("| 1 |")][0]
        edited = filled.replace(first_row, first_row[:-6] + " sarcasm | use a sentiment lexicon |")
        path.write_text(edited, encoding="utf-8")
        assert error_review.analysis_columns_filled(edited)
        written = error_review.write_protected(path, empty)
        assert written.name == "failure_analysis.new.md"
        assert path.read_text(encoding="utf-8") == edited  # my text is untouched
        path.write_text(empty, encoding="utf-8")  # an unfilled file may be refreshed
        assert error_review.write_protected(path, empty) == path


def test_results_md_keeps_text_outside_markers():
    blocks = make_results_md.build_blocks(LOCAL)
    assert all(f"<!-- AUTO:{name} -->" in make_results_md.TEMPLATE for name in blocks)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "results.md"
        assert make_results_md.update_results_file(path, blocks, "local") == []
        text = path.read_text(encoding="utf-8")
        for heading in ("Overview", "Data and preprocessing", "Model 1 (n-gram bag)", "Model 2 (Transformer)",
                        "Model 3 (bigram BiGRU)", "Results", "Comparison and observations", "Hardware disclosure",
                        "Limitations and future work"):
            assert f"## {heading}" in text, heading
        assert "TODO (my words)" in text
        edited = text.replace("TODO (my words): limitations and future work.", "MY OWN WORDS about limits.")
        edited = edited.replace("<!-- AUTO:overview -->\n", "<!-- AUTO:overview -->\nSTALE AUTO TEXT\n")
        edited += "\nMY FOOTNOTE\n"
        path.write_text(edited, encoding="utf-8")
        make_results_md.update_results_file(path, blocks, "local")
        again = path.read_text(encoding="utf-8")
        assert "MY OWN WORDS about limits." in again and "MY FOOTNOTE" in again
        assert "STALE AUTO TEXT" not in again
        make_results_md.update_results_file(path, blocks, "local")
        assert path.read_text(encoding="utf-8") == again  # running twice changes nothing


def test_notebook_has_nine_sections():
    notebook = build_notebook.build()
    headers = [c.source for c in notebook.cells if c.cell_type == "markdown" and re.match(r"## \d\.", c.source)]
    assert [h.split(".")[0] for h in headers] == [f"## {i}" for i in range(1, 10)], headers
    kinds = [c.cell_type for c in notebook.cells][1:]
    assert kinds == ["markdown", "code"] * 9
    nbformat.validate(notebook)
    saved = nbformat.read(build_notebook.NOTEBOOK_PATH, as_version=4)  # the file written by build_notebook.py
    assert sum(1 for c in saved.cells if c.cell_type == "markdown" and re.match(r"## \d\.", c.source)) == 9


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
