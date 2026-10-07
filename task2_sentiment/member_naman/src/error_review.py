"""Pick 20 test errors for the manual error analysis. This script only collects candidates and facts.

The error types and the fixes are written by hand afterwards: the two analysis columns stay empty.

Run:  python src/error_review.py --config local [--model bigru] [--seed 42]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
from data import decode_escapes
from dataset import get_loader
from predict import MODEL_NAMES, load_best_model, read_test_raw
from slices import make_slices
from utils import Logger, load_config, load_json, project_root

BUCKET_SIZE = 5
MIN_SLICE_REVIEWS = 200
ANALYSIS_COLUMNS = ["Error type (my analysis)", "Testable fix (my proposal)"]
ERROR_TYPE_REMINDER = ["sarcasm", "negation scope", "mixed sentiment", 'contrast "but" clause', "label noise",
                       "truncation", "rare words or names", "very short review", "other"]


def best_model_by_validation(cfg):
    """The model with the highest best_val_accuracy (validation, never test)."""
    best_name, best_accuracy = None, -1.0
    for name in MODEL_NAMES:
        path = cfg["paths"]["output_dir"] / name / "summary.json"
        if path.exists():
            accuracy = load_json(path)["best_val_accuracy"]
            if accuracy > best_accuracy:
                best_name, best_accuracy = name, accuracy
    if best_name is None:
        raise SystemExit("No trained model found: run train.py first")
    return best_name, best_accuracy


def worst_slice(pred, masks):
    """(slice name, macro-F1, size) of the weakest slice with at least 200 reviews; None if no slice is big enough."""
    y, hat = pred["true_label"].to_numpy(), pred["predicted_label"].to_numpy()
    scores = []
    for name, mask in masks.items():
        if mask.sum() >= MIN_SLICE_REVIEWS:
            scores.append((float(metrics.macro_f1_from_counts(*metrics.confusion_counts(y[mask], hat[mask]))), name, int(mask.sum())))
    if not scores:
        return None
    f1, name, size = min(scores)
    return name, f1, size


def select_candidates(pred, masks, seed):
    """Choose 4 buckets of 5 distinct errors. Returns (DataFrame with a 'bucket' column, list of notes).

    Row positions refer to `pred` (the test set in its original order). Buckets, in this order:
      confident false positives (true 0, highest probability), confident false negatives (true 1, lowest
      probability), near-threshold errors (smallest |p - 0.5|), and random errors from the worst slice.
    A bucket with too few candidates is filled with the remaining errors (most confident first,
    or random for the slice bucket) and a note says so.
    """
    prob = pred["positive_probability"].to_numpy()
    wrong = (pred["predicted_label"] != pred["true_label"]).to_numpy()
    true = pred["true_label"].to_numpy()
    confidence = np.abs(prob - 0.5)
    rng = np.random.default_rng(seed)
    chosen, notes = [], []  # chosen: list of (row position, bucket name)

    def take(bucket, candidates):
        """Add up to 5 of the candidates (already in priority order) that are not used yet."""
        used = {row for row, _ in chosen}
        picked = [int(i) for i in candidates if int(i) not in used][:BUCKET_SIZE]
        chosen.extend((row, bucket) for row in picked)
        if len(picked) < BUCKET_SIZE:
            used = {row for row, _ in chosen}
            leftovers = [int(i) for i in np.where(wrong)[0] if int(i) not in used]
            if bucket != "slice-specific failure":
                leftovers.sort(key=lambda i: -confidence[i])
            else:
                leftovers = [int(i) for i in rng.permutation(leftovers)]
            fill = leftovers[:BUCKET_SIZE - len(picked)]
            chosen.extend((row, bucket) for row in fill)
            notes.append(f"Bucket '{bucket}' had only {len(picked)} candidates; filled {len(fill)} from the remaining errors.")

    false_pos = np.where(wrong & (true == 0))[0]
    take("confident false positive", false_pos[np.argsort(-prob[false_pos], kind="stable")])
    false_neg = np.where(wrong & (true == 1))[0]
    take("confident false negative", false_neg[np.argsort(prob[false_neg], kind="stable")])
    errors = np.where(wrong)[0]
    take("near-threshold error", errors[np.argsort(confidence[errors], kind="stable")])

    weak = worst_slice(pred, masks)
    if weak is None:
        notes.append("No slice has at least 200 reviews; the slice bucket uses random errors from all slices.")
        in_slice = errors
    else:
        in_slice = np.where(wrong & masks[weak[0]])[0]
    used = {row for row, _ in chosen}
    pool = [int(i) for i in in_slice if int(i) not in used]
    take("slice-specific failure", rng.permutation(pool) if pool else [])

    rows = pd.DataFrame(chosen, columns=["row", "bucket"])
    assert rows["row"].is_unique, "a review was selected twice"
    return rows, notes


def attention_words(cfg, model_name, rows, vocab, top_k=5):
    """For the bigru: the top_k words by attention weight for each selected review (on the truncated input)."""
    if model_name != "bigru":
        return [""] * len(rows)
    model, _ = load_best_model(cfg, model_name, torch.device("cpu"))
    id_to_word = {i: w for w, i in vocab.items()}
    builder = get_loader(cfg, model_name, "test", train=False).builder
    result = []
    for row in rows:
        batch = builder.make(np.array([row]))
        with torch.no_grad():
            _, attn = model(batch["ids"], batch["lengths"], return_attention=True)
        weights = attn[0].numpy()
        top = np.argsort(-weights)[:top_k]
        result.append(", ".join(f"{id_to_word[int(batch['ids'][0, t])]} ({weights[t]:.2f})" for t in top))
    return result


def one_line(text, limit):
    """Single-line, table-safe version of a review, cut to `limit` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:limit] + ("..." if len(text) > limit else "")
    return text.replace("|", "\\|")


def render_markdown(model_name, selected_by, run_name, candidates, notes, weak):
    lines = [f"# Failure analysis: {model_name} ({run_name})", "",
             f"Model chosen by best validation accuracy ({selected_by}). Candidates only: the two right-hand columns are for my own analysis.", "",
             "Error types to choose from (reminder only): " + "; ".join(ERROR_TYPE_REMINDER) + ".", ""]
    if weak:
        lines.append(f"Weakest slice (macro-F1, at least {MIN_SLICE_REVIEWS} reviews): **{weak[0]}** with macro-F1 {weak[1]:.4f} on {weak[2]} reviews.")
    for note in notes:
        lines.append(f"- NOTE: {note}")
    lines += ["", "| # | Bucket | orig_index | True | Pred | Prob | Review text (first 600 chars) | Top attention words | "
              + " | ".join(ANALYSIS_COLUMNS) + " |", "|" + "---|" * (8 + len(ANALYSIS_COLUMNS))]
    for _, r in candidates.iterrows():
        cells = [str(r["rank"]), r["bucket"], str(r["orig_index"]), str(r["true_label"]), str(r["predicted_label"]),
                 f"{r['positive_probability']:.3f}", one_line(r["text"], 600), one_line(r["top_attention_words"], 200), "", ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def analysis_columns_filled(markdown_text):
    """True if any row of the failure-analysis table has text in either of the last two (analysis) columns."""
    for line in markdown_text.splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
        if len(cells) >= 10 and cells[0].strip().isdigit() and (cells[-1].strip() or cells[-2].strip()):
            return True
    return False


def write_protected(path, text):
    """Write text to path, unless path already holds my filled-in analysis: then write <name>.new.md instead."""
    path = Path(path)
    if path.exists() and analysis_columns_filled(path.read_text(encoding="utf-8")):
        path = path.with_name(path.stem + ".new.md")
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select 20 test errors for the manual error analysis")
    parser.add_argument("--config", default="local")
    parser.add_argument("--model", default=None, choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = Logger(cfg["paths"]["output_dir"] / "error_review.log")
    if args.model:
        model_name = args.model
        selected_by = f"chosen with --model; its validation accuracy {load_json(cfg['paths']['output_dir'] / model_name / 'summary.json')['best_val_accuracy']:.4f}"
    else:
        model_name, accuracy = best_model_by_validation(cfg)
        selected_by = f"{accuracy:.4f}"
    pred = pd.read_csv(cfg["paths"]["output_dir"] / model_name / "test_predictions.csv")
    test_raw = read_test_raw(cfg)
    assert (pred["orig_index"].to_numpy() == test_raw["orig_index"].to_numpy()).all()
    assert (pred["true_label"].to_numpy() == test_raw["label"].to_numpy()).all()
    masks = make_slices(test_raw["text"].tolist(), test_raw["raw_words"].to_numpy())

    rows, notes = select_candidates(pred, masks, args.seed)
    weak = worst_slice(pred, masks)
    with np.load(cfg["paths"]["processed_dir"] / "arrays.npz") as npz:
        n_tokens = npz["test_n_tokens"]
    vocab = load_json(cfg["paths"]["processed_dir"] / "vocab.json")

    positions = rows["row"].to_numpy()
    candidates = pd.DataFrame({
        "rank": np.arange(1, len(rows) + 1), "bucket": rows["bucket"], "orig_index": pred["orig_index"].to_numpy()[positions],
        "true_label": pred["true_label"].to_numpy()[positions], "predicted_label": pred["predicted_label"].to_numpy()[positions],
        "positive_probability": pred["positive_probability"].to_numpy()[positions],
        "text": [decode_escapes(test_raw["text"][i]) for i in positions],
        "raw_words": test_raw["raw_words"].to_numpy()[positions],
        "cleaned_tokens": n_tokens[positions], "truncated_at_max_len": n_tokens[positions] > cfg["data"]["max_len"],
        "top_attention_words": attention_words(cfg, model_name, positions, vocab),
    })

    out_dir = cfg["paths"]["output_dir"] / "error_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "candidates.csv", index=False)
    markdown = render_markdown(model_name, selected_by, cfg["run_name"], candidates, notes, weak)
    written = [write_protected(out_dir / "failure_analysis.md", markdown)]
    if cfg["run_name"] == "gpu":
        written.append(write_protected(project_root() / "failure_analysis.md", markdown))
    for path in written:
        if path.name.endswith(".new.md"):
            logger.log(f"{path.name[:-7]}.md already has my analysis: wrote {path.name} instead")

    logger.log(f"error review | run {cfg['run_name']} | model {model_name} (best validation accuracy {selected_by})")
    logger.log(f"bucket sizes: {rows['bucket'].value_counts().to_dict()}")
    logger.log(f"worst slice: {weak}")
    for note in notes:
        logger.log(f"NOTE: {note}")


if __name__ == "__main__":
    main()
