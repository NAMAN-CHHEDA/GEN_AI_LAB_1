"""Exploratory data analysis on the processed data.

Run:  python src/eda.py --config gpu      (writes tables, plots and json to outputs/<run_name>/eda/)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import SPECIAL_TOKENS, clean_text, get_data, load_raw_dataset
from slices import make_slices
from utils import Logger, load_config, save_json

CLASS_NAMES = {0: "negative", 1: "positive"}
CLASS_COLORS = {0: "#c0392b", 1: "#2471a3"}
MAX_LEN_CANDIDATES = [128, 192, 256, 320, 384, 512]
PERCENTILES = [50, 75, 90, 95, 99]


def show_table(log, title, df):
    """Log a dataframe as plain text."""
    log(f"\n=== {title} ===")
    log(df.to_string())


def plain(obj):
    """Turn numpy numbers into normal python numbers so the summary can be saved as json."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# a) class balance
# ---------------------------------------------------------------------------

def class_balance(data, eda_dir, log):
    rows = []
    for name in ("train", "val", "test"):
        labels = data[name].labels
        rows.append({"split": name, "total": len(labels), "negative": int((labels == 0).sum()),
                     "positive": int((labels == 1).sum()), "positive_share": round(float(labels.mean()), 4)})
    table = pd.DataFrame(rows).set_index("split")
    show_table(log, "Class balance", table)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["negative"], width=0.4, color=CLASS_COLORS[0], label="negative (0)")
    ax.bar(x + 0.2, table["positive"], width=0.4, color=CLASS_COLORS[1], label="positive (1)")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("reviews")
    ax.set_title("Class balance per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eda_dir / "class_balance.png", dpi=150)
    plt.close(fig)
    return table.to_dict(orient="index")


# ---------------------------------------------------------------------------
# b), c), d), e) lengths
# ---------------------------------------------------------------------------

def length_histogram(values, labels, title, xlabel, path):
    p99 = np.percentile(values, 99)
    bins = np.linspace(0, p99, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in (0, 1):
        ax.hist(np.clip(values[labels == label], 0, p99), bins=bins, histtype="step", density=True,
                linewidth=1.8, color=CLASS_COLORS[label], label=CLASS_NAMES[label])
    for p, style in ((50, ":"), (90, "--"), (95, "-.")):
        value = np.percentile(values, p)
        ax.axvline(value, color="black", linestyle=style, linewidth=1, label=f"p{p} = {value:.0f}")
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} (clipped at p99 = {p99:.0f})")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def length_stats(train, eda_dir, log):
    columns = {"raw_words": train.raw_words.astype(np.int64), "tokens": train.n_tokens.astype(np.int64)}
    rows = []
    for var, values in columns.items():
        for group, mask in (("all", np.ones(len(values), bool)), ("negative", train.labels == 0),
                            ("positive", train.labels == 1)):
            v = values[mask]
            row = {"variable": var, "group": group, "mean": round(float(v.mean()), 1), "median": np.percentile(v, 50)}
            for p in PERCENTILES[1:]:
                row[f"p{p}"] = np.percentile(v, p)
            row["max"] = v.max()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(eda_dir / "length_stats.csv", index=False)
    show_table(log, "Length statistics (train)", table.set_index(["variable", "group"]))

    length_histogram(columns["raw_words"], train.labels, "Raw words per review (train)", "raw words",
                     eda_dir / "length_raw_words.png")
    length_histogram(columns["tokens"], train.labels, "Cleaned tokens per review (train)", "cleaned tokens",
                     eda_dir / "length_tokens.png")
    return table


def truncation_table(train, eda_dir, log):
    n = train.n_tokens.astype(np.int64)
    rows = []
    for max_len in MAX_LEN_CANDIDATES:
        cut = n > max_len
        lost = (n[cut] - max_len) / n[cut]
        rows.append({"max_len": max_len, "pct_reviews_cut": round(100 * float(cut.mean()), 2),
                     "avg_pct_tokens_lost_in_cut": round(100 * float(lost.mean()), 2) if cut.any() else 0.0})
    table = pd.DataFrame(rows)
    table.to_csv(eda_dir / "truncation_table.csv", index=False)
    show_table(log, "Truncation (train)", table.set_index("max_len"))
    return table


# ---------------------------------------------------------------------------
# f) most class-associated words
# ---------------------------------------------------------------------------

def top_words(train, vocab, eda_dir, log, min_count=50, top_n=20):
    id_to_word = {i: w for w, i in vocab.items()}
    lengths = np.diff(train.offsets)
    token_labels = np.repeat(train.labels, lengths)  # label of every token
    size = len(vocab)
    counts = {c: np.bincount(train.ids[token_labels == c], minlength=size).astype(np.float64) for c in (0, 1)}
    total = counts[0] + counts[1]
    # add-1 smoothed log-odds: log P(word | positive) - log P(word | negative)
    p1 = (counts[1] + 1) / (counts[1].sum() + size)
    p0 = (counts[0] + 1) / (counts[0].sum() + size)
    log_odds = np.log(p1) - np.log(p0)
    ok = (total >= min_count) & (np.arange(size) >= len(SPECIAL_TOKENS))
    candidates = np.where(ok)[0]
    ordered = candidates[np.argsort(log_odds[candidates])]
    rows = []
    for label, chosen in ((0, ordered[:top_n]), (1, ordered[::-1][:top_n])):
        for i in chosen:
            rows.append({"class": CLASS_NAMES[label], "word": id_to_word[i], "log_odds_pos_vs_neg": round(float(log_odds[i]), 3),
                         "count_neg": int(counts[0][i]), "count_pos": int(counts[1][i])})
    table = pd.DataFrame(rows)
    table.to_csv(eda_dir / "top_words.csv", index=False)
    result = {name: table[table["class"] == name]["word"].tolist() for name in ("negative", "positive")}
    log("\n=== Top words per class (train, count >= 50, log-odds) ===")
    log("negative: " + ", ".join(result["negative"]))
    log("positive: " + ", ".join(result["positive"]))
    return result


# ---------------------------------------------------------------------------
# g) quality checks
# ---------------------------------------------------------------------------

def quality_checks(data, raw_train_texts, raw_train_all, test_texts, log):
    train_set = set(raw_train_texts)
    result = {
        "train_reviews": len(raw_train_texts),
        "train_duplicate_extra_copies": len(raw_train_texts) - len(train_set),
        "test_reviews_also_in_train_rows": sum(t in train_set for t in test_texts),
        "reviews_under_5_tokens": {n: int((data[n].n_tokens < 5).sum()) for n in ("train", "val", "test")},
        "oov_token_percent": {n: data["info"]["oov_token_percent"][n] for n in ("val", "test")},
    }
    full_set = set(raw_train_all)  # all 560,000 original train reviews, including those used for validation
    result["test_reviews_also_in_full_original_train_split"] = sum(t in full_set for t in test_texts)
    log("\n=== Quality checks ===")
    for key, value in result.items():
        log(f"{key}: {value}")
    return result


# ---------------------------------------------------------------------------
# h) lemmatizer changes
# ---------------------------------------------------------------------------

def lemmatizer_changes(texts, cfg, eda_dir, log, top_n=40):
    keep_digits = cfg["data"]["keep_digits"]
    changes = Counter()
    for text in texts:
        plain_tokens = clean_text(text, lemmatize=False, keep_digits=keep_digits)
        lemma_tokens = clean_text(text, lemmatize=True, keep_digits=keep_digits)
        changes.update((w, l) for w, l in zip(plain_tokens, lemma_tokens) if w != l)
    table = pd.DataFrame([{"word": w, "lemma": l, "count": c} for (w, l), c in changes.most_common(top_n)])
    table.to_csv(eda_dir / "lemmatizer_changes.csv", index=False)
    show_table(log, f"Most frequent lemmatizer changes (first {len(texts)} train reviews)", table)
    return table.to_dict(orient="records")


# ---------------------------------------------------------------------------
# i) slices
# ---------------------------------------------------------------------------

def slice_sizes(test_csv_path, eda_dir, log):
    test_df = pd.read_csv(test_csv_path, dtype={"text": str}, keep_default_na=False)
    masks = make_slices(test_df["text"].tolist(), test_df["raw_words"].to_numpy())
    labels = test_df["label"].to_numpy()
    rows = [{"slice": "all", "reviews": len(test_df), "share_pct": 100.0, "positive_share": round(float(labels.mean()), 3)}]
    for name, mask in masks.items():
        rows.append({"slice": name, "reviews": int(mask.sum()), "share_pct": round(100 * float(mask.mean()), 2),
                     "positive_share": round(float(labels[mask].mean()), 3) if mask.any() else None})
    table = pd.DataFrame(rows)
    table.to_csv(eda_dir / "slice_sizes.csv", index=False)
    show_table(log, "Evaluation slice sizes (test)", table.set_index("slice"))
    return table.to_dict(orient="records")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA on the processed Yelp data")
    parser.add_argument("--config", default="local")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    eda_dir = cfg["paths"]["output_dir"] / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(cfg["paths"]["output_dir"] / "eda.log")
    log = logger.log
    log(f"EDA for config '{args.config}'")

    data = get_data(cfg, logger)
    train = data["train"]
    summary = {"config": args.config, "info": data["info"]}

    summary["class_balance"] = class_balance(data, eda_dir, log)
    summary["length_stats"] = length_stats(train, eda_dir, log).to_dict(orient="records")
    summary["truncation"] = truncation_table(train, eda_dir, log).to_dict(orient="records")
    summary["top_words"] = top_words(train, data["vocab"], eda_dir, log)

    # raw texts come from the cached Hugging Face dataset (the processed folder only keeps test text)
    raw = load_raw_dataset(cfg["data"]["dataset"])
    raw_train_all = raw["train"]["text"]
    raw_train_texts = [raw_train_all[i] for i in train.orig_index]
    test_texts = pd.read_csv(cfg["paths"]["processed_dir"] / "test_raw.csv", dtype={"text": str},
                             keep_default_na=False)["text"].tolist()
    summary["quality_checks"] = quality_checks(data, raw_train_texts, raw_train_all, test_texts, log)
    save_json(plain(summary["quality_checks"]), eda_dir / "quality_checks.json")

    summary["lemmatizer_changes"] = lemmatizer_changes(raw_train_texts[:5000], cfg, eda_dir, log)
    summary["slice_sizes"] = slice_sizes(cfg["paths"]["processed_dir"] / "test_raw.csv", eda_dir, log)

    save_json(plain(summary), eda_dir / "eda_summary.json")
    log(f"\nSaved EDA files to {eda_dir.relative_to(cfg['paths']['output_dir'].parent.parent).as_posix()}")


if __name__ == "__main__":
    main()
