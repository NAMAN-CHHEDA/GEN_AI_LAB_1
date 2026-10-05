#!/usr/bin/env python3
"""Rebuild Task 2 vocab with the same recipe as the training notebooks and save
artifacts under data_processed/. Does not retrain models.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MEMBER_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = MEMBER_DIR / "data_processed"
CACHE_DIR = MEMBER_DIR.parent / "data" / "hf_cache"

VOCAB_SIZE = 30000
MAX_LEN = 160
SEED = 42
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1


def ensure_nltk() -> None:
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> list[str]:
    text = text.lower()
    text = text.replace("\\n", " ").replace("\\r", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]


def main() -> None:
    ensure_nltk()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stop_words = set(stopwords.words("english")) - {"no", "not", "nor", "never"}
    lemmatizer = WordNetLemmatizer()

    print("Loading fancyzhx/yelp_polarity ...")
    dataset = load_dataset("fancyzhx/yelp_polarity", cache_dir=str(CACHE_DIR))
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.10,
        random_state=SEED,
        stratify=train_df["label"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_split = test_df.reset_index(drop=True)

    print("Preprocessing train split for vocabulary ...")
    train_tokens = [
        preprocess_text(t, stop_words, lemmatizer)
        for t in tqdm(train_split["text"], desc="Preprocess train")
    ]

    word_counts: Counter[str] = Counter()
    for tokens in tqdm(train_tokens, desc="Building vocabulary"):
        word_counts.update(tokens)

    word_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in word_counts.most_common(VOCAB_SIZE - 2):
        word_to_idx[word] = len(word_to_idx)

    lengths = pd.Series([len(t) for t in train_tokens])
    split_sizes = {
        "train": int(len(train_split)),
        "val": int(len(val_split)),
        "test": int(len(test_split)),
        "max_len": MAX_LEN,
        "vocab_size": int(len(word_to_idx)),
        "vocab_cap": VOCAB_SIZE,
        "seed": SEED,
        "length_p50": float(lengths.quantile(0.50)),
        "length_p90": float(lengths.quantile(0.90)),
        "length_p95": float(lengths.quantile(0.95)),
        "train_label_counts": train_split["label"].value_counts().sort_index().to_dict(),
        "val_label_counts": val_split["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test_split["label"].value_counts().sort_index().to_dict(),
    }

    vocab_path = OUT_DIR / "vocab.json"
    sizes_path = OUT_DIR / "split_sizes.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(word_to_idx, f)
    with sizes_path.open("w", encoding="utf-8") as f:
        json.dump(split_sizes, f, indent=2)

    # Refresh preprocessing_info with concrete vocab artifact paths
    info_path = OUT_DIR / "preprocessing_info.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())
    info["status"] = "vocab_and_split_metadata_saved"
    info["artifacts"] = {
        "vocab_json": "task2_sentiment/sarvesh/data_processed/vocab.json",
        "split_sizes_json": "task2_sentiment/sarvesh/data_processed/split_sizes.json",
        "preprocessing_info_json": "task2_sentiment/sarvesh/data_processed/preprocessing_info.json",
    }
    info["vocabulary"] = {
        "built_from": "training split only",
        "special_tokens": [PAD_TOKEN, UNK_TOKEN],
        "max_len": MAX_LEN,
        "vocab_size": len(word_to_idx),
        "vocab_cap": VOCAB_SIZE,
        "seed": SEED,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    print("Wrote", vocab_path, f"({len(word_to_idx)} tokens)")
    print("Wrote", sizes_path)
    print("Wrote", info_path)


if __name__ == "__main__":
    main()
