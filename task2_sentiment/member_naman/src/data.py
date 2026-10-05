"""Data step: load Yelp Polarity, validate, clean, split, build the vocabulary and encode.

Run:  python src/data.py --config local        (add --force to rebuild)
"""

import argparse
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import utils` works from anywhere

from utils import Logger, load_config, load_json, log_config, project_root, save_json

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PAD, UNK, BOS = 0, 1, 2  # BOS is reserved for the bigram of the first position later
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>"]

# Only neutral function words are removed. Negations and intensifiers carry sentiment, so keep them.
STOPWORDS = frozenset("""
a an the and or of to in on at by for with from as is are was were be been being am
it its this that these those there here has have had do does did
""".split())

# Bump this whenever clean_text changes, so old cached data is rebuilt.
CLEANING_VERSION = 2

# Words the lemmatizer must NOT change (it would turn "us" into "u", "vegas" into "vega", ...).
LEMMA_EXCEPTIONS = frozenset({"us", "vegas", "las", "mrs"})

# Config keys that change the processed data. (max_len and head_fraction only matter later, at training time.)
CACHE_KEYS = ["dataset", "train_samples", "val_samples", "val_fraction", "test_samples", "vocab_max",
              "min_freq", "lemmatize", "keep_digits", "stopwords"]

# Words that must never be treated as stopwords (checked as soon as this file is imported).
PROTECTED_WORDS = frozenset("""
not no nor never but however although very too so quite only just more most less least few
than against without i we you he she they
""".split())
assert not (PROTECTED_WORDS & STOPWORDS), f"protected words in STOPWORDS: {PROTECTED_WORDS & STOPWORDS}"


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------

# One escape: a surrogate pair, a single \uXXXX, or one of \n \r \t \" \\
_ESCAPE_RE = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"  # surrogate pair
    r"|\\u([0-9a-fA-F]{4})"                                            # single \uXXXX
    r'|\\([nrt"\\])'                                                    # simple escapes
)


def _decode_one_escape(match):
    high, low, single, simple = match.groups()
    if high is not None:  # join a surrogate pair into one real character
        return chr(0x10000 + ((int(high, 16) - 0xD800) << 10) + (int(low, 16) - 0xDC00))
    if single is not None:
        code = int(single, 16)
        if 0xD800 <= code <= 0xDFFF:  # a lone surrogate is not a real character
            return " "
        return chr(code)
    if simple in "nrt":
        return " "
    return simple  # \" -> "   and   \\ -> \


def decode_escapes(text):
    """Turn literal escape sequences (backslash + letter) into real characters."""
    try:
        return _ESCAPE_RE.sub(_decode_one_escape, text)
    except (ValueError, OverflowError):
        return text


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&#?\w+;")
_CURLY_APOSTROPHE_RE = re.compile("[‘’ʼ`´]")
_SUFFIX_RE = re.compile(r"'(?:re|s|m|ll|ve|d)\b")  # it's, we're, I'm, they'll, I've, he'd
_NOT_RE = re.compile(r"n't\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


@lru_cache(maxsize=None)
def _lemmatizer():
    """Create the WordNet lemmatizer, downloading the NLTK data on first use."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    for package in ("wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)
    return WordNetLemmatizer()


@lru_cache(maxsize=None)
def lemmatize_word(word):
    """Lemma of one word (cached, because the same words appear again and again)."""
    if word in LEMMA_EXCEPTIONS:
        return word
    return _lemmatizer().lemmatize(word)


def clean_text(text, lemmatize, keep_digits):
    """Raw review -> list of lowercase tokens."""
    text = decode_escapes(text.lower())
    text = _CURLY_APOSTROPHE_RE.sub("'", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)

    # Contractions: special cases first, then the general n't rule, then the other suffixes
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("cannot", "can not").replace("ain't", "is not")
    text = _NOT_RE.sub(" not", text)
    text = _SUFFIX_RE.sub("", text)

    text = (_NON_ALNUM_RE if keep_digits else _NON_ALPHA_RE).sub(" ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    if lemmatize:
        tokens = [lemmatize_word(w) for w in tokens]
    return [sys.intern(w) for w in tokens]  # intern: equal words share one string object (saves memory)


# ----------------------------------------------------------------------------
# Raw data: loading and validation
# ----------------------------------------------------------------------------

def hf_cache_dir():
    return project_root().parent / "data" / "hf_cache"


def load_raw_dataset(dataset_name):
    from datasets import load_dataset
    return load_dataset(dataset_name, cache_dir=str(hf_cache_dir()))


def print_label_check(raw_split, log):
    """Show 2 raw reviews per label so a human can check which label means what."""
    log("Label check (first 2 train reviews for each label, first 200 chars):")
    shown = {0: 0, 1: 0}
    samples = []
    for text, label in zip(raw_split["text"], raw_split["label"]):
        if label in shown and shown[label] < 2:
            shown[label] += 1
            line = f"  label={label}: {text[:200]!r}"
            samples.append(line)
            log(line)
        if all(n == 2 for n in shown.values()):
            break
    return samples


def validate_split(raw_split, name):
    """Drop bad rows. Returns the clean rows plus a report with the drop and escape counts."""
    texts, labels, orig_index = [], [], []
    dropped = {"not_a_string": 0, "empty": 0, "bad_label": 0}
    escapes = {"newline": 0, "quote": 0, "unicode": 0, "any": 0}
    for i, (text, label) in enumerate(zip(raw_split["text"], raw_split["label"])):
        if not isinstance(text, str):
            dropped["not_a_string"] += 1
        elif not text.strip():
            dropped["empty"] += 1
        elif label not in (0, 1):
            dropped["bad_label"] += 1
        else:
            texts.append(text)
            labels.append(label)
            orig_index.append(i)
            has_n = "\\n" in text
            has_q = '\\"' in text
            has_u = re.search(r"\\u[0-9a-fA-F]{4}", text) is not None
            escapes["newline"] += has_n
            escapes["quote"] += has_q
            escapes["unicode"] += has_u
            escapes["any"] += has_n or has_q or has_u
    report = {"rows_in_split": len(raw_split), "rows_kept": len(texts),
              "dropped": dropped, "reviews_with_escapes": escapes}
    return texts, np.array(labels, dtype=np.uint8), np.array(orig_index, dtype=np.int64), report


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def split_train_val(n_rows, seed, train_samples, val_samples, val_fraction):
    """Positions (into the valid train rows) for train and validation. They never overlap.

    Validation is carved out of a seeded permutation first, then train is taken from the rest.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    n_val = val_samples if val_samples is not None else round(val_fraction * n_rows)
    val_pos = permutation[:n_val]
    train_pos = permutation[n_val:]
    if train_samples is not None:
        assert train_samples <= len(train_pos), "train_samples + validation is more than the train split"
        train_pos = train_pos[:train_samples]
    return train_pos, val_pos


def choose_test(n_rows, seed, test_samples):
    """Positions of the test reviews. A random subset is sorted so the original order is kept."""
    if test_samples is None or test_samples >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed + 1)  # a different stream from the train/val permutation
    return np.sort(rng.choice(n_rows, size=test_samples, replace=False))


# ----------------------------------------------------------------------------
# Vocabulary and encoding
# ----------------------------------------------------------------------------

def build_vocab(word_counts, min_freq, vocab_max):
    """word -> id. Specials first, then words with count >= min_freq, most frequent first."""
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    candidates = [(w, c) for w, c in word_counts.items() if c >= min_freq]
    candidates.sort(key=lambda wc: (-wc[1], wc[0]))
    for word, _ in candidates[: vocab_max - len(SPECIAL_TOKENS)]:
        vocab[word] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab):
    """Tokens -> ids (unknown words become UNK). An empty review becomes a single UNK."""
    if not tokens:
        return [UNK]
    return [vocab.get(w, UNK) for w in tokens]


def encode_split(token_lists, vocab):
    """Encode all reviews. Returns a list of int32 arrays, the token counts and OOV statistics."""
    id_arrays, n_tokens = [], []
    oov, total = 0, 0
    for tokens in token_lists:
        ids = np.array(encode_tokens(tokens, vocab), dtype=np.int32)
        id_arrays.append(ids)
        n_tokens.append(len(tokens))
        if tokens:  # the filler UNK of an empty review is not counted as OOV
            oov += int((ids == UNK).sum())
            total += len(ids)
    return id_arrays, np.array(n_tokens, dtype=np.int32), oov, total


def to_csr(id_arrays):
    """Flat id array plus offsets: review i is ids[offsets[i]:offsets[i+1]]."""
    lengths = np.array([len(a) for a in id_arrays], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return np.concatenate(id_arrays).astype(np.int32), offsets


class SplitData:
    """One split (train, val or test) read back from arrays.npz."""

    def __init__(self, arrays, name):
        self.name = name
        self.ids = arrays[f"{name}_ids"]
        self.offsets = arrays[f"{name}_offsets"]
        self.labels = arrays[f"{name}_labels"]
        self.orig_index = arrays[f"{name}_orig_index"]
        self.raw_words = arrays[f"{name}_raw_words"]
        self.n_tokens = arrays[f"{name}_n_tokens"]

    def __len__(self):
        return len(self.labels)

    def review(self, i):
        """Token ids of review i (no truncation) as a numpy array."""
        return self.ids[self.offsets[i]:self.offsets[i + 1]]


# ----------------------------------------------------------------------------
# Building and caching
# ----------------------------------------------------------------------------

def class_counts(labels):
    counts = np.bincount(labels, minlength=2)
    return {"label_0": int(counts[0]), "label_1": int(counts[1])}


def build_data(cfg, logger):
    """Do the whole pipeline once and write everything to processed_dir."""
    log = logger.log
    t_start = time.time()
    dcfg, seed = cfg["data"], cfg["seed"]
    if dcfg["stopwords"] != "minimal":
        raise ValueError("only stopwords: minimal is implemented")
    processed_dir = cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {dcfg['dataset']} (cache: {hf_cache_dir().relative_to(project_root().parent).as_posix()})")
    raw = load_raw_dataset(dcfg["dataset"])
    label_check = print_label_check(raw["train"], log)

    # 1) validate raw rows
    train_texts_all, train_labels_all, train_orig_all, train_report = validate_split(raw["train"], "train")
    test_texts_all, test_labels_all, test_orig_all, test_report = validate_split(raw["test"], "test")
    for name, rep in (("train", train_report), ("test", test_report)):
        log(f"Validation {name}: {rep['rows_in_split']} rows, kept {rep['rows_kept']}, "
            f"dropped {rep['dropped']}, reviews with literal escapes {rep['reviews_with_escapes']}")

    # 2) splits (positions into the valid rows)
    train_pos, val_pos = split_train_val(len(train_texts_all), seed, dcfg["train_samples"],
                                         dcfg["val_samples"], dcfg["val_fraction"])
    test_pos = choose_test(len(test_texts_all), seed, dcfg["test_samples"])
    selected = {
        "train": (train_texts_all, train_labels_all, train_orig_all, train_pos),
        "val": (train_texts_all, train_labels_all, train_orig_all, val_pos),
        "test": (test_texts_all, test_labels_all, test_orig_all, test_pos),
    }
    texts = {name: [t[p] for p in pos] for name, (t, _, _, pos) in selected.items()}
    labels = {name: l[pos] for name, (_, l, _, pos) in selected.items()}
    orig_index = {name: o[pos] for name, (_, _, o, pos) in selected.items()}

    # 3) clean (the clock only runs for cleaning, to estimate the cost of the full dataset)
    tokens, clean_seconds = {}, 0.0
    for name in ("train", "val", "test"):
        t0 = time.time()
        tokens[name] = [clean_text(t, dcfg["lemmatize"], dcfg["keep_digits"]) for t in texts[name]]
        clean_seconds += time.time() - t0
        log(f"Cleaned {name}: {len(tokens[name])} reviews")
    n_cleaned = sum(len(v) for v in tokens.values())
    reviews_per_sec = n_cleaned / clean_seconds
    log(f"Cleaning speed: {reviews_per_sec:.0f} reviews/s ({n_cleaned} reviews in {clean_seconds:.1f}s)")

    # 4) vocabulary from TRAIN tokens only
    train_counts = Counter(w for toks in tokens["train"] for w in toks)
    vocab = build_vocab(train_counts, dcfg["min_freq"], dcfg["vocab_max"])
    log(f"Vocabulary: {len(vocab)} ids (from {len(train_counts)} distinct train words)")

    # 5) encode (no truncation)
    arrays, oov_pct, empty_after_cleaning = {}, {}, {}
    for name in ("train", "val", "test"):
        id_arrays, n_tokens, oov, total = encode_split(tokens[name], vocab)
        ids, offsets = to_csr(id_arrays)
        arrays[f"{name}_ids"] = ids
        arrays[f"{name}_offsets"] = offsets
        arrays[f"{name}_labels"] = labels[name]
        arrays[f"{name}_orig_index"] = orig_index[name]
        arrays[f"{name}_raw_words"] = np.array([min(len(t.split()), 65535) for t in texts[name]],
                                               dtype=np.uint16)
        arrays[f"{name}_n_tokens"] = n_tokens
        oov_pct[name] = round(100 * oov / max(total, 1), 3)
        empty_after_cleaning[name] = int((n_tokens == 0).sum())

    # 6) write everything
    np.savez(processed_dir / "arrays.npz", **arrays)
    save_json(vocab, processed_dir / "vocab.json")
    pd.DataFrame({
        "orig_index": orig_index["test"], "text": texts["test"],
        "label": labels["test"], "raw_words": arrays["test_raw_words"],
    }).to_csv(processed_dir / "test_raw.csv", index=False)
    with open(processed_dir / "examples.txt", "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(f"--- train review {i} (label {labels['train'][i]}) ---\n")
            f.write(f"RAW:     {texts['train'][i]}\n")
            f.write(f"CLEANED: {' '.join(tokens['train'][i])}\n\n")

    build_seconds = time.time() - t_start
    info = {
        "data_config": dcfg,
        "seed": seed,
        "cache_key": cache_key(cfg),
        "vocab_size": len(vocab),
        "counts": {name: len(labels[name]) for name in labels},
        "class_counts": {name: class_counts(labels[name]) for name in labels},
        "dropped_rows": {"train": train_report["dropped"], "test": test_report["dropped"]},
        "rows_in_raw_split": {"train": train_report["rows_in_split"], "test": test_report["rows_in_split"]},
        "reviews_with_escapes": {"train": train_report["reviews_with_escapes"],
                                 "test": test_report["reviews_with_escapes"]},
        "empty_after_cleaning": empty_after_cleaning,
        "oov_token_percent": oov_pct,
        "mean_tokens_per_review": {n: round(float(arrays[f"{n}_n_tokens"].mean()), 2) for n in labels},
        "clean_seconds": round(clean_seconds, 2),
        "cleaning_reviews_per_sec": round(reviews_per_sec, 1),
        "build_seconds": round(build_seconds, 2),
        "label_check": label_check,
    }
    save_json(info, processed_dir / "split_info.json")  # written last: its presence marks a finished build
    log(f"Build finished in {build_seconds:.1f}s")


def cache_key(cfg):
    """The settings that decide what the processed data looks like."""
    key = {name: cfg["data"][name] for name in CACHE_KEYS}
    key["seed"] = cfg["seed"]
    key["cleaning_version"] = CLEANING_VERSION
    return key


def cache_is_valid(cfg):
    """True if processed_dir holds a finished build made with the same cache key."""
    processed_dir = cfg["paths"]["processed_dir"]
    needed = ["split_info.json", "arrays.npz", "vocab.json"]
    if not all((processed_dir / name).exists() for name in needed):
        return False
    info = load_json(processed_dir / "split_info.json")
    return info.get("cache_key") == cache_key(cfg)


def get_data(cfg, logger=None, force=False):
    """Return {"train", "val", "test": SplitData, "vocab": dict, "info": dict}, building if needed."""
    logger = logger or Logger(cfg["paths"]["output_dir"] / "data_build.log")
    if force or not cache_is_valid(cfg):
        build_data(cfg, logger)
    else:
        logger.log("Reusing cached processed data (same cache key)")
    processed_dir = cfg["paths"]["processed_dir"]
    with np.load(processed_dir / "arrays.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}
    return {
        "train": SplitData(arrays, "train"),
        "val": SplitData(arrays, "val"),
        "test": SplitData(arrays, "test"),
        "vocab": load_json(processed_dir / "vocab.json"),
        "info": load_json(processed_dir / "split_info.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Build (or reuse) the processed Yelp data")
    parser.add_argument("--config", default="local", help="config name in configs/ (local or gpu)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is valid")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles choke on some characters
    cfg = load_config(args.config)
    logger = Logger(cfg["paths"]["output_dir"] / "data_build.log")
    log_config(logger, cfg)
    data = get_data(cfg, logger, force=args.force)

    info = data["info"]
    logger.log("--- Summary ---")
    logger.log(f"vocab size: {info['vocab_size']}")
    for name in ("train", "val", "test"):
        logger.log(f"{name}: {info['counts'][name]} reviews, classes {info['class_counts'][name]}, "
                   f"mean tokens {info['mean_tokens_per_review'][name]}, "
                   f"OOV {info['oov_token_percent'][name]}%, empty after cleaning {info['empty_after_cleaning'][name]}")
    logger.log(f"example: train review 0 has {len(data['train'].review(0))} token ids")


if __name__ == "__main__":
    main()
