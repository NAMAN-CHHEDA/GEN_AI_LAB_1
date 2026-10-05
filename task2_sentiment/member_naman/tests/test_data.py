"""Plain-assert tests for data.py. Run:  python tests/test_data.py   (no pytest needed)."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import copy
import tempfile

from data import (CACHE_KEYS, PAD, cache_is_valid, cache_key, PROTECTED_WORDS, STOPWORDS, UNK, BOS, build_vocab, clean_text,
                  encode_tokens, split_train_val)


def clean(text):
    return clean_text(text, lemmatize=True, keep_digits=True)


def test_negation_kept_and_no_punctuation():
    tokens = clean("I can't recommend it, wasn't good!!")
    assert "not" in tokens, tokens
    assert all(t.isalnum() for t in tokens), tokens


def test_wont():
    assert clean("won't")[:2] == ["will", "not"], clean("won't")


def test_sentiment_words_survive():
    tokens = clean("but very too never")
    assert tokens == ["but", "very", "too", "never"], tokens


def test_digits_and_lemma():
    tokens = clean("We gave it 2 stars")
    assert "2" in tokens and "star" in tokens, tokens


def test_url_and_html_removed():
    tokens = clean("Nice <b>place</b> see https://example.com/x?y=1 and www.foo.com &amp; more")
    joined = " ".join(tokens)
    assert "http" not in joined and "www" not in joined and "<" not in joined and "amp" not in joined, tokens
    assert "nice" in tokens and "place" in tokens, tokens


def test_escapes_decoded():
    tokens = clean("Great\\nfood \\u00fc")
    assert tokens == ["great", "food"], tokens  # the decoded u-umlaut is not a-z, so it is dropped
    from data import decode_escapes
    assert decode_escapes("Great\\nfood \\u00fc") == "Great food ü"
    assert decode_escapes('say \\"hi\\" \\\\') == 'say "hi" \\'
    assert decode_escapes("\\ud83d\\ude00") == "\U0001F600"  # surrogate pair -> one character


def test_lemma_exceptions():
    tokens = clean("us vegas las mrs stars")
    assert tokens == ["us", "vegas", "las", "mrs", "star"], tokens


def test_stopwords_protected():
    assert not (PROTECTED_WORDS & STOPWORDS)
    for word in ("not", "no", "nor", "never", "but", "very", "too", "so", "i", "they", "without"):
        assert word not in STOPWORDS, word


def test_vocab_train_only():
    train = [["good", "food", "good"], ["bad", "food"]]
    val = [["terrible", "food"]]
    vocab = build_vocab(Counter(w for toks in train for w in toks), min_freq=1, vocab_max=100)
    assert vocab["<pad>"] == PAD == 0 and vocab["<unk>"] == UNK == 1 and vocab["<bos>"] == BOS == 2
    assert "good" in vocab and "bad" in vocab and "food" in vocab
    assert "terrible" not in vocab
    assert encode_tokens(val[0], vocab)[0] == UNK
    assert min(v for w, v in vocab.items() if not w.startswith("<")) == 3


def test_empty_review_is_single_unk():
    assert encode_tokens([], {"<pad>": 0, "<unk>": 1, "<bos>": 2}) == [UNK]
    assert clean("!!! ??? the") == []  # cleans to nothing, which encode_tokens turns into [UNK]


def test_split_is_repeatable_and_disjoint():
    a_train, a_val = split_train_val(1000, 42, 300, 100, 0.1)
    b_train, b_val = split_train_val(1000, 42, 300, 100, 0.1)
    assert (a_train == b_train).all() and (a_val == b_val).all()
    assert len(a_train) == 300 and len(a_val) == 100
    assert not (set(a_train.tolist()) & set(a_val.tolist()))
    full_train, full_val = split_train_val(1000, 42, None, None, 0.1)  # val_fraction path
    assert len(full_val) == 100 and len(full_train) == 900
    assert not (set(full_train.tolist()) & set(full_val.tolist()))


def _fake_cache(cfg, folder):
    """Make a processed_dir that looks like a finished build for cfg."""
    folder = Path(folder)
    for name in ("arrays.npz", "vocab.json"):
        (folder / name).write_text("x")
    (folder / "split_info.json").write_text(json.dumps({"cache_key": cache_key(cfg)}))


def _toy_cfg(folder):
    data = {"dataset": "d", "train_samples": 5, "val_samples": 2, "val_fraction": 0.1, "test_samples": 3,
            "vocab_max": 100, "min_freq": 3, "lemmatize": True, "keep_digits": True,
            "stopwords": "minimal", "max_len": 128, "head_fraction": 0.25}
    return {"seed": 42, "paths": {"processed_dir": Path(folder)}, "data": data}


def test_cache_ignores_max_len():
    with tempfile.TemporaryDirectory() as folder:
        cfg = _toy_cfg(folder)
        _fake_cache(cfg, folder)
        assert cache_is_valid(cfg)
        other = copy.deepcopy(cfg)
        other["data"]["max_len"] = 256
        other["data"]["head_fraction"] = 0.5
        assert cache_is_valid(other)


def test_cache_invalidated_by_min_freq():
    with tempfile.TemporaryDirectory() as folder:
        cfg = _toy_cfg(folder)
        _fake_cache(cfg, folder)
        other = copy.deepcopy(cfg)
        other["data"]["min_freq"] = 5
        assert not cache_is_valid(other)


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:  # AssertionError or any other error counts as a failure
            failed += 1
            print(f"FAIL: {name} -> {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
