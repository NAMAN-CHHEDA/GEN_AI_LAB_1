"""Evaluation slices: boolean masks that pick out groups of reviews."""

import re

import numpy as np

NEGATION_RE = re.compile(r"\b(not|no|never)\b|n't")
CONTRAST_RE = re.compile(r"\b(but|however|although)\b")


def make_slices(raw_texts, raw_words):
    """Return {slice name: boolean array}, one entry per review. Uses the raw (uncleaned) text."""
    raw_words = np.asarray(raw_words)
    lowered = [t.lower().replace("’", "'") for t in raw_texts]  # curly apostrophes -> n't still matches
    return {
        "short": raw_words < 50,
        "medium": (raw_words >= 50) & (raw_words <= 200),
        "long": raw_words > 200,
        "has_negation": np.array([NEGATION_RE.search(t) is not None for t in lowered]),
        "has_contrast": np.array([CONTRAST_RE.search(t) is not None for t in lowered]),
    }
