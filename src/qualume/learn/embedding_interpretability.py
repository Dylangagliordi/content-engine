"""Shared embedding-interpretability primitives — the ranking logic common
to every "make this opaque PCA axis human-readable" fix built this session
(okn_mirror.ml_engine_okn, learn.text_embedding_features, and now
learn.stored_embedding_features). Extracted here once both the text-keyword
and metadata-contrast variants needed the same high/low-projection ranking
step, rather than tripling the sort/slice logic across three modules.

okn_mirror deliberately does NOT import this (it's a standalone mirror of
upstream okn-analytics' own code, kept dependency-free from qualume's own
learn/* package by design) — its copy stays a verbatim duplicate on purpose.
"""
import numpy as np


def rank_high_low_indices(embeddings: np.ndarray, component_vector: np.ndarray,
                           n_examples: int = 5) -> tuple[list[int], list[int]]:
    """Project every row of `embeddings` onto one PCA component direction,
    sort, and return the (high_indices, low_indices) at the two extremes —
    the shared first step behind every embedding-interpretability technique
    in this codebase. Returns ([], []) if there's nothing meaningful to
    split (fewer than 2 real rows)."""
    projections = embeddings @ component_vector
    order = np.argsort(projections)
    n = min(n_examples, len(order) // 2) if len(order) >= 2 else 0
    if n == 0:
        return [], []
    high_idx = list(order[-n:][::-1])
    low_idx = list(order[:n])
    return high_idx, low_idx


def tfidf_keywords(texts: list[str], n_keywords: int = 5) -> list[str]:
    """Top TF-IDF keywords for one group of texts (the high or low side of
    an axis) — shared helper, same technique used independently in
    okn_mirror, text_embedding_features (pre-refactor), and now here."""
    non_empty = [t for t in texts if len(str(t).strip()) > 2]
    if len(non_empty) < 2:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=n_keywords, stop_words="english",
                                 token_pattern=r"(?u)\b[#\w][\w]{2,}\b")
        tfidf.fit(non_empty)
        return list(tfidf.get_feature_names_out())
    except Exception:
        return []


def mode_contrast(metadata_high: list[dict], metadata_low: list[dict], keys: list[str]) -> dict:
    """For embeddings with no associated free text (e.g. visual frame
    embeddings — nothing was ever embedded AS text), contrast structured
    metadata instead: for each key, the most common value among the HIGH
    group vs the most common value among the LOW group. A real, grounded
    interpretation technique for non-text embeddings — not as rich as a
    keyword contrast, but genuinely read as "high = mostly X, low = mostly
    Y" rather than an opaque number.

    Returns {key: {"high_mode": value_or_None, "high_count": int,
    "low_mode": value_or_None, "low_count": int}} for every key, even if
    one side has no non-null values for it (counts/modes are None/0 then,
    not fabricated)."""
    from collections import Counter

    result = {}
    for key in keys:
        high_vals = [m.get(key) for m in metadata_high if m.get(key)]
        low_vals = [m.get(key) for m in metadata_low if m.get(key)]
        high_counter = Counter(high_vals)
        low_counter = Counter(low_vals)
        high_mode, high_count = high_counter.most_common(1)[0] if high_counter else (None, 0)
        low_mode, low_count = low_counter.most_common(1)[0] if low_counter else (None, 0)
        result[key] = {
            "high_mode": high_mode, "high_count": high_count,
            "low_mode": low_mode, "low_count": low_count,
        }
    return result
