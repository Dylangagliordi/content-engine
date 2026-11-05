"""Stored (precomputed) embedding features — transcript_embeddings and
visual_embeddings (Voyage voyage-3.5 / voyage-multimodal-3.5, 1024-dim,
already computed and persisted by learn.transcript_embedding /
learn.visual_embedding for HDBSCAN clustering) folded into the SAME
prediction model as learn.text_embedding_features's hook_text embeddings
(sentence-transformers) and learn.features's engineered video-content
stats — the full three-source merge requested after the okn-vs-qualume
comparison: "how can we use all [of them] in the prediction model and
have them all be interpretable".

WHY these are PRECOMPUTED, not re-embedded here (unlike hook_text, which
is cheap to encode on-the-fly with a local model): transcript/visual
embeddings already exist as real DB rows from a real paid Voyage API
call (transcript_embeddings/visual_embeddings tables) — re-calling the
API here would duplicate real cost for no benefit. This module only
FITS A PCA REDUCTION over whichever already-stored vectors happen to be
available for a given training set; it never calls Voyage itself.

Real coverage gap, not glossed over: not every training item has a
stored transcript/visual embedding (those pipelines ran independently,
on their own schedule, against a different real-world backlog than
annotation). A row missing either vector contributes None for those
features — same "refuse rather than guess" discipline as everywhere
else, not a fabricated zero-vector standing in for "we don't know".

INTERPRETABILITY, the two different real techniques needed here:
- transcript axis: the ORIGINAL embedded text (hook_text + full_transcript,
  reconstructed via transcript_embedding.build_embeddable_text — the exact
  same join used when the vector was first created) is real text, so the
  same TF-IDF high/low keyword contrast as text_embedding_features applies
  directly.
- visual axis: nothing was ever embedded AS TEXT for visual_embeddings
  (it's raw video frames) — TF-IDF cannot apply. Instead, contrast
  structured per-item metadata already captured in shot_breakdown
  (dominant setting/shot_type/color_temperature, each item's own mode
  across its shots) between the high/low groups on that axis — a real,
  grounded interpretation technique for a non-text embedding, via
  embedding_interpretability.mode_contrast.

Fold-safety (same note as text_embedding_features): fit_stored_embeddings
is a pure fit-on-these-vectors-only function. Cross-validation must fit a
fresh instance per fold on that fold's own training vectors, never on the
full dataset.
"""
import logging
from collections import Counter
from typing import Optional

from qualume.learn.embedding_interpretability import mode_contrast, rank_high_low_indices, tfidf_keywords

logger = logging.getLogger("qualume.learn.stored_embedding_features")

MIN_VECTORS_FOR_PCA = 10
MAX_COMPONENTS = 8

VISUAL_METADATA_KEYS = ("setting", "shot_type", "color_temperature")


def dominant_shot_metadata(shot_breakdown: list[dict] | None) -> dict:
    """One item's representative visual metadata: the mode (most common
    real value) of setting/shot_type/color_temperature across its own
    shots. Used as the interpretability "aux" for a visual embedding's
    axis contrast — real, derived-from-the-actual-shots data, not a
    fabricated label. Empty dict (not None) for an item with no
    shot_breakdown — mode_contrast already handles missing keys per-group
    by simply counting fewer/zero real values, no special-casing needed
    here."""
    if not shot_breakdown:
        return {}
    result = {}
    for key in VISUAL_METADATA_KEYS:
        values = [s.get(key) for s in shot_breakdown if s.get(key)]
        if values:
            result[key] = Counter(values).most_common(1)[0][0]
    return result


def fit_stored_embeddings(vectors: list[list[float]], aux: list) -> Optional[dict]:
    """Fit a PCA reduction over already-computed embedding vectors
    (transcript_embeddings.embedding or visual_embeddings.embedding,
    already parsed from JSONB into plain float lists by the caller).
    `aux[i]` is interpretability context for `vectors[i]` — either the
    original embedded text (transcript) or a dominant_shot_metadata dict
    (visual); stored alongside so explain_*_component can look it up
    later without re-querying the DB.

    Returns None (never fabricates) below MIN_VECTORS_FOR_PCA real
    vectors — same honest-gap discipline as
    text_embedding_features.fit_hook_text_embeddings. Caller must filter
    to only the (vector, aux) pairs where vector is not None BEFORE
    calling this — this function assumes every vector passed in is real."""
    import numpy as np
    from sklearn.decomposition import PCA

    if len(vectors) < MIN_VECTORS_FOR_PCA:
        return None
    arr = np.array(vectors, dtype=float)
    n_components = min(MAX_COMPONENTS, max(2, len(vectors) // 10), arr.shape[1], len(vectors) - 1)
    if n_components < 2:
        return None
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(arr)
    return {"pca": pca, "embeddings": arr, "aux": list(aux), "n_components": n_components}


def transform_stored_embedding(fitted: dict, vector: list[float]) -> list[float]:
    """Project one (possibly new) already-computed embedding vector
    through a FITTED PCA pipeline (from fit_stored_embeddings). Returns a
    plain list of floats, length fitted['n_components']."""
    import numpy as np
    reduced = fitted["pca"].transform(np.array([vector], dtype=float))
    return [float(v) for v in reduced[0]]


def explain_transcript_component(fitted: dict, component_idx: int, n_examples: int = 5, n_keywords: int = 5) -> dict:
    """TF-IDF keyword/example contrast for one transcript-embedding PCA
    axis — `fitted["aux"]` must be the original embedded TEXT per vector
    (see fit_stored_embeddings). Same technique/shape as
    text_embedding_features.explain_component; never raises."""
    try:
        component_vector = fitted["pca"].components_[component_idx]
        high_idx, low_idx = rank_high_low_indices(fitted["embeddings"], component_vector, n_examples)
        if not high_idx:
            return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}
        texts = fitted["aux"]
        high_texts = [texts[i] for i in high_idx]
        low_texts = [texts[i] for i in low_idx]
        return {
            "high_keywords": tfidf_keywords(high_texts, n_keywords),
            "low_keywords": tfidf_keywords(low_texts, n_keywords),
            "high_examples": [str(t)[:80] for t in high_texts[:3]],
            "low_examples": [str(t)[:80] for t in low_texts[:3]],
        }
    except Exception as e:
        logger.warning("Failed to explain transcript embedding component %s: %s", component_idx, e)
        return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}


def explain_visual_component(fitted: dict, component_idx: int, n_examples: int = 8) -> dict:
    """Metadata-mode contrast for one visual-embedding PCA axis —
    `fitted["aux"]` must be dominant_shot_metadata() dicts, one per
    vector. Visual embeddings have no associated text, so this uses
    embedding_interpretability.mode_contrast instead of TF-IDF. Never
    raises."""
    try:
        component_vector = fitted["pca"].components_[component_idx]
        high_idx, low_idx = rank_high_low_indices(fitted["embeddings"], component_vector, n_examples)
        if not high_idx:
            return {key: {"high_mode": None, "high_count": 0, "low_mode": None, "low_count": 0}
                     for key in VISUAL_METADATA_KEYS}
        aux = fitted["aux"]
        high_meta = [aux[i] for i in high_idx]
        low_meta = [aux[i] for i in low_idx]
        return mode_contrast(high_meta, low_meta, list(VISUAL_METADATA_KEYS))
    except Exception as e:
        logger.warning("Failed to explain visual embedding component %s: %s", component_idx, e)
        return {key: {"high_mode": None, "high_count": 0, "low_mode": None, "low_count": 0}
                 for key in VISUAL_METADATA_KEYS}
