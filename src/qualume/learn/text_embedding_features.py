"""Hook-text embedding features — the merge point between qualume's own
video-content features (learn/features.py's cuts_per_min, setting_change_count,
etc., video-only) and okn_mirror's caption-embedding approach
(okn_mirror/ml_engine_okn.py's emb_N + the interpretability fix built on top
of it).

WHY hook_text specifically (not full_transcript/caption): hook_text is the
ONE text field present on BOTH owned and external CreativeDNA
(models/contracts.py: CreativeDNA.hook_text, ExternalVideoAnnotation.hook_text)
— the only real text signal that works for video (TikTok/Reels) AND, once an
owned-content training pipeline exists, IG statics too (statics have no
shot_breakdown/full_transcript/duration at all; they DO have hook_text — the
headline/opening line of the creative). Building this on hook_text, not on
qualume's video-only engineered features, is what makes this module reusable
across workflow_types rather than TikTok-only.

Mirrors okn_mirror's _explain_embedding_component technique exactly (same
sentence-transformers model, same PCA-then-keyword-contrast interpretability
approach) rather than reinventing it — see that module for the original.

Fold-safety: fit_hook_text_embeddings is a pure fit-on-these-texts-only
function. Callers doing cross-validation MUST fit a fresh instance per
training fold and only ever .transform() held-out text with that fold's own
fitted PCA — never fit on the full dataset and then cross-validate, or the
embedding space itself leaks test-set information into every fold (textbook
data leakage). learn.prediction.cross_validate_predictor does this correctly;
see that function.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("qualume.learn.text_embedding_features")

_embedding_model = None
HAS_EMBEDDINGS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    pass

MIN_TEXTS_FOR_EMBEDDING = 10  # below this, PCA on embeddings is unreliable/overfits trivially
MAX_COMPONENTS = 8


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None and HAS_EMBEDDINGS:
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model


def fit_hook_text_embeddings(texts: list[str]) -> Optional[dict]:
    """Fit a PCA-reduction over the sentence embeddings of `texts` (hook_text
    values; empty string for missing — see module docstring on why an empty
    hook_text is itself a meaningful, embeddable signal, not a value to skip).

    Returns None if sentence-transformers is unavailable or there are too
    few texts to fit PCA meaningfully (same "degrade gracefully, don't
    fabricate a model" discipline as okn_mirror) — never raises.

    Returns {"pca": fitted PCA, "model_name": str, "n_components": int,
    "embeddings": raw sentence embeddings (for interpretability lookups),
    "texts": the original texts (for interpretability lookups)}."""
    if not HAS_EMBEDDINGS or len(texts) < MIN_TEXTS_FOR_EMBEDDING:
        return None
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        raw_embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    except Exception as e:
        logger.warning("Hook-text embedding failed: %s", e)
        return None

    from sklearn.decomposition import PCA

    n_components = min(MAX_COMPONENTS, max(2, len(texts) // 10), raw_embeddings.shape[1], len(texts) - 1)
    if n_components < 2:
        return None
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(raw_embeddings)
    return {
        "pca": pca, "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "n_components": n_components, "embeddings": raw_embeddings, "texts": list(texts),
    }


def transform_hook_text(fitted: dict, text: str) -> list[float]:
    """Project one (possibly new) hook_text through a FITTED embedding+PCA
    pipeline (from fit_hook_text_embeddings). Returns a plain list of floats,
    length fitted['n_components'] — ready to append to a feature vector."""
    model = _get_embedding_model()
    if model is None:
        raise RuntimeError("sentence-transformers unavailable, cannot transform new text")
    raw = model.encode([text or ""], show_progress_bar=False)
    reduced = fitted["pca"].transform(raw)
    return [float(v) for v in reduced[0]]


def explain_component(fitted: dict, component_idx: int, n_examples: int = 5, n_keywords: int = 5) -> dict:
    """Same technique as okn_mirror.ml_engine_okn.MLEngine._explain_embedding_component:
    project the fitting texts onto one PCA axis, contrast the high/low
    extremes via TF-IDF keywords + raw examples, so a human can read what
    that axis represents instead of an opaque emb_N number.

    Returns {"high_keywords", "low_keywords", "high_examples", "low_examples"}
    — never raises, degrades to empty lists on any internal failure."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        component_vector = fitted["pca"].components_[component_idx]
        projections = fitted["embeddings"] @ component_vector
        texts = fitted["texts"]

        order = np.argsort(projections)
        n = min(n_examples, len(order) // 2) if len(order) >= 2 else 0
        if n == 0:
            return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}

        high_idx = order[-n:][::-1]
        low_idx = order[:n]
        high_texts = [texts[i] for i in high_idx]
        low_texts = [texts[i] for i in low_idx]

        def _top_keywords(group: list[str]) -> list[str]:
            non_empty = [t for t in group if len(t.strip()) > 2]
            if len(non_empty) < 2:
                return []
            try:
                tfidf = TfidfVectorizer(max_features=n_keywords, stop_words="english",
                                         token_pattern=r"(?u)\b[#\w][\w]{2,}\b")
                tfidf.fit(non_empty)
                return list(tfidf.get_feature_names_out())
            except Exception:
                return []

        return {
            "high_keywords": _top_keywords(high_texts),
            "low_keywords": _top_keywords(low_texts),
            "high_examples": [str(t)[:80] for t in high_texts[:3]],
            "low_examples": [str(t)[:80] for t in low_texts[:3]],
        }
    except Exception as e:
        logger.warning("Failed to explain hook-text embedding component %s: %s", component_idx, e)
        return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}
