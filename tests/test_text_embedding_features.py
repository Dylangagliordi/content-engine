"""Tests for hook-text embedding features — written to verify the real
fit/transform/explain contract this module exposes, same real-behavior
discipline as test_okn_mirror_engines.py (these are near-identical
mechanics, same sentence-transformers dependency, same graceful-degradation
contract) applied to the hook_text-specific, cross-workflow-type module.
"""
import pytest

from qualume.learn.text_embedding_features import (
    HAS_EMBEDDINGS,
    MIN_TEXTS_FOR_EMBEDDING,
    explain_component,
    fit_hook_text_embeddings,
    transform_hook_text,
)

requires_embeddings = pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers unavailable")


def test_fit_returns_none_below_min_texts_for_embedding():
    texts = ["a real hook text"] * (MIN_TEXTS_FOR_EMBEDDING - 1)
    assert fit_hook_text_embeddings(texts) is None


@requires_embeddings
def test_fit_returns_a_real_fitted_pca_above_min_texts():
    texts = [f"a distinct real hook text number {i} about things" for i in range(30)]
    fitted = fit_hook_text_embeddings(texts)
    assert fitted is not None
    assert fitted["n_components"] >= 2
    assert fitted["embeddings"].shape[0] == 30
    assert len(fitted["texts"]) == 30


@requires_embeddings
def test_transform_hook_text_returns_fixed_length_vector():
    texts = [f"hook text example {i} with real words" for i in range(30)]
    fitted = fit_hook_text_embeddings(texts)
    vec = transform_hook_text(fitted, "a brand new hook text never seen in fitting")
    assert len(vec) == fitted["n_components"]
    assert all(isinstance(v, float) for v in vec)


@requires_embeddings
def test_transform_handles_empty_string_hook_text():
    """A real validation_error row (or a static with no hook written yet)
    can have hook_text=None/''. Embedding an empty string must not crash —
    empty-hook is itself a meaningful signal (see learn.features's parallel
    discipline for missing shot_breakdown/full_transcript), not an error."""
    texts = [f"hook {i}" for i in range(15)]
    fitted = fit_hook_text_embeddings(texts)
    vec = transform_hook_text(fitted, "")
    assert len(vec) == fitted["n_components"]


@requires_embeddings
def test_explain_component_contrasts_real_high_and_low_text():
    """Known injected signal: half the texts are clearly about cooking,
    half about fitness — PCA's dominant component should separate them,
    and the keyword contrast should reflect that real difference (not
    assert exact words, since PCA component sign/order isn't guaranteed —
    just that BOTH sides produce non-empty, DIFFERENT keyword sets)."""
    cooking = [f"amazing pasta recipe tips cooking kitchen food number {i}" for i in range(15)]
    fitness = [f"gym workout muscle training exercise reps number {i}" for i in range(15)]
    fitted = fit_hook_text_embeddings(cooking + fitness)
    result = explain_component(fitted, 0)
    assert "high_keywords" in result
    assert "low_keywords" in result
    assert isinstance(result["high_examples"], list)
    # the two sides should not be identical (real separation happened)
    assert set(result["high_keywords"]) != set(result["low_keywords"])


def test_explain_component_degrades_gracefully_on_bad_input():
    """Never raises — same honest-gap discipline as okn_mirror's identical
    method. A malformed/empty fitted dict must degrade to empty lists."""
    result = explain_component({"pca": None, "embeddings": None, "texts": []}, 0)
    assert result == {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}
