"""Tests for stored (precomputed) embedding features — the transcript_
embeddings/visual_embeddings merge, written FIRST per this project's TDD
discipline. These fit/transform pure functions never touch the DB
themselves (DB fetching is training_data.py's job) — tests pass synthetic
vectors + aux directly, same as text_embedding_features's own tests.
"""
import numpy as np
import pytest

from qualume.learn.stored_embedding_features import (
    MIN_VECTORS_FOR_PCA,
    dominant_shot_metadata,
    explain_transcript_component,
    explain_visual_component,
    fit_stored_embeddings,
    transform_stored_embedding,
)


def _random_vector(seed: int, dim: int = 1024) -> list[float]:
    rng = np.random.RandomState(seed)
    return list(rng.rand(dim))


def test_fit_returns_none_below_min_vectors():
    vectors = [_random_vector(i) for i in range(MIN_VECTORS_FOR_PCA - 1)]
    assert fit_stored_embeddings(vectors, aux=["text"] * len(vectors)) is None


def test_fit_returns_a_real_fitted_pca_above_min_vectors():
    vectors = [_random_vector(i) for i in range(30)]
    aux = [f"text {i}" for i in range(30)]
    fitted = fit_stored_embeddings(vectors, aux)
    assert fitted is not None
    assert fitted["n_components"] >= 2
    assert fitted["embeddings"].shape[0] == 30
    assert fitted["aux"] == aux


def test_transform_stored_embedding_returns_fixed_length_vector():
    vectors = [_random_vector(i) for i in range(30)]
    fitted = fit_stored_embeddings(vectors, aux=list(range(30)))
    new_vector = _random_vector(999)
    result = transform_stored_embedding(fitted, new_vector)
    assert len(result) == fitted["n_components"]
    assert all(isinstance(v, float) for v in result)


def test_explain_transcript_component_contrasts_real_text():
    """Known injected signal: construct vectors with a REAL separating
    direction (not random) so PCA's dominant component actually captures
    it, same discipline as text_embedding_features's own cooking/fitness
    test — proves the keyword contrast reflects a genuine axis, not noise."""
    rng = np.random.RandomState(0)
    base = rng.rand(1024)
    direction = rng.rand(1024)
    cooking_vectors = [list(base + direction * (1 + 0.1 * i)) for i in range(15)]
    fitness_vectors = [list(base - direction * (1 + 0.1 * i)) for i in range(15)]
    vectors = cooking_vectors + fitness_vectors
    cooking_text = [f"amazing pasta recipe cooking kitchen food number {i}" for i in range(15)]
    fitness_text = [f"gym workout muscle training exercise reps number {i}" for i in range(15)]
    aux = cooking_text + fitness_text

    fitted = fit_stored_embeddings(vectors, aux)
    result = explain_transcript_component(fitted, 0)
    assert "high_keywords" in result
    assert "low_keywords" in result
    assert set(result["high_keywords"]) != set(result["low_keywords"])


def test_explain_transcript_component_degrades_gracefully_on_bad_input():
    result = explain_transcript_component({"pca": None, "embeddings": None, "aux": []}, 0)
    assert result == {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}


def test_dominant_shot_metadata_returns_the_mode_per_key():
    shots = [
        {"setting": "kitchen", "shot_type": "medium_shot", "color_temperature": "warm"},
        {"setting": "kitchen", "shot_type": "close_up", "color_temperature": "warm"},
        {"setting": "gym", "shot_type": "medium_shot", "color_temperature": "cool"},
    ]
    result = dominant_shot_metadata(shots)
    assert result["setting"] == "kitchen"
    assert result["color_temperature"] == "warm"


def test_dominant_shot_metadata_handles_empty_shot_breakdown():
    assert dominant_shot_metadata(None) == {}
    assert dominant_shot_metadata([]) == {}


def test_explain_visual_component_contrasts_real_metadata():
    """Known injected signal: a real separating direction in the vectors,
    correlated with kitchen-vs-gym metadata — proves mode_contrast reflects
    a genuine axis, not just the first two rows by coincidence."""
    rng = np.random.RandomState(1)
    base = rng.rand(1024)
    direction = rng.rand(1024)
    kitchen_vectors = [list(base + direction * (1 + 0.1 * i)) for i in range(10)]
    gym_vectors = [list(base - direction * (1 + 0.1 * i)) for i in range(10)]
    vectors = kitchen_vectors + gym_vectors
    kitchen_meta = [{"setting": "kitchen", "shot_type": "medium_shot", "color_temperature": "warm"}] * 10
    gym_meta = [{"setting": "gym", "shot_type": "wide_shot", "color_temperature": "cool"}] * 10
    aux = kitchen_meta + gym_meta

    fitted = fit_stored_embeddings(vectors, aux)
    result = explain_visual_component(fitted, 0)
    assert "setting" in result
    assert result["setting"]["high_mode"] != result["setting"]["low_mode"]


def test_explain_visual_component_degrades_gracefully_on_bad_input():
    result = explain_visual_component({"pca": None, "embeddings": None, "aux": []}, 0)
    for key in ("setting", "shot_type", "color_temperature"):
        assert result[key]["high_mode"] is None
