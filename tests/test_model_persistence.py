"""Tests for model persistence — real save/load round-trip, using a real
temp directory (not mocked) so this exercises real joblib serialization
of a real trained RandomForestRegressor, not just a dict of plain values.
"""
import shutil

import pytest

from qualume.learn.prediction import build_feature_vector, predict_performance, train_predictor


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Redirect MODELS_DIR to a real temp directory for every test in
    this file — never touches the real data/models/ directory, and
    cleans up automatically via pytest's tmp_path."""
    import qualume.learn.model_persistence as mp
    monkeypatch.setattr(mp, "MODELS_DIR", tmp_path / "models")
    yield
    if (tmp_path / "models").exists():
        shutil.rmtree(tmp_path / "models", ignore_errors=True)


def _shot(start, end):
    return {"start_sec": start, "end_sec": end, "shot_type": "medium_shot",
            "setting": "kitchen", "color_temperature": "warm"}


def _annotation(energy_level=3, shots=None, duration=20):
    return {
        "hook_type": "question", "format_tag": "talking_head", "energy_level": energy_level,
        "shot_breakdown": shots or [_shot(0, 10), _shot(10, 20)],
        "full_transcript": [{"start_sec": 0, "text": "hello world"}],
        "video_duration_seconds": duration,
    }


def _real_rows(n=25):
    return [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(n)]


def test_save_predictor_writes_a_real_model_file_and_metadata_file():
    from qualume.learn.model_persistence import load_metadata, model_path, save_predictor

    trained = train_predictor(_real_rows(), min_sample_size=20, use_text_embeddings=False)
    path = save_predictor(trained, niche="zz_test_niche", n_training_rows=25)

    assert path.exists()
    assert path == model_path("zz_test_niche")
    meta = load_metadata("zz_test_niche")
    assert meta["niche"] == "zz_test_niche"
    assert meta["n_training_rows"] == 25
    assert "trained_at" in meta


def test_load_predictor_round_trips_to_a_real_working_model():
    """Real round-trip: save, then load in what's functionally a fresh
    context (new dict, no shared Python object identity), then predict
    with it — proving the ACTUAL RandomForestRegressor object survived
    joblib serialization correctly, not just that a file got written."""
    from qualume.learn.model_persistence import load_predictor, save_predictor

    rows = []
    for i in range(30):
        is_fast = i % 2 == 0
        shots = [_shot(s, s + (1 if is_fast else 10)) for s in range(0, 20, 1 if is_fast else 10)]
        ann = _annotation(shots=shots, duration=20)
        ratio = 2.0 if is_fast else 0.5
        rows.append((ann, ratio))
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    save_predictor(trained, niche="zz_test_roundtrip", n_training_rows=30)

    loaded = load_predictor("zz_test_roundtrip")
    assert loaded is not None
    assert loaded is not trained  # genuinely a different object, not just returning the same ref

    fast_vec = build_feature_vector(_annotation(shots=[_shot(s, s + 1) for s in range(0, 20)], duration=20))
    slow_vec = build_feature_vector(_annotation(shots=[_shot(0, 20)], duration=20))
    fast_pred = predict_performance(loaded, fast_vec)
    slow_pred = predict_performance(loaded, slow_vec)
    assert fast_pred > slow_pred  # the loaded model still learned the real relationship


def test_load_predictor_returns_none_for_a_niche_with_no_saved_model():
    """Real, expected state — a niche nobody has trained yet. None, not
    an exception, same refuse-rather-than-guess discipline as
    learn.baseline returning None for an account with no snapshots."""
    from qualume.learn.model_persistence import load_predictor

    assert load_predictor("zz_test_never_trained") is None


def test_load_metadata_returns_none_for_a_niche_with_no_saved_model():
    from qualume.learn.model_persistence import load_metadata

    assert load_metadata("zz_test_never_trained") is None


def test_has_saved_predictor_reflects_real_file_presence():
    from qualume.learn.model_persistence import has_saved_predictor, save_predictor

    assert has_saved_predictor("zz_test_presence") is False
    trained = train_predictor(_real_rows(), min_sample_size=20, use_text_embeddings=False)
    save_predictor(trained, niche="zz_test_presence", n_training_rows=25)
    assert has_saved_predictor("zz_test_presence") is True


def test_save_predictor_metadata_tracks_which_embedders_were_actually_fit():
    """Real honest-staleness-tracking requirement: a caller loading a
    saved model later must be able to tell WHICH embedders this specific
    saved model used, without loading the full model and poking at its
    internals."""
    from qualume.learn.model_persistence import load_metadata, save_predictor

    trained = train_predictor(_real_rows(), min_sample_size=20, use_text_embeddings=False,
                               use_transcript_embeddings=False, use_visual_embeddings=False)
    save_predictor(trained, niche="zz_test_embedders", n_training_rows=25)
    meta = load_metadata("zz_test_embedders")
    assert meta["embedders_used"] == {"text": False, "transcript": False, "visual": False}


def test_save_predictor_overwrites_a_previous_save_for_the_same_niche():
    """Same 'latest wins' discipline as everywhere else in this project —
    re-training and re-saving a niche must replace the old artifact, not
    accumulate versioned files nobody asked for."""
    from qualume.learn.model_persistence import load_metadata, save_predictor

    trained1 = train_predictor(_real_rows(20), min_sample_size=20, use_text_embeddings=False)
    save_predictor(trained1, niche="zz_test_overwrite", n_training_rows=20)
    first_meta = load_metadata("zz_test_overwrite")

    trained2 = train_predictor(_real_rows(40), min_sample_size=20, use_text_embeddings=False)
    save_predictor(trained2, niche="zz_test_overwrite", n_training_rows=40)
    second_meta = load_metadata("zz_test_overwrite")

    assert second_meta["n_training_rows"] == 40
    assert second_meta["n_training_rows"] != first_meta["n_training_rows"]


def test_niche_to_filename_handles_real_niche_strings_safely():
    """Real niche strings contain spaces, slashes, apostrophes — e.g.
    "women's health supplements", "food/cooking/recipe content" — must
    not crash or collide when turned into filesystem paths."""
    from qualume.learn.model_persistence import _niche_to_filename

    assert "/" not in _niche_to_filename("food/cooking/recipe content")
    assert "'" not in _niche_to_filename("women's health supplements")
    # two different real niches must not collapse to the same filename
    assert (_niche_to_filename("fitness/workout content")
            != _niche_to_filename("fitness workout content"))


def _pooled_rows(n=60):
    rows = []
    for i in range(n):
        niche = "fitness/workout content" if i % 2 == 0 else "food/cooking/recipe content"
        ann = _annotation(energy_level=i % 5 + 1)
        ann["niche"] = niche
        ratio = 2.0 if niche == "fitness/workout content" else 0.5
        rows.append((ann, ratio))
    return rows


def test_save_pooled_predictor_writes_a_real_model_file_and_metadata_file():
    from qualume.learn.model_persistence import load_pooled_metadata, pooled_model_path, save_pooled_predictor

    trained = train_predictor(_pooled_rows(), min_sample_size=50, use_text_embeddings=False)
    niches = ["fitness/workout content", "food/cooking/recipe content"]
    path = save_pooled_predictor(trained, n_training_rows=60, niches_represented=niches)

    assert path.exists()
    assert path == pooled_model_path()
    meta = load_pooled_metadata()
    assert meta["n_training_rows"] == 60
    assert meta["n_niches_represented"] == 2
    assert meta["niches_represented"] == sorted(niches)
    assert "trained_at" in meta


def test_load_pooled_predictor_round_trips_to_a_real_working_model():
    from qualume.learn.model_persistence import load_pooled_predictor, save_pooled_predictor

    trained = train_predictor(_pooled_rows(), min_sample_size=50, use_text_embeddings=False)
    save_pooled_predictor(trained, n_training_rows=60,
                           niches_represented=["fitness/workout content", "food/cooking/recipe content"])

    loaded = load_pooled_predictor()
    assert loaded is not None
    assert loaded is not trained

    fitness_vec = build_feature_vector(_annotation())
    fitness_vec["niche"] = "fitness/workout content"
    food_vec = build_feature_vector(_annotation())
    food_vec["niche"] = "food/cooking/recipe content"
    fitness_pred = predict_performance(loaded, fitness_vec)
    food_pred = predict_performance(loaded, food_vec)
    assert fitness_pred > food_pred  # the loaded pooled model still learned the real signal


def test_load_pooled_predictor_returns_none_when_nothing_saved_yet():
    from qualume.learn.model_persistence import load_pooled_predictor

    assert load_pooled_predictor() is None


def test_has_saved_pooled_predictor_reflects_real_file_presence():
    from qualume.learn.model_persistence import has_saved_pooled_predictor, save_pooled_predictor

    assert has_saved_pooled_predictor() is False
    trained = train_predictor(_pooled_rows(), min_sample_size=50, use_text_embeddings=False)
    save_pooled_predictor(trained, n_training_rows=60,
                           niches_represented=["fitness/workout content", "food/cooking/recipe content"])
    assert has_saved_pooled_predictor() is True


def test_save_pooled_predictor_overwrites_a_previous_save():
    from qualume.learn.model_persistence import load_pooled_metadata, save_pooled_predictor

    trained1 = train_predictor(_pooled_rows(60), min_sample_size=50, use_text_embeddings=False)
    save_pooled_predictor(trained1, n_training_rows=60,
                           niches_represented=["fitness/workout content", "food/cooking/recipe content"])
    first_meta = load_pooled_metadata()

    trained2 = train_predictor(_pooled_rows(80), min_sample_size=50, use_text_embeddings=False)
    save_pooled_predictor(trained2, n_training_rows=80,
                           niches_represented=["fitness/workout content", "food/cooking/recipe content"])
    second_meta = load_pooled_metadata()

    assert second_meta["n_training_rows"] == 80
    assert second_meta["n_training_rows"] != first_meta["n_training_rows"]


def test_pooled_and_niche_keyed_saves_do_not_collide():
    """The pooled path and the niche-keyed path must coexist on disk
    without clobbering each other - two genuinely separate artifacts."""
    from qualume.learn.model_persistence import (
        has_saved_pooled_predictor,
        has_saved_predictor,
        save_pooled_predictor,
        save_predictor,
    )

    niche_trained = train_predictor(_real_rows(), min_sample_size=20, use_text_embeddings=False)
    save_predictor(niche_trained, niche="zz_test_coexist", n_training_rows=25)

    pooled_trained = train_predictor(_pooled_rows(), min_sample_size=50, use_text_embeddings=False)
    save_pooled_predictor(pooled_trained, n_training_rows=60,
                           niches_represented=["fitness/workout content", "food/cooking/recipe content"])

    assert has_saved_predictor("zz_test_coexist") is True
    assert has_saved_pooled_predictor() is True
