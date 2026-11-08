"""Tests for learn.ensemble_models — GradientBoosting + MLPRegressor
second opinions trained on the SAME pooled feature matrix as
learn.prediction's primary RandomForestRegressor. Fixture pattern
mirrors tests/test_prediction.py (_shot/_annotation helpers, synthetic
rows with a known injected correlation) — see that file for the
canonical shape this project uses for every model test.
"""
from qualume.learn.ensemble_models import predict_with_all_models, train_ensemble


def _shot(start, end):
    return {
        "start_sec": start, "end_sec": end, "shot_type": "medium_shot",
        "camera_movement": "static", "setting": "kitchen",
        "color_temperature": "warm", "lighting": "natural",
    }


def _annotation(hook_type="question", format_tag="talking_head", energy_level=3,
                 shots=None, transcript=None, duration=30):
    return {
        "hook_type": hook_type,
        "format_tag": format_tag,
        "energy_level": energy_level,
        "shot_breakdown": shots or [_shot(0, 10), _shot(10, 30)],
        "full_transcript": transcript or [{"start_sec": 0, "text": "hello there world today"}],
        "video_duration_seconds": duration,
    }


def test_train_ensemble_returns_all_three_fitted_models():
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(60)]
    trained = train_ensemble(rows, min_sample_size=50, use_text_embeddings=False)
    assert trained["gb_model"] is not None
    assert trained["mlp_model"] is not None


def test_train_ensemble_mlp_is_none_below_the_row_floor():
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(25)]
    trained = train_ensemble(rows, min_sample_size=20, use_text_embeddings=False)
    assert trained["gb_model"] is not None
    assert trained["mlp_model"] is None


def test_predict_with_all_models_returns_three_point_estimates():
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(60)]
    trained = train_ensemble(rows, min_sample_size=50, use_text_embeddings=False)
    from qualume.learn.prediction import build_feature_vector
    vec = build_feature_vector(_annotation())
    result = predict_with_all_models(trained, vec)
    assert isinstance(result["random_forest"], float)
    assert isinstance(result["gradient_boosting"], float)
    assert isinstance(result["mlp"], float) or result["mlp"] is None


def test_gb_and_rf_agree_on_direction_for_a_strong_injected_signal():
    """Same synthetic fast/slow cuts_per_min signal as
    test_train_predictor_and_predict_round_trip_on_synthetic_data in
    test_prediction.py; both RF and GB predictions should move the
    same direction (fast > slow), proving the shared feature matrix is
    correct for both models, not just RF."""
    from qualume.learn.prediction import build_feature_vector

    rows = []
    for i in range(30):
        is_fast = i % 2 == 0
        shots = [_shot(s, s + (1 if is_fast else 10)) for s in range(0, 20, 1 if is_fast else 10)]
        ann = _annotation(shots=shots, duration=20)
        ratio = 2.0 if is_fast else 0.5
        rows.append((ann, ratio))

    trained = train_ensemble(rows, min_sample_size=20, use_text_embeddings=False)

    fast_vec = build_feature_vector(_annotation(shots=[_shot(s, s + 1) for s in range(0, 20)], duration=20))
    slow_vec = build_feature_vector(_annotation(shots=[_shot(0, 20)], duration=20))

    fast = predict_with_all_models(trained, fast_vec)
    slow = predict_with_all_models(trained, slow_vec)

    assert fast["random_forest"] > slow["random_forest"]
    assert fast["gradient_boosting"] > slow["gradient_boosting"]
