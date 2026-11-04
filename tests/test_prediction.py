"""Tests for the prediction-model scaffold — written FIRST per this
project's TDD discipline.

Per plans/learn-engine-idea-sourcing-architecture.md: random forest/ML
prediction models were explicitly ruled out for INSIGHT GENERATION
(wrong tool — no clean interpretable effect-size-with-p-value), but
explicitly flagged as the right tool for a DIFFERENT, later job:
"predicting a new draft's likely performance before publish". This
module is that job, not insight mining — it may reuse the same
engineered/categorical features, but the OUTPUT is a performance
prediction, not a statistical claim with a p-value.

Not meaningful with today's real 58 videos (tiny n for ML, single
niche) — built and tested as a correctly-shaped SCAFFOLD now, same
"capture the shape before the volume" discipline as replication.py and
validation.py, so training against real volume later is a data problem,
not a missing-pipeline problem.
"""
from qualume.learn.prediction import build_feature_vector, train_predictor, predict_performance
from qualume.learn.text_embedding_features import HAS_EMBEDDINGS
import pytest

requires_embeddings = pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers unavailable")


def _shot(start, end):
    return {
        "start_sec": start, "end_sec": end, "shot_type": "medium_shot",
        "camera_movement": "static", "setting": "kitchen",
        "setting_category": "home_indoor", "color_temperature": "warm",
        "dominant_colors": ["white"], "lighting": "bright_natural",
        "subject_action": "doing something",
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


def test_build_feature_vector_combines_engineered_and_categorical_features():
    vec = build_feature_vector(_annotation())
    # engineered (continuous) features present
    assert "cuts_per_min" in vec
    assert "avg_shot_length_sec" in vec
    # categorical features present, one-hot-style keys
    assert vec["hook_type"] == "question"
    assert vec["format_tag"] == "talking_head"
    assert vec["energy_level"] == 3


def test_build_feature_vector_handles_missing_annotation_fields_gracefully():
    """A real validation_error row has typed fields NULL - must not crash
    building a feature vector, just produce Nones for what's missing."""
    partial = {
        "hook_type": None, "format_tag": None, "energy_level": None,
        "shot_breakdown": [], "full_transcript": [], "video_duration_seconds": 30,
    }
    vec = build_feature_vector(partial)
    assert vec["hook_type"] is None
    assert vec["cuts_per_min"] is None


def test_train_predictor_requires_a_minimum_real_sample_size():
    """Training on too few real rows is actively misleading (overfits
    completely) - fail loud rather than silently returning a useless
    model that looks real."""
    import pytest
    rows = [(_annotation(), 1.0)] * 3  # way below any sane minimum
    with pytest.raises(ValueError):
        train_predictor(rows, min_sample_size=20)


def test_train_predictor_and_predict_round_trip_on_synthetic_data():
    """Not a claim this is a GOOD model at this n - just that the
    train -> predict plumbing is correct: given a clean synthetic
    correlation (more cuts/min -> higher ratio), the model's predictions
    should at least move in the right direction between a fast-paced and
    a slow-paced vector, proving the pipeline wires features to outcome
    correctly rather than just returning noise or a constant."""
    rows = []
    for i in range(30):
        is_fast = i % 2 == 0
        shots = [_shot(s, s + (1 if is_fast else 10)) for s in range(0, 20, 1 if is_fast else 10)]
        ann = _annotation(shots=shots, duration=20)
        ratio = 2.0 if is_fast else 0.5
        rows.append((ann, ratio))

    model = train_predictor(rows, min_sample_size=20)

    fast_vec = build_feature_vector(_annotation(shots=[_shot(s, s + 1) for s in range(0, 20)], duration=20))
    slow_vec = build_feature_vector(_annotation(shots=[_shot(0, 20)], duration=20))

    fast_pred = predict_performance(model, fast_vec)
    slow_pred = predict_performance(model, slow_vec)
    assert fast_pred > slow_pred


def test_predict_performance_returns_a_plain_float():
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(25)]
    model = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    vec = build_feature_vector(_annotation())
    result = predict_performance(model, vec)
    assert isinstance(result, float)


# ──────────────────────────────────────────────
# Merged model: hook_text embedding features (okn_mirror-style), the real
# merge requested after the okn-vs-qualume comparison — hook_text is the
# ONE text field present on BOTH owned (static) and external (video)
# CreativeDNA, so this is the cross-workflow-type merge point, not a
# TikTok-only addition. See learn/text_embedding_features.py's module
# docstring for the full rationale.
# ──────────────────────────────────────────────

def test_build_feature_vector_without_text_embedder_is_unchanged():
    """Backward compatibility: no text_embedder arg -> no emb_hook_* keys
    at all, identical shape to before this feature existed."""
    vec = build_feature_vector(_annotation())
    assert not any(k.startswith("emb_hook_") for k in vec)


@requires_embeddings
def test_build_feature_vector_with_text_embedder_adds_emb_hook_features():
    from qualume.learn.text_embedding_features import fit_hook_text_embeddings
    texts = [f"hook text example {i} about real things" for i in range(15)]
    fitted = fit_hook_text_embeddings(texts)
    ann = _annotation()
    ann["hook_text"] = "a brand new hook"
    vec = build_feature_vector(ann, text_embedder=fitted)
    emb_keys = [k for k in vec if k.startswith("emb_hook_")]
    assert len(emb_keys) == fitted["n_components"]


@requires_embeddings
def test_train_predictor_fits_and_stores_a_text_embedder_by_default():
    rows = []
    for i in range(30):
        ann = _annotation(hook_type="a" if i % 2 == 0 else "b", energy_level=i % 5 + 1)
        ann["hook_text"] = f"real hook number {i}"
        rows.append((ann, float(i % 5 + 1)))
    trained = train_predictor(rows, min_sample_size=20)
    assert trained["text_embedder"] is not None
    assert trained["use_text_embeddings"] is True


def test_train_predictor_use_text_embeddings_false_skips_embedder_entirely():
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(25)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    assert trained["text_embedder"] is None


@requires_embeddings
def test_predict_performance_for_annotation_handles_the_embedder_automatically():
    """The real ergonomics fix: a caller predicting a NEW item shouldn't
    have to manually thread the trained model's fitted embedder through
    build_feature_vector themselves — this convenience wrapper does it."""
    from qualume.learn.prediction import predict_performance_for_annotation
    rows = []
    for i in range(30):
        is_fast = i % 2 == 0
        shots = [_shot(s, s + (1 if is_fast else 10)) for s in range(0, 20, 1 if is_fast else 10)]
        ann = _annotation(shots=shots, duration=20)
        ann["hook_text"] = "fast paced hook" if is_fast else "slow paced hook"
        ratio = 2.0 if is_fast else 0.5
        rows.append((ann, ratio))
    trained = train_predictor(rows, min_sample_size=20)

    fast_ann = _annotation(shots=[_shot(s, s + 1) for s in range(0, 20)], duration=20)
    fast_ann["hook_text"] = "fast paced hook"
    result = predict_performance_for_annotation(trained, fast_ann)
    assert isinstance(result, float)


def test_feature_importance_attaches_interpretation_to_emb_hook_features():
    """Real gap this closes (user: 'whats the point of [embeddings] if we
    cant see them', applied here to the merged prediction model): any
    emb_hook_* feature surfaced by feature_importance must carry a human-
    readable interpretation, same technique as okn_mirror's fix, not an
    opaque label."""
    from qualume.learn.prediction import feature_importance
    rows = []
    for i in range(30):
        topic = "cooking" if i % 2 == 0 else "fitness"
        ann = _annotation(energy_level=i % 5 + 1)
        ann["hook_text"] = f"{topic} related hook text number {i}"
        ratio = 2.0 if topic == "cooking" else 0.8
        rows.append((ann, ratio))
    trained = train_predictor(rows, min_sample_size=20)
    if trained["text_embedder"] is None:
        pytest.skip("sentence-transformers unavailable in this environment")
    importances = feature_importance(trained)
    emb_features = [f for f in importances if f["raw_feature"].startswith("emb_hook_")]
    for f in emb_features:
        assert "interpretation" in f
