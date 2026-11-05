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


# ──────────────────────────────────────────────
# Three-way embedding merge: hook_text (live) + transcript/visual
# (precomputed, learn.stored_embedding_features) — "how can we use ALL
# [text/transcript/visual embeddings] in the prediction model and have
# them all be interpretable".
# ──────────────────────────────────────────────

def _random_vector(seed: int, dim: int = 1024) -> list[float]:
    import numpy as np
    rng = np.random.RandomState(seed)
    return list(rng.rand(dim))


def _annotation_with_stored_embeddings(i: int, topic: str):
    ann = _annotation(energy_level=i % 5 + 1)
    ann["hook_text"] = f"{topic} hook text number {i}"
    ann["transcript_embedding_vector"] = _random_vector(i)
    ann["transcript_embeddable_text"] = f"{topic} full transcript spoken words number {i}"
    ann["visual_embedding_vector"] = _random_vector(1000 + i)
    ann["shot_breakdown"] = [{"start_sec": 0, "end_sec": 10, "setting": topic, "shot_type": "medium_shot"}]
    return ann


def test_build_feature_vector_with_all_three_embedders_adds_all_three_blocks():
    from qualume.learn.stored_embedding_features import fit_stored_embeddings
    rows = [(_annotation_with_stored_embeddings(i, "cooking" if i % 2 == 0 else "fitness"), 1.0)
            for i in range(20)]
    transcript_embedder = fit_stored_embeddings(
        [ann["transcript_embedding_vector"] for ann, _ in rows],
        [ann["transcript_embeddable_text"] for ann, _ in rows],
    )
    visual_embedder = fit_stored_embeddings(
        [ann["visual_embedding_vector"] for ann, _ in rows],
        [{"setting": "x"}] * 20,
    )
    ann = rows[0][0]
    vec = build_feature_vector(ann, transcript_embedder=transcript_embedder, visual_embedder=visual_embedder)
    assert any(k.startswith("emb_transcript_") for k in vec)
    assert any(k.startswith("emb_visual_") for k in vec)
    assert not any(k.startswith("emb_hook_") for k in vec)  # text_embedder not passed


def test_build_feature_vector_gives_none_for_missing_stored_vector():
    """An item with NO stored transcript/visual embedding yet (real,
    incomplete coverage) must get None for that block's features, not a
    fabricated zero vector."""
    from qualume.learn.stored_embedding_features import fit_stored_embeddings
    vectors = [_random_vector(i) for i in range(20)]
    embedder = fit_stored_embeddings(vectors, ["text"] * 20)
    ann = _annotation()
    ann["transcript_embedding_vector"] = None  # this item has no stored vector
    vec = build_feature_vector(ann, transcript_embedder=embedder)
    emb_keys = [k for k in vec if k.startswith("emb_transcript_")]
    assert len(emb_keys) == embedder["n_components"]
    assert all(vec[k] is None for k in emb_keys)


def test_train_predictor_fits_transcript_and_visual_embedders_from_rows():
    rows = [(_annotation_with_stored_embeddings(i, "cooking" if i % 2 == 0 else "fitness"), 1.0)
            for i in range(30)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    assert trained["transcript_embedder"] is not None
    assert trained["visual_embedder"] is not None
    assert any(k.startswith("emb_transcript_") for k in trained["emb_keys"])
    assert any(k.startswith("emb_visual_") for k in trained["emb_keys"])


def test_train_predictor_skips_transcript_visual_when_disabled():
    rows = [(_annotation_with_stored_embeddings(i, "x"), 1.0) for i in range(25)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False,
                               use_transcript_embeddings=False, use_visual_embeddings=False)
    assert trained["transcript_embedder"] is None
    assert trained["visual_embedder"] is None
    assert trained["emb_keys"] == []


def test_train_predictor_handles_partial_coverage_gracefully():
    """Real coverage shape: SOME items have a stored transcript/visual
    vector, some don't (those pipelines run independently). The embedder
    still fits on whichever rows DO have a vector, and the model trains
    on the full row set with None for the rest — must not crash."""
    rows = []
    for i in range(30):
        ann = _annotation_with_stored_embeddings(i, "cooking" if i % 2 == 0 else "fitness")
        if i % 3 == 0:
            ann["transcript_embedding_vector"] = None
            ann["visual_embedding_vector"] = None
        rows.append((ann, 1.0))
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    assert trained["transcript_embedder"] is not None  # enough real rows still had one


def test_feature_importance_uses_metadata_contrast_for_visual_not_tfidf():
    """Real gap this closes: visual embeddings have no associated text at
    all, so feature_importance must use mode_contrast (structured
    metadata), not an empty/broken TF-IDF attempt, for emb_visual_*."""
    from qualume.learn.prediction import feature_importance
    rows = [(_annotation_with_stored_embeddings(i, "cooking" if i % 2 == 0 else "fitness"),
              2.0 if i % 2 == 0 else 0.5) for i in range(30)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    importances = feature_importance(trained, top_n=30)
    visual_features = [f for f in importances if f["raw_feature"].startswith("emb_visual_")]
    for f in visual_features:
        assert "interpretation" in f
        assert "setting" in f["interpretation"]
        assert "high_mode" in f["interpretation"]["setting"]


# ──────────────────────────────────────────────
# Real correlation + full feature listing — user: "does it tell us how
# correlated? how can we have that... also do it for every feature,
# also list every feature". feature_importance() previously only gave
# sklearn's Gini/variance-reduction importance (how much the model
# RELIES on a feature), which is NOT a correlation and carries no sign.
# This adds a real Spearman correlation (direction-aware) for every
# continuous/ordinal/embedding feature, and a category-breakdown (the
# categorical equivalent of a correlation sign) for hook_type/format_tag,
# and changes the default to return EVERY feature, not just top_n.
# ──────────────────────────────────────────────

def test_feature_importance_lists_every_feature_by_default():
    """top_n=None (the new default) must return every feature in the
    model, not just the top 10 — real gap: 'also list every feature'."""
    from qualume.learn.prediction import feature_importance
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(25)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    total_features = (len(trained["categorical_keys"]) + len(trained["continuous_keys"])
                       + len(trained["ordinal_keys"]) + len(trained["emb_keys"]))
    result = feature_importance(trained)
    assert len(result) == total_features


def test_feature_importance_still_respects_explicit_top_n():
    from qualume.learn.prediction import feature_importance
    rows = [(_annotation(energy_level=i % 5 + 1), float(i % 5 + 1)) for i in range(25)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained, top_n=3)
    assert len(result) == 3


def test_feature_importance_includes_real_spearman_correlation_for_continuous_features():
    """Known injected signal: cuts_per_min strongly correlated with the
    target (same relationship test_train_predictor_and_predict_round_trip_on_synthetic_data
    already proves the MODEL learns) — feature_importance must report
    the actual correlation coefficient and a direction string, not just
    an importance weight with no sign."""
    from qualume.learn.prediction import feature_importance
    rows = []
    for i in range(30):
        is_fast = i % 2 == 0
        shots = [_shot(s, s + (1 if is_fast else 10)) for s in range(0, 20, 1 if is_fast else 10)]
        ann = _annotation(shots=shots, duration=20)
        ratio = 2.0 if is_fast else 0.5
        rows.append((ann, ratio))
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained)
    cuts = next(f for f in result if f["raw_feature"] == "cuts_per_min")
    assert cuts["correlation"] is not None
    assert cuts["correlation"] > 0  # more cuts/min -> higher ratio, by construction
    assert cuts["direction"] == "higher is better"


def test_feature_importance_direction_says_lower_is_better_for_negative_correlation():
    from qualume.learn.prediction import feature_importance
    rows = []
    for i in range(30):
        is_slow = i % 2 == 0
        shots = [_shot(s, s + (10 if is_slow else 1)) for s in range(0, 20, 10 if is_slow else 1)]
        ann = _annotation(shots=shots, duration=20)
        ratio = 2.0 if is_slow else 0.5  # slower (fewer cuts) -> higher ratio here
        rows.append((ann, ratio))
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained)
    cuts = next(f for f in result if f["raw_feature"] == "cuts_per_min")
    assert cuts["correlation"] < 0
    assert cuts["direction"] == "lower is better"


def test_feature_importance_includes_correlation_for_embedding_features():
    """Embedding axes also get a real correlation number (on top of the
    existing keyword/metadata interpretation) — same "how correlated"
    answer applies to emb_* features, not just engineered ones."""
    from qualume.learn.prediction import feature_importance
    rows = [(_annotation_with_stored_embeddings(i, "cooking" if i % 2 == 0 else "fitness"),
              2.0 if i % 2 == 0 else 0.5) for i in range(30)]
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained)
    emb_features = [f for f in result if f["raw_feature"].startswith("emb_")]
    assert len(emb_features) > 0
    for f in emb_features:
        assert "correlation" in f
        assert "direction" in f


def test_feature_importance_includes_category_breakdown_for_categorical_features():
    """Categorical features (hook_type, format_tag) have no single
    correlation number — the real equivalent is a per-category
    breakdown of the actual target values, sorted so the best/worst
    category is obvious, same directionality info a correlation sign
    gives for continuous features."""
    from qualume.learn.prediction import feature_importance
    rows = []
    for i in range(30):
        hook = "question" if i % 2 == 0 else "statement"
        ratio = 3.0 if hook == "question" else 0.5  # real injected signal
        rows.append((_annotation(hook_type=hook, energy_level=i % 5 + 1), ratio))
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained)
    hook_feature = next(f for f in result if f["raw_feature"] == "hook_type")
    assert "category_breakdown" in hook_feature
    breakdown = hook_feature["category_breakdown"]
    values_by_category = {b["value"]: b["median_target"] for b in breakdown}
    assert values_by_category["question"] > values_by_category["statement"]


def test_feature_importance_correlation_is_none_when_insufficient_real_data():
    """A feature with no real variance (e.g. every row has the same
    energy_level, or an embedding block that never got fit) can't
    produce a meaningful correlation — must report None, not a
    fabricated/NaN-masquerading-as-zero number."""
    from qualume.learn.prediction import feature_importance
    rows = [(_annotation(energy_level=3), float(i)) for i in range(25)]  # energy_level constant
    trained = train_predictor(rows, min_sample_size=20, use_text_embeddings=False)
    result = feature_importance(trained)
    energy = next(f for f in result if f["raw_feature"] == "energy_level")
    assert energy["correlation"] is None
    assert energy["direction"] == "no variance / insufficient data"
