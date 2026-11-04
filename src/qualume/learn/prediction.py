"""Prediction-model scaffold — DIFFERENT job from learn.mining/
learn.replication, per plans/learn-engine-idea-sourcing-architecture.md:
"random forest ... may show up later for a DIFFERENT job — predicting a
new draft's likely performance before publish ... that's prediction, not
insight-mining". learn.mining gives interpretable claims with a p-value
(the right tool for "is X associated with Y"); this module gives a raw
performance prediction for one specific draft (the right tool for "how
well will THIS do"), explicitly NOT meant to produce a statistical claim.

Random forest regression chosen per the plan's own reasoning: handles a
mix of continuous (learn.features) and categorical fields without manual
interaction terms, robust to the inevitable outliers in view-count data,
and degrades gracefully (rather than catastrophically) with a small
real n — appropriate for a model that starts genuinely small and grows
as more accounts/videos are ingested.

MERGED MODEL (added after the okn-vs-qualume comparison, per explicit
user request): hook_text embedding features (learn/text_embedding_features.py,
same technique as okn_mirror) are now OPTIONAL additional features
alongside the video-content engineered features (learn/features.py).
hook_text is the one field present on BOTH owned (static) and external
(video) CreativeDNA — see text_embedding_features.py's module docstring
for why this, not caption/full_transcript, is the real cross-workflow-type
merge point. Enabled by default (use_text_embeddings=True) when
sentence-transformers is available; set False to train on video-content
features alone (e.g. for a fast sanity check, or when sentence-transformers
isn't installed in that environment — degrades gracefully, never crashes).

Every emb_hook_N feature surfaced by feature_importance() carries a real
human-readable interpretation (high/low keyword contrast from the actual
training hook_text values), not an opaque PCA label — the same
interpretability fix built for okn_mirror, applied here (see
text_embedding_features.explain_component).
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from qualume.learn.features import engineered_features
from qualume.learn.text_embedding_features import (
    HAS_EMBEDDINGS,
    explain_component,
    fit_hook_text_embeddings,
    transform_hook_text,
)

CATEGORICAL_FIELDS = ("hook_type", "format_tag")
ORDINAL_FIELDS = ("energy_level",)

READABLE_FEATURE_NAMES = {
    "hook_type": "Hook Type", "format_tag": "Format", "energy_level": "Energy Level",
    "cuts_per_min": "Cuts Per Minute", "avg_shot_length_sec": "Avg Shot Length",
    "shot_length_stddev": "Shot Length Variability", "color_temperature_consistency": "Color Temp Consistency",
    "setting_change_count": "Setting Changes", "shot_type_variety": "Shot Type Variety",
    "speech_coverage_ratio": "Speech Coverage", "words_per_second": "Speech Speed",
}


def build_feature_vector(annotation: dict, text_embedder: dict | None = None) -> dict:
    """One video's (or, once an owned-content training pipeline exists, one
    static's) full feature vector: engineered continuous features
    (learn.features, derived from shot_breakdown/full_transcript — VIDEO
    ONLY, absent/None for a static) plus the categorical/ordinal fields a
    model can also use directly, plus OPTIONAL hook-text embedding features
    (emb_hook_0..N) when `text_embedder` is given (a dict from
    learn.text_embedding_features.fit_hook_text_embeddings) — these work for
    BOTH video and static annotations, since hook_text exists on both.
    `annotation` is a plain dict with keys matching the relevant
    external_creative_dna/creative_dna columns plus video_duration_seconds —
    callers assemble this from a DB row or (in tests) a fixture; this
    function has no DB dependency itself.

    Missing/None fields produce None in the vector, never a guessed
    default — a real validation_error row's NULL fields must stay NULL
    through to the model, not silently become a fake 0 or 'unknown'
    category that looks like real data.

    text_embedder=None (the default when called directly, NOT what
    train_predictor passes — see that function) means NO emb_hook_* keys
    at all, identical shape to before this feature existed — strict
    backward compatibility for any existing caller."""
    duration = annotation.get("video_duration_seconds")
    shots = annotation.get("shot_breakdown") or []
    transcript = annotation.get("full_transcript") or []

    vec = dict(engineered_features(shots, transcript, duration)) if duration else {
        "cuts_per_min": None, "avg_shot_length_sec": None, "shot_length_stddev": None,
        "color_temperature_consistency": None, "setting_change_count": None,
        "shot_type_variety": None, "speech_coverage_ratio": None, "words_per_second": None,
    }
    for field in CATEGORICAL_FIELDS + ORDINAL_FIELDS:
        vec[field] = annotation.get(field)

    if text_embedder is not None:
        hook_text = annotation.get("hook_text") or ""
        embedded = transform_hook_text(text_embedder, hook_text)
        for i, value in enumerate(embedded):
            vec[f"emb_hook_{i}"] = value

    return vec


def train_predictor(rows: list[tuple[dict, float]], min_sample_size: int = 50,
                     use_text_embeddings: bool = True):
    """Train a RandomForestRegressor on (annotation_dict, performance_ratio)
    pairs. Raises ValueError below min_sample_size — training on too few
    real rows produces a model that overfits completely and looks real
    while being useless; failing loud here is the honest behavior, not
    silently returning a model nobody should trust.

    use_text_embeddings (default True): fits a hook_text embedder
    (learn.text_embedding_features.fit_hook_text_embeddings) on THIS
    training set's own hook_text values and folds emb_hook_* features into
    every row. Fit happens ONCE here, on the full set passed in — callers
    doing cross-validation must fit a fresh embedder per fold (see
    text_embedding_features's fold-safety note) rather than reusing this
    function's single fit across folds. Set False to train on video-content
    features alone; also degrades to False automatically (text_embedder
    stays None) if sentence-transformers isn't installed or there aren't
    enough real hook_text values to fit PCA meaningfully — same honest
    "can't build this, don't fake it" discipline as fit_hook_text_embeddings
    itself.

    Returns an opaque model object (an encoder + the fitted regressor + the
    fitted text embedder, if any); pass it straight to predict_performance
    or predict_performance_for_annotation, don't inspect its internals."""
    if len(rows) < min_sample_size:
        raise ValueError(
            f"train_predictor needs at least {min_sample_size} real rows, got {len(rows)} — "
            "training on fewer would overfit completely and produce a model that "
            "looks real but isn't trustworthy"
        )

    text_embedder = None
    if use_text_embeddings and HAS_EMBEDDINGS:
        hook_texts = [ann.get("hook_text") or "" for ann, _ in rows]
        text_embedder = fit_hook_text_embeddings(hook_texts)

    vectors = [build_feature_vector(ann, text_embedder=text_embedder) for ann, _ in rows]
    targets = [ratio for _, ratio in rows]

    categorical_keys = list(CATEGORICAL_FIELDS)
    emb_keys = [k for k in vectors[0].keys() if k.startswith("emb_hook_")]
    continuous_keys = [k for k in vectors[0].keys()
                        if k not in categorical_keys and k not in ORDINAL_FIELDS and k not in emb_keys]
    ordinal_keys = list(ORDINAL_FIELDS)

    raw_categorical = [[v.get(k) if v.get(k) is not None else "__missing__" for k in categorical_keys]
                        for v in vectors]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded_categorical = encoder.fit_transform(raw_categorical)

    X = []
    for i, v in enumerate(vectors):
        row = list(encoded_categorical[i])
        row += [v.get(k) if v.get(k) is not None else -1.0 for k in continuous_keys]
        row += [v.get(k) if v.get(k) is not None else -1.0 for k in ordinal_keys]
        row += [v.get(k) if v.get(k) is not None else 0.0 for k in emb_keys]
        X.append(row)

    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X, targets)

    return {
        "model": model,
        "encoder": encoder,
        "categorical_keys": categorical_keys,
        "continuous_keys": continuous_keys,
        "ordinal_keys": ordinal_keys,
        "emb_keys": emb_keys,
        "text_embedder": text_embedder,
        "use_text_embeddings": text_embedder is not None,
    }


def _vectorize_for_predict(trained: dict, feature_vector: dict) -> list:
    categorical_keys = trained["categorical_keys"]
    continuous_keys = trained["continuous_keys"]
    ordinal_keys = trained["ordinal_keys"]
    emb_keys = trained.get("emb_keys", [])

    raw_categorical = [[feature_vector.get(k) if feature_vector.get(k) is not None else "__missing__"
                         for k in categorical_keys]]
    encoded_categorical = trained["encoder"].transform(raw_categorical)[0]

    row = list(encoded_categorical)
    row += [feature_vector.get(k) if feature_vector.get(k) is not None else -1.0 for k in continuous_keys]
    row += [feature_vector.get(k) if feature_vector.get(k) is not None else -1.0 for k in ordinal_keys]
    row += [feature_vector.get(k) if feature_vector.get(k) is not None else 0.0 for k in emb_keys]
    return row


def predict_performance(trained, feature_vector: dict) -> float:
    """Predict the performance-vs-median ratio for one draft's feature
    vector (from build_feature_vector), using a model from
    train_predictor. Returns a plain float — a point estimate, not a
    statistical claim with a confidence interval (that distinction
    matters: this is the prediction job, not the insight-mining job).

    feature_vector must already have emb_hook_* keys if the model was
    trained with text embeddings — build it via build_feature_vector(ann,
    text_embedder=trained["text_embedder"]), or just call
    predict_performance_for_annotation instead, which does that for you."""
    row = _vectorize_for_predict(trained, feature_vector)
    return float(trained["model"].predict([row])[0])


def predict_performance_for_annotation(trained: dict, annotation: dict) -> float:
    """Convenience wrapper: given a trained model and a raw annotation dict
    (same shape as what train_predictor's rows used), automatically builds
    the feature vector using the model's OWN fitted text_embedder (if any)
    and predicts. The real ergonomics fix over predict_performance: a caller
    predicting a new draft's performance shouldn't have to manually know
    whether this model was trained with embeddings or thread the fitted
    embedder through build_feature_vector themselves."""
    vec = build_feature_vector(annotation, text_embedder=trained.get("text_embedder"))
    return predict_performance(trained, vec)


def feature_importance(trained: dict, top_n: int = 10) -> list[dict]:
    """Ranked feature importances from a trained model's RandomForestRegressor
    — real gap this closes (user: "whats the point of [embeddings] if we
    cant see them"): any emb_hook_N feature in the ranking gets a real
    human-readable interpretation attached (high/low keyword contrast
    from the actual training hook_text values, via
    text_embedding_features.explain_component — same technique built for
    okn_mirror), not left as an opaque PCA label.

    Returns a list of {feature, raw_feature, importance, pct} dicts,
    sorted descending by importance, optionally with an "interpretation"
    key for emb_hook_* features. Does not filter by a minimum importance
    threshold (unlike okn_mirror's >0.01 cutoff) — top_n caps the list
    length instead, since this is meant for direct inspection, not a
    report section."""
    model = trained["model"]
    feature_names = (trained["categorical_keys"] + trained["continuous_keys"]
                      + trained["ordinal_keys"] + trained.get("emb_keys", []))
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for name, imp in ranked:
        entry = {
            "feature": READABLE_FEATURE_NAMES.get(name, name),
            "raw_feature": name,
            "importance": round(float(imp), 4),
            "pct": round(float(imp) * 100, 1),
        }
        if name.startswith("emb_hook_") and trained.get("text_embedder") is not None:
            component_idx = int(name.split("_")[-1])
            entry["interpretation"] = explain_component(trained["text_embedder"], component_idx)
        results.append(entry)
    return results
