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

NOT meaningful yet at today's real scale (58 videos, 1 niche, tiny by ML
standards) — built and tested now as a correctly-shaped scaffold, same
"capture the shape before the volume" discipline as replication.py and
validation.py in this same module.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from qualume.learn.features import engineered_features

CATEGORICAL_FIELDS = ("hook_type", "format_tag")
ORDINAL_FIELDS = ("energy_level",)


def build_feature_vector(annotation: dict) -> dict:
    """One video's full feature vector: engineered continuous features
    (learn.features, derived from shot_breakdown/full_transcript) plus
    the categorical/ordinal fields a model can also use directly.
    `annotation` is a plain dict with keys matching the relevant
    external_creative_dna columns plus video_duration_seconds — callers
    assemble this from a DB row or (in tests) a fixture; this function
    has no DB dependency itself.

    Missing/None fields produce None in the vector, never a guessed
    default — a real validation_error row's NULL fields must stay NULL
    through to the model, not silently become a fake 0 or 'unknown'
    category that looks like real data."""
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
    return vec


def train_predictor(rows: list[tuple[dict, float]], min_sample_size: int = 50):
    """Train a RandomForestRegressor on (annotation_dict, performance_ratio)
    pairs. Raises ValueError below min_sample_size — training on too few
    real rows produces a model that overfits completely and looks real
    while being useless; failing loud here is the honest behavior, not
    silently returning a model nobody should trust.

    Returns an opaque model object (an encoder + the fitted regressor);
    pass it straight to predict_performance, don't inspect its internals."""
    if len(rows) < min_sample_size:
        raise ValueError(
            f"train_predictor needs at least {min_sample_size} real rows, got {len(rows)} — "
            "training on fewer would overfit completely and produce a model that "
            "looks real but isn't trustworthy"
        )

    vectors = [build_feature_vector(ann) for ann, _ in rows]
    targets = [ratio for _, ratio in rows]

    categorical_keys = list(CATEGORICAL_FIELDS)
    continuous_keys = [k for k in vectors[0].keys() if k not in categorical_keys and k not in ORDINAL_FIELDS]
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
        X.append(row)

    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X, targets)

    return {
        "model": model,
        "encoder": encoder,
        "categorical_keys": categorical_keys,
        "continuous_keys": continuous_keys,
        "ordinal_keys": ordinal_keys,
    }


def predict_performance(trained, feature_vector: dict) -> float:
    """Predict the performance-vs-median ratio for one draft's feature
    vector (from build_feature_vector), using a model from
    train_predictor. Returns a plain float — a point estimate, not a
    statistical claim with a confidence interval (that distinction
    matters: this is the prediction job, not the insight-mining job)."""
    categorical_keys = trained["categorical_keys"]
    continuous_keys = trained["continuous_keys"]
    ordinal_keys = trained["ordinal_keys"]

    raw_categorical = [[feature_vector.get(k) if feature_vector.get(k) is not None else "__missing__"
                         for k in categorical_keys]]
    encoded_categorical = trained["encoder"].transform(raw_categorical)[0]

    row = list(encoded_categorical)
    row += [feature_vector.get(k) if feature_vector.get(k) is not None else -1.0 for k in continuous_keys]
    row += [feature_vector.get(k) if feature_vector.get(k) is not None else -1.0 for k in ordinal_keys]

    return float(trained["model"].predict([row])[0])
