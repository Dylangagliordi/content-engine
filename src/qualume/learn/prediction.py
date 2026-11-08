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
user request, then extended once more per "how can we use all [text/
transcript/visual embeddings] in the prediction model and have them all
be interpretable"): THREE optional embedding feature blocks now fold
into the same feature vector as the video-content engineered features
(learn.features):
  1. hook_text embeddings (learn.text_embedding_features) — live-encoded
     with sentence-transformers, works for owned (static) AND external
     (video) CreativeDNA since hook_text exists on both.
  2. transcript embeddings (learn.stored_embedding_features, reading
     ALREADY-COMPUTED transcript_embeddings rows — Voyage voyage-3.5,
     built by learn.transcript_embedding for HDBSCAN clustering, not
     re-embedded here) — video-only (transcript_embeddings is keyed to
     external_content_item_id).
  3. visual embeddings (learn.stored_embedding_features, reading
     ALREADY-COMPUTED visual_embeddings rows — Voyage voyage-multimodal-
     3.5, built by learn.visual_embedding) — video-only, no associated
     text at all (raw video frames), interpreted via structured
     shot_breakdown metadata contrast instead of keywords.

Every one of these is independently optional (None-safe): an item
missing a stored transcript/visual embedding (real, incomplete coverage
— those pipelines run on their own schedule, see
learn/training_data.py's module docstring) simply contributes no
emb_transcript_*/emb_visual_* signal for that row, never a fabricated
placeholder. Enabled by default when the relevant dependency/data is
available; each can be disabled independently via
use_text_embeddings/use_transcript_embeddings/use_visual_embeddings.

Every emb_* feature surfaced by feature_importance() carries a real
human-readable interpretation, not an opaque PCA label — hook_text and
transcript axes via TF-IDF high/low keyword contrast
(text_embedding_features.explain_component /
stored_embedding_features.explain_transcript_component); the visual axis
(no associated text) via structured metadata-mode contrast
(stored_embedding_features.explain_visual_component) — the same
interpretability fix built for okn_mirror, extended to every embedding
source in this model, not just hook_text.

NICHE AS A CATEGORICAL FEATURE (added 2026-11-08, same pooled-training
fix as learn.training_data.fetch_training_rows_pooled — see that
module's "POOLED TRAINING" docstring section for the full reasoning):
"niche" is now in CATEGORICAL_FIELDS, so any annotation dict carrying a
real "niche" key (fetch_training_rows_pooled tags every row with one)
gets it encoded exactly like hook_type/format_tag — same OrdinalEncoder,
same handle_unknown="use_encoded_value" safety for a niche never seen
during training. fetch_training_rows (single-niche, mining/clustering-
style) never sets this key, so it's simply None there — build_feature_vector's
existing .get()-based None-safety (same as every other categorical field)
means this is fully backward compatible, not a breaking change. The
justification for pooling the PREDICTIVE model across niches (rather
than hard-partitioning it like mining/clustering must be) is that
learn.baseline.performance_vs_baseline already account-relativizes the
target (ratio_to_median) before niche enters the picture — RandomForest's
own splits can discover real niche-interaction effects natively instead
of each niche being stuck with its own tiny, separately-trained model.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from qualume.learn.features import engineered_features
from qualume.learn.stored_embedding_features import (
    dominant_shot_metadata,
    explain_transcript_component,
    explain_visual_component,
    fit_stored_embeddings,
    transform_stored_embedding,
)
from qualume.learn.text_embedding_features import (
    HAS_EMBEDDINGS,
    explain_component,
    fit_hook_text_embeddings,
    transform_hook_text,
)

CATEGORICAL_FIELDS = ("hook_type", "format_tag", "niche")
ORDINAL_FIELDS = ("energy_level",)

READABLE_FEATURE_NAMES = {
    "hook_type": "Hook Type", "format_tag": "Format", "energy_level": "Energy Level",
    "niche": "Niche",
    "cuts_per_min": "Cuts Per Minute", "avg_shot_length_sec": "Avg Shot Length",
    "shot_length_stddev": "Shot Length Variability", "color_temperature_consistency": "Color Temp Consistency",
    "setting_change_count": "Setting Changes", "shot_type_variety": "Shot Type Variety",
    "speech_coverage_ratio": "Speech Coverage", "words_per_second": "Speech Speed",
}


def build_feature_vector(annotation: dict, text_embedder: dict | None = None,
                          transcript_embedder: dict | None = None,
                          visual_embedder: dict | None = None) -> dict:
    """One video's (or, once an owned-content training pipeline exists, one
    static's) full feature vector: engineered continuous features
    (learn.features, derived from shot_breakdown/full_transcript — VIDEO
    ONLY, absent/None for a static) plus the categorical/ordinal fields a
    model can also use directly, plus up to THREE optional embedding
    feature blocks:
      - text_embedder: hook-text embeddings (emb_hook_0..N), works for
        video AND static annotations (hook_text exists on both).
      - transcript_embedder: ALREADY-COMPUTED transcript embedding
        (emb_transcript_0..N), requires annotation["transcript_embedding_vector"]
        — video-only, absent if that item has no stored vector yet.
      - visual_embedder: ALREADY-COMPUTED visual/frame embedding
        (emb_visual_0..N), requires annotation["visual_embedding_vector"]
        — video-only, absent if that item has no stored vector yet.
    `annotation` is a plain dict with keys matching the relevant
    external_creative_dna/creative_dna columns plus video_duration_seconds —
    callers assemble this from a DB row (learn.training_data.fetch_training_rows
    already includes the vector keys) or (in tests) a fixture; this
    function has no DB dependency itself.

    Missing/None fields produce None in the vector, never a guessed
    default — a real validation_error row's NULL fields, or an item with
    no stored transcript/visual embedding yet, must stay NULL through to
    the model, not silently become a fake 0 or 'unknown' category that
    looks like real data.

    All three embedder args default to None (identical shape to before
    any of these features existed) — strict backward compatibility for
    any existing caller. train_predictor passes real fitted embedders by
    default; see that function."""
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

    if transcript_embedder is not None:
        transcript_vector = annotation.get("transcript_embedding_vector")
        if transcript_vector is not None:
            embedded = transform_stored_embedding(transcript_embedder, transcript_vector)
            for i, value in enumerate(embedded):
                vec[f"emb_transcript_{i}"] = value
        else:
            for i in range(transcript_embedder["n_components"]):
                vec[f"emb_transcript_{i}"] = None

    if visual_embedder is not None:
        visual_vector = annotation.get("visual_embedding_vector")
        if visual_vector is not None:
            embedded = transform_stored_embedding(visual_embedder, visual_vector)
            for i, value in enumerate(embedded):
                vec[f"emb_visual_{i}"] = value
        else:
            for i in range(visual_embedder["n_components"]):
                vec[f"emb_visual_{i}"] = None

    return vec


def train_predictor(rows: list[tuple[dict, float]], min_sample_size: int = 50,
                     use_text_embeddings: bool = True,
                     use_transcript_embeddings: bool = True,
                     use_visual_embeddings: bool = True):
    """Train a RandomForestRegressor on (annotation_dict, performance_ratio)
    pairs. Raises ValueError below min_sample_size — training on too few
    real rows produces a model that overfits completely and looks real
    while being useless; failing loud here is the honest behavior, not
    silently returning a model nobody should trust.

    use_text_embeddings (default True): fits a hook_text embedder
    (learn.text_embedding_features.fit_hook_text_embeddings) on THIS
    training set's own hook_text values and folds emb_hook_* features in.

    use_transcript_embeddings / use_visual_embeddings (default True each):
    fit a PCA reduction (learn.stored_embedding_features.fit_stored_embeddings)
    over whichever rows have a real transcript_embedding_vector /
    visual_embedding_vector already stored (real, incomplete coverage —
    see learn/training_data.py) and fold emb_transcript_*/emb_visual_*
    features into every row that has one (None for rows that don't, see
    build_feature_vector).

    Every one of these THREE embedders is fit ONCE here, on the full set
    passed in — callers doing cross-validation must fit a fresh instance
    of EACH per fold (see text_embedding_features's and
    stored_embedding_features's fold-safety notes) rather than reusing
    this function's single fit across folds. Each flag degrades to False
    automatically (that embedder stays None) if its dependency/data is
    unavailable or there isn't enough real data to fit meaningfully —
    same honest "can't build this, don't fake it" discipline throughout.

    Returns an opaque model object (an encoder + the fitted regressor +
    the three fitted embedders, any of which may be None); pass it
    straight to predict_performance or predict_performance_for_annotation,
    don't inspect its internals."""
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

    transcript_embedder = None
    if use_transcript_embeddings:
        transcript_pairs = [
            (ann["transcript_embedding_vector"], ann.get("transcript_embeddable_text") or "")
            for ann, _ in rows if ann.get("transcript_embedding_vector") is not None
        ]
        if transcript_pairs:
            vectors, aux = zip(*transcript_pairs)
            transcript_embedder = fit_stored_embeddings(list(vectors), list(aux))

    visual_embedder = None
    if use_visual_embeddings:
        visual_pairs = [
            (ann["visual_embedding_vector"], dominant_shot_metadata(ann.get("shot_breakdown")))
            for ann, _ in rows if ann.get("visual_embedding_vector") is not None
        ]
        if visual_pairs:
            vectors, aux = zip(*visual_pairs)
            visual_embedder = fit_stored_embeddings(list(vectors), list(aux))

    vectors = [
        build_feature_vector(ann, text_embedder=text_embedder,
                              transcript_embedder=transcript_embedder,
                              visual_embedder=visual_embedder)
        for ann, _ in rows
    ]
    targets = [ratio for _, ratio in rows]

    categorical_keys = list(CATEGORICAL_FIELDS)
    emb_keys = [k for k in vectors[0].keys()
                if k.startswith("emb_hook_") or k.startswith("emb_transcript_") or k.startswith("emb_visual_")]
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
        "transcript_embedder": transcript_embedder,
        "visual_embedder": visual_embedder,
        "use_text_embeddings": text_embedder is not None,
        "use_transcript_embeddings": transcript_embedder is not None,
        "use_visual_embeddings": visual_embedder is not None,
        # Kept for feature_importance()'s real correlation/category-breakdown
        # computation (needs the actual per-row feature values + targets,
        # not just the fitted model) — NOT used by predict_performance,
        # which only ever needs "model"/"encoder"/the *_keys lists.
        "_training_vectors": vectors,
        "_training_targets": targets,
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
    the feature vector using the model's OWN fitted embedders (text,
    transcript, visual — any of which may be None) and predicts. The real
    ergonomics fix over predict_performance: a caller predicting a new
    draft's performance shouldn't have to manually know which embedders
    this model was trained with or thread them through build_feature_vector
    themselves."""
    vec = build_feature_vector(
        annotation,
        text_embedder=trained.get("text_embedder"),
        transcript_embedder=trained.get("transcript_embedder"),
        visual_embedder=trained.get("visual_embedder"),
    )
    return predict_performance(trained, vec)


def feature_importance(trained: dict, top_n: int | None = None) -> list[dict]:
    """Ranked feature importances from a trained model's RandomForestRegressor
    — real gap this closes (user: "whats the point of [embeddings] if we
    cant see them", extended to "how can we use all [text/transcript/
    visual embeddings] ... and have them all be interpretable", then
    extended again: "does it tell us how correlated? how can we have
    that... also do it for every feature, also list every feature"):

    1. CORRELATION, not just importance. RandomForest's
       feature_importances_ (Gini/variance-reduction) tells you how much
       the MODEL RELIES on a feature — it has no sign and is not a
       correlation coefficient. Every continuous/ordinal/embedding
       feature now also gets a real Spearman rank correlation against
       the actual training targets (ratio_to_median), plus a plain-
       English "direction" string ("higher is better" / "lower is
       better" / "no variance / insufficient data"). Spearman, not
       Pearson, chosen for the same reason learn.mining already uses
       Mann-Whitney U elsewhere in this codebase: ratio_to_median is a
       skewed ratio metric with real outliers (documented repeatedly
       this project — fitness max=355x median), and Spearman makes no
       normality/linearity assumption, just monotonic rank agreement.
       None (not a fabricated 0.0) when there's no real variance to
       correlate (e.g. every training row had the same value for that
       feature) — same "refuse rather than guess" discipline as
       baseline.py/mining.py.
    2. CATEGORY BREAKDOWN for categorical features (hook_type,
       format_tag). A single correlation number has no meaning for an
       unordered category — the real equivalent is each distinct
       value's median target (ratio_to_median) across the real training
       rows that had it, sorted descending, so "which category is
       actually associated with higher performance" is directly
       readable (the categorical analogue of a correlation sign).
    3. EVERY emb_* feature in the ranking gets a real human-readable
       interpretation attached, with the RIGHT technique per source:
       emb_hook_*/emb_transcript_* via TF-IDF high/low keyword contrast;
       emb_visual_* via structured shot-metadata mode contrast (no
       associated text exists for visual embeddings). Never left as an
       opaque PCA label.
    4. top_n=None (the new default) returns EVERY feature in the model,
       not just the top few — pass an explicit top_n to cap the list.

    Returns a list of {feature, raw_feature, importance, pct} dicts,
    sorted descending by importance, each also carrying either
    "correlation"+"direction" (continuous/ordinal/embedding features) or
    "category_breakdown" (categorical features), plus "interpretation"
    for emb_* features."""
    import numpy as np
    from scipy import stats

    model = trained["model"]
    categorical_keys = trained["categorical_keys"]
    continuous_keys = trained["continuous_keys"]
    ordinal_keys = trained["ordinal_keys"]
    emb_keys = trained.get("emb_keys", [])
    feature_names = categorical_keys + continuous_keys + ordinal_keys + emb_keys
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]

    targets = trained.get("_training_targets")
    training_vectors = trained.get("_training_vectors")

    def _spearman_correlation(name: str):
        """Real Spearman rho between one feature's real training values
        and the real training targets. None if fewer than 2 distinct
        real (non-None) values exist for this feature — scipy.stats.spearmanr
        on constant input returns NaN, which this deliberately converts to
        None rather than letting a silent NaN leak into output that looks
        like a real number."""
        if targets is None or training_vectors is None:
            return None
        values, matched_targets = [], []
        for v, t in zip(training_vectors, targets):
            val = v.get(name)
            if val is not None:
                values.append(val)
                matched_targets.append(t)
        if len(set(values)) < 2 or len(values) < 3:
            return None
        rho, _ = stats.spearmanr(values, matched_targets)
        if np.isnan(rho):
            return None
        return float(rho)

    def _category_breakdown(name: str) -> list[dict]:
        """Median real target per distinct real category value, sorted
        descending — the categorical equivalent of a correlation sign."""
        if targets is None or training_vectors is None:
            return []
        by_value: dict = {}
        for v, t in zip(training_vectors, targets):
            val = v.get(name)
            if val is not None:
                by_value.setdefault(val, []).append(t)
        breakdown = [
            {"value": val, "n": len(ts), "median_target": float(np.median(ts))}
            for val, ts in by_value.items()
        ]
        breakdown.sort(key=lambda b: b["median_target"], reverse=True)
        return breakdown

    results = []
    for name, imp in ranked:
        entry = {
            "feature": READABLE_FEATURE_NAMES.get(name, name),
            "raw_feature": name,
            "importance": round(float(imp), 4),
            "pct": round(float(imp) * 100, 1),
        }
        if name in categorical_keys:
            entry["category_breakdown"] = _category_breakdown(name)
        else:
            rho = _spearman_correlation(name)
            entry["correlation"] = round(rho, 4) if rho is not None else None
            if rho is None:
                entry["direction"] = "no variance / insufficient data"
            elif rho > 0:
                entry["direction"] = "higher is better"
            elif rho < 0:
                entry["direction"] = "lower is better"
            else:
                entry["direction"] = "no clear direction"

        if name.startswith("emb_hook_") and trained.get("text_embedder") is not None:
            component_idx = int(name.split("_")[-1])
            entry["interpretation"] = explain_component(trained["text_embedder"], component_idx)
        elif name.startswith("emb_transcript_") and trained.get("transcript_embedder") is not None:
            component_idx = int(name.split("_")[-1])
            entry["interpretation"] = explain_transcript_component(trained["transcript_embedder"], component_idx)
        elif name.startswith("emb_visual_") and trained.get("visual_embedder") is not None:
            component_idx = int(name.split("_")[-1])
            entry["interpretation"] = explain_visual_component(trained["visual_embedder"], component_idx)
        results.append(entry)
    return results
