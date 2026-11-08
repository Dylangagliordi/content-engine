"""Second-opinion models trained on the SAME pooled feature matrix as
learn.prediction's primary RandomForestRegressor — ported from
okn_mirror/ml_engine_okn.py's _feature_importance (GradientBoosting) and
_engagement_predictor (MLPRegressor), per explicit user decision: "add in
isolation forest and k means and gradient boost and mlp regressor, all
wired into one system."

WHY second opinions, not replacements: RandomForest is more robust to
overfitting on small/noisy per-niche-interaction data (most of this
corpus sits at n=50-350 per niche even pooled); GradientBoosting usually
achieves better raw accuracy but is more hyperparameter-sensitive and
overfitting-prone at this scale; MLPRegressor is the weakest choice in
the tabular-ML literature at this n, but still a genuinely independent
model family. Agreement between RF/GB/MLP on which features matter is
real, currently-unused corroborating signal; disagreement is worth a
human looking closer at, not silently averaged away.

Trained on the IDENTICAL pooled-with-niche-feature matrix as
learn.prediction.train_predictor (see that module + learn.training_data's
fetch_training_rows_pooled) - not a separate feature pipeline.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from qualume.learn.prediction import _vectorize_for_predict, train_predictor

MIN_ROWS_FOR_MLP = 50  # okn_mirror's own MIN_POSTS_FOR_NN=15 floor was too
# low for THIS codebase's much higher-dimensional feature vector (up to
# ~30 embedding dims x 3 sources) - use the same floor as min_sample_size's
# own default rather than inventing a second number.


def train_ensemble(rows: list[tuple[dict, float]], min_sample_size: int = 50, **kwargs) -> dict:
    """Trains learn.prediction's primary RandomForest (via train_predictor,
    same kwargs/embedder logic, unchanged) PLUS a GradientBoostingRegressor
    and an MLPRegressor on the identical encoded feature matrix. Returns
    the RandomForest's `trained` dict (same shape as train_predictor's own
    return value, so model_persistence.py keeps working unchanged) with
    two added keys: "gb_model" (fitted GradientBoostingRegressor) and
    "mlp_model" (fitted MLPRegressor, or None if fewer than
    MIN_ROWS_FOR_MLP real rows - same honest "can't build this, don't fake
    it" discipline as every embedder in this codebase)."""
    trained = train_predictor(rows, min_sample_size=min_sample_size, **kwargs)

    X = [_vectorize_for_predict(trained, v) for v in trained["_training_vectors"]]
    y = trained["_training_targets"]

    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X, y)
    trained["gb_model"] = gb
    trained["gb_feature_importances"] = dict(zip(
        trained["categorical_keys"] + trained["continuous_keys"] + trained["ordinal_keys"] + trained["emb_keys"],
        gb.feature_importances_,
    ))

    if len(rows) >= MIN_ROWS_FOR_MLP:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation="relu", solver="adam",
                            max_iter=2000, random_state=42, early_stopping=True,
                            validation_fraction=0.15, alpha=0.01)
        mlp.fit(X_scaled, y)
        trained["mlp_model"] = mlp
        trained["mlp_scaler"] = scaler
    else:
        trained["mlp_model"] = None
        trained["mlp_scaler"] = None

    return trained


def predict_with_all_models(trained: dict, feature_vector: dict) -> dict:
    """Returns {"random_forest": float, "gradient_boosting": float,
    "mlp": float | None} - all three models' point estimates for the
    same feature vector, so a caller can see agreement/disagreement
    directly rather than only getting RandomForest's single number."""
    row = _vectorize_for_predict(trained, feature_vector)
    result = {
        "random_forest": float(trained["model"].predict([row])[0]),
        "gradient_boosting": float(trained["gb_model"].predict([row])[0]),
    }
    if trained.get("mlp_model") is not None:
        scaled_row = trained["mlp_scaler"].transform([row])
        result["mlp"] = float(trained["mlp_model"].predict(scaled_row)[0])
    else:
        result["mlp"] = None
    return result
