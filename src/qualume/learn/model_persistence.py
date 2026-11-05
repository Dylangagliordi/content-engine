"""Model persistence — save/load a trained predictor so train_predictor
doesn't have to re-fetch rows and retrain from scratch every single call.

Real gap this closes (surfaced when the user asked "is it fully built
and merged and using every model"): learn.prediction.train_predictor
has never persisted anything — every call to predict_performance_for_
annotation implies re-running fetch_training_rows + train_predictor
first, in-process, every time. No model registry, no saved artifact,
nothing reusable across processes/requests. That's fine for a one-off
analysis script; it's a real blocker for wiring prediction into
anything live (e.g. run_batch.py scoring a new draft before publish —
see run_batch_scoring.py) where re-training per request would be slow
and wasteful.

Uses joblib (already a real dependency via scikit-learn, confirmed
installed — not a new one introduced here) — the standard, scikit-
learn-recommended serialization for sklearn estimators (handles
RandomForestRegressor/OrdinalEncoder/PCA objects correctly; raw pickle
would technically also work but joblib is more efficient for the numpy
arrays this objects owns, per sklearn's own persistence docs).

Stored per-niche (one file per niche, not one global model) — same
reasoning as training_data.fetch_training_rows being niche-REQUIRED,
not optional: a cross-niche model would be meaningless the same way
cross-niche mining/clustering was until the previous fix in this
session. Files live in data/models/ (gitignored, regenerable — never
committed, same as data/okn_mirror_results.json).

What's SAVED alongside the trained dict, for honest staleness/provenance
tracking (not just the model itself):
  - niche: which niche this model is for
  - trained_at: ISO timestamp
  - n_training_rows: real sample size this model was trained on
  - embedders_used: which of text/transcript/visual embeddings were
    actually fit (None vs a real fitted embedder) — so a caller loading
    a stale model can SEE what it was built with before trusting it,
    without having to inspect trained["text_embedder"] etc. themselves.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

logger = logging.getLogger("qualume.learn.model_persistence")

MODELS_DIR = Path("data/models")


def _niche_to_filename(niche: str) -> str:
    """Sanitize a real niche string (which can contain spaces, slashes,
    apostrophes — e.g. "women's health supplements",
    "food/cooking/recipe content") into a safe, COLLISION-RESISTANT
    filename. Real bug caught by this module's own test suite: a naive
    character-replacement sanitizer collapses "fitness/workout content"
    and "fitness workout content" to the identical string
    "fitness_workout_content" — two different real niches would then
    silently overwrite each other's saved model on disk, with no error
    and no indication anything went wrong. Fixed with a short SHA-256
    hash suffix of the EXACT original string appended to the readable
    slug — the slug stays human-skimmable in a directory listing, the
    hash guarantees two distinct niche strings never collide regardless
    of what punctuation they differ by."""
    import hashlib

    safe = "".join(c if c.isalnum() else "_" for c in niche.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    digest = hashlib.sha256(niche.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def model_path(niche: str) -> Path:
    return MODELS_DIR / f"{_niche_to_filename(niche)}.joblib"


def metadata_path(niche: str) -> Path:
    return MODELS_DIR / f"{_niche_to_filename(niche)}.meta.json"


def save_predictor(trained: dict, niche: str, n_training_rows: int) -> Path:
    """Persist a trained predictor (from learn.prediction.train_predictor)
    to disk, keyed by niche. Overwrites any existing saved model for this
    niche — same "latest wins, no versioned history" discipline as the
    rest of this project's non-append-only data (this is a cache/
    artifact, not a performance snapshot that must never be overwritten).

    Strips the two internal `_training_vectors`/`_training_targets` keys
    before saving — those are large (one dict + float per training row)
    and only needed transiently by feature_importance() in the SAME
    process that just trained the model; a caller reloading a saved
    model later doesn't need them for predict_performance and
    feature_importance degrades gracefully (empty correlation/breakdown)
    without them, same honest-gap discipline as everywhere else — not
    silently wrong, just less detailed on a later process's call to
    feature_importance.

    Returns the path written to."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in trained.items() if not k.startswith("_")}
    path = model_path(niche)
    joblib.dump(to_save, path)

    meta = {
        "niche": niche,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_training_rows": n_training_rows,
        "embedders_used": {
            "text": trained.get("text_embedder") is not None,
            "transcript": trained.get("transcript_embedder") is not None,
            "visual": trained.get("visual_embedder") is not None,
        },
    }
    metadata_path(niche).write_text(json.dumps(meta, indent=2))
    logger.info("Saved predictor for niche=%r to %s (n=%s)", niche, path, n_training_rows)
    return path


def load_predictor(niche: str) -> dict | None:
    """Load a previously-saved predictor for this niche, or None if none
    exists yet (never raises on a missing file — "no model saved yet" is
    a real, expected state for a niche nobody has trained, same
    "refuse rather than guess" discipline as learn.baseline returning
    None for an account with no snapshots)."""
    path = model_path(niche)
    if not path.exists():
        return None
    return joblib.load(path)


def load_metadata(niche: str) -> dict | None:
    """The saved metadata (trained_at/n_training_rows/embedders_used) for
    a niche's model, or None if none exists — lets a caller decide
    whether a saved model is stale (e.g. trained_at too long ago, or
    n_training_rows much smaller than the current real count) WITHOUT
    loading the full model artifact first."""
    path = metadata_path(niche)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def has_saved_predictor(niche: str) -> bool:
    return model_path(niche).exists()
