#!/usr/bin/env python3
"""Train + save the ONE pooled ensemble predictor across every real
niche clearing the overall sample-size floor - real fix (2026-11-08) for
the per-niche gap this script used to have: of 56 real niches with any
data, only a handful ever cleared a 50-row floor ALONE, leaving the
other 48+ niches permanently un-modeled. Pooling the training data
(niche folded in as a categorical feature - see
learn.training_data.fetch_training_rows_pooled / learn.prediction's
"niche" categorical field / learn.ensemble_models) lets every real row
contribute to ONE shared model instead of each niche needing its own
50-row floor in isolation.

Run this periodically (manually, or wire into a cron/scheduled job
later) to keep the saved pooled model fresh as new annotated data
accumulates - predict_performance_for_annotation / run_batch scoring
load whatever was last saved here, they don't retrain on demand.

Trains THREE models on the identical pooled feature matrix via
learn.ensemble_models.train_ensemble: RandomForestRegressor (primary,
persisted via model_persistence.save_pooled_predictor exactly as
before), plus GradientBoostingRegressor and MLPRegressor as second
opinions (gb_model/mlp_model land inside the same saved dict - see
ensemble_models.train_ensemble's own docstring for why these are kept
as corroborating signal, not a replacement for RandomForest).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qualume.learn.ensemble_models import train_ensemble
from qualume.learn.model_persistence import save_pooled_predictor
from qualume.learn.prediction import feature_importance
from qualume.learn.training_data import fetch_training_rows_pooled

MIN_SAMPLE_SIZE = 50


def main():
    # active_only is not a concept here the way it is for
    # list_distinct_niches(active_only=...) in the old per-niche loop -
    # fetch_training_rows_pooled(niches=None) already pulls every real
    # niche with annotated+baselined data regardless of whether its
    # account is still actively polled today, same "already-collected
    # history stays valid" reasoning as the old per-niche script had.
    print("Fetching pooled training rows across every real niche...")
    rows = fetch_training_rows_pooled(niches=None)
    print(f"Pooled rows: {len(rows)}")

    if len(rows) < MIN_SAMPLE_SIZE:
        print(f"FAILED: only {len(rows)} real rows, below the {MIN_SAMPLE_SIZE}-row floor. "
              f"Not saving a model trained on too little data.")
        return

    niches_represented = sorted({ann.get("niche") for ann, _ in rows if ann.get("niche")})
    print(f"Niches represented: {len(niches_represented)}")

    t0 = time.time()
    trained = train_ensemble(rows, min_sample_size=MIN_SAMPLE_SIZE)
    elapsed = time.time() - t0

    path = save_pooled_predictor(trained, n_training_rows=len(rows), niches_represented=niches_represented)
    print(f"[{elapsed:.1f}s] Trained + saved pooled predictor -> {path}")

    print(f"  RandomForest: real model, {('GB' if trained.get('gb_model') else 'no GB')}, "
          f"{('MLP' if trained.get('mlp_model') else 'no MLP (below MIN_ROWS_FOR_MLP)')}")

    fi = feature_importance(trained, top_n=10)
    print("\nTop 10 features (RandomForest importance):")
    for f in fi:
        print(f"  {f['feature']}: {f['pct']}%")


if __name__ == "__main__":
    main()
