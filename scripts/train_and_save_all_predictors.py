#!/usr/bin/env python3
"""Train + save a real predictor for every real niche clearing the
sample-size floor. Run this periodically (manually, or wire into a
cron/scheduled job later) to keep saved models fresh as new annotated
data accumulates — predict_performance_for_annotation / run_batch
scoring load whatever was last saved here, they don't retrain on demand.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qualume.db.watchlist import list_distinct_niches
from qualume.learn.model_persistence import save_predictor
from qualume.learn.prediction import train_predictor
from qualume.learn.training_data import fetch_training_rows

MIN_SAMPLE_SIZE = 50


def main():
    # active_only=False (unlike orchestrate.run_ingestion_cycle's live
    # mining/clustering loop, which defaults to active_only=True):
    # training uses ALREADY-INGESTED historical data regardless of
    # whether an account is still being actively polled today — real
    # discovery while verifying this script against production data:
    # fitness/workout content, women's health supplements, and food/
    # cooking/recipe content (the flagship niches with the most real
    # annotated+baselined data) are ALL currently deactivated accounts
    # (42 inactive accounts total). Deactivating future polling for an
    # account doesn't retroactively invalidate its real, already-
    # collected performance history — a model should still train on it.
    niches = list_distinct_niches(active_only=False)
    print(f"Checking {len(niches)} real active niches for the {MIN_SAMPLE_SIZE}-row floor...")
    trained_count = 0
    for niche in niches:
        rows = fetch_training_rows(niche)
        if len(rows) < MIN_SAMPLE_SIZE:
            continue
        t0 = time.time()
        try:
            trained = train_predictor(rows, min_sample_size=MIN_SAMPLE_SIZE)
            path = save_predictor(trained, niche=niche, n_training_rows=len(rows))
            print(f"  [{time.time()-t0:.1f}s] {niche} (n={len(rows)}) -> {path}")
            trained_count += 1
        except Exception as e:
            print(f"  FAILED {niche}: {e}")
    print(f"\nTrained and saved {trained_count} real niche models.")


if __name__ == "__main__":
    main()
