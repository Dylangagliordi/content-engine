#!/usr/bin/env python3
"""Side-by-side comparison: qualume's own RandomForest (learn.prediction,
trained on Gemini-annotated video features) vs okn_mirror's GradientBoosting
feature importance + the rest of okn's ML suite (trained on caption text +
engagement + timing only) — run on the SAME real niches, and where possible
the SAME real items, so the two approaches' outputs are directly comparable.

Three comparison modes per niche:
  A) qualume RF  -> on items that have been Gemini-annotated AND have a
     performance snapshot (the only items qualume's own pipeline can use).
  B) okn_mirror, SAME item set as (A) -> true apples-to-apples: same 50-300
     real videos, two different feature sets (annotation-derived vs
     caption/metrics-derived), two different models (RF vs GradientBoosting).
  C) okn_mirror, FULL niche dataset (every item with a snapshot, annotated
     or not) -> shows okn_mirror's real coverage advantage: it can use ALL
     the data qualume's annotation backlog is currently gatekeeping.

Writes real results to data/okn_vs_qualume_comparison.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qualume.learn.prediction import build_feature_vector, train_predictor
from qualume.learn.training_data import _fetch_candidate_items
from qualume.learn.baseline import performance_vs_baseline
from qualume.okn_mirror.loader import load_niche_dataframe
from qualume.okn_mirror.ml_engine_okn import run_ml

NICHES = [
    "fitness/workout content",
    "women's health supplements",
    "food/cooking/recipe content",
    "mom tok / parenting",
    "small business/entrepreneur content",
    "general health & nutrition",
    "menopause/hormone health",
]
MIN_SAMPLE_SIZE = 50


def fetch_training_rows_with_ids(niche: str):
    """Same as learn.training_data.fetch_training_rows but also returns the
    item_id for each row, so okn_mirror can be filtered to the EXACT same
    item set for a true apples-to-apples comparison (mode B above)."""
    candidates = _fetch_candidate_items(niche)
    rows = []
    for c in candidates:
        try:
            perf = performance_vs_baseline(c["item_id"])
        except ValueError:
            continue
        annotation = {
            "hook_type": c["hook_type"], "format_tag": c["format_tag"],
            "energy_level": c["energy_level"], "shot_breakdown": c["shot_breakdown"],
            "full_transcript": c["full_transcript"], "video_duration_seconds": c["video_duration_seconds"],
        }
        rows.append((c["item_id"], annotation, float(perf["ratio_to_median"])))
    return rows


def run_qualume_rf(niche: str, rows_with_ids):
    rows = [(ann, ratio) for _, ann, ratio in rows_with_ids]
    if len(rows) < MIN_SAMPLE_SIZE:
        return {"status": "insufficient_data", "n": len(rows)}
    trained = train_predictor(rows, min_sample_size=MIN_SAMPLE_SIZE)
    model = trained["model"]
    vectors = [build_feature_vector(ann) for ann, _ in rows]
    targets = [ratio for _, ratio in rows]
    categorical_keys = trained["categorical_keys"]
    continuous_keys = trained["continuous_keys"]
    ordinal_keys = trained["ordinal_keys"]
    raw_categorical = [[v.get(k) if v.get(k) is not None else "__missing__" for k in categorical_keys] for v in vectors]
    encoded_categorical = trained["encoder"].transform(raw_categorical)
    X = []
    for i, v in enumerate(vectors):
        row = list(encoded_categorical[i])
        row += [v.get(k) if v.get(k) is not None else -1.0 for k in continuous_keys]
        row += [v.get(k) if v.get(k) is not None else -1.0 for k in ordinal_keys]
        X.append(row)
    in_sample_r2 = model.score(X, targets)
    feature_names = categorical_keys + continuous_keys + ordinal_keys
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    ratios = targets
    return {
        "status": "ok", "n": len(rows), "in_sample_r2": round(float(in_sample_r2), 4),
        "top_features": [{"feature": n, "importance": round(float(i), 4)} for n, i in importances[:6] if i > 0.01],
        "ratio_range": {"min": round(min(ratios), 2), "max": round(max(ratios), 2),
                         "median": round(sorted(ratios)[len(ratios)//2], 2)},
    }


def summarize_okn(result: dict) -> dict:
    fi = result.get("feature_importance", {})
    cc = result.get("content_clusters", {})
    an = result.get("anomalies", {})
    return {
        "status": result.get("status"),
        "n": result.get("n_posts"),
        "feature_importance_r2": fi.get("r2_score"),
        "top_features": fi.get("top_features", [])[:6],
        "n_clusters": cc.get("n_clusters"),
        "clusters": [
            {"label": c["label"], "size": c["size"], "avg_engagement_rate": c["avg_engagement_rate"],
             "dominant_type": c["dominant_type"]}
            for c in cc.get("clusters", [])
        ],
        "total_anomalies": an.get("total_anomalies"),
        "momentum_score": result.get("momentum_score", {}).get("total_score"),
    }


def main():
    comparison = {}
    for niche in NICHES:
        print(f"\n{'='*70}\n{niche}\n{'='*70}")
        rows_with_ids = fetch_training_rows_with_ids(niche)
        print(f"qualume annotated+baselined items: {len(rows_with_ids)}")

        print("-- A) qualume RF (annotation-derived features) --")
        t0 = time.time()
        qualume_result = run_qualume_rf(niche, rows_with_ids)
        print(f"   status={qualume_result.get('status')}  ({time.time()-t0:.1f}s)")

        item_ids = {item_id for item_id, _, _ in rows_with_ids}
        full_df = load_niche_dataframe(niches=[niche], min_niche_size=1)

        print(f"-- B) okn_mirror, SAME {len(item_ids)} items (apples-to-apples) --")
        matched_df = full_df[full_df["post_id"].isin(item_ids)]
        t0 = time.time()
        if len(matched_df) >= 10:
            okn_matched = run_ml(matched_df, niche)
        else:
            okn_matched = {"status": "insufficient_data", "n_posts": len(matched_df)}
        print(f"   status={okn_matched.get('status')}  n={len(matched_df)}  ({time.time()-t0:.1f}s)")

        print(f"-- C) okn_mirror, FULL niche dataset ({len(full_df)} items, annotated or not) --")
        t0 = time.time()
        okn_full = run_ml(full_df, niche)
        print(f"   status={okn_full.get('status')}  n={len(full_df)}  ({time.time()-t0:.1f}s)")

        comparison[niche] = {
            "qualume_rf_annotated_subset": qualume_result,
            "okn_mirror_same_subset": summarize_okn(okn_matched),
            "okn_mirror_full_dataset": summarize_okn(okn_full),
        }

    out_path = Path("data/okn_vs_qualume_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\n\nFull comparison written to {out_path}")


if __name__ == "__main__":
    main()
