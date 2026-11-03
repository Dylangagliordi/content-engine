"""Real training run: fetch each qualifying niche's real DB rows, train
learn.prediction's RandomForestRegressor, report real R^2 + feature
importances. First time this scaffold has ever been run on real data.
"""
import sys
sys.path.insert(0, "src")

from sklearn.ensemble import RandomForestRegressor

from qualume.learn.training_data import fetch_training_rows
from qualume.learn.prediction import train_predictor, build_feature_vector

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

for niche in NICHES:
    print(f"\n{'='*70}\n{niche}\n{'='*70}")
    rows = fetch_training_rows(niche=niche)
    print(f"real rows fetched: {len(rows)}")

    if len(rows) < MIN_SAMPLE_SIZE:
        print(f"SKIPPED: below {MIN_SAMPLE_SIZE}-row floor")
        continue

    try:
        trained = train_predictor(rows, min_sample_size=MIN_SAMPLE_SIZE)
    except ValueError as e:
        print(f"SKIPPED: {e}")
        continue

    model: RandomForestRegressor = trained["model"]

    # Real R^2 on the SAME data it trained on (in-sample fit, not a
    # held-out test score — reported honestly as such, not dressed up
    # as generalization performance).
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
    print(f"in-sample R^2: {in_sample_r2:.4f}")

    feature_names = categorical_keys + continuous_keys + ordinal_keys
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    print("top feature importances:")
    for name, imp in importances[:8]:
        if imp > 0.01:
            print(f"  {name:30s} {imp:.4f}")

    ratios = targets
    print(f"performance ratio range: min={min(ratios):.2f} max={max(ratios):.2f} "
          f"median={sorted(ratios)[len(ratios)//2]:.2f}")
