#!/usr/bin/env python3
"""Standalone runner for okn_mirror — the faithful okn-analytics port.
NOT wired into orchestrate.run_ingestion_cycle; this is a comparison tool,
run on demand, not a production pipeline stage. See src/qualume/okn_mirror/
for the port itself and its documented deltas from upstream.

Usage:
    uv run python scripts/run_okn_mirror.py                       # every niche >= min size
    uv run python scripts/run_okn_mirror.py --niche "food/cooking/recipe content"
    uv run python scripts/run_okn_mirror.py --min-niche-size 100
    uv run python scripts/run_okn_mirror.py --skip-embeddings      # skip models 11-14 (faster)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qualume.okn_mirror.analyze_okn import analyze
from qualume.okn_mirror.loader import load_niche_dataframe
from qualume.okn_mirror.ml_engine_okn import run_ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niche", action="append", help="restrict to this niche (repeatable)")
    parser.add_argument("--min-niche-size", type=int, default=50)
    parser.add_argument("--out", default="data/okn_mirror_results.json")
    args = parser.parse_args()

    df = load_niche_dataframe(niches=args.niche, min_niche_size=args.min_niche_size)
    print(f"Loaded {len(df)} rows across {df['platform'].nunique()} niches "
          f"(min_niche_size={args.min_niche_size})")
    if df.empty:
        print("No data qualifies — nothing to run.")
        return

    print("\n=== analyze_okn (8 cross-niche modules) ===")
    t0 = time.time()
    analysis = analyze(df)
    print(f"  done in {time.time() - t0:.1f}s")

    ml_results = {}
    niches = df["platform"].unique().tolist()
    for i, niche in enumerate(niches):
        ndf = df[df["platform"] == niche]
        print(f"\n=== ml_engine_okn ({i+1}/{len(niches)}): '{niche}' ({len(ndf)} posts) ===")
        t0 = time.time()
        try:
            ml_results[niche] = run_ml(ndf, niche)
            status = ml_results[niche].get("status")
            print(f"  status={status}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            ml_results[niche] = {"status": "failed", "error": str(e)}
            print(f"  FAILED: {e}  ({time.time() - t0:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"analysis": analysis, "ml_results": ml_results}, indent=2, default=str
    ))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
