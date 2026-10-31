# Qualume — Session Handoff (2026-10-30, visual mining + BH correction)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified live against real data, not assumed)
- 307/307 tests passing. Git clean.
- n8n (6-stage cycle, launchd-managed server) runs every 6 hours: poll -> download -> annotate ->
  classify -> mine (Phase A, declared taxonomy) -> cluster (Phase B, emergent/HDBSCAN).
- 95 external videos, all fully processed. 70 have real transcripts and are embedded; 25
  genuinely have no spoken copy (product-only/voiceover shots), correctly excluded.
- on_screen_text accuracy VERIFIED this session: extracted real video frames via ffmpeg, ran
  independent tesseract OCR against 4 frames from 2 videos, confirmed matches against stored
  text (3/4 clean, 1/4 fragment-level due to busy background) — the data is trustworthy, not
  fabricated.

## Real gap closed: visual/presenter data now mined (was captured, never tested before today)
New `learn/array_mining.py` handles two shapes the existing mining modules didn't cover:
- `mine_array_value` — JSONB array columns (music_mood_v2, script_structure_tags), no hardcoded
  taxonomy, real distinct values auto-discovered from data
- `mine_nested_field_value` — JSONB array-of-objects with a nested field (presenters[].
  relatable_archetype / apparent_age_range / apparent_gender_presentation / delivery_affect)
Wired into `run_mining_pass` as 2 new loops. Added `delivery_mode` to the existing flat-column
mineable list too (quick win, was just never added).

## Real gap closed: multiple-comparison correction (Benjamini-Hochberg)
New `learn/multiple_comparison.py` — textbook BH procedure (hand-verified worked example in
tests). Wired into BOTH `run_mining_pass` (restructured to collect all p-values first, THEN
decide significance as one batch) and `run_cluster_pass`.

**Real, honest result from running this against actual data**: the mining pass now tests 72
features/cycle (up from 46 before this session's visual-field wiring). Under BH correction,
**0/72 survive** — versus 10/72 under the old flat p<0.05 (including what had looked like the
strongest signal in the whole dataset, presenters.delivery_affect='excitedly sharing' at 3.68x).
This means the old mining "findings" were very likely mostly noise amplified by running that many
simultaneous tests against only 95 videos, not real signal. **User explicitly chose to keep BH as
the new standard** rather than relax it — 0 honest findings beats false confidence. All 10 old
proposed mining insights are now correctly tier='retired' in the DB.

Clustering pass is far less affected by this (only ~7 tests/cycle, not 72) — still 3/7 survive,
unchanged: "live-shopping event announcements" (5.16x), two underperforming clusters.

## Real bugs found and fixed this session (beyond the two features above)
- 3 pre-existing orchestrate tests were missing mocks for run_mining_pass/run_cluster_pass
  (gap dating to when those stages were added LAST session) — once those stages started doing
  real work this session, the tests started hanging 10+ minutes each on live DB/Voyage calls
  instead of running as fast mocked units. Fixed, confirmed <4s now.
- Same unmocked-module-constant pattern hit 5 pre-existing run_mining_pass tests the moment the
  new array/nested loops went live (missing MINEABLE_ARRAY_COLUMNS/MINEABLE_NESTED_FIELDS mocks).
- 2 more leaked test-fixture accounts found in the real watchlist mid-session (4th real incident
  of this exact gap across sessions — no separate test DB remains the single most recurring real
  problem this project has hit).

## What's actually NOT done yet (don't assume otherwise)
- With 0 mining findings currently live, there is nothing from the declared-taxonomy pipeline to
  act on right now — that's correct and honest, not broken. Needs more real data volume (more
  videos, more accounts) before it can produce statistically trustworthy findings again.
- Cross-account/cross-niche replication auto-check (learn.replication.check_segment_replication,
  built, tested) still not wired into either automated pipeline.
- No owned content published — insights_owned pipeline (seed_owned_insight/record_owned_evidence)
  has nothing real to validate against yet.
- shot_breakdown's rich per-shot data (camera movement, lighting, colors, setting changes) is
  STILL not mined directly — only the derived continuous features in learn/features.py exist
  (built for the prediction.py random forest, not wired into mining as correlational tests yet).
  Real next visual-data gap if more signal is wanted from that layer.
- Instagram account-risk decision (Part 3) — still unresolved, untouched for weeks.

## Immediate next action (pick one)
1. Let the pipeline run for real weeks/months — more data volume is the actual fix for "0 findings
   survive BH correction right now"; nothing to build, just time + continued automated polling.
2. Wire shot_breakdown's continuous features (cuts_per_min, avg_shot_length, color_temperature_
   consistency, etc. — already computed in learn/features.py) into a numeric-correlation mining
   pass (needs Spearman/Pearson correlation, not the Mann-Whitney group-comparison tests used
   everywhere else — a genuinely different statistical test, not built yet).
3. Wire learn.replication's cross-account check into both pipelines as an additional filter.
4. Finally build the separate test DB (Part 1 tech debt) —4 real incidents now, the strongest
   case yet for prioritizing this over any single new feature.
5. Publish one of the 5 draft IG statics to finally start the insights_owned validation loop.
