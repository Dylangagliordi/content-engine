# Qualume — Session Handoff (2026-11-04, lifecycle gaps closed: niche-scoping, model persistence, run_batch scoring)

## Start here next session
Read this file first, not chat history. Most recent section is at the TOP — see
"2026-11-08: merge into one coherent system" below for the latest state.

## 2026-11-08: merge into one coherent system — per-niche ML removed, 4 okn_mirror pieces ported, pooled predictor live

Executed `.hermes/plans/2026-11-08_140848-merge-coherent-content-intelligence-system.md`
top to bottom (that plan already had the design decisions settled from earlier in this
session — see it for the full reasoning if picking this up fresh). Summary of what
actually shipped, every number verified live against the real Postgres instance, not
assumed:

### What changed
1. **`learn/prediction.py`**: `"niche"` added to `CATEGORICAL_FIELDS` — the predictive
   model now treats niche as a feature (same OrdinalEncoder mechanism as
   hook_type/format_tag) instead of being hard-partitioned by niche. Mining/clustering/
   anomaly-detection/tiering stay niche-REQUIRED (peer-relative claims need a peer
   group) — only the RandomForest/GB/MLP predictive models pool across niches now.
2. **`learn/ensemble_models.py`** (new): `train_ensemble()`/`predict_with_all_models()`
   — GradientBoostingRegressor + MLPRegressor trained on the IDENTICAL pooled feature
   matrix as the primary RandomForest, as second-opinion corroborating signal (ported
   from okn_mirror's `_feature_importance`/`_engagement_predictor`).
3. **`learn/anomaly_detection.py`** (new): `detect_anomalies(niche)` — IsolationForest,
   niche-scoped, flags `viral_outlier`/`underperformer_outlier` with no prior hypothesis
   needed (genuinely new capability vs. `learn.replication`, which only checks a KNOWN
   claim). Ported from okn_mirror's `_anomaly_detection`.
4. **`learn/performance_tiers.py`** (new): `compute_performance_tiers(niche)` — KMeans
   forced k=2-3, niche-scoped, buckets into `[TOP]`/`[AVG]`/`[LOW]` ordinal tiers — a
   distinct job from HDBSCAN's open-ended structural clusters. Ported from okn_mirror's
   `_content_clustering`.
5. **`orchestrate.py`**: anomaly detection + performance tiering wired in as the 8th/9th
   niche-scoped stages (same per-niche isolation shape as mining/clustering/
   visual_clustering — one niche's failure never blocks another, or another stage).
6. **`learn/model_persistence.py`**: added `save_pooled_predictor`/`load_pooled_predictor`/
   `load_pooled_metadata`/`has_saved_pooled_predictor` — a SEPARATE function pair from
   the niche-keyed `save_predictor`/`load_predictor` (both coexist on disk without
   collision, verified by a real test). Pooled metadata tracks `niches_represented`
   (sorted list + count) so a caller can see at a glance which real niches actually
   contributed, without loading the full artifact.
7. **`scripts/train_and_save_all_predictors.py`**: rewritten from "loop over niches,
   save N models" to "fetch pooled rows once, train one ensemble, save once."
8. **`run_batch.py`**: `score_variant_if_model_available` now loads the ONE pooled model
   (`has_saved_pooled_predictor`/`load_pooled_predictor`) and passes niche as part of
   the annotation dict (handled via `CATEGORICAL_FIELDS`) instead of looking up a
   per-niche model file. Works correctly with niche=None too (unseen-value safety, same
   as hook_type).

### Real numbers, verified live (not assumed)
- Pooled training rows: **2,149** (up from the old per-niche floor that left 48+ of 56
  real niches permanently un-modeled).
- Niches represented in the pooled dataset: **56** (the watchlist grew since the
  original 27-niche count cited earlier this session — confirms that number is stale;
  56 is the current real figure).
- `"niche"` as a feature: real, honest, mid-tier importance — **7.6-7.8%**, 3rd place in
  top-15 RandomForest feature importance (not dominant, not negligible) — validates the
  pooling decision without overclaiming it.
- Real production training run (`scripts/train_and_save_all_predictors.py`): RF + GB +
  MLP all trained successfully in ~33s, saved to `data/models/pooled_predictor.joblib`
  (gitignored, 25MB — the embedding dimensionality makes this artifact much larger than
  any single old per-niche model was).
- `run_batch.score_variant_if_model_available` verified live against the real saved
  pooled model: a real ScriptVariant scored successfully both with a real niche
  (fitness/workout content → 11.57x account-median) and with niche=None (19.91x).
- Full test suite: **486 passed**, 0 failures (up from 455 at the end of the previous
  session). Zero leftover `zz_test_*`/`test_*` DB fixtures after the run — every test's
  `finally`-block cleanup held.

### Real bug found + fixed as a byproduct (not part of the original plan, but blocking verification)
`tests/test_orchestrate.py` (17 of 18 tests) and both tests in
`tests/test_orchestrate_pause_ingestion.py` never mocked
`find_pending_content_type_classification`/`classify_and_store_content_type` — every
run of these tests was silently making REAL Claude API calls against the live
content_type backlog (504 pending items at time of discovery) and REAL DB queries
against all 56 real niches for mining/clustering, on every single test invocation. This
had been true since the content_type stage was added (well before this session) and
made the orchestrate test files take minutes instead of seconds while spending real API
budget per test run. Found because adding the new anomaly_detection/performance_tiers
stages made the pre-existing problem bad enough to time out outright. Fixed by adding
the missing mocks everywhere real_ingestion_cycle() is exercised; `test_orchestrate.py`
now runs in ~5s (was timing out at 180s+).

### What's explicitly OUT of scope (confirmed, don't re-add without asking)
okn_mirror's momentum score, content fatigue, posting cadence, TF-IDF caption-lift,
topic discovery, similar-post KNN, hashtag clusters, and all 8 rule-based
`analyze_okn.py` modules — discussed earlier this session as "genuinely new, not yet
built" but the user's final instruction narrowed scope to exactly: niche-split removal
+ IsolationForest + KMeans + GradientBoosting + MLPRegressor. Cross-validation is ALSO
still explicitly out of scope for this plan (see below) — don't conflate "add ensemble
models" with "cross-validate the ensemble."

## What's actually NOT done yet (don't assume otherwise)
- CROSS-VALIDATION — flagged repeatedly across many sessions now, still not built.
  Every R²/importance/correlation number produced so far (RF, GB, MLP, okn_mirror's own
  GB) is IN-SAMPLE only. Still the single most load-bearing open item before trusting
  any of these numbers for a real decision.
- `scripts/train_and_save_all_predictors.py` is not scheduled anywhere (cron/
  orchestrator) — the pooled model will go stale as new annotated data accumulates
  unless someone re-runs it by hand.
- `learn/anomaly_detection.py`/`learn/performance_tiers.py` are wired into
  `orchestrate.py` but have never run a FULL live cycle against production data yet
  (only verified via direct unit-level calls + the orchestrator's mocked test suite) —
  watch the first few real scheduled cycles and confirm they don't error out at real
  volume (56 niches × 2 new stages per cycle).
- Foundational docs rebuild, LangGraph agent skeleton, roster skill update — untouched,
  carried forward from earlier sessions.
- Instagram account-risk decision, owned-content publishing loop (actually publishing a
  generated static, not just scoring the draft), separate test DB — all still open,
  unchanged.

## Immediate next action when picking this back up
1. Cross-validation — still the single most load-bearing open item, flagged repeatedly.
2. Decide whether `train_and_save_all_predictors.py` (now pooled, one run instead of
   N) should run on a schedule so the saved pooled model doesn't go stale.
3. Watch the first few real live orchestrator cycles with anomaly_detection/
   performance_tiers actually running against production data (not just mocked tests)
   — confirm no crash/timeout at real 56-niche volume.
4. Run a real `run_batch.py` batch end-to-end via the actual CLI entry point (`python -m
   qualume.run_batch`) to confirm the pooled-model scoring path works outside direct
   function calls too — only verified via direct calls + tests so far.

---

## Project state (verified live via DB queries this session, not assumed)
- 17,999 total external_content_items, 1,553 annotated (external_creative_dna rows), 65
  distinct niches in watched_accounts (159 active accounts, 112 inactive).
- insights_external: still thin (2 proposed, 39 retired as of last check) — the niche-scoping
  fix below should start improving this going forward, but hasn't had a full live cycle to
  prove it yet.
- 8 real per-niche prediction models saved to data/models/ (gitignored) via
  scripts/train_and_save_all_predictors.py: fitness/workout, women's health supplements,
  food/cooking/recipe, mom tok/parenting, small business/entrepreneur, general health &
  nutrition, menopause/hormone health, Nephilim/giants content.
- Git clean. Last 5 commits (newest first): 2ddcb42 (run_batch scoring wired in), cdf96ea
  (model persistence), d9a195a (niche-scoped orchestrator loop), 6d637d3 (prior session log),
  150985c (Spearman correlation).

## What happened this session (second half): "is it fully built and using every model?" → 3 real gaps found and fixed

User asked directly whether the prediction model built in the first half of this session was
"fully built and merged and using every model." Answer, verified against real code: **no** —
the model was real and verified, but lived on three disconnected islands: (1) ingestion +
mining/clustering ran live but mining/clustering called with niche=None despite accepting a
niche param since Nov 2; (2) the prediction model was never persisted — every call retrained
from scratch; (3) run_batch.py (the actual script that writes/persists real drafts) never
called the prediction model at all, despite the model's own docstring saying its purpose was
"predicting a new draft's likely performance before publish." User said "fix those gaps" —
all three fixed this session, in order:

### Gap #1 fixed (commit d9a195a): niche-scoped mining/clustering wired into orchestrate
New `db.watchlist.list_distinct_niches(active_only=True)`. `orchestrate.run_ingestion_cycle`'s
mining/clustering/visual_clustering stages now loop over every real niche and call each pass
SCOPED to that niche (same per-item isolation discipline as every other stage — one niche's
failure doesn't block the rest). Verified live: 51 real active niches returned right now.

### Gap #2 fixed (commit cdf96ea): model persistence
New `learn.model_persistence` (joblib-based save/load, keyed per niche, with a metadata JSON
tracking trained_at/n_training_rows/which embedders were fit). New
`scripts/train_and_save_all_predictors.py`. Real bug caught by the test suite itself: a naive
filename sanitizer collapsed "fitness/workout content" and "fitness workout content" to the
identical filename — fixed with a SHA-256 hash suffix. Real discovery while running the
training script: `active_only=True` (matching the live orchestrator's default) returned ZERO
of the flagship niches — fitness/workout, women's-health, food/cooking are ALL on currently
DEACTIVATED accounts (112 inactive). Training deliberately uses `active_only=False` (a real,
documented, deliberate difference from the live mining loop) since deactivating future polling
doesn't invalidate an account's already-collected performance history. 8 real models trained
and saved; verified with a genuine fresh-process load+predict round trip.

### Gap #3 fixed (commit 2ddcb42): predict_performance_for_annotation wired into run_batch.py
New migration 020: `content_items.predicted_score` + `content_items.niche` (both nullable,
applied live to Postgres). New `run_batch.score_variant_if_model_available(variant, niche)` —
loads the niche's saved model and scores the variant's real hook_text; returns None (never
fabricates) when no niche given or no saved model exists. `run_batch()` gained an optional
`--niche` CLI arg; scores every surviving variant after the eval gate, before persisting.
Explicitly OBSERVATIONAL — does not gate/reject, same honesty discipline as the eval-gate
not silently overreaching into human decisions. Verified live against the real saved
fitness/workout model: a real ScriptVariant hook_text scored 3.84x account-median.

## Honest state of the lifecycle now (verified, not assumed)
Ingestion (live, scheduled) → embedding (two separate schedules: transcript via Voyage, visual
via Voyage multimodal, NOT gated on annotation) → mining/clustering (live, NOW niche-scoped) →
prediction (train_and_save_all_predictors.py — still a MANUAL script, not yet scheduled) →
run_batch.py (now DOES call the saved model, if one exists for the given niche). The remaining
manual step: nothing re-trains/re-saves predictors automatically as new annotated data
accumulates — scripts/train_and_save_all_predictors.py needs to be re-run by hand (or wired
into a cron job) to keep saved models fresh. Visual frame embeddings are used in prediction
but still not fused into the main HDBSCAN clustering comparison (that fusion question remains
explicitly deferred, unchanged from 2026-10-31).

## What's actually NOT done yet (don't assume otherwise)
- CROSS-VALIDATION — flagged repeatedly, still not built. Every R²/importance/correlation
  number from this session (and the merged model generally) is IN-SAMPLE only. Three separate
  embedders each have documented fold-safety leakage risk. Still the single most load-bearing
  open item before trusting any of these numbers for a real decision.
- train_and_save_all_predictors.py is not scheduled anywhere (cron/orchestrator) — models will
  go stale as new annotated data accumulates unless someone re-runs it by hand.
- Foundational docs rebuild, LangGraph agent skeleton, roster skill update — untouched, carried
  forward from earlier sessions.
- Instagram account-risk decision, owned-content publishing loop (actually publishing a
  generated static, not just scoring the draft), separate test DB — all still open, unchanged.

## Immediate next action when picking this back up
1. Cross-validation — the single most load-bearing open item, flagged three times now.
2. Decide whether train_and_save_all_predictors.py should run on a schedule (cron job or a new
   stage in orchestrate.run_ingestion_cycle) so saved models don't silently go stale.
3. Run a real run_batch.py batch with --niche set to confirm the full generate→score→persist
   path end-to-end in the real CLI (only verified so far via direct function calls, not the
   actual `python -m qualume.run_batch` entry point).
4. Watch the first few real live cycles with niche-scoped mining/clustering and see if
   insights_external actually grows past its current 2-proposed-row floor.

## Earlier context (preserved from the previous handoff, still accurate)
Mass TikTok ingestion, annotation-throughput fixes, content_type classifier, and the first
random-forest training all predate this session and are described in git log (commits
f48444d through 2e40411). The okn_mirror port + three-way embedding merge + Spearman
correlation work (first half of THIS session) is in commits 995e406 through 150985c.


## What happened this session: okn-analytics mirror + full embedding/correlation merge into prediction

### 1. okn_mirror — faithful port of CyberSystema/okn-analytics (commit 995e406, a119460)
User re-analyzed a previously-identified reference repo (CyberSystema/okn-analytics, a
social-media-intelligence pipeline: 8 rule-based analysis modules + 14 sklearn ML models) and
asked to mimic it "almost exactly" for a real side-by-side comparison. Built as a standalone
package (`src/qualume/okn_mirror/`), NOT wired into orchestrate.py, NOT touching
learn/mining.py or learn/transcript_clustering.py:
- `loader.py` — maps qualume's real Postgres schema into okn's exact UNIFIED_SCHEMA
  DataFrame. Key adaptation: okn's real axis of comparison is PLATFORM (one org, several
  platforms); qualume's is NICHE (one platform, many niches) — every "platform" loop in the
  port iterates over niche strings instead.
- `analyze_okn.py` — line-for-line port of OKNAnalyzer's 8 modules.
- `ml_engine_okn.py` — line-for-line port of all 14 ML models (NN predictor, GradientBoosting
  feature importance, KMeans clustering, IsolationForest anomalies, TF-IDF caption NLP,
  content fatigue, cadence, momentum score, root-cause analysis, topic discovery, similar-post
  predictor, hashtag clusters).
- Ran for real against 15,947 items / 44 niches via `scripts/run_okn_mirror.py` — all 44
  completed cleanly after fixing 2 real bugs found by actually running it (not just
  unit-testing): a PCA n_components edge case, and an engagement_rate dtype-upcast bug
  (int64→object the moment ANY row anywhere had reach=0, breaking every single niche's
  sklearn calls until fixed).
- Built `_explain_embedding_component` — real interpretability fix after the user pushed back
  on raw opaque `emb_N` labels ("whats the point of embeddings if we cant see them"):
  projects captions onto one PCA axis, contrasts high/low extremes via TF-IDF keywords + raw
  examples. Wired into both feature_importance and root_cause_analysis.
- `scripts/compare_okn_vs_qualume.py` — real side-by-side runner (qualume's own RF on
  annotation-derived features vs okn_mirror's GB on caption/metrics features, same real item
  sets, 7 niches). Results in `data/okn_vs_qualume_comparison.json` (gitignored, regenerable).

### 2. Three-way embedding merge into learn.prediction (commits 588dc70, 0c18570)
User asked to merge the okn-style approach into qualume's OWN prediction model (not just
compare), then asked two clarifying questions first: (a) can this work for IG statics? — no,
not directly: owned CreativeDNA has no shot_breakdown/full_transcript/duration at all
(video-only), but `hook_text` exists on BOTH owned and external CreativeDNA, so the merge is
built on hook_text specifically, making it reusable for statics once an owned-content training
pipeline exists. (b) why does qualume only have cuts_per_min-style features, not vectors? —
answer: qualume never built a text-embedding step at all, full stop, not a design choice.
Then user asked "we have text embedding AND video embedding right" — pointed out the real
asymmetry (hook_text embeddings were merged, but the ALREADY-COMPUTED transcript_embeddings/
visual_embeddings tables — 712+779 real rows — were sitting unused by prediction,
clustering-only) and the user said "use all [of them] in the prediction model, have them all
be interpretable":
- New `learn/text_embedding_features.py` — hook_text embeddings via sentence-transformers
  (live-encoded, local model, works for video AND static annotations).
- New `learn/stored_embedding_features.py` — fits a PCA reduction over the ALREADY-COMPUTED
  transcript_embeddings/visual_embeddings vectors (never re-calls Voyage — would duplicate
  real paid API cost). Transcript axis interpreted via TF-IDF keyword contrast on the exact
  original embedded text; visual axis (no associated text — raw video frames) interpreted via
  structured shot_breakdown metadata (setting/shot_type/color_temperature) mode contrast.
- New `learn/embedding_interpretability.py` — shared ranking/TF-IDF helpers extracted once
  both embedding modules needed the same logic (okn_mirror deliberately NOT refactored to use
  this — stays a dependency-free standalone mirror of upstream, by design).
- `learn/training_data.py` — now LEFT JOINs transcript_embeddings/visual_embeddings (real,
  INCOMPLETE coverage, never excludes or fabricates); reconstructs the exact transcript text
  via the same build_embeddable_text join learn.transcript_embedding itself uses, with a real
  stale-text edge case handled (if re-annotation changed hook_text/full_transcript after the
  embedding was computed, falls back to no text rather than showing a mismatch).
- `learn/prediction.py` — build_feature_vector/train_predictor/feature_importance now take 3
  independent None-safe embedder args; each can be toggled via
  use_text_embeddings/use_transcript_embeddings/use_visual_embeddings (all default True,
  each degrades to None automatically if unavailable).
- Real bug found+fixed during this work: training_data.py never selected hook_text from the DB
  at all (missing from the original SELECT) — every annotation's hook_text was silently None
  until fixed.
- Verified live against all 7 niches clearing the 50-row floor — all three embedders fit
  successfully despite real partial coverage (e.g. food/cooking: 110/150 transcript, 22/150
  visual). Every emb_* feature in every niche's top-10 carried a real, readable
  interpretation. Example: menopause niche's #1 feature was emb_visual_0 (0.63 importance) —
  HIGH = "outdoor porch" settings, LOW = "kitchen".

### 3. Real correlation + full feature listing (commit 150985c)
User asked "does it tell us how correlated? ... do it for every feature, list every feature" —
pointed out that RandomForest's feature_importances_ is NOT a correlation (no sign, measures
model reliance not directional relationship). Added:
- Real Spearman rank correlation (scipy.stats.spearmanr) for every continuous/ordinal/
  embedding feature against the actual training targets (ratio_to_median), with a plain
  "higher is better"/"lower is better"/"no variance / insufficient data" direction string.
  Spearman chosen for the same reason learn.mining already uses Mann-Whitney U (skewed ratio
  metric, real outliers, no normality assumption needed). Returns None, never a fabricated
  0.0, when there's no real variance to correlate.
- category_breakdown for categorical features (hook_type, format_tag) — median real target
  per distinct category, sorted — the categorical equivalent of a correlation sign.
- top_n changed from a hardcoded 10 to None (returns EVERY feature) by default.
- Honest finding, surfaced not glossed over: correlation magnitudes came back weak
  (|rho| ~0.01-0.22) even for RandomForest-important features — expected, since RF captures
  non-linear/interaction effects a single pairwise correlation can't see. This gap is now
  visible in the output instead of hidden.

## What's actually NOT done yet (don't assume otherwise)
- CROSS-VALIDATION — flagged repeatedly this session, still not built. Every R²/importance/
  correlation number produced so far (qualume's RF, okn_mirror's GB, the merged model) is
  IN-SAMPLE only. Three separate embedders (text/transcript/visual) each have their own
  documented fold-safety leakage risk (must fit fresh per fold, never on the full set then
  CV) — this is the natural, most load-bearing next step before trusting any of these numbers
  for a real decision.
- Niche+content_type-scoped clustering/mining was never wired into the live orchestrator
  (orchestrate.run_ingestion_cycle still calls run_mining_pass/run_cluster_pass/
  run_visual_cluster_pass unscoped, niche=None) — flagged as open since Nov 2-3, still open.
- Visual frame embeddings are now used in PREDICTION (this session's work) but still NOT fed
  into the main HDBSCAN clustering comparison loop against transcript clustering — that
  fusion question (raised back on 2026-10-31) remains explicitly deferred.
- Foundational docs rebuild (avatar/offer-brief/research), LangGraph agent skeleton (PM, Data
  Engineer, General Analyst, Greenbabe Analyst), roster skill update — all untouched, carried
  forward from earlier sessions.
- Instagram account-risk decision, owned-content publishing loop, separate test DB for
  pollution-proofing — all still open, unchanged.

## Immediate next action when picking this back up
1. Build the cross-validated version of the merged prediction model (fresh embedder fit per
   fold for all three embedding sources) — the single most load-bearing open item.
2. Decide whether to wire niche+content_type-scoped clustering into the live orchestrator now
   that the embedding work gives a second, independent signal to cross-check cluster labels
   against.
3. Annotation backlog is still the long-pole bottleneck for the Gemini-annotation-dependent
   features (video-content engineered features, hook_text/transcript axes) — only 1,553/17,186
   items annotated. The visual-embedding/okn_mirror paths don't need annotation at all and
   already cover more data; consider leaning on those more while annotation catches up.

## Earlier context (2026-10-31 mass ingestion — resolved, kept for history)
Mass TikTok ingestion (112→220+ accounts, 8,857+ videos) ran to completion since this note was
written; annotation backlog climbed from ~374 toward the 1,553 figure above via the Nov 3
Gemini-fallback-model throughput fix (commit f48444d). Content_type classifier (selling/
organic) shipped and wired into the cycle (commit 7fbd42e). Random forest first trained on
real data Nov 3 (commit 2e40411) — superseded by this session's merged model.
