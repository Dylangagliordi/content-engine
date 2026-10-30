# Qualume — Session Handoff (2026-10-30)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified, not assumed)
- 255/255 tests passing. Git clean.
- Watchlist: 2 TikTok accounts active (@pinkstork, @wishgardenherbs). A leaked test fixture
  account (test_orch_ad047945) was found active in the REAL watchlist and getting polled on
  real scheduled cycles — found via the live server's error log, deleted, confirmed clean.
- 95 external_content_items, all 95 fully annotated (v6 presenter/scene) and copy-classified
  (22-flag taxonomy). n8n's scheduled polling found 35 new videos beyond the original 60 this
  session — confirms the pipeline is actually discovering new content, not just idle.
- n8n (`qualume-n8n` container, localhost:5678) workflow is PUBLISHED/ACTIVE — real scheduled
  runs every 6 hours, confirmed live (manually triggered one, watched it actually poll/download/
  annotate/classify real new content end to end).
- Orchestration server (`qualume.server`) now runs as a real launchd user service
  (`~/Library/LaunchAgents/com.qualume.orchestration-server.plist`), not a manually-started
  terminal process. RunAtLoad + KeepAlive — survives logout and Mac restart, auto-restarts on
  crash (verified live: kill -9'd the real PID, launchd relaunched within 3s). Does NOT survive
  the Mac sleeping/being off — that's still the N100/always-on-host migration, unchanged.
- Part 5b (pattern mining) is now AUTOMATED, not a manual script: `learn.run_mining_pass.
  run_mining_pass()` runs as run_ingestion_cycle's 5th stage on every 6h cycle. Tests every
  mineable DNA column + all 22 copy-technique flags, upserts p<0.05 findings into
  insights_external (keyed on stable identity so repeat runs refresh in place, not duplicate),
  retires proposed-tier rows that stop clearing significance. Currently 7 real findings
  (n=95, properly deduped) — see below for what they say.

## Two real bugs found and fixed this session (both via live verification, not assumed)
1. **orchestrate.py hardcoded 'v5'** in both find_pending_annotation and
   find_pending_classification's SQL, left over from before the v6 schema bump. Every newly-
   annotated-at-v6 video silently never got picked up for classification — no error, just
   permanently missing data. Fixed: both now import the live ANALYSIS_PROMPT_VERSION constant.
2. **Pseudo-replication in all 3 mining modules** (learn.mining, learn.copy_mining,
   learn.replication) — none of them filtered external_performance_snapshots to each video's
   LATEST row. Once scheduled polling started re-polling existing videos (2-3 snapshot rows per
   video), every statistical test counted the same video 2-3x as independent observations,
   artificially shrinking every p-value. A real run before the fix showed 22/49 "significant"
   findings; after the fix, 7/46 on the identical dataset — the former was mostly this bug.
   Fixed: all 3 modules now join against only each video's latest snapshot by timestamp.

## Current real insights_external findings (tier=proposed, n=95, all source=internal_mining)
1. format_tag='talking_head' outperforms (1.35x vs 0.94x baseline, p=0.0036)
2. copy technique hook_promise_without_mechanism outperforms (p=0.0069)
3. format_tag='voiceover_broll' underperforms (0.73x vs baseline, p=0.0109)
4. copy technique hook_pain_callout underperforms (p=0.0112)
5. caption_style='none' underperforms (0.33x vs baseline, p=0.0195)
6. copy technique creates_urgency outperforms (1.91x, p=0.0440)
7. hook_type='question' outperforms (1.63x, p=0.0490)
All proposed-tier, n=95, correlational not causal — low confidence by design until owned
evidence (Part 5c/experiments loop) tests them against OUR OWN account, which requires
publishing owned content first (still zero published).

## What's actually NOT done yet (don't assume otherwise)
- No owned content has ever been published (5 draft IG statics sit unpublished in
  data/assets/) — insights_owned / real performance-based learning can't start until that
  manual publish step happens. The 7 insights_external findings above are external-only,
  capped at lower trust by design until tested against owned data.
- Part 5c (playbook authoring from accepted-tier insights) not started — nothing is
  accepted-tier yet, nothing has graduated past proposed.
- Instagram scraping account decision (Part 3) — still unresolved, untouched for weeks.
- Separate test DB — tests still write into the real dev Postgres (flagged repeatedly,
  not yet fixed). This session specifically found a real leaked test row in production data
  from this exact gap — worth prioritizing before it recurs.

## Immediate next action (pick one)
1. Let the automated pipeline run for a while (days) and watch the 7 insights stabilize or
   shift as more data comes in via scheduled polling — nothing urgent to build, just observe.
2. Publish one of the 5 draft IG statics to start the insights_owned / real validation loop —
   the actual mechanism (learn.validation.seed_owned_insight / record_owned_evidence) is already
   built and tested, just waiting on real owned performance data to mean anything.
3. Finally build the separate test DB (Part 1 tech debt) — this session's leaked-fixture-in-
   production incident is a concrete argument for prioritizing it now rather than later.
