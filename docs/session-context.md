# Qualume — Session Handoff (2026-10-31, watchlist expansion + concurrency)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified live against real data, not assumed)
- 312/312 tests passing. Git clean.
- n8n (6-stage cycle, launchd-managed server) runs every 6 hours: poll -> download -> annotate ->
  classify -> mine (Phase A, declared taxonomy) -> cluster (Phase B, emergent/HDBSCAN).
- launchd server RESTARTED this session (new PID) to pick up new concurrency code — was running
  stale code since last session's edits were never deployed.

## Real work completed this session (picking up from 2026-10-30's in-progress state)
1. Committed the controlled-concurrency code that was left uncommitted last session:
   `_run_stage_over_items(max_workers=...)` in orchestrate.py — Gemini annotation and Claude
   classification now run 8-way parallel (I/O-bound network calls, no TikTok-facing risk).
   poll/download stay strictly sequential (real TikTok ban/rate-limit risk). 5 TDD tests, all pass.
2. Raised `DEFAULT_MAX_VIDEOS_PER_ACCOUNT` from 30 to 100 (user's explicit choice) — needed for the
   5000-video volume goal; 30 x ~115 accounts only ceilings at ~3450.
3. Built `scripts/verify_candidate_handles.py` — verifies candidate TikTok handles are real via
   yt-dlp metadata-only extraction (playlistend=1, paced 1.5s between checks) BEFORE adding to the
   watchlist. Ran it against the 93-98 handles 5 parallel subagents researched last session (fitness,
   food, beauty, finance/business, DTC brand non-health). Result: 93/98 valid, 5 rejected (private
   account, zero videos, or unresolvable TikTok user ID) — real verification caught real dead
   handles, not assumed valid.
4. Added all 93 verified handles to the watchlist via `manage_watchlist add`. **Watchlist grew from
   20 -> 112 active accounts**, now spanning: original health/supplements/wellness/mom-tok niches
   PLUS fitness, food/cooking, beauty/skincare, personal finance, small business, and non-health
   DTC/TikTok-Shop brands. Breadth-first per user's "all sorts of niches" request.
5. Kicked off a real ingestion cycle via `POST /run-ingestion-cycle` against the restarted server.
   Confirmed real progress during the session: 413 -> 553 -> 736+ videos, growing steadily as it
   worked through the new 112-account watchlist with the higher per-account cap. This was still
   running when the session ended — it's a long sequential poll/download pass (TikTok-facing, by
   design not parallelized), expect it to keep climbing toward the 5000 goal over hours, not
   minutes. Check `external_content_items` count to see where it landed.

## What's actually NOT done yet (don't assume otherwise)
- The 5000-video goal is NOT yet reached — was at ~736 and climbing when this session ended. The
  n8n 6-hour cycle + this session's long-running ingestion pass will keep adding more over time with
  zero further manual intervention needed.
- Mining/clustering have NOT been re-run against the expanded dataset yet this session (the
  ingestion cycle call includes mining/clustering as stages 5-6, so it may happen automatically once
  that cycle finishes — but no one has looked at fresh numbers yet). Check insights tables once
  ingestion settles.
- 0 mining findings survived BH correction as of last session's end (at 95 videos, 72 tests) — with
  the big breadth increase (112 accounts, new niches), re-run the real numbers once there's enough
  volume per new niche/segment to even be testable (min_group_size=5 threshold).
- Cross-account/cross-niche replication auto-check (learn.replication.check_segment_replication,
  built, tested) still not wired into either automated pipeline.
- No owned content published — insights_owned pipeline (seed_owned_insight/record_owned_evidence)
  has nothing real to validate against yet.
- shot_breakdown's rich per-shot data still not mined directly (same gap as before — needs Spearman/
  Pearson correlation test, not built yet).
- Instagram account-risk decision (Part 3) — still unresolved, untouched for weeks.
- Separate test DB (Part 1 tech debt) — still not built; 4+ real fixture-leak incidents logged across
  sessions, still the strongest outstanding case for prioritizing this.

## Immediate next action (pick one)
1. Check on the long-running ingestion pass from this session — how far did it get toward 5000
   videos, did mining/clustering re-run cleanly against the new niche breadth, any errors in new
   accounts (e.g. brand accounts with few/no transcript-bearing videos skewing coverage)?
2. Re-run mining pass by hand once there's real new-niche volume and look at actual BH-corrected
   findings count vs. the 0/72 baseline from last session.
3. Wire shot_breakdown's continuous features into a numeric-correlation mining pass (Spearman/Pearson
   — a genuinely different statistical test than the Mann-Whitney used everywhere else).
4. Wire learn.replication's cross-account check into both pipelines as an additional filter.
5. Finally build the separate test DB (Part 1 tech debt).
6. Publish one of the 5 draft IG statics to finally start the insights_owned validation loop.
