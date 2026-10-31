# Qualume — Session Handoff (2026-10-31, mass ingestion in flight)

## Start here next session
Read this file first, not chat history. Check `data/ingestion_progress.log` FIRST for the current
real numbers (appends one line every 60s: total/accounts/downloaded/annotated/classified) — it'll
tell you if the long-running ingestion job from this session already finished.

## Project state (verified live, not assumed)
- 312/312 tests passing. Git clean except the two files added at session end (committed below).
- launchd server (`com.qualume.orchestration-server`) is RUNNING and mid-cycle — do not kill it
  without checking progress first, it's doing real multi-hour work (see below).
- A background monitor (`scripts/monitor_ingestion.py`, PID visible via `ps aux | grep monitor_ingestion`)
  is logging DB counts to `data/ingestion_progress.log` every 60s. Still running when this session
  ended. Safe to leave running — it's read-only, no side effects, trivial CPU cost. Kill it with
  `pkill -f monitor_ingestion` once the ingestion job finishes if you don't want it anymore.

## What happened this session: real mass-scale ingestion, in progress
1. Picked up last session's uncommitted work: committed the 8-worker concurrency code for
   annotation/classification, raised per-account video cap 30->100, restarted the launchd server
   (it had been running stale pre-concurrency code).
2. Verified the ~98 TikTok handles researched by last session's subagents are REAL via a new
   `scripts/verify_candidate_handles.py` (yt-dlp metadata-only check, paced). 93/98 valid.
3. Added all 93 to the watchlist: **active accounts went from 20 -> 112**, covering fitness, food,
   beauty, finance, small business, and non-health DTC/TikTok-Shop brands on top of the original
   health/wellness/mom-tok niches.
4. Triggered one real `POST /run-ingestion-cycle` call against the restarted server. This single
   call is STILL RUNNING (stages run sequentially: poll -> download -> annotate -> classify -> mine
   -> cluster). Real numbers observed live during the session:
   - Videos ingested: 413 -> 8,857 (blew past the 5000-video goal on the download/poll stage alone)
   - Accounts with content: 13 -> 93
   - Downloaded: 8,833/8,857 (effectively done; 24 failing on dead/private/no-video-format TikTok
     links — expected, isolated per-item failures, not a pipeline bug)
   - Annotation (the real current bottleneck): climbing steadily but slowly — was at ~374/8,857
     when the session ended, with 8,585 still pending. At the observed pace this is realistically a
     multi-hour tail, not minutes.
   - Classification: stuck at 96 — it's the LAST stage before mining/clustering within this one
     sequential cycle call, so it won't move until the annotation backlog (8,585 videos) is cleared.
   - Mining/clustering have NOT run yet against the new data — they're stages 5-6, still blocked
     behind annotation+classification finishing.
5. User explicitly chose to let this run fully unattended — do not feel obligated to babysit it.

## What's actually NOT done yet (don't assume otherwise)
- Mining/clustering have not touched the new 93-account, fitness/food/beauty/finance/DTC-expanded
  dataset at all yet. 0/72 findings survived BH correction at last measurement (95 videos, 2
  accounts) — re-run and look at REAL numbers once annotation+classification finish and mining
  actually executes against the new breadth. Don't assume more data automatically means findings.
- Some niches (fitness, food, beauty, finance, business, DTC brand) are completely new and have
  never been through classification or mining even once — first real signal on whether these niches
  behave similarly to the original health/supplements content, or differently, is still unknown.
- 24 TikTok download failures (dead/private accounts, no video formats) are logged in
  `logs/orchestration-server.error.log` — not urgent, but worth a quick look to see if any are
  salvageable (e.g. wrong handle) vs. genuinely dead.
- Cross-account/cross-niche replication check, shot_breakdown correlation mining, owned-content
  publishing loop, Instagram account-risk decision, separate test DB — all still open from before,
  unchanged this session.

## Immediate next action when picking this back up
1. Check `data/ingestion_progress.log` tail — did annotation/classification/mining/clustering
   actually finish? If yes, look at real mining/clustering output for the first time against the
   new 112-account, multi-niche dataset.
2. If still running, decide whether to just let it keep going, or whether 8 concurrent Gemini/Claude
   workers should be tuned up/down based on real observed throughput this session provided.
3. Check the 24 download failures — any genuinely fixable (typo'd handle, etc.)?
