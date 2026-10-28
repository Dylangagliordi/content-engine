# Qualume — Session Handoff (2026-10-28)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified, not assumed)
- 238/238 tests passing. Git clean.
- Watchlist: 2 TikTok accounts active (@pinkstork, @wishgardenherbs, both `watchlist_type=competitor`).
- 60 external_content_items ingested. ALL 60 now annotated at v6 (presenter/avatar + scene
  description) — backfill completed 2026-10-28, verified against real DB rows (not mocked).
  Two of the 60 initially had no downloaded video file (`local_video_path` NULL); ran
  `tiktok_reader.download_pending_videos()` to pull them, then annotated — both succeeded.
- Gemini's earlier persistent 503 ("high demand") outage on `gemini-3.1-flash-lite` has cleared.
  Confirmed via live calls, not assumption — note if it recurs, it's a known transient external
  issue, not a code bug (same pattern seen before).
- Copy-technique classifier v2 (ordered 22-flag technique sequence) built and live-verified
  2026-10-17/19, improved accuracy over v1 on the same real video.
- n8n (`qualume-n8n` container, OrbStack, localhost:5678, healthz OK) wired to a real local HTTP
  server wrapping the ingestion pipeline (`host.docker.internal`). Workflow (Schedule -> HTTP ->
  /run-ingestion-cycle -> poll -> download -> Gemini -> Claude -> mine) is IMPORTED into n8n's UI
  but NOT YET ACTIVATED — still a manual step for the user.
- Hermes runs on Claude Pro subscription auth (CLAUDE_CODE_OAUTH_TOKEN), not API credits. Qualume's
  own `.env` ANTHROPIC_API_KEY is separate and unaffected. Three distinct cost meters.

## What's actually NOT done yet (don't assume otherwise)
- n8n workflow exists but is inactive — nothing runs on a schedule yet, every run so far has been
  triggered manually from a terminal.
- Learn engine (Part 5: pattern mining, insights tables, feature discovery) — designed in
  `plans/learn-engine-idea-sourcing-architecture.md` but the actual mining/correlation step against
  the now-fully-v6-annotated 60-video dataset has NOT been run. This is the natural next real step:
  there's finally a complete, consistent dataset (presenter + scene + copy-technique + shot-level
  data, all at current schema versions) to mine something from.
- Instagram scraping account decision (greenbabe vs. secondary) — still unresolved, still blocking
  Part 3. Untouched for several sessions now.
- Zero owned content has ever been published (5 draft IG statics sit unpublished in
  data/assets/) — so insights_owned / real performance-based learning can't start until that
  manual publish step happens.

## Immediate next action (pick one, both are real unblocking moves)
1. Activate the n8n workflow so ingestion runs on a schedule without a terminal session driving it
   — this is a one click in the n8n UI at this point, everything behind it is already wired and
   tested live.
2. Start the actual Learn-engine mining pass (Part 5) against the 60 v6-annotated external videos
   — first real data run of the pattern-mining design that's existed on paper since 2026-10-12.
   Welch's t-test / Mann-Whitney U per-feature first, per the design doc — not random forest yet.

## Known gaps (not touched recently, still open)
- Instagram account-risk decision (Part 3) — unresolved.
- No owned content published — insights_owned pipeline has nothing real to work with yet.
- Separate test DB — tests still write into the real dev Postgres (Part 1 tech debt, flagged
  repeatedly, not yet fixed).
