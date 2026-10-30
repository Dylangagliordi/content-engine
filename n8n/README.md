# Qualume n8n integration

## What this is

One workflow (`qualume_ingestion_cycle.json`) that runs the whole pipeline
on a schedule: poll watchlist -> download pending videos -> annotate
pending (Gemini v5) -> classify copy for newly-annotated videos — via
`src/qualume/orchestrate.run_ingestion_cycle()`.

n8n itself only has 2 nodes: a Schedule Trigger, and one HTTP Request node
that calls a local server (`src/qualume/server.py`) wrapping that one
Python function. n8n does NOT reimplement any pipeline logic — it's purely
the scheduler/trigger, which is the right division of labor (Python/
Postgres stays the source of truth and does all real work; n8n just calls
it on a cadence instead of you running a script by hand).

## Why a local HTTP server instead of an n8n Execute Command / Code node

n8n's own Code node runs inside its Docker container, which doesn't have
this project's Python venv, Postgres driver, or `.env` — it would need a
separate, parallel Python environment INSIDE the n8n container, duplicating
real setup. A tiny local server that n8n reaches via Docker's
`host.docker.internal` (confirmed live, 2026-10-17 — `docker exec
qualume-n8n wget -qO- http://host.docker.internal:8420/health` returns
`{"status": "ok"}` from inside the real `qualume-n8n` container, no extra
Docker networking config needed) keeps the pipeline running in the ONE
already-correct environment.

## Setup (one-time)

1. Start the orchestration server (keep it running — see "Running it for
   real" below for how to keep it alive long-term):
   ```bash
   cd ~/qualume && uv run python -m qualume.server
   ```
   Confirms it's up: `curl http://localhost:8420/health` → `{"status": "ok"}`

2. Open n8n: http://localhost:5678 (log in / set up your n8n account if
   this is the first time — n8n requires its own login, separate from
   anything in this project; I can't do this step for you).

3. Import the workflow: n8n UI → Workflows → **Import from File** →
   select `~/qualume/n8n/qualume_ingestion_cycle.json`.

4. Open the imported workflow, check the Schedule Trigger node's interval
   (defaults to every 6 hours — change it if you want a different
   cadence), then toggle the workflow **Active** in the top-right.

That's it — n8n will now call `POST /run-ingestion-cycle` on that
schedule, and you'll see each run (success/failure, how long it took,
the full JSON result) in n8n's Executions tab.

## Running the server for real (not just a one-off terminal)

Set up as a launchd user service (2026-10-30) — survives logout, Mac
restart, AND a crash (auto-restarts), without needing a terminal open:

- Plist: `~/Library/LaunchAgents/com.qualume.orchestration-server.plist`
  (`RunAtLoad=true` + `KeepAlive=true` — starts on login, restarts on any
  exit including a crash, confirmed live by `kill -9`'ing the real PID
  and watching launchd relaunch it within 3s)
- Logs: `~/qualume/logs/orchestration-server.log` /
  `orchestration-server.error.log`
- Manage it:
  ```bash
  launchctl list | grep qualume                                          # check it's running
  launchctl unload ~/Library/LaunchAgents/com.qualume.orchestration-server.plist   # stop
  launchctl load ~/Library/LaunchAgents/com.qualume.orchestration-server.plist     # start
  tail -f ~/qualume/logs/orchestration-server.log                        # watch logs live
  ```
- **This only survives while THIS Mac is on.** The Mac sleeping still
  pauses it (launchd doesn't wake the machine) — the real fix for that
  remains the N100/always-on-host migration (Part 9), unchanged. This
  closes the "dies if the terminal closes" gap specifically, not the
  "Mac has to stay powered on" one.

## What a run actually does, and what's isolated from what

Every stage inside `run_ingestion_cycle()` is wrapped so one stage's
total failure doesn't block the others, and one item's failure within a
stage doesn't block the rest of that stage:
- `poll` — sweeps all active watched accounts (same as running
  `poll_tiktok_watchlist.py` by hand)
- `download` — downloads any video files not yet on disk
- `annotation` — runs Gemini annotation (current `ANALYSIS_PROMPT_VERSION`
  from `gemini_watcher.py`, v6 as of this writing — read live, not
  hardcoded; see "Real limits" below for why that matters) on any video
  with a file but no current-version `external_creative_dna` row yet
- `classification` — runs the copy-technique classifier on any video with
  a current-version annotation but no `external_copy_analysis` row yet
- `mining` (added 2026-10-30) — re-runs Part 5b's statistical pass
  (`learn.run_mining_pass.run_mining_pass`) over the WHOLE dataset every
  cycle: every mineable `external_creative_dna` column + every one of the
  22 copy-technique flags, judged against each account's own performance
  median. Findings clearing p<0.05 get upserted into `insights_external`
  (keyed on stable identity, so a repeat finding refreshes one row rather
  than piling up duplicates every 6 hours); findings that stop clearing
  significance as more data comes in get marked `tier='retired'`
  (proposed-tier rows only — never touches `accepted`/`testing` rows,
  which have real owned-evidence backing beyond just this pass).

Each stage's result (or `{"error": ...}` if the WHOLE stage failed
outright) comes back in the HTTP response and is visible in n8n's
Executions tab — that's your real visibility into what happened each
cycle, no separate logging needed.

## Real limits worth knowing

- **No auth on the server.** It binds to `127.0.0.1` only by default
  (see `DEFAULT_HOST` in `server.py`) — don't change that without adding
  real auth, since a POST to this endpoint triggers real Gemini/Claude
  API spend and real DB writes.
- **One cycle at a time, no overlap protection yet.** If a cycle is still
  running when the next schedule fires, n8n will fire a second HTTP
  request — the server doesn't currently queue or reject concurrent
  requests. Not a real risk at today's volume (annotation/classification
  stages are sequential and fast per-item), but worth building a real
  lock if cycles start taking longer than the schedule interval.
- **Re-classification on a new copy_classifier prompt version is NOT
  automatic.** `find_pending_classification` only looks for items with
  NO analysis row at all — bumping `ANALYSIS_PROMPT_VERSION` and
  re-running everyone is a deliberate, separate batch job (same as the
  v1→v6 Gemini re-annotations earlier), not something this cycle does
  on its own.
- **Fixed 2026-10-30, worth knowing it existed:** `find_pending_annotation`
  / `find_pending_classification` used to hardcode the literal string
  `'v5'` instead of reading the live `ANALYSIS_PROMPT_VERSION` constant.
  After the v6 schema bump, every newly-annotated video silently never
  got picked up for classification — no error, just permanently missing
  data on every scheduled run. Both functions now import the live
  constant, so this specific failure mode can't recur at the next version
  bump without a code change here too.
- **Fixed 2026-10-30, worth knowing it existed:** `learn.mining`,
  `learn.copy_mining`, and `learn.replication` all joined against
  `external_performance_snapshots` without filtering to each video's
  LATEST snapshot — a video re-polled 3 times (the real shape scheduled
  polling produces on every refresh) got pseudo-replicated 3x in every
  statistical test, artificially inflating sample sizes and shrinking
  every p-value (a real production run showed 22/49 "significant"
  findings before the fix; 7/46 after — the former was mostly noise from
  this bug, not real signal). All three modules now join against only
  each video's latest snapshot by timestamp.
