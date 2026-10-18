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

The server needs to be alive whenever n8n's schedule fires. Options:
- **Simplest right now**: keep a terminal open running
  `uv run python -m qualume.server` (fine for testing, not for unattended
  use — dies if the terminal/Mac sleeps).
- **Real unattended setup (the N100/always-on-host plan)**: once the
  pipeline moves to an always-on host, run this as a proper background
  service (systemd unit, or Docker Compose `profiles` alongside Postgres/
  n8n) so it survives reboots and doesn't depend on a terminal staying
  open. Not built yet — flag it when you're ready to wire this up on the
  real hardware.

## What a run actually does, and what's isolated from what

Every stage inside `run_ingestion_cycle()` is wrapped so one stage's
total failure doesn't block the others, and one item's failure within a
stage doesn't block the rest of that stage:
- `poll` — sweeps all active watched accounts (same as running
  `poll_tiktok_watchlist.py` by hand)
- `download` — downloads any video files not yet on disk
- `annotation` — runs Gemini v5 annotation on any video with a file but
  no v5 `external_creative_dna` row yet
- `classification` — runs the copy-technique classifier on any video with
  a v5 annotation but no `external_copy_analysis` row yet

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
  v3→v4→v5 Gemini re-annotations earlier), not something this cycle does
  on its own.
