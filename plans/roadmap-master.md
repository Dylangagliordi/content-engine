# Qualume — Master Roadmap (granular, living document)

> Supersedes the phase ordering in `v2-engineering-plan.md` §6 and
> `execution-plan-phase-0-1.md` where they conflict with what actually got built and
> reprioritized. Those docs are still correct on architecture/principles — this doc is the
> up-to-date, granular, continuously-updated task list and status tracker. Update this file
> whenever a task completes or a decision gets made; don't let it go stale.

**Last updated:** 2026-10-10. **Test suite status:** 33/33 passing. **Reality check:** fully
greenfield — zero owned posts published, zero owned performance data, TikTok reading engine
working live, Instagram reading blocked on a pending decision, nothing running on a schedule yet,
everything currently hosted on a Mac via Docker (OrbStack), not yet moved to dedicated hardware.

---

## Status legend
- ✅ Done & verified (tests passing, and/or live-verified against real data)
- 🚧 In progress / partially built
- ⛔ Blocked (on a decision, external dependency, or account access)
- ⬜ Not started
- 🔁 Recurring/ongoing (not a one-time task — maintained continuously)

---

## Locked decisions (don't re-litigate these without a reason)

- **LLM provider:** Anthropic Claude only. Tiers: `claude-haiku-4-5` (low), `claude-sonnet-4-5`
  (medium), `claude-opus-4-5` (high) — verified live against the real API 2026-10-10. Re-verify
  periodically; Anthropic's lineup moves.
- **Infra:** Docker Compose (Postgres 17 + n8n), currently on this Mac via OrbStack. Designed to be
  fully portable — no Mac-specific assumptions — so migration to other hardware is just copying the
  project folder and re-running `docker compose up`.
- **Hosting target:** Raspberry Pi 5 (8GB), favored over cloud VPS for two reasons: (1) cost — ~$110
  one-time + ~$1/mo electricity vs. ~$5-6/mo forever, breakeven ~18-20 months; (2) **residential IP
  is less likely to be detected/blocked by TikTok/Instagram anti-bot systems** than a datacenter/VPS
  IP range, which matters more for this specific workload than the dollar difference. **Not yet
  purchased** — still running on the Mac.
- **Pilot content workflow:** IG statics (simplest Design engine — no video assembly). TikTok/Reels
  video workflow is future work, not started.
- **Business account:** "greenbabe" (Instagram). Zero posts published as of this writing.
- **Priority reordering (2026-10-10):** user explicitly redirected focus to the READ engine before
  continuing the WRITE/DESIGN loop — "everything relies on that." This roadmap reflects that
  reordering: Read engine work is pulled forward, ahead of finishing Phase 1's publish step.

## Open decisions (blocking specific future tasks — resolve before starting them)

1. **Instagram scraping account.** IG requires a logged-in session to read ANY content (confirmed
   live — even public posts return "not available" without login). Using greenbabe directly risks
   the real business account getting flagged/restricted. Options: (a) separate/secondary IG account
   for scraping only, (b) use greenbabe and accept the risk, (c) skip IG reading entirely for now.
   **Not yet decided.** Blocks: all Instagram-reading tasks below.
2. **Raspberry Pi purchase timing.** Not yet bought. Blocks: 24/7 continuous operation, residential-IP
   benefit, true "runs without my computer open" capability. Does NOT block continued development —
   everything is being built portable, Mac-hosted for now.
3. **Scheduling mechanism choice.** n8n (already running in Docker, has a built-in scheduler/cron
   node) vs. plain Linux cron vs. systemd timers. Leaning n8n since it's already deployed and gives a
   UI for monitoring runs, but not yet decided or wired up.

---

## PART 1 — Foundation (Phase 0) — ✅ DONE

- ✅ Docker (OrbStack) installed, verified
- ✅ Project scaffolded (`~/qualume/`, uv-managed Python project)
- ✅ Core deps installed: langgraph, langchain-anthropic, pydantic, psycopg2-binary, pillow, pytest,
  yt-dlp
- ✅ `.env` + real Anthropic API key, verified live
- ✅ Postgres + n8n running via docker-compose
- ✅ Core schema migrated (8 tables: content_items, creative_dna, performance_snapshots,
  ad_performance, experiments, insights, playbooks, eval_runs)
- ✅ Pydantic contracts for Write/Design pipeline (`models/contracts.py`) — TDD'd
- ✅ Model-tiering helper (`llm/pick_llm.py`) — TDD'd, live-verified
- ✅ Eval harness skeleton (deterministic constraint checks; LLM-judge layer not yet added)
- ✅ v0 `ig_static` playbook seeded (both DB row and Hermes skill `ig-static-playbook`), explicitly
  tagged `source: external_field_notes, tier: proposed` — honest about being unvalidated
- ✅ DB connection helper (`db/connection.py`), config loader (`config.py`), playbook loader
  (`db/playbook_loader.py`)

**Remaining Phase 0 debt (not blocking, but real):**
- ⬜ Separate test DB/schema — tests currently write fixture rows into the real dev Postgres
  instance; cleaned up manually each time, which is error-prone (already caused two stray-commit
  incidents). Fix before this becomes a bigger problem.
- ⬜ LLM-judge layer for the eval harness (currently only deterministic must_include/must_avoid
  string checks — no actual quality judgment yet).

---

## PART 2 — Reading Engine: TikTok — ✅ MOSTLY DONE, 🔁 needs scheduling

**Goal:** continuously discover and record data from TikTok videos/accounts you choose, with zero
login required, deduped so re-polling never creates duplicate work.

- ✅ Schema: `watched_accounts`, `external_content_items`, `external_performance_snapshots` —
  structurally separate from owned content (different trust level, per v2 plan §2.1)
- ✅ `db/watchlist.py` — add/list/deactivate watched accounts (3 tests)
- ✅ `ingest/tiktok_reader.py` — fetch via yt-dlp (no login), parse/normalize, dedup-insert (5 tests,
  live-verified: polled a real account twice, second run correctly found 0 new / 5 duplicates)
- ✅ `poll_tiktok_watchlist.py` — sweeps all active watched accounts, isolates per-account failures so
  one blocked/broken account doesn't kill the whole run, polite delay between accounts (2 tests)
- ✅ `manage_watchlist.py` — CLI for add/list/deactivate, live-verified
- 🔁 **Populate the real watchlist** — currently empty (test account added and removed). You add real
  TikTok accounts/competitors via `uv run python -m qualume.manage_watchlist add tiktok <handle>`.
  This is ongoing, not a one-time task — add/prune as your competitive set evolves.
- ⬜ **Scheduling** — poller currently only runs when manually invoked. Needs to run automatically on
  a cadence (every N hours). Blocked on open decision #3 above (n8n vs cron vs systemd).
- ⬜ **Hashtag/trend-page support** — currently only supports per-account polling. `yt-dlp` also
  supports hashtag pages the same way (confirmed in research) — extending `fetch_account_videos`-style
  logic to hashtags is a small, well-understood addition once accounts are proven out at volume.
- ⬜ **Rate-limit/backoff handling** — one test video hit a transient IP-block error during live
  testing. Currently no retry/backoff logic — a blocked request just fails and is logged (isolated per
  Task 2's poller design), but doesn't retry later. Add exponential backoff + a "temporarily skip this
  account for N hours" cooldown if blocks become frequent.
- ⬜ **Monitoring/alerting on poller health** — per v2 plan §5.5 ("alerting on pipeline health is a
  separate concern from alerting on content performance") — right now failures just print to stdout.
  Needs a persistent log/alert (e.g. a Hermes cron digest, or write failures to a dedicated table) so
  a silently-broken poller doesn't go unnoticed for days.

---

## PART 3 — Reading Engine: Instagram — ⛔ BLOCKED on account decision

**Goal:** same capability as TikTok, but IG requires a logged-in session for ANY content view
(confirmed live, not assumed) — this entire part is gated on open decision #1 above.

- ⛔ Decide which account takes the scraping risk (separate account / greenbabe directly / skip)
- ⬜ Once decided: obtain and securely store session cookies (never commit to git; `.env` or a
  dedicated secrets file, gitignored)
- ⬜ `ingest/instagram_reader.py` — same shape as the TikTok reader (fetch/parse/dedup), but using
  authenticated yt-dlp calls (`--cookies` flag) or a logged-in browser session as fallback
- ⬜ Separate, explicit rate-limiting — IG's detection is more aggressive than TikTok's per research;
  cadence should be conservative by default (e.g. a handful of accounts polled a few times a day, not
  hourly) with an easy dial to adjust
- ⬜ Schema: likely reuse `watched_accounts`/`external_content_items`/`external_performance_snapshots`
  with `platform='instagram'` rather than new tables — IG and TikTok data shapes are similar enough
- ⬜ Explicit account-health monitoring — if the scraping account itself gets restricted, you want to
  know immediately (e.g. a canary check that confirms the account can still browse normally), not
  discover it days later when the whole pipeline's gone quiet

---

## PART 4 — Write + Design Engine: IG Statics — 🚧 MOSTLY DONE, publish step paused

**Goal:** generate and render IG static post content from a brief, gated by an eval check, ready for
(manual, for now) publishing.

- ✅ Write engine (`graphs/write_ig_static.py`) — LangGraph stateful graph: load_playbook → ideate →
  generate variants → self-critique. Model-tiered. Live-verified with real Claude output.
- ✅ Design engine (`graphs/design_ig_static.py`) — Pillow template rendering, deterministic,
  render-determinism tested, visually verified.
- ✅ Batch runner (`run_batch.py`) — Write → eval gate → Design → persist, with a real bug caught and
  fixed mid-build (eval gate's keyword-matching logic was silently borrowing an unrelated golden
  example's required terms — fixed with a stopword-based per-brief keyword extractor + regression
  tests).
- ✅ Manual performance-entry CLI (`ingest/manual_entry.py`) — append-only, tested.
- ⛔ **Manual publish (Task 1.4)** — 5 real generated images exist in `data/assets/`, status='draft'
  in Postgres. Paused: you haven't yet decided which image to post, and haven't handed over posting
  access/intent. This is intentionally a human decision point, not a blocker to fix in code.
- ⬜ **Instagram Graph API setup for OWN account** (Task 1.6, distinct from Part 3's scraping-other-
  accounts concern) — needed eventually for automated publishing and for pulling official analytics on
  greenbabe's own posts (this is the legitimate, ToS-compliant, official API path — very different
  from Part 3's scraping approach). Not started. No urgency until Task 1.4 produces a real published
  post worth tracking.
- ⬜ **Real design templates / brand kit** — current template is a single plain cream square generated
  programmatically, no real branding. Fine for proving the pipeline, not fine for actual greenbabe
  content. Needs real brand assets (colors, fonts, logo) once you're ready to publish for real.

---

## PART 5 — Learn Engine — ⬜ NOT STARTED (the actual "learning" part)

**Goal:** turn raw ingested content (both owned, once published, and external/competitor) into
structured, versioned, statistically-validated insights that actually improve future generations.
This is the part that makes "24/7 learning" literally true rather than just "24/7 data collection."

**Full concrete design (schema, video-vibe-capture research, idea-sourcing mechanics):**
`plans/learn-engine-idea-sourcing-architecture.md` — written 2026-10-12, read it before starting
any 5a/5b/5c work below. It answers the question "when Write fires, where does the idea actually
come from" with two sourcing modes (inspired-by-a-specific-video, and from-the-aggregate-summary),
and specifies the `external_creative_dna` table, multi-watchlist support (`watchlist_type`), and
the `insights.applies_to_segment` retrieval support that 5a-5c below assume exist.

### 5a. Multimodal annotation (turns raw video/image into structured CreativeDNA)
- ⬜ Annotation pipeline using a vision-capable Claude call: watch/view each `external_content_items`
  row's video (or still frames) and `content_items` row's rendered image, extract hook_type,
  hook_text, visual_hook_description, format_tag, script_structure_tags, cta_type, audio info,
  social_proof_used — populate `creative_dna` (owned) and a new `external_creative_dna` table
  (external) with the same shape, kept in separate tables per the owned/external trust split.
- ⬜ **Vibe/tone capture, not just literal content** (see architecture doc for full research): don't
  rely on frames+transcript prose alone. Compute pacing (`PySceneDetect`, cuts-per-minute) and
  music mood (`beatlyze-analyzer`, BPM/energy/valence from the audio waveform) deterministically
  rather than having the LLM guess; use anchored numeric scales (not "rate 1-10" unanchored) for
  the model-observed fields (`energy_level`, `tone_category`). Real precedent exists for this
  shape: arXiv 2512.21402 builds essentially this pipeline. Worth a real side-by-side test of
  Gemini's native video+audio understanding vs. the Claude frame-extraction workaround specifically
  for this stage before committing either way — Gemini's architecture fuses audio+visual natively,
  which the frame workaround can't fully replicate.
- ⬜ Schema-validated structured output (reject/retry on malformed annotation output, per v2 plan's
  "fail loud" principle) — don't silently accept free-text junk into a structured field.
- ⬜ Golden eval set for annotation quality — a small hand-labeled set of known videos with "ground
  truth" tags, used to catch regressions when the annotation prompt changes.
- ⬜ Cost/sampling strategy — annotating every single external video at volume isn't free; decide
  early whether to annotate everything or sample (e.g. annotate top-N by views/engagement, skip the
  long tail) once real volume is flowing.

### 5b. Pattern mining (statistical, not vibes)
- ⬜ Correlation analysis: `CreativeDNA` fields vs. performance metrics, segmented by workflow_type
  and by owned/external lane — produces candidate insight statements with effect size, sample size,
  and a significance measure attached (not just "this looks good").
- ⬜ False-positive guard: a test using shuffled/randomized labels on synthetic data to confirm the
  miner doesn't "find patterns" in noise — a correctness check on the miner itself, done once before
  trusting any of its real output.
- ⬜ Insight tiering lifecycle: `proposed` → `testing` → `accepted` → `retired`, each transition
  logged (the `insights` table already has a `state_history` JSONB column for this, unused so far).
- ⬜ Minimum sample size gating — don't let a pattern seen in 3 data points masquerade as validated;
  define and enforce a real threshold before promotion. The architecture doc sets a floor of ≥20
  analyzed videos in scope before idea-sourcing Mode A (inspired-by-a-specific-video) activates.
- ⬜ **Writes into `insights` with segment tags, queried live at generation time** (new
  `applies_to_segment` column + feature registry, see architecture doc) — replaces an earlier
  draft of this design that used a static periodically-regenerated summary doc; a live filtered
  query per-brief is more useful than one flat document, see the doc's "Retrieval, not a static
  summary doc" section.

### 5c. Playbook authoring (closes the loop back into generation)
- ⬜ Compile `accepted`-tier insights into a structured diff against the current playbook version for
  a workflow_type.
- ⬜ Human-readable diff surfaced to you (e.g. via a Hermes cron digest) for review/approval before
  anything auto-patches.
- ⬜ On approval: `skill_manage(action='patch')` against the relevant Hermes playbook skill AND a new
  versioned row in the `playbooks` table — keep the Hermes skill and the DB in sync, always.
- ⬜ Canary/shadow rollout for new playbook versions — run old vs. new on a small slice before full
  cutover, rather than a hard switch (v2 plan §4.4).

**Minimum viable data before 5b/5c are worth running:** you need SOME real performance history —
either greenbabe's own published posts (Part 4's publish step) accumulating snapshots over days/weeks,
or a meaningful volume of annotated external content with known engagement numbers. Right now there's
neither at volume. Part 5 work can be built and unit-tested against synthetic data before that, but
shouldn't be trusted against real output until real data exists.

---

## PART 6 — Synthetic Audience Scoring — ⬜ NOT STARTED

**Goal:** pre-flight score draft content against simulated persona reviewers before it's ever
published or spent on, per the v2 plan's cross-cutting pre-flight stage.

- ⬜ Define initial ICP personas (can start from assumptions/market research, refine once real
  customer data exists)
- ⬜ Build the scoring graph: feed a `ScriptVariant` + rendered asset to a panel of persona prompts,
  get predicted reception
- ⬜ **Calibration test** — back-test persona predictions against actual outcomes once real
  performance data exists; track prediction accuracy as a monitored metric, re-calibrate personas if
  accuracy drifts. Don't trust this blindly — it needs ongoing validation, not one-time setup.

---

## PART 7 — Explore/Exploit Formalization — ⬜ NOT STARTED

**Goal:** make the batch-generation explore/exploit split (already a parameter in `CampaignBrief`,
currently a fixed default, not actually driven by any learned policy) a real, data-informed decision.

- ⬜ Replace the fixed 70/30 default with a policy informed by accumulated insight confidence —
  e.g. explore more aggressively early (nothing proven yet) and shift toward exploit as the playbook
  accumulates `accepted`-tier insights.
- ⬜ Track outcomes per explore/exploit bucket explicitly (the `ScriptVariant` contract already tags
  this) so you can later measure whether explore-bucket content is actually surfacing useful new
  patterns or just wasting generation budget.

---

## PART 8 — Second Content Workflow: TikTok/Reels Video — ⬜ NOT STARTED

**Goal:** apply the same Write→Design→eval→publish loop to short-form video, reusing all shared
substrate (contracts, eval harness pattern, playbook/insight tables) but building a video-specific
Design engine.

- ⬜ Video-specific `CreativeDNA` fields matter differently than statics — pacing/cuts, audio
  selection, visual pattern-interrupt placement in the first 1-3 seconds (repeatedly flagged in field
  notes as the single most important variable for short-form video specifically)
- ⬜ Design engine video lane: template/avatar/UGC-source selection, caption/on-screen-text overlay,
  audio/track selection, assembly/render — meaningfully more complex than the static Pillow-rendering
  approach; likely needs either a proper video-editing library (moviepy, ffmpeg-python) or an
  external video-gen/avatar service
- ⬜ Visual-hook-presence test (v2 plan §2.4) — automated check that every rendered video actually has
  *something* in frame in the first 1-3 seconds, not a blank/logo card — catches the most-cited
  failure mode mechanically rather than relying on manual review
- ⬜ Render-determinism test adapted for video (same input/seed → same output) — video rendering has
  more non-determinism sources (codec settings, frame timing) than static image rendering, worth
  explicit attention
- ⬜ A seeded v0 playbook for this workflow too, same honesty discipline (`tier: proposed,
  source: external_field_notes`) until real data validates anything

---

## PART 9 — Infrastructure Maturity — ⬜ MOSTLY NOT STARTED

- ⬜ **Raspberry Pi migration** — purchase, set up Docker on Pi OS, copy the project folder, verify
  `docker compose up` brings up the identical stack, migrate the `.env` (new secrets, don't reuse
  the Mac's literally, rotate the DB password at minimum), point scheduling at the Pi.
- ⬜ **Remote access setup** — Tailscale (recommended — zero-config VPN, works well with a home Pi)
  or port-forwarding, so you can check on the Pi / pull data / review generated content without
  physical access.
- ⬜ **Automated backups** — Postgres data living only on a single home Pi's SD card/SSD is a real
  risk (SD card failure is common). Set up scheduled `pg_dump` backups to somewhere off-device (cloud
  storage, another machine) before trusting the Pi with irreplaceable accumulated insight history.
- ⬜ **Health monitoring / alerting** — a lightweight check (even just a Hermes cron job pinging the
  Pi's Postgres/poller) that tells you if the whole stack goes down, rather than silently finding out
  weeks later that nothing's been collected.
- ⬜ **Scheduling wired up for real** (see Part 2's open item #3) — whichever mechanism is chosen,
  needs to survive reboots (the Pi losing power and restarting shouldn't require you to manually
  restart everything).

---

## PART 10 — Cross-Cutting / Ongoing Engineering Hygiene — 🔁 ONGOING

These aren't one-time tasks, they're standards to keep maintaining as everything above gets built:

- 🔁 Keep `docs/session-context.md` updated as the project's "memory" — don't let it drift from
  reality the way a stale plan document would.
- 🔁 Every new generation/annotation prompt gets a version id; every playbook change gets logged;
  every content_item stays traceable to exactly what produced it (v2 plan §3.2 versioning discipline)
  — don't let this discipline erode as more pieces get added under time pressure.
- 🔁 Full test suite stays green — currently 33/33. Treat any skipped/ignored failing test as a
  blocker, not a thing to revisit later.
- 🔁 Keep test fixtures from polluting real data (the known gap flagged in Part 1 — a separate test
  DB would fix this permanently; until then, manual cleanup discipline after any DB-writing test run).
- 🔁 Re-verify live model IDs/API behavior periodically — providers change model names and platform
  scraping behavior (TikTok/Instagram) changes without notice; don't assume something that worked
  once still works months later without a quick live check.

---

## Suggested near-term execution order (not rigid — revisit as decisions resolve)

1. Populate the real TikTok watchlist with accounts you actually care about (Part 2, unblocked now).
2. Decide the scheduling mechanism and wire up the poller to run automatically (Part 2/Part 9).
3. Resolve the Instagram account decision (Part 3, open decision #1) — or explicitly defer it.
4. Build the annotation pass (Part 5a) against accumulated external TikTok data — this is where
   "reading" starts turning into "learning," and doesn't require owned performance data to start
   being useful (external data alone is enough to start recognizing creative patterns, even before
   you've published anything yourself).
5. In parallel, whenever you're ready: pick one of the 5 generated IG static images and actually
   publish it (Part 4's paused manual step) so owned performance data starts accumulating too.
6. Raspberry Pi purchase + migration (Part 9) whenever convenient — nothing above strictly requires
   it, it's a hosting upgrade, not a feature unlock (except the residential-IP reliability benefit for
   sustained scraping).
7. Pattern mining + playbook authoring (Part 5b/5c) once there's enough real data (owned or external)
   to mine something meaningful from — resist the urge to build this before there's real signal to
   validate it against.
8. Second workflow (Part 8, TikTok/Reels video) once the IG statics loop has proven itself end to end
   with at least one real playbook update driven by real data.
