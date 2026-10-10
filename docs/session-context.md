# Qualume — Project Context & Session Log

This file is the durable record of how this project's plan evolved, so a new session (human or
agent) can pick up the thread without needing the original chat transcript.

**Project:** Qualume (Quality Volume) — an AI content-creation engine that batch-generates
TikTok/Reels videos, IG statics, and future formats, each as its own workflow, and feeds performance
data into a learning loop that figures out what makes a good hook, headline, format, audio choice, etc.

**Planned stack:** Hermes (cockpit + playbook store), LangGraph (reasoning pipelines), n8n
(integration/orchestration plumbing).

---

## Folder layout

```
~/qualume/
  plans/          versioned plan documents (read v2 first, v1 kept for history)
  transcripts/    raw transcripts of reference videos used as research input
  docs/           this file + any future design docs
  data/           (empty, reserved for future local data/fixtures)
  src/            (empty, reserved for future code)
```

## Plan version history

- **`plans/v1-initial-plan.md`** — first draft. Built the architecture too directly off two
  Hormozi/ecommerce YouTube videos (treated one ad-agency operator's specific workflow as the
  reference architecture). **Superseded** — kept only for history.
- **`plans/v2-engineering-plan.md`** — **current plan.** Rebuilt from first-principles AI/software
  engineering practice (typed contracts, versioning, statistical significance gating, eval harnesses,
  explore/exploit, systematic debugging, tracer-bullet rollout) per explicit user correction (see
  "Key decisions" below). The two videos were demoted to Appendix A — labeled as unverified field
  notes/hypotheses to test, not architecture to copy.

Read `plans/v2-engineering-plan.md` top to bottom for the actual plan. This doc is index + rationale,
not a replacement.

## Key decisions / correction history (why the plan looks the way it does)

1. **User's original ask:** build a content-creation engine across multiple formats (each its own
   workflow), running on its own generated data, with a research arm to learn from other creators'
   content too — mission "Qualume = Quality Volume." Stack: Hermes + LangGraph + n8n. Gave two
   reference videos for context before planning.
2. **First plan (v1)** synthesized the two videos' specific workflow (Hormozi-style ad-agency
   self-learning pipeline; AI-run ecommerce launch) directly into the core architecture — e.g. the
   whole ingest→annotate→mine→playbook→generate→publish loop and the "v0 playbook" heuristics were
   near-paraphrases of one practitioner's specific ad-optimization workflow.
3. **User called this out:** "make sure u didnt just make the plan off the vids cus that wasnt my
   intention." Correct instinct — one small-sample practitioner case study (the source himself
   reports ~50% of his AI-informed ad batch flopped) is a weak foundation for core architecture.
4. **Plan rebuilt (v2)** from engineering principles: typed/versioned contracts at every pipeline
   boundary, idempotent+immutable data handling, explicit explore/exploit allocation (not pure
   copy-the-winners), statistical significance tiers before any mined pattern reaches a live
   playbook, shrinking (not zero) human-approval gates, systematic-debugging discipline (reproduce →
   trace → ranked hypotheses → one variable at a time → "3+ failed fixes = question the architecture"),
   and TDD-style eval-harness-first discipline. The videos were moved to an appendix as hypotheses to
   test against real data, not facts.
5. **Third video reviewed** (LangGraph data-agent tutorial, 7yOmi4IX-Rs, ~5hr hands-on build) — not
   yet folded into the plan as a dedicated section. Summary below. Useful concrete LangGraph patterns:
   Pydantic state (fail-loud validation), stateful-orchestration graphs vs. React tool-calling loops,
   model-tiering per node (cheap/medium/expensive LLM by task complexity), LLM-as-judge nodes with
   isolated prompts (exclude raw user input to prevent prompt injection into judge calls), supervisor/
   router pattern dispatching to composable sub-graphs, and a concrete LangGraph gotcha: parallel
   conditional branches must not both independently return/mutate the same state key (causes
   `InvalidUpdateError`) — route via explicit node-name mapping instead.
6. **Open/pending:** user asked whether to fold these LangGraph patterns into a new "§9 LangGraph
   implementation patterns" section of the v2 plan (with pseudo-code per pillar's graph) — not done
   yet as of this writing. Do that next if picking this project back up.

## Summary: Video 3 — LangGraph Data Agent tutorial (7yOmi4IX-Rs)

5-hour hands-on tutorial building a multi-agent "Data Agent" that routes natural-language business
questions to either a SQL Analyst sub-agent or an ETL (Python/pandas) Analyst sub-agent.

**Two core graph patterns taught:**

- **Stateful orchestration graph** (used for the SQL Analyst): fixed node sequence —
  `curate_question → add_schema_context → generate_SQL → is_safe?(conditional) → execute_SQL/cancel
  → represent_final_answer`. State is a Pydantic BaseModel (chosen over TypedDict specifically
  because Pydantic fails loudly/immediately on bad data — framed as a feature, not an inconvenience).
  Context engineering (injecting full DB schema + sample rows) is a dedicated node, not assumed.
  Model-tiering: cheap/low-end model for simple rewrites, higher-end for SQL generation itself, cheap
  again for final-answer formatting.
- **React (reasoning+acting) loop** (used for ETL Analyst): one LLM node bound to `@tool`-decorated
  Python functions; no fixed sequence. LLM reasons → calls a tool → tool node executes and appends a
  ToolMessage → LLM reasons again → repeats until done. Tool descriptions/docstrings are the only
  thing guiding the LLM's tool choice, so they must be detailed. Generated code (e.g. pandas
  transforms) executed server-side via `exec()` (flagged as fine for local dev; sandboxing/task
  queues recommended for production).
- **Supervisor/router pattern**: a third, simple LLM node with structured output
  (`answer: "SQL"|"ETL"`) classifies incoming requests and a conditional edge dispatches to whichever
  compiled sub-agent graph applies. Sub-graphs are composable — callable like any function from a
  parent node.

**Other patterns worth reusing in Qualume:**
- LLM-as-judge pattern: a dedicated node with its own narrow Pydantic output schema
  (`answer: yes/no`, `comments`) judges something (SQL safety, in this case) — deliberately excludes
  raw user input from the judge's prompt to prevent the user from manipulating the judge via the same
  channel used to generate the thing being judged.
- Real bug worth pre-empting: when two conditional branches each independently try to
  "return the full state," LangGraph throws `InvalidUpdateError — can receive only one value per
  step" because both branches attempt to update the same state key concurrently. Fix: route via an
  explicit node-name string/mapping from the conditional function, not parallel state mutation. This
  directly matters for Qualume's planned fan-out steps (hook×body matrix expansion, multi-variant
  scoring).

**Relevance mapping to Qualume's v2 plan:**
- Model-tiering per node → apply to Write/Design engines (cheap model for ideation/formatting,
  expensive model only for the generation step that must be structurally/creatively correct).
- LLM-as-judge with isolated prompts → direct pattern for the synthetic-audience pre-flight scoring
  step and any "is this generated asset safe/on-brand to publish" gate.
- Supervisor + composable sub-graphs → natural LangGraph shape for Qualume's Read/Learn/Write/Design
  pillars: one top-level router dispatching to per-workflow_type sub-graphs (tiktok_reel generation
  graph, ig_static generation graph, etc.), each independently compiled and testable.
- Parallel-state-update gotcha → explicit thing to design around wherever Qualume's graphs fan out.

---

## Next steps (as of this writing)
1. Decide whether to add a "LangGraph implementation patterns" section to the v2 plan with
   pseudo-code per pillar (Read/Learn/Write/Design) incorporating the patterns above. (Still open.)
2. Execute `plans/execution-plan-phase-0-1.md` — see below, this is now the active work.

## Phase 0/1 tooling decisions (confirmed 2026-10-10)

These resolve the "open questions" that were blocking a concrete execution plan:
- **LLM provider: Anthropic (Claude) only.** Used for annotation (Claude vision handles IG-static
  image annotation directly), generation, and judge/critique calls. No Gemini/OpenAI in Phase 0/1.
- **Infra: all local via Docker** on this machine (not cloud/hosted). Docker was NOT installed on
  this machine as of 2026-10-10 — `docker --version` returned "command not found", no Docker
  Desktop/OrbStack found in /Applications. First execution task installs OrbStack via
  `brew install --cask orbstack` (needs explicit user approval — installs system software).
- **Phase 1 pilot workflow: IG statics** (confirmed, matches v2 plan's recommendation — simplest
  Design engine, no video assembly).
- **Data reality: fully greenfield.** One Instagram account exists, zero posts, zero performance
  data, no Meta Developer App / Graph API access configured yet. This is the single biggest factor
  shaping the execution plan's Phase 1 shape:
  - Automated IG ingestion (Graph API) requires IG Business/Creator conversion + FB Page link + Meta
    Developer App + access token — a multi-day external-approval dependency that must NOT block the
    tracer bullet.
  - Solution: Phase 1 splits ingestion into **1a (manual performance entry via a small CLI script,
    unblocked, start immediately)** and **1b (real Graph API ingestion, parallel task, no hard
    blocking dependency on 1a)**.
  - Phase 1 publish step is also manual (you post to IG by hand after reviewing generated assets) —
    automating posting needs the same Graph API access as ingestion, not worth building before
    Write/Design output is even proven worth posting.
  - Design engine (Phase 1) uses **Pillow-based template rendering** (headline/body text placed onto
    pre-made template backgrounds) rather than an image-gen model/service — simplest, zero-API-cost,
    fully deterministic option to prove the pipeline shape first. Swapping in a real design-gen tool
    is an explicit fast-follow, not a Phase 1 blocker.

## Execution plan

**`plans/execution-plan-phase-0-1.md`** is the concrete, bite-sized build plan for Phase 0
(foundations: Docker/Postgres/n8n stack, Python project with uv, Pydantic contracts, model-tiering
helper, eval harness skeleton, seeded v0 playbook) and Phase 1 (tracer bullet: Write engine → Design
engine → eval gate → manual publish → manual performance logging, IG statics only, no automated
learning yet). Each task has exact file paths, code, and a verify step.

## Phase 0 — COMPLETE (2026-10-10)

All 10 tasks done, all 4 exit checks green:
- OrbStack installed (`brew install --cask orbstack`) — Docker CLI works via `~/.orbstack/bin`
  (auto-added to `.bash_profile`, new shells pick it up; this session needed a manual `export PATH`).
- Project scaffolded under `~/qualume/src/qualume/{db,models,llm,graphs,eval,ingest}` + `tests/`.
- `uv init` + deps installed: langgraph, langchain-anthropic, pydantic, psycopg2-binary,
  python-dotenv, pytest, pillow. All import cleanly.
- `.env` created from `.env.example` (gitignored); real `ANTHROPIC_API_KEY` set by user and
  verified live against the Anthropic API (made an actual API call, got a response).
- Verified live Claude model IDs via `anthropic.Anthropic().models.list()` — do NOT trust
  placeholder model names, the real working ones as of 2026-10-10 are: `claude-haiku-4-5`,
  `claude-sonnet-4-5`, `claude-opus-4-5` (all three confirmed working via live `.invoke()` calls).
- `docker-compose.yml` brings up Postgres 17 (port 5433) + n8n (port 5678). Both confirmed running
  and responsive.
- Schema migration (`src/qualume/db/migrations/001_init_schema.sql`) applied — 8 tables exist:
  content_items, creative_dna, performance_snapshots, ad_performance, experiments, insights,
  playbooks, eval_runs.
- Pydantic contracts (`src/qualume/models/contracts.py`) built TDD-style (RED confirmed before
  implementation) — `RawContentRecord`, `CreativeDNA`, `CampaignBrief`, `ScriptVariant`,
  `DesignRequest`, `DesignOutput`. 3 tests passing.
- Model-tiering helper (`src/qualume/llm/pick_llm.py`) — `pick_llm("low"|"medium"|"high")` using the
  verified model IDs above. 2 tests passing.
- Eval harness skeleton (`src/qualume/eval/harness.py`) — deterministic `must_include`/`must_avoid`
  constraint checks (no LLM judge yet, intentionally — that's added once there's a real Write
  prompt to score). 3 tests passing. Golden set seeded at
  `src/qualume/eval/golden_sets/ig_static_write_v1.jsonl` (3 hand-written examples).
- v0 `ig_static` playbook seeded BOTH as a DB row (`playbooks` table,
  `source: external_field_notes, tier: proposed`) and as a Hermes skill (`ig-static-playbook`,
  readable via `skill_view(name='ig-static-playbook')`) — explicitly marked unproven, not to be
  cited as "what works" until real performance data validates it.
- All 8 pytest tests passing. Git history: 6 commits, from initial scaffold through Phase 0 complete.

**Known gotcha for next session:** `docker`/`docker compose` need
`export PATH="$HOME/.orbstack/bin:$PATH"` in any NEW terminal session that was started before
OrbStack was installed, OR just open a fresh terminal (OrbStack already added itself to
`~/.bash_profile` so brand-new shells get it automatically).

## Immediate next action

Start Phase 1 (tracer bullet) per `plans/execution-plan-phase-0-1.md` §2: Task 1.1 (Write engine
LangGraph for ig_static), then 1.2 (Design engine, Pillow template rendering), 1.3 (wire into a batch
script with eval gate), 1.4 (manual review/publish), 1.5 (manual performance-entry CLI), and 1.6
(start the Instagram Graph API access setup in parallel — has external approval latency, don't block
on it).

## Phase 1 — IN PROGRESS (2026-10-10)

Tasks 1.1-1.3, 1.5 done and verified live. Task 1.4 (manual publish) is blocked on YOU, not on code —
waiting for you to pick one of 5 generated images and actually post it to Instagram. Task 1.6
(Graph API setup) not started yet.

- **1.1 Write engine** (`src/qualume/graphs/write_ig_static.py`) — LangGraph stateful-orchestration
  graph: load_playbook → ideate_angles → generate_variants → self_critique. Model tiering:
  pick_llm("medium") for ideation/generation, pick_llm("low") for the critique pass. Verified live
  against real Claude API — produced genuinely distinct, on-topic hook/body/CTA copy. Unit tests use
  mocked LLMs (don't hit the real API in tests) — 2 tests, both passing.
- **1.2 Design engine** (`src/qualume/graphs/design_ig_static.py`) — Pillow template rendering, NOT
  an image-gen model (deliberate Phase 1 simplification — deterministic, zero-API-cost). Template:
  `data/templates/template_cream_square.png` (plain cream 1080x1080 square, generated programmatically,
  no real brand kit yet since greenfield). Render-determinism test confirms byte-identical output for
  identical input. Visually verified via screenshot — clean, legible layout, no overflow. 4 tests
  passing.
- **1.3 Batch runner** (`src/qualume/run_batch.py`) — wires Write → eval gate → Design → Postgres
  persist. **Caught and fixed a real bug during build**: the original eval-gate design tried to match
  a brief to the "closest" golden-set example by keyword overlap, silently falling back to an
  unrelated example (e.g. requiring the word "journal" in a wallet ad) when no match existed — this
  would have silently broken the gate for almost any real brief. Fixed by splitting the gate into (a)
  a universal must_avoid list (banned claims like "guaranteed results" — unioned across all golden
  examples, applies to any brief) and (b) a per-brief must_include derived from the brief's own text
  via a stopword-based keyword extractor, tested against both the original and an unrelated product
  category to confirm it generalizes. 4 regression tests added for this fix, all passing.
- **1.5 Manual performance-entry CLI** (`src/qualume/ingest/manual_entry.py`) — Phase 1a per the
  execution plan's greenfield adaptation (no Graph API access yet, log stats by hand). Append-only
  (never upserts) — tested explicitly that logging twice produces two rows, not one updated row, per
  the v2 plan's immutability principle. 3 tests passing, verified live against the real DB.
- Also built along the way (not originally itemized as separate tasks but needed): `src/qualume/
  config.py` (env loader) and `src/qualume/db/connection.py` (psycopg2 connection helper) —
  prerequisites for everything above; `src/qualume/db/playbook_loader.py` (loads the active playbook
  row for a workflow_type, raises loudly if none exists) — 2 tests passing.
- **Current state**: 5 real draft content_items in Postgres (2 productivity-journal variants, 3
  water-bottle variants from a deliberately different product category used to test the eval-gate
  fix), each with a rendered PNG in `data/assets/` (gitignored — generated artifacts, not source) and
  full creative_dna + provenance (playbook_version, prompt_versions_used for ideation/write/design).
  23/23 tests passing across the whole project.
- **Gitignore gotcha hit twice**: `data/assets/*.png` only ignored top-level files, not files in
  subdirectories created during tests (e.g. `data/assets/_test_tmp/...`) — two stray test-generated
  PNGs got committed before this was caught and fixed. Current rule is `data/assets/` (whole dir
  excluded). If a new nested asset dir pattern appears, verify it's actually excluded before
  committing, don't assume.

### Still blocked / not started
- **Task 1.4 (manual publish)**: genuinely needs YOU — review the 5 generated images (one shown
  inline in session: a stainless-steel water-bottle ad, "Your gym bottle smells like a locker room by
  Friday"), pick one, post it to Instagram yourself. Not something Hermes can or should do
  autonomously. Once posted, update that content_item's status to 'published' in Postgres:
  `UPDATE content_items SET status='published', published_at=now() WHERE id='<id>';` — then start
  logging daily stats via Task 1.5's CLI.
- **Task 1.6 (Graph API access)**: not started. No hard dependency on anything else — can run in
  parallel whenever picked up. See execution plan §Task 1.6 for the exact steps (convert IG account
  to Business/Creator, link FB Page, create Meta Developer App, generate token).

## Reading engine pivot (2026-10-10) — user redirected focus

After Phase 1 build progress above, user explicitly redirected: "lets focus on the reading engine and
build up from there since everything relies on that." This is the right call — Read is the foundation
the Learn engine depends on, and nothing existed for it yet. Also clarified real intent: wants to
pull data from TikTok/Instagram videos "I choose" (not just own-account analytics), record as much
data as wanted, and eventually run this 24/7 on dedicated hardware (Raspberry Pi) or cloud, learning
continuously what content/videos perform well.

### Platform research findings (verified live, not guessed)

- **TikTok**: public video/profile metadata fully readable with `yt-dlp`, NO login required. Verified
  live: pulled real views/likes/comments/reposts/duration/audio-track data from real public videos
  and from a real account's full profile feed (`https://www.tiktok.com/@handle` returns a playlist of
  all recent videos). One test video hit a transient "IP blocked" error — intermittent, not universal;
  residential IPs (i.e. a home-based Pi) are less likely to trigger this than datacenter/VPS IPs.
- **Instagram**: confirmed BLOCKED without an authenticated session — tested via both `yt-dlp` (error:
  "Instagram sent an empty media response... check if accessible without being logged in") AND a real
  browser session (two different live, currently-public Reels both showed "Post isn't available" when
  not logged in). This is a hard platform-level requirement, not a Pi/IP issue — login cookies from a
  real IG account are required for ANY IG content reading, own or others'. Carries real ToS violation
  risk and account-flagging risk (temporary action blocks, shadow-restriction, possible disable) for
  whichever account is used to scrape. User asked about using the business's real "greenbabe" account
  for this — explicitly flagged the risk (flagging/restriction would hit the real business account,
  not a disposable one) and recommended a separate/secondary account instead. User has NOT yet decided
  (question outstanding as of this writing) — do not build IG scraping against greenbabe without an
  explicit decision on this.
- **Industry validation (web research, not guessing)**: this exact pattern (scheduled polling of
  chosen accounts/hashtags → normalized schema → dataset → LLM pattern-mining) is a well-established,
  commercially-productized approach — found real examples: SociaVault, Exolyt, Virlo, Kalodata (all
  commercial "social media intelligence" products built on this), and a notable open-source reference,
  `ScrapeCreators/social-media-research-skills` (2,900+ GitHub stars) — an AI-agent skill toolkit for
  exactly this kind of research (outlier-post-finder, transcript-intelligence, creator-profile-teardown
  skills). Apify's own documented architecture guide confirmed: self-hosting beats their paid platform
  specifically at "stable, moderate-volume workloads" (2 vCPU/2GB RAM tier) — which matches this
  project's actual expected usage (a handful of watched accounts, polled every few hours), validating
  the self-hosted approach already being built rather than needing a paid scraping API.

### Raspberry Pi vs cloud decision (discussed, Pi favored, not yet purchased)

- Pi 5 (8GB) ≈ $100-130 one-time + ~$1/month electricity (5-8W continuous draw for this workload).
  Cloud VPS ≈ $5-6/month forever (no ceiling). Breakeven ~18-20 months; Pi cheaper after that and
  increasingly so over time.
- Bigger reason than pure cost: **residential IP from a home Pi is less likely to be detected/blocked
  by TikTok/Instagram's anti-bot systems** than a datacenter/VPS IP range — this matters more for this
  specific use case than the dollar difference.
- Important nuance clarified to user: Pi fixes IP-detection risk for BOTH platforms, but does NOT
  remove Instagram's login-wall requirement — that's a separate, harder constraint Pi doesn't solve.
- Pi also generalizes well beyond this one engine: once there's an always-on Linux box, it can run
  any number of scheduled jobs (systemd timers, cron, n8n's built-in scheduler) independent of whether
  the Mac is open or Hermes is running — useful framing since user explicitly wants "more cron jobs"
  over time.
- User has not yet purchased a Pi. Current build is happening on the Mac, designed to be fully
  portable (Docker Compose + Python project, no Mac-specific assumptions) so migration later is just
  copying the project folder and running the same `docker compose up` on the Pi.

### What's actually been built for the reading engine (Phase: Read engine, TikTok-only so far)

New DB tables (migration `src/qualume/db/migrations/002_external_content.sql`), deliberately
SEPARATE from the owned-content tables (`content_items` etc. from Phase 0/1) per the v2 plan's
"two lanes, structurally separate, different trust levels" principle — external/competitor content
should never be structurally conflatable with Qualume's own generated+published content:
- `watched_accounts` — which TikTok (future: IG) accounts are being tracked, platform+handle unique,
  active/inactive flag so you can pause watching an account without losing history.
- `external_content_items` — one row per discovered video: platform_video_id (the TikTok ID — this
  is the dedup key), url, caption, duration, audio track/artist, posted_at, full raw yt-dlp metadata
  JSON (so nothing is lost even if we haven't modeled every field yet).
- `external_performance_snapshots` — append-only, same pattern as owned-content's
  `performance_snapshots`: views/likes/comments/shares over time, not just latest values.

Code (`src/qualume/`):
- `db/watchlist.py` — add/list/deactivate watched accounts. 3 tests passing.
- `ingest/tiktok_reader.py` — `fetch_account_videos()` (wraps yt-dlp, pulls a profile's recent videos,
  no login), `parse_video_entry()` (normalizes yt-dlp's raw dict into our schema, handles missing
  fields gracefully rather than crashing), `ingest_videos_for_account()` (the core dedup logic — checks
  platform_video_id before inserting, so re-polling an unchanged account inserts nothing new). 5 tests
  passing, including a live-data verification run (polled a real account twice — first run: 5 new
  videos; second run: 0 new, 5 correctly recognized as duplicates).
- `poll_tiktok_watchlist.py` — the actual script a cron job/scheduler would invoke: sweeps every
  active watched account, polite delay between accounts (avoid looking like a traffic burst), and
  critically: **one account's failure doesn't kill the whole sweep** (try/except per-account, logged
  and counted, loop continues) — tested explicitly with a simulated broken account among working ones.
  2 tests passing.
- `manage_watchlist.py` — convenience CLI (`add`/`list`/`deactivate`) for managing the watchlist by
  hand. Verified live end-to-end.

38 tests passing project-wide as of this commit. Live-verified against real TikTok data (not just
mocks) — a real account (@scout2015, used only as a smoke-test target, then removed from the
watchlist since it's not a genuine account to track) was polled twice, proving both the extraction
and the dedup logic work against actual TikTok responses, not just test fixtures.

### Immediate next steps for the reading engine
1. User decides on the IG scraping account question (greenbabe directly vs. separate account vs.
   skip IG for now) — currently unresolved.
2. User populates the real watchlist with actual TikTok accounts/competitors they want tracked, via
   `uv run python -m qualume.manage_watchlist add tiktok <handle>`.
3. Set up actual scheduling (cron, or n8n's built-in scheduler since n8n is already running in
   Docker) to call `poll_tiktok_watchlist.py` on a cadence — not yet wired up, currently a manual-run
   script.
4. Eventually: the annotation/Learn-engine pass over `external_content_items` (multimodal annotation,
   pattern mining per v2 plan §2.2) — not started, this session only built Read.
5. Raspberry Pi purchase + migration — not started, current work is fully portable to it when ready.
