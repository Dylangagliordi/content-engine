# Qualume — AI Content Creation & Learning Engine — Plan

> Mission: "Quality Volume" — batch-generate content across formats (TikTok/Reels, IG statics, etc.),
> publish it, measure it, and feed the results into a learning loop that gets smarter about what
> makes a good hook, format, headline, audio choice, etc. Stack: Hermes (cockpit + skills-as-playbooks),
> LangGraph (reasoning pipelines), n8n (plumbing/orchestration/integrations).

**Status:** planning only — no build yet. This doc is the blueprint to execute against.

---

## 1. What the two reference videos actually teach us

### Video 1 — Hormozi ad-strategy / self-learning ad agent (xJ99dBYhpRI)
Core system described, generalized:

1. **Ingest organic performance data** from IG/FB via API — views, saves, comments, follows per post.
   Saves/follows are weighted as *stronger* signal than views (save = "I want this again", follow =
   "I want more of this creator" — both beat vanity view counts).
2. **Multimodal annotation, not just transcription.** Feed video into Gemini to get: spoken transcript
   AND on-screen text/captions AND a description of the visual hook/pattern-interrupt in the first
   few seconds. The hook is frequently visual/textual, not verbal — a transcript-only pipeline misses
   the single most important variable.
3. **Dual data feed**: organic content performance + paid ad performance (CPM/CPC/ROAS) both get
   annotated the same way and fed into one reasoning model (Claude Code in his case).
4. **Clean attribution matters.** Platform self-reported attribution (Meta/Google) is unreliable;
   he layers in a third-party attribution tool (Hyros) so ROAS numbers used for learning are trustworthy.
   Garbage attribution in = garbage pattern-mining out.
5. **Pattern-mining produces a reusable "skill"** — a living rubric capturing: how to write hooks
   that convert, which formats actually produce ROAS (not just views/cheap CPC) — skits vs talking-head
   vs raw vs high-production — how to structure a script body, how/when to use social proof, what
   kills conversions.
6. **Generation is batched as a hook×body matrix** — e.g. 3 bodies × 10 hooks = 30 variants from one
   filming session, explicitly produced because "there's such a massive outlier in how good the hooks are."
7. **Synthetic audience pre-flight testing** — simulated ICP personas (deal-breakers, buying triggers,
   scam-detectors, etc.) react to a draft ad before it's produced/spent-on, predicting performance and
   feeding rewrite suggestions back to the generator.
8. **Closed loop**: publish → re-ingest performance → re-annotate → re-mine patterns → rubric updates →
   next batch is better. Volume of data in is what makes the loop converge.
9. **Planned extension: competitor/other-creator content** as a secondary, lower-weighted training
   signal (you can get views/comments on competitors but not saves/ROAS) — used mainly to surface
   *unique angles*, and explicitly generalizable to other platforms (he mentions YouTube).

### Video 2 — AI-run ecommerce business end to end (uX6zwf4b8sM)
This is less about the content-learning loop and more about **full-stack autonomous operation** —
still highly relevant because Qualume needs the same operational scaffolding around content:

1. **Multi-agent role split** — one agent plans/researches, one builds, one runs ads, one handles ops —
   coordinated from one cockpit (Claude Desktop + CLI), each given narrow, well-defined objectives.
2. **Decisions get written down as a PRD** before execution — conversational decisions get compiled
   into a structured doc that a coding/build agent then executes against. This is the discipline that
   keeps multi-agent systems from drifting.
3. **Competitor research precedes creative generation** — scrape competitor ad libraries for tactics/
   offers/positioning into a doc, *then* brief the creative-generation step from that doc.
4. **Big batch generation is explicit here too** — "a PRD for 50 ad creatives" generated in one pass.
5. **Organic + paid are built in parallel from day one**, not sequentially — paid ads validate the
   offer fast; organic (with a cold-start bootstrap via paid reposts/shout-outs from adjacent accounts)
   builds a compounding owned-audience asset.
6. **Systematic debugging of underperformance**: when results don't match expectations, check the
   offer/trust layer first (price, social proof, credibility signals), then the technical funnel
   (they found a broken, slow-loading checkout only by debugging "high traffic, zero sales").
7. **Offer iteration based on data**: price was dropped, free shipping + bonuses added, bundles added
   to lift perceived value/AOV — each change driven by a specific observed failure, not guesswork.

### Synthesis — the shape of the system both videos converge on
```
[Ingest] -> [Multimodal Annotate] -> [Normalize to shared schema] -> [Pattern-mine / learn]
    -> [Playbook / rubric, versioned] -> [Batch-generate variants, informed by playbook]
    -> [Synthetic-audience pre-flight score] -> [Publish organic + paid] -> back to [Ingest]
```
Plus a parallel, lower-trust branch: **external research** (competitors / other creators) feeding
the same annotate → normalize → pattern-mine stages, kept separate/weighted so it informs but doesn't
pollute the owned-data signal.

---

## 2. Architecture for Qualume

### 2.1 Shared data model (the backbone — build once, reuse across every workflow)
A single Postgres schema so TikTok/Reels, IG statics, and future formats are cross-comparable and the
learning loop can find patterns *across* formats, not just within one.

Core tables:
- `content_items` — id, workflow_type (tiktok_reel / ig_static / ...), platform, status, batch_id,
  playbook_version_used, created_at, published_at, asset_url(s)
- `creative_dna` — one row per content_item: hook_type, hook_text, visual_hook_description
  (from multimodal annotation), format_tag (skit/talking-head/raw/high-production/carousel/single-image),
  script_structure_tags, cta_type, audio_id/track, caption_style, pacing_notes, social_proof_used
- `performance_snapshots` — content_item_id, ts, views, likes, saves, comments, shares, follows_gained,
  watch_time/retention if available
- `ad_performance` — content_item_id, spend, CPM, CPC, ROAS, attribution_source (platform vs
  clean-attribution tool), conversions
- `external_benchmark_items` — same shape as content_items/creative_dna but source=competitor/creator,
  confidence_weight (lower, since save/ROAS data usually unavailable)
- `experiments` — batch_id, hypothesis, variant_list (hook×body matrix), workflow_type, outcome_summary
- `playbooks` — workflow_type, version, rubric_text/structured_json, generated_at, source_insight_ids,
  approved_by/approved_at (human gate)
- `insights` — id, statement, supporting_data_refs, confidence, status (proposed/approved/rejected),
  applies_to_workflow_type

Object storage (S3/R2/Supabase storage) for raw video/image assets, generated creative assets, and
annotation artifacts (Gemini outputs).

### 2.2 Tooling split — who does what and why

**n8n — plumbing and integrations.** Good at: scheduled polling of platform/content APIs or scraping
providers, webhook receivers, posting content out (via Ayrshare/Buffer-style APIs or native platform
APIs), retry/error handling on flaky third-party calls, file movement between storage, Slack/Telegram
notifications, and — critically — **triggering LangGraph runs via HTTP** so the reasoning work doesn't
live in n8n's node-graph (which gets unmanageable for actual LLM reasoning chains).

**LangGraph — the reasoning pipelines.** Deployed as a LangGraph server (local or LangGraph Cloud),
each pipeline exposed as a callable graph/endpoint:
- `ingest_annotate_graph` — pulls raw content + calls Gemini (or GPT-4o-vision/Claude) for multimodal
  annotation, writes normalized rows to `creative_dna`
- `generation_graph[workflow_type]` — one graph per content workflow (tiktok_reel, ig_static, ...):
  ideation → copy/script → asset assembly → batch variant expansion (hook×body matrix)
- `synthetic_audience_graph` — scores draft variants against ICP personas before publish/spend
- `insight_mining_graph` — periodic job: correlate creative_dna + performance across winners/losers/
  middle, output candidate `insights` rows
- `playbook_authoring_graph` — turns approved insights into a diffed rubric update for a given
  `playbooks` row (human approval gate before commit, at least through Phase 3)

**Hermes — the cockpit, supervisor, and playbook store.** This is the key structural idea: Hermes's
own **skill system is a natural home for the "playbook"** concept from Video 1 ("Claude learns a
skill that has an understanding of what creates outliers"). Concretely:
- Each workflow gets a living skill, e.g. `tiktok-reel-playbook`, `ig-static-playbook`, maintained via
  `skill_manage(action='patch')` whenever `playbook_authoring_graph` proposes (and you approve) an update.
- Hermes cron jobs run the daily/weekly digest: "here's what won this week, here's the proposed
  playbook diff, approve/reject" — delivered to you for a lightweight human-in-the-loop gate.
- Hermes is also where you issue ad hoc commands ("generate 20 more hook variants for angle X",
  "pull last week's top 5 by ROAS") without having to open n8n or the LangGraph server directly.
- Hermes (this session, via delegate_task/subagents) is also the most practical place to actually
  *build* the n8n workflows and LangGraph graphs — i.e., use Hermes as the dev agent for Qualume itself.

### 2.3 Per-workflow content engines
Each content type (TikTok/Reels, IG statics, future formats) is its own LangGraph generation graph but
shares the backbone schema and the ingest/annotate/mine/playbook infrastructure. Common internal shape:
```
ideation (playbook-informed angle+hook ideas)
  -> script/copy generation
  -> asset generation or sourcing (template/UGC/AI-avatar/stock)
  -> synthetic-audience scoring (filter/rank variants pre-production or pre-spend)
  -> batch variant expansion (N hooks x M bodies/templates)
  -> tag with creative_dna, queue for publish
```
Differences live in the asset-generation step (video edit pipeline vs static/carousel image pipeline)
and in which creative_dna fields are even meaningful (e.g. "pacing/cuts" doesn't apply to a static post,
but "headline placement/copy density" does).

### 2.4 External research pipeline (the piece explicitly called out as needing its own plan)
Separate but schema-compatible pipeline for learning from content you don't own:
1. **Source definition** — competitor accounts, adjacent-niche creators, hashtag/sound trend lists,
   per workflow_type (don't mix a TikTok competitor list into the IG-static research set).
2. **Ingestion** — n8n-scheduled pulls via a vetted scraping/API provider (e.g. Apify, EnsembleData,
   Phyllo, or official platform APIs where available — TikTok Research API, YouTube Data API, Meta
   Content Library). Respect ToS; prefer official/vetted providers over raw scraping.
3. **Annotation** — same Gemini multimodal pass as owned content → populate `external_benchmark_items`.
4. **Confidence weighting** — explicit lower weight in the mining graph: we only get public engagement
   (views/likes/comments/shares), never saves or true conversion/ROAS, so outputs from this branch are
   treated as "angle/format hypotheses to test" rather than "proven winners."
5. **Output** — feeds `insight_mining_graph` as a secondary input; primary use case is surfacing format/
   hook/trend ideas we haven't tried yet, cross-checked once we test them with our own data.
6. **Cross-platform extensibility** — same pipeline shape reused for YouTube Shorts, etc., per Video 1's
   closing point that this generalizes beyond IG/FB.

### 2.5 Playbook v0 — seed it now, don't wait for data
Volume takes time to accumulate before pattern-mining is statistically meaningful. Seed each workflow's
initial playbook skill directly from these two transcripts so day-1 generations aren't starting blind:
- Content that *doesn't look like an ad* outperforms traditional direct-response-formula ads; prefer
  repurposing top organic content as ad creative over producing bespoke "salesy" creative.
- Weight **saves and follows** as the leading indicator of "ready to become an ad," not raw views.
- Every annotation pass must capture the **visual/on-screen hook**, not just spoken transcript — hooks
  are frequently non-verbal in the first 3-5 seconds.
- Generate in **hook×body batches** (e.g., 10 hooks × 3 bodies) per production session, not one-offs.
- Validate demand before producing content/ads for a product/offer — don't invent demand.
- Build organic and paid distribution **in parallel from day one**; bootstrap organic cold-start via
  paid shout-outs/reposts in adjacent niches if needed.
- When a batch underperforms expectations, debug **offer/trust first, technical funnel second** before
  assuming the creative itself failed.
- Use bundles/bonuses to test perceived-value/price elasticity rather than only testing creative.
- Once ad spend is non-trivial, invest in clean attribution (not platform self-reported) before trusting
  ROAS numbers for learning.

---

## 3. Phased build roadmap

### Phase 0 — Foundations (infra, no content yet)
- Stand up Postgres (Supabase is a reasonable default — gives DB + storage + auth in one) and object
  storage for assets.
- Design/migrate the shared schema from §2.1.
- Stand up a LangGraph server (local Docker first; LangGraph Cloud later if needed) and an n8n instance
  (self-hosted via Docker is fine to start).
- Decide the ingestion provider(s) for owned-account analytics (native Graph API for IG/FB; TikTok
  Business API) and for external/competitor data (pick one vetted provider — don't build 3 scrapers).
- Decide the multimodal annotation model (Gemini per the video's approach is a reasonable default;
  note cost/volume tradeoff — annotate every item at first, revisit sampling once volume is high).
- Create the two seed Hermes skills: `tiktok-reel-playbook`, `ig-static-playbook`, pre-populated with
  the v0 rubric from §2.5.

### Phase 1 — MVP loop, ONE workflow only (recommend: TikTok/Reels first — richer signal, matches both
videos' focus)
- Build `generation_graph[tiktok_reel]`: ideation → script → batch hook×body expansion (no synthetic
  audience yet — add that in Phase 2).
- Build `ingest_annotate_graph` against your own published content.
- n8n: scheduled pull of your own account's performance data → push into Postgres → trigger annotate
  graph for anything new.
- No automated learning yet — just get the full loop operating end-to-end and *visible* (a simple
  dashboard or even a Hermes cron digest of "here's what we published this week and how it did").
- Exit criteria: you can trigger a batch generation, publish it, and see real performance data land
  back in the DB, tagged with creative_dna, within the same week.

### Phase 2 — Add scoring + first learned playbook update
- Build `synthetic_audience_graph` with an initial persona set (can start from assumptions, refine as
  real customer/follower data accumulates).
- Build `insight_mining_graph` v1 — simple correlation/clustering over whatever data has accumulated.
- Build `playbook_authoring_graph` — proposes a diff; you approve via Hermes; committed via
  `skill_manage(action='patch')` against `tiktok-reel-playbook`.
- Exit criteria: one real playbook update has been proposed, reviewed, and merged based on actual
  performance data (not just the v0 seed).

### Phase 3 — External research pipeline
- Stand up the competitor/creator ingestion per §2.4 for the TikTok/Reels workflow first.
- Feed into the same annotate/mine pipeline with explicit lower confidence weighting.
- Exit criteria: at least one insight sourced from external data has been tested in an actual batch
  and its outcome recorded (proving the hypothesis loop closes, win or lose).

### Phase 4 — Automate the learning cadence
- Put `insight_mining_graph` and `playbook_authoring_graph` on a schedule (weekly is a reasonable
  starting cadence given batch volume needed for signal) via n8n or Hermes cron, still with a human
  approval gate before any playbook/skill is patched.

### Phase 5 — Replicate for IG statics (and future formats)
- Reuse all shared infra; build only the workflow-specific `generation_graph[ig_static]` and adapt
  which creative_dna fields matter (headline/copy density/carousel structure vs video pacing).
- Cross-pollinate: let `insight_mining_graph` look for patterns that hold *across* formats, not just
  within one (e.g., a hook phrasing that works in both video and static copy).

### Phase 6 — Scale external research + cross-platform
- Extend ingestion to YouTube Shorts and any other relevant platform, same schema.
- Revisit annotation sampling strategy as volume grows (full-annotate recent/outlier content, sample
  the long middle to control cost).

---

## 4. Risks & open questions (resolve before/while building, not after)

- **API/ToS access**: official organic insights APIs require owned Business accounts; competitor data
  requires either an official research API (limited) or a vetted third-party provider — raw scraping
  carries ToS and ban risk. Pick providers deliberately in Phase 0, don't default to ad hoc scraping.
- **Attribution**: for paid ad learning to mean anything, platform self-reported ROAS isn't trustworthy
  at scale — plan for a clean-attribution layer (Hyros-equivalent) once ad spend is material.
- **Annotation cost**: multimodal annotation of every single video adds up fast at "volume" scale —
  budget this explicitly and plan the sampling strategy for Phase 6 rather than letting cost surprise you.
- **Minimum viable data volume**: pattern-mining needs enough batch volume before signal is trustworthy;
  lean on the v0 seeded playbook (heuristics from these two videos) for early batches rather than
  expecting the learning loop to produce anything useful on week one.
- **IP/compliance on external research**: the goal is angle/pattern inspiration, not content reuse —
  build a compliance checkpoint into the generation graph so external-derived ideas get transformed,
  not copied.
- **Schema commitment**: the shared creative_dna/performance schema is the thing hardest to change
  later — get a second pass of review on §2.1 before writing migrations, since every workflow and every
  pipeline downstream depends on it.
- **Tool choice confirmation needed from you**: which DB/storage (Supabase vs raw Postgres+S3), which
  annotation model (Gemini vs alternative), which organic/competitor data providers, and where n8n/
  LangGraph will actually be hosted (local Docker vs cloud) are all still open and will shape Phase 0
  concretely — worth a short decision pass before Phase 0 kicks off.

---

## 5. Suggested immediate next actions
1. Confirm/choose the Phase 0 tooling decisions above (DB/storage, annotation model, data providers,
   hosting).
2. I can draft the actual Postgres schema (DDL) and the two seed Hermes skills
   (`tiktok-reel-playbook`, `ig-static-playbook`) with the v0 rubric from §2.5 right away — these have
   no external dependencies and unblock everything else.
3. Stand up a throwaway LangGraph + n8n docker-compose locally to validate the "n8n triggers LangGraph
   via webhook" integration pattern before building real graphs against it.
4. Pick the single workflow (recommend TikTok/Reels) and single data source to pilot Phase 1 against,
   so the first real loop closes on something narrow before expanding.
