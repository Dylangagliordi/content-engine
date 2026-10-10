# Qualume — AI Content Creation & Learning Engine — Plan (v2)

> Supersedes `2026-10-10_qualume-content-engine-plan.md`. That version leaned too heavily on two
> reference YouTube videos as architecture. This version is designed from engineering first
> principles — modularity, contracts, observability, testing, and systematic debugging — with the
> videos demoted to Appendix A: field notes from other practitioners, treated as unverified
> hypotheses to test, not facts to build around.

**Mission:** Qualume = Quality Volume. Batch-generate content across formats (TikTok/Reels, IG
statics, and future formats), each as its own workflow, publish it, measure it, and run a disciplined
learning loop that produces increasingly reliable, *tested* answers to: what makes a good hook, which
headline/format patterns work, which audio choices perform, etc.

**Stack:** Hermes (cockpit, playbook store, human-in-the-loop gate), LangGraph (reasoning pipelines),
n8n (integration/orchestration plumbing).

---

## 1. Engineering principles this plan is built on

These are the standards every component below is held to. If a later design decision conflicts with
one of these, the principle wins — flag it and resolve the conflict explicitly rather than silently
cutting corners.

1. **You cannot improve what you cannot measure, and you cannot trust what you haven't tested.**
   Before any generation pipeline ships, there must be an eval harness capable of scoring its output
   against a held-out golden set. Build the harness *before or alongside* the generator, not after.
2. **Explicit contracts at every boundary.** Every hand-off between pipeline stages (ingest→annotate,
   annotate→mine, mine→playbook, playbook→generate, generate→publish) is a typed schema (Pydantic
   models), versioned, and validated on write. A stage that receives malformed input fails loudly and
   immediately — never silently coerces or guesses.
3. **Everything that can be versioned, is versioned.** Prompts, playbooks, generation-graph code,
   eval golden sets, and schema definitions all carry a version id. Every generated content item
   records exactly which versions produced it. This is what makes debugging a bad batch possible
   after the fact instead of "I think it was the new hook prompt, not sure."
4. **Idempotency and immutability over cleverness.** Ingestion is idempotent (dedupe by platform post
   id + content hash). Performance data is an append-only event log, never overwritten in place — you
   need the full time-series to compute decay/velocity, and you need history to debug "why does this
   number look different than last week."
5. **Separate exploration from exploitation, on purpose.** A system that only ever regenerates what
   worked last time will mode-collapse onto a local optimum and stop discovering anything new. Treat
   batch generation as an explicit explore/exploit allocation (e.g., 70/30 split, or a bandit policy),
   not purely "copy the winners."
6. **Statistical discipline before trusting a pattern.** An "insight" is a hypothesis until it clears
   a minimum sample size and significance bar. Mined patterns are tiered: `proposed` (just mined,
   unproven) → `testing` (actively being A/B'd against the status quo) → `accepted` (cleared
   significance, promoted into the live playbook) → `retired` (performance has decayed, platform/
   audience shifted). Never let `proposed` skip straight into the generation prompt.
7. **Human-in-the-loop gates shrink over time, they don't start at zero.** Every automated decision
   that writes to a playbook or spends money starts with a human approval step. As confidence in a
   given gate's reliability is earned (tracked explicitly — see §5.4), graduate it to auto-approve
   with monitoring, never the reverse.
8. **Systematic debugging over guessing.** When a batch underperforms, follow root-cause process (see
   §6) — reproduce, trace, form ranked hypotheses, test one variable at a time — rather than
   immediately blaming "the hook" or "the algorithm" and re-rolling.
9. **Fail loud, fail cheap, fail early.** Validate inputs at ingestion, not three stages downstream.
   Prefer a pipeline that halts a batch with a clear error over one that silently produces 30 broken
   variants you discover only after spending ad budget on them.
10. **Build the smallest end-to-end slice first (tracer bullet), not every stage in parallel.** One
    workflow, one platform, one full loop closed end-to-end, before breadth.

---

## 2. The four engine pillars

The user's framing — reading, learning, writing, design — maps directly onto four subsystems. Each
is built as an independently testable LangGraph subgraph with its own input/output contract, its own
eval harness, and its own owner-of-truth data tables. They share one substrate (§3) but should never
be so tightly coupled that you can't unit-test "Write" without a live "Read" pipeline running.

### 2.1 READ — the research/ingestion engine
**Purpose:** acquire raw material for both halves of the loop — our own published content's
performance, and external content for angle/pattern discovery.

**Two lanes, kept structurally separate (different trust levels, different schemas-of-origin):**
- **Owned-performance lane**: pull our published content_items' engagement/ad metrics on a schedule.
  Source of truth for everything the Learn engine mines.
- **External-research lane**: pull competitor/adjacent-creator public content for a lower-confidence,
  "hypothesis surfacing" use only (see §2.2's confidence tiering) — never merged into the
  high-trust owned dataset.

**Contract (output):** a validated `RawContentRecord` — platform, external_id, content_type, raw
asset ref(s), raw metric snapshot, source_lane (owned|external), pulled_at. Nothing downstream reads
anything less structured than this.

**Testing strategy:**
- Unit test the platform-API/provider adapters against recorded fixture responses (don't hit live
  APIs in CI) — assert correct parsing, correct handling of pagination, correct handling of partial/
  missing fields.
- Idempotency test: running ingestion twice on the same window produces zero duplicate rows.
- Contract test: adapter output always validates against `RawContentRecord`; a provider API schema
  change should break this test loudly, not corrupt the DB silently.
- Fixture-based "poison input" tests: malformed API responses, rate-limit errors, partial outages —
  assert graceful degradation (retry/backoff/dead-letter), not pipeline crash.

**Debugging hooks:** every ingestion run logs a structured summary (records pulled, records rejected
+ reason, provider latency, retry count) keyed by run_id, queryable later without re-running anything.

### 2.2 LEARN — the pattern-mining / insight engine
**Purpose:** turn `RawContentRecord` + performance history into tiered, versioned, *trustworthy*
insights, and compile accepted insights into a playbook diff.

**Pipeline stages:**
1. **Multimodal annotation** — every content item (video or static) gets decomposed into structured
   `CreativeDNA` fields (hook type, hook text, visual-hook description, format tag, script structure,
   CTA type, audio/track id, caption style, pacing, social proof used). This is a model call (vision-
   capable LLM) with a strict output schema — reject and retry on schema-invalid output rather than
   accepting free text.
2. **Correlation / pattern mining** — statistical pass (not a single LLM vibe-check) correlating
   `CreativeDNA` fields against outcome metrics, segmented by workflow_type and by owned vs external
   lane. Output: candidate insight statements with effect size, sample size, and a significance
   measure attached — e.g., "hooks tagged visual_pattern_interrupt have X% higher save-rate than
   talking-head opens, n=47, p=0.03." Insights without enough sample size are explicitly logged as
   `insufficient_data`, not discarded — they re-enter next cycle's pool.
3. **Insight tiering** — see principle 6 above. Movement between tiers is a logged state transition,
   not an in-place update, so the full history of "we thought X, then disproved it" is auditable.
4. **Playbook diff authoring** — accepted insights get compiled into a structured diff against the
   current playbook version for a given workflow_type. The diff is human-readable (this is what gets
   shown to you for approval) and machine-readable (this is what actually patches the Hermes skill).

**Contract (input):** `CreativeDNA` rows + `PerformanceSnapshot` time series + lane tag.
**Contract (output):** `Insight` rows (tiered) and `PlaybookDiff` objects.

**Testing strategy:**
- Annotation schema-validation tests with a battery of real sample videos/images — assert the model
  reliably returns valid structured output; track schema-failure rate as a monitored metric, not
  something you only notice when it breaks badly.
- **Golden eval set for annotation quality**: a small, human-labeled set of content items with
  "ground truth" hook/format tags. New annotation-prompt versions must match the golden set above a
  threshold before replacing the production annotation prompt — this is a regression test for a
  prompt, same discipline as a regression test for code.
- Statistical-mining unit tests on synthetic data with known-injected correlations — assert the
  mining stage actually recovers the known signal and does NOT flag noise as significant (false
  positive rate check using random/shuffled labels — if the miner "finds a pattern" in shuffled data,
  the significance test is broken).
- Playbook-diff tests: given a fixed set of accepted insights, assert the diff output is deterministic
  and correctly formatted for the Hermes skill patch target.

**Debugging hooks:** every `Insight` row stores its full provenance — which content_items, which
annotation-prompt version, which mining-run id produced it — so if a playbook update turns out wrong,
you can trace it back to the exact inputs instead of re-deriving blind.

### 2.3 WRITE — the copy/script generation engine
**Purpose:** generate hooks, scripts, headlines, captions, CTAs — the verbal/textual layer — informed
by the current playbook, allocated across explore/exploit.

**Pipeline stages:**
1. **Ideation** — playbook-informed angle generation for a given campaign/product brief.
2. **Explore/exploit allocation** — decide, per batch, what fraction of variants are "safe" (close to
   proven playbook patterns) vs "exploratory" (deliberately testing an unproven or `proposed`-tier
   insight, or a completely novel angle). This ratio is an explicit, tunable parameter, not emergent.
3. **Hook×body matrix expansion** — generate the N×M variant grid for the batch.
4. **Self-critique / rubric scoring pass** — score each draft variant against the structured playbook
   rubric before it leaves the Write engine (cheap, fast, catches obviously-bad output before it even
   reaches synthetic-audience scoring).

**Contract (input):** campaign brief (product, offer, constraints) + active `Playbook` version +
explore/exploit ratio parameter.
**Contract (output):** `ScriptVariant` rows — hook_text, body_text, cta_text, tags, which playbook
version/prompt version produced it, explore_or_exploit flag.

**Testing strategy:**
- Golden eval set of (brief → expected-quality-characteristics) pairs, scored by an LLM-judge
  calibrated periodically against human review — run on every prompt change before deploy, track
  score deltas, block deploy on regression past a threshold.
- Diversity test: assert a batch of N variants isn't just N near-duplicates (embedding-similarity
  check across the batch) — duplicate variants waste generation budget and corrupt the
  explore/exploit signal downstream.
- Constraint-adherence tests: required elements (brand name, legal disclaimers, banned claims) are
  present — deterministic string/regex checks, not just vibes.
- Explore/exploit ratio test: assert actual output matches configured ratio within tolerance.

**Debugging hooks:** every `ScriptVariant` traces back to brief + playbook version + prompt version +
explore/exploit flag, so a bad batch can be isolated to "the exploit lane regressed" vs "the whole
batch is bad" vs "this specific prompt version has a problem."

### 2.4 DESIGN — the visual/asset generation engine
**Purpose:** turn a `ScriptVariant` into an actual asset — video edit (for Reels/TikTok) or static/
carousel image (for IG statics) — including audio selection and pacing/visual-hook execution.

**Pipeline stages (branches by workflow_type):**
- Video lane: template/avatar/UGC-source selection → visual pattern-interrupt placement (the first
  1-3 seconds, called out repeatedly in field notes — see Appendix A — as disproportionately
  important) → caption/on-screen-text overlay → audio/track selection → assembly/render.
- Static lane: headline/copy placement → layout/template selection → design-system/brand consistency
  check → render.

**Contract (input):** `ScriptVariant` + workflow_type + asset constraints (brand kit, aspect ratio,
duration limits per platform).
**Contract (output):** `ContentItem` with rendered asset ref + full `CreativeDNA` populated (so no
duplicate annotation work is needed later — this engine already knows its own hook type, format tag,
etc., because it designed it).

**Testing strategy:**
- Render smoke tests: every asset meets hard platform constraints (duration, aspect ratio, file size,
  caption readability/contrast) before it's eligible for publish — automated checks, not manual
  inspection per item.
- Visual-hook-presence test: assert every video asset actually has *something* in frame in the first
  1-3 seconds that isn't a blank/logo card (this is the single most-cited failure mode in field notes
  — a pipeline producing visually-dead opens is a design-engine bug, catch it mechanically).
- Brand-consistency test: automated check against brand kit (colors/fonts/logo placement) for statics.
- A/B render-determinism test: same `ScriptVariant` + same seed produces the same asset, so render
  bugs are reproducible, not flaky.

**Debugging hooks:** render pipeline logs every asset-assembly decision (which template, which stock/
audio asset id, which overlay) tied to the ContentItem id.

### 2.5 Pre-flight: synthetic-audience scoring (cross-cutting, sits between Write/Design and Publish)
Before a batch spends any money or organic slot, score it against a persona panel (ICP-based synthetic
reviewers) for predicted reception. This is a filter/rank step, not a pipeline stage owned by one
pillar — it reads from both Write and Design outputs.

**Testing strategy:** calibration test — periodically back-test persona-panel predictions against
actual outcomes for already-published content; track prediction accuracy as a first-class monitored
metric. If calibration drifts (panel predictions stop correlating with real outcomes), that's a
trigger to refresh personas, not silently keep trusting them.

---

## 3. Shared substrate

### 3.1 Data model (summary — unchanged in shape from v1, confirmed sound on review)
- `content_items` — id, workflow_type, platform, status, batch_id, playbook_version_used,
  prompt_versions_used (json: {ideation, generation, annotation}), lane (owned|external),
  created_at, published_at, asset_url(s)
- `creative_dna` — one row per content_item, structured fields per §2.2/§2.4
- `performance_snapshots` — append-only event log: content_item_id, ts, views, likes, saves, comments,
  shares, follows_gained, retention/watch-time if available
- `ad_performance` — content_item_id, spend, CPM, CPC, ROAS, attribution_source, conversions
- `experiments` — batch_id, hypothesis, explore_exploit_ratio, variant_list, workflow_type, outcome
- `insights` — id, statement, tier (proposed/testing/accepted/retired), effect_size, sample_size,
  p_value_or_equivalent, supporting_data_refs, applies_to_workflow_type, state_history (json log)
- `playbooks` — workflow_type, version, rubric_json, generated_at, source_insight_ids, approved_by,
  approved_at
- `eval_runs` — stage (annotation/write/design), golden_set_version, prompt_version, score, pass/fail,
  run_at — this table is what makes "did this prompt change regress quality" answerable with data,
  not memory.

### 3.2 Versioning discipline
Every mutable artifact (prompt text, playbook, eval golden set, schema) gets a monotonic version id
and a changelog entry. Nothing is edited in place — a "change" is always a new version plus a pointer
update, so any `content_item` can always resolve exactly which versions of everything produced it.

### 3.3 Orchestration role split (not video-derived — pure tool-fit reasoning)
- **n8n**: scheduled polling, webhook receivers, publish-out integrations, retry/backoff on flaky
  third-party calls, cross-system notifications. Triggers LangGraph runs over HTTP; does not contain
  reasoning logic itself.
- **LangGraph**: every pipeline in §2 as a graph/server endpoint, each independently invokable and
  independently testable (this is why the contracts in §2 matter — they're also the graph's unit-test
  seams).
- **Hermes**: cockpit for ad hoc commands, the human-approval checkpoint for playbook diffs
  (`skill_manage(action='patch')` against per-workflow playbook skills), and cron-scheduled digest/
  reporting jobs.

---

## 4. Eval & testing strategy (first-class, not an afterthought)

This section exists because "testing and debugging principles" was an explicit requirement — treat it
as load-bearing, not boilerplate.

1. **Golden sets per pillar.** Read (fixture API responses), Learn (synthetic data with known
   correlations + human-labeled annotation samples), Write (brief→quality-characteristic pairs),
   Design (asset constraint checks). Each golden set is versioned; growing it is ongoing work, not a
   one-time setup task.
2. **LLM-as-judge, calibrated.** Where human-equivalent judgment is needed (is this hook good), use an
   LLM judge, but periodically sample its scores against actual human review and actual downstream
   performance to catch judge drift — never trust an uncalibrated judge indefinitely.
3. **Regression gates before any prompt/playbook ships.** A new prompt or playbook version must clear
   its golden-set threshold before it replaces the production version. Treat this exactly like CI for
   code — no merge without green tests.
4. **Shadow / canary before full cutover.** A new playbook version runs alongside the current one on a
   small slice of a batch first (explicit A/B, randomized assignment, not "last week vs this week"
   which confounds on time/seasonality/audience drift). Only promote to 100% after the canary clears
   significance.
5. **Statistical significance bar for insight promotion.** Defined in §2.2 tiering — do not let
   pattern-mining noise reach the playbook without clearing a sample-size/significance threshold.
6. **Monitor for drift, not just regression.** Track prediction-accuracy of synthetic-audience scoring
   and annotation schema-failure rate over time as first-class metrics — platforms change algorithms,
   audiences change tastes, and a playbook that was right in month 1 can quietly go stale by month 4.

---

## 5. Debugging & observability strategy

1. **Full provenance tracing.** Every `content_item` resolves to: which brief, which playbook version,
   which prompt versions (ideation/write/design/annotation), which explore/exploit bucket, which batch.
   This is the tight feedback loop that systematic-debugging requires — when a batch underperforms,
   you can reproduce "what exactly produced this" without guessing.
2. **Structured run logs at every stage boundary**, keyed by run_id: inputs received, outputs
   produced, rejects + reasons, latency, retry count. This is what lets you answer "where did it
   break" in a multi-component system without adding ad hoc print statements after the fact.
3. **When a batch underperforms, follow root-cause process, not a guess:**
   - Reproduce: can you re-run the exact pipeline with the exact versions that produced the batch?
   - Trace: walk the provenance chain — was it the brief, the playbook, the specific prompt version,
     the render step, or the publish/targeting step?
   - Form ranked hypotheses before changing anything; test one variable at a time (one playbook
     change, re-run, compare — not five simultaneous changes).
   - If 3+ hypotheses fail to explain it: stop and question whether the architecture/approach itself
     is wrong (e.g., maybe the signal doesn't exist in the data you have) rather than continuing to
     patch around it.
4. **Dead-letter queues, not silent drops**, for every n8n integration point (failed ingestion,
   failed publish, failed webhook) — a human-reviewable queue beats a silently-lost batch.
5. **Alerting on pipeline health, separate from alerting on content performance.** A digest about
   "this week's winning hooks" and an alert about "ingestion has failed 3 runs in a row" are different
   concerns and should not be mixed into one notification channel.

---

## 6. Phased roadmap (tracer-bullet first, breadth later)

### Phase 0 — Substrate + harness (no generation yet)
- Stand up Postgres + object storage; migrate schema from §3.1.
- Stand up LangGraph server + n8n instance (Docker locally to start).
- Build the `eval_runs` table and the first golden sets (even small, 10-20 examples each) for
  annotation and write quality *before* building the generators themselves.
- Build the provenance/versioning scaffolding (version ids on prompts/playbooks from day one — this
  is much harder to retrofit than to build in from the start).
- Seed Hermes skill(s) for each planned workflow's playbook, versioned v0 — can draw on field notes in
  Appendix A as a starting hypothesis set, explicitly tagged `tier: proposed, source: external_field_notes`
  so it's clear these haven't been validated on your own data yet.

### Phase 1 — Tracer bullet: ONE workflow, ONE platform, full loop, no learning yet
- Pick one workflow_type (recommend starting with whichever has the simplest asset-generation step —
  likely IG statics, since the Design engine for statics is lower-complexity than video, letting you
  validate Read→Write→Design→Publish→Read-back plumbing before tackling video assembly).
- Build Read (owned-lane only), Write, Design, and a manual-trigger publish step for this one
  workflow. No synthetic-audience scoring, no automated mining yet.
- Exit criteria: a batch can be generated, pass its eval-harness gate, get published, and have real
  performance data land back in `performance_snapshots`, fully traceable end-to-end, within one week.
- This phase proves the contracts in §2 actually hold under real data, not just on paper.

### Phase 2 — Add scoring + first real (tested) insight
- Build synthetic-audience pre-flight scoring; back-test its calibration against Phase 1's real
  outcomes before trusting it going forward.
- Build the Learn pipeline's mining stage against accumulated Phase 1 data; run the false-positive
  check (shuffled-label test) before trusting any output.
- Produce one real `insight`, carry it through the full tier lifecycle (proposed→testing→accepted),
  and author one real playbook diff, reviewed and approved by you.
- Exit criteria: one playbook update traced to real data, not the v0 seed, with its promotion decision
  backed by a documented significance check.

### Phase 3 — Explore/exploit + canary discipline
- Formalize the explore/exploit ratio in the Write engine.
- Implement shadow/canary testing for playbook version rollout (§4.4) rather than hard cutovers.
- Exit criteria: at least one playbook version has gone through canary → promotion, with the canary's
  comparison data retained for audit.

### Phase 4 — External research lane
- Stand up the external/competitor ingestion lane (§2.1), kept structurally separate and lower-trust.
- Feed into Learn as `lane=external` insights, explicitly tiered lower-confidence.
- Exit criteria: one externally-sourced hypothesis has been tested via the explore lane and its
  real outcome recorded — proving the lane actually produces validated learning, not just scraped
  noise.

### Phase 5 — Second workflow (video — TikTok/Reels)
- Reuse all shared substrate; build the workflow-specific Design-engine video lane (visual-hook
  placement, audio selection, render assembly) and its asset-specific eval/testing additions (visual-
  hook-presence check, render-determinism check).
- Exit criteria: same tracer-bullet exit criteria as Phase 1, for the new workflow.

### Phase 6 — Automate cadence + cross-workflow insight mining
- Put mining/playbook-authoring on a schedule (weekly to start) with human approval gate retained.
- Extend mining to look for patterns that hold *across* workflow_types, not just within one.
- Revisit annotation-cost/sampling strategy as volume grows — full-annotate recent/outlier content,
  sample the long middle.

### Phase 7 — Graduate gates based on earned trust
- For any human-approval gate that has a track record of ≥N consecutive correct human-aligned
  decisions with no overrides, propose (to you) graduating it to auto-approve-with-monitoring.
  Never graduate a gate unilaterally — this is itself a reviewed decision.

---

## 7. Risks & open questions

- **API/ToS access** for both owned-analytics pull and external-research ingestion — resolve data
  provider choices in Phase 0; prefer official APIs or vetted providers over raw scraping.
- **Attribution reliability** for paid-ad ROAS — platform self-reported attribution is known-unreliable;
  budget for a clean-attribution layer once ad spend is material, before trusting ROAS-based insights.
- **Annotation cost at volume** — full multimodal annotation of every item is not free; the sampling
  strategy in Phase 6 needs real cost data from Phases 1-2 to size correctly.
- **Judge/calibration drift** — any LLM-judge or synthetic-audience panel needs periodic recalibration
  against real outcomes; build this into the schedule, don't treat it as one-time setup.
- **Schema commitment** — the shared data model is the hardest thing to change later; a second review
  pass on §3.1 before writing migrations is worth the time.
- **IP/compliance on external research** — the external lane is for angle/pattern inspiration only;
  the generation engines must transform, not copy, anything derived from competitor content.
- **Minimum viable sample size** — don't expect meaningful automated insights before Phase 1-2 have
  accumulated enough batches; lean on the explicitly-tagged `proposed, source: external_field_notes`
  seed playbook for early batches, and treat it with appropriate skepticism until validated.

---

## 8. Immediate next actions
1. Confirm Phase 0 tooling choices: DB/storage (Supabase vs raw Postgres+S3), annotation model
   (needs vision capability), owned-analytics API access per platform, external-research provider,
   hosting for n8n/LangGraph (local Docker vs cloud).
2. I can draft the Postgres DDL (§3.1) and the Pydantic contract models (§2's per-pillar
   input/output schemas) immediately — no external dependencies, unblocks everything else.
3. Stand up a throwaway golden-set + eval-harness skeleton (even 10 hand-picked examples) before
   writing a single generation prompt — this is the principle-1 "build the harness first" requirement
   made concrete.
4. Pick the Phase 1 pilot workflow (recommend IG statics for lowest Design-engine complexity) and
   confirm before building anything.

---

## Appendix A — Field notes from external practitioners (context only, not architecture)

These come from two YouTube videos the user flagged as context: an ad-agency operator describing a
self-learning Meta-ads content pipeline, and a solo operator running an AI-driven ecommerce launch.
Treat every item below as an **unverified hypothesis worth testing against your own data**, not as a
design requirement. None of it should be hard-coded as "accepted" in any playbook — it enters at
`tier: proposed, source: external_field_notes` per §6 Phase 0, and only earns playbook-acceptance
through the same statistical process as internally-mined insights.

**From the ad-agency / self-learning-ads video:**
- Hypothesis: content that doesn't read as "an ad" outperforms traditional direct-response-formula ads.
- Hypothesis: saves and follows are leading indicators of ad-readiness, more predictive than raw views.
- Hypothesis: the hook is frequently visual/on-screen in the first few seconds, not the spoken opening
  line — an annotation pipeline that only transcribes audio will systematically miss this.
- Observed practice: batch generation as a hook×body matrix (e.g., 10 hooks × 3 bodies) per production
  session.
- Observed practice: layering a third-party attribution tool because platform-native attribution was
  judged unreliable for training a model.
- Observed practice (unvalidated by the source themselves — they note roughly half the AI-informed
  batch "flopped"): this is explicit evidence that unvalidated hypotheses from one small sample can
  fail at scale, reinforcing why §1 principle 6 (statistical discipline) matters rather than trusting
  any single practitioner's framework wholesale.
- Planned extension (by the source, not yet proven even by them): feeding in competitor content as a
  secondary, lower-weighted signal.

**From the AI-run-ecommerce video:**
- Observed practice: compiling decisions into a structured brief/PRD before handing off to an
  execution agent.
- Observed practice: competitor research performed before creative generation, informing positioning.
- Observed practice: building organic and paid distribution in parallel from day one rather than
  sequentially.
- Observed practice: when traffic came in but conversions didn't, the operator root-caused the
  offer/trust layer first (price, social proof, perceived legitimacy) and only after that found a
  genuine technical bug (slow-loading checkout) — a real-world illustration of why §5's
  root-cause-before-fix discipline matters, since two different causes were tangled together and
  guessing would have fixed the wrong one first.

**Why these are appendix material, not architecture:** both are single-operator, single-niche case
studies with small sample sizes by the sources' own account (one explicitly reports roughly 50% of
their AI-informed batch underperforming). They're useful as a source of testable hypotheses and as a
sanity check that the loop shape (ingest→annotate→mine→playbook→generate→publish) is a reasonable
industry pattern — but the actual engineering rigor (contracts, versioning, statistical gating, eval
harnesses, explore/exploit, canarying) in this plan is what should make Qualume's learning loop more
reliable than either of the described ad hoc systems, not a copy of them.
