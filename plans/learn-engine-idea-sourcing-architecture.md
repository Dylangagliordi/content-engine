# Learn Engine & Idea-Sourcing Architecture (Part 5 design)

Written 2026-10-12, revised 2026-10-12 (same day, after further design discussion — see
"Revision log" at the bottom). This turns roadmap Part 5 ("Learn engine — not started") into a concrete
design. Nothing here is built yet — this is the target to build toward, not a status report.
Supersedes the vague Part 5 placeholder in `plans/roadmap-master.md`.

## The problem this solves

Today, the Write engine's `ideate_angles` node is told: here's a brief (what we're promoting)
and a rubric (how to write well) — invent some angles. That's generation from a style guide, not
generation from evidence. Nothing connects a generated idea to anything actually observed to work
in the real world. This doc fixes that.

## Two things, kept deliberately separate (NOT two documents — see "Retrieval, not a static
## summary doc" below for why the second one is a live query, not a file)

1. **The Playbook** (exists today, `playbooks` table, v0→v2) — HOW to write: tone, copywriting
   mechanics, format rules, the claims-safety gate. Stays as-is.
2. **The structured insight store** (`insights` table, already exists, currently empty) — WHAT's
   actually happening: statistically-validated, tiered, segment-tagged claims mined from the
   Read+Learn engines. Queried live per-brief at generation time, not compiled into a static doc.
   Only becomes real once there's real data to mine (today: zero accounts watched).

## Schema additions needed

### `external_creative_dna` — external content's missing DNA table
`external_content_items` currently stores only raw metadata (caption, duration, audio track
name) — nothing structured about what the video *is*. Owned content has `creative_dna` for this;
external needs the same, kept structurally separate per the owned-vs-external trust principle,
plus fields owned content doesn't need (we chose our own tone, we don't need to detect it):

```sql
CREATE TABLE external_creative_dna (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_content_item_id UUID NOT NULL REFERENCES external_content_items(id),

    -- mirrors owned creative_dna's shape, for direct comparability
    hook_type                 TEXT,
    hook_text                 TEXT,              -- transcribed/observed, not authored
    visual_hook_description   TEXT,
    format_tag                 TEXT,
    script_structure_tags     TEXT[],
    cta_type                   TEXT,
    caption_style              TEXT,
    social_proof_used          BOOLEAN DEFAULT false,

    -- vibe fields, external-only (see "How AI watches a video" below)
    energy_level                SMALLINT,          -- 1-5, anchored scale
    tone_category                TEXT,              -- enum, not free text
    pacing_cuts_per_min          FLOAT,             -- COMPUTED, not LLM-guessed
    music_mood                   TEXT[],            -- tags, from audio analysis
    voice_delivery_style         TEXT,

    -- unproven/candidate features land here, NOT as new typed columns (see "Features are
    -- learned too" below) — a feature only gets promoted to a real column once its
    -- utility_score clears a threshold with real sample size behind it
    extended_features            JSONB DEFAULT '{}',

    -- provenance (same discipline as everywhere else in this codebase)
    analysis_model                TEXT NOT NULL,
    analysis_prompt_version       TEXT NOT NULL,
    analyzed_at                    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Features are learned too, not a fixed list decided today

The five vibe fields above (`energy_level`, `tone_category`, etc.) are a reasonable starting set,
not a claim that we already know what matters. Hardcoding a permanent feature list upfront repeats
the same mistake a hand-written-forever playbook would make. Two distinct things need to evolve
over time, not one:

- **Features** — the *dimensions* we measure (does "energy_level" exist as a tracked thing at all)
- **Insights** — *claims* about relationships between features and outcomes (the existing
  `insights` table already handles this half)

```sql
CREATE TABLE creative_dna_feature_registry (
    id                TEXT PRIMARY KEY,          -- e.g. 'energy_level'
    data_type          TEXT NOT NULL,             -- int/float/text/bool/text[]
    description         TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'proposed',  -- proposed → trial → stable → deprecated
    storage_location     TEXT NOT NULL DEFAULT 'extended',  -- 'extended' (jsonb) or 'column' (promoted)
    utility_score         FLOAT,                   -- latest effect-size summary across insights using it
    sample_size           INTEGER NOT NULL DEFAULT 0,
    proposed_by           TEXT NOT NULL,            -- 'human' or 'llm_discovery'
    added_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_evaluated_at      TIMESTAMPTZ
);
```

New candidate features come from a distinct Learn-engine sub-step, **feature discovery** — not
correlation mining (which needs features to already exist) but open-ended difference-spotting:
periodically ask the model something like "here are the top-20 and bottom-20 performing videos in
this niche by [metric] — what dimension differs between them that we are NOT currently capturing
in the DNA schema?" A proposed feature starts in `extended_features` (JSONB, cheap to try, no
schema churn); once pattern mining shows real `utility_score` with real sample size behind it, it
gets promoted to a proper typed column via a real migration — proof before commitment, not a guess
upfront, same discipline the rest of this project already uses for insight tiers.

### `watched_accounts.watchlist_type` — multiple watchlist support
Current table has no concept of *why* an account is tracked. One-column addition:

```sql
ALTER TABLE watched_accounts ADD COLUMN watchlist_type TEXT NOT NULL DEFAULT 'general';
```

Watchlist types to support: `competitor` (direct niche competitors), `cross_niche` (big accounts
in unrelated niches — catches patterns that transcend topic, e.g. a hook structure that works in
both fitness and finance content says something about hooks, not about fitness), `aspirational`
(top accounts in-niche but not direct competitors), `emerging` (fast-growing small accounts — what
works at 5M followers may not transfer to a near-zero account), `format_specific` (tracked for one
skill, e.g. "best at hooks," regardless of niche relevance).

### `insights` table — extend, don't replace (this is what feeds the Write engine, not a doc)
The `insights` table already exists with good bones: `tier`, `effect_size`, `sample_size`,
`p_value`, `supporting_data_refs`, `state_history`. It's missing segment granularity — today it
only has `applies_to_workflow_type`, nothing for niche/format/platform:

```sql
ALTER TABLE insights ADD COLUMN applies_to_segment JSONB DEFAULT '{}';
-- e.g. {"niche": "fitness", "platform": "tiktok", "format": "talking_head"}
```

No new "summary" table is needed — see "Retrieval, not a static summary doc" below for why a
flat document was the wrong shape for this, and what replaces it.

### Own-content performance needs a baseline — judge against OUR median, not raw numbers

**This is design reasoning from conversation, not a sourced research finding** — flagging that
explicitly after an earlier mischaracterization implied otherwise. A single post's raw
view/save/like count can't be judged in absolute terms — it has to be judged against what's
normal for OUR account specifically (same problem exists on the external-mining side too: a
video from a 5M-follower account and a video from a 2K-follower account need different baselines,
or pattern mining ends up just measuring "which accounts are big," not what makes content work).

The fix: compute a rolling median/stddev baseline per account (ours included) and judge each
post/video's performance as a z-score or percentile against THAT account's own recent normal —
not stored as a persisted column (would violate the append-only principle already in place for
`performance_snapshots`), computed fresh as a windowed aggregate query at mining/validation time.

**One post does not "prove" or "disprove" an insight on its own.** An insight is a claim about an
aggregate tendency, not a deterministic prediction for every instance — one data point barely
moves a real statistical estimate, and a single post carries confounds beyond the insight itself
(timing, that day's audience mood, Design-rendering quality, multiple insights combined in one
piece pulling different directions). A below-baseline post is ONE MORE DATA POINT into the
insight's running effect-size/sample-size/p-value recalculation — exactly the same mechanism
external mining already uses, just a second, smaller, higher-trust evidence pool.

### Two insights tables, not one — owned and external need different trust levels

Mirrors a principle already deeply baked into this project elsewhere (CLAUDE.md principle 5:
owned vs. external content never conflated, `content_items` vs. `external_content_items`). The
same split belongs on insights:

- **`insights_external`** — mined purely from watched-account data. Potentially high volume
  (hundreds+), capped at a lower trust ceiling even at `tier: accepted` — "this worked for other
  accounts" is not the same claim as "this works for us."
- **`insights_owned`** — mined from and tested against our own `content_items` +
  `performance_snapshots`, using OUR account's own median as baseline (see above). This is the
  one that actually drives generation decisions, and it can meaningfully DIVERGE from the
  external version of the same claim (accepted externally off a huge competitor sample, still
  `proposed` or even contradicted against our own data) — one merged table couldn't represent
  that divergence cleanly, two tables can.
- **Future, lower priority, not built now**: `insights_meta` — patterns about patterns (e.g. "hook
  insights have higher average effect size than CTA insights across the board") — a genuinely
  distinct kind of analysis, not urgent with zero data to support it yet.

Both tables keep the same schema shape (`tier`, `effect_size`, `sample_size`, `p_value`,
`applies_to_segment`, `state_history`) — it's the trust/provenance boundary that's structurally
separated, not the data model.

### How an external insight actually gets validated against our account — use `experiments`, don't invent a new mechanism

The `experiments` table already exists in this schema (`batch_id`, `hypothesis`,
`explore_exploit_ratio`, `workflow_type`, `outcome_summary`) and is the real, already-designed
mechanism for this — it was under-referenced in earlier drafts of this doc. An external insight
gets deliberately tested by: generating a real batch through `run_batch.py` with an explicit
`hypothesis` recorded (the insight being tested), tagging the resulting `content_items` via
`grounding_refs`, publishing, then rolling the real (baseline-normalized) performance back into
`insights_owned` as that hypothesis's first real evidence. This is "testing" in the literal
`tier: proposed → testing → accepted` sense — `testing` tier specifically means "an `experiments`
row exists for this and we're waiting on owned-account evidence," not just "waiting on more
external data."

### Closing the loop — your own results re-validate the insights that inspired them

```sql
ALTER TABLE content_items ADD COLUMN grounding_refs JSONB DEFAULT '{}';
-- e.g. {"inspired_by_video": "<external_content_item_id>", "insights_used": ["<insight_id>", ...]}
```

When a published piece's real performance comes back via `performance_snapshots`, that becomes
additional evidence FOR OR AGAINST the specific insight(s) that inspired it — not just external
data feeding insights one-way, but owned results re-validating (or invalidating) them too. This is
what makes the loop genuinely closed rather than a one-way pipeline from external data to content.

## How AI actually "watches" a video — researched, not guessed

Claude has no native video/audio understanding (confirmed earlier this project). The baseline
workaround (yt-dlp downloads the real video, ffmpeg extracts frames, transcript from captions or
Whisper, frames+transcript fed to one Claude prompt) works and is cheap (~$0.01-0.03/short clip)
but may under-capture *vibe* specifically — mood, pacing, energy — since it's a text+static-image
representation of something inherently audiovisual and temporal.

**Research findings (2026-10-12), each grounded in a real source, not speculation:**

- **DECIDED (2026-10-12): Gemini is the video-watching model for this stage.** Confirmed via
  Google's own docs (ai.google.dev/gemini-api/docs/video-understanding) that Gemini's native video
  understanding genuinely fuses audio+visual — not frames-then-transcript bolted together. Newer
  Gemini 3.x Flash models have an "agentic" mode that adaptively navigates the timeline instead of
  uniform sampling; Google's own claimed numbers: up to 88% fewer tokens, 66% lower cost, up to 7%
  higher accuracy vs static sampling. This is a **deliberate, scoped exception** to the
  project's "Claude only" rule — Gemini is used ONLY for the video-watching/annotation stage,
  combined with the computed signals below; every other stage (Write, Design, self-critique,
  pattern mining, playbook authoring) stays Claude-only as before. Update `CLAUDE.md`'s "Claude
  only (Anthropic) — no OpenAI/Gemini" line to carve out this one exception explicitly, so it
  reads as an intentional decision, not an inconsistency to "fix" later.
- **A near-exact precedent exists**: arXiv 2512.21402, "Understanding Virality: A Rubric-based
  Vision-Language Model Framework for Short-Form Edutainment Evaluation" (2026) builds essentially
  this pipeline — queries a VLM per video for 12-20 visual + 8-12 audio textual descriptors
  covering "motion dynamics, transitions, background music characteristics, pacing, and speaker
  tone," embeds them, clusters them to find patterns. Validates the whole approach is sound, not
  just the individual tools.
- **Pacing should be computed, not LLM-guessed.** `PySceneDetect` (github.com/Breakthrough/
  PySceneDetect) computes cuts-per-minute directly via ffmpeg scene-change detection — a mature,
  maintained library. Average-shot-length is an established decades-old film-analysis metric
  (cinemetrics), not something being invented here. More reliable than asking an LLM to eyeball
  pacing from sparse frames.
- **Music mood should be computed, not described in prose.** `beatlyze-analyzer` (PyPI) extracts
  BPM, energy, valence, danceability, loudness, mood tags directly from the audio waveform via
  real signal processing (RMS energy, spectral centroid, chroma) — positions itself as an open
  replacement for Spotify's deprecated Audio Features API. Heavier option if more rigor is wanted
  later: `Essentia` (MTG, mtg.github.io/essentia-labs) with pretrained TensorFlow mood/genre
  classifiers trained on Discogs-scale data.
- **Anchored numeric scales beat unanchored ones for LLM consistency.** A real example found (a
  creative-writing profiler tool) uses `pacing: 0-100` with explicit decile anchors ("0-20:
  extremely slow, meditative, philosophical" ... "81-100: maximum intensity"). Unanchored scales
  ("rate the energy 1-10") drift inconsistently run to run; anchored ones don't.

### Recommended pipeline shape (3 parts per video, not one prose description)

1. **Computed signals** (no LLM, cheap, deterministic): PySceneDetect → cuts-per-minute.
   beatlyze-analyzer on the extracted audio track → BPM, energy, valence, mood tags.
2. **Model-observed signals** (LLM, real judgment needed): Gemini, given the real video with
   native audio+video (decided above — not frames+transcript, which is the fallback if Gemini
   access/cost becomes a blocker) → structured JSON with anchored scales: `energy_level`
   (1-5, anchored), `tone_category` (enum, not free text), `voice_delivery_style`,
   `visual_hook_description`.
3. **Fuse into one `external_creative_dna` row** — computed numbers and the model's qualitative
   read live side by side, so later pattern-mining can correlate e.g. "high cut-rate + high-energy
   tone" against real view/save performance — a testable, falsifiable insight, not just "the AI
   said this felt energetic."

## Where ideas actually come from (replacing today's "invent from a style guide")

Two sourcing modes, both requiring the schema above to exist first:

### Mode A — Inspired by a specific high-performing video
Once a watchlist (or niche, or format-specific subset) has accumulated **≥20 analyzed videos**
(deliberate floor — fewer and you're copying one lucky video, not finding a pattern), a new graph
node queries `external_content_items` + `external_creative_dna` + `external_performance_snapshots`,
ranks by actual performance within the relevant scope, and selects a genuine top performer. That
video's `external_creative_dna` — its real hook structure, format, tone, NOT its literal content —
becomes grounding context fed into `ideate_angles`: "here is a real video that performed well, and
here's structurally what made it work — write an original angle for OUR product that applies the
same structural pattern." Retrieval-augmented ideation, not copying.

### Mode B — From live-queried structured insights (not a static summary doc)
The Learn engine continuously mines performance data into the `insights` table — cross-video
patterns ("hooks under 3s with a visible problem outperform scripted intros by X%"), gated through
the existing `tier: proposed→testing→accepted` discipline. At generation time, `gather_inspiration_
node` runs a live query — filtered by `applies_to_workflow_type` and `applies_to_segment` matching
THIS brief's niche/platform/format, ranked by `effect_size`/`sample_size`/recency, top-K results —
and injects those specific rows into `ideate_angles`. See "Retrieval, not a static summary doc"
below for the full reasoning behind this shape.

## Retrieval, not a static summary doc

A single flat "data overview" document (the original shape of this design, same-day revision) has
real problems once you think through how it's actually used: it compresses away nuance (niche,
format, confidence, sample size all flatten into prose), it's identical for every brief when what
actually matters at generation time is "what's relevant to THIS brief," it has no natural way to
rank by confidence or decay stale findings, and it isn't queryable.

The fix: the `insights` table (already built, with `tier`/`effect_size`/`sample_size`/`p_value`)
IS the source of truth — a live, per-brief filtered query replaces what would have been a static
doc. `gather_inspiration_node` becomes a real retrieval step, not a "read this file" step.

**On semantic retrieval**: pgvector is NOT currently installed in this project's Postgres
(confirmed via `pg_available_extensions` — not present in the `postgres:17` image as configured).
Start with tag/category-based filtering (the `applies_to_segment` JSONB above — niche, platform,
format) rather than adding a new extension before it's proven necessary, consistent with this
project's "don't build ahead of need" principle. If tag-based filtering proves too coarse once
there's real volume, that's the deliberate point to evaluate pgvector — not before.

**A human-facing report still has value — it just isn't what Claude is fed.** A readable
dashboard/digest, rendered FROM the structured `insights` table on demand (or via a periodic
Hermes cron job), gives visibility into what the system has learned without that artifact being
load-bearing for generation. Structured rows are the source of truth; a doc, if one exists, is a
rendering of them — not the reverse.

### Graph change required
A new node, `gather_inspiration_node`, runs before `ideate_angles` in the Write graph. It
populates `state.inspiration_source` (a specific external video reference, Mode A) and/or
`state.retrieved_insights` (a list of matched `insights` rows, Mode B) — likely both together once
enough data exists. The ideation prompt changes from "invent angles for this brief" to "here's
grounding evidence — invent angles for this brief that apply what's actually been observed to
work."

## Statistical method for mining — a real decision, not left vague

Simple per-feature tests FIRST, not regression or random forest, at least initially:
- **Welch's t-test / Mann-Whitney U** for a continuous outcome (views, saves) split by a
  binary/categorical DNA field (e.g. `energy_level=high` group vs `energy_level=low` group).
- **Chi-squared** for categorical-vs-categorical association (e.g. does `hook_type=question`
  correlate with higher completion-rate bucket than `hook_type=statement`).
These are simple enough to unit-test against synthetic data with known injected correlations
(the false-positive guard already planned in roadmap 5b), and they produce exactly the clean
effect-size-with-p-value statement the `insights` tables are shaped to store.

**Regression is a deliberate later upgrade**, not needed on day one: becomes genuinely useful once
there are enough continuous features and enough volume to control for confounders simultaneously
— e.g. "does pacing matter independent of video length and niche" (a real question raised in
design discussion 2026-10-12) needs length/niche as covariates, which simple per-feature tests
can't do alone.

**Random forest (or any ML prediction model) is the wrong tool for INSIGHT GENERATION specifically**
— it gives feature importances, not a clean interpretable "X causes Y, here's the effect size and
confidence" statement, and needs far more data than this project will have for a long while to
avoid overfitting on noise. It may show up later for a DIFFERENT job — predicting a new draft's
likely performance before publish — but that's prediction, not insight-mining, and is explicitly
out of scope for this doc.

## Planner vs. do-er — an ad hoc Analyst sub-agent, reusing a pattern this project already knows

Raised in design discussion 2026-10-12: the periodic pattern-miner above only answers questions it
was built to ask on a schedule. A real question like "does pacing matter independent of length and
niche" needs something that can go run the right statistical test on demand. This maps directly
onto the **SQL Analyst pattern** from the LangGraph tutorial summarized early in this project
(stateful orchestration: curate the raw question -> inject schema/context -> generate a query ->
safety-check it -> execute -> represent the result in plain language) — nearly exactly suited to
this job, reusing a pattern this project has real tested understanding of rather than inventing
something new.

Three distinct roles emerge, not one monolithic Learn engine:
- **Periodic pattern miner** (5b, scheduled, mostly deterministic) — finds new candidate insights
  via the t-test/chi-squared process above.
- **Ad hoc Analyst** (new, SQL-Analyst-shaped) — answers specific on-demand questions, including
  ones that want real regression with covariates (the pacing-independent-of-length-and-niche case
  is a perfect example of when this sub-agent, not the scheduled miner, is the right tool).
- **Feature discovery** (already designed above) — open-ended difference-spotting for uncaptured
  dimensions.

Correctly NOT built now — zero data exists yet, nothing to analyze. Capturing the shape so it's
not re-derived from scratch later.

## Checklist enforcement — how an accepted insight actually becomes a rule, not just prose

**Grounding, flagged honestly**: this section was prompted by a single operator's podcast Q&A
anecdote (summarized/transcribed 2026-10-12, not a published paper or tested framework) describing
a "make 100, see what worked, add 2 checklist items, repeat" loop. Treat the general shape as a
plausible hypothesis worth adapting, not a proven methodology to copy verbatim — same discipline
as the original two reference videos in Appendix A.

The gap this closes: "5c: tier + author playbook" was previously vague about HOW an accepted
insight becomes an actual rule — it implied "add a sentence of prose to the rubric and hope Claude
follows it." This project already has a stronger precedent for exactly this problem: the
anti-fabrication fix didn't add advisory prose, it added a deterministic `unsupported_claims` check
that OVERRIDES the model's own judgment. A checklist item is only as good as whether it's actually
checked, not just suggested — so every accepted insight gets classified by how it's enforced, not
just pasted into the playbook as another instruction:

- **`eval_check`** — if the insight is mechanically checkable (e.g. "hooks under N words
  outperform longer ones"), it becomes a new deterministic function in the eval harness
  (`src/qualume/eval/harness.py`), same pattern as the existing claims gate. Hard pass/fail,
  never advisory.
- **`design_constraint`** — if it's structural/layout (e.g. "product visible in first 2 seconds"),
  it becomes a rule inside the Design engine's template logic directly — Design is already
  deterministic plain Python, so this is adding a constraint there, never relying on an LLM to
  remember it at generation time.
- **`prompt_guidance`** — only genuinely unenforceable, judgment-based insights (tone, vibe,
  "feels right") fall back to prose in the system prompt. Weakest tier, last resort, not the
  default.

Each generated checklist item stores `derived_from_insight_id` (traceable back to exactly which
insight produced it — same provenance discipline as everything else here) and gets its own
pass-rate tracked after it's added, reusing the before/after measurement pattern already proven
out on the retry loop (73%→100%, not vibes). An item that always trivially passes may be
redundant; one that keeps failing either needs a stronger enforcement tier or should be
reconsidered as not actually true.

**Cap the batch size of new items per playbook version** — the transcript's "next hundred, add two
points" (not fifty) taken literally: bumping too many checklist items into one playbook version at
once makes it impossible to isolate which addition actually caused a measured change, the same
confound-avoidance reasoning already applied to the insight-validity critic itself. A small,
explicit cap (e.g. 2-3 new enforced items per version bump) keeps each version's effect
measurable.

## Explore/exploit cycling ("the accordion") — a real policy for an existing unused field

**Grounding, same caveat as above** — this is adapted from the same single podcast anecdote
("you just keep going back and forth between better and more... it's a very normal cadence"),
treated as a hypothesis to adapt and test, not copy as proven fact.

`experiments.explore_exploit_ratio` already exists in the schema but nothing currently sets or
uses it — it's a real field with no real policy behind it yet. The anecdote maps directly onto a
concrete, implementable rule:

1. **Explore phase** (many cheap variants, lower per-piece effort): default when a focus area
   (niche/format/segment) has thin `insights_owned` evidence — low `sample_size`, no `accepted`
   tier yet. Wide `explore_exploit_ratio`, e.g. mostly-explore.
2. **Exploit phase** (fewer, higher-effort variants refining a winning pattern): triggered once
   `insights_owned` for that focus area shows strengthening `effect_size`/`sample_size` (i.e. a
   pattern is emerging, not yet fully proven) — shift the ratio toward exploit, concentrate batch
   effort on refining that specific pattern rather than spreading thin.
3. **Reset to explore**: once exploit-phase marginal gains flatten (measured the same way the
   retry-benefit comparison measured "worth it or not" — before/after numbers, not a guess), shift
   back toward explore for that focus area. The cycle repeats, not a one-time switch.

Implementation shape: a new function, `determine_explore_exploit_ratio(focus_area) -> float`,
reads directly off `insights_owned` for that focus area and returns the ratio; `run_batch.py` calls
it when creating a new `experiments` row instead of a hardcoded/manual value. This is the first
concrete link between the Learn engine's statistical output and how much volume/effort the Write
engine actually spends per batch — previously those two sides of the system had no real
connection.

**Correctly not built now** — needs real `insights_owned` data to have anything to key off of;
captured here so the policy is designed before the field is ever touched, not invented ad hoc
later.

## Build sequencing (depends on, in order)

1. TikTok watchlist actually populated (Part 2, already unblocked — zero accounts today)
2. `external_creative_dna` (incl. `extended_features`) + `creative_dna_feature_registry` +
   `watchlist_type` + split `insights` into `insights_external`/`insights_owned` (both with
   `applies_to_segment`) + `content_items.grounding_refs` schema migration
3. The video-watching pipeline: Gemini (model-observed signals) + PySceneDetect/beatlyze-analyzer
   (computed signals) → fused `external_creative_dna` row
4. 20+ videos analyzed in at least one watchlist/niche before Mode A can activate
5. First `insights_external` rows once there's enough analyzed volume for cross-video mining
   (simple per-feature tests — t-test/chi-squared — not regression/random forest, see "Statistical
   method for mining" above) — Mode B activates here. Feature-discovery sub-step can start running
   in parallel once there's any meaningful volume, independent of whether Mode A/B are live yet.
6. `gather_inspiration_node` added to the Write graph, wired to both modes
7. Re-validate: does grounded ideation actually produce better content than today's ungrounded
   version? Needs a real A/B comparison (same discipline as the retry-loop and claims-gate
   verdicts already done this project) before trusting it, not assumed.
8. Once real owned content has been published and gotten performance data back: wire the
   `grounding_refs` feedback loop, judged against OUR account's own rolling median baseline (not
   raw numbers), so published results accumulate into `insights_owned` and re-validate (or weaken)
   the specific external insights that inspired them — via a real `experiments` row with a stated
   hypothesis, not an ad hoc one-off check. `determine_explore_exploit_ratio` starts getting real
   signal to key off of at this point too.
9. Once `insights_owned`/`insights_external` have `accepted`-tier rows: route each one through
   the checklist-enforcement classification (`eval_check` / `design_constraint` /
   `prompt_guidance`) rather than defaulting everything to prose — build the eval-harness/
   design-engine hooks for the first `eval_check`/`design_constraint` items when they actually
   exist, not speculatively now.

## Open questions, not resolved here

- Exact `tone_category` enum values and `energy_level` anchor text — draft during actual
  implementation against real sample videos, not invented abstractly now.
- Whether Mode A and Mode B run together on every batch or are selected contextually — likely
  "both, weighted," but undecided until there's real data to test against.
- Exact `utility_score` formula and promotion threshold for `creative_dna_feature_registry` —
  needs real insight data to calibrate against, can't be meaningfully set abstractly now.
- Cadence for the feature-discovery sub-step (every N new videos? scheduled? triggered by
  mining-stage stagnation?) — decide once there's real volume to observe a cadence against.
- Exact baseline window for own-account median (trailing N posts? trailing N days? both?) — needs
  real publishing cadence data to pick sensibly, can't be set abstractly now.
- When does the ad hoc Analyst sub-agent actually get built — likely well after the scheduled
  miner and feature-discovery are already running and producing real questions worth asking it,
  not before.

## Revision log

- **2026-10-12 (initial)**: first draft — fixed 5-field vibe schema, "Data Overview" as a
  periodically-regenerated static document, Gemini flagged as "worth testing."
- **2026-10-12 (same-day revision)**: replaced the static Data Overview document with live
  retrieval against the existing `insights` table (`applies_to_segment` addition) — a document
  can't be ranked/filtered per-brief the way a structured query can. Added
  `creative_dna_feature_registry` + `extended_features` so the feature set itself can grow via a
  feature-discovery sub-step, not stay frozen at today's 5-field guess. Confirmed Gemini as the
  decided video-watching model (scoped exception to Claude-only), not an open question. Added
  `content_items.grounding_refs` to close the loop from owned-content performance back into the
  insights that inspired it.
- **2026-10-12 (third revision, same week)**: added own-account median-baseline normalization for
  judging owned-content performance (explicitly flagged as design reasoning from conversation, NOT
  a sourced research finding — corrected an earlier ambiguous attribution). Split `insights` into
  `insights_external` and `insights_owned` (different trust levels, can diverge on the same claim)
  plus a noted future `insights_meta`. Tied owned-insight validation explicitly to the existing
  `experiments` table instead of inventing a new mechanism. Decided the mining statistical method
  (simple per-feature tests — t-test/chi-squared — first; regression as a later upgrade for
  confounder-controlled questions; random forest ruled out for insight-generation specifically,
  noted as possibly useful later for a different job — performance prediction). Added the
  planner/do-er split (periodic miner vs. ad hoc Analyst vs. feature discovery), tied to the SQL
  Analyst LangGraph pattern already understood in this project. Corrected the `m13v/
  social-autoposter` repo characterization in docs/cost-tooling-research-2026-10-10.md (it's a
  Reddit/X reply-drafting+autopost plugin, not a content-performance schema — kept on the research
  list regardless per explicit user instruction, for a possible separate future project) and
  clarified `ViralLab/SMDT`'s "normalization" means cross-platform schema standardization, not
  account-baseline performance normalization — don't conflate the two.
- **2026-10-12 (fourth revision)**: added two sections prompted by a single operator podcast Q&A
  transcript (flagged explicitly as unverified hypothesis, not proven methodology — same
  discipline as the Appendix A reference videos). (1) Checklist enforcement: accepted insights now
  get classified by HOW they're enforced (`eval_check` in the eval harness / `design_constraint`
  in the Design engine / `prompt_guidance` as last-resort prose) instead of all defaulting to
  playbook prose, each tracked back to `derived_from_insight_id` with its own measured pass rate,
  capped at a small number of new items per playbook version so effects stay isolated and
  measurable. (2) Explore/exploit cycling: a real policy (`determine_explore_exploit_ratio`) for
  the previously-unused `experiments.explore_exploit_ratio` field — wide-explore when
  `insights_owned` evidence for a focus area is thin, shift to exploit as effect_size/sample_size
  strengthen, reset to explore once exploit-phase gains measurably flatten. First concrete link
  between the Learn engine's statistical output and the Write engine's actual batch effort/volume.
