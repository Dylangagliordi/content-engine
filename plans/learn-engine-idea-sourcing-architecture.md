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

## Build sequencing (depends on, in order)

1. TikTok watchlist actually populated (Part 2, already unblocked — zero accounts today)
2. `external_creative_dna` (incl. `extended_features`) + `creative_dna_feature_registry` +
   `watchlist_type` + `insights.applies_to_segment` + `content_items.grounding_refs` schema
   migration
3. The video-watching pipeline: Gemini (model-observed signals) + PySceneDetect/beatlyze-analyzer
   (computed signals) → fused `external_creative_dna` row
4. 20+ videos analyzed in at least one watchlist/niche before Mode A can activate
5. First `insights` rows once there's enough analyzed volume for cross-video mining — Mode B
   activates here. Feature-discovery sub-step (open-ended difference-spotting on top/bottom
   performers) can start running in parallel once there's any meaningful volume, independent of
   whether Mode A/B are live yet.
6. `gather_inspiration_node` added to the Write graph, wired to both modes
7. Re-validate: does grounded ideation actually produce better content than today's ungrounded
   version? Needs a real A/B comparison (same discipline as the retry-loop and claims-gate
   verdicts already done this project) before trusting it, not assumed.
8. Once real owned content has been published and gotten performance data back: wire the
   `grounding_refs` feedback loop so published results re-validate (or weaken) the specific
   insights that inspired them, not just feed new external data one-way.

## Open questions, not resolved here

- Exact `tone_category` enum values and `energy_level` anchor text — draft during actual
  implementation against real sample videos, not invented abstractly now.
- Whether Mode A and Mode B run together on every batch or are selected contextually — likely
  "both, weighted," but undecided until there's real data to test against.
- Exact `utility_score` formula and promotion threshold for `creative_dna_feature_registry` —
  needs real insight data to calibrate against, can't be meaningfully set abstractly now.
- Cadence for the feature-discovery sub-step (every N new videos? scheduled? triggered by
  mining-stage stagnation?) — decide once there's real volume to observe a cadence against.

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
