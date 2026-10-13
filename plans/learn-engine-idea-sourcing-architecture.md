# Learn Engine & Idea-Sourcing Architecture (Part 5 design)

Written 2026-10-12. This turns roadmap Part 5 ("Learn engine — not started") into a concrete
design. Nothing here is built yet — this is the target to build toward, not a status report.
Supersedes the vague Part 5 placeholder in `plans/roadmap-master.md`.

## The problem this solves

Today, the Write engine's `ideate_angles` node is told: here's a brief (what we're promoting)
and a rubric (how to write well) — invent some angles. That's generation from a style guide, not
generation from evidence. Nothing connects a generated idea to anything actually observed to work
in the real world. This doc fixes that.

## Two documents, kept deliberately separate

1. **The Playbook** (exists today, `playbooks` table, v0→v2) — HOW to write: tone, copywriting
   mechanics, format rules, the claims-safety gate. Stays as-is.
2. **The Data Overview** (new, does not exist yet) — WHAT's actually happening: a standing,
   periodically-regenerated summary of cross-video patterns mined from the Read+Learn engines.
   Only becomes real once there's real data to summarize (today: zero accounts watched).

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

    -- provenance (same discipline as everywhere else in this codebase)
    analysis_model                TEXT NOT NULL,
    analysis_prompt_version       TEXT NOT NULL,
    analyzed_at                    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

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

### `data_overview` — versioned, same discipline as playbooks
```sql
CREATE TABLE data_overview (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version          TEXT NOT NULL,
    summary_json     JSONB NOT NULL,     -- mined, cross-video patterns with tier/confidence
    videos_analyzed  INTEGER NOT NULL,   -- sample size this version was built from
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```
Every generated piece of content should stay traceable to which data_overview version informed
it — same `*_version_used` provenance pattern already used for playbooks.

## How AI actually "watches" a video — researched, not guessed

Claude has no native video/audio understanding (confirmed earlier this project). The baseline
workaround (yt-dlp downloads the real video, ffmpeg extracts frames, transcript from captions or
Whisper, frames+transcript fed to one Claude prompt) works and is cheap (~$0.01-0.03/short clip)
but may under-capture *vibe* specifically — mood, pacing, energy — since it's a text+static-image
representation of something inherently audiovisual and temporal.

**Research findings (2026-10-12), each grounded in a real source, not speculation:**

- **Gemini's native video understanding genuinely fuses audio+visual** (confirmed via Google's
  own docs: ai.google.dev/gemini-api/docs/video-understanding) — not frames-then-transcript
  bolted together. Newer Gemini 3.x Flash models have an "agentic" mode that adaptively navigates
  the timeline instead of uniform sampling; Google's own claimed numbers: up to 88% fewer tokens,
  66% lower cost, up to 7% higher accuracy vs static sampling. **Real case for using Gemini
  specifically for this one analysis stage**, even with the rest of the stack staying Claude-only
  — worth a real side-by-side test before committing either way.
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
2. **Model-observed signals** (LLM, real judgment needed): frames+transcript (or native Gemini
   video+audio — worth testing both) → structured JSON with anchored scales: `energy_level`
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

### Mode B — From the aggregate data overview
The Learn engine periodically (after every N new videos, or on a schedule) re-mines everything
into the `data_overview` table — cross-video patterns ("hooks under 3s with a visible problem
outperform scripted intros by X%"), gated through the same `tier: proposed→testing→accepted`
statistical-validation discipline already designed into the v2 engineering plan for playbook
insights. That summary gets injected into `ideate_angles` as background context alongside the
playbook.

### Graph change required
A new node, `gather_inspiration_node`, runs before `ideate_angles` in the Write graph. It
populates `state.inspiration_source` (a specific external video reference, Mode A) and/or
`state.data_overview_summary` (Mode B) — likely both together once enough data exists. The
ideation prompt changes from "invent angles for this brief" to "here's grounding evidence — invent
angles for this brief that apply what's actually been observed to work."

## Build sequencing (depends on, in order)

1. TikTok watchlist actually populated (Part 2, already unblocked — zero accounts today)
2. `external_creative_dna` + `watchlist_type` schema migration
3. The video-watching pipeline (computed signals + model-observed signals → fused DNA row)
4. 20+ videos analyzed in at least one watchlist/niche before Mode A can activate
5. First `data_overview` version once there's enough analyzed volume for cross-video mining —
   Mode B activates here
6. `gather_inspiration_node` added to the Write graph, wired to both modes
7. Re-validate: does grounded ideation actually produce better content than today's ungrounded
   version? Needs a real A/B comparison (same discipline as the retry-loop and claims-gate
   verdicts already done this project) before trusting it, not assumed.

## Open questions, not resolved here

- Gemini-for-vibe-analysis vs. staying Claude-only for this one stage — real tradeoff, needs a
  side-by-side test on a handful of real videos before deciding, not a guess.
- Exact `tone_category` enum values and `energy_level` anchor text — draft during actual
  implementation against real sample videos, not invented abstractly now.
- Whether Mode A and Mode B run together on every batch or are selected contextually — likely
  "both, weighted," but undecided until there's real data to test against.
