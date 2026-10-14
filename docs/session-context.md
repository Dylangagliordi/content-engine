# Qualume — Session Handoff (2026-10-12)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified, not assumed)
- 51/51 tests passing. Git clean. Playbook at v2 (anti-fabrication claims rule + vyralcontent
  hook hypotheses). Retry loop on by default (max_attempts=2, measured 73%->100% pass-rate lift
  post-claims-fix). Zero TikTok accounts watched yet — reading engine built/tested but idle.
  5 unpublished draft IG statics in data/assets/, nothing has ever actually been published.
- Hermes itself now runs on Claude Pro subscription auth (CLAUDE_CODE_OAUTH_TOKEN), not API
  credits — switched 2026-10-10 after diagnosing real spend. Qualume's own `.env` key is separate
  and unaffected. Three distinct cost meters, still true: this Hermes session, Qualume's own API
  key, Claude Code's Pro-plan session usage (separate daily cap).

## Current focus: designing Part 5 (Learn engine) before building it
Full design: `plans/learn-engine-idea-sourcing-architecture.md` (revised twice same-week, see its
own Revision log). Core shape as of 2026-10-12:
- `external_creative_dna` (external content's structured-DNA table, mirrors owned `creative_dna`
  + vibe fields) — NOT YET migrated into the DB.
- `creative_dna_feature_registry` + `extended_features` JSONB — features are learned/promoted
  over time via a "feature discovery" sub-step, not hardcoded forever.
- Gemini DECIDED as the video-watching model (native audio+video fusion) — deliberate scoped
  exception to Claude-only, every other stage stays Claude. CLAUDE.md still needs a line updated
  to reflect this explicitly (attempted once, write requires explicit user approval on that
  protected file, prompt timed out — ask user directly next time, don't force-retry).
- No static "data overview" doc — retrieval against the existing `insights` table instead
  (`applies_to_segment` JSONB addition), ranked by effect_size/sample_size/recency per-brief.
  pgvector NOT installed (checked) — tag-based filtering first, semantic search is a real future
  upgrade, not built ahead of need.
- `content_items.grounding_refs` — closes the loop: a published post's real performance
  re-validates (or weakens) the specific insight(s) that inspired it.
- **Not yet added to the doc, discussed 2026-10-12, needs folding in next session**:
  - Own-content performance should be judged against OUR OWN account's rolling median/stddev
    baseline (not raw numbers) before counting as evidence for/against an insight — this was
    MY design reasoning in conversation, NOT sourced from any GitHub repo (corrected a
    mischaracterization where it was implied to be research-backed). Compute fresh at
    query/mining time, don't persist as a column (append-only principle).
  - Split `insights` into `insights_external` (mined from watched accounts, capped lower trust
    even at tier=accepted) and `insights_owned` (tested against our own account, the one that
    actually drives decisions) — mirrors the existing owned-vs-external table separation
    principle already applied elsewhere in this schema. Third, lower-priority future table:
    `insights_meta` (patterns about patterns, e.g. "hook insights have higher effect size than
    CTA insights on average") — not urgent, no data to support it yet.
  - The existing `experiments` table (batch_id, hypothesis, explore_exploit_ratio,
    workflow_type, outcome_summary) is the actual mechanism for deliberately testing an insight
    against our account — already built, under-referenced in the design doc so far, tie it in
    explicitly rather than inventing a new mechanism.
  - User's "planner vs do-er" question connects directly to the SQL Analyst LangGraph pattern
    already researched in this project (way back, video summary, stateful orchestration): a
    periodic pattern-miner (scheduled, mostly deterministic) vs. an ad hoc Analyst sub-agent
    (SQL-analyst-shaped, answers on-demand questions like "does pacing matter independent of
    length and niche" via real regression with those as covariates) vs. feature-discovery
    (already designed). Not built, correctly deferred — zero data exists yet.
  - Statistical method for mining: simple per-feature tests first (Welch's t-test / Mann-Whitney
    U for continuous-vs-binary, chi-squared for categorical), NOT random forest (wrong tool for
    clean interpretable effect-size-with-p-value statements) — regression is a deliberate later
    upgrade once there's enough volume to control for confounders (e.g. the pacing-independent-
    of-length-and-niche question). This reasoning exists in chat, not yet written into the doc.

## Repo research — corrections made 2026-10-12, keep these straight
- `vyralcontent/content-skills` — MIT, free, genuinely useful, hook/algorithm hypotheses already
  folded into playbook v2. Confirmed good.
- `m13v/social-autoposter` ("S4L") — re-verified 2026-10-12, CORRECTED an earlier wrong
  characterization (it is NOT a content-performance schema, it's a Reddit/X reply-drafting +
  autopost plugin). Explicitly kept on the research list per user instruction — they may build a
  SEPARATE future automation for comment/thread replies, unrelated to Qualume's content-creation
  engine. Don't pull it into Qualume's scope; it's a parked idea for later, different project.
- `ViralLab/SMDT` — re-verified 2026-10-12. Its "normalization" = cross-PLATFORM schema
  standardization (Twitter fields vs Reddit fields -> one shared schema), NOT account-baseline
  performance normalization. Don't conflate the two meanings — caught and corrected this
  conflation in docs/cost-tooling-research-2026-10-10.md already.
- PySceneDetect (pacing/cuts-per-min), beatlyze-analyzer (audio mood, PyPI), Essentia (heavier
  audio-mood option, MTG) — all confirmed real and usable, in the architecture doc.

## Known gap, not yet fixed
Instagram scraping account decision (greenbabe vs. secondary) — still unresolved, still blocking
Part 3. Not touched this session.

## Immediate next action
Fold the "not yet added" bullet list above into
`plans/learn-engine-idea-sourcing-architecture.md` as a proper revision (same pattern as its
existing two prior revisions) before any schema migration gets built. Nothing should be built
until the watchlist actually has real accounts in it (still zero) — that remains the actual
unblocking action if the user wants to move from design to real data.
