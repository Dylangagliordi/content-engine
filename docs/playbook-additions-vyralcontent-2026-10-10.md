# Playbook Additions — vyralcontent/content-skills findings (2026-10-10)

Free, MIT-licensed, platform-mechanics-grounded hypotheses pulled from `vyralcontent/content-skills`
(`viral-hooks` and `viral-tiktok-content` skill files, read in full). Per project convention, these
enter as `tier: proposed, source: external_field_notes` — unvalidated against our own data until
proven via our own eval/performance loop, same discipline as the original Hormozi/ecommerce video
hypotheses already in the v0 playbook.

**Staged here, not yet written to the DB** — to be folded in alongside the anti-fabrication rubric
fix (queued, pending Claude Code session reset) as a single clean playbook version bump rather than
two racing edits to the same row.

## Applies to ig_static (current playbook) — new rubric candidates

- **Three-layer hook principle**: a hook isn't just the opening line of text — it's the combination
  of (1) the visual/first-frame, (2) the verbal/spoken line (N/A for static but relevant once we
  build tiktok_video), and (3) on-screen text. For statics specifically, this suggests the rubric
  should check that the headline text and the background image/visual aren't saying two different,
  disconnected things — right now `design_ig_static.py` has no check that headline and visual
  reinforce each other, it just renders whatever text onto whatever template.
- **Named anti-patterns worth adding as explicit must_avoid-style checks**: hooks that ask a generic
  question with no stakes ("Did you know...?"), hooks that bury the actual value 2+ sentences in,
  hooks that oversell ("You won't believe...") without a concrete specific claim right after.

## Applies to a future tiktok_video playbook (not yet built, no workflow_type row exists yet)

- **Completion-rate benchmarks by length** — concrete numbers worth citing as a reference until we
  have our own data: >100% (loop/rewatch) for ≤15s content, ~70% for 15-30s, ~50% for 30-60s, ~40%
  for 60s+. Practical implication for scripting: favor shorter cuts until our own data says
  otherwise, since shorter clips have a structurally easier bar to clear.
- **Saves/shares outweigh likes** — consistent with what the Hormozi video already told us (now have
  two independent sources agreeing on this, worth noting as corroborating rather than just one
  anecdote).
- **"Test-then-expand" distribution mechanic** — TikTok re-tests every new video against a small
  lookalike cohort regardless of the account's follower count or past performance. Practical
  implications for Qualume: (1) a new/small test account is a fair way to validate content ideas,
  since it's not structurally disadvantaged vs. a big account for a given video's first test; (2) a
  video that flops initially can still "re-enter testing" later — don't treat a low first-24h number
  as a permanent verdict the way we might for a one-shot ad platform.

## Not adopted / flagged as lower confidence
The repo's named-creator hook framework references (attributing specific techniques to specific
creators like Hormozi/MrBeast/Brunson) are useful as flavor/reference but not restated here as rules
— attributing a technique to a creator doesn't make it generalizable, and we already have the
discipline of not over-indexing on any single source's claims (this is the same correction applied
to the original two YouTube videos in plan v2).

## Next step
Fold the "applies to ig_static" section above into the next playbook version bump, together with the
anti-fabrication claims/honesty rubric category (queued separately). Hold the tiktok_video section
until that workflow actually exists (Part 8 of the roadmap) — creating a playbook row with no
consuming code yet would be premature per the "don't build ahead of what's used" principle already
in the v2 plan.
