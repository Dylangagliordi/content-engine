# Qualume: Project Context (for Claude Code)

This file is auto-loaded by Claude Code CLI whenever it runs in this directory. Keep it accurate —
it's the fastest way to get Claude Code oriented without re-explaining the project every session.

## What this project is

Qualume ("Quality Volume") is an AI content-creation and learning engine. It has two main halves:
1. **Reading/Learning** — continuously discover content from chosen TikTok (and eventually Instagram)
   accounts, store it, and eventually mine it for patterns (what hooks/formats/topics perform).
2. **Writing/Design** — generate and render original IG static posts (and eventually TikTok/Reels
   videos) for the business account "greenbabe," gated by an eval check before anything publishes.

Full architecture: `plans/v2-engineering-plan.md`. Full granular roadmap + status:
`plans/roadmap-master.md` (this is the living source of truth — check it before starting new work).
Session history/decisions: `docs/session-context.md`.

## Stack

- Python, managed with `uv` (not pip/poetry — use `uv add`, `uv run`)
- LangGraph + langchain-anthropic for the reasoning pipelines (Write/Design engines)
- Postgres 17 (via Docker/OrbStack) for all state — 11 tables as of this writing, see
  `src/qualume/db/migrations/*.sql` for schema
- n8n (via Docker) for scheduling/orchestration — not yet wired up to anything real
- Claude only (Anthropic) — no OpenAI/Gemini. Model tiers via `src/qualume/llm/pick_llm.py`
  (`low`=haiku, `medium`=sonnet, `high`=opus — verify these model IDs are still current if it's been
  a while: `python -c "import anthropic; print([m.id for m in anthropic.Anthropic().models.list().data])"`)
- `yt-dlp` for TikTok content reading (no login required, confirmed working)
- Pillow for IG static image rendering (template-based, deterministic — not an image-gen model yet)

## Engineering principles (non-negotiable, see v2-engineering-plan.md §1 for full rationale)

1. **TDD, strictly.** Write the failing test first, confirm it fails for the right reason, then
   implement, then confirm green. Every module in this codebase was built this way — don't break
   the pattern.
2. **Typed contracts everywhere.** Pydantic models for all pipeline boundaries
   (`src/qualume/models/contracts.py`). Fail loud on bad input, never silently coerce.
3. **Everything versioned.** Prompts, playbooks, schema — track version IDs so any output is
   traceable back to exactly what produced it.
4. **Append-only performance data.** Never UPDATE a performance snapshot — INSERT a new one. Full
   time-series matters more than "current value."
5. **Owned vs. external content are structurally separate tables**, never conflated. Content you
   generate (`content_items`, `creative_dna`) is NOT the same trust level as content scraped from
   other accounts (`external_content_items`, `external_performance_snapshots`).
6. **Tests must clean up their own fixtures.** There is currently NO separate test database — tests
   write directly into the real dev Postgres instance. This has already caused real problems (test
   rows polluting real counts, had to be manually cleaned up twice). ALWAYS delete any rows a test
   creates, in a `finally` block, even on failure. Flagged in roadmap Part 1 as technical debt to fix
   properly (separate test DB/schema) — until then, be disciplined about cleanup.
7. **No unverified model names or API assumptions.** If referencing a Claude model ID, an external
   API's field, or anything else that could be stale, verify it live before trusting it. This
   codebase has already caught real bugs this way (see git log — "fixed eval-gate keyword bug").

## Commands

```bash
# Run tests (always do this before AND after any change)
cd ~/qualume && uv run pytest tests/ -v

# Add a dependency
uv add <package>

# Docker stack (Postgres + n8n)
export PATH="$HOME/.orbstack/bin:$PATH"  # needed in shells opened before OrbStack install
docker compose up -d
docker compose ps

# Manage the TikTok watchlist
uv run python -m qualume.manage_watchlist add tiktok <handle>
uv run python -m qualume.manage_watchlist list

# Run the TikTok poller manually (not yet scheduled)
uv run python -m qualume.poll_tiktok_watchlist

# Generate a content batch (Write -> eval gate -> Design -> persist)
uv run python -m qualume.run_batch --brief "<product/topic>" --goal "<goal>" --count 5
```

## What NOT to do without asking first

- Don't commit `.env` (it's gitignored — keep it that way; it holds the real Anthropic API key and
  DB password).
- Don't post anything to Instagram or any real social account autonomously — publishing is always a
  manual, human-approved step in this project.
- Don't build Instagram *scraping* (reading other accounts' content) — this is explicitly blocked on
  an unresolved account-risk decision (see roadmap Part 3). Don't assume "greenbabe" should be used
  for scraping without an explicit go-ahead.
- Don't silently change the eval-gate or playbook logic — these are deliberately conservative/
  honest-about-uncertainty by design (see the `tier: proposed, source: external_field_notes` pattern
  in the seeded playbook) — don't "improve" them into overconfidence.
