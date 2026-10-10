# Qualume — Session Handoff (2026-10-10, end of session)

## Start here next session
Read this file first. Don't re-derive context from chat history — it's expensive. Detail docs
referenced below have the full writeups if something needs deeper verification.

## Project state
- Repo: ~/qualume, git-initialized, clean working tree as of this handoff (verify with `git log -5`)
- Stack: Postgres 17 + n8n via OrbStack/docker-compose (both running), Python/uv, LangGraph,
  Claude (Anthropic direct API + Claude Code CLI on Pro plan)
- Built & working: Write engine (ig_static, LangGraph, with self_critique retry loop — currently
  OFF by default, max_attempts=1), Design engine (Pillow, plain Python by design), run_batch.py
  (Write->eval->Design->Postgres), TikTok reading engine (yt-dlp, no login, dedup verified live),
  manual performance-logging CLI
- Blocked: Instagram reading (needs account decision — greenbabe vs secondary, unresolved),
  manual IG publish (waiting on user to post + report back content_item_id), IG Graph API setup

## PENDING — check this first
A cron job (`qualume-antifab-retry-once`, id 69ed304fa788) was scheduled for 5:15pm ET today to
retry a Claude Code task that hit a session-usage limit. Run `cronjob(action='list')` to check its
status/output — it was told to leave changes staged (not committed) and report real verified
numbers. If it completed, review its diff before committing. If it also failed, the task is in
docs/retry-loop-verdict-2026-10-10.md's "Two proposed fixes" section — re-run manually, scoped
smaller per the lesson learned (see below).

## Key recent decisions (full detail in docs/, don't re-litigate)
- `docs/tool-division-of-labor.md`: Hermes=planning/research/verification, Claude Code=heavy
  coding (delegated), n8n=plumbing only, LangGraph=the actual engine logic
- `docs/retry-loop-verdict-2026-10-10.md`: conditional retry loop built & tested, but measured
  live to add zero benefit at current rubric strictness — found a bigger problem instead: the
  eval gate passes fabricated stats/claims (real examples logged). Fix queued (see PENDING above).
- `docs/playbook-additions-vyralcontent-2026-10-10.md`: free MIT-licensed TikTok algorithm
  insights staged, NOT yet written to the DB — waiting to fold in alongside the anti-fabrication
  rubric fix as one clean playbook version bump (currently v0) instead of two racing edits.
- `docs/cost-tooling-research-2026-10-10.md`: diagnosed the ~$20 Qualume API spend (no caching,
  not a leak); prompt caching added to Write engine but currently dormant (rubric too small to
  hit Anthropic's cache-eligibility threshold, ~2.6k tokens) — will activate automatically as the
  playbook grows, no further action needed.

## Lesson learned this session — apply going forward
- **Keep Hermes sessions short and scoped.** This session ran to ~91M cumulative tokens / ~$34
  (via `hermes insights --days 1`) from being one long continuous thread covering many unrelated
  phases. Start a fresh session per work phase instead of one marathon thread.
- **Push multi-step research (web_search/web_extract chains) into `delegate_task` subagents**
  instead of inline — keeps the raw search noise out of the main session's context.
- **Scope Claude Code tasks smaller.** A single 6-step delegated task hit the Pro plan's session
  usage limit mid-run. Split big asks into 2-3 smaller invocations.
- Three separate cost meters exist, don't conflate them: (1) this Hermes session's own token
  cost, (2) Qualume's direct Anthropic API spend (`.env` key), (3) Claude Code's Pro-plan session
  usage (separate limit, resets daily).

## Immediate next actions (pick up here)
1. Check the 5:15pm cron job result, verify its diff yourself (don't trust self-report), commit
   if good.
2. Fold the staged vyralcontent playbook additions in alongside it — one version bump (v0 -> v1).
3. User still needs to decide: Instagram scraping account (greenbabe vs secondary) — blocking
   Part 3 of the roadmap (see plans/roadmap-master.md).
4. User offered to manually export a Canva design as PNG for review (OAuth Connect API setup
   deferred as higher-effort than needed for a one-off look).
