# Qualume — Tool Division of Labor

> Who does what, and why. Written 2026-10-10 after the user requested a clear split and asked that
> heavy coding move to Claude Code going forward. Keep this updated if the split changes.

## Quick reference

| Tool | Role | Scope |
|---|---|---|
| **Hermes** (this chat) | Orchestrator, planner, decision-maker, account/credential gatekeeper | Everything that needs judgment, user interaction, or cross-cutting coordination |
| **Claude Code** | Heavy coding executor | Writing/editing code, running tests, git operations — delegated from Hermes |
| **n8n** | Scheduling + integration plumbing | Cron-like triggers, webhook receivers, connecting services together — NOT reasoning logic |
| **LangGraph** | Reasoning pipelines | The actual Write/Design/Learn graphs — the "thinking" parts of the content engine itself |
| **Claude API (Anthropic)** | The model doing the thinking | Called BY the LangGraph pipelines and by Claude Code — not a separate orchestration layer |
| **Postgres** | State/memory | Every table of data the whole system reads and writes |
| **Raspberry Pi (future)** | Hosting | Where Postgres + the schedulers + pollers physically run, unattended, 24/7 |

---

## Hermes — what it's for

Hermes (this conversation) is the **project's cockpit and decision layer**, not primarily a code
editor going forward. Concretely, Hermes handles:

- **Planning and architecture** — writing/maintaining the plans (`v2-engineering-plan.md`,
  `roadmap-master.md`), making the calls on what gets built in what order, catching architectural
  issues before they become code.
- **Research** — web research (like the Pi-vs-cloud and scraping-pattern research done earlier),
  evaluating tools/approaches, reading documentation.
- **Anything needing your explicit input** — account/credential decisions (Instagram scraping
  account, Pi purchase timing), approvals (publish gate, playbook promotion), clarifying questions.
  Claude Code cannot ask you questions mid-task — if a task needs a judgment call from you, it has
  to happen here first, then get handed to Claude Code as a resolved, concrete instruction.
- **Small, fast fixes and verification** — a one-line config change, running the test suite to check
  status, inspecting the database, verifying something works live. Not every single code change
  needs to go through Claude Code — trivial things are often faster to just do directly.
- **Delegating and reviewing Claude Code's work** — dispatching coding tasks to Claude Code with
  full context, then checking what it actually did (reading the diff, running tests, verifying
  claims) before considering a task done. Claude Code's self-report of "done" is not itself proof —
  Hermes verifies.
- **Cross-session memory** — this chat is what accumulates the project's history/decisions
  (`docs/session-context.md`), since Claude Code sessions are more task-scoped and don't carry the
  same long-running relationship context.

**What Hermes should NOT do going forward:** write large/complex features end-to-end by hand in this
chat. That's the work being handed to Claude Code now — Hermes should delegate substantial coding
tasks rather than grinding through every TDD cycle manually in this window.

---

## Claude Code — what it's for

Claude Code is the **coding workhorse**. Verified installed, authenticated, and working on this
project as of today (confirmed it can read `CLAUDE.md` and reason correctly about project
conventions). From now on, substantial code changes should be delegated to it rather than written
directly in Hermes. Concretely:

- **Implementing features** — e.g. "build the annotation pipeline for Part 5a," "add retry/backoff
  to the TikTok poller," "wire up the Instagram reader once the account decision is made."
- **Writing and running tests** — Claude Code has shell access and can run `pytest` itself, so it can
  do full TDD red/green cycles autonomously within a delegated task.
- **Refactoring** — cleaning up code, fixing the known test-fixture-pollution debt (building a real
  separate test DB), anything that touches a lot of files mechanically.
- **Git operations** — commits, diffs, reviewing changes — Claude Code is comfortable operating git
  directly within a task.

**How Hermes invokes it, in practice:**
- **Print mode (`claude -p "..."`)** for self-contained, one-shot tasks — "add a function that does
  X, with tests" — no back-and-forth needed, fastest and cheapest.
- **Interactive mode (via tmux)** for anything that benefits from iteration — a multi-step feature
  where Claude Code might need to course-correct based on test failures, or where Hermes wants to
  monitor progress and steer mid-task.
- Always pointed at `workdir="~/qualume"` so it loads `CLAUDE.md` and has the right project context
  automatically.
- Given `--allowedTools` scoped to what the task actually needs (e.g. `Read,Edit,Bash` for a coding
  task, not broader than necessary).

**What Claude Code should NOT do without Hermes/the user first resolving it:**
- Make account/credential decisions (Instagram scraping account choice, anything requiring a login).
- Publish anything to a real social account.
- Silently relax the deliberately-conservative patterns already in the codebase (the honest
  `tier: proposed` playbook tagging, the eval gate's fail-loud behavior) — these were built carefully
  and shouldn't get "optimized away" by an agent trying to make tests pass faster.

---

## n8n — what it's for

n8n is **plumbing only** — it should never contain actual reasoning/business logic. Its job:

- **Scheduling** — triggering the TikTok poller (and eventually the Learn-engine's mining/playbook
  jobs) on a recurring cadence. This is the next piece to actually wire up (currently the poller only
  runs when manually invoked).
- **Webhooks/integration glue** — if/when real publish automation or external API callbacks get
  built, n8n is where those connections live, not hardcoded into Python scripts.
- **Visibility** — n8n's UI gives a visual way to see what's scheduled and whether recent runs
  succeeded/failed, which plain cron doesn't give you for free.

n8n is already running (Docker, port 5678) but nothing is wired into it yet — it's provisioned but
unused. First real n8n task: scheduling the TikTok poller (roadmap Part 2/9, open decision #3).

---

## LangGraph — what it's for

LangGraph is **where the actual AI reasoning pipelines live** — the Write engine, Design engine, and
eventually the Learn engine's annotation/mining/playbook-authoring stages are all LangGraph graphs.
This is conceptually different from both Hermes and Claude Code: those are tools *you* (and Hermes)
use to build and operate Qualume; LangGraph graphs are *part of Qualume itself* — the code that
actually generates content and will eventually learn from data, running independent of whether
Hermes or Claude Code are even open.

- Already built: `write_ig_static.py`, `design_ig_static.py` (both LangGraph graphs).
- Not yet built: the Learn engine's graphs (annotation, pattern-mining, playbook-authoring — roadmap
  Part 5).

---

## The Claude API itself — what it's for

Worth being explicit since it's easy to conflate with "Claude Code": the **Claude API** (Anthropic's
model-serving endpoint) is the actual intelligence being called both by LangGraph's pipelines (via
`pick_llm()` in the codebase) and by Claude Code (as the model powering its own coding ability).
It's not a separate orchestration layer you interact with directly — it's the thing underneath both
LangGraph and Claude Code. There is no separate "Anthropic cloud hosting" product — Anthropic only
serves model inference, not general compute/hosting (clarified in an earlier session discussion).

---

## Updated workflow going forward

For any new Qualume coding task:

1. **Hermes** figures out what needs to be built (checking `roadmap-master.md`), resolves any open
   decisions that block it (asking the user if needed), and writes a clear, self-contained task
   description.
2. **Hermes delegates to Claude Code** (print mode for simple/contained tasks, interactive+tmux for
   complex/iterative ones), pointed at the project directory so `CLAUDE.md` loads automatically.
3. **Claude Code implements it** — TDD cycle, tests passing, committed.
4. **Hermes verifies** — reads the diff/result, runs the test suite independently, checks against
   real data if relevant (not just trusting Claude Code's self-report).
5. **Hermes updates the roadmap/session-context docs** to reflect what's now done, and reports back
   to the user.

Small one-off things (checking a DB value, a one-line fix, answering a question) can still happen
directly in Hermes without the full delegation ceremony — this workflow is for substantial coding
work, not everything.
