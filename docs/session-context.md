# Qualume — Project Context & Session Log

This file is the durable record of how this project's plan evolved, so a new session (human or
agent) can pick up the thread without needing the original chat transcript.

**Project:** Qualume (Quality Volume) — an AI content-creation engine that batch-generates
TikTok/Reels videos, IG statics, and future formats, each as its own workflow, and feeds performance
data into a learning loop that figures out what makes a good hook, headline, format, audio choice, etc.

**Planned stack:** Hermes (cockpit + playbook store), LangGraph (reasoning pipelines), n8n
(integration/orchestration plumbing).

---

## Folder layout

```
~/qualume/
  plans/          versioned plan documents (read v2 first, v1 kept for history)
  transcripts/    raw transcripts of reference videos used as research input
  docs/           this file + any future design docs
  data/           (empty, reserved for future local data/fixtures)
  src/            (empty, reserved for future code)
```

## Plan version history

- **`plans/v1-initial-plan.md`** — first draft. Built the architecture too directly off two
  Hormozi/ecommerce YouTube videos (treated one ad-agency operator's specific workflow as the
  reference architecture). **Superseded** — kept only for history.
- **`plans/v2-engineering-plan.md`** — **current plan.** Rebuilt from first-principles AI/software
  engineering practice (typed contracts, versioning, statistical significance gating, eval harnesses,
  explore/exploit, systematic debugging, tracer-bullet rollout) per explicit user correction (see
  "Key decisions" below). The two videos were demoted to Appendix A — labeled as unverified field
  notes/hypotheses to test, not architecture to copy.

Read `plans/v2-engineering-plan.md` top to bottom for the actual plan. This doc is index + rationale,
not a replacement.

## Key decisions / correction history (why the plan looks the way it does)

1. **User's original ask:** build a content-creation engine across multiple formats (each its own
   workflow), running on its own generated data, with a research arm to learn from other creators'
   content too — mission "Qualume = Quality Volume." Stack: Hermes + LangGraph + n8n. Gave two
   reference videos for context before planning.
2. **First plan (v1)** synthesized the two videos' specific workflow (Hormozi-style ad-agency
   self-learning pipeline; AI-run ecommerce launch) directly into the core architecture — e.g. the
   whole ingest→annotate→mine→playbook→generate→publish loop and the "v0 playbook" heuristics were
   near-paraphrases of one practitioner's specific ad-optimization workflow.
3. **User called this out:** "make sure u didnt just make the plan off the vids cus that wasnt my
   intention." Correct instinct — one small-sample practitioner case study (the source himself
   reports ~50% of his AI-informed ad batch flopped) is a weak foundation for core architecture.
4. **Plan rebuilt (v2)** from engineering principles: typed/versioned contracts at every pipeline
   boundary, idempotent+immutable data handling, explicit explore/exploit allocation (not pure
   copy-the-winners), statistical significance tiers before any mined pattern reaches a live
   playbook, shrinking (not zero) human-approval gates, systematic-debugging discipline (reproduce →
   trace → ranked hypotheses → one variable at a time → "3+ failed fixes = question the architecture"),
   and TDD-style eval-harness-first discipline. The videos were moved to an appendix as hypotheses to
   test against real data, not facts.
5. **Third video reviewed** (LangGraph data-agent tutorial, 7yOmi4IX-Rs, ~5hr hands-on build) — not
   yet folded into the plan as a dedicated section. Summary below. Useful concrete LangGraph patterns:
   Pydantic state (fail-loud validation), stateful-orchestration graphs vs. React tool-calling loops,
   model-tiering per node (cheap/medium/expensive LLM by task complexity), LLM-as-judge nodes with
   isolated prompts (exclude raw user input to prevent prompt injection into judge calls), supervisor/
   router pattern dispatching to composable sub-graphs, and a concrete LangGraph gotcha: parallel
   conditional branches must not both independently return/mutate the same state key (causes
   `InvalidUpdateError`) — route via explicit node-name mapping instead.
6. **Open/pending:** user asked whether to fold these LangGraph patterns into a new "§9 LangGraph
   implementation patterns" section of the v2 plan (with pseudo-code per pillar's graph) — not done
   yet as of this writing. Do that next if picking this project back up.

## Summary: Video 3 — LangGraph Data Agent tutorial (7yOmi4IX-Rs)

5-hour hands-on tutorial building a multi-agent "Data Agent" that routes natural-language business
questions to either a SQL Analyst sub-agent or an ETL (Python/pandas) Analyst sub-agent.

**Two core graph patterns taught:**

- **Stateful orchestration graph** (used for the SQL Analyst): fixed node sequence —
  `curate_question → add_schema_context → generate_SQL → is_safe?(conditional) → execute_SQL/cancel
  → represent_final_answer`. State is a Pydantic BaseModel (chosen over TypedDict specifically
  because Pydantic fails loudly/immediately on bad data — framed as a feature, not an inconvenience).
  Context engineering (injecting full DB schema + sample rows) is a dedicated node, not assumed.
  Model-tiering: cheap/low-end model for simple rewrites, higher-end for SQL generation itself, cheap
  again for final-answer formatting.
- **React (reasoning+acting) loop** (used for ETL Analyst): one LLM node bound to `@tool`-decorated
  Python functions; no fixed sequence. LLM reasons → calls a tool → tool node executes and appends a
  ToolMessage → LLM reasons again → repeats until done. Tool descriptions/docstrings are the only
  thing guiding the LLM's tool choice, so they must be detailed. Generated code (e.g. pandas
  transforms) executed server-side via `exec()` (flagged as fine for local dev; sandboxing/task
  queues recommended for production).
- **Supervisor/router pattern**: a third, simple LLM node with structured output
  (`answer: "SQL"|"ETL"`) classifies incoming requests and a conditional edge dispatches to whichever
  compiled sub-agent graph applies. Sub-graphs are composable — callable like any function from a
  parent node.

**Other patterns worth reusing in Qualume:**
- LLM-as-judge pattern: a dedicated node with its own narrow Pydantic output schema
  (`answer: yes/no`, `comments`) judges something (SQL safety, in this case) — deliberately excludes
  raw user input from the judge's prompt to prevent the user from manipulating the judge via the same
  channel used to generate the thing being judged.
- Real bug worth pre-empting: when two conditional branches each independently try to
  "return the full state," LangGraph throws `InvalidUpdateError — can receive only one value per
  step" because both branches attempt to update the same state key concurrently. Fix: route via an
  explicit node-name string/mapping from the conditional function, not parallel state mutation. This
  directly matters for Qualume's planned fan-out steps (hook×body matrix expansion, multi-variant
  scoring).

**Relevance mapping to Qualume's v2 plan:**
- Model-tiering per node → apply to Write/Design engines (cheap model for ideation/formatting,
  expensive model only for the generation step that must be structurally/creatively correct).
- LLM-as-judge with isolated prompts → direct pattern for the synthetic-audience pre-flight scoring
  step and any "is this generated asset safe/on-brand to publish" gate.
- Supervisor + composable sub-graphs → natural LangGraph shape for Qualume's Read/Learn/Write/Design
  pillars: one top-level router dispatching to per-workflow_type sub-graphs (tiktok_reel generation
  graph, ig_static generation graph, etc.), each independently compiled and testable.
- Parallel-state-update gotcha → explicit thing to design around wherever Qualume's graphs fan out.

---

## Next steps (as of this writing)
1. Decide whether to add a "LangGraph implementation patterns" section to the v2 plan with
   pseudo-code per pillar (Read/Learn/Write/Design) incorporating the patterns above. (Still open.)
2. Execute `plans/execution-plan-phase-0-1.md` — see below, this is now the active work.

## Phase 0/1 tooling decisions (confirmed 2026-10-10)

These resolve the "open questions" that were blocking a concrete execution plan:
- **LLM provider: Anthropic (Claude) only.** Used for annotation (Claude vision handles IG-static
  image annotation directly), generation, and judge/critique calls. No Gemini/OpenAI in Phase 0/1.
- **Infra: all local via Docker** on this machine (not cloud/hosted). Docker was NOT installed on
  this machine as of 2026-10-10 — `docker --version` returned "command not found", no Docker
  Desktop/OrbStack found in /Applications. First execution task installs OrbStack via
  `brew install --cask orbstack` (needs explicit user approval — installs system software).
- **Phase 1 pilot workflow: IG statics** (confirmed, matches v2 plan's recommendation — simplest
  Design engine, no video assembly).
- **Data reality: fully greenfield.** One Instagram account exists, zero posts, zero performance
  data, no Meta Developer App / Graph API access configured yet. This is the single biggest factor
  shaping the execution plan's Phase 1 shape:
  - Automated IG ingestion (Graph API) requires IG Business/Creator conversion + FB Page link + Meta
    Developer App + access token — a multi-day external-approval dependency that must NOT block the
    tracer bullet.
  - Solution: Phase 1 splits ingestion into **1a (manual performance entry via a small CLI script,
    unblocked, start immediately)** and **1b (real Graph API ingestion, parallel task, no hard
    blocking dependency on 1a)**.
  - Phase 1 publish step is also manual (you post to IG by hand after reviewing generated assets) —
    automating posting needs the same Graph API access as ingestion, not worth building before
    Write/Design output is even proven worth posting.
  - Design engine (Phase 1) uses **Pillow-based template rendering** (headline/body text placed onto
    pre-made template backgrounds) rather than an image-gen model/service — simplest, zero-API-cost,
    fully deterministic option to prove the pipeline shape first. Swapping in a real design-gen tool
    is an explicit fast-follow, not a Phase 1 blocker.

## Execution plan

**`plans/execution-plan-phase-0-1.md`** is the concrete, bite-sized build plan for Phase 0
(foundations: Docker/Postgres/n8n stack, Python project with uv, Pydantic contracts, model-tiering
helper, eval harness skeleton, seeded v0 playbook) and Phase 1 (tracer bullet: Write engine → Design
engine → eval gate → manual publish → manual performance logging, IG statics only, no automated
learning yet). Each task has exact file paths, code, and a verify step. Phase 2+ stays at
roadmap-level in `v2-engineering-plan.md` until Phase 1's exit criteria are met — there's no real
performance data to mine insights from until then.

Immediate next action if picking this back up: run Task 0.1 (`brew install --cask orbstack`) —
flagged in the execution plan as needing explicit approval since it installs system software — then
proceed through Tasks 0.2–0.10 in order.
