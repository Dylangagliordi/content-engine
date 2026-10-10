# Qualume — Execution Plan: Phase 0 (Foundations) + Phase 1 (Tracer Bullet)

> Companion to `plans/v2-engineering-plan.md`. That doc is the architecture/roadmap. This doc is
> the concrete, bite-sized build plan for the first two phases only — Phase 2+ stay at
> roadmap-level in v2 until Phase 1 actually closes the loop and tells us what we got wrong.

**Confirmed decisions this plan is built on (locked in 2026-10-10):**
- LLM provider: **Anthropic (Claude) only** for now — annotation, generation, and judge calls all use
  Claude. Claude's vision models handle IG-static image annotation directly; no Gemini needed since
  the pilot workflow is static images, not video.
- Infra: **all local via Docker** on this machine. No cloud/hosted services in Phase 0/1.
- Pilot workflow: **IG statics** (simplest Design engine — no video assembly).
- Data reality: **fully greenfield.** One IG account exists, zero posts, zero performance data, no
  Meta Developer App / Graph API access configured yet. This changes Phase 1's shape (see §2below)
  — the loop has to start without owned historical data and without automated ingestion blocking it.

**Key adaptation because of greenfield status:**
Automated Instagram ingestion (Graph API) requires: IG account converted to Business/Creator →
linked to a Facebook Page → a Meta Developer App → app review or at minimum a generated access
token. That's a multi-day external dependency (Meta review queues can be slow) and must not block
closing the tracer-bullet loop. So Phase 1 ingestion is split:
- **1a (now, unblocked):** manual performance entry — a small script/CLI that logs
  views/likes/saves/comments by hand into `performance_snapshots` once a published post has stats.
- **1b (fast-follow, parallel task):** set up the real Graph API path so ingestion automates once
  approved — tracked as its own task, not a blocker for 1a.

---

## 0. Repo/folder conventions for everything below

All paths are relative to `~/qualume/` unless stated otherwise.

```
~/qualume/
  docker-compose.yml
  .env                        (gitignored — real secrets)
  .env.example                (committed — template)
  pyproject.toml              (uv-managed Python project)
  src/
    qualume/
      __init__.py
      config.py               (env/config loader)
      db/
        __init__.py
        connection.py
        migrations/
          001_init_schema.sql
      models/
        __init__.py
        contracts.py           (Pydantic contracts: RawContentRecord, CreativeDNA, ScriptVariant, etc.)
      llm/
        __init__.py
        pick_llm.py             (model-tiering helper, Claude haiku/sonnet/opus by task level)
      graphs/
        __init__.py
        write_ig_static.py      (LangGraph: Write engine for ig_static)
        design_ig_static.py     (LangGraph: Design engine for ig_static)
      eval/
        __init__.py
        harness.py               (runs a golden set against current prompt version, scores, logs to eval_runs)
        golden_sets/
          ig_static_write_v1.jsonl
          ig_static_annotation_v1.jsonl
      ingest/
        __init__.py
        manual_entry.py          (Phase 1a: CLI to log performance stats by hand)
        ig_graph_api.py           (Phase 1b: real API ingestion, stubbed until access exists)
  tests/
    test_contracts.py
    test_pick_llm.py
    test_write_graph.py
    test_design_graph.py
    test_eval_harness.py
  data/
    assets/                    (generated images land here before manual publish)
    templates/                 (static design templates — Pillow/HTML templates)
  docs/
    session-context.md         (already exists — keep updating it)
  plans/
    v1-initial-plan.md
    v2-engineering-plan.md
    execution-plan-phase-0-1.md   (this file)
```

---

## 1. Phase 0 — Foundations

**Exit criteria for all of Phase 0:** Postgres is running with the full schema migrated; the Python
project has working Pydantic contracts, a model-tiering helper, and a running (even if trivial) eval
harness; one playbook skill is seeded; `docker compose up` brings the whole local stack up from a
clean checkout.

### Task 0.1 — Install Docker locally
**Objective:** get a working Docker runtime on this Mac (currently not installed — verified via
`docker --version` returning "command not found", no Docker Desktop/OrbStack found in /Applications).

**Action (needs your go-ahead — installs system software):**
```bash
brew install --cask orbstack
```
OrbStack recommended over Docker Desktop on macOS: lighter resource footprint, faster startup, free
for personal use, drop-in Docker CLI compatible.

**Verify:**
```bash
docker --version
docker compose version
```
Expected: both print version strings without error.

### Task 0.2 — Scaffold the project skeleton
**Files:** create all directories/empty `__init__.py` files per §0 folder tree above.

```bash
cd ~/qualume
mkdir -p src/qualume/{db/migrations,models,llm,graphs,eval/golden_sets,ingest}
mkdir -p tests data/assets data/templates
touch src/qualume/__init__.py src/qualume/db/__init__.py src/qualume/models/__init__.py \
      src/qualume/llm/__init__.py src/qualume/graphs/__init__.py src/qualume/eval/__init__.py \
      src/qualume/ingest/__init__.py
```

**Verify:** `find src -name "*.py" | sort` shows the expected empty files.

### Task 0.3 — Initialize the Python project with uv
**Files:** `pyproject.toml`

```bash
cd ~/qualume
uv init --no-readme --name qualume
uv add langgraph langchain-anthropic pydantic psycopg2-binary python-dotenv pytest pillow
```

**Verify:** `uv run python -c "import langgraph, langchain_anthropic, pydantic, psycopg2, PIL; print('ok')"`
prints `ok`.

### Task 0.4 — `.env` template and real `.env`
**Files:** `.env.example` (committed), `.env` (gitignored, already covered by existing `.gitignore`)

`.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=qualume
POSTGRES_USER=qualume
POSTGRES_PASSWORD=changeme
N8N_PORT=5678
```

Copy it to a real `.env` and fill in your actual Anthropic key:
```bash
cp .env.example .env
```
Then edit `.env` by hand to set `ANTHROPIC_API_KEY` and a real `POSTGRES_PASSWORD`. **Do not paste
your API key into chat** — edit the file directly.

**Verify:** `.env` exists, is listed in `.gitignore`, and `git status` does not show it as trackable.

### Task 0.5 — docker-compose for Postgres + n8n
**Files:** `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:17
    container_name: qualume-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - qualume_pg_data:/var/lib/postgresql/data

  n8n:
    image: n8nio/n8n:latest
    container_name: qualume-n8n
    restart: unless-stopped
    ports:
      - "${N8N_PORT}:5678"
    environment:
      - N8N_SECURE_COOKIE=false
    volumes:
      - qualume_n8n_data:/home/node/.n8n

volumes:
  qualume_pg_data:
  qualume_n8n_data:
```

Note: LangGraph is run as a local Python process in Phase 0/1 (not containerized yet) — it's invoked
directly by scripts/tests, not as a standalone server. Containerizing it as a `langgraph-server` is a
fast-follow once there's an actual n8n→LangGraph HTTP trigger to wire up (Phase 1b or later).

```bash
cd ~/qualume
docker compose up -d
```

**Verify:**
```bash
docker compose ps
# both qualume-postgres and qualume-n8n should show "running"
docker exec -it qualume-postgres psql -U qualume -d qualume -c "select 1;"
# expect: returns 1
```
n8n UI should be reachable at `http://localhost:5678`.

### Task 0.6 — Write the Postgres schema migration
**Files:** `src/qualume/db/migrations/001_init_schema.sql`

```sql
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type TEXT NOT NULL,            -- e.g. 'ig_static'
    platform TEXT NOT NULL,                 -- e.g. 'instagram'
    status TEXT NOT NULL DEFAULT 'draft',   -- draft | published | rejected
    batch_id UUID,
    playbook_version_used TEXT,
    prompt_versions_used JSONB,             -- {"ideation": "v1", "write": "v1", "design": "v1"}
    lane TEXT NOT NULL DEFAULT 'owned',     -- owned | external
    asset_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ
);

CREATE TABLE creative_dna (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    hook_type TEXT,
    hook_text TEXT,
    visual_hook_description TEXT,
    format_tag TEXT,
    script_structure_tags TEXT[],
    cta_type TEXT,
    caption_style TEXT,
    social_proof_used BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE performance_snapshots (
    id BIGSERIAL PRIMARY KEY,
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    views INT, likes INT, saves INT, comments INT, shares INT, follows_gained INT,
    entry_method TEXT NOT NULL DEFAULT 'manual'  -- manual | api
);

CREATE TABLE ad_performance (
    id BIGSERIAL PRIMARY KEY,
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    spend NUMERIC, cpm NUMERIC, cpc NUMERIC, roas NUMERIC,
    attribution_source TEXT, conversions INT
);

CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    hypothesis TEXT,
    explore_exploit_ratio NUMERIC,
    workflow_type TEXT NOT NULL,
    outcome_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'proposed',  -- proposed | testing | accepted | retired
    effect_size NUMERIC,
    sample_size INT,
    p_value NUMERIC,
    supporting_data_refs JSONB,
    applies_to_workflow_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'internal_mining',  -- internal_mining | external_field_notes
    state_history JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type TEXT NOT NULL,
    version TEXT NOT NULL,
    rubric_json JSONB NOT NULL,
    source_insight_ids UUID[],
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(workflow_type, version)
);

CREATE TABLE eval_runs (
    id BIGSERIAL PRIMARY KEY,
    stage TEXT NOT NULL,            -- annotation | write | design
    golden_set_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    score NUMERIC,
    passed BOOLEAN,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()
```

```bash
docker exec -i qualume-postgres psql -U qualume -d qualume < src/qualume/db/migrations/001_init_schema.sql
```

**Verify:**
```bash
docker exec -it qualume-postgres psql -U qualume -d qualume -c "\dt"
```
Expect 8 tables listed: content_items, creative_dna, performance_snapshots, ad_performance,
experiments, insights, playbooks, eval_runs.

### Task 0.7 — Write the Pydantic contract models
**Files:** `src/qualume/models/contracts.py`

```python
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class RawContentRecord(BaseModel):
    """Output contract of the READ engine."""
    platform: str
    external_id: str
    content_type: Literal["static", "video"]
    asset_refs: list[str] = Field(default_factory=list)
    raw_metrics: dict = Field(default_factory=dict)
    source_lane: Literal["owned", "external"]
    pulled_at: datetime


class CreativeDNA(BaseModel):
    """Structured annotation output — one per content_item."""
    content_item_id: UUID
    hook_type: Optional[str] = None
    hook_text: Optional[str] = None
    visual_hook_description: Optional[str] = None
    format_tag: Optional[str] = None
    script_structure_tags: list[str] = Field(default_factory=list)
    cta_type: Optional[str] = None
    caption_style: Optional[str] = None
    social_proof_used: bool = False


class CampaignBrief(BaseModel):
    """Input contract to the WRITE engine."""
    product_or_topic: str
    offer_or_goal: str
    constraints: list[str] = Field(default_factory=list)
    playbook_version: str
    explore_exploit_ratio: float = Field(default=0.3, ge=0.0, le=1.0)


class ScriptVariant(BaseModel):
    """Output contract of the WRITE engine."""
    hook_text: str
    body_text: str
    cta_text: str
    tags: list[str] = Field(default_factory=list)
    playbook_version: str
    prompt_version: str
    explore_or_exploit: Literal["explore", "exploit"]


class DesignRequest(BaseModel):
    """Input contract to the DESIGN engine."""
    script_variant: ScriptVariant
    workflow_type: Literal["ig_static"]
    aspect_ratio: str = "1:1"
    brand_kit_path: Optional[str] = None


class DesignOutput(BaseModel):
    """Output contract of the DESIGN engine — becomes a content_item + creative_dna row."""
    asset_path: str
    workflow_type: Literal["ig_static"]
    creative_dna: CreativeDNA
```

**Test file:** `tests/test_contracts.py`
```python
import pytest
from pydantic import ValidationError
from qualume.models.contracts import CampaignBrief, ScriptVariant


def test_campaign_brief_requires_playbook_version():
    with pytest.raises(ValidationError):
        CampaignBrief(product_or_topic="x", offer_or_goal="y", constraints=[])


def test_campaign_brief_rejects_out_of_range_ratio():
    with pytest.raises(ValidationError):
        CampaignBrief(
            product_or_topic="x", offer_or_goal="y",
            playbook_version="v0", explore_exploit_ratio=1.5
        )


def test_script_variant_rejects_invalid_explore_flag():
    with pytest.raises(ValidationError):
        ScriptVariant(
            hook_text="h", body_text="b", cta_text="c",
            playbook_version="v0", prompt_version="v0",
            explore_or_exploit="maybe"
        )
```

**Verify (RED then GREEN per TDD):**
```bash
cd ~/qualume
uv run pytest tests/test_contracts.py -v
```
Write the test file first, confirm it fails with `ModuleNotFoundError` (contracts.py doesn't exist
yet), then write `contracts.py`, re-run, confirm all 3 pass.

### Task 0.8 — Model-tiering helper (Claude-only, per the LangGraph video pattern)
**Files:** `src/qualume/llm/pick_llm.py`

```python
import os
from langchain_anthropic import ChatAnthropic

_MODELS = {
    "low": "claude-haiku-4-5",      # cheap cleanup/formatting tasks
    "medium": "claude-sonnet-4-5",  # default generation/annotation
    "high": "claude-opus-4-1",      # structurally-critical generation (rare, expensive)
}


def pick_llm(level: str = "medium", **kwargs) -> ChatAnthropic:
    """Return a Claude chat model tiered by task complexity.

    level: 'low' | 'medium' | 'high'
    Model names above are placeholders for whatever the current Claude
    lineup is at build time — check Anthropic's docs for current model IDs
    before running this for real.
    """
    level = level.lower()
    if level not in _MODELS:
        raise ValueError(f"Unknown model level: {level}")
    return ChatAnthropic(
        model=_MODELS[level],
        temperature=kwargs.pop("temperature", 0),
        **kwargs,
    )
```

**Test:** `tests/test_pick_llm.py`
```python
import pytest
from qualume.llm.pick_llm import pick_llm


def test_pick_llm_rejects_unknown_level():
    with pytest.raises(ValueError):
        pick_llm("ultra")


def test_pick_llm_returns_object_for_each_valid_level():
    for level in ("low", "medium", "high"):
        llm = pick_llm(level)
        assert llm is not None
```

**Verify:** `uv run pytest tests/test_pick_llm.py -v` — both pass. (This test doesn't call the live
API, just checks object construction — keep it that way so tests don't burn API credits or need
network in CI.)

**Note:** before Phase 1 generation actually runs, manually verify the exact current Claude model
IDs against Anthropic's docs — model names drift and this file will need real IDs, not placeholders.

### Task 0.9 — Minimal eval harness skeleton
**Files:** `src/qualume/eval/harness.py`, `src/qualume/eval/golden_sets/ig_static_write_v1.jsonl`

Golden set format (one JSON object per line) — hand-curate 10-15 examples since there's no real
performance data yet to derive them from:
```json
{"brief": "Promote a $20 productivity journal, audience: young professionals", "must_include": ["journal"], "must_avoid": ["guaranteed results", "doctor"], "min_quality_notes": "hook should state a concrete problem, not a vague claim"}
```

```python
# src/qualume/eval/harness.py
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvalResult:
    stage: str
    golden_set_version: str
    prompt_version: str
    total: int
    passed: int

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_constraint_checks(generated_text: str, example: dict) -> bool:
    """Deterministic checks only at this stage — no LLM judge yet.
    Returns True if generated_text satisfies must_include/must_avoid."""
    text_lower = generated_text.lower()
    for must in example.get("must_include", []):
        if must.lower() not in text_lower:
            return False
    for avoid in example.get("must_avoid", []):
        if avoid.lower() in text_lower:
            return False
    return True
```

**Test:** `tests/test_eval_harness.py`
```python
from qualume.eval.harness import run_constraint_checks


def test_run_constraint_checks_fails_on_missing_required_term():
    example = {"must_include": ["journal"], "must_avoid": []}
    assert run_constraint_checks("Buy this amazing planner today", example) is False


def test_run_constraint_checks_fails_on_banned_term():
    example = {"must_include": [], "must_avoid": ["guaranteed results"]}
    assert run_constraint_checks("This journal gives guaranteed results", example) is False


def test_run_constraint_checks_passes_clean_text():
    example = {"must_include": ["journal"], "must_avoid": ["guaranteed results"]}
    assert run_constraint_checks("This journal helps you plan your week", example) is True
```

**Verify:** `uv run pytest tests/test_eval_harness.py -v` — all 3 pass.

This is intentionally the simplest possible version (deterministic string checks only — no LLM
judge yet). The LLM-judge layer gets added in Phase 1 once there's an actual Write-engine prompt to
score; building the harness shell first means the judge can be plugged in without redesigning
anything.

### Task 0.10 — Seed the `ig_static` playbook as a Hermes skill (v0, explicitly unproven)
Create a Hermes skill `ig-static-playbook` whose content is a structured rubric, version `v0`,
tagged `source: external_field_notes` per the engineering plan's Appendix A discipline — this is the
seed heuristic set, not proven architecture. Store the same rubric as a row in the `playbooks` table
so the DB and the skill agree.

```sql
INSERT INTO playbooks (workflow_type, version, rubric_json, approved_by, approved_at)
VALUES (
  'ig_static', 'v0',
  '{"hooks": ["state a concrete problem in plain language, avoid vague claims"],
    "format": ["one clear headline + one supporting line, avoid clutter"],
    "social_proof": ["use sparingly, one credible data point beats three vague ones"],
    "source": "external_field_notes", "tier": "proposed"}'::jsonb,
  'dylan', now()
);
```

**Verify:** `select workflow_type, version, rubric_json->'source' from playbooks;` returns the row.

### Phase 0 exit check
```bash
cd ~/qualume
docker compose ps                    # postgres + n8n both running
uv run pytest tests/ -v              # all tests pass
docker exec -it qualume-postgres psql -U qualume -d qualume -c "\dt"   # 8 tables
docker exec -it qualume-postgres psql -U qualume -d qualume -c "select version from playbooks;"  # v0 row
```
All four must succeed before moving to Phase 1.

---

## 2. Phase 1 — Tracer bullet (IG statics, manual publish, no automated learning yet)

**Exit criteria:** you can run one command that generates an IG static batch (Write → Design), the
output passes the eval-harness gate, you manually review and post one to Instagram, and you can
manually log its performance back into `performance_snapshots` with full provenance (which playbook
version, which prompt version produced it) traceable end-to-end.

### Task 1.1 — Write engine: ideation + copy generation for ig_static
**Files:** `src/qualume/graphs/write_ig_static.py`

Build a LangGraph graph (stateful orchestration pattern, Pydantic state) with nodes:
`load_playbook → ideate_angle → generate_hook_body_cta → self_critique_against_playbook`.
Use `pick_llm("low")` for the critique pass, `pick_llm("medium")` for generation.
Output: a list of `ScriptVariant` objects, each tagged `explore_or_exploit` per a simple fixed ratio
parameter (start at 70% exploit / 30% explore since there's no real performance data yet to justify
anything fancier — a bandit policy is premature until Phase 2+).

**Test:** `tests/test_write_graph.py` — mock the LLM call (don't hit the real API in this test);
assert the graph produces the correct count of variants and that each validates against
`ScriptVariant`.

**Manual smoke test (hits real Claude API once, costs a few cents):**
```bash
uv run python -m qualume.graphs.write_ig_static --brief "promote a $20 productivity journal" --count 5
```
Expect 5 valid `ScriptVariant` objects printed, each with distinct hook text (diversity check — if
they're near-duplicates, the prompt needs work before moving on).

### Task 1.2 — Design engine: static image renderer
**Files:** `src/qualume/graphs/design_ig_static.py`, `data/templates/*.png` (2-3 simple template
backgrounds to start)

For Phase 1, use **Pillow-based template rendering** (not an image-gen model) — place headline/body
text from a `ScriptVariant` onto a pre-made template background. This is the simplest, most
controllable, zero-API-cost option for a first static asset, and keeps the Design engine's render
step fully deterministic (same input → same output, satisfying the render-determinism test from the
v2 plan). Swapping in an actual design-gen model/service is a fast-follow once the pipeline shape is
proven — don't let image-gen tool selection block the tracer bullet.

Pipeline: `select_template → place_headline_and_body → brand_consistency_check → render_png`.

**Test:** `tests/test_design_graph.py` — given a fixed `ScriptVariant` and a fixed template, assert
the rendered output has deterministic byte-identical output across two runs (seed everything), and
assert text actually fits within template bounds (no overflow/cutoff).

### Task 1.3 — Wire Write → Design → eval gate into one batch script
**Files:** `src/qualume/run_batch.py` (a thin script, not a graph — just sequences the two graphs and
the eval harness)

```python
# Pseudocode shape
briefs = [...]
variants = write_ig_static.run(brief, count=5)
for v in variants:
    passed = run_constraint_checks(v.hook_text + v.body_text, golden_example)
    if not passed:
        continue  # reject, do not pass to Design
    asset = design_ig_static.run(v)
    insert_content_item(asset, playbook_version="v0", prompt_version="v1")
```

**Verify:** running this produces N rendered PNGs in `data/assets/`, each with a matching
`content_items` + `creative_dna` row in Postgres, `status='draft'`.

```bash
docker exec -it qualume-postgres psql -U qualume -d qualume -c \
  "select id, workflow_type, status, playbook_version_used from content_items order by created_at desc limit 5;"
```

### Task 1.4 — Manual review + manual publish
No automation here on purpose — you look at the rendered PNGs in `data/assets/`, pick the best one,
post it to Instagram yourself through the app. Then mark it published:
```sql
UPDATE content_items SET status='published', published_at=now() WHERE id='<the one you posted>';
```
This manual step is intentional for Phase 1 — automating IG posting requires the same Graph API
access that automating ingestion does (see Task 1.6), and isn't worth building before you've proven
the Write/Design output is even worth posting.

### Task 1.5 — Manual performance entry (Phase 1a — unblocked path)
**Files:** `src/qualume/ingest/manual_entry.py`

```python
import argparse
from qualume.db.connection import get_connection


def log_snapshot(content_item_id: str, views: int, likes: int, saves: int,
                  comments: int, shares: int = 0, follows_gained: int = 0):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO performance_snapshots
               (content_item_id, views, likes, saves, comments, shares, follows_gained, entry_method)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 'manual')""",
            (content_item_id, views, likes, saves, comments, shares, follows_gained),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("content_item_id")
    parser.add_argument("--views", type=int, default=0)
    parser.add_argument("--likes", type=int, default=0)
    parser.add_argument("--saves", type=int, default=0)
    parser.add_argument("--comments", type=int, default=0)
    parser.add_argument("--shares", type=int, default=0)
    parser.add_argument("--follows-gained", type=int, default=0)
    args = parser.parse_args()
    log_snapshot(args.content_item_id, args.views, args.likes, args.saves,
                 args.comments, args.shares, args.follows_gained)
    print("Snapshot logged.")
```

Use this daily for the first published post:
```bash
uv run python -m qualume.ingest.manual_entry <content_item_id> --views 340 --likes 28 --saves 4 --comments 2
```

**Verify:** querying `performance_snapshots` shows a growing time series (append-only, multiple rows
over days) for that content_item_id, `entry_method='manual'`.

### Task 1.6 — Set up real Instagram Graph API access (Phase 1b — parallel, not blocking)
This is a setup/account-configuration task, not a coding task, and has external-approval latency —
start it now so it's ready by the time Phase 2 needs automated ingestion at volume:
1. Convert the Instagram account to a **Professional (Business or Creator)** account in the IG app
   settings.
2. Link it to a **Facebook Page** (required for Graph API access even though you're not using FB
   itself).
3. Create a **Meta Developer App** at developers.facebook.com, add the **Instagram Graph API**
   product.
4. Generate a long-lived access token for the account (via Graph API Explorer or the app's token
   tool).
5. Store the token in `.env` as `IG_GRAPH_API_TOKEN` (never commit it).
6. Leave `src/qualume/ingest/ig_graph_api.py` as a stub for now — implement it once the token exists
   and there's actually published content with stats to pull.

**This task has no "verify" step yet** — it's infrastructure setup that unblocks later work, not
something with a pass/fail test today. Revisit once you have the token.

### Phase 1 exit check
```bash
uv run pytest tests/ -v                     # all green, including new Write/Design tests
# at least one content_item row with status='published'
docker exec -it qualume-postgres psql -U qualume -d qualume -c \
  "select ci.id, ci.status, count(ps.id) as snapshots from content_items ci \
   left join performance_snapshots ps on ps.content_item_id = ci.id \
   where ci.status='published' group by ci.id, ci.status;"
```
Expect at least one published row with 1+ snapshot rows accumulating over time.

---

## 3. What happens after Phase 1 (pointer back to v2 plan, not re-specified here)

Once Phase 1's exit criteria are met — real (even if small/manual) performance data exists, traced
end-to-end to specific playbook/prompt versions — Phase 2 of `v2-engineering-plan.md` applies as
written: build synthetic-audience scoring, back-test its calibration, run the mining pipeline's
false-positive check, and produce the first real statistically-validated insight. Do not start Phase
2 work before Phase 1's exit checks above are all green — there's no real data to mine yet otherwise.

---

## 4. Immediate TODO (in order)

1. `brew install --cask orbstack` (Task 0.1) — **needs your approval, installs system software.**
2. Run Tasks 0.2–0.10 in order (scaffolding, schema, contracts, model picker, eval harness skeleton,
   seed playbook) — all safe, no system-level side effects beyond the project folder + Docker
   containers.
3. Set your real `ANTHROPIC_API_KEY` in `.env` by hand.
4. Run Phase 0 exit check — confirm all four commands pass before touching Phase 1.
5. Build Tasks 1.1–1.5 (Write, Design, batch runner, manual publish, manual performance entry).
6. In parallel with step 5, start Task 1.6 (Instagram Business account conversion + Meta Developer
   App) since it has external latency — don't wait to start it.
7. Run Phase 1 exit check.
