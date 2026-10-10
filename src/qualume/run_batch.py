"""Batch runner: Write -> eval gate -> Design -> persist to Postgres.

Phase 1 tracer-bullet script (not a LangGraph graph itself — this sequences
two independently-compiled graphs plus the eval harness, matching the v2
plan's "each pillar independently testable" principle: Write and Design stay
decoupled, this script is the only thing that knows about both).
"""

import argparse
import json
import re
import uuid

from qualume.graphs import write_ig_static, design_ig_static
from qualume.models.contracts import CampaignBrief, DesignRequest
from qualume.eval.harness import load_golden_set, run_constraint_checks
from qualume.db.connection import get_connection

GOLDEN_SET_PATH = "src/qualume/eval/golden_sets/ig_static_write_v1.jsonl"


def build_universal_avoid_list(golden_set: list[dict]) -> list[str]:
    """Union of every golden-set example's must_avoid terms.

    These are universal compliance/quality bars (banned claims like "cures",
    "guaranteed results") that should apply to ANY brief, regardless of
    product — unlike must_include terms, which are necessarily specific to
    whatever product a given golden example happens to describe and do not
    generalize to an arbitrary new brief.
    """
    seen: set[str] = set()
    for example in golden_set:
        for term in example.get("must_avoid", []):
            seen.add(term.lower())
    return sorted(seen)


_KEYWORD_STOPWORDS = {"for", "to", "that", "who", "which", "with"}


def _extract_product_keyword(brief_text: str) -> str | None:
    """Best-effort extraction of the core product noun from a brief string.

    Heuristic, not exact: takes the word immediately before the first
    "stopword" (for/to/that/who/which/with) in the clause before any comma,
    since product nouns are typically followed by a qualifier phrase
    ("... wallet for men", "... journal that helps you focus"). Falls back
    to the clause's last word if no stopword is found. Used only to build a
    per-batch must_include check so the gate verifies the generated copy
    actually mentions the thing being promoted, instead of borrowing an
    unrelated golden example's must_include list (the bug this replaces —
    see docs/session-context.md for the writeup).
    """
    first_clause = brief_text.split(",")[0].strip()
    words = re.findall(r"[A-Za-z]+", first_clause)
    if not words:
        return None
    for i, word in enumerate(words):
        if word.lower() in _KEYWORD_STOPWORDS and i > 0:
            return words[i - 1].lower()
    return words[-1].lower()


def build_eval_example_for_brief(brief_text: str, universal_avoid: list[str]) -> dict:
    """Construct a per-batch eval example: dynamic must_include (derived from
    this specific brief) + the universal must_avoid list (applies to all
    briefs). Does NOT depend on finding a matching golden-set entry."""
    keyword = _extract_product_keyword(brief_text)
    return {
        "must_include": [keyword] if keyword else [],
        "must_avoid": universal_avoid,
    }


def insert_content_item(design_output, batch_id: str, playbook_version: str, write_prompt_version: str) -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO content_items
                   (workflow_type, platform, status, batch_id, playbook_version_used,
                    prompt_versions_used, lane, asset_path)
                   VALUES (%s, %s, 'draft', %s, %s, %s, 'owned', %s)
                   RETURNING id""",
                (
                    design_output.workflow_type,
                    "instagram",
                    batch_id,
                    playbook_version,
                    json.dumps({
                        "ideation": write_prompt_version,
                        "write": write_prompt_version,
                        "design": design_ig_static.PROMPT_VERSION,
                    }),
                    design_output.asset_path,
                ),
            )
            content_item_id = cur.fetchone()[0]

            dna = design_output.creative_dna
            cur.execute(
                """INSERT INTO creative_dna
                   (content_item_id, hook_text, format_tag, script_structure_tags,
                    cta_type, caption_style, social_proof_used)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    content_item_id,
                    dna.hook_text,
                    dna.format_tag,
                    dna.script_structure_tags,
                    dna.cta_type,
                    dna.caption_style,
                    dna.social_proof_used,
                ),
            )
        conn.commit()
        return str(content_item_id)
    finally:
        conn.close()


def run_batch(brief_text: str, goal: str, count: int, output_dir: str = "data/assets") -> list[str]:
    batch_id = str(uuid.uuid4())
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    universal_avoid = build_universal_avoid_list(golden_set)
    eval_example = build_eval_example_for_brief(brief_text, universal_avoid)

    brief = CampaignBrief(
        product_or_topic=brief_text,
        offer_or_goal=goal,
        playbook_version="pending",  # resolved inside the Write graph
    )
    variants = write_ig_static.run(brief, count=count)

    content_item_ids = []
    rejected = 0
    for variant in variants:
        full_text = f"{variant.hook_text} {variant.body_text} {variant.cta_text}"
        if not run_constraint_checks(full_text, eval_example):
            rejected += 1
            continue

        design_req = DesignRequest(script_variant=variant, workflow_type="ig_static")
        design_output = design_ig_static.run(design_req, output_dir=output_dir)

        content_item_id = insert_content_item(
            design_output,
            batch_id=batch_id,
            playbook_version=variant.playbook_version,
            write_prompt_version=variant.prompt_version,
        )
        content_item_ids.append(content_item_id)

    print(f"Batch {batch_id}: {len(variants)} written, "
          f"{rejected} rejected by eval gate, {len(content_item_ids)} rendered+persisted.")
    return content_item_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full Write->Design->persist batch.")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--goal", default="drive engagement and sales")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    ids = run_batch(args.brief, args.goal, args.count)
    print("\ncontent_item ids:")
    for i in ids:
        print(" ", i)
