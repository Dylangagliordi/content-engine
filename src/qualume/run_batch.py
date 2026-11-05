"""Batch runner: Write -> eval gate -> Design -> score -> persist to Postgres.

Phase 1 tracer-bullet script (not a LangGraph graph itself — this sequences
two independently-compiled graphs plus the eval harness, matching the v2
plan's "each pillar independently testable" principle: Write and Design stay
decoupled, this script is the only thing that knows about both).

SCORING (added 2026-11-04, real fix for a real gap): learn.prediction's
trained model existed and was fully verified against production data, but
was never actually CALLED anywhere in the real generation pipeline —
predict_performance_for_annotation's own module docstring describes its
whole purpose as "predicting a new draft's likely performance before
publish", and that step never happened here until now. Each generated
ScriptVariant is now scored against its niche's saved model (if one exists —
see learn.model_persistence) BEFORE persisting, and the score is written to
content_items.predicted_score (migration 020). This is observational only
— does NOT gate publishing or filter which variants get persisted (unlike
the eval constraint-check gate above, which does reject variants); a
low predicted_score is a signal for a human to weigh, not an automatic
rejection, same "don't silently overreach into decisions a human should
make" discipline as the eval-gate/playbook honesty principles in CLAUDE.md.
NULL (never a fabricated number) when no real saved model exists for the
given niche, or no niche was given at all.
"""

import argparse
import json
import re
import uuid

from qualume.graphs import write_ig_static, design_ig_static
from qualume.models.contracts import CampaignBrief, DesignRequest, ScriptVariant
from qualume.eval.harness import load_golden_set, run_constraint_checks
from qualume.db.connection import get_connection
from qualume.learn.model_persistence import has_saved_predictor, load_predictor
from qualume.learn.prediction import predict_performance_for_annotation

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


def score_variant_if_model_available(variant: ScriptVariant, niche: str | None) -> float | None:
    """Score one generated ScriptVariant against its niche's saved
    prediction model (learn.model_persistence), if one exists. Returns
    None — never a fabricated number — when niche is None (no niche was
    given to this batch run) or no model has been saved for that niche
    yet (below the training floor, or scripts/train_and_save_all_predictors.py
    hasn't been run for it). This is the real wiring gap closed 2026-11-04:
    the trained model existed and was verified, but nothing in the real
    generation pipeline ever called it before persisting a draft.

    Builds the annotation dict from the variant's own real fields
    (hook_type comes from the DESIGN stage's CreativeDNA, not available
    yet at this Write-stage point — see run_batch's call site for how
    hook_type is filled in from the design_output instead) — hook_text
    is what actually drives the score for an IG static today, since
    statics have no shot_breakdown/full_transcript/duration at all (see
    learn.prediction's own module docstring on why hook_text is the
    cross-workflow-type merge point)."""
    if niche is None:
        return None
    if not has_saved_predictor(niche):
        return None
    trained = load_predictor(niche)
    annotation = {
        "hook_type": None, "format_tag": None, "energy_level": None,
        "shot_breakdown": [], "full_transcript": [], "video_duration_seconds": None,
        "hook_text": variant.hook_text,
    }
    return predict_performance_for_annotation(trained, annotation)


def insert_content_item(design_output, batch_id: str, playbook_version: str, write_prompt_version: str,
                         niche: str | None = None, predicted_score: float | None = None) -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO content_items
                   (workflow_type, platform, status, batch_id, playbook_version_used,
                    prompt_versions_used, lane, asset_path, niche, predicted_score)
                   VALUES (%s, %s, 'draft', %s, %s, %s, 'owned', %s, %s, %s)
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
                    niche,
                    predicted_score,
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


def run_batch(brief_text: str, goal: str, count: int, output_dir: str = "data/assets",
              niche: str | None = None) -> list[str]:
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
    scored_count = 0
    for variant in variants:
        full_text = f"{variant.hook_text} {variant.body_text} {variant.cta_text}"
        if not run_constraint_checks(full_text, eval_example):
            rejected += 1
            continue

        design_req = DesignRequest(script_variant=variant, workflow_type="ig_static")
        design_output = design_ig_static.run(design_req, output_dir=output_dir)

        predicted_score = score_variant_if_model_available(variant, niche=niche)
        if predicted_score is not None:
            scored_count += 1

        content_item_id = insert_content_item(
            design_output,
            batch_id=batch_id,
            playbook_version=variant.playbook_version,
            write_prompt_version=variant.prompt_version,
            niche=niche,
            predicted_score=predicted_score,
        )
        content_item_ids.append(content_item_id)

    print(f"Batch {batch_id}: {len(variants)} written, "
          f"{rejected} rejected by eval gate, {len(content_item_ids)} rendered+persisted, "
          f"{scored_count} scored against a real saved model.")
    return content_item_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full Write->Design->persist batch.")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--goal", default="drive engagement and sales")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--niche", default=None,
                         help="Real niche to score each draft against (see "
                              "scripts/train_and_save_all_predictors.py) — omit to skip scoring.")
    args = parser.parse_args()

    ids = run_batch(args.brief, args.goal, args.count, niche=args.niche)
    print("\ncontent_item ids:")
    for i in ids:
        print(" ", i)
