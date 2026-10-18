"""Copy-technique classifier — Claude-only, NOT the Gemini video-watching
exception. A text classification pass over copy that's ALREADY been
extracted somewhere else (full_transcript/on_screen_text/hook_text from
a video, or just on-screen/body copy from a static) — no video, no
audio, no new Gemini upload, ever.

classify_copy() is deliberately FORMAT-AGNOSTIC: plain text/list
arguments, no DB dependency, no assumption about which table the copy
came from. It works unchanged for:
  - external organic TikTok video (full_transcript + on_screen_text +
    hook_text all present — real data exists today, see
    classify_and_store_external)
  - external paid/VSL/advertorial video (same shape, just different
    content — the 19-flag taxonomy was deliberately kept broad enough
    to cover both organic and paid framing, e.g. uses_risk_reversal and
    discovery_moment_framing are real techniques in VSLs/advertorials
    that organic clips use less often, but the classifier doesn't care
    which — it just reports what's actually there)
  - a static ad or static organic post (on_screen_text/caption only, no
    transcript/hook_text at all — not yet ingested anywhere in this
    project, but the function already handles this shape)
  - OWNED content (content_items/creative_dna) once real owned copy
    exists — not wired up yet (classify_and_store_external is external-
    only for now), but nothing about classify_copy() itself is
    external-specific.

19 boolean flags (see models.contracts.CopyAnalysis for the full field
list + descriptions), grounded in real reference material: Winning Hook
Criteria (7 hook levers), Features/Benefits/Outcomes framework, the VOC
doc's own real customer language, and the VSL/Advetorial breakdowns
(mechanism explanation, discovery framing, risk reversal, status shift).
Multi-label by design — a real video/static can trip several at once.
"""
from typing import NamedTuple, Optional

from pydantic import ValidationError

from qualume.llm.pick_llm import pick_llm
from qualume.models.contracts import CopyAnalysis

ANALYSIS_PROMPT_VERSION = "v1"

ANNOTATION_PROMPT = """\
You are analyzing the COPY (spoken words, on-screen text, and caption) of a \
short-form social video or static ad/post for a content-pattern research \
database. You are given text only — no video, no image, no audio. Judge \
ONLY the copy techniques actually present in the text below; do not guess \
at anything you cannot see in the text itself.

For EACH of the 19 fields, answer true if the technique is clearly present \
anywhere in the given text, false if it is clearly absent, or leave it \
null if the available text genuinely doesn't give you enough to tell \
(e.g. very little text, or the field depends on something like spoken \
delivery you cannot observe from text alone). A real piece of copy can be \
true on MANY fields at once — this is not a single-choice classification.

Fields (grouped by where they typically occur, though copy doesn't have \
to follow this structure):

HOOK-LEVEL (the opening few seconds/first line):
- hook_curiosity_open_loop: teases a result/reveal without naming the mechanism
- hook_pain_callout: names the viewer's felt pain directly
- hook_promise_without_mechanism: states the outcome/benefit, withholds the "how"
- hook_specificity: numbers/timeframes/quantities in the opening
- hook_credibility_signal: authority/expert/clinical claim right in the opening
- hook_pattern_interrupt: a belief-breaking/"sacred cow smash" opener
- hook_status_shift: "people you look up to aren't doing what you do" framing

PROBLEM/AGITATION:
- mentions_customer_problem: names the problem the product solves, in any form
- uses_voc_language: uses raw/colloquial customer phrasing, not clinical/marketing language
- agitates_pain_emotionally: escalates consequence/shame/fear, not just states the problem
- burns_alternatives: explicitly dismisses other things the viewer already tried

MECHANISM/SOLUTION:
- explains_mechanism: explains WHY/HOW it works, not just that it works
- discovery_moment_framing: introduces the product as a casual/organic discovery, not a hard pitch
- uses_sensory_visual_language: translates a benefit into a felt/visual metaphor
- features_framed_as_benefits: a feature is explicitly tied to why it matters to the viewer

PROOF/CREDIBILITY:
- uses_social_proof: testimonials, review counts, "X customers" claims
- uses_authority_or_expert_proof: credentials, institutions, "doctor/nurse/herbalist reveals"
- uses_before_after_contrast: explicit then-vs-now framing

CLOSE/CTA:
- creates_urgency: time pressure, scarcity, "while it's still available"
- uses_risk_reversal: guarantee, "no questions asked", free trial
- cta_direct_address: direct "you" framing in the call to action
- promises_identity_or_status_shift: the ask is framed as becoming a different kind of person, not just buying

Return ONLY the structured fields. Do not invent text that isn't given to you.
"""


class ClassificationResult(NamedTuple):
    annotation: Optional[CopyAnalysis]
    raw_response: dict
    validation_error: Optional[str]


def _build_prompt(hook_text, full_transcript, on_screen_text, caption) -> str:
    parts = [ANNOTATION_PROMPT, "\n--- COPY TO ANALYZE ---\n"]
    if hook_text:
        parts.append(f"HOOK TEXT: {hook_text}\n")
    if full_transcript:
        transcript_text = " ".join(
            seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in full_transcript
        )
        parts.append(f"FULL SPOKEN TRANSCRIPT: {transcript_text}\n")
    if on_screen_text:
        text_items = " | ".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in on_screen_text
        )
        parts.append(f"ON-SCREEN TEXT: {text_items}\n")
    if caption:
        parts.append(f"CAPTION: {caption}\n")
    return "".join(parts)


def classify_copy(
    hook_text: str | None,
    full_transcript: list | None,
    on_screen_text: list | None,
    caption: str | None,
) -> ClassificationResult:
    """Classify whatever copy IS available — at least one of hook_text,
    full_transcript, on_screen_text, caption must be non-empty (raises
    ValueError otherwise, rather than silently classifying nothing).

    Deliberately takes plain text/list arguments, not a DB row or any
    particular table's shape — this is what makes it reusable across
    external video, external static, and (later) owned content without
    any changes to this function."""
    if not any([hook_text, full_transcript, on_screen_text, caption]):
        raise ValueError(
            "classify_copy needs at least one of hook_text, full_transcript, "
            "on_screen_text, or caption — nothing to classify"
        )

    prompt = _build_prompt(hook_text, full_transcript, on_screen_text, caption)
    llm = pick_llm("low").with_structured_output(CopyAnalysis)

    try:
        annotation = llm.invoke(prompt)
    except ValidationError as e:
        return ClassificationResult(annotation=None, raw_response={}, validation_error=str(e))

    return ClassificationResult(
        annotation=annotation,
        raw_response=annotation.model_dump(),
        validation_error=None,
    )


def classify_and_store_external(external_content_item_id: str) -> str:
    """Pull an external video's already-stored copy (hook_text/
    full_transcript/on_screen_text from external_creative_dna, caption
    from external_content_items — whatever annotation version exists,
    latest wins, same discipline as learn.mining) and store a new
    external_copy_analysis row.

    Raises ValueError if the content item doesn't exist, or if it has no
    external_creative_dna row at all yet (nothing to classify from —
    run gemini_watcher.annotate_and_store first)."""
    from qualume.db.connection import get_connection
    from psycopg2.extras import Json

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT caption FROM external_content_items WHERE id = %s",
                (external_content_item_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"external_content_items row {external_content_item_id} not found")
            caption = row[0]

            cur.execute(
                """SELECT hook_text, full_transcript, on_screen_text
                   FROM external_creative_dna
                   WHERE external_content_item_id = %s
                   ORDER BY analyzed_at DESC LIMIT 1""",
                (external_content_item_id,),
            )
            dna_row = cur.fetchone()
            if dna_row is None:
                raise ValueError(
                    f"external_content_items row {external_content_item_id} has no "
                    "external_creative_dna row yet — run gemini_watcher.annotate_and_store first"
                )
            hook_text, full_transcript, on_screen_text = dna_row
    finally:
        conn.close()

    result = classify_copy(
        hook_text=hook_text, full_transcript=full_transcript,
        on_screen_text=on_screen_text, caption=caption,
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if result.annotation is not None:
                ann = result.annotation
                cur.execute(
                    """INSERT INTO external_copy_analysis
                       (external_content_item_id,
                        hook_curiosity_open_loop, hook_pain_callout,
                        hook_promise_without_mechanism, hook_specificity,
                        hook_credibility_signal, hook_pattern_interrupt, hook_status_shift,
                        mentions_customer_problem, uses_voc_language,
                        agitates_pain_emotionally, burns_alternatives,
                        explains_mechanism, discovery_moment_framing,
                        uses_sensory_visual_language, features_framed_as_benefits,
                        uses_social_proof, uses_authority_or_expert_proof,
                        uses_before_after_contrast,
                        creates_urgency, uses_risk_reversal,
                        cta_direct_address, promises_identity_or_status_shift,
                        raw_response, validation_error,
                        analysis_model, analysis_prompt_version)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        external_content_item_id,
                        ann.hook_curiosity_open_loop, ann.hook_pain_callout,
                        ann.hook_promise_without_mechanism, ann.hook_specificity,
                        ann.hook_credibility_signal, ann.hook_pattern_interrupt, ann.hook_status_shift,
                        ann.mentions_customer_problem, ann.uses_voc_language,
                        ann.agitates_pain_emotionally, ann.burns_alternatives,
                        ann.explains_mechanism, ann.discovery_moment_framing,
                        ann.uses_sensory_visual_language, ann.features_framed_as_benefits,
                        ann.uses_social_proof, ann.uses_authority_or_expert_proof,
                        ann.uses_before_after_contrast,
                        ann.creates_urgency, ann.uses_risk_reversal,
                        ann.cta_direct_address, ann.promises_identity_or_status_shift,
                        Json(result.raw_response), None,
                        "claude", ANALYSIS_PROMPT_VERSION,
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO external_copy_analysis
                       (external_content_item_id, raw_response, validation_error,
                        analysis_model, analysis_prompt_version)
                       VALUES (%s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        external_content_item_id, Json(result.raw_response),
                        result.validation_error, "claude", ANALYSIS_PROMPT_VERSION,
                    ),
                )
            analysis_id = cur.fetchone()[0]
        conn.commit()
        return str(analysis_id)
    finally:
        conn.close()
