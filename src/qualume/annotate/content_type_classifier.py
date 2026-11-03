"""content_type (selling/organic) classifier — Claude-only, same text-
classification shape as copy_classifier (no video, no new Gemini call,
works off copy already extracted into external_creative_dna/
external_content_items). Real gap this closes (migration 017): niche
scopes WHAT TOPIC a video is about; content_type scopes WHETHER it's a
selling/ad piece vs organic content — a second, independent axis.

Meta Ad Library videos are selling-by-definition and get seeded
automatically at ingest (meta_ads_reader.parse_ad_entry) with zero
classifier calls needed. TikTok content needs this real classification
pass instead, since an organic TikTok account (especially a DTC brand
or TikTok Shop seller) can post BOTH organic story-time content AND
hard-sell pieces — content_type can't be inferred from the account
alone.

Deliberately a SIMPLE binary judgment, not a new multi-label taxonomy
like the 22 copy techniques — content_type is one either/or call about
the whole video, not a set of co-occurring signals.
"""
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field, ValidationError

from qualume.llm.pick_llm import pick_llm

ANALYSIS_PROMPT_VERSION = "v1"

ANNOTATION_PROMPT = """\
You are classifying a short-form social video's COPY (spoken words, on-screen \
text, and caption) for a content-pattern research database. You are given \
text only — no video, no image, no audio.

Classify this video as exactly ONE of:
- "selling": the video is a paid ad or a direct sales piece — it names/shows \
a specific product, makes a purchase ask or CTA (shop now, link in bio, buy, \
discount code), or is clearly structured as a sales pitch (problem -> \
product -> offer -> CTA), even if the tone is casual/organic-feeling. A \
"soft sell" piece with a product mention and purchase CTA still counts as \
selling, not organic.
- "organic": the video is personal/story/entertainment/educational content \
with NO product promotion, purchase ask, or sales CTA — even if it's posted \
by a brand's own account (e.g. a day-in-the-life clip, a myth-busting \
explainer, a relatable storytime with no product mentioned).

Judge ONLY what is actually in the text below — do not guess at intent you \
cannot see in the text itself. If the text gives genuinely ambiguous signal \
(e.g. too short/sparse to tell), still pick your best-supported single \
answer and note the ambiguity in confidence_note; do not invent a third \
category.

TEXT TO CLASSIFY:
{text}

Respond with ONLY the structured classification.\
"""


class ContentTypeClassification(BaseModel):
    """Structured output for the content_type classifier — see module
    docstring for the selling/organic distinction."""
    content_type: str = Field(description="Either 'selling' or 'organic'")
    confidence_note: str = Field(
        description="One short sentence citing the specific text that drove this judgment"
    )


class ContentTypeResult(NamedTuple):
    content_type: Optional[str]
    raw_response: dict
    validation_error: Optional[str]


def _build_prompt(hook_text, full_transcript, on_screen_text, caption) -> str:
    parts = []
    if hook_text:
        parts.append(f"HOOK TEXT: {hook_text}")
    if full_transcript:
        transcript_text = " ".join(
            seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in full_transcript
        )
        parts.append(f"FULL SPOKEN TRANSCRIPT: {transcript_text}")
    if on_screen_text:
        text_items = " | ".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in on_screen_text
        )
        parts.append(f"ON-SCREEN TEXT: {text_items}")
    if caption:
        parts.append(f"CAPTION: {caption}")
    return ANNOTATION_PROMPT.format(text="\n".join(parts))


def classify_content_type(
    hook_text: str | None,
    full_transcript: list | None,
    on_screen_text: list | None,
    caption: str | None,
) -> ContentTypeResult:
    """Classify whatever copy IS available — same precondition and
    format-agnostic discipline as copy_classifier.classify_copy (plain
    text/list args, no DB, reusable across any table's shape)."""
    if not any([hook_text, full_transcript, on_screen_text, caption]):
        raise ValueError(
            "classify_content_type needs at least one of hook_text, full_transcript, "
            "on_screen_text, or caption — nothing to classify"
        )

    prompt = _build_prompt(hook_text, full_transcript, on_screen_text, caption)
    llm = pick_llm("low").with_structured_output(ContentTypeClassification)

    try:
        annotation = llm.invoke(prompt)
    except ValidationError as e:
        return ContentTypeResult(content_type=None, raw_response={}, validation_error=str(e))

    return ContentTypeResult(
        content_type=annotation.content_type,
        raw_response=annotation.model_dump(),
        validation_error=None,
    )


def classify_and_store_content_type(external_content_item_id: str) -> str:
    """Pull an external TikTok video's already-stored copy (same
    latest-version-wins discipline as copy_classifier.classify_and_
    store_external) and write the result directly to external_content_
    items.content_type (migration 017) — no separate results table,
    since this is a single scalar field on the item itself, not a
    multi-row analysis like copy technique occurrences.

    Raises ValueError if the content item doesn't exist, or has no
    external_creative_dna row yet (nothing to classify from)."""
    from qualume.db.connection import get_connection

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

    result = classify_content_type(
        hook_text=hook_text, full_transcript=full_transcript,
        on_screen_text=on_screen_text, caption=caption,
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE external_content_items SET content_type = %s WHERE id = %s",
                (result.content_type, external_content_item_id),
            )
        conn.commit()
    finally:
        conn.close()

    return external_content_item_id
