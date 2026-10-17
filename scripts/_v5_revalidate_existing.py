"""Re-validate existing external_creative_dna rows against the CURRENT
ExternalVideoAnnotationV5 (no new Gemini call - the raw_gemini_response
is already stored and unchanged). Used once, after removing v5's wrong
duration-ceiling validator, to fix the 25 rows that were incorrectly
rejected under the old rule but are real, good data under the current one.
"""
import json

from pydantic import ValidationError

from qualume.annotate.gemini_watcher import (
    _check_coverage,
    derive_hook_fields,
)
from qualume.db.connection import get_connection
from qualume.models.contracts import ExternalVideoAnnotationV5
from psycopg2.extras import Json

with open("/tmp/v5_reannotate_ids.txt") as f:
    item_ids = [line.strip() for line in f if line.strip()]

print(f"=== re-validating {len(item_ids)} rows against the fixed schema ===", flush=True)

fixed, still_bad = 0, 0

for i, item_id in enumerate(item_ids, 1):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ecd.id, ecd.raw_gemini_response, eci.duration_seconds
                   FROM external_creative_dna ecd
                   JOIN external_content_items eci ON eci.id = ecd.external_content_item_id
                   WHERE ecd.external_content_item_id = %s AND ecd.analysis_prompt_version = 'v5'""",
                (item_id,),
            )
            dna_id, raw_response, duration_seconds = cur.fetchone()

        try:
            ann = ExternalVideoAnnotationV5(**raw_response)
        except ValidationError as e:
            print(f"[{i}/{len(item_ids)}] {item_id} STILL FAILS: {e}", flush=True)
            still_bad += 1
            continue

        hook_text, visual_hook_description = derive_hook_fields(
            ann.script_structure_timeline, ann.full_transcript, ann.shot_breakdown
        )
        warnings = _check_coverage(ann, duration_seconds)
        coverage_warning = "; ".join(warnings) if warnings else None

        with conn.cursor() as cur:
            cur.execute(
                """UPDATE external_creative_dna SET
                    validation_error = NULL,
                    coverage_warning = %s,
                    hook_type = %s, hook_text = %s, visual_hook_description = %s,
                    format_tag = %s, script_structure_tags = %s, cta_type = %s,
                    caption_style = %s, social_proof_used = %s, energy_level = %s,
                    tone_category = %s, voice_delivery_style = %s,
                    script_structure_timeline = %s, on_screen_text = %s,
                    emotion_timeline = %s, pacing_qualitative = %s, music_mood_v2 = %s,
                    shot_breakdown = %s, full_transcript = %s
                   WHERE id = %s""",
                (
                    coverage_warning,
                    ann.hook_type, hook_text, visual_hook_description,
                    ann.format_tag, ann.script_structure_tags, ann.cta_type,
                    ann.caption_style, ann.social_proof_used, ann.energy_level,
                    ann.tone_category, ann.voice_delivery_style,
                    Json([b.model_dump() for b in ann.script_structure_timeline]),
                    Json([t.model_dump() for t in ann.on_screen_text]),
                    Json([e.model_dump() for e in ann.emotion_timeline]),
                    ann.pacing_qualitative, ann.music_mood,
                    Json([s.model_dump() for s in ann.shot_breakdown]),
                    Json([t.model_dump() for t in ann.full_transcript]),
                    dna_id,
                ),
            )
        conn.commit()
        print(f"[{i}/{len(item_ids)}] {item_id} -> {dna_id} FIXED"
              + (f" (coverage_warning: {coverage_warning})" if coverage_warning else ""), flush=True)
        fixed += 1
    finally:
        conn.close()

print(f"\n=== DONE: {fixed} fixed, {still_bad} still genuinely fail validation ===", flush=True)
