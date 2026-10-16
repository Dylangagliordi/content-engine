"""Gemini video-watching annotation — model-observed signals only.

Gemini is a deliberate, scoped exception to the Claude-only rule: it is used
ONLY here, for native audio+video understanding (see
plans/learn-engine-idea-sourcing-architecture.md, "How AI actually watches a
video"). Since v2, every signal — including music mood and pacing — comes
from this one call; there is no separate signal-processing job. Pacing is
qualitative only (Gemini samples video at 1 FPS, so exact cut counts are
unreliable). v3 adds per-shot visual composition and the full spoken
transcript, aiming for enough structured detail that a new video could be
briefed as stylistically/structurally equivalent — not just a similar hook.
"""
import json
import os
import time
from typing import NamedTuple

from google import genai
from google.genai import types
from pydantic import ValidationError

from qualume import config  # noqa: F401 — loads .env into os.environ
from qualume.db.connection import get_connection
from psycopg2.extras import Json

from qualume.models.contracts import ExternalVideoAnnotationV3

ANALYSIS_PROMPT_VERSION = "v3"

# gemini-3.8-flash is the newest/most capable and real on this key, but is
# returning persistent 503 ServerError specifically on video input as of
# 2026-10-15 (plain text succeeds on it; video fails identically across
# 3.8-flash, flash-latest, AND 3.7-flash — isolated via direct testing, not
# assumed). gemini-3.1-flash-lite verified live: uploads, polls ACTIVE, and
# returns an accurate video description. Using it until 3.8-flash's video path
# stabilizes — re-check periodically, same re-verify command pattern as before:
#   uv run python -c "from google import genai; import os; from dotenv import load_dotenv; load_dotenv(); print(genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.get(model='models/gemini-3.8-flash'))"
GEMINI_MODEL = "gemini-3.1-flash-lite"

FILE_POLL_INTERVAL_SECONDS = 2
FILE_POLL_TIMEOUT_SECONDS = 300

ANNOTATION_PROMPT = """\
You are annotating a short-form social video (TikTok) for a content-pattern \
research database. Watch the whole video, using BOTH the visuals and the audio \
(speech, music, sound effects), and describe what is actually observable. \
Do not guess at performance, intent, or anything not present in the video.

Fields:
- hook_type: the mechanism used in the first ~3 seconds to stop the scroll \
(snake_case, e.g. question, bold_claim, problem_statement, visual_reveal, \
pattern_interrupt, story_open).
- hook_text: the exact words spoken or shown on screen during the hook. Empty \
string if there are none.
- visual_hook_description: one or two sentences on what is on screen in the \
first ~3 seconds.
- format_tag: the overall format (snake_case, e.g. talking_head, \
voiceover_broll, skit, tutorial, slideshow, product_demo, green_screen).
- script_structure_tags: the ordered beats of the video (snake_case, e.g. \
hook, problem, demo, payoff, cta).
- cta_type: the call to action (snake_case, e.g. follow, comment, \
link_in_bio, shop, save, none). Use "none" if there is no call to action.
- caption_style: the style of on-screen text (e.g. auto_captions, \
bold_center, minimal, none).
- social_proof_used: true ONLY if the video shows reviews, testimonials, \
customer counts, press mentions, or expert endorsement.
- energy_level: an integer on this anchored scale, judged from pace of \
speech, movement, editing and music together:
  1 = calm, slow, meditative
  2 = relaxed, unhurried
  3 = moderate, conversational
  4 = upbeat, lively, quick
  5 = maximum intensity, frenetic
- tone_category: the single dominant tone, from the allowed values only.
- voice_delivery_style: how the voice is delivered (e.g. conversational \
direct-to-camera, whispered ASMR, fast-talking hype, text-to-speech, no voice).

Timestamped fields. All times are seconds from the start of the video \
(decimals allowed); every list must be in chronological order.
- script_structure_timeline: the same beats as script_structure_tags, each \
with start_sec and end_sec, covering the video from start to end.
- on_screen_text: every distinct piece of text shown on screen (captions, \
titles, overlays, stickers, labels), transcribed verbatim, each with the \
timestamp_sec it first appears. Do not include spoken words that are not \
shown as text. Empty list if there is no on-screen text.
- emotion_timeline: how the emotion of the video shifts over time, judged \
from facial expression, voice and music together. One entry at 0, then one \
for each noticeable shift, as a single lowercase word (e.g. curious, \
frustrated, surprised, relieved, excited, calm, nostalgic). A video with one \
steady emotion has a single entry.
- pacing_qualitative: the overall editing and visual pace — slow, medium or \
fast. Give a qualitative read, do not try to count cuts.
- pacing_rationale: one sentence on what you observed that justifies the \
pacing read (e.g. shot length, camera movement, speech speed).
- music_mood: mood tags for the music, from listening to the audio, lowercase \
(e.g. upbeat, melancholic, suspenseful, dreamy, aggressive, lofi). Empty list \
if there is no music.

Shot-level and transcript fields. These are the most granular fields — the \
goal is enough detail that someone could brief a NEW video with the same \
visual style and structure without rewatching this one.
- shot_breakdown: every distinct camera shot/setup, in chronological order, \
covering the whole video start to end (no gaps). For each shot give: \
start_sec, end_sec, shot_type (e.g. close_up, medium_shot, wide_shot, \
extreme_close_up, pov), camera_movement (e.g. static, handheld, pan, \
zoom_in, zoom_out, tracking), setting (where it takes place, e.g. kitchen, \
bedroom, outdoor park), color_palette (dominant colors observed, named not \
hex, e.g. "warm pinks and whites", "muted earth tones"), lighting (e.g. \
bright_natural, dim, backlit, studio, golden_hour), and subject_action (what \
the person/subject is doing, one short phrase). A beat from \
script_structure_timeline can span multiple shots if the camera setup \
changes within it.
- full_transcript: every spoken utterance in the video, in chronological \
order, transcribed verbatim, each with the start_sec it begins. This is the \
COMPLETE transcript, not just the hook — include everything said, not only \
the opening few seconds. Empty list if there is no speech.
"""


class AnnotationResult(NamedTuple):
    annotation: ExternalVideoAnnotationV3 | None  # None if schema validation failed
    raw_response: dict  # complete parsed Gemini response, including unmapped keys —
                         # ALWAYS populated when Gemini returned anything parseable,
                         # even if `annotation` is None. Nothing gets silently dropped.
    validation_error: str | None = None  # set when annotation is None, explains why


def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add a line `GEMINI_API_KEY=<your key>` to "
            "~/qualume/.env (get a key at https://aistudio.google.com/apikey), then re-run."
        )
    # Passed explicitly so a stray GOOGLE_API_KEY (which the SDK would prefer)
    # can't silently take over.
    return genai.Client(api_key=api_key)


def _wait_until_active(client: genai.Client, uploaded):
    deadline = time.monotonic() + FILE_POLL_TIMEOUT_SECONDS
    file = uploaded
    while file.state.name == "PROCESSING":
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Gemini file {file.name} still PROCESSING after {FILE_POLL_TIMEOUT_SECONDS}s"
            )
        time.sleep(FILE_POLL_INTERVAL_SECONDS)
        file = client.files.get(name=file.name)
    if file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file {file.name} ended in state {file.state.name}, not ACTIVE")
    return file


def annotate_video(local_video_path: str) -> AnnotationResult:
    """Upload one video to Gemini, have it watch it, return validated DNA
    alongside the complete raw response."""
    if not os.path.isfile(local_video_path):
        raise FileNotFoundError(f"Video file not found: {local_video_path}")
    client = _gemini_client()

    uploaded = client.files.upload(file=local_video_path)
    try:
        video = _wait_until_active(client, uploaded)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[video, ANNOTATION_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExternalVideoAnnotationV3,
            ),
        )
        # Validate the raw text ourselves: the SDK's response.parsed silently
        # becomes None on a validation failure instead of raising.
        if not response.text:
            raise RuntimeError(f"Gemini returned no text for {local_video_path}")
        raw_response = json.loads(response.text)
        if not isinstance(raw_response, dict):
            raise RuntimeError(
                f"Gemini response for {local_video_path} is not a JSON object: {response.text[:200]}"
            )
        try:
            annotation = ExternalVideoAnnotationV3.model_validate(raw_response)
            return AnnotationResult(annotation=annotation, raw_response=raw_response)
        except ValidationError as e:
            # Don't lose the raw response just because our strict schema (e.g.
            # chronological-order checks) rejected it — row 5 above. Capture
            # what Gemini actually said; annotate_and_store decides how to
            # store a row even when typed-field validation failed.
            return AnnotationResult(
                annotation=None, raw_response=raw_response, validation_error=str(e)
            )
    finally:
        # Uploaded files auto-expire after ~48h; delete now to not leave them around.
        client.files.delete(name=uploaded.name)


def annotate_and_store(external_content_item_id: str) -> str:
    """Annotate one external_content_items row's video; insert and return the
    new external_creative_dna row id.

    If Gemini's output fails our strict schema validation (e.g. an
    out-of-order timestamp), the row is still saved: raw_gemini_response
    and validation_error are populated, every typed field we can pull
    straight out of the raw dict (best-effort, no re-validation) is filled,
    and anything we genuinely can't trust stays NULL. Nothing is silently
    discarded just because one field didn't fit the strict model."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT local_video_path FROM external_content_items WHERE id = %s",
                (external_content_item_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"external_content_items row {external_content_item_id} not found")
        local_video_path = row[0]
        if local_video_path is None:
            raise ValueError(
                f"external_content_items row {external_content_item_id} has NULL local_video_path — "
                "download the video first (tiktok_reader.download_pending_videos)"
            )

        result = annotate_video(local_video_path)
        ann, raw_response, validation_error = (
            result.annotation, result.raw_response, result.validation_error
        )

        if ann is not None:
            # Clean path: fully validated against the strict schema.
            values = dict(
                hook_type=ann.hook_type, hook_text=ann.hook_text,
                visual_hook_description=ann.visual_hook_description, format_tag=ann.format_tag,
                script_structure_tags=ann.script_structure_tags, cta_type=ann.cta_type,
                caption_style=ann.caption_style, social_proof_used=ann.social_proof_used,
                energy_level=ann.energy_level, tone_category=ann.tone_category,
                voice_delivery_style=ann.voice_delivery_style,
                script_structure_timeline=Json([b.model_dump() for b in ann.script_structure_timeline]),
                on_screen_text=Json([t.model_dump() for t in ann.on_screen_text]),
                emotion_timeline=Json([e.model_dump() for e in ann.emotion_timeline]),
                pacing_qualitative=ann.pacing_qualitative, music_mood_v2=ann.music_mood,
                shot_breakdown=Json([s.model_dump() for s in ann.shot_breakdown]),
                full_transcript=Json([t.model_dump() for t in ann.full_transcript]),
            )
        else:
            # Validation failed — pull what we safely can straight from the raw
            # dict (no re-validation, best-effort scalars only; nested/typed
            # fields that caused the failure are left NULL rather than guessed).
            def _raw(key, default=None):
                return raw_response.get(key, default)

            values = dict(
                hook_type=_raw("hook_type"), hook_text=_raw("hook_text"),
                visual_hook_description=_raw("visual_hook_description"),
                format_tag=_raw("format_tag"),
                script_structure_tags=_raw("script_structure_tags") or [],
                cta_type=_raw("cta_type"), caption_style=_raw("caption_style"),
                social_proof_used=bool(_raw("social_proof_used", False)),
                energy_level=_raw("energy_level"), tone_category=_raw("tone_category"),
                voice_delivery_style=_raw("voice_delivery_style"),
                script_structure_timeline=None, on_screen_text=None, emotion_timeline=None,
                pacing_qualitative=_raw("pacing_qualitative"), music_mood_v2=_raw("music_mood"),
                shot_breakdown=None, full_transcript=None,
            )

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                       (external_content_item_id, hook_type, hook_text, visual_hook_description,
                        format_tag, script_structure_tags, cta_type, caption_style,
                        social_proof_used, energy_level, tone_category, voice_delivery_style,
                        script_structure_timeline, on_screen_text, emotion_timeline,
                        pacing_qualitative, music_mood_v2, shot_breakdown, full_transcript,
                        raw_gemini_response, validation_error,
                        analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    external_content_item_id, values["hook_type"], values["hook_text"],
                    values["visual_hook_description"], values["format_tag"],
                    values["script_structure_tags"], values["cta_type"], values["caption_style"],
                    values["social_proof_used"], values["energy_level"], values["tone_category"],
                    values["voice_delivery_style"], values["script_structure_timeline"],
                    values["on_screen_text"], values["emotion_timeline"],
                    values["pacing_qualitative"], values["music_mood_v2"],
                    values["shot_breakdown"], values["full_transcript"],
                    Json(raw_response), validation_error,
                    GEMINI_MODEL, ANALYSIS_PROMPT_VERSION,
                ),
            )
            dna_id = str(cur.fetchone()[0])
        conn.commit()
        return dna_id
    finally:
        conn.close()
