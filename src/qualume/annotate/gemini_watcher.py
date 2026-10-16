"""Gemini video-watching annotation — model-observed signals only.

Gemini is a deliberate, scoped exception to the Claude-only rule: it is used
ONLY here, for native audio+video understanding (see
plans/learn-engine-idea-sourcing-architecture.md, "How AI actually watches a
video"). Computed signals (pacing_cuts_per_min, music_mood) are a separate,
future job and are never asked of Gemini.
"""
import os
import time

from google import genai
from google.genai import types

from qualume import config  # noqa: F401 — loads .env into os.environ
from qualume.db.connection import get_connection
from qualume.models.contracts import ExternalVideoAnnotation

ANALYSIS_PROMPT_VERSION = "v1"

# Taken from the model list shipped in google-genai 2.31.0 (described there as
# the current most capable Flash model). NOT yet verified against the live API
# because no key was configured at build time — verify before the first real run:
#   uv run python -c "from google import genai; import os; print([m.name for m in genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.list()])"
GEMINI_MODEL = "gemini-3.8-flash"

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
"""


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


def annotate_video(local_video_path: str) -> ExternalVideoAnnotation:
    """Upload one video to Gemini, have it watch it, return validated DNA."""
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
                response_schema=ExternalVideoAnnotation,
            ),
        )
        # Validate the raw text ourselves: the SDK's response.parsed silently
        # becomes None on a validation failure instead of raising.
        if not response.text:
            raise RuntimeError(f"Gemini returned no text for {local_video_path}")
        return ExternalVideoAnnotation.model_validate_json(response.text)
    finally:
        # Uploaded files auto-expire after ~48h; delete now to not leave them around.
        client.files.delete(name=uploaded.name)


def annotate_and_store(external_content_item_id: str) -> str:
    """Annotate one external_content_items row's video; insert and return the
    new external_creative_dna row id."""
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

        ann = annotate_video(local_video_path)

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                       (external_content_item_id, hook_type, hook_text, visual_hook_description,
                        format_tag, script_structure_tags, cta_type, caption_style,
                        social_proof_used, energy_level, tone_category, voice_delivery_style,
                        analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    external_content_item_id, ann.hook_type, ann.hook_text,
                    ann.visual_hook_description, ann.format_tag, ann.script_structure_tags,
                    ann.cta_type, ann.caption_style, ann.social_proof_used, ann.energy_level,
                    ann.tone_category, ann.voice_delivery_style,
                    GEMINI_MODEL, ANALYSIS_PROMPT_VERSION,
                ),
            )
            dna_id = str(cur.fetchone()[0])
        conn.commit()
        return dna_id
    finally:
        conn.close()
