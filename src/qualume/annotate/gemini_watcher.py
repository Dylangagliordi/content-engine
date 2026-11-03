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
v4 stops storing Gemini's independent hook_text/visual_hook_description
(derived in Python from the transcript/shots instead — see
derive_hook_fields) and makes per-shot color/setting queryable. v5 adds
clearer prompt wording against merging distinct shots/sentences together
and records coverage gaps against the real video duration (shots/transcript
stopping short of the video's known duration). An earlier v5 draft also
hard-rejected any shot/segment over a fixed duration, but real rejected rows
turned out to be legitimate long single takes and sentences, not merged
ones — duration alone can't tell those apart, so that check was removed
rather than kept as a false-positive source; see ExternalVideoAnnotationV5's
docstring in contracts.py. Temperature is left at the API default —
Google's own Gemini 3 guidance (confirmed live,
ai.google.dev/gemini-api/docs/gemini-3) warns lower values risk looping/
degraded output on Gemini 3 models, which GEMINI_MODEL is.
"""
import json
import os
import time
from typing import NamedTuple, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from qualume import config  # noqa: F401 — loads .env into os.environ
from qualume.db.connection import get_connection
from psycopg2.extras import Json

from qualume.models.contracts import (
    ExternalVideoAnnotationV5,
    ExternalVideoAnnotationV6,
    PresenterDescription,
    ScriptBeat,
    ShotBreakdownV4,
    TranscriptSegment,
)

ANALYSIS_PROMPT_VERSION = "v6"

# gemini-3.8-flash is the newest/most capable and real on this key, but is
# returning persistent 503 ServerError specifically on video input as of
# 2026-10-15 (plain text succeeds on it; video fails identically across
# 3.8-flash, flash-latest, AND 3.7-flash — isolated via direct testing, not
# assumed). gemini-3.1-flash-lite verified live: uploads, polls ACTIVE, and
# returns an accurate video description. Using it until 3.8-flash's video path
# stabilizes — re-check periodically, same re-verify command pattern as before:
#   uv run python -c "from google import genai; import os; from dotenv import load_dotenv; load_dotenv(); print(genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.get(model='models/gemini-3.8-flash'))"
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Real fix, 2026-11-03 ("ensure annotations get sped up"): GEMINI_MODEL's
# free-tier quota is 500 requests/day (confirmed live via Google's own
# rate-limits docs, ai.google.dev/gemini-api/docs/rate-limits — "Limits
# vary depending on the specific model being used"), and the real
# annotation backlog at the time this was added was ~10,400 videos —
# 500/day alone would take 20+ days to clear. gemini-3.5-flash-lite is a
# genuinely SEPARATE model with its OWN separate 500 RPD free-tier quota
# (same docs: limits are per-model, not pooled across a project) —
# verified live that it successfully watches video (plain-text prompt
# succeeds), roughly doubling real daily annotation throughput for free.
#
# BUT: verified live that gemini-3.5-flash-lite returns a persistent 503
# ServerError specifically when a strict response_schema is attached
# alongside video input (same general 503-on-video-input class of issue
# as gemini-3.8-flash/flash-latest/3.7-flash above, just a narrower
# trigger — schema+video, not video alone). Dropping the strict schema
# (keep response_mime_type='application/json', drop response_schema)
# avoids the 503 and produces real, correctly-shaped JSON — annotate_video
# falls back to this model+config specifically ON a 503 from the primary
# model, and routes the result through this project's EXISTING degraded-
# path validation (best-effort field extraction, raw_response/
# validation_error stored) rather than a new code path, since that path
# already handles "valid JSON, doesn't fully match the strict schema."
# Re-verify periodically (same pattern as GEMINI_MODEL above) in case a
# future SDK/model update fixes the schema+video 503 and a strict call
# becomes possible here too.
GEMINI_MODEL_FALLBACK = "gemini-3.5-flash-lite"


# A single generate_content call has no cross-call state; the only lever for
# run-to-run consistency on the same video is sampling variance. Gemini's API
# accepts 0.0 (range 0.0-2.0, and the SDK field is a plain Optional[float], so
# 0.0 isn't dropped as falsy), so near-zero is not about an SDK edge case — 0.1
# CONFIRMED (checked live against Google's own docs, ai.google.dev/
# gemini-api/docs/gemini-3, 2026-10-16): for ALL Gemini 3 models, Google
# "strongly recommends keeping temperature at its default value of 1.0" —
# lower values are documented to risk looping/degraded output, specifically
# on Gemini 3 models. GEMINI_MODEL (gemini-3.1-flash-lite) IS a Gemini 3
# model, so a low temperature here would likely make output WORSE, not more
# consistent — the opposite of FIX 2's original intent. Do not set a
# temperature override; let the SDK default (1.0) apply. If GEMINI_MODEL
# ever changes to a non-Gemini-3 model, this guidance may no longer apply —
# re-check before reintroducing a temperature override.
GEMINI_TEMPERATURE = None  # None = use the SDK/API default, not a override

# Unaccounted-for stretch of the real video (between/around shots) that gets
# flagged in coverage_warning. Small gaps are normal Gemini timestamp slop.
MAX_COVERAGE_GAP_SECONDS = 3.0

# Separate, more generous tolerance for trailing silence after the last
# transcript segment: a video can legitimately go quiet for several seconds
# before it ends (lingering shot, on-screen-text-only CTA, outro music) —
# that's not "Gemini stopped transcribing early", so it needs a looser
# threshold than the between-shots gap check above.
MAX_TRANSCRIPT_TRAILING_SILENCE_SECONDS = 8.0

FILE_POLL_INTERVAL_SECONDS = 2
FILE_POLL_TIMEOUT_SECONDS = 300

# v5's segmentation wording asks for one entry per real cut/sentence and NOT
# to merge distinct ones together, but explicitly allows genuinely long single
# takes/sentences — v5 originally also had hard duration ceilings enforcing
# this as a contract validator, removed after real rejected rows turned out
# to be legitimate long single shots/sentences, not merged ones. See
# ExternalVideoAnnotationV5's docstring in contracts.py for the full story.
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
with start_sec and end_sec, covering the video from start to end. Name the \
opening scroll-stopping beat exactly "hook".
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
- shot_breakdown: one entry PER CAMERA CUT, in chronological order, covering \
the whole video start to end (no gaps). Every time the picture cuts to a new \
shot, angle, framing or setting, start a new entry — even if the setting is \
the same and the cut is only a second long. Do NOT merge several cuts into \
one entry. A single unbroken take (e.g. one person talking to camera with \
no cuts) is real and should be reported as ONE long entry — don't \
artificially split a shot that genuinely has no cut in it. For each shot give: \
start_sec, end_sec, shot_type (e.g. close_up, medium_shot, wide_shot, \
extreme_close_up, pov), camera_movement (e.g. static, handheld, pan, \
zoom_in, zoom_out, tracking), setting (where it takes place, e.g. kitchen, \
bedroom, outdoor park), setting_category (exactly one of: home_indoor, \
home_outdoor, studio, public_indoor, public_outdoor, vehicle, other), \
color_temperature (exactly one of: warm, cool, neutral, mixed), \
dominant_colors (one or more basic color names, most dominant first, ONLY \
from: red, orange, yellow, green, blue, purple, pink, brown, black, white, \
gray, beige — e.g. ["pink", "white"]), lighting (e.g. bright_natural, dim, \
backlit, studio, golden_hour), and subject_action (what the person/subject \
is doing, one short phrase). A beat from \
script_structure_timeline can span multiple shots if the camera setup \
changes within it.
- full_transcript: the spoken words, in chronological order, transcribed \
verbatim, as ONE ENTRY PER SENTENCE (or per short phrase if the speaker \
doesn't use full sentences), each with the start_sec that sentence begins. \
Do NOT merge multiple sentences into one entry, and never return the whole \
transcript as a single entry. A single genuinely long sentence is fine as \
one entry — the point is not merging separate sentences together, not \
hitting a time limit. A typical 30-second talking video has roughly 6-12 \
entries. This is the COMPLETE transcript, not just the hook — include everything said, \
through to the end of the video. Empty list if there is no speech.

Presenter/avatar and topic fields.
- presenters: every DISTINCT real person visibly on camera at any point, in \
order of first appearance. Empty list if nobody is ever visibly shown (e.g. \
pure product shots, voiceover-over-broll with no face on screen). For each \
presenter, describe ONLY what is visibly observable:
  - apparent_age_range: the apparent age bucket from how they visibly present \
(child, teen, 20s, 30s, 40s, 50s, 60_plus, or unclear) — an observation about \
presentation, not a claim about their real age.
  - apparent_gender_presentation: male, female, or androgynous_or_unclear — \
an observation about presentation, not a claim about real gender identity.
  - relatable_archetype: the relatable role/persona they read as to a viewer \
(snake_case, e.g. mom, nurse, young_professional, grandma, fitness_coach, \
regular_person).
  - wardrobe_style: what they are visibly wearing/styled as (e.g. "casual \
loungewear", "medical scrubs", "business casual", "athleisure").
  - delivery_affect: their visible delivery style as a GERUND PHRASE that \
reads naturally in "a mom in her kitchen, ___ about X" — e.g. "passionately \
talking", "calmly confiding", "excitedly sharing", "tearfully explaining", \
"casually chatting".
- delivery_mode: how the content is delivered overall — direct_to_camera, \
voiceover_only, pov, interview_style, or no_presenter if nobody is ever on \
screen.
- video_topic_summary: one plain sentence summarizing what this video is \
actually ABOUT — the real subject matter being discussed/shown (e.g. "night \
sweats and hot flashes during menopause", "a morning skincare routine", \
"picky eating in toddlers").
"""


# Used when no beat is literally named "hook" (Gemini names beats freely,
# e.g. "cold_open"/"problem"). Matches the "first ~3 seconds" the prompt has
# always used to define the hook, so this degrades gracefully rather than
# failing a whole annotation over a beat label.
HOOK_FALLBACK_WINDOW_SEC = (0.0, 3.0)


def _hook_window(timeline: list[ScriptBeat]) -> tuple[float, float]:
    for beat in timeline:
        if beat.beat.strip().lower() == "hook":
            return beat.start_sec, beat.end_sec
    return HOOK_FALLBACK_WINDOW_SEC


def _article_phrase(presenter) -> str:
    """'a mom' / 'an elderly nurse' etc - article + archetype, snake_case
    archetype turned into natural words."""
    archetype = presenter.relatable_archetype.replace("_", " ").strip()
    article = "an" if archetype[:1].lower() in "aeiou" else "a"
    return f"{article} {archetype}"


def derive_scene_description(
    presenters: list,
    setting: str | None,
    video_topic_summary: str | None,
) -> str:
    """Build the one-sentence scene description the user specifically
    asked for: "this vid was a mom in her kitchen passionately talking
    about night sweats". Deterministic composition from structured
    fields — single source of truth, same discipline as
    derive_hook_fields (never ask Gemini for this sentence directly,
    always compose it from the typed presenter/setting/topic fields so
    it can't silently drift from them).

    presenters: list[PresenterDescription], already-parsed (from a valid
    ExternalVideoAnnotationV6) or already-dict-shaped (degraded path) —
    accepts either, same best-effort-on-the-degraded-path discipline as
    the rest of gemini_watcher.
    - Empty list -> "Product/visual-only footage in <setting>, about
      <topic>." (no person ever visibly shown — e.g. pure product shots).
    - One presenter -> "A <archetype> in <setting>, <delivery_affect>
      about <topic>."
    - Multiple presenters -> each archetype joined with "and", one
      shared delivery_affect/setting/topic (real multi-presenter nuance,
      e.g. two different delivery affects for two people, is a possible
      future refinement — not built here, kept simple and honest about
      what's actually composed).

    setting=None (shot_breakdown unavailable, e.g. degraded path) or
    video_topic_summary=None (annotation validation failed) degrade
    gracefully — omit that clause rather than crash or fabricate a
    placeholder value."""
    def _as_presenter(p):
        from qualume.models.contracts import PresenterDescription
        return p if isinstance(p, PresenterDescription) else PresenterDescription(**p)

    people = [_as_presenter(p) for p in presenters]

    if not people:
        subject = "Product/visual-only footage"
        affect_clause = ""
    else:
        archetypes = " and ".join(_article_phrase(p) for p in people)
        subject = archetypes[0].upper() + archetypes[1:]
        affect_clause = people[0].delivery_affect.strip()

    sentence = subject
    if setting:
        setting_clean = setting.strip()
        article = "an" if setting_clean[:1].lower() in "aeiou" else "a"
        sentence += f" in {article} {setting_clean}"

    if affect_clause:
        sentence += f", {affect_clause}"
        if video_topic_summary:
            sentence += f" about {video_topic_summary.strip()}"
    elif video_topic_summary:
        sentence += f", about {video_topic_summary.strip()}"

    sentence = sentence.strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _shot_phrase(shot: ShotBreakdownV4) -> str:
    kind = shot.shot_type.replace("_", " ").strip().removesuffix(" shot")
    article = "An" if kind[:1].lower() in "aeiou" else "A"
    return f"{article} {kind} shot"


def derive_hook_fields(
    timeline: list[ScriptBeat],
    transcript: list[TranscriptSegment],
    shots: list[ShotBreakdownV4],
) -> tuple[str, str]:
    """Return (hook_text, visual_hook_description) derived deterministically
    from the hook window — the single source of truth for the stored values.

    Gemini answers hook_text/visual_hook_description itself too, but those
    independent answers drifted from its own full_transcript on real output
    (dropped "I found the perfect one"), so they're never stored.

    Window: the beat named "hook" in script_structure_timeline, else the
    first 3 seconds. hook_text: every transcript segment whose start_sec is in
    [start, end), space-joined in order ("" if none). visual: the first shot
    overlapping the window, plus a note of the cut if more shots overlap
    ("" if none)."""
    start, end = _hook_window(timeline)

    segments = sorted(transcript, key=lambda t: t.start_sec)
    hook_text = " ".join(
        t.text.strip() for t in segments if start <= t.start_sec < end and t.text.strip()
    )

    overlapping = [
        s for s in sorted(shots, key=lambda s: s.start_sec)
        if s.start_sec < end and s.end_sec > start
    ]
    if not overlapping:
        return hook_text, ""
    first = overlapping[0]
    visual = f"{_shot_phrase(first)} in {first.setting.strip()}: {first.subject_action.strip().rstrip('.')}."
    if len(overlapping) > 1:
        nxt = overlapping[1]
        lead = (
            "Then cuts to" if len(overlapping) == 2
            else f"Then cuts through {len(overlapping) - 1} more shots, starting with"
        )
        phrase = _shot_phrase(nxt)
        visual += f" {lead} {phrase[0].lower()}{phrase[1:]}: {nxt.subject_action.strip().rstrip('.')}."
    return hook_text, visual


def _check_coverage(
    annotation: ExternalVideoAnnotationV6, video_duration_seconds: float | None
) -> list[str]:
    """Return human-readable warnings for a schema-valid annotation that
    doesn't actually cover the real video (empty list = looks complete).

    Deliberately warnings, not a ValidationError: the contract's validators
    only see Gemini's own timestamps, so they can't tell that shots ending
    at 10s are wrong for a 32s video — but the shots/transcript that ARE
    there are still valid data worth keeping. Stored in coverage_warning
    so incomplete rows stay queryable after the fact."""
    if video_duration_seconds is None:
        return ["video duration unknown (external_content_items.duration_seconds is NULL); coverage not checked"]
    duration = float(video_duration_seconds)
    warnings = []

    shots = sorted(annotation.shot_breakdown, key=lambda s: s.start_sec)
    if not shots:
        warnings.append(f"shot_breakdown is empty for a {duration:g}s video")
    covered_until = 0.0
    for shot in shots:
        if shot.start_sec - covered_until > MAX_COVERAGE_GAP_SECONDS:
            warnings.append(
                f"shot_breakdown has a {shot.start_sec - covered_until:.1f}s gap "
                f"({covered_until}-{shot.start_sec}s) with no shot"
            )
        covered_until = max(covered_until, shot.end_sec)
    if shots and duration - covered_until > MAX_COVERAGE_GAP_SECONDS:
        warnings.append(f"shot_breakdown stops at {covered_until}s of a {duration:g}s video")
    if covered_until - duration > MAX_COVERAGE_GAP_SECONDS:
        warnings.append(f"shot_breakdown runs to {covered_until}s, past the real {duration:g}s video end")

    # Empty transcript is legitimate (silent video), so only a non-empty one
    # that stops early is flagged — uses its own, more generous tolerance
    # than the shot-gap check above, since trailing silence before a video
    # ends is normal (lingering shot, on-screen CTA, outro music).
    if annotation.full_transcript:
        last = max(t.start_sec for t in annotation.full_transcript)
        if duration - last > MAX_TRANSCRIPT_TRAILING_SILENCE_SECONDS:
            warnings.append(
                f"full_transcript's last segment starts at {last}s of a {duration:g}s video "
                f"({duration - last:.1f}s with no new segment)"
            )
    return warnings


_M = TypeVar("_M", bound=BaseModel)


def _parse_raw_list(model: type[_M], items) -> list[_M] | None:
    """Best-effort parse of one raw list for the degraded path; None if it's
    missing or any entry doesn't fit (we don't guess at partial data)."""
    if not isinstance(items, list):
        return None
    try:
        return [model.model_validate(i) for i in items]
    except ValidationError:
        return None


class AnnotationResult(NamedTuple):
    annotation: ExternalVideoAnnotationV6 | None  # None if schema validation failed
    raw_response: dict  # complete parsed Gemini response, including unmapped keys —
                         # ALWAYS populated when Gemini returned anything parseable,
                         # even if `annotation` is None. Nothing gets silently dropped.
    validation_error: str | None = None  # set when annotation is None, explains why
    coverage_warning: str | None = None  # "; "-joined _check_coverage warnings for a
                                          # VALID annotation; None if complete or invalid
    model_used: str = GEMINI_MODEL  # real fix, 2026-11-03: WHICH model actually produced
                                     # this result — GEMINI_MODEL on the normal path,
                                     # GEMINI_MODEL_FALLBACK when the primary 503'd and the
                                     # fallback served instead. annotate_and_store persists
                                     # this into analysis_model; defaulting to GEMINI_MODEL
                                     # keeps every pre-fallback caller's behavior unchanged.


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


def annotate_video(local_video_path: str, video_duration_seconds: float | None) -> AnnotationResult:
    """Upload one video to Gemini, have it watch it, return validated DNA
    alongside the complete raw response. video_duration_seconds is the real
    duration (yt-dlp metadata), passed in rather than looked up so this
    stays DB-free; it's only used for the coverage check.

    Real fallback (2026-11-03): on a 503 from GEMINI_MODEL (the real,
    confirmed-live daily-quota/high-demand failure mode, not a generic
    catch-all), retries once against GEMINI_MODEL_FALLBACK WITHOUT a
    strict response_schema (confirmed live: the schema is what triggers
    an identical 503 on this second model specifically for video input).
    A non-503 error (e.g. a genuine 400 bad request) is NOT retried —
    propagates as before, so this fix doesn't mask real bugs behind a
    second model's different failure mode."""
    if not os.path.isfile(local_video_path):
        raise FileNotFoundError(f"Video file not found: {local_video_path}")
    client = _gemini_client()

    uploaded = client.files.upload(file=local_video_path)
    try:
        video = _wait_until_active(client, uploaded)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[video, ANNOTATION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExternalVideoAnnotationV6,
                    temperature=GEMINI_TEMPERATURE,
                ),
            )
            model_used = GEMINI_MODEL
        except genai_errors.ServerError as e:
            if getattr(e, "code", None) != 503:
                raise
            # Real fallback: same video, second model, no strict schema
            # (confirmed live this specific combination avoids the 503 on
            # GEMINI_MODEL_FALLBACK). Re-raises anything else untouched.
            response = client.models.generate_content(
                model=GEMINI_MODEL_FALLBACK,
                contents=[video, ANNOTATION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=GEMINI_TEMPERATURE,
                ),
            )
            model_used = GEMINI_MODEL_FALLBACK
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
            annotation = ExternalVideoAnnotationV6.model_validate(raw_response)
        except ValidationError as e:
            # Don't lose the raw response just because our strict schema (e.g.
            # chronological-order checks) rejected it — row 5 above. Capture
            # what Gemini actually said; annotate_and_store decides how to
            # store a row even when typed-field validation failed. This same
            # path also catches the fallback model's occasional out-of-enum
            # values (e.g. tone_category='excited') — no new handling needed.
            return AnnotationResult(
                annotation=None, raw_response=raw_response, validation_error=str(e),
                model_used=model_used,
            )
        warnings = _check_coverage(annotation, video_duration_seconds)
        return AnnotationResult(
            annotation=annotation, raw_response=raw_response,
            coverage_warning="; ".join(warnings) or None,
            model_used=model_used,
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
    discarded just because one field didn't fit the strict model.

    hook_text / visual_hook_description are ALWAYS the derive_hook_fields
    values, never Gemini's independent answers (those stay only in
    raw_gemini_response).

    coverage_warning is set when a VALID annotation doesn't span the real
    video duration (see _check_coverage); NULL on the degraded path, where
    validation_error already says what's wrong."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT local_video_path, duration_seconds FROM external_content_items WHERE id = %s",
                (external_content_item_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"external_content_items row {external_content_item_id} not found")
        local_video_path, duration_seconds = row
        if local_video_path is None:
            raise ValueError(
                f"external_content_items row {external_content_item_id} has NULL local_video_path — "
                "download the video first (tiktok_reader.download_pending_videos)"
            )

        result = annotate_video(local_video_path, duration_seconds)
        ann, raw_response, validation_error, coverage_warning, model_used = (
            result.annotation, result.raw_response, result.validation_error,
            result.coverage_warning, result.model_used,
        )

        if ann is not None:
            # Clean path: fully validated against the strict schema.
            # Override Gemini's own hook answers with the derived ones.
            hook_text, visual_hook_description = derive_hook_fields(
                ann.script_structure_timeline, ann.full_transcript, ann.shot_breakdown
            )
            values = dict(
                hook_type=ann.hook_type, hook_text=hook_text,
                visual_hook_description=visual_hook_description, format_tag=ann.format_tag,
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
                presenters=Json([p.model_dump() for p in ann.presenters]),
                delivery_mode=ann.delivery_mode,
                video_topic_summary=ann.video_topic_summary,
                scene_description=derive_scene_description(
                    ann.presenters,
                    ann.shot_breakdown[0].setting if ann.shot_breakdown else None,
                    ann.video_topic_summary,
                ),
            )
        else:
            # Validation failed — pull what we safely can straight from the raw
            # dict (no re-validation, best-effort scalars only; nested/typed
            # fields that caused the failure are left NULL rather than guessed).
            def _raw(key, default=None):
                return raw_response.get(key, default)

            # Derive the hook fields best-effort from the raw lists. A bad
            # timeline just means the 3s fallback window; an unusable
            # transcript/shots means NULL — never Gemini's independent
            # hook_text, which would reintroduce the second source of truth.
            timeline = _parse_raw_list(ScriptBeat, _raw("script_structure_timeline")) or []
            transcript = _parse_raw_list(TranscriptSegment, _raw("full_transcript"))
            shots = _parse_raw_list(ShotBreakdownV4, _raw("shot_breakdown"))
            hook_text = visual_hook_description = None
            if transcript is not None:
                hook_text = derive_hook_fields(timeline, transcript, [])[0]
            if shots is not None:
                visual_hook_description = derive_hook_fields(timeline, [], shots)[1]

            # Best-effort presenters/scene_description on the degraded path —
            # same discipline as hook fields above: only compute from data
            # that actually parsed, leave NULL rather than guess.
            presenters_parsed = _parse_raw_list(PresenterDescription, _raw("presenters"))
            raw_topic = _raw("video_topic_summary")
            scene_description = None
            if presenters_parsed is not None and raw_topic:
                first_shot_setting = shots[0].setting if shots else None
                scene_description = derive_scene_description(
                    presenters_parsed, first_shot_setting, raw_topic
                )

            values = dict(
                hook_type=_raw("hook_type"), hook_text=hook_text,
                visual_hook_description=visual_hook_description,
                format_tag=_raw("format_tag"),
                script_structure_tags=_raw("script_structure_tags") or [],
                cta_type=_raw("cta_type"), caption_style=_raw("caption_style"),
                social_proof_used=bool(_raw("social_proof_used", False)),
                energy_level=_raw("energy_level"), tone_category=_raw("tone_category"),
                voice_delivery_style=_raw("voice_delivery_style"),
                script_structure_timeline=None, on_screen_text=None, emotion_timeline=None,
                pacing_qualitative=_raw("pacing_qualitative"), music_mood_v2=_raw("music_mood"),
                shot_breakdown=None, full_transcript=None,
                presenters=Json(_raw("presenters")) if presenters_parsed is not None else None,
                delivery_mode=_raw("delivery_mode"),
                video_topic_summary=raw_topic,
                scene_description=scene_description,
            )

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                       (external_content_item_id, hook_type, hook_text, visual_hook_description,
                        format_tag, script_structure_tags, cta_type, caption_style,
                        social_proof_used, energy_level, tone_category, voice_delivery_style,
                        script_structure_timeline, on_screen_text, emotion_timeline,
                        pacing_qualitative, music_mood_v2, shot_breakdown, full_transcript,
                        presenters, delivery_mode, video_topic_summary, scene_description,
                        raw_gemini_response, validation_error, coverage_warning,
                        analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    values["presenters"], values["delivery_mode"],
                    values["video_topic_summary"], values["scene_description"],
                    Json(raw_response), validation_error, coverage_warning,
                    model_used, ANALYSIS_PROMPT_VERSION,
                ),
            )
            dna_id = str(cur.fetchone()[0])
        conn.commit()
        return dna_id
    finally:
        conn.close()
