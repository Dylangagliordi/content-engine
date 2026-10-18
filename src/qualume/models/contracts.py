from datetime import datetime
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


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


ToneCategory = Literal[
    "comedic", "serious", "sincere", "sarcastic", "dramatic",
    "chill", "chaotic", "wholesome", "urgent", "informative",
]


class ExternalVideoAnnotation(BaseModel):
    """Model-observed signals for one external video — the structured output
    Gemini is constrained to (response_schema). Becomes part of an
    external_creative_dna row. v1 (analysis_prompt_version='v1'); kept for
    existing rows/callers — new annotations use ExternalVideoAnnotationV2."""
    hook_type: str = Field(description="Hook mechanism in the first ~3s, snake_case, e.g. question, bold_claim, problem_statement, visual_reveal, pattern_interrupt")
    hook_text: str = Field(description="Exact words spoken or shown on screen in the hook; empty string if none")
    visual_hook_description: str = Field(description="What is visually on screen during the first ~3s")
    format_tag: str = Field(description="Overall video format, snake_case, e.g. talking_head, voiceover_broll, skit, tutorial, slideshow, product_demo")
    script_structure_tags: list[str] = Field(description="Ordered beats of the video, snake_case, e.g. hook, problem, demo, payoff, cta")
    cta_type: str = Field(description="Call to action, snake_case, e.g. follow, comment, link_in_bio, shop, none")
    caption_style: str = Field(description="Style of on-screen text/captions, e.g. auto_captions, bold_center, minimal, none")
    social_proof_used: bool = Field(description="True only if the video shows reviews, testimonials, numbers of customers, press, or expert endorsement")
    energy_level: int = Field(ge=1, le=5, description="1=calm/meditative, 2=relaxed, 3=moderate/conversational, 4=upbeat/lively, 5=maximum intensity/frenetic")
    tone_category: ToneCategory = Field(description="Single dominant tone")
    voice_delivery_style: str = Field(description="How the voice is delivered, e.g. conversational direct-to-camera, whispered ASMR, fast-talking hype, text-to-speech, no voice")


class ScriptBeat(BaseModel):
    beat: str = Field(description="Beat name, snake_case, e.g. hook, problem, demo, payoff, cta")
    start_sec: float = Field(ge=0, description="Beat start, seconds from video start")
    end_sec: float = Field(ge=0, description="Beat end, seconds from video start")

    @model_validator(mode="after")
    def _end_not_before_start(self):
        if self.end_sec < self.start_sec:
            raise ValueError(f"beat {self.beat!r} ends ({self.end_sec}) before it starts ({self.start_sec})")
        return self


class OnScreenText(BaseModel):
    text: str = Field(description="Literal on-screen text/caption/overlay, verbatim")
    timestamp_sec: float = Field(ge=0, description="When the text first appears, seconds from video start")


class EmotionPoint(BaseModel):
    timestamp_sec: float = Field(ge=0, description="When this emotion begins, seconds from video start")
    emotion: str = Field(description="Dominant emotion from this point on, one lowercase word, e.g. curious, frustrated, relieved, excited, calm")


PacingQualitative = Literal["slow", "medium", "fast"]


def _require_non_decreasing(values: list[float], field: str) -> None:
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError(f"{field} must be in chronological order, got timestamps {values}")


class ExternalVideoAnnotationV2(ExternalVideoAnnotation):
    """v2 adds timestamped structure and audio-derived signals, all asked of
    Gemini in the same call (no separate PySceneDetect/beatlyze job). Pacing
    is qualitative only: Gemini samples video at 1 FPS, so a precise
    cuts-per-minute count is structurally unreliable and is never asked for."""
    script_structure_timeline: list[ScriptBeat] = Field(description="Ordered beats of the video with start/end seconds")
    on_screen_text: list[OnScreenText] = Field(description="Every distinct piece of on-screen text, in order of appearance; empty if none")
    emotion_timeline: list[EmotionPoint] = Field(description="Emotional shifts across the video, in chronological order, starting at 0")
    pacing_qualitative: PacingQualitative = Field(description="Overall editing/visual pace: slow, medium, or fast")
    pacing_rationale: str = Field(description="One sentence justifying the pacing read from what is observable")
    music_mood: list[str] = Field(description="Mood tags for the music, from the audio, lowercase, e.g. upbeat, melancholic, suspenseful, lofi; empty if no music")

    @model_validator(mode="after")
    def _timelines_in_order(self):
        _require_non_decreasing([b.start_sec for b in self.script_structure_timeline], "script_structure_timeline")
        _require_non_decreasing([t.timestamp_sec for t in self.on_screen_text], "on_screen_text")
        _require_non_decreasing([e.timestamp_sec for e in self.emotion_timeline], "emotion_timeline")
        return self


class ShotBreakdown(BaseModel):
    """One camera shot/setup — the actual reproducible unit for visual
    recreation (finer-grained than script_structure_timeline's story beats:
    one beat like 'demo' can span several shots). v3's shape — kept
    unchanged so the real stored v3 row still validates against this exact
    class (versioning principle: every row traceable to exactly what
    produced it). v4 uses ShotBreakdownV4 below instead of mutating this."""
    start_sec: float = Field(ge=0, description="Shot start, seconds from video start")
    end_sec: float = Field(ge=0, description="Shot end, seconds from video start")
    shot_type: str = Field(description="e.g. close_up, medium_shot, wide_shot, extreme_close_up, pov")
    camera_movement: str = Field(description="e.g. static, handheld, pan, zoom_in, zoom_out, tracking")
    setting: str = Field(description="Where this shot takes place, e.g. kitchen, bedroom, outdoor park")
    color_palette: str = Field(description="Dominant colors observed, named not hex, e.g. 'warm pinks and whites', 'muted earth tones'")
    lighting: str = Field(description="e.g. bright_natural, dim, backlit, studio, golden_hour")
    subject_action: str = Field(description="What the person/subject is doing in this shot, one short phrase")

    @model_validator(mode="after")
    def _end_not_before_start(self):
        if self.end_sec < self.start_sec:
            raise ValueError(f"shot ends ({self.end_sec}) before it starts ({self.start_sec})")
        return self


ColorTemperature = Literal["warm", "cool", "neutral", "mixed"]

# Deliberately small and fixed so shots can be GROUP BY'd / filtered in SQL
# (v3's free-prose color_palette — 'bright warm tones', 'deep reds and bright
# whites' — couldn't be). Unlike shot_type/lighting etc, basic color names
# aren't a "learned feature" space, so a closed vocabulary loses nothing.
BasicColor = Literal[
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown", "black", "white", "gray", "beige",
]

SettingCategory = Literal[
    "home_indoor", "home_outdoor", "studio", "public_indoor", "public_outdoor", "vehicle", "other",
]


class ShotBreakdownV4(BaseModel):
    """v4's shot shape — a NEW class, not a mutation of ShotBreakdown, so v3's
    real stored row stays validatable against the class it was actually
    produced by. color_palette (free prose) is REMOVED, replaced by
    color_temperature + dominant_colors (fixed vocabularies). setting_category
    is a coarse queryable grouping added ALONGSIDE free-text setting, which
    keeps the human-readable detail — not redundant with it, unlike
    color_palette would have been with the new color fields."""
    start_sec: float = Field(ge=0, description="Shot start, seconds from video start")
    end_sec: float = Field(ge=0, description="Shot end, seconds from video start")
    shot_type: str = Field(description="e.g. close_up, medium_shot, wide_shot, extreme_close_up, pov")
    camera_movement: str = Field(description="e.g. static, handheld, pan, zoom_in, zoom_out, tracking")
    setting: str = Field(description="Where this shot takes place, e.g. kitchen, bedroom, outdoor park")
    setting_category: SettingCategory = Field(description="Coarse setting group for querying")
    color_temperature: ColorTemperature = Field(description="Overall color temperature of the shot")
    dominant_colors: list[BasicColor] = Field(min_length=1, description="Dominant colors in the shot, basic color names only, most dominant first")
    lighting: str = Field(description="e.g. bright_natural, dim, backlit, studio, golden_hour")
    subject_action: str = Field(description="What the person/subject is doing in this shot, one short phrase")

    @model_validator(mode="after")
    def _end_not_before_start(self):
        if self.end_sec < self.start_sec:
            raise ValueError(f"shot ends ({self.end_sec}) before it starts ({self.start_sec})")
        return self


class TranscriptSegment(BaseModel):
    """One spoken utterance — the FULL transcript, not just the hook.
    hook_text (v1) only ever captured the opening few seconds; this covers
    the entire video."""
    start_sec: float = Field(ge=0, description="Segment start, seconds from video start")
    text: str = Field(description="Exact words spoken, verbatim")


class ExternalVideoAnnotationV3(ExternalVideoAnnotationV2):
    """v3 adds per-shot visual composition (color, lighting, camera, setting)
    and the complete spoken transcript — the goal being enough structured
    detail to brief a stylistically/structurally equivalent new video, not
    just a similar-shaped script. The exact source file (audio_track/
    audio_artist already captured losslessly via external_content_items from
    yt-dlp metadata) + raw_gemini_response cover anything this typed schema
    doesn't anticipate."""
    shot_breakdown: list[ShotBreakdown] = Field(description="Every distinct camera shot, in order, covering the whole video")
    full_transcript: list[TranscriptSegment] = Field(description="Every spoken utterance in the video, in order, verbatim; empty if no speech")

    @model_validator(mode="after")
    def _v3_timelines_in_order(self):
        _require_non_decreasing([s.start_sec for s in self.shot_breakdown], "shot_breakdown")
        _require_non_decreasing([t.start_sec for t in self.full_transcript], "full_transcript")
        return self


class ExternalVideoAnnotationV4(ExternalVideoAnnotationV3):
    """v4 fixes two problems found reviewing real v3 output:

    1. Redundancy. hook_text / visual_hook_description were answered by Gemini
       independently, in the same call as full_transcript / shot_breakdown, so
       the two answers drifted (real video: full_transcript's first segment
       had "I found the perfect one", hook_text omitted it). Gemini still
       fills them here (they're inherited and still in the response schema),
       but they are NOT stored: gemini_watcher.derive_hook_fields recomputes
       both from script_structure_timeline + full_transcript + shot_breakdown
       before persisting, so there's one source of truth. Gemini's own answers
       survive only in raw_gemini_response.
    2. Queryability. ShotBreakdown.color_palette (free prose) became
       color_temperature + dominant_colors (fixed vocabularies), plus a coarse
       setting_category — a NEW shot class, ShotBreakdownV4, overriding the
       field below. v3's own ShotBreakdown class is untouched, so the real
       stored v3 row still validates against ExternalVideoAnnotationV3. No
       new top-level columns; the changes live inside shot_breakdown's JSONB.

    Learned-label fields (hook_type, format_tag, shot_type, lighting, ...)
    deliberately stay free strings — see creative_dna_feature_registry."""
    shot_breakdown: list[ShotBreakdownV4] = Field(description="Every distinct camera shot, in order, covering the whole video")


class ExternalVideoAnnotationV5(ExternalVideoAnnotationV4):
    """v5 originally added hard duration ceilings on shot_breakdown/
    full_transcript (reject anything over ~8-10s as 'under-segmented').
    Removed after reviewing real rejected rows: a 60s single static
    talking-head shot and a 9-word, 9-second sentence are both completely
    ordinary content, not merged segments — duration alone can't tell
    real long takes/sentences apart from actual under-segmentation, so the
    check was rejecting good data, which violates this project's
    never-silently-discard-real-data rule. The real v4 bug (one segment
    spanning the ENTIRE 32s script, several complete sentences glued
    together) had a different, actually-distinguishing signature —
    multiple sentence-ending punctuation marks inside one segment — which
    could be a future, better-targeted check if this recurs, but isn't
    built here. For now: same fields/schema as v4, no extra validation.
    coverage_warning (gemini_watcher._check_coverage) still separately
    flags shots/transcript that stop short of the real video duration —
    that check is about completeness, not segment length, and stays."""
    pass


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


class CopyAnalysis(BaseModel):
    """Structured output of the copy-technique classifier
    (annotate/copy_classifier.py) — a text-only Claude pass over a video's
    already-stored full_transcript/on_screen_text/hook_text, NOT a new
    video-watching call. 19 boolean flags, multi-label by design (a real
    video can trip several at once). Grounded in real reference material
    (Winning Hook Criteria, Features/Benefits/Outcomes, VOC doc, VSL and
    Advetorial breakdowns) — see copy_classifier.py's module docstring.

    Every field is Optional[bool]: None means "the classifier genuinely
    couldn't tell" (e.g. no speech/on-screen text at all), which is a
    different claim than False ("present and confidently absent") — never
    collapse the two."""
    hook_curiosity_open_loop: Optional[bool] = Field(
        description="Teases a result/reveal without naming the mechanism")
    hook_pain_callout: Optional[bool] = Field(
        description="Names the viewer's felt pain directly in the first ~3s")
    hook_promise_without_mechanism: Optional[bool] = Field(
        description="States the outcome/benefit, withholds the 'how'")
    hook_specificity: Optional[bool] = Field(
        description="Numbers/timeframes/quantities in the hook")
    hook_credibility_signal: Optional[bool] = Field(
        description="Authority/expert/clinical claim in the hook itself")
    hook_pattern_interrupt: Optional[bool] = Field(
        description="A belief-breaking/'sacred cow smash' opener")
    hook_status_shift: Optional[bool] = Field(
        description="'People you look up to aren't doing what you do' framing")

    mentions_customer_problem: Optional[bool] = Field(
        description="Names the problem the product solves, in any form")
    uses_voc_language: Optional[bool] = Field(
        description="Uses raw/colloquial customer phrasing, not clinical/marketing language")
    agitates_pain_emotionally: Optional[bool] = Field(
        description="Escalates consequence/shame/fear, not just states the problem")
    burns_alternatives: Optional[bool] = Field(
        description="Explicitly dismisses other things the viewer already tried")

    explains_mechanism: Optional[bool] = Field(
        description="Explains WHY/HOW it works, not just that it works")
    discovery_moment_framing: Optional[bool] = Field(
        description="Introduces the product as a casual/organic discovery, not a hard pitch")
    uses_sensory_visual_language: Optional[bool] = Field(
        description="Translates a benefit into a felt/visual metaphor")
    features_framed_as_benefits: Optional[bool] = Field(
        description="A feature is explicitly tied to why it matters to the viewer")

    uses_social_proof: Optional[bool] = Field(
        description="Testimonials, review counts, 'X customers' claims")
    uses_authority_or_expert_proof: Optional[bool] = Field(
        description="Credentials, institutions, 'doctor/nurse/herbalist reveals'")
    uses_before_after_contrast: Optional[bool] = Field(
        description="Explicit then-vs-now framing")

    creates_urgency: Optional[bool] = Field(
        description="Time pressure, scarcity, 'while it's still available'")
    uses_risk_reversal: Optional[bool] = Field(
        description="Guarantee, 'no questions asked', free trial")
    cta_direct_address: Optional[bool] = Field(
        description="Direct 'you' framing in the call to action")
    promises_identity_or_status_shift: Optional[bool] = Field(
        description="The ask is framed as becoming a different kind of person, not just buying")
