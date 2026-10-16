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
