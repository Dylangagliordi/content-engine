from datetime import datetime
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


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
    external_creative_dna row. Deliberately excludes pacing_cuts_per_min and
    music_mood: those are computed from the signal, never LLM-guessed."""
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
