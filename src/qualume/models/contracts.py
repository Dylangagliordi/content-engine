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
