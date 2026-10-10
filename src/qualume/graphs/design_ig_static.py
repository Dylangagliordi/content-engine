"""DESIGN engine — ig_static workflow.

Pillow-based template rendering (NOT an image-gen model). Deliberately the
simplest, zero-API-cost, fully deterministic option for Phase 1 — same input
always produces byte-identical output, satisfying the render-determinism test
from the engineering plan (v2 plan §2.4). Swapping in a real image-gen
tool/service is an explicit fast-follow, not a Phase 1 blocker.

Pipeline (not a LangGraph graph — this stage has no LLM decision points, it's
a deterministic render pipeline, so a plain function pipeline is the right
tool, not an unnecessary graph wrapper):

    select_template -> place_headline_and_body -> brand_consistency_check -> render_png
"""

import os
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qualume.models.contracts import DesignRequest, DesignOutput, CreativeDNA

PROMPT_VERSION = "v1"

TEMPLATE_PATH = "data/templates/template_cream_square.png"
FONT_PATH_HEADLINE = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_PATH_BODY = "/System/Library/Fonts/Supplemental/Arial.ttf"
TEXT_COLOR = (30, 27, 22)
MARGIN = 80
MAX_TEXT_WIDTH = 920  # 1080 canvas minus margins, leaves breathing room


def fit_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels for the given font.

    Deterministic: same text + font + max_width always produces the same
    line breaks. Raises ValueError on empty text — fail loud rather than
    silently rendering a blank line (a blank hook is exactly the "visually
    dead open" failure mode the v2 plan flags as a mechanical design-engine
    bug to catch, not something to ever treat as valid output).
    """
    if not text.strip():
        raise ValueError("Cannot fit empty text — a blank hook/body is treated as a bug, not valid output.")

    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def select_template(workflow_type: str) -> Image.Image:
    if workflow_type != "ig_static":
        raise ValueError(f"No template available for workflow_type={workflow_type!r}")
    return Image.open(TEMPLATE_PATH).convert("RGB")


def place_headline_and_body(img: Image.Image, hook_text: str, body_text: str, cta_text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    headline_font = ImageFont.truetype(FONT_PATH_HEADLINE, 64)
    body_font = ImageFont.truetype(FONT_PATH_BODY, 36)
    cta_font = ImageFont.truetype(FONT_PATH_HEADLINE, 32)

    y = MARGIN + 60

    hook_lines = fit_text_to_width(hook_text, headline_font, MAX_TEXT_WIDTH)
    for line in hook_lines:
        draw.text((MARGIN, y), line, font=headline_font, fill=TEXT_COLOR)
        bbox = draw.textbbox((0, 0), line, font=headline_font)
        y += (bbox[3] - bbox[1]) + 16

    y += 40  # gap between hook block and body block

    body_lines = fit_text_to_width(body_text, body_font, MAX_TEXT_WIDTH)
    for line in body_lines:
        draw.text((MARGIN, y), line, font=body_font, fill=TEXT_COLOR)
        bbox = draw.textbbox((0, 0), line, font=body_font)
        y += (bbox[3] - bbox[1]) + 10

    # CTA pinned near the bottom, not flowing with body — keeps it visually
    # distinct regardless of how long the body text runs.
    cta_y = img.height - MARGIN - 50
    draw.text((MARGIN, cta_y), cta_text.upper(), font=cta_font, fill=TEXT_COLOR)

    return img


def brand_consistency_check(img: Image.Image) -> None:
    """Minimal automated check for Phase 1 — no real brand kit yet (greenfield).

    Asserts basic render sanity: correct canvas size, non-blank image (catches
    the "visually dead" failure mode mechanically rather than trusting manual
    review). Raises AssertionError (fail loud) if violated.
    """
    assert img.size == (1080, 1080), f"Unexpected canvas size: {img.size}"
    extrema = img.getextrema()
    # If every channel's min==max, the image is a single flat color — i.e.
    # nothing was actually drawn on top of the template.
    is_flat = all(lo == hi for lo, hi in extrema)
    assert not is_flat, "Rendered image is a single flat color — text was not drawn."


def render_png(img: Image.Image, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # Deterministic filename would collide across runs; use a uuid for the
    # filename but keep render *content* deterministic (tested separately).
    filename = f"ig_static_{uuid.uuid4().hex}.png"
    path = str(Path(output_dir) / filename)
    img.save(path, format="PNG")
    return path


def run(req: DesignRequest, output_dir: str = "data/assets") -> DesignOutput:
    img = select_template(req.workflow_type)
    img = place_headline_and_body(
        img,
        req.script_variant.hook_text,
        req.script_variant.body_text,
        req.script_variant.cta_text,
    )
    brand_consistency_check(img)
    asset_path = render_png(img, output_dir)

    creative_dna = CreativeDNA(
        content_item_id=uuid.uuid4(),  # placeholder — real content_item_id assigned on DB insert
        hook_text=req.script_variant.hook_text,
        format_tag="ig_static_template",
        script_structure_tags=req.script_variant.tags,
        cta_type="direct",
        caption_style="headline_body_cta",
        social_proof_used=False,
    )
    return DesignOutput(
        asset_path=asset_path,
        workflow_type=req.workflow_type,
        creative_dna=creative_dna,
    )
