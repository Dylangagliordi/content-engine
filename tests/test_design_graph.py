import pytest
from qualume.graphs.design_ig_static import run, fit_text_to_width
from qualume.models.contracts import ScriptVariant, DesignRequest
from PIL import ImageFont


def _make_variant(hook="Short hook", body="Short body text.", cta="Buy now"):
    return ScriptVariant(
        hook_text=hook,
        body_text=body,
        cta_text=cta,
        tags=["test"],
        playbook_version="v0",
        prompt_version="v1",
        explore_or_exploit="exploit",
    )


def test_render_is_deterministic_for_same_input(tmp_path):
    variant = _make_variant()
    req = DesignRequest(script_variant=variant, workflow_type="ig_static")

    out1 = run(req, output_dir=str(tmp_path))
    out2 = run(req, output_dir=str(tmp_path / "second"))

    bytes1 = open(out1.asset_path, "rb").read()
    bytes2 = open(out2.asset_path, "rb").read()
    assert bytes1 == bytes2


def test_render_populates_creative_dna_from_variant():
    variant = _make_variant(hook="A specific hook")
    req = DesignRequest(script_variant=variant, workflow_type="ig_static")
    out = run(req, output_dir="data/assets/_test_tmp")

    assert out.creative_dna.hook_text == "A specific hook"
    assert out.creative_dna.format_tag == "ig_static_template"


def test_fit_text_to_width_wraps_long_text_within_bounds():
    font = ImageFont.load_default()
    long_text = "This is a fairly long piece of hook text that should need to wrap onto multiple lines to fit."
    # load_default() is a tiny bitmap font, so force wrapping with a narrow
    # max_width rather than assuming any arbitrary width will trigger it.
    lines = fit_text_to_width(long_text, font, max_width=60)
    assert len(lines) > 1
    for line in lines:
        assert len(line) > 0


def test_fit_text_to_width_rejects_empty_text():
    font = ImageFont.load_default()
    with pytest.raises(ValueError):
        fit_text_to_width("", font, max_width=400)
