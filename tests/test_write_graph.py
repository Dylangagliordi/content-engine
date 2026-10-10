from unittest.mock import patch, MagicMock

from qualume.graphs.write_ig_static import (
    run,
    AngleList,
    VariantDraft,
    CritiqueResult,
)
from qualume.models.contracts import CampaignBrief, ScriptVariant


def _fake_structured_llm(responses):
    """Return a MagicMock chain: llm.with_structured_output(X).invoke(prompt) -> responses.pop(0)."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = responses
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


@patch("qualume.graphs.write_ig_static.load_active_playbook")
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_write_graph_produces_expected_variant_count_and_contract(mock_pick_llm, mock_load_playbook):
    mock_load_playbook.return_value = {
        "version": "v0",
        "rubric": {"hooks": ["be concrete"], "format": ["one headline"], "social_proof": ["sparingly"]},
    }

    angle_response = AngleList(angles=["angle one", "angle two"])
    variant_responses = [
        VariantDraft(hook_text="Hook A", body_text="Body A", cta_text="Shop now"),
        VariantDraft(hook_text="Hook B", body_text="Body B", cta_text="Learn more"),
    ]
    critique_responses = [
        CritiqueResult(passes=True, notes="fine"),
        CritiqueResult(passes=True, notes="fine"),
    ]

    # pick_llm is called ONCE per node (not per variant) — each node then
    # calls .invoke() on that same llm object once per item it's iterating.
    call_sequence = [
        _fake_structured_llm([angle_response]),       # ideate_angles_node: 1 call
        _fake_structured_llm(variant_responses),       # generate_variants_node: 2 calls (1/angle)
        _fake_structured_llm(critique_responses),      # self_critique_node: 2 calls (1/variant)
    ]
    mock_pick_llm.side_effect = call_sequence

    brief = CampaignBrief(
        product_or_topic="test product",
        offer_or_goal="test goal",
        playbook_version="pending",
    )
    variants = run(brief, count=2)

    assert len(variants) == 2
    for v in variants:
        assert isinstance(v, ScriptVariant)
        assert v.playbook_version == "v0"
        assert v.prompt_version == "v1"
        assert v.explore_or_exploit in ("explore", "exploit")


@patch("qualume.graphs.write_ig_static.load_active_playbook")
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_write_graph_filters_out_failed_critique(mock_pick_llm, mock_load_playbook):
    mock_load_playbook.return_value = {
        "version": "v0",
        "rubric": {"hooks": ["be concrete"]},
    }

    angle_response = AngleList(angles=["angle one"])
    variant_response = VariantDraft(hook_text="Bad hook", body_text="Bad body", cta_text="Buy")
    critique_response = CritiqueResult(passes=False, notes="too vague")

    call_sequence = [
        _fake_structured_llm([angle_response]),
        _fake_structured_llm([variant_response]),
        _fake_structured_llm([critique_response]),
    ]
    mock_pick_llm.side_effect = call_sequence

    brief = CampaignBrief(
        product_or_topic="test product",
        offer_or_goal="test goal",
        playbook_version="pending",
    )
    variants = run(brief, count=1)

    assert len(variants) == 0
