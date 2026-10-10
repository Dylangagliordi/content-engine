from unittest.mock import patch, MagicMock

import pytest

from qualume.graphs import write_ig_static
from qualume.graphs.write_ig_static import (
    run,
    AngleList,
    VariantDraft,
    CritiqueResult,
)
from qualume.models.contracts import CampaignBrief, ScriptVariant


@pytest.fixture(autouse=True)
def _isolated_attempt_log(tmp_path, monkeypatch):
    """Keep self_critique's per-attempt log out of the real data/ directory."""
    monkeypatch.setattr(write_ig_static, "ATTEMPT_LOG_PATH", str(tmp_path / "attempts.jsonl"))


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
        assert v.prompt_version == "v2"
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
    # max_attempts=1 = no retry; the retry path is covered in test_write_retry_loop.py.
    variants = run(brief, count=1, max_attempts=1)

    assert len(variants) == 0


def _system_block(messages):
    """Return the single text block of the leading SystemMessage in an invoke() call."""
    from langchain_core.messages import SystemMessage

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[0].content, list) and len(messages[0].content) == 1
    return messages[0].content[0]


@patch("qualume.graphs.write_ig_static.load_active_playbook")
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_every_llm_call_puts_static_rubric_in_cached_system_block(mock_pick_llm, mock_load_playbook):
    """Prompt caching: the rubric + fixed framing must sit in a SystemMessage block
    marked cache_control, byte-identical across calls within a node, and the
    per-call variable content (brief, angle, copy) must NOT be in that block."""
    mock_load_playbook.return_value = {
        "version": "v0",
        "rubric": {"hooks": ["be concrete"], "format": ["one headline"]},
    }
    ideate = _fake_structured_llm([AngleList(angles=["angle one", "angle two"])])
    generate = _fake_structured_llm([
        VariantDraft(hook_text="Hook A", body_text="Body A", cta_text="Shop now"),
        VariantDraft(hook_text="Hook B", body_text="Body B", cta_text="Learn more"),
    ])
    critique = _fake_structured_llm([
        CritiqueResult(passes=True, notes="fine"),
        CritiqueResult(passes=True, notes="fine"),
    ])
    mock_pick_llm.side_effect = [ideate, generate, critique]

    brief = CampaignBrief(
        product_or_topic="UNIQUE_PRODUCT", offer_or_goal="UNIQUE_GOAL", playbook_version="pending",
    )
    run(brief, count=2)

    for node_llm in (ideate, generate, critique):
        calls = node_llm.with_structured_output.return_value.invoke.call_args_list
        blocks = [_system_block(c.args[0]) for c in calls]
        for b in blocks:
            assert b["cache_control"] == {"type": "ephemeral"}
            assert "be concrete" in b["text"]
            for variable in ("UNIQUE_PRODUCT", "UNIQUE_GOAL", "angle one", "Hook A"):
                assert variable not in b["text"]
        assert all(b == blocks[0] for b in blocks)
