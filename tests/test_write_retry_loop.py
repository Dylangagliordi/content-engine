"""Conditional retry loop for self_critique (fail -> regenerate with notes -> re-judge).

Uses a schema-keyed scripted LLM instead of the per-node call_sequence used in
test_write_graph.py: the loop re-enters nodes a variable number of times, so
the fake must answer by *what* is being asked (AngleList / VariantDraft /
CritiqueResult), not by how many times pick_llm happened to be called.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from qualume.graphs import write_ig_static
from qualume.graphs.write_ig_static import (
    AngleList,
    CritiqueResult,
    MAX_ATTEMPTS_CEILING,
    PROMPT_VERSION,
    RETRY_PROMPT_VERSION,
    VariantDraft,
    WriteState,
    route_after_critique,
    run,
)
from qualume.models.contracts import CampaignBrief, ScriptVariant

PLAYBOOK = {"version": "v0", "rubric": {"hooks": ["be concrete"]}}


class ScriptedLLM:
    """Stands in for pick_llm(level): with_structured_output(Schema).invoke(msgs)
    pops the next scripted response for that Schema and records the messages.
    Running out of responses raises IndexError -> an unexpected extra call fails loudly."""

    def __init__(self, scripts):
        self.scripts = {schema: list(responses) for schema, responses in scripts.items()}
        self.calls = {schema: [] for schema in scripts}

    def __call__(self, level="medium", **kwargs):
        return self

    def with_structured_output(self, schema):
        structured = MagicMock()

        def _invoke(messages):
            self.calls[schema].append(messages)
            return self.scripts[schema].pop(0)

        structured.invoke.side_effect = _invoke
        return structured


def _brief():
    return CampaignBrief(product_or_topic="test product", offer_or_goal="test goal",
                         playbook_version="pending")


def _read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(write_ig_static, "ATTEMPT_LOG_PATH", str(path))
    return path


@patch("qualume.graphs.write_ig_static.load_active_playbook", return_value=PLAYBOOK)
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_failed_variant_is_retried_with_notes_and_passes_on_attempt_2(mock_pick_llm, _pb, log_path):
    llm = ScriptedLLM({
        AngleList: [AngleList(angles=["angle one", "angle two"])],
        VariantDraft: [
            VariantDraft(hook_text="Good hook", body_text="Good body", cta_text="Shop"),
            VariantDraft(hook_text="Vague hook", body_text="Vague body", cta_text="Buy"),
            VariantDraft(hook_text="Fixed hook", body_text="Fixed body", cta_text="Buy"),  # retry
        ],
        CritiqueResult: [
            CritiqueResult(passes=True, notes="fine"),
            CritiqueResult(passes=False, notes="NOTE_HOOK_TOO_VAGUE"),
            CritiqueResult(passes=True, notes="now concrete"),  # re-judge of the retry
        ],
    })
    mock_pick_llm.side_effect = llm

    variants = run(_brief(), count=2, max_attempts=2)

    # Both survive: angle one on attempt 1, angle two on attempt 2.
    assert [v.hook_text for v in variants] == ["Good hook", "Fixed hook"]
    assert all(isinstance(v, ScriptVariant) for v in variants)
    assert variants[0].prompt_version == PROMPT_VERSION
    assert variants[1].prompt_version == f"{PROMPT_VERSION}+{RETRY_PROMPT_VERSION}"
    assert variants[1].tags == ["angle two"]  # same angle kept across the retry

    # Only the failed variant was regenerated (2 first-pass + 1 retry), and only
    # it was re-judged (2 + 1).
    assert len(llm.calls[VariantDraft]) == 3
    assert len(llm.calls[CritiqueResult]) == 3

    # The retry prompt carries the same angle, the rejected draft, and the judge's notes —
    # in the per-call user turn, NOT the cached system block.
    retry_msgs = llm.calls[VariantDraft][2]
    system_text = retry_msgs[0].content[0]["text"]
    user_text = retry_msgs[1].content
    assert "Angle: angle two" in user_text
    assert "Vague hook" in user_text
    assert "NOTE_HOOK_TOO_VAGUE" in user_text
    assert "NOTE_HOOK_TOO_VAGUE" not in system_text
    # Same static prefix as first-pass generation, so the cache is shared.
    assert system_text == llm.calls[VariantDraft][0][0].content[0]["text"]

    rows = _read_log(log_path)
    assert [(r["variant_angle"], r["attempt_number"], r["passed"], r["decision"]) for r in rows] == [
        ("angle one", 1, True, "accepted"),
        ("angle two", 1, False, "retry"),
        ("angle two", 2, True, "accepted"),
    ]
    assert rows[1]["critique_notes"] == "NOTE_HOOK_TOO_VAGUE"
    assert rows[1]["generated_by"] == "generate_variants"
    assert rows[2]["generated_by"] == "regenerate_failed"
    assert len({r["run_id"] for r in rows}) == 1
    assert all(r["max_attempts"] == 2 for r in rows)


@patch("qualume.graphs.write_ig_static.load_active_playbook", return_value=PLAYBOOK)
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_max_attempts_1_preserves_no_retry_behavior(mock_pick_llm, _pb, log_path):
    llm = ScriptedLLM({
        AngleList: [AngleList(angles=["angle one"])],
        VariantDraft: [VariantDraft(hook_text="Bad", body_text="Bad", cta_text="Buy")],
        CritiqueResult: [CritiqueResult(passes=False, notes="too vague")],
    })
    mock_pick_llm.side_effect = llm

    assert run(_brief(), count=1, max_attempts=1) == []
    assert len(llm.calls[VariantDraft]) == 1
    rows = _read_log(log_path)
    assert [(r["attempt_number"], r["passed"], r["decision"]) for r in rows] == [
        (1, False, "dropped_max_attempts"),
    ]


@patch("qualume.graphs.write_ig_static.load_active_playbook", return_value=PLAYBOOK)
@patch("qualume.graphs.write_ig_static.pick_llm")
def test_hard_ceiling_stops_loop_when_judge_always_fails(mock_pick_llm, _pb, log_path):
    n = MAX_ATTEMPTS_CEILING
    llm = ScriptedLLM({
        AngleList: [AngleList(angles=["angle one"])],
        # Script MORE responses than the ceiling allows — if the loop overran,
        # it would happily consume them, so the call-count assert is meaningful.
        VariantDraft: [VariantDraft(hook_text=f"H{i}", body_text="B", cta_text="C") for i in range(n + 5)],
        CritiqueResult: [CritiqueResult(passes=False, notes=f"fail {i}") for i in range(n + 5)],
    })
    mock_pick_llm.side_effect = llm

    variants = run(_brief(), count=1, max_attempts=1000)

    assert variants == []
    assert len(llm.calls[VariantDraft]) == n
    assert len(llm.calls[CritiqueResult]) == n
    rows = _read_log(log_path)
    assert [r["attempt_number"] for r in rows] == list(range(1, n + 1))
    assert [r["decision"] for r in rows] == ["retry"] * (n - 1) + ["dropped_ceiling"]
    # Each retry is fed the notes from the attempt immediately before it.
    for i in range(1, n):
        assert f"fail {i - 1}" in llm.calls[VariantDraft][i][1].content


def test_route_after_critique_returns_node_names():
    variant = ScriptVariant(hook_text="h", body_text="b", cta_text="c", tags=["a"],
                            playbook_version="v0", prompt_version="v2", explore_or_exploit="explore")
    base = dict(brief=_brief(), angles=["a"], variants=[variant], critique_notes=["x"])

    failed_with_budget = WriteState(**base, max_attempts=2, attempts=[1], verdicts=[False])
    failed_no_budget = WriteState(**base, max_attempts=2, attempts=[2], verdicts=[False])
    passed = WriteState(**base, max_attempts=2, attempts=[1], verdicts=[True])
    over_ceiling = WriteState(**base, max_attempts=99, attempts=[MAX_ATTEMPTS_CEILING], verdicts=[False])

    assert route_after_critique(failed_with_budget) == "regenerate_failed"
    assert route_after_critique(failed_no_budget) == "finalize"
    assert route_after_critique(passed) == "finalize"
    assert route_after_critique(over_ceiling) == "finalize"


def test_max_attempts_must_be_at_least_one():
    with pytest.raises(ValueError):
        WriteState(brief=_brief(), max_attempts=0)
