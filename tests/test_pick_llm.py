import pytest
from qualume.llm.pick_llm import pick_llm


def test_pick_llm_rejects_unknown_level():
    with pytest.raises(ValueError):
        pick_llm("ultra")


def test_pick_llm_returns_object_for_each_valid_level():
    for level in ("low", "medium", "high"):
        llm = pick_llm(level)
        assert llm is not None
