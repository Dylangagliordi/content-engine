import pytest
from qualume.db.playbook_loader import load_active_playbook


def test_load_active_playbook_raises_when_none_exists():
    with pytest.raises(ValueError):
        load_active_playbook("nonexistent_workflow_type")


def test_load_active_playbook_returns_rubric_for_ig_static():
    # v1 (seeded by qualume.db.seed_playbooks) adds the anti-fabrication `claims` rule.
    result = load_active_playbook("ig_static")
    assert result["version"] == "v1"
    assert "rubric" in result
    assert "hooks" in result["rubric"]
    assert "claims" in result["rubric"]
