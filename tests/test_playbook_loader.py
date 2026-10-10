import pytest
from qualume.db.playbook_loader import load_active_playbook


def test_load_active_playbook_raises_when_none_exists():
    with pytest.raises(ValueError):
        load_active_playbook("nonexistent_workflow_type")


def test_load_active_playbook_returns_rubric_for_ig_static():
    # v2 (seeded by qualume.db.seed_playbooks) is the newest: v1's anti-fabrication
    # `claims` rule plus free vyralcontent hook hypotheses. Assert structure, not an
    # exact version pin, so seeding a new version doesn't require touching this test
    # every time (the loader's actual contract is "always the newest row").
    result = load_active_playbook("ig_static")
    assert "rubric" in result
    assert "hooks" in result["rubric"]
    assert "claims" in result["rubric"]
