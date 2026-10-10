import pytest

from qualume.db.connection import get_connection
from qualume.db.playbook_loader import load_active_playbook
from qualume.db.seed_playbooks import PLAYBOOKS, seed_playbook

TEST_WORKFLOW = "test_seed_playbooks_tmp"


def _ig_static(version):
    return next(p for p in PLAYBOOKS if p["workflow_type"] == "ig_static" and p["version"] == version)


def test_ig_static_v1_adds_claims_rule_and_keeps_v0_rubric_verbatim():
    v0, v1 = _ig_static("v0"), _ig_static("v1")
    claims = v1["rubric"]["claims"]
    assert any("statistics" in rule and "brief" in rule for rule in claims)
    assert any("testimonial" in rule and "brief" in rule for rule in claims)
    # Everything else (incl. the honest tier/source labels) carries over unchanged.
    assert {k: v for k, v in v1["rubric"].items() if k != "claims"} == v0["rubric"]
    assert "claims" not in v0["rubric"]


def _cleanup(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM playbooks WHERE workflow_type = %s", (TEST_WORKFLOW,))
    conn.commit()


def test_seed_playbook_is_rerunnable_and_versions_are_immutable():
    conn = get_connection()
    try:
        assert seed_playbook(conn, TEST_WORKFLOW, "v0", {"hooks": ["a"]}) == "inserted"
        assert seed_playbook(conn, TEST_WORKFLOW, "v0", {"hooks": ["a"]}) == "exists"
        # Same version, different rubric: fail loud instead of silently editing history.
        with pytest.raises(ValueError):
            seed_playbook(conn, TEST_WORKFLOW, "v0", {"hooks": ["changed"]})
    finally:
        _cleanup(conn)
        conn.close()


def test_later_seeded_version_becomes_active_even_in_one_session():
    conn = get_connection()
    try:
        seed_playbook(conn, TEST_WORKFLOW, "v0", {"hooks": ["a"]})
        seed_playbook(conn, TEST_WORKFLOW, "v1", {"hooks": ["b"]})
        assert load_active_playbook(TEST_WORKFLOW)["version"] == "v1"
    finally:
        _cleanup(conn)
        conn.close()
