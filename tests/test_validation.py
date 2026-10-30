"""Tests for the insights_external -> insights_owned weighting/validation
mechanism — written FIRST per this project's TDD discipline.

Per plans/learn-engine-idea-sourcing-architecture.md: insights_external
is "this worked for other accounts" (lower trust ceiling even at
tier:accepted); insights_owned is "this works for us" (drives generation
decisions, CAN DIVERGE from the external version). The real mechanism
("use experiments, don't invent a new mechanism"): an external insight
gets tested via a real experiments row with a stated hypothesis, then
real owned performance (judged against OUR OWN account's median baseline,
same mechanism as external) becomes evidence for or against the LINKED
insights_owned row — which can end up with a different tier/effect_size
than its external parent.

Zero real owned content/performance exists yet (0 content_items published,
0 performance_snapshots) — this is a correctly-timed SCAFFOLD, built and
tested with synthetic evidence now so it's ready the moment real owned
results exist, per the project's "capture the shape before the volume"
discipline (same pattern as learn.replication).
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.learn.validation import (
    propose_external_insight,
    seed_owned_insight,
    record_owned_evidence,
    upsert_external_insight,
    retire_stale_external_insights,
)


def _cleanup(statement_prefix: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM insights_owned WHERE statement LIKE %s",
                (f"{statement_prefix}%",),
            )
            cur.execute(
                "DELETE FROM insights_external WHERE statement LIKE %s",
                (f"{statement_prefix}%",),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def prefix():
    p = f"zz_test_{uuid.uuid4().hex[:8]}: "
    _cleanup(p)
    yield p
    _cleanup(p)


def test_propose_external_insight_inserts_proposed_tier_row(prefix):
    insight_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tier, effect_size, sample_size, p_value, source FROM insights_external WHERE id = %s",
                (insight_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row[0] == "proposed"
    assert float(row[1]) == 0.35
    assert row[2] == 24
    assert float(row[3]) == 0.02
    assert row[4] == "internal_mining"


def test_seed_owned_insight_links_back_to_external_parent(prefix):
    external_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    owned_id = seed_owned_insight(external_id, applies_to_workflow_type="tiktok_video")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tier, sample_size, derived_from_external_insight_id FROM insights_owned WHERE id = %s",
                (owned_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row[0] == "proposed"
    assert row[1] == 0  # zero owned evidence yet - real, not guessed
    assert str(row[2]) == external_id


def test_seed_owned_insight_rejects_unknown_external_parent():
    with pytest.raises(ValueError):
        seed_owned_insight(str(uuid.uuid4()), applies_to_workflow_type="tiktok_video")


def test_record_owned_evidence_accumulates_sample_size(prefix):
    external_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    owned_id = seed_owned_insight(external_id, applies_to_workflow_type="tiktok_video")

    record_owned_evidence(owned_id, confirmed=True, content_item_id=str(uuid.uuid4()))
    record_owned_evidence(owned_id, confirmed=True, content_item_id=str(uuid.uuid4()))
    record_owned_evidence(owned_id, confirmed=False, content_item_id=str(uuid.uuid4()))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sample_size, state_history FROM insights_owned WHERE id = %s",
                (owned_id,),
            )
            sample_size, state_history = cur.fetchone()
    finally:
        conn.close()
    assert sample_size == 3  # one real data point per piece of evidence, not re-weighted
    assert len(state_history) == 3  # every piece of evidence is traceable, not just a running count


def test_record_owned_evidence_can_diverge_from_external_tier(prefix):
    """The real point of the split: owned evidence can CONTRADICT the
    external insight (accepted externally off a huge competitor sample,
    still proposed or even rejected against OUR OWN account) - the owned
    row's tier is driven by OWNED evidence, never inherited wholesale
    from its external parent."""
    external_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    owned_id = seed_owned_insight(external_id, applies_to_workflow_type="tiktok_video")

    # Repeated, consistent OWNED contradiction - the hypothesis just does
    # not hold for OUR account, regardless of external's 24-sample accept.
    for _ in range(5):
        record_owned_evidence(owned_id, confirmed=False, content_item_id=str(uuid.uuid4()))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT owned.tier, external_insight.tier FROM insights_owned owned "
                "JOIN insights_external external_insight ON external_insight.id = owned.derived_from_external_insight_id "
                "WHERE owned.id = %s",
                (owned_id,),
            )
            owned_tier, external_tier = cur.fetchone()
    finally:
        conn.close()
    assert external_tier == "proposed"  # untouched - this function never mutates the external row
    assert owned_tier == "rejected"  # owned evidence drove its OWN tier, independently


def test_record_owned_evidence_promotes_to_accepted_on_consistent_confirmation(prefix):
    external_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    owned_id = seed_owned_insight(external_id, applies_to_workflow_type="tiktok_video")

    for _ in range(5):
        record_owned_evidence(owned_id, confirmed=True, content_item_id=str(uuid.uuid4()))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tier FROM insights_owned WHERE id = %s", (owned_id,))
            (tier,) = cur.fetchone()
    finally:
        conn.close()
    assert tier == "accepted"


def test_record_owned_evidence_stays_testing_with_thin_or_mixed_evidence(prefix):
    """Not enough evidence yet, or genuinely mixed results - must stay at
    'testing', not prematurely claim accepted or rejected off 1-2 points."""
    external_id = propose_external_insight(
        statement=f"{prefix}hook_type=question outperforms",
        effect_size=0.35, sample_size=24, p_value=0.02,
        applies_to_workflow_type="tiktok_video",
    )
    owned_id = seed_owned_insight(external_id, applies_to_workflow_type="tiktok_video")

    record_owned_evidence(owned_id, confirmed=True, content_item_id=str(uuid.uuid4()))
    record_owned_evidence(owned_id, confirmed=False, content_item_id=str(uuid.uuid4()))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tier FROM insights_owned WHERE id = %s", (owned_id,))
            (tier,) = cur.fetchone()
    finally:
        conn.close()
    assert tier == "testing"


def test_record_owned_evidence_rejects_unknown_owned_insight():
    with pytest.raises(ValueError):
        record_owned_evidence(str(uuid.uuid4()), confirmed=True, content_item_id=str(uuid.uuid4()))


def test_upsert_external_insight_inserts_new_row_on_first_call(prefix):
    insight_id = upsert_external_insight(
        statement=f"{prefix}format_tag=voiceover_broll underperforms",
        effect_size=0.57, sample_size=60, p_value=0.0021,
        applies_to_workflow_type="tiktok_video",
        identity={"mined_from": "external_creative_dna.format_tag", "value": "voiceover_broll"},
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tier, sample_size, state_history FROM insights_external WHERE id = %s",
                (insight_id,),
            )
            tier, sample_size, state_history = cur.fetchone()
    finally:
        conn.close()
    assert tier == "proposed"
    assert sample_size == 60
    assert len(state_history) == 1
    assert state_history[0]["event"] == "proposed"


def test_upsert_external_insight_refreshes_same_identity_instead_of_duplicating(prefix):
    """The real point: a scheduled mining pass re-running the SAME finding
    (same identity) every cycle must update one row, not pile up N
    duplicate 'proposed' rows for an unchanged claim."""
    identity = {"mined_from": "external_creative_dna.format_tag", "value": "voiceover_broll"}
    first_id = upsert_external_insight(
        statement=f"{prefix}format_tag=voiceover_broll underperforms (n=60)",
        effect_size=0.57, sample_size=60, p_value=0.0021,
        applies_to_workflow_type="tiktok_video", identity=identity,
    )
    second_id = upsert_external_insight(
        statement=f"{prefix}format_tag=voiceover_broll underperforms (n=95)",
        effect_size=0.61, sample_size=95, p_value=0.0015,
        applies_to_workflow_type="tiktok_video", identity=identity,
    )
    assert first_id == second_id  # same row, refreshed, not a new one

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM insights_external WHERE id = %s",
                (first_id,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT sample_size, p_value, state_history FROM insights_external WHERE id = %s",
                (first_id,),
            )
            sample_size, p_value, state_history = cur.fetchone()
    finally:
        conn.close()
    assert sample_size == 95  # latest numbers, not the first pass's stale ones
    assert float(p_value) == 0.0015
    assert len(state_history) == 2  # proposed, then refreshed — full history kept
    assert state_history[1]["event"] == "refreshed"


def test_upsert_external_insight_different_identity_creates_a_separate_row(prefix):
    id_a = upsert_external_insight(
        statement=f"{prefix}format_tag=voiceover_broll underperforms",
        effect_size=0.57, sample_size=60, p_value=0.0021,
        applies_to_workflow_type="tiktok_video",
        identity={"mined_from": "external_creative_dna.format_tag", "value": "voiceover_broll"},
    )
    id_b = upsert_external_insight(
        statement=f"{prefix}hook_pain_callout underperforms",
        effect_size=0.76, sample_size=60, p_value=0.0398,
        applies_to_workflow_type="tiktok_video",
        identity={"mined_from": "external_copy_analysis.hook_pain_callout", "value": True},
    )
    assert id_a != id_b


def test_upsert_external_insight_never_overwrites_a_different_source(prefix):
    """A hand-authored insight (source != 'internal_mining') that happens to
    share an identity must never get silently clobbered by an automatic
    mining-pass upsert."""
    manual_id = upsert_external_insight(
        statement=f"{prefix}manually curated claim",
        effect_size=0.5, sample_size=10, p_value=0.01,
        applies_to_workflow_type="tiktok_video",
        identity={"mined_from": "manual_review", "value": "x"},
        source="manual",
    )
    auto_id = upsert_external_insight(
        statement=f"{prefix}mining found the same identity string",
        effect_size=0.9, sample_size=60, p_value=0.001,
        applies_to_workflow_type="tiktok_video",
        identity={"mined_from": "manual_review", "value": "x"},
        source="internal_mining",
    )
    assert manual_id != auto_id

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT statement, source FROM insights_external WHERE id = %s", (manual_id,))
            statement, source = cur.fetchone()
    finally:
        conn.close()
    assert source == "manual"
    assert "manually curated claim" in statement  # untouched by the auto upsert


def test_retire_stale_external_insights_retires_no_longer_significant_proposed_rows(prefix):
    identity_a = {"mined_from": "external_creative_dna.format_tag", "value": "voiceover_broll"}
    identity_b = {"mined_from": "external_copy_analysis.hook_pain_callout", "value": True}
    id_a = upsert_external_insight(
        statement=f"{prefix}finding A", effect_size=0.57, sample_size=60, p_value=0.0021,
        applies_to_workflow_type="tiktok_video", identity=identity_a,
    )
    id_b = upsert_external_insight(
        statement=f"{prefix}finding B", effect_size=0.76, sample_size=60, p_value=0.0398,
        applies_to_workflow_type="tiktok_video", identity=identity_b,
    )

    # Next mining pass: only finding A is still significant.
    retired = retire_stale_external_insights(
        applies_to_workflow_type="tiktok_video",
        still_significant_identities=[identity_a],
    )
    assert retired == 1

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tier FROM insights_external WHERE id = %s", (id_a,))
            tier_a = cur.fetchone()[0]
            cur.execute("SELECT tier, state_history FROM insights_external WHERE id = %s", (id_b,))
            tier_b, state_history_b = cur.fetchone()
    finally:
        conn.close()
    assert tier_a == "proposed"  # still significant, untouched
    assert tier_b == "retired"
    assert state_history_b[-1]["event"] == "retired"


def test_retire_stale_external_insights_never_touches_accepted_or_testing_tier(prefix):
    identity = {"mined_from": "external_creative_dna.format_tag", "value": "voiceover_broll"}
    insight_id = upsert_external_insight(
        statement=f"{prefix}finding", effect_size=0.57, sample_size=60, p_value=0.0021,
        applies_to_workflow_type="tiktok_video", identity=identity,
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE insights_external SET tier = 'accepted' WHERE id = %s", (insight_id,))
        conn.commit()
    finally:
        conn.close()

    retired = retire_stale_external_insights(
        applies_to_workflow_type="tiktok_video", still_significant_identities=[],
    )
    assert retired == 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tier FROM insights_external WHERE id = %s", (insight_id,))
            tier = cur.fetchone()[0]
    finally:
        conn.close()
    assert tier == "accepted"  # never demoted by a mining pass, only proposed-tier is
