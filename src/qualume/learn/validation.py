"""insights_external -> insights_owned validation scaffold — per
plans/learn-engine-idea-sourcing-architecture.md: "use experiments, don't
invent a new mechanism" and "closing the loop... re-validate the
insights that inspired them".

Real sequence this supports (not all built yet — only the owned-tier
logic is):
1. Mining/replication produces a candidate -> propose_external_insight.
2. A real experiments row gets created (NOT built here — run_batch.py's
   job, tier:testing in the architecture doc's sense) with an explicit
   hypothesis referencing this external insight -> seed_owned_insight
   creates the linked insights_owned row, zero evidence yet.
3. Each time a piece of OWNED content tied to that hypothesis gets real
   performance data back, judged against OUR OWN account's median
   baseline (same mechanism as external, learn.baseline) ->
   record_owned_evidence. The owned row's tier is driven ENTIRELY by
   accumulated OWNED evidence — it can diverge from (confirm, contradict,
   or just not yet have enough data to match) its external parent's
   tier, which is NEVER mutated by this module.

Zero real owned content/performance exists yet (0 published, 0
snapshots) — built as a scaffold now, tested with synthetic evidence,
ready the moment real results exist.
"""
TIER_PROMOTION_THRESHOLD = 5  # min pieces of evidence before leaving 'testing'
CONFIRMATION_RATE_FOR_ACCEPT = 0.8  # >=80% confirmed -> accepted
CONFIRMATION_RATE_FOR_REJECT = 0.2  # <=20% confirmed -> rejected
# Between those two thresholds (mixed evidence) stays 'testing' even past
# TIER_PROMOTION_THRESHOLD — a real split verdict shouldn't be forced into
# accepted/rejected just because enough data points exist.


def propose_external_insight(
    statement: str,
    effect_size: float,
    sample_size: int,
    p_value: float,
    applies_to_workflow_type: str,
    applies_to_segment: dict | None = None,
) -> str:
    """Insert a new insights_external row at tier='proposed' — the
    output shape of learn.mining/learn.replication's results, made
    durable and queryable rather than living only in a one-off script's
    stdout."""
    from qualume.db.connection import get_connection
    import json

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO insights_external
                   (statement, effect_size, sample_size, p_value,
                    applies_to_workflow_type, applies_to_segment)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (statement, effect_size, sample_size, p_value,
                 applies_to_workflow_type, json.dumps(applies_to_segment or {})),
            )
            insight_id = cur.fetchone()[0]
        conn.commit()
        return str(insight_id)
    finally:
        conn.close()


def seed_owned_insight(external_insight_id: str, applies_to_workflow_type: str) -> str:
    """Create the linked insights_owned row for an external insight that's
    being tested against OUR account (real experiments row with a stated
    hypothesis is the trigger for this, per the plan — not built here).
    Starts at tier='proposed', sample_size=0 — zero owned evidence exists
    yet, and that's reported honestly, not guessed or inherited from the
    external parent's sample_size."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT statement FROM insights_external WHERE id = %s", (external_insight_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"insights_external row {external_insight_id} not found")
            statement = row[0]

            cur.execute(
                """INSERT INTO insights_owned
                   (statement, sample_size, applies_to_workflow_type,
                    derived_from_external_insight_id)
                   VALUES (%s, 0, %s, %s)
                   RETURNING id""",
                (statement, applies_to_workflow_type, external_insight_id),
            )
            owned_id = cur.fetchone()[0]
        conn.commit()
        return str(owned_id)
    finally:
        conn.close()


def record_owned_evidence(owned_insight_id: str, confirmed: bool, content_item_id: str) -> None:
    """One real published piece's performance (already judged against OUR
    account's own median baseline — the caller's job, same mechanism as
    external) becomes ONE data point of evidence for or against this
    owned insight. Appends to state_history (full traceability of every
    real data point, not just a running tally) and re-derives tier from
    the accumulated confirmation rate — the ONLY thing driving this row's
    tier; its external parent is never read or mutated here.

    Tier logic: stays 'testing' until TIER_PROMOTION_THRESHOLD pieces of
    evidence exist, THEN promotes to 'accepted' (>=80% confirmed),
    'rejected' (<=20% confirmed), or stays 'testing' (genuinely mixed —
    a real split verdict, not forced to a side just because there's
    enough data)."""
    from qualume.db.connection import get_connection
    import json
    from datetime import datetime, timezone

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sample_size, state_history FROM insights_owned WHERE id = %s",
                (owned_insight_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"insights_owned row {owned_insight_id} not found")
            sample_size, state_history = row

            state_history = list(state_history or [])
            state_history.append({
                "confirmed": confirmed,
                "content_item_id": content_item_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            new_sample_size = sample_size + 1
            confirmation_rate = sum(1 for e in state_history if e["confirmed"]) / new_sample_size

            if new_sample_size < TIER_PROMOTION_THRESHOLD:
                tier = "testing"
            elif confirmation_rate >= CONFIRMATION_RATE_FOR_ACCEPT:
                tier = "accepted"
            elif confirmation_rate <= CONFIRMATION_RATE_FOR_REJECT:
                tier = "rejected"
            else:
                tier = "testing"

            cur.execute(
                """UPDATE insights_owned
                   SET sample_size = %s, state_history = %s, tier = %s
                   WHERE id = %s""",
                (new_sample_size, json.dumps(state_history), tier, owned_insight_id),
            )
        conn.commit()
    finally:
        conn.close()
