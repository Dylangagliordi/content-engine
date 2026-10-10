from qualume.db.connection import get_connection


def load_active_playbook(workflow_type: str) -> dict:
    """Load the most recent playbook version for a workflow_type.

    Returns {"version": str, "rubric": dict}. Raises ValueError if no
    playbook row exists for this workflow_type — fail loud rather than
    silently generating content with no rubric at all.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT version, rubric_json FROM playbooks
                   WHERE workflow_type = %s
                   ORDER BY generated_at DESC
                   LIMIT 1""",
                (workflow_type,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(
            f"No playbook found for workflow_type={workflow_type!r}. "
            "Seed one before generating content (see v2 plan §2.2 / Phase 0 Task 0.10)."
        )

    version, rubric_json = row
    return {"version": version, "rubric": rubric_json}
