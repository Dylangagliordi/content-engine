"""Re-runnable playbook seeding: every playbook version lives here, in code.

    uv run python -m qualume.db.seed_playbooks

Each (workflow_type, version) is immutable once written: re-running is a no-op
for versions already in the DB, and a version whose rubric here differs from
the DB row fails loud rather than silently rewriting history. A rubric change
is always a NEW version row (load_active_playbook picks the newest), so every
content_items.playbook_version_used stays traceable to the exact rubric that
produced it.
"""

import json

from qualume.db.connection import get_connection

_IG_STATIC_V0 = {
    "tier": "proposed",
    "source": "external_field_notes",
    "hooks": ["state a concrete problem in plain language, avoid vague claims"],
    "format": ["one clear headline + one supporting line, avoid clutter"],
    "social_proof": ["use sparingly, one credible data point beats three vague ones"],
}

PLAYBOOKS = [
    {
        "workflow_type": "ig_static",
        "version": "v0",
        "rubric": _IG_STATIC_V0,
        "approved_by": "dylan",
    },
    {
        # Anti-fabrication fix (docs/retry-loop-verdict-2026-10-10.md): the judge
        # was passing invented stats, features, and testimonials. Content-safety
        # rules, not a performance hypothesis — v0 rules carry over unchanged.
        "workflow_type": "ig_static",
        "version": "v1",
        "rubric": {
            **_IG_STATIC_V0,
            "claims": [
                "no statistics, percentages, test results, user/customer counts, or specific "
                "product features that are not explicitly present in the brief text; this "
                "overrides social_proof: a data point that is not in the brief must not be used",
                "personal testimonials/anecdotes (first-person stories, customer quotes, "
                "before/after experiences) must not be presented as fact unless the brief "
                "explicitly provides them",
            ],
        },
        "approved_by": None,  # pending human review
    },
]


def seed_playbook(conn, workflow_type: str, version: str, rubric: dict,
                  approved_by: str | None = None) -> str:
    """Insert one playbook version if absent. Returns "inserted" or "exists".

    Raises ValueError if the version already exists with a different rubric.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rubric_json FROM playbooks WHERE workflow_type = %s AND version = %s",
            (workflow_type, version),
        )
        row = cur.fetchone()
        if row is not None:
            if row[0] != rubric:
                raise ValueError(
                    f"Playbook {workflow_type}/{version} already exists with a different rubric. "
                    "Versions are immutable — add a new version instead."
                )
            return "exists"
        # clock_timestamp(), not the now() default: now() is fixed per transaction,
        # so versions seeded together would tie on generated_at and make the
        # "newest version is active" ordering in load_active_playbook ambiguous.
        cur.execute(
            """INSERT INTO playbooks (workflow_type, version, rubric_json, approved_by,
                                      approved_at, generated_at)
               VALUES (%s, %s, %s, %s, CASE WHEN %s IS NULL THEN NULL ELSE clock_timestamp() END,
                       clock_timestamp())""",
            (workflow_type, version, json.dumps(rubric), approved_by, approved_by),
        )
    conn.commit()
    return "inserted"


def main():
    conn = get_connection()
    try:
        for p in PLAYBOOKS:
            status = seed_playbook(conn, p["workflow_type"], p["version"], p["rubric"],
                                   p["approved_by"])
            print(f"{p['workflow_type']}/{p['version']}: {status}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
