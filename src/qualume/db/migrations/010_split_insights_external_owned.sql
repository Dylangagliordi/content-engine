-- Split the single `insights` table into insights_external/insights_owned,
-- per plans/learn-engine-idea-sourcing-architecture.md ("Two insights
-- tables, not one"). Mirrors the owned-vs-external trust separation
-- already enforced elsewhere (content_items vs external_content_items,
-- creative_dna vs external_creative_dna) — insights_external is "this
-- worked for other accounts" (lower trust ceiling even at tier:accepted),
-- insights_owned is "this works for us" (the one that actually drives
-- generation decisions, can diverge from the external version of the
-- same claim).
--
-- Also adds applies_to_segment (JSONB — e.g. {"niche": "wellness",
-- "platform": "tiktok", "format": "talking_head"}), missing from the
-- original single table per the plan.
--
-- The original `insights` table has 0 rows (confirmed live) and nothing
-- references it via foreign key, so this drops and recreates rather than
-- a data-preserving ALTER — there is no data to lose.

DROP TABLE IF EXISTS insights;

CREATE TABLE insights_external (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement                 TEXT NOT NULL,
    tier                      TEXT NOT NULL DEFAULT 'proposed',
    -- proposed -> testing -> accepted -> retired, plus rejected
    -- (rejected rows stay stored permanently with the critic's reason in
    -- state_history, so the system never re-discovers/re-tests the same
    -- spurious candidate on a future mining pass)
    effect_size               NUMERIC,
    sample_size               INTEGER,
    p_value                   NUMERIC,
    applies_to_segment        JSONB DEFAULT '{}',
    applies_to_workflow_type  TEXT NOT NULL,
    supporting_data_refs      JSONB DEFAULT '{}',
    source                    TEXT NOT NULL DEFAULT 'internal_mining',
    state_history             JSONB NOT NULL DEFAULT '[]',
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE insights_owned (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement                 TEXT NOT NULL,
    tier                      TEXT NOT NULL DEFAULT 'proposed',
    effect_size               NUMERIC,
    sample_size               INTEGER,
    p_value                   NUMERIC,
    applies_to_segment        JSONB DEFAULT '{}',
    applies_to_workflow_type  TEXT NOT NULL,
    supporting_data_refs      JSONB DEFAULT '{}',
    source                    TEXT NOT NULL DEFAULT 'internal_mining',
    state_history             JSONB NOT NULL DEFAULT '[]',
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- owned-only: an insights_owned row that originated as a test of a
    -- specific insights_external claim (per the plan's "use experiments,
    -- don't invent a new mechanism" — this is just the traceability link
    -- back to which external insight inspired this one, if any; a
    -- homegrown owned-only pattern with no external origin leaves this NULL)
    derived_from_external_insight_id UUID REFERENCES insights_external(id)
);
