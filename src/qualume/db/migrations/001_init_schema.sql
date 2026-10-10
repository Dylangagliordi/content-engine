CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type TEXT NOT NULL,            -- e.g. 'ig_static'
    platform TEXT NOT NULL,                 -- e.g. 'instagram'
    status TEXT NOT NULL DEFAULT 'draft',   -- draft | published | rejected
    batch_id UUID,
    playbook_version_used TEXT,
    prompt_versions_used JSONB,             -- {"ideation": "v1", "write": "v1", "design": "v1"}
    lane TEXT NOT NULL DEFAULT 'owned',     -- owned | external
    asset_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ
);

CREATE TABLE creative_dna (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    hook_type TEXT,
    hook_text TEXT,
    visual_hook_description TEXT,
    format_tag TEXT,
    script_structure_tags TEXT[],
    cta_type TEXT,
    caption_style TEXT,
    social_proof_used BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE performance_snapshots (
    id BIGSERIAL PRIMARY KEY,
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    views INT, likes INT, saves INT, comments INT, shares INT, follows_gained INT,
    entry_method TEXT NOT NULL DEFAULT 'manual'  -- manual | api
);

CREATE TABLE ad_performance (
    id BIGSERIAL PRIMARY KEY,
    content_item_id UUID NOT NULL REFERENCES content_items(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    spend NUMERIC, cpm NUMERIC, cpc NUMERIC, roas NUMERIC,
    attribution_source TEXT, conversions INT
);

CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    hypothesis TEXT,
    explore_exploit_ratio NUMERIC,
    workflow_type TEXT NOT NULL,
    outcome_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'proposed',  -- proposed | testing | accepted | retired
    effect_size NUMERIC,
    sample_size INT,
    p_value NUMERIC,
    supporting_data_refs JSONB,
    applies_to_workflow_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'internal_mining',  -- internal_mining | external_field_notes
    state_history JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type TEXT NOT NULL,
    version TEXT NOT NULL,
    rubric_json JSONB NOT NULL,
    source_insight_ids UUID[],
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(workflow_type, version)
);

CREATE TABLE eval_runs (
    id BIGSERIAL PRIMARY KEY,
    stage TEXT NOT NULL,            -- annotation | write | design
    golden_set_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    score NUMERIC,
    passed BOOLEAN,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
