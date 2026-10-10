-- External/competitor content ingestion — structurally separate from owned
-- content_items per v2 plan §2.1 ("two lanes, kept structurally separate,
-- different trust levels, different schemas-of-origin"). Content pulled from
-- accounts/hashtags you don't own and didn't write — used for pattern
-- discovery only, never merged into the owned-content learning signal.

CREATE TABLE watched_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,           -- 'tiktok' | 'instagram' (future)
    handle TEXT NOT NULL,             -- e.g. 'scout2015' (no @ prefix)
    active BOOLEAN NOT NULL DEFAULT true,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT,
    UNIQUE(platform, handle)
);

CREATE TABLE external_content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    watched_account_id UUID REFERENCES watched_accounts(id),
    platform TEXT NOT NULL,
    platform_video_id TEXT NOT NULL,  -- TikTok's own video ID — the dedup key
    url TEXT NOT NULL,
    caption TEXT,
    duration_seconds INT,
    audio_track TEXT,
    audio_artist TEXT,
    posted_at TIMESTAMPTZ,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_metadata JSONB,               -- full yt-dlp info dict, for fields we haven't modeled yet
    UNIQUE(platform, platform_video_id)
);

CREATE TABLE external_performance_snapshots (
    id BIGSERIAL PRIMARY KEY,
    external_content_item_id UUID NOT NULL REFERENCES external_content_items(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    views INT, likes INT, comments INT, shares INT, saves INT
);
