-- external_content_items: Meta Ad Library specific fields, added
-- 2026-11-02 alongside the meta_ads platform reader.
--
-- Verified LIVE on 2026-11-02 against a real US commercial ad (Depop,
-- Library ID 1647772813204375) before writing this: the Ad Library
-- detail panel shows NO impression/reach number for an ordinary US
-- commercial ad — only Library ID, started-running date, platforms,
-- and creative/copy. This matches Meta's own docs (impression ranges
-- are only published for EU/UK-delivered ads under DSA, or for
-- political/issue ads) but was confirmed firsthand, not assumed. So
-- there is deliberately NO impressions column here — storing a
-- fabricated/estimated number would violate this project's honesty
-- discipline (CLAUDE.md principles #2, #7).
--
-- ads_using_creative: the real scale proxy Meta DOES publish — "N ads
-- use this creative" on the card, i.e. how many separate ad entries
-- share the same copy/creative. Higher count is a real (if rough)
-- signal that an advertiser is running/scaling that specific creative,
-- which is the honest substitute for a missing spend/impression number.
--
-- ad_started_running_at: the ad's "Started running on <date>" field —
-- reliably available for every ad regardless of region/type, unlike
-- impressions.
ALTER TABLE external_content_items
    ADD COLUMN ad_started_running_at DATE,
    ADD COLUMN ads_using_creative INT;
