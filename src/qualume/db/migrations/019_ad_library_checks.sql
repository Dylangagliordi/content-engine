-- ad_library_checks: append-only longevity tracking for Meta Ad Library
-- videos, added 2026-11-02 alongside the meta_ads reader.
--
-- WHY A SEPARATE TABLE instead of reusing external_performance_snapshots:
-- that table's columns (views, likes, comments, shares, saves) have no
-- honest value for an ad — Meta does not publish any of those numbers
-- for an ordinary US commercial ad (confirmed live 2026-11-02 against a
-- real ad, see migration 018's comment). Storing NULLs in every column
-- every snapshot would be pointless noise, and storing a fabricated
-- number would violate this project's honesty discipline (CLAUDE.md
-- principles #2, #7). So this is a distinct, append-only table with the
-- fields that ARE honestly available and meaningful for an ad:
--
-- still_active: was this Library ID still showing as an active ad on
-- this check? An advertiser pays to keep an ad running — if it weren't
-- converting, standard media-buying practice is to kill it fast. So
-- "still active after N days" is a real, if indirect, winner signal —
-- the honest substitute for the missing spend/impression number, built
-- the same way TikTok's view-count time series is: by re-polling over
-- real time, not estimated once.
--
-- ads_using_creative: Meta's own "N ads use this creative" count at
-- this check — can grow over time as an advertiser scales a winning
-- creative into more ad variants, a real (if rough) scale signal.
--
-- Same append-only-never-UPDATE discipline as external_performance_
-- snapshots (CLAUDE.md principle #4): every check is a new row, so the
-- full history of "how long did this stay active" is reconstructable,
-- not just a single current-state flag.
CREATE TABLE ad_library_checks (
    id BIGSERIAL PRIMARY KEY,
    external_content_item_id UUID NOT NULL REFERENCES external_content_items(id),
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    still_active BOOLEAN NOT NULL,
    ads_using_creative INT
);
