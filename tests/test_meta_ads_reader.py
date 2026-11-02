"""Tests for the Meta Ad Library reader — written FIRST per this
project's TDD discipline.

Real gap this fixes vs. the free, public facebook-ads-library-mcp
scraper this is adapted from: that scraper serializes the rendered page
to MARKDOWN via crawl4ai, and markdown has no syntax for <video src=...>
(only images), so every ad's actual video file gets silently dropped —
you only get a static thumbnail. Confirmed live 2026-11-02 that real
ad_snapshot video ads DO have a real <video> element with a src/
currentSrc in the rendered DOM; this module pulls that directly via a
small JS snippet run in-page (same approach Apify's paid actor uses to
get its videoHdUrl), zipped back to parsed ad cards in DOM order.

Also real, confirmed live 2026-11-02 against an actual US commercial ad
(Depop): there is NO impression/reach/spend number published for an
ordinary US commercial ad anywhere in the Ad Library UI — only Library
ID, started-running date, platforms, and ads_using_creative ("N ads use
this creative"). So this module does NOT fabricate an impressions
field — see migration 018/019 comments for the full reasoning. Run date
+ ads_using_creative + a longevity re-check over time (ad_library_
checks table) are the honest substitutes.
"""
from qualume.ingest.meta_ads_reader import (
    parse_ad_library_markdown,
    attach_video_urls,
    parse_ad_entry,
)


SAMPLE_MARKDOWN = """
Some header junk before the first card

Library ID: 1111111111111111
Started running on Jan 5, 2026
Platforms Facebook Instagram
[Greenbabe](https://www.facebook.com/greenbabewellness/) 
**Sponsored**
Finally sleep through the night again. Our postpartum tincture helps new moms find calm.
[![](https://scontent.fbcdn.net/thumb1.jpg) Finally sleep through the night again. Shop now ](https://l.facebook.com/l.php?u=https%3A%2F%2Fhellogreenbabe.com%2Fproducts%2Finner-peace&h=abc)
Active

Library ID: 2222222222222222
Started running on Feb 1, 2026
**2 ads** use this creative
Platforms Facebook
[Greenbabe](https://www.facebook.com/greenbabewellness/) 
**Sponsored**
Stop the postpartum rage cycle naturally.
[![](https://scontent.fbcdn.net/thumb2.jpg) Stop the postpartum rage cycle naturally. Learn more ](https://l.facebook.com/l.php?u=https%3A%2F%2Fhellogreenbabe.com%2Fpages%2Fabout&h=def)
Active
"""


def test_parse_ad_library_markdown_extracts_library_ids_in_order():
    ads = parse_ad_library_markdown(SAMPLE_MARKDOWN)
    assert [a["library_id"] for a in ads] == ["1111111111111111", "2222222222222222"]


def test_parse_ad_library_markdown_extracts_started_running_date():
    ads = parse_ad_library_markdown(SAMPLE_MARKDOWN)
    assert ads[0]["started_running"] == "Jan 5, 2026"
    assert ads[1]["started_running"] == "Feb 1, 2026"


def test_parse_ad_library_markdown_extracts_ads_using_creative_defaulting_to_one():
    ads = parse_ad_library_markdown(SAMPLE_MARKDOWN)
    assert ads[0]["ads_using_creative"] == 1  # no "N ads use this creative" line present
    assert ads[1]["ads_using_creative"] == 2


def test_parse_ad_library_markdown_extracts_advertiser():
    ads = parse_ad_library_markdown(SAMPLE_MARKDOWN)
    assert ads[0]["advertiser"] == "Greenbabe"
    assert ads[0]["advertiser_handle"] == "greenbabewellness"


def test_parse_ad_library_markdown_decodes_landing_url():
    ads = parse_ad_library_markdown(SAMPLE_MARKDOWN)
    assert ads[0]["landing_url"] == "https://hellogreenbabe.com/products/inner-peace"


def test_parse_ad_library_markdown_dedupes_repeated_library_ids():
    """Infinite-scroll re-rendering can duplicate a card already seen —
    same real shape the free scraper handles, confirmed necessary there."""
    doubled = SAMPLE_MARKDOWN + "\nLibrary ID: 1111111111111111\nStarted running on Jan 5, 2026\nActive\n"
    ads = parse_ad_library_markdown(doubled)
    assert len(ads) == 2  # still 2, not 3


def test_parse_ad_library_markdown_handles_no_cards_gracefully():
    assert parse_ad_library_markdown("nothing here") == []


# --- Real fix: attaching video URLs that markdown parsing drops --------


def test_attach_video_urls_zips_video_src_to_ad_cards_in_dom_order():
    ads = [
        {"library_id": "1111111111111111"},
        {"library_id": "2222222222222222"},
    ]
    video_srcs = [
        "https://video.fbcdn.net/o1/v/t2/ad1_hd.mp4",
        "https://video.fbcdn.net/o1/v/t2/ad2_hd.mp4",
    ]
    result = attach_video_urls(ads, video_srcs)
    assert result[0]["video_url"] == "https://video.fbcdn.net/o1/v/t2/ad1_hd.mp4"
    assert result[1]["video_url"] == "https://video.fbcdn.net/o1/v/t2/ad2_hd.mp4"


def test_attach_video_urls_leaves_video_url_none_for_image_only_ads():
    """An image-only ad (no <video> element in its card) must not get a
    video URL from a LATER card's video — real risk of a naive flat zip
    if some cards have no video and the counts don't line up 1:1."""
    ads = [
        {"library_id": "1111111111111111", "has_video_placeholder": True},
        {"library_id": "2222222222222222", "has_video_placeholder": False},
        {"library_id": "3333333333333333", "has_video_placeholder": True},
    ]
    video_srcs = ["https://video.fbcdn.net/ad1.mp4", "https://video.fbcdn.net/ad3.mp4"]
    result = attach_video_urls(ads, video_srcs)
    assert result[0]["video_url"] == "https://video.fbcdn.net/ad1.mp4"
    assert result[1]["video_url"] is None
    assert result[2]["video_url"] == "https://video.fbcdn.net/ad3.mp4"


def test_attach_video_urls_handles_fewer_videos_than_placeholders_gracefully():
    """Real failure mode: page didn't fully hydrate, fewer <video> src
    values came back than expected — must not crash or misassign."""
    ads = [
        {"library_id": "1", "has_video_placeholder": True},
        {"library_id": "2", "has_video_placeholder": True},
    ]
    result = attach_video_urls(ads, ["https://video.fbcdn.net/only_one.mp4"])
    assert result[0]["video_url"] == "https://video.fbcdn.net/only_one.mp4"
    assert result[1]["video_url"] is None


# --- Normalizing a parsed ad card into our external_content_items schema


def test_parse_ad_entry_maps_fields_correctly():
    ad = {
        "library_id": "1111111111111111",
        "started_running": "Jan 5, 2026",
        "ads_using_creative": 2,
        "advertiser": "Greenbabe",
        "advertiser_handle": "greenbabewellness",
        "landing_url": "https://hellogreenbabe.com/products/inner-peace",
        "body": "Finally sleep through the night again.",
        "video_url": "https://video.fbcdn.net/ad1_hd.mp4",
    }
    parsed = parse_ad_entry(ad, country="US")
    assert parsed["platform_video_id"] == "1111111111111111"
    assert parsed["url"] == "https://www.facebook.com/ads/library/?id=1111111111111111"
    assert parsed["caption"] == "Finally sleep through the night again."
    assert parsed["ad_started_running_at"] is not None
    assert parsed["ad_started_running_at"].year == 2026
    assert parsed["ad_started_running_at"].month == 1
    assert parsed["ad_started_running_at"].day == 5
    assert parsed["ads_using_creative"] == 2
    assert parsed["video_url"] == "https://video.fbcdn.net/ad1_hd.mp4"
    assert parsed["content_type"] == "selling"  # every Ad Library entry is selling-by-definition


def test_parse_ad_entry_handles_missing_started_running_gracefully():
    ad = {"library_id": "999", "advertiser": "X"}
    parsed = parse_ad_entry(ad, country="US")
    assert parsed["ad_started_running_at"] is None
    assert parsed["platform_video_id"] == "999"


# --- Downloading the real video file ------------------------------------
# requests is always mocked here: a real download would hit Meta's live
# CDN from the test suite, same reasoning as tiktok_reader's own
# download tests mocking yt_dlp.


def test_download_ad_video_writes_file_and_returns_path(tmp_path):
    dest = tmp_path / "raw_videos"
    from unittest.mock import patch, MagicMock

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.iter_content.return_value = [b"fake", b" mp4 ", b"bytes"]
    fake_resp.raise_for_status.return_value = None

    with patch("qualume.ingest.meta_ads_reader.requests.get", return_value=fake_resp) as mock_get:
        from qualume.ingest.meta_ads_reader import download_ad_video
        path = download_ad_video(
            "https://video.fbcdn.net/ad1_hd.mp4?token=abc", "1111111111111111", dest_dir=str(dest)
        )

    assert mock_get.called
    assert path == str(dest / "1111111111111111.mp4")
    with open(path, "rb") as f:
        assert f.read() == b"fake mp4 bytes"


def test_download_ad_video_is_idempotent_when_file_already_exists(tmp_path):
    from unittest.mock import patch, MagicMock
    dest = tmp_path / "raw_videos"
    dest.mkdir()
    existing = dest / "222.mp4"
    existing.write_bytes(b"already downloaded")

    with patch("qualume.ingest.meta_ads_reader.requests.get") as mock_get:
        from qualume.ingest.meta_ads_reader import download_ad_video
        path = download_ad_video("https://video.fbcdn.net/ad2.mp4", "222", dest_dir=str(dest))

    mock_get.assert_not_called()  # no network hit
    assert path == str(existing)
    assert existing.read_bytes() == b"already downloaded"
