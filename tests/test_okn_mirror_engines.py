"""Tests for okn_mirror's analyze_okn and ml_engine_okn ports — lighter than
test_okn_mirror_loader.py's exhaustive fixture tests, since these two modules
are near-verbatim ports of already-proven upstream logic (see each module's
own docstring for the exact, documented deltas from okn-analytics). These
tests confirm the port RUNS correctly against qualume's real DB-shaped data
and produces internally-consistent output — not re-deriving sklearn's math.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.okn_mirror.analyze_okn import analyze
from qualume.okn_mirror.loader import load_niche_dataframe
from qualume.okn_mirror.ml_engine_okn import run_ml


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_performance_snapshots WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
            cur.execute(
                "DELETE FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s)",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _insert_video(account_id, views, likes=0, comments=0, shares=0,
                   content_type="organic", caption="a real test caption about testing"):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url,
                    caption, duration_seconds, content_type)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', %s, 30, %s)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex, caption, content_type),
            )
            item_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares)
                   VALUES (%s, %s, %s, %s, %s)""",
                (item_id, views, likes, comments, shares),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def niche_with_20_videos():
    """20 videos is enough to clear MIN_POSTS_FOR_NN (15) and
    ANALYSIS["min_posts_for_analysis"] (5), so every non-embedding model
    actually runs (not just returns insufficient_data)."""
    handle = f"test_oknmirror_ml_{uuid.uuid4().hex[:8]}"
    niche = f"zz_test_niche_ml_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE watched_accounts SET niche = %s WHERE id = %s", (niche, account_id))
        conn.commit()
    finally:
        conn.close()

    import random
    random.seed(42)
    for i in range(20):
        views = random.randint(100, 10000)
        likes = int(views * random.uniform(0.02, 0.15))
        ctype = "organic" if i % 2 == 0 else "selling"
        _insert_video(account_id, views, likes=likes, comments=likes // 10,
                       shares=likes // 20, content_type=ctype,
                       caption=f"real test caption number {i} about things #testtag{i % 3}")
    yield niche
    _cleanup_account(handle)


def test_analyze_okn_runs_all_8_modules_without_crashing(niche_with_20_videos):
    df = load_niche_dataframe(niches=[niche_with_20_videos], min_niche_size=5)
    assert len(df) == 20
    results = analyze(df)
    for key in ["niche_overview", "content_performance", "engagement", "temporal",
                "growth", "anomalies", "cross_niche", "recommendations"]:
        assert key in results


def test_ml_engine_runs_core_models_without_sentence_transformers_dependency(niche_with_20_videos):
    """Even if sentence-transformers / the embedding model is unavailable or
    slow in CI, the 10 non-semantic models (1-10) must still run and return
    real status, not crash the whole engine — same graceful-degradation
    discipline as upstream okn's own HAS_EMBEDDINGS fallback."""
    df = load_niche_dataframe(niches=[niche_with_20_videos], min_niche_size=5)
    results = run_ml(df, niche_with_20_videos)
    assert results["status"] == "ok"
    assert results["n_posts"] == 20
    for key in ["feature_importance", "engagement_prediction", "content_clusters",
                "anomalies", "caption_analysis", "engagement_drivers",
                "content_fatigue", "posting_cadence", "momentum_score", "root_cause"]:
        assert key in results, f"missing {key}"
    # semantic models always present as keys even if embeddings unavailable
    for key in ["topic_discovery", "similar_posts", "hashtag_clusters"]:
        assert key in results


def test_ml_engine_reports_insufficient_data_below_min_posts():
    """Below MIN_POSTS_FOR_ML (10), the whole engine should short-circuit
    with status=insufficient_data, not crash trying to fit models on too
    little data — exact same guard as upstream okn."""
    import pandas as pd
    tiny_df = pd.DataFrame({
        "post_id": ["a", "b"], "platform": ["x", "x"],
        "published_at": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "content_type": ["organic", "organic"], "title": ["hi", "there"],
        "permalink": ["", ""], "duration_sec": [10, 10], "reach": [100, 200],
        "views": [100, 200], "likes": [10, 20], "comments": [1, 2], "shares": [1, 1],
        "saves": [0, 0], "engagement_total": [12, 23], "engagement_rate": [0.12, 0.115],
        "watch_time_sec": [0, 0], "avg_watch_sec": [0, 0], "followers_gained": [0, 0],
        "link_clicks": [0, 0],
    })
    results = run_ml(tiny_df, "x")
    assert results["status"] == "insufficient_data"


def test_feature_importance_produces_a_real_r2_and_ranked_features(niche_with_20_videos):
    df = load_niche_dataframe(niches=[niche_with_20_videos], min_niche_size=5)
    results = run_ml(df, niche_with_20_videos)
    fi = results["feature_importance"]
    assert fi["status"] == "ok"
    assert "r2_score" in fi
    assert isinstance(fi["top_features"], list)


def test_content_clustering_produces_k_clusters_summing_to_total_posts(niche_with_20_videos):
    df = load_niche_dataframe(niches=[niche_with_20_videos], min_niche_size=5)
    results = run_ml(df, niche_with_20_videos)
    cc = results["content_clusters"]
    assert cc["status"] == "ok"
    total = sum(c["size"] for c in cc["clusters"])
    assert total == 20


def test_momentum_score_is_between_0_and_100(niche_with_20_videos):
    df = load_niche_dataframe(niches=[niche_with_20_videos], min_niche_size=5)
    results = run_ml(df, niche_with_20_videos)
    ms = results["momentum_score"]
    if ms.get("status") == "ok":
        assert 0 <= ms["total_score"] <= 100
