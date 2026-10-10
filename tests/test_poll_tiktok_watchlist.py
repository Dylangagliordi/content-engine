from unittest.mock import patch

from qualume.poll_tiktok_watchlist import run_poll_cycle


@patch("qualume.poll_tiktok_watchlist.list_watched_accounts")
def test_run_poll_cycle_handles_empty_watchlist(mock_list):
    mock_list.return_value = []
    result = run_poll_cycle()
    assert result == {
        "accounts_polled": 0,
        "accounts_failed": 0,
        "new_videos": 0,
        "skipped_duplicates": 0,
    }


@patch("qualume.poll_tiktok_watchlist.ingest_videos_for_account")
@patch("qualume.poll_tiktok_watchlist.list_watched_accounts")
@patch("qualume.poll_tiktok_watchlist.time.sleep")  # don't actually sleep in tests
def test_run_poll_cycle_isolates_one_account_failure_from_others(mock_sleep, mock_list, mock_ingest):
    mock_list.return_value = [
        {"id": "a1", "handle": "good_account_1"},
        {"id": "a2", "handle": "broken_account"},
        {"id": "a3", "handle": "good_account_2"},
    ]

    def side_effect(account_id, handle, max_videos=30):
        if handle == "broken_account":
            raise RuntimeError("simulated scrape failure")
        return {"new_videos": 1, "skipped_duplicates": 0}

    mock_ingest.side_effect = side_effect

    result = run_poll_cycle()

    assert result["accounts_polled"] == 2
    assert result["accounts_failed"] == 1
    assert result["new_videos"] == 2
