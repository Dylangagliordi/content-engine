"""Tests for the orchestration HTTP server — written FIRST per this
project's TDD discipline.

Real production incident this session (2026-10-31): server.py used a
single-threaded http.server.HTTPServer. While one POST /run-ingestion-
cycle call was in flight (which, on a backlog of thousands of videos,
realistically runs for hours to days), the server could not answer
ANYTHING else — not GET /health, not n8n's next scheduled trigger every
6 hours. A curl client that gave up waiting (hit its own --max-time) had
no way to know whether the server-side work was still happening,
deadlocked, or crashed — it just silently disconnected from an orphaned
request. This test suite proves make_server() now uses a threading
server so /health stays answerable DURING a long-running cycle.
"""
import threading
import time
import urllib.request
from http.server import HTTPServer
from unittest.mock import patch

import pytest

from qualume.server import make_server


def test_make_server_returns_a_threading_server_not_single_threaded():
    """Real regression test for the incident: HTTPServer (non-threading)
    handles one request at a time, full stop — ThreadingHTTPServer
    handles each request in its own thread. Checking the actual class,
    not just behavior, because the behavioral symptom (a blocked /health
    during a slow request) is slow and flaky to test directly."""
    server = make_server(("127.0.0.1", 0))  # port 0 = OS picks a free port
    try:
        assert not isinstance(type(server), type(HTTPServer)) or True  # placeholder, real assert below
        from http.server import ThreadingHTTPServer
        assert isinstance(server, ThreadingHTTPServer)
    finally:
        server.server_close()


def test_health_check_responds_while_a_slow_cycle_is_in_flight():
    """Real end-to-end proof, not just a type check: start a real server
    on a real local port, have a slow /run-ingestion-cycle request
    occupy it, and confirm a concurrent GET /health still gets answered
    promptly instead of queuing behind the slow request."""
    with patch("qualume.server.run_ingestion_cycle") as mock_cycle:
        def slow_cycle():
            time.sleep(1.5)
            return {"poll": {}, "download": {}}
        mock_cycle.side_effect = slow_cycle

        server = make_server(("127.0.0.1", 0))
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Fire the slow POST in its own thread, don't wait for it here.
            def fire_slow_post():
                try:
                    urllib.request.urlopen(
                        f"http://{host}:{port}/run-ingestion-cycle",
                        data=b"", timeout=5,
                    )
                except Exception:
                    pass
            post_thread = threading.Thread(target=fire_slow_post)
            post_thread.start()

            time.sleep(0.3)  # let the slow POST actually start being handled

            # The real assertion: /health must respond fast, not hang
            # behind the in-flight slow POST.
            start = time.monotonic()
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
                elapsed = time.monotonic() - start
                assert resp.status == 200

            assert elapsed < 1.0  # well under the 1.5s the slow cycle takes

            post_thread.join(timeout=5)
        finally:
            server.shutdown()
            server.server_close()


def test_concurrent_cycle_request_is_rejected_not_run_in_parallel():
    """Real second risk ThreadingHTTPServer introduces on its own (found
    while fixing the /health-blocking bug): if n8n's 6-hour schedule
    fires a NEW /run-ingestion-cycle while a previous multi-hour cycle
    is still running, naive threading would run them concurrently —
    doubling up ANNOTATION_CONCURRENCY/CLASSIFICATION_CONCURRENCY worker
    pools and recreating the exact memory-exhaustion incident that
    forced those constants down from 8 to 4 in the first place. A
    second POST while one is already in flight must be rejected (409),
    not silently run in parallel."""
    import threading
    import time
    import urllib.error
    import urllib.request
    from unittest.mock import patch as mock_patch

    from qualume.server import make_server

    with mock_patch("qualume.server.run_ingestion_cycle") as mock_cycle:
        mock_cycle.side_effect = lambda: time.sleep(1.0) or {}

        server = make_server(("127.0.0.1", 0))
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            first = threading.Thread(
                target=lambda: urllib.request.urlopen(
                    f"http://{host}:{port}/run-ingestion-cycle", data=b"", timeout=5)
            )
            first.start()
            time.sleep(0.3)  # let the first request actually start running

            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(
                    f"http://{host}:{port}/run-ingestion-cycle", data=b"", timeout=3)
            assert exc_info.value.code == 409

            first.join(timeout=5)
            assert mock_cycle.call_count == 1  # never ran twice concurrently
        finally:
            server.shutdown()
            server.server_close()
