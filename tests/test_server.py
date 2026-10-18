"""Tests for the local HTTP server wrapping orchestrate.run_ingestion_cycle
— the real n8n integration surface. n8n's Schedule Trigger -> HTTP
Request node calls POST /run-ingestion-cycle on this server (reachable
from inside the n8n Docker container via host.docker.internal, confirmed
live against the real qualume-n8n container).

stdlib http.server only, deliberately — this project has no web
framework dependency anywhere else (FastAPI/Flask), and a single POST
endpoint that calls one existing function doesn't justify adding one.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from qualume.server import make_server


@pytest.fixture
def running_server():
    with patch("qualume.server.run_ingestion_cycle") as mock_cycle:
        mock_cycle.return_value = {
            "poll": {"accounts_polled": 1},
            "download": {"downloaded": 0},
            "annotation": {"succeeded": 0, "failed": 0, "errors": []},
            "classification": {"succeeded": 0, "failed": 0, "errors": []},
        }
        server = make_server(("127.0.0.1", 0))  # port 0 = OS picks a free port
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)
        yield f"http://127.0.0.1:{port}", mock_cycle
        server.shutdown()


def test_post_run_ingestion_cycle_returns_the_real_result(running_server):
    base_url, mock_cycle = running_server
    req = urllib.request.Request(f"{base_url}/run-ingestion-cycle", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    assert body == mock_cycle.return_value
    mock_cycle.assert_called_once()


def test_get_health_returns_200_without_running_the_cycle(running_server):
    """A lightweight health check n8n (or anything else) can poll without
    triggering a real, expensive ingestion run."""
    base_url, mock_cycle = running_server
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
        assert resp.status == 200
    mock_cycle.assert_not_called()


def test_get_run_ingestion_cycle_is_rejected_not_silently_run(running_server):
    """Only POST should trigger a real (expensive, side-effecting) cycle —
    a GET must not accidentally trigger it, e.g. from a browser preview
    or a misconfigured monitoring tool."""
    base_url, mock_cycle = running_server
    req = urllib.request.Request(f"{base_url}/run-ingestion-cycle", method="GET")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 405
    mock_cycle.assert_not_called()


def test_unknown_path_returns_404(running_server):
    base_url, mock_cycle = running_server
    req = urllib.request.Request(f"{base_url}/not-a-real-path", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 404


def test_cycle_exception_returns_500_not_a_silent_empty_response(running_server):
    """If run_ingestion_cycle itself raises unexpectedly (it shouldn't,
    since every stage inside it already catches its own exceptions —
    but a genuinely unexpected bug must surface as a real error to n8n,
    not a silent empty/broken response that looks like success."""
    base_url, mock_cycle = running_server
    mock_cycle.side_effect = RuntimeError("totally unexpected bug")
    req = urllib.request.Request(f"{base_url}/run-ingestion-cycle", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 500
