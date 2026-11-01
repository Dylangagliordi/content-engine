"""Minimal local HTTP server exposing orchestrate.run_ingestion_cycle to
n8n. Run this on the Mac/N100 (wherever Postgres/the pipeline actually
runs); n8n's Schedule Trigger -> HTTP Request node calls
POST http://host.docker.internal:<port>/run-ingestion-cycle on a cadence.
host.docker.internal is confirmed reachable from inside the real
qualume-n8n Docker container (OrbStack resolves it automatically; no
extra Docker networking config needed).

stdlib http.server only — this project has no web framework dependency
anywhere else, and one POST endpoint calling one existing function
doesn't justify adding FastAPI/Flask. Deliberately NOT exposed outside
localhost by default (see DEFAULT_HOST) — this has no auth and triggers
real side-effecting work (Gemini/Claude API calls, DB writes); binding
it beyond localhost is the operator's explicit choice, not the default.

ThreadingHTTPServer, not plain HTTPServer (real production incident,
2026-10-31): with a single-threaded HTTPServer, one in-flight POST
/run-ingestion-cycle call (which realistically runs for HOURS on a large
backlog) blocked GET /health entirely — no way to tell from outside
whether the server was alive, deadlocked, or crashed while a cycle ran.
ThreadingHTTPServer handles each request in its own thread, so /health
stays answerable throughout a long cycle. run_ingestion_cycle's own DB
connections are already per-call, short-lived (get_connection/conn.close()
per query inside each sub-function, never held across the whole cycle),
so concurrent requests don't share stale connections — no new
thread-safety concern introduced by this change.

_cycle_lock (added same session, same incident): ThreadingHTTPServer on
its own introduces a NEW risk — if n8n's 6-hour schedule fires a fresh
/run-ingestion-cycle while a previous multi-hour cycle is still running,
naive threading would run BOTH concurrently, doubling up orchestrate.py's
own worker pools (ANNOTATION_CONCURRENCY etc.) and recreating the exact
memory-exhaustion incident that forced those constants down from 8 to 4.
A non-blocking lock.acquire(blocking=False) rejects a second concurrent
request with 409 rather than queueing or running it in parallel — the
operator (or n8n's own retry/skip-if-running logic) decides what to do
next, this server never silently doubles up its own real work.

Usage: `uv run python -m qualume.server` (reads PORT from env, defaults
to 8420 - arbitrary, unlikely to collide with anything else on the host).
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from qualume.orchestrate import run_ingestion_cycle

DEFAULT_HOST = "127.0.0.1"  # localhost-only by default - see module docstring
DEFAULT_PORT = 8420

_cycle_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/run-ingestion-cycle":
            # Only POST may trigger the real, expensive, side-effecting
            # cycle (real Gemini/Claude API calls, real DB writes) — a
            # GET (e.g. from a browser preview, or a misconfigured
            # monitoring probe) must never accidentally run it.
            self._send_json(405, {"error": "use POST to run the ingestion cycle"})
        else:
            self._send_json(404, {"error": f"no such path: {self.path}"})

    def do_POST(self):
        if self.path != "/run-ingestion-cycle":
            self._send_json(404, {"error": f"no such path: {self.path}"})
            return
        if not _cycle_lock.acquire(blocking=False):
            # A cycle is already running — real 2026-10-31 incident
            # prevention, see module docstring. Reject, don't queue or
            # run concurrently.
            self._send_json(409, {"error": "a cycle is already running — try again later"})
            return
        try:
            result = run_ingestion_cycle()
            self._send_json(200, result)
        except Exception as e:
            # Every stage inside run_ingestion_cycle already catches its
            # own exceptions (see orchestrate.py) — reaching here means a
            # genuinely unexpected bug, which must surface as a real
            # error to n8n (failed workflow run, visible in its UI), not
            # a silent empty/broken response that looks like success.
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _cycle_lock.release()

    def log_message(self, format, *args):
        pass  # keep test output / systemd logs clean; real logging can be added later


def make_server(address: tuple[str, int] | None = None) -> ThreadingHTTPServer:
    if address is None:
        address = (DEFAULT_HOST, int(os.environ.get("PORT", DEFAULT_PORT)))
    return ThreadingHTTPServer(address, _Handler)


if __name__ == "__main__":
    server = make_server()
    host, port = server.server_address
    print(f"qualume orchestration server listening on http://{host}:{port}")
    print("  GET  /health               - liveness check, does nothing")
    print("  POST /run-ingestion-cycle   - poll -> download -> annotate -> classify")
    server.serve_forever()
