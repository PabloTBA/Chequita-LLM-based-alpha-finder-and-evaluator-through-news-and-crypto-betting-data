"""
api_server.py
=============
FastAPI server that wraps PipelineOrchestrator and streams every print()
statement to the browser in real-time via Server-Sent Events (SSE).

Start with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
  POST /api/run        Start the pipeline (JSON body optional — see RunRequest)
  GET  /api/logs       SSE stream — one line per event, ends with data: [DONE]
  GET  /api/status     {"status": "idle|running|done|error", "run_date": ..., ...}
  GET  /api/report     Last full report as plain-text Markdown
  GET  /api/summary    Last trader summary as plain-text Markdown

Default parameters (matching: python pipeline_orchestrator.py
                              --days 7 --max-tickers 5
                              --min-volume 10000 --max-markets 30)
-----------------------------------------------------------------
  days          = 7
  max_tickers   = 5
  min_volume    = 10 000
  max_markets   = 30
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import sys
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Chequita Alpha Finder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your front-end origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (single-run server — one pipeline run at a time) ─────────────

_state: dict[str, Any] = {
    "status":       "idle",   # idle | running | done | error
    "run_date":     None,
    "SummaryReport": None,
    "TraderReport":  None,
    "error":        None,
    "params":       None,
}

# Lines of stdout are pushed here by the capture writer and consumed by SSE.
_log_queue: queue.Queue[str] = queue.Queue()

# Lock prevents two simultaneous pipeline runs.
_run_lock = threading.Lock()


# ── Stdout capture ────────────────────────────────────────────────────────────

class _QueueWriter(io.TextIOBase):
    """
    Drop-in replacement for sys.stdout that:
      1. Writes every character to the original terminal (so docker logs / CLI work).
      2. Buffers text until a newline, then pushes complete lines to _log_queue.

    Because sys.stdout is a module-level reference shared by all threads,
    replacing it once (before spawning the pipeline's ThreadPoolExecutor) is
    sufficient to capture output from every worker thread.
    """

    def __init__(self, real: io.TextIOBase) -> None:
        self._real = real
        self._buf  = ""

    def write(self, text: str) -> int:
        # Mirror to terminal
        self._real.write(text)
        self._real.flush()

        # Buffer and split on newlines
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            _log_queue.put(line)
        return len(text)

    def flush(self) -> None:
        # If an incomplete line is pending (e.g. from a progress indicator),
        # flush it so the browser always sees the latest output.
        if self._buf:
            _log_queue.put(self._buf)
            self._buf = ""
        self._real.flush()

    def fileno(self) -> int:
        # Some third-party libraries call fileno(); delegate to real stdout.
        return self._real.fileno()

    # Make the wrapper look like a real text stream to libraries that check.
    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real, "errors", "replace")

    def readable(self)  -> bool: return False
    def writable(self)  -> bool: return True
    def seekable(self)  -> bool: return False


# ── Pipeline runner ───────────────────────────────────────────────────────────

_STREAM_END = "__PIPELINE_STREAM_END__"   # sentinel pushed at end of every run


def _run_pipeline(cfg: dict, run_date: str | None) -> None:
    """
    Runs inside a daemon thread.
    Replaces sys.stdout for the duration so every print() — including those
    from DiagnosticsEngine, MonteCarloEngine, Backtester, etc. — appears in
    the SSE stream.
    """
    real_stdout = sys.stdout
    sys.stdout  = _QueueWriter(real_stdout)
    try:
        from datetime import datetime as _dt
        _state["status"] = "running"
        _state["error"]  = None

        _log_queue.put(
            f"[API] Pipeline started — {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        _log_queue.put(
            f"[API] Parameters: window_days={cfg.get('window_days')}  "
            f"max_tickers={cfg.get('max_tickers')}  "
            f"market_min_volume={cfg.get('market_min_volume')}  "
            f"market_max_markets={cfg.get('market_max_markets')}"
        )

        from pipeline_orchestrator import PipelineOrchestrator
        result = PipelineOrchestrator(cfg).run(run_date)

        _state["status"]       = "done"
        _state["run_date"]     = result.get("run_date")
        _state["SummaryReport"] = result.get("SummaryReport")
        _state["TraderReport"]  = result.get("TraderReport")

        _log_queue.put(f"[API] Done.  SummaryReport → {result.get('SummaryReport')}")
        _log_queue.put(f"[API]        TraderReport  → {result.get('TraderReport')}")

    except Exception as exc:
        import traceback
        _state["status"] = "error"
        _state["error"]  = str(exc)
        _log_queue.put(f"[API] ERROR: {exc}")
        _log_queue.put(traceback.format_exc())

    finally:
        # Restore real stdout before pushing the sentinel
        sys.stdout = real_stdout
        # Flush any buffered partial line
        if isinstance(sys.stdout, _QueueWriter):
            sys.stdout.flush()
        _log_queue.put(_STREAM_END)


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    """
    All fields are optional — defaults exactly match the CLI invocation:
        python pipeline_orchestrator.py --days 7 --max-tickers 5
                                        --min-volume 10000 --max-markets 30
    """
    date:        str | None = None    # YYYY-MM-DD; None → yesterday UTC+8
    days:        int        = 14      # news summary window (capped at 14)
    max_tickers: int        = 15      # max tickers to fully analyse
    min_volume:  float      = 10_000  # min prediction-market volume (USD)
    max_markets: int        = 30      # max prediction markets to store/embed


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/run")
def start_run(req: RunRequest = RunRequest()):
    """
    Start the pipeline in a background thread.
    Returns 409 if a run is already in progress.

    Called when the user clicks Generate in the front-end.
    """
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in progress. "
                   "Wait for it to finish or check /api/status.",
        )

    # Drain any leftover log lines from a previous run so /api/logs is clean.
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break

    # ── Build LLM client and config ───────────────────────────────────────────
    try:
        import ollama
        from dotenv import load_dotenv
        load_dotenv()

        def llm(prompt: str) -> str:
            resp = ollama.chat(
                model="qwen3:14b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            return (
                resp.message.content
                if hasattr(resp, "message")
                else resp["message"]["content"]
            )

        cfg: dict[str, Any] = {
            "benzinga_api_key":   os.getenv("BENZINGA_API"),
            "llm_client":         llm,
            "output_dir":         "reports",
            "cache_dir":          "data/cache",
            "chroma_dir":         "data/chroma",
            "initial_portfolio":  100_000.0,
            "window_days":        min(req.days, 14),   # cap at 14 (same as CLI)
            "max_tickers":        req.max_tickers,
            "market_min_volume":  req.min_volume,
            "market_max_markets": req.max_markets,
        }
    except Exception as exc:
        _run_lock.release()
        raise HTTPException(status_code=500, detail=f"Setup failed: {exc}")

    # Store readable params in state (no llm_client callable — not serialisable)
    _state["params"] = {k: v for k, v in cfg.items() if k != "llm_client"}

    def _worker() -> None:
        try:
            _run_pipeline(cfg, req.date)
        finally:
            _run_lock.release()

    threading.Thread(target=_worker, daemon=True, name="pipeline-run").start()

    return {
        "status":  "started",
        "params":  _state["params"],
    }


@app.get("/api/logs")
async def stream_logs():
    """
    Server-Sent Events endpoint.
    Each event carries one line of stdout from the running pipeline.
    The stream closes automatically when the pipeline finishes.

    Front-end usage:
        const es = new EventSource('/api/logs');
        es.onmessage = e => console.log(e.data);
    """
    async def _generate():
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Block for up to 0.4 s so we can yield SSE keepalives
                line: str = await loop.run_in_executor(
                    None, lambda: _log_queue.get(timeout=0.4)
                )

                if line == _STREAM_END:
                    yield "data: [DONE]\n\n"
                    break

                # SSE framing: newlines inside the line would break the protocol,
                # so replace embedded newlines with a visible marker.
                safe = line.replace("\r", "").replace("\n", " ↵ ")
                yield f"data: {safe}\n\n"

            except queue.Empty:
                # No log line yet — send a keepalive comment to prevent the
                # browser / proxy from closing an idle connection.
                yield ": keepalive\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


@app.get("/api/status")
def get_status():
    """Current pipeline state.  Poll this after the SSE stream closes."""
    return _state


@app.get("/api/report", response_class=PlainTextResponse)
def get_report():
    """Return the last full pipeline report as Markdown text."""
    path = _state.get("SummaryReport")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No report available yet.")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/api/summary", response_class=PlainTextResponse)
def get_summary():
    """Return the last trader summary as Markdown text."""
    path = _state.get("TraderReport")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No summary available yet.")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
