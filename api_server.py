"""
api_server.py
=============
FastAPI backend that runs the pipeline in a background thread and
streams progress logs back to the frontend via polling.

Run with:
    pip install fastapi uvicorn
    python api_server.py
"""

import subprocess
import sys
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id → { status, logs, report_path, summary_path }
_jobs: dict[str, dict] = {}
_ROOT = Path(__file__).parent


class RunRequest(BaseModel):
    days:        int   = 7
    max_tickers: int   = 5
    min_volume:  float = 10_000.0
    max_markets: int   = 30


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":       "running",
        "logs":         [],
        "report_path":  None,
        "summary_path": None,
    }

    def _run():
        cmd = [
            sys.executable, "pipeline_orchestrator.py",
            "--days",        str(req.days),
            "--max-tickers", str(req.max_tickers),
            "--min-volume",  str(req.min_volume),
            "--max-markets", str(req.max_markets),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip()
            _jobs[job_id]["logs"].append(line)
            # Detect report paths from pipeline output
            if "Full report" in line and "→" in line:
                _jobs[job_id]["report_path"] = line.split("→")[-1].strip()
            if "Trader summary" in line and "→" in line:
                _jobs[job_id]["summary_path"] = line.split("→")[-1].strip()

        proc.wait()
        _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "error"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in _jobs:
        return {"status": "not_found", "logs": [], "report_path": None, "summary_path": None}
    return _jobs[job_id]


@app.get("/api/report/{job_id}")
def get_report(job_id: str):
    job = _jobs.get(job_id)
    if not job or not job.get("report_path"):
        return {"content": None}
    try:
        content = (_ROOT / job["report_path"]).read_text(encoding="utf-8")
        return {"content": content}
    except Exception:
        return {"content": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
