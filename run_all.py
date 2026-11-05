"""Regenerate every transcript in transcripts/ from the current code, in task order, under MOCK_LLM.

    python run_all.py

Each script runs in its own process, and its output (stdout + stderr) goes to transcripts/<file>.txt.
The live WebSocket check (ws_client.py) needs a running server, so uvicorn is started on a free local
port for it and stopped afterwards; the log lines the server wrote during that run are appended."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "transcripts"
LOG = ROOT / "logs" / "app.jsonl"
# Graded transcripts must use MOCK_LLM. Unbuffered output keeps warnings and prints in true order.
ENV = {**os.environ, "MOCK_LLM": "true", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

STEPS = [                                            # (transcript, script) in task order
    ("01_dataset.txt", "dataset.py"),
    ("02_knowledge_base.txt", "knowledge_base.py"),
    ("03_chunking_and_indexing.txt", "rag_core.py"),
    ("04_threshold_calibration.txt", "calibrate_threshold.py"),
    ("04_grounded_generation.txt", "grounded_demo.py"),
    ("05_chunking_comparison.txt", "rag_compare.py"),
    ("06_order_status_tool.txt", "tools.py"),
    ("07_09_crew_and_schema.txt", "crew_demo.py"),
    ("08_session_memory.txt", "memory_demo.py"),
    ("10_guardrails.txt", "guardrails_demo.py"),
    ("11_12_api_and_logging.txt", "api_demo.py"),
    ("11_websocket_live.txt", None),                 # uvicorn + ws_client.py
    ("13_evaluation.txt", "evaluation.py"),
    ("14_review_stage.txt", "review_demo.py"),
    ("15_governance.txt", "governance_demo.py"),
    ("16_cache.txt", "cache_demo.py"),
]


def run(transcript: str, script: str, *args: str, header: str = "") -> None:
    t0 = time.perf_counter()
    with open(OUT / transcript, "w", encoding="utf-8") as f:
        f.write(header + f"$ python {' '.join([script, *args])}\n\n")
        f.flush()
        code = subprocess.call([sys.executable, script, *args], cwd=ROOT, env=ENV,
                               stdout=f, stderr=subprocess.STDOUT)
    print(f"  {transcript:<32} exit {code}  {time.perf_counter() - t0:5.1f}s")
    if code:
        sys.exit(f"{script} failed: see transcripts/{transcript}")


def run_websocket_live(transcript: str) -> None:
    with socket.socket() as s:                       # pick a free local port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    start = LOG.stat().st_size if LOG.exists() else 0
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", str(port),
                               "--log-level", "warning"], cwd=ROOT, env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 180
        while True:                                  # wait until the server answers /health
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).close()
                break
            except OSError:
                if server.poll() is not None or time.time() > deadline:
                    sys.exit("uvicorn did not start")
                time.sleep(0.5)
        run(transcript, "ws_client.py", f"ws://127.0.0.1:{port}/ws/chat",
            header=f"# server: python -m uvicorn api:app --port {port}\n")
        time.sleep(1)                                # let the server log the first client's exit
    finally:
        server.terminate()
        server.wait(timeout=30)
    with open(LOG, "rb") as f:
        f.seek(start)
        new_lines = f.read().decode("utf-8").strip()
    with open(OUT / transcript, "a", encoding="utf-8") as f:
        f.write(f"\n--- logs/app.jsonl lines the server wrote during this run ---\n{new_lines}\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print(f"Writing transcripts to {OUT.name}/ (MOCK_LLM=true)")
    for transcript, script in STEPS:
        if script is None:
            run_websocket_live(transcript)
        else:
            run(transcript, script)
    print("Done.")
