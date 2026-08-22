from __future__ import annotations

from collections import deque
from time import perf_counter

from flask import Flask, jsonify, request

from stonks_solver import solve_cases

app = Flask(__name__)
RECENT = deque(maxlen=5)


@app.get("/")
@app.get("/health")
@app.get("/v3/health")
@app.get("/v4/health")
def health():
    return jsonify({"ok": True, "service": "time-travelling-stonks-man"})


@app.post("/v3")
@app.post("/v4")
@app.post("/stonks")
@app.post("/api/stonks")
@app.post("/v3/stonks")
@app.post("/v4/stonks")
def stonks():
    started = perf_counter()
    payload = request.get_json(force=True, silent=False)
    if isinstance(payload, dict):
        cases = [payload]
        single = True
    else:
        cases = payload or []
        single = False

    result = solve_cases(cases)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    RECENT.append({
        "path": request.path,
        "case_count": len(cases),
        "elapsed_ms": elapsed_ms,
        "input": payload,
        "output": result[0] if single else result,
    })
    return jsonify(result[0] if single else result)


@app.get("/debug")
@app.get("/v3/debug")
@app.get("/v4/debug")
def debug():
    return jsonify({"recent": list(RECENT)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
