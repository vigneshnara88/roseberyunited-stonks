from __future__ import annotations

from flask import Flask, jsonify, request

from stonks_solver import solve_cases

app = Flask(__name__)


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "time-travelling-stonks-man"})


@app.post("/stonks")
def stonks():
    payload = request.get_json(force=True, silent=False)
    if isinstance(payload, dict):
        cases = [payload]
        single = True
    else:
        cases = payload or []
        single = False

    result = solve_cases(cases)
    return jsonify(result[0] if single else result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
