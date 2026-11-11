"""Flask frontend for Quantum Computing LLM."""

import os
import logging
from flask import Flask, render_template, request, jsonify
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def backend_response(response):
    """Keep upstream failures useful even when the backend does not return JSON."""
    try:
        data = response.json()
    except ValueError:
        logger.error("Backend returned non-JSON response (HTTP %s)", response.status_code)
        return jsonify({
            "error": "Backend returned an invalid response. Check the backend logs."
        }), 502

    if not isinstance(data, dict):
        return jsonify({"error": "Backend returned an invalid response."}), 502
    if not response.ok:
        message = data.get("error") or data.get("detail")
        if not isinstance(message, str):
            message = "Backend could not process the request."
        return jsonify({"error": message}), response.status_code
    return jsonify(data), response.status_code


@app.route("/")
def index():
    """Serve main page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Proxy query to backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=request.get_json(),
            timeout=300
        )
        return backend_response(response)
    except requests.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.ConnectionError:
        return jsonify({"error": "Cannot connect to backend."}), 503
    except requests.RequestException:
        logger.exception("Backend query request failed")
        return jsonify({"error": "Backend request failed. Please try again."}), 502


@app.route("/api/health", methods=["GET"])
def api_health():
    """Proxy health check to backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return backend_response(response)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": "Backend unavailable"}), 503


@app.route("/health", methods=["GET"])
def health():
    """Frontend health check."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
