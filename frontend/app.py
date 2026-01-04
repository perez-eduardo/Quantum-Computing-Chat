"""Flask frontend for Quantum Computing LLM."""

import os
import logging
from flask import Flask, render_template, request, jsonify
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
        return jsonify(response.json()), response.status_code
    except requests.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.ConnectionError:
        return jsonify({"error": "Cannot connect to backend."}), 503
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Proxy health check to backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": "Backend unavailable"}), 503


@app.route("/health", methods=["GET"])
def health():
    """Frontend health check."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
