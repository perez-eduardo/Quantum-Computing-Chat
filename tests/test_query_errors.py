"""Regression tests for upstream failures that previously became JSON parse errors."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import main as backend
from frontend import app as frontend


class FrontendQueryTests(unittest.TestCase):
    def setUp(self):
        self.client = frontend.app.test_client()

    def query(self, response):
        with patch.object(frontend.requests, "post", return_value=response):
            return self.client.post("/api/query", json={"question": "What is a qubit?"})

    def test_plain_text_backend_failure(self):
        response = Mock(status_code=500, ok=False)
        response.json.side_effect = ValueError("Expecting value")
        result = self.query(response)
        self.assertEqual(result.status_code, 502)
        self.assertIn("Check the backend logs", result.json["error"])

    def test_fastapi_error_detail_is_preserved(self):
        response = Mock(status_code=503, ok=False)
        response.json.return_value = {"detail": "Knowledge retrieval is unavailable."}
        result = self.query(response)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json, {"error": "Knowledge retrieval is unavailable."})

    def test_invalid_json_shape(self):
        response = Mock(status_code=200, ok=True)
        response.json.return_value = ["unexpected"]
        self.assertEqual(self.query(response).status_code, 502)

    def test_success_is_preserved(self):
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {"answer": "A qubit is a quantum bit."}
        result = self.query(response)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, response.json.return_value)

    def test_connection_failure_and_timeout(self):
        for error, status in [(requests.ConnectionError(), 503), (requests.Timeout(), 504)]:
            with self.subTest(status=status), patch.object(frontend.requests, "post", side_effect=error):
                result = self.client.post("/api/query", json={"question": "What is a qubit?"})
                self.assertEqual(result.status_code, status)
                self.assertIsInstance(result.json["error"], str)


class BackendQueryTests(unittest.IsolatedAsyncioTestCase):
    async def query(self, question="What is a qubit?"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=backend.app), base_url="http://test"
        ) as client:
            return await client.post("/query", json={"question": question})

    async def test_retrieval_failure_is_json(self):
        retriever = Mock()
        retriever.search.side_effect = RuntimeError("private database connection details")
        with patch.object(backend, "retriever", retriever):
            result = await self.query()
        self.assertEqual(result.status_code, 503)
        self.assertIn("Knowledge retrieval", result.json()["detail"])
        self.assertNotIn("private", result.text)

    async def test_generation_failure_is_json(self):
        retriever = Mock()
        retriever.search.return_value = [{"question": "Q?", "answer": "A", "source": "test", "similarity": 0.9}]
        llm = Mock()
        llm.generate.side_effect = RuntimeError("private API error")
        with patch.object(backend, "retriever", retriever), patch.object(backend, "get_groq", return_value=llm):
            result = await self.query()
        self.assertEqual(result.status_code, 502)
        self.assertIn("Answer generation failed", result.json()["detail"])
        self.assertNotIn("private", result.text)

    async def test_empty_question_is_still_rejected(self):
        result = await self.query(" ")
        self.assertEqual(result.status_code, 400)


if __name__ == "__main__":
    unittest.main()
