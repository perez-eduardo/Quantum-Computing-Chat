"""Modal API inference for Quantum Computing LLM."""

import logging
import requests

logger = logging.getLogger(__name__)


class ModalInference:
    """Modal API client for custom model inference."""
    
    def __init__(self, url: str, timeout: int = 300):
        self.url = url
        self.timeout = timeout
    
    def generate(self, context: str, question: str) -> str:
        """Generate an answer by calling Modal API."""
        try:
            response = requests.post(
                self.url,
                json={"context": context, "question": question},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "")
        except requests.Timeout:
            logger.error("Modal API timeout")
            raise RuntimeError("Custom model request timed out")
        except requests.RequestException as e:
            logger.error(f"Modal API error: {e}")
            raise RuntimeError(f"Failed to connect to custom model: {e}")
