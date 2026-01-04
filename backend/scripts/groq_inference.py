"""Groq API inference for Quantum Computing LLM."""

import logging
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantum computing assistant for beginners. 
Answer using the provided context. Keep explanations simple and accessible.
Do not use complex math or equations. Be concise but thorough."""


class GroqInference:
    """Groq API client for LLM inference."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, context: str, question: str) -> str:
        """Generate an answer given context and question."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")
