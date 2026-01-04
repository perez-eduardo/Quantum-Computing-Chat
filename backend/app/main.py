"""FastAPI application for Quantum Computing LLM."""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

# Add scripts to path
SCRIPTS_PATH = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from retrieval import Retriever
from groq_inference import GroqInference
from modal_inference import ModalInference
from app.config import (
    GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS,
    MODAL_URL, validate_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
retriever: Optional[Retriever] = None
groq_inference: Optional[GroqInference] = None
modal_inference: Optional[ModalInference] = None


def get_groq() -> GroqInference:
    """Lazy initialization for Groq client."""
    global groq_inference
    if groq_inference is None:
        groq_inference = GroqInference(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
    return groq_inference


def get_modal() -> Optional[ModalInference]:
    """Lazy initialization for Modal client."""
    global modal_inference
    if modal_inference is None and MODAL_URL:
        modal_inference = ModalInference(url=MODAL_URL)
    return modal_inference


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_suggested_question(original: str, answer: str, results: List[dict]) -> Optional[str]:
    """Find a relevant follow-up question from retrieved results."""
    if not results:
        return None
    
    answer_words = set(
        w.lower().strip(".,!?()[]{}:;\"'") 
        for w in answer.split() 
        if len(w) > 5
    )
    candidates = []
    
    for r in results:
        q = r.get("question", "").strip()
        if not q:
            continue
        sim = text_similarity(original, q)
        if sim > 0.6:
            continue
        matches = sum(1 for w in answer_words if w in q.lower())
        candidates.append({"question": q, "similarity": sim, "matches": matches})
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: (-x["matches"], x["similarity"]))
    return candidates[0]["question"]


def build_context(results: List[dict], top_k: int = 3) -> str:
    """Build context string from retrieved results."""
    parts = [f"Q: {r['question']} A: {r['answer'][:300]}" for r in results[:top_k]]
    return " ".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global retriever
    logger.info("Starting Quantum Computing LLM API...")
    validate_config()
    retriever = Retriever()
    logger.info("Ready")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Quantum Computing LLM API",
    version="4.0.0",
    lifespan=lifespan
)


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    model: str = "groq"


class Source(BaseModel):
    question: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    answer: str
    sources: List[Source]
    response_time_ms: int
    suggested_question: Optional[str]
    model_used: str


class HealthResponse(BaseModel):
    status: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint."""
    return Response(status_code=204)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a question and return an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start = time.time()
    
    # Retrieve context
    results = retriever.search(request.question, top_k=5)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    context = build_context(results, top_k=3)
    
    # Select model and generate
    if request.model == "custom":
        modal = get_modal()
        if not modal:
            raise HTTPException(status_code=400, detail="Custom model not configured")
        logger.info("Using Custom Model (Modal)")
        llm = modal
        model_used = "custom"
    else:
        logger.info("Using Groq")
        llm = get_groq()
        model_used = "groq"
    
    answer = llm.generate(context, request.question)
    suggested = get_suggested_question(request.question, answer, results)
    elapsed_ms = int((time.time() - start) * 1000)
    
    sources = [
        Source(
            question=r["question"][:100],
            source=r["source"],
            similarity=round(r["similarity"], 4)
        )
        for r in results[:3]
    ]
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed_ms,
        suggested_question=suggested,
        model_used=model_used
    )
