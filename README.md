# Quantum Computing LLM

A full-stack ML application for answering quantum computing questions. Features dual-mode inference with Groq API (fast) and a custom 140M parameter transformer (demo).

**Live Demo:** https://quantum-computing-llm.up.railway.app

## Architecture

```
User → Frontend (Flask) → Backend (FastAPI) → Voyage AI (embeddings) → Neon (vector search)
                                                                              ↓
                                                         model="groq"   → Groq API (~2-3s)
                                                         model="custom" → Modal GPU (~30-60s)
```

## Quick Start

### 1. Set Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required services:
- [Voyage AI](https://www.voyageai.com/) for embeddings
- [Neon](https://neon.tech/) PostgreSQL with pgvector
- [Groq](https://groq.com/) for fast inference
- [Modal](https://modal.com/) for custom model (optional)

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

### 3. Run Frontend

```bash
cd frontend
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
python app.py
```

Visit http://localhost:3000

On Windows PowerShell, you can install both services into one local environment
from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
Copy-Item .env.example .env
# Fill in .env with your service credentials before starting the backend.
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Then start the frontend in a second terminal from the repository root:

```powershell
.\.venv\Scripts\python -m flask --app frontend.app run --host 127.0.0.1 --port 3000
```

### Troubleshooting query errors

The frontend `/health` endpoint only checks the frontend process. Use
`/api/health` to check backend connectivity. Query failures identify retrieval
or model service errors; the backend logs contain the underlying exception.
Non-JSON backend responses return a readable gateway error instead of a JSON
parsing exception.

Run the error-handling regression tests from the repository root:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py      # Environment configuration
│   │   └── main.py        # FastAPI endpoints
│   └── scripts/
│       ├── retrieval.py       # RAG retrieval (Voyage + Neon)
│       ├── groq_inference.py  # Groq API client
│       └── modal_inference.py # Modal API client
│
└── frontend/
    ├── app.py             # Flask server
    ├── templates/         # Jinja templates
    └── static/            # CSS, JS, images
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Send question, receive answer |

### POST /query

```json
{
  "question": "What is a qubit?",
  "model": "groq"
}
```

Response:
```json
{
  "answer": "A qubit is...",
  "sources": [...],
  "response_time_ms": 2500,
  "model_used": "groq"
}
```

## Deployment

Both services deploy to Railway. See `Dockerfile` in each folder.

## License

MIT
