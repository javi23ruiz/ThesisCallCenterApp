# ThesisCallCenterApp

Minimal FastAPI backend with a dark, chat-style frontend inspired by the provided UI.

## Backend (FastAPI)

1. Create a virtual environment and install dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure OpenAI credentials (required for the call center responses):

```bash
export OPENAI_API_KEY=your_api_key_here
# Optional: choose a model (default: gpt-4o-mini)
export OPENAI_API_MODEL=gpt-4o-mini
```

3. Run the API locally:

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health` → health check
- `POST /ask` → body: `{ "question": "..." }` → returns `{ "answer": "<AI reply as a live call center agent>" }`

## Frontend

Open `frontend/index.html` in your browser. The page will call `http://localhost:8000/ask`.

If you need a local static server instead of opening the file directly:

```bash
cd frontend
python3 -m http.server 5500
```

Then navigate to `http://localhost:5500`.
