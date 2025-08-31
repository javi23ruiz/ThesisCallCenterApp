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

# Google Cloud Speech-to-Text (voice input)
# Provide Application Default Credentials via a JSON credentials file path
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/credentials.json
```

3. Run the API locally:

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health` → health check
- `POST /ask` → body: `{ "question": "...", "session_id": "<id>" }` → returns `{ "answer": "<AI reply as a live call center agent>" }`
- `POST /stt` → multipart form data with `audio` (webm/opus) → returns `{ "text": "transcript" }`

## Frontend

Open `frontend/index.html` in your browser. The page will call `http://localhost:8000/ask`.

If you need a local static server instead of opening the file directly:

```bash
cd frontend
python3 -m http.server 5500
```

Then navigate to `http://localhost:5500`.

### Voice input
- Click the phone button to start/stop recording. It records WebM/Opus and sends it to `/stt`.
- The transcript is posted to `/ask` using your persistent session to preserve memory.
