from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from fastapi import UploadFile, File, Form
from fastapi import WebSocket, WebSocketDisconnect, Body
from google.cloud import speech
from google.cloud import texttospeech
import asyncio
from queue import Queue
import threading
from fastapi.responses import Response, JSONResponse
import random
import time


app = FastAPI(title="Thesis Call Center API")

# Allow the local frontend to call the backend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class RagScoreRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- LLM setup (loaded once) ---
load_dotenv()
_openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
_openai_model: str = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

_llm: Optional[ChatOpenAI] = None
_chain = None
_chain_with_history = None

# simple in-memory store for chat histories
_session_store: dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

if _openai_api_key:
    _llm = ChatOpenAI(model=_openai_model, temperature=0.3)

    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a professional live call center agent for an insurance company. "
                    "Speak in a friendly, concise, and helpful tone. "
                    "Confirm understanding, ask clarifying questions only when needed, and provide actionable steps. "
                    "If policy-specific details are required but not provided, explain how the customer can find them. "
                    "Avoid over-promising, never fabricate policy details, and offer to escalate to a human agent when appropriate. "
                    "Use short paragraphs and bullet points where helpful."
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Customer question: {question}")
        ]
    )

    _chain = _prompt | _llm | StrOutputParser()
    _chain_with_history = RunnableWithMessageHistory(
        _chain,
        lambda session_id: _get_session_history(session_id),
        input_messages_key="question",
        history_messages_key="history",
    )

# --- Google Speech-to-Text setup ---
_speech_client: Optional[speech.SpeechClient] = None
_speech_init_error: Optional[str] = None
_gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
try:
    if _gcp_credentials_path and os.path.exists(_gcp_credentials_path):
        # Use explicit service account file to avoid ADC ambiguity
        _speech_client = speech.SpeechClient.from_service_account_file(_gcp_credentials_path)
    else:
        # Attempt default credentials (gcloud auth/application default)
        _speech_client = speech.SpeechClient()
except Exception as e:
    _speech_client = None
    _speech_init_error = str(e)

# --- Google Text-to-Speech setup ---
_tts_client: Optional[texttospeech.TextToSpeechClient] = None
try:
    _tts_client = texttospeech.TextToSpeechClient()
except Exception:
    _tts_client = None


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if _chain is None:
        return {
            "answer": (
                "The assistant is not configured. Please set OPENAI_API_KEY in your environment "
                "and restart the server."
            )
        }

    session_id = req.session_id or "default"
    answer: str = _chain_with_history.invoke(
        {"question": req.question},
        config={"configurable": {"session_id": session_id}},
    )
    return {"answer": answer}


@app.post("/rag/score")
def rag_score(req: RagScoreRequest) -> dict:
    """Temporary scoring endpoint.
    Returns a mock confidence in [0,1]. Later, replace with a real model.
    """
    confidence = random.random()
    time.sleep(2)
    return {"confidence": confidence}


@app.post("/stt")
async def stt(
    audio: UploadFile = File(...),
    language_code: str = Form("en-US"),
) -> dict:
    if _speech_client is None:
        return {"text": "", "error": f"Speech client not configured. {_speech_init_error or 'Set GOOGLE_APPLICATION_CREDENTIALS and restart.'}"}

    data = await audio.read()
    if not data:
        return {"text": "", "error": "Empty audio."}

    try:
        # Try to infer encoding from uploaded MIME type
        mime = (audio.content_type or "").lower()
        if "ogg" in mime:
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        elif "webm" in mime:
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        elif "mp3" in mime:
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
        elif "wav" in mime or "x-wav" in mime or "wave" in mime:
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        else:
            # reasonable default for modern browsers
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS

        audio_msg = speech.RecognitionAudio(content=data)
        config = speech.RecognitionConfig(
            encoding=encoding,
            enable_automatic_punctuation=True,
            language_code=language_code,
        )
        response = _speech_client.recognize(config=config, audio=audio_msg)
        transcript_parts = [result.alternatives[0].transcript for result in response.results if result.alternatives]
        transcript = " ".join(transcript_parts).strip()
        return {"text": transcript}
    except Exception as e:
        return {"text": "", "error": str(e)}


@app.websocket("/stt/stream")
async def stt_stream(ws: WebSocket):
    await ws.accept()
    if _speech_client is None:
        await ws.send_json({"error": "Speech client not configured"})
        await ws.close()
        return

    # Prepare streaming config
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
        ),
        interim_results=True,
        single_utterance=False,
    )

    loop = asyncio.get_event_loop()
    audio_queue: Queue[Optional[bytes]] = Queue()

    def request_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def recognizer():
        try:
            # Pass config explicitly; generator yields only audio chunks
            responses = _speech_client.streaming_recognize(
                config=streaming_config,
                requests=request_generator(),
            )
            for response in responses:
                for result in response.results:
                    text = result.alternatives[0].transcript if result.alternatives else ""
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json({"text": text, "is_final": result.is_final}),
                        loop,
                    )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(ws.send_json({"error": str(e)}), loop)

    thread = threading.Thread(target=recognizer, daemon=True)
    thread.start()

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                audio_queue.put(msg["bytes"]) 
            elif msg.get("type") in ("websocket.disconnect",):
                break
            else:
                # Text message used as control message: "STOP"
                if msg.get("text") == "STOP":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        audio_queue.put(None)
        await ws.close()


@app.post("/tts")
def tts(text: str = Body(..., media_type="text/plain")):
    if _tts_client is None:
        return JSONResponse({"error": "TTS client not configured. Set GOOGLE_APPLICATIONS_CREDENTIALS."}, status_code=500)

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
    response = _tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return Response(content=response.audio_content, media_type="audio/mpeg")


# Helpful for `python backend/server.py` during local development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)


