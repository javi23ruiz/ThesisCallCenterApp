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


# Helpful for `python backend/server.py` during local development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)


