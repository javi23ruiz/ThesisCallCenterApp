const apiBase = "http://localhost:8000";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function renderMessage(role, text) {
  const messages = document.getElementById("messages");
  const item = el("div", `msg ${role}`);
  const avatar = el("div", "avatar", role === "assistant" ? "IA" : "U");
  const bubble = el("div", "bubble");
  if (role === "assistant" && typeof marked !== "undefined") {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }
  item.appendChild(avatar);
  item.appendChild(bubble);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function getSessionId() {
  const KEY = "chat_session_id";
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = Math.random().toString(36).slice(2);
    localStorage.setItem(KEY, id);
  }
  return id;
}

async function ask(question) {
  const res = await fetch(`${apiBase}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: getSessionId() })
  });
  if (!res.ok) throw new Error("Request failed");
  const data = await res.json();
  return data.answer;
}

function setupChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question");
  const callBtn = document.querySelector(".btn.btn-call");
  let mediaRecorder;
  let chunks = [];
  let chosenMime = null;

  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/ogg;codecs=opus",
      "audio/webm"
    ];
    for (const c of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) {
        return c;
      }
    }
    return null;
  }

  async function startRecording() {
    chosenMime = pickMimeType();
    if (!chosenMime) {
      renderMessage("assistant", "Voice recording is not supported in this browser. Please use the latest Chrome.");
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: chosenMime });
    chunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      const blobType = chosenMime.includes("ogg") ? "audio/ogg" : "audio/webm";
      const blob = new Blob(chunks, { type: blobType });
      const formData = new FormData();
      const filename = blobType === "audio/ogg" ? "audio.ogg" : "audio.webm";
      formData.append("audio", blob, filename);
      try {
        const res = await fetch(`${apiBase}/stt`, { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const transcript = (data.text || "").trim();
        if (transcript) {
          renderMessage("user", transcript);
          const answer = await ask(transcript);
          renderMessage("assistant", answer);
        }
      } catch (err) {
        renderMessage("assistant", `Voice transcription failed. ${err?.message ? "(" + err.message + ")" : ""}`);
        console.error(err);
      }
    };
    mediaRecorder.start();
    callBtn.textContent = "⏹";
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      callBtn.textContent = "📞";
    }
  }

  callBtn.addEventListener("click", async () => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      await startRecording();
    } else {
      stopRecording();
    }
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    renderMessage("user", q);
    try {
      const answer = await ask(q);
      renderMessage("assistant", answer);
    } catch (err) {
      renderMessage("assistant", "Oops, something went wrong.");
      console.error(err);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderMessage("assistant", "Hello! I'm your insurance assistant. How can I help?");
  setupChat();
});


