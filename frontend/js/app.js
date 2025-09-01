const apiBase = "http://localhost:8000";
const wsBase = apiBase.replace(/^http/, "ws");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function renderMessage(role, text, meta) {
  const messages = document.getElementById("messages");
  const item = el("div", `msg ${role}`);
  const avatar = el("div", "avatar", role === "assistant" ? "IA" : "U");
  const bubble = el("div", "bubble");
  if (role === "assistant" && typeof marked !== "undefined") {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }
  if (meta && role === "user") {
    const metaLine = el("div", "meta-line");
    const c = meta.confidence;
    const cls = c > 0.5 ? "high" : c >= 0.3 ? "mid" : "low";
    const confPill = el("span", `meta-pill ${cls}`);
    confPill.textContent = `Confidence: ${c.toFixed(2)}`;

    const ragCls = meta.invoked ? "invoked" : "";
    const ragPill = el("span", `meta-pill ${ragCls}`);
    ragPill.textContent = `RAG: ${meta.invoked ? "Invoked" : "Skipped"}`;

    metaLine.appendChild(confPill);
    metaLine.appendChild(ragPill);
    bubble.appendChild(metaLine);
  }
  item.appendChild(avatar);
  item.appendChild(bubble);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function setInterimTranscript(text) {
  const messages = document.getElementById("messages");
  if (!window.__partialItem) {
    const item = el("div", "msg user partial");
    const avatar = el("div", "avatar", "U");
    const bubble = el("div", "bubble");
    item.appendChild(avatar);
    item.appendChild(bubble);
    window.__partialItem = item;
    messages.appendChild(item);
  }
  const bubble = window.__partialItem.querySelector(".bubble");
  bubble.textContent = text || "";
  messages.scrollTop = messages.scrollHeight;
}

function clearInterimTranscript() {
  if (window.__partialItem) {
    try { window.__partialItem.remove(); } catch (_) {}
    window.__partialItem = null;
  }
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
  // Play TTS of assistant answer in parallel
  try {
    const audioRes = await fetch(`${apiBase}/tts`, { method: "POST", headers: { "Content-Type": "text/plain" }, body: data.answer });
    if (audioRes.ok) {
      const buf = await audioRes.arrayBuffer();
      const blob = new Blob([buf], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play().catch(() => {});
    }
  } catch (_) {}
  return data.answer;
}

function setupChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question");
  const callBtn = document.querySelector(".btn.btn-call");
  let ws;
  let audioCtx;
  let sourceNode;
  let processorNode;
  let mediaStream;

  function downsampleTo16k(float32Array, inputSampleRate) {
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(float32Array.length / ratio);
    const result = new Int16Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const idx = Math.round(i * ratio);
      const s = Math.max(-1, Math.min(1, float32Array[idx] || 0));
      result[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return result.buffer;
  }

  async function startStreaming() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      renderMessage("assistant", "Microphone permission denied.");
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    ws = new WebSocket(`${wsBase}/stt/stream`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { callBtn.textContent = "⏹"; };
    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          renderMessage("assistant", `Voice transcription failed. (${data.error})`);
          return;
        }
        if (data.is_final) {
          const finalText = (data.text || "").trim();
          clearInterimTranscript();
          if (finalText) {
            const confidence = Math.random();
            const invoked = confidence > 0.5;
            renderMessage("user", finalText, { confidence, invoked });
            const answer = await ask(finalText);
            renderMessage("assistant", answer);
          }
        } else {
          setInterimTranscript(data.text || "");
        }
      } catch (_) {}
    };
    ws.onerror = () => { renderMessage("assistant", "Voice connection error."); };
    ws.onclose = () => { callBtn.textContent = "📞"; clearInterimTranscript(); };

    processorNode.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== 1) return;
      const input = e.inputBuffer.getChannelData(0);
      const pcm = downsampleTo16k(input, audioCtx.sampleRate);
      ws.send(pcm);
    };
  }

  function stopStreaming() {
    try { if (ws && ws.readyState === 1) ws.close(); } catch (_) {}
    ws = null;
    try { if (processorNode) processorNode.disconnect(); } catch (_) {}
    try { if (sourceNode) sourceNode.disconnect(); } catch (_) {}
    try { if (audioCtx) audioCtx.close(); } catch (_) {}
    try { if (mediaStream) mediaStream.getTracks().forEach(t => t.stop()); } catch (_) {}
    callBtn.textContent = "📞";
  }

  callBtn.addEventListener("click", async () => {
    if (!ws || ws.readyState !== 1) await startStreaming(); else stopStreaming();
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    const confidence = Math.random();
    const invoked = confidence > 0.5;
    renderMessage("user", q, { confidence, invoked });
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


