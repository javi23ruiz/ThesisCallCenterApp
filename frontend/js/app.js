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
  const avatar = el("div", "avatar");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "currentColor");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  if (role === "assistant") {
    // simple headset icon
    path.setAttribute("d", "M12 3a7 7 0 0 0-7 7v3a3 3 0 0 0 3 3h1v-6H6v-0a6 6 0 1 1 12 0v0h-3v6h1a3 3 0 0 0 3-3V10a7 7 0 0 0-7-7z");
  } else {
    // user circle icon
    path.setAttribute("d", "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 5a3 3 0 1 1 0 6 3 3 0 0 1 0-6zm0 12a7 7 0 0 1-5.916-3.17A5 5 0 0 1 12 13a5 5 0 0 1 5.916 2.83A7 7 0 0 1 12 19z");
  }
  svg.appendChild(path);
  avatar.appendChild(svg);
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

function updateRagSidebar(snippet) {
  const container = document.getElementById("rag-list");
  if (!container) return;
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <div class="card-header">
      <span class="tag">Retrieved</span>
      <span class="muted">Dynamic</span>
    </div>
    <h3>${(snippet.title || "Related Section").replace(/</g, "&lt;")}</h3>
    <div class="highlight"><strong>${(snippet.heading || "Match").replace(/</g, "&lt;")}:</strong> ${(snippet.excerpt || "").replace(/</g, "&lt;")}</div>
    <ul class="bullets">${(snippet.points || []).map(p => `<li>${p.replace(/</g, "&lt;")}</li>`).join("")}</ul>
  `;
  container.appendChild(card);
  // Scroll to the bottom so the newest appears after the previous ones
  try { container.scrollTop = container.scrollHeight; } catch (_) {}
}

function showRagPopup(show) {
  const elPopup = document.getElementById("rag-popup");
  if (!elPopup) return;
  if (show) elPopup.classList.add("show"); else elPopup.classList.remove("show");
}

function showRagPopupTimed(durationMs = 6000) {
  showRagPopup(true);
  try { clearTimeout(showRagPopupTimed._t); } catch (_) {}
  showRagPopupTimed._t = setTimeout(() => showRagPopup(false), durationMs);
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
  let audioEnded = Promise.resolve();
  try {
    const audioRes = await fetch(`${apiBase}/tts`, { method: "POST", headers: { "Content-Type": "text/plain" }, body: data.answer });
    if (audioRes.ok) {
      const buf = await audioRes.arrayBuffer();
      const blob = new Blob([buf], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play().catch(() => {});
      audioEnded = new Promise((resolve) => {
        audio.addEventListener("ended", resolve, { once: true });
        audio.addEventListener("error", resolve, { once: true });
      });
    }
  } catch (_) {}
  return { answer: data.answer, audioEnded };
}

function setupChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question");
  const callBtn = document.querySelector(".btn.btn-call");
  const sendBtn = document.querySelector(".btn.btn-send");
  let isBusy = false;
  function setBusy(busy) {
    isBusy = !!busy;
    input.disabled = isBusy;
    if (sendBtn) sendBtn.disabled = isBusy;
    if (callBtn) callBtn.disabled = isBusy;
  }
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
    if (isBusy) return;
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
            let confidence = 0;
            try {
              const r = await fetch(`${apiBase}/rag/score`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: finalText }) });
              const rd = await r.json();
              confidence = typeof rd.confidence === "number" ? rd.confidence : 0;
            } catch (_) { confidence = 0; }
            const invoked = confidence > 0.5;
            renderMessage("user", finalText, { confidence, invoked });
            if (invoked) {
              showRagPopup(true);
              updateRagSidebar({
                title: "Retrieved Document",
                heading: "Relevant Passage",
                excerpt: finalText,
                points: ["Snippet generated from user utterance", "Confidence > 0.5", "RAG mocked"],
              });
            }
            setBusy(true);
            stopStreaming();
            const { answer } = await ask(finalText);
            renderMessage("assistant", answer);
            if (invoked) { showRagPopup(false); }
            setBusy(false);
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
    if (isBusy) return;
    if (!ws || ws.readyState !== 1) await startStreaming(); else stopStreaming();
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    if (isBusy) return;
    input.value = "";
    // Ask backend for confidence (mocked now, model later)
    let confidence = 0;
    try {
      const r = await fetch(`${apiBase}/rag/score`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: q }) });
      const rd = await r.json();
      confidence = typeof rd.confidence === "number" ? rd.confidence : 0;
    } catch (_) { confidence = 0; }
    const invoked = confidence > 0.5;
    renderMessage("user", q, { confidence, invoked });
    if (invoked) {
      showRagPopup(true);
      updateRagSidebar({
        title: "Retrieved Document",
        heading: "Relevant Passage",
        excerpt: q,
        points: ["Snippet generated from user message", "Confidence > 0.5", "RAG mocked"],
      });
    }
    try {
      setBusy(true);
      const { answer } = await ask(q);
      renderMessage("assistant", answer);
      if (invoked) { showRagPopup(false); }
      setBusy(false);
    } catch (err) {
      renderMessage("assistant", "Oops, something went wrong.");
      console.error(err);
      setBusy(false);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderMessage("assistant", "Hello! I'm your insurance assistant. How can I help?");
  setupChat();
});


