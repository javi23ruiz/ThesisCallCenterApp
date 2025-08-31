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


