const statusEl = document.querySelector("#status");
const messagesEl = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const healthButton = document.querySelector("#healthButton");

function appendMessage(role, text, meta = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  article.appendChild(bubble);
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function refreshHealth() {
  statusEl.textContent = "检查本地模型服务...";
  try {
    const response = await fetch("/health");
    const data = await response.json();
    statusEl.textContent = `网关：${data.status} / Ollama：${data.ollama} / 模型：${data.model}`;
  } catch (error) {
    statusEl.textContent = `网关状态检查失败：${error}`;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  appendMessage("user", message);
  messageInput.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "生成中";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const model = data.model ? ` / ${data.model}` : "";
    appendMessage("assistant", data.answer, `${data.route}${model} / ${data.latency_ms}ms`);
  } catch (error) {
    appendMessage("assistant", `请求失败：${error.message}`);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "发送";
    messageInput.focus();
  }
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

healthButton.addEventListener("click", refreshHealth);
refreshHealth();
