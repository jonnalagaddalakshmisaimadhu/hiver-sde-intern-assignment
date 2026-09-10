/**
 * @AppleSupport AI Support Agent - Frontend Controller
 * Vanilla JS connection to FastAPI /api/inquire backend
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const messagesList = document.getElementById("messages-list");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  
  // Modal Elements
  const infoBtn = document.getElementById("info-btn");
  const infoModal = document.getElementById("info-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalDismissBtn = document.getElementById("modal-dismiss-btn");

  // Telemetry Elements
  const telemetryToggleBtn = document.getElementById("telemetry-toggle-btn");
  const telemetryPane = document.getElementById("telemetry-pane");
  const telemetryIntent = document.getElementById("telemetry-intent");
  const telemetryConfBar = document.getElementById("telemetry-confidence-bar");
  const telemetryConfText = document.getElementById("telemetry-confidence-text");
  const telemetryDecision = document.getElementById("telemetry-decision");
  const telemetrySimilarity = document.getElementById("telemetry-similarity");
  const telemetryLatency = document.getElementById("telemetry-latency");
  const taxonomyTags = document.querySelectorAll(".tax-tag");

  // Mobile Telemetry Sidebar Toggle
  if (telemetryToggleBtn && telemetryPane) {
    telemetryToggleBtn.addEventListener("click", () => {
      const isOpen = telemetryPane.classList.toggle("mobile-open");
      telemetryToggleBtn.classList.toggle("active", isOpen);
    });
  }

  // Auto-resize textarea
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
  });

  // Enter to send (Shift+Enter for newline)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Handle Quick Sample Chips
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      if (q) {
        userInput.value = q;
        userInput.focus();
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Clear Chat Action
  clearChatBtn.addEventListener("click", () => {
    const welcome = document.getElementById("welcome-message");
    messagesList.innerHTML = "";
    if (welcome) {
      messagesList.appendChild(welcome);
    }
    resetTelemetry();
  });

  // Modal Controls ('i' Symbol)
  function openModal() {
    infoModal.classList.add("open");
    infoModal.setAttribute("aria-hidden", "false");
  }

  function closeModal() {
    infoModal.classList.remove("open");
    infoModal.setAttribute("aria-hidden", "true");
  }

  infoBtn.addEventListener("click", openModal);
  modalCloseBtn.addEventListener("click", closeModal);
  modalDismissBtn.addEventListener("click", closeModal);

  // Close modal when clicking backdrop outside card
  infoModal.addEventListener("click", (e) => {
    if (e.target === infoModal) {
      closeModal();
    }
  });

  // ESC to close modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && infoModal.classList.contains("open")) {
      closeModal();
    }
  });

  // Submit Query to Backend
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Append User Message Bubble
    appendUserMessage(query);

    // Reset Input Bar
    userInput.value = "";
    userInput.style.height = "auto";
    setLoading(true);

    // Append Temporary Typing Indicator
    const typingIndicator = appendTypingIndicator();
    scrollToBottom();

    try {
      const response = await fetch("/api/inquire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();

      // Remove typing indicator and render agent reply
      typingIndicator.remove();
      appendAgentResponse(data);
      updateTelemetry(data);

    } catch (err) {
      console.error("Inquiry error:", err);
      typingIndicator.remove();
      appendErrorMessage(`Error: Could not reach agent server (${err.message}). Is the backend running on port 8000?`);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  });

  function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    if (isLoading) {
      sendBtn.classList.add("loading");
    } else {
      sendBtn.classList.remove("loading");
      userInput.focus();
    }
  }

  function appendUserMessage(text) {
    const card = document.createElement("div");
    card.className = "message-card user-message";
    card.innerHTML = `
      <div class="message-avatar">
        <div class="avatar-user">You</div>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Customer</span>
          <span class="message-timestamp">${getTimeString()}</span>
        </div>
        <p class="message-text">${escapeHtml(text)}</p>
      </div>
    `;
    messagesList.appendChild(card);
  }

  function appendTypingIndicator() {
    const card = document.createElement("div");
    card.className = "message-card agent-message typing-indicator-card";
    card.innerHTML = `
      <div class="message-avatar">
        <div class="avatar-apple"></div>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Apple Support Assistant</span>
          <span class="message-timestamp">Analyzing inquiry...</span>
        </div>
        <p class="message-text" style="color: var(--text-muted); font-style: italic;">
          Classifying intent & retrieving historical evidence...
        </p>
      </div>
    `;
    messagesList.appendChild(card);
    return card;
  }

  function appendAgentResponse(data) {
    const card = document.createElement("div");
    card.className = "message-card agent-message";

    const isAuto = data.decision === "AUTO_HANDLE";
    const badgeClass = isAuto ? "meta-auto-handle" : "meta-escalate";
    const badgeText = isAuto ? "✓ AUTO_HANDLE" : "🚨 ESCALATE TO HUMAN";
    const rationaleClass = isAuto ? "" : "escalate-rationale";

    // Evidence Accordion HTML
    let evidenceHtml = "";
    if (data.evidence && data.evidence.length > 0) {
      const topSim = data.evidence[0].similarity;
      evidenceHtml = `
        <div class="evidence-accordion">
          <button type="button" class="evidence-toggle-btn" aria-expanded="false">
            <span>🔍 Grounding Evidence (${data.evidence.length} historical matches, Top Sim: ${(topSim).toFixed(3)})</span>
            <span class="chevron">▼</span>
          </button>
          <div class="evidence-drawer">
            ${data.evidence.map((ev, idx) => `
              <div class="evidence-item">
                <span class="evidence-sim-tag">[Match ${idx + 1} &bull; Similarity ${(ev.similarity).toFixed(3)}]</span>
                <span class="evidence-inbound"><strong>Customer:</strong> "${escapeHtml(ev.customer_message)}"</span>
                <span class="evidence-resolution"><strong>Resolution:</strong> "${escapeHtml(ev.brand_response)}"</span>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="message-avatar">
        <div class="avatar-apple"></div>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Apple Support Assistant</span>
          <span class="message-timestamp">${getTimeString()}</span>
        </div>
        <p class="message-text">${escapeHtml(data.reply)}</p>
        
        <div class="rationale-box ${rationaleClass}">
          <strong>${badgeText}:</strong> ${escapeHtml(data.reason)}
        </div>

        <div class="agent-meta-bar">
          <span class="meta-pill meta-intent">🎯 ${data.intent} (${Math.round(data.intent_confidence * 100)}%)</span>
          <span class="meta-pill ${badgeClass}">${badgeText}</span>
          <span class="meta-latency">⚡ ${data.latency_ms} ms</span>
        </div>

        ${evidenceHtml}
      </div>
    `;

    // Attach accordion toggle behavior
    const toggleBtn = card.querySelector(".evidence-toggle-btn");
    const drawer = card.querySelector(".evidence-drawer");
    if (toggleBtn && drawer) {
      toggleBtn.addEventListener("click", () => {
        const isOpen = drawer.classList.contains("open");
        if (isOpen) {
          drawer.classList.remove("open");
          toggleBtn.setAttribute("aria-expanded", "false");
          toggleBtn.querySelector(".chevron").textContent = "▼";
        } else {
          drawer.classList.add("open");
          toggleBtn.setAttribute("aria-expanded", "true");
          toggleBtn.querySelector(".chevron").textContent = "▲";
        }
      });
    }

    messagesList.appendChild(card);
  }

  function appendErrorMessage(errText) {
    const card = document.createElement("div");
    card.className = "message-card agent-message";
    card.innerHTML = `
      <div class="message-avatar">
        <div class="avatar-apple" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">!</div>
      </div>
      <div class="message-content" style="border-color: rgba(239, 68, 68, 0.4);">
        <p class="message-text" style="color: #fca5a5;">${escapeHtml(errText)}</p>
      </div>
    `;
    messagesList.appendChild(card);
  }

  function updateTelemetry(data) {
    // 1. Update Intent & Confidence
    telemetryIntent.textContent = data.intent;
    const confPct = Math.round(data.intent_confidence * 100);
    telemetryConfBar.style.width = `${confPct}%`;
    telemetryConfText.textContent = `${confPct}%`;

    // 2. Update Decision Badge
    if (data.decision === "AUTO_HANDLE") {
      telemetryDecision.textContent = "AUTO_HANDLE";
      telemetryDecision.className = "badge-status-auto";
    } else {
      telemetryDecision.textContent = "ESCALATE";
      telemetryDecision.className = "badge-status-escalate";
    }

    // 3. Update Similarity & Latency
    const topSim = (data.evidence && data.evidence.length > 0) ? data.evidence[0].similarity.toFixed(3) : "0.000";
    telemetrySimilarity.textContent = topSim;
    telemetryLatency.textContent = `${data.latency_ms} ms`;

    // 4. Highlight Active Taxonomy Tag
    taxonomyTags.forEach((tag) => {
      if (tag.getAttribute("data-intent") === data.intent) {
        tag.classList.add("active");
      } else {
        tag.classList.remove("active");
      }
    });
  }

  function resetTelemetry() {
    telemetryIntent.textContent = "Standby";
    telemetryConfBar.style.width = "0%";
    telemetryConfText.textContent = "--%";
    telemetryDecision.textContent = "Standby";
    telemetryDecision.className = "badge-status-neutral";
    telemetrySimilarity.textContent = "--";
    telemetryLatency.textContent = "-- ms";
    taxonomyTags.forEach((t) => t.classList.remove("active"));
  }

  function scrollToBottom() {
    messagesList.scrollTop = messagesList.scrollHeight;
  }

  function getTimeString() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
