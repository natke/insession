import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Tab switching
function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => {
        c.classList.remove("active");
        c.classList.add("hidden");
      });
      tab.classList.add("active");
      const target = document.getElementById(`tab-${tab.getAttribute("data-tab")}`);
      if (target) {
        target.classList.remove("hidden");
        target.classList.add("active");
      }
    });
  });
}

// Recording state
let isRecording = false;

function initRecording() {
  const btnRecord = document.getElementById("btn-record") as HTMLButtonElement;
  const btnStop = document.getElementById("btn-stop") as HTMLButtonElement;
  const indicator = document.getElementById("recording-indicator")!;
  const output = document.getElementById("transcription-output")!;

  btnRecord.addEventListener("click", async () => {
    const clientName = (document.getElementById("client-name") as HTMLInputElement).value.trim()
      || "Client";
    const dateStr = new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    const sessionName = `${clientName} — ${dateStr}`;

    isRecording = true;
    btnRecord.disabled = true;
    btnStop.disabled = false;
    indicator.classList.remove("hidden");
    output.innerHTML = "";

    // Call Rust backend to start transcription
    try {
      await invoke("start_transcription", { sessionName });
      pollTranscription(output);
    } catch (e) {
      output.innerHTML = `<p class="placeholder">Error: ${e}</p>`;
    }
  });

  btnStop.addEventListener("click", async () => {
    isRecording = false;
    btnRecord.disabled = false;
    btnStop.disabled = true;
    indicator.classList.add("hidden");

    try {
      const sessionName = await invoke("stop_transcription") as string;
      if (sessionName) {
        output.innerHTML += `\n<p style="color:#4361ee; margin-top:0.5rem">✓ Session "${sessionName}" saved. Embedding for RAG...</p>`;
        // Auto-embed the session
        try {
          await invoke("embed_session", { sessionName });
          output.innerHTML += `<p style="color:#4361ee">✓ Embedded for search.</p>`;
        } catch (embErr) {
          output.innerHTML += `<p style="color:#e63946">⚠ Embedding error: ${embErr}</p>`;
        }
      }
      loadSessions();
    } catch (e) {
      console.error(e);
    }
  });
}

async function pollTranscription(output: HTMLElement) {
  while (isRecording) {
    try {
      const text = await invoke("get_transcription_buffer") as string;
      if (text) {
        output.textContent = text;
        output.scrollTop = output.scrollHeight;
      }
    } catch (_) { /* ignore */ }
    await new Promise((r) => setTimeout(r, 500));
  }
}

// Sessions list
async function loadSessions() {
  const list = document.getElementById("sessions-list")!;
  try {
    const sessions = await invoke("list_sessions") as Array<{ name: string; date: string; transcript: string }>;
    if (sessions.length === 0) {
      list.innerHTML = '<p class="placeholder">No sessions recorded yet.</p>';
      return;
    }
    list.innerHTML = sessions
      .map((s) => `
        <div class="session-item" data-session="${s.name}">
          <div class="session-header">
            <span class="name">${s.name}</span>
            <span class="date">${s.date}</span>
          </div>
          <div class="session-transcript" style="display:none;">
            <pre>${s.transcript}</pre>
          </div>
        </div>
      `)
      .join("");

    // Click to expand/collapse transcript
    list.querySelectorAll(".session-item").forEach((item) => {
      item.querySelector(".session-header")!.addEventListener("click", () => {
        const transcript = item.querySelector(".session-transcript") as HTMLElement;
        transcript.style.display = transcript.style.display === "none" ? "block" : "none";
      });
    });

    // Auto-embed sessions that haven't been embedded yet
    for (const s of sessions) {
      try {
        await invoke("embed_session", { sessionName: s.name });
      } catch (_) { /* already embedded or no transcript */ }
    }
  } catch (_) {
    list.innerHTML = '<p class="placeholder">No sessions recorded yet.</p>';
  }
}

// Query
let sessionsEmbedded = false;
function initQuery() {
  const form = document.getElementById("query-form") as HTMLFormElement;
  const input = document.getElementById("query-input") as HTMLInputElement;
  const results = document.getElementById("query-results")!;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Add a pending entry for this query
    const entry = document.createElement("div");
    entry.className = "query-entry";
    entry.innerHTML = `<div class="query-question">❓ ${query}</div><div class="query-answer placeholder">Searching...</div>`;
    results.prepend(entry);
    input.value = "";

    try {
      // Auto-embed if needed (only once per app session)
      if (!sessionsEmbedded) {
        const sessions = await invoke("list_sessions") as Array<{ name: string; transcript: string }>;
        for (const s of sessions) {
          if (s.transcript) {
            try { await invoke("embed_session", { sessionName: s.name }); } catch (_) {}
          }
        }
        sessionsEmbedded = true;
      }

      const answer = await invoke("query_sessions", { query }) as {
        answer: string;
        sources: string[];
      };
      entry.querySelector(".query-answer")!.innerHTML = `
        <p>${answer.answer}</p>
        ${answer.sources.length > 0
          ? `<div class="source">📄 Sources: ${answer.sources.join(", ")}</div>`
          : ""}
      `;
      entry.querySelector(".query-answer")!.classList.remove("placeholder");
    } catch (e) {
      entry.querySelector(".query-answer")!.innerHTML = `<p class="error">Error: ${e}</p>`;
      entry.querySelector(".query-answer")!.classList.remove("placeholder");
    }
  });
}

// Init
window.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initRecording();
  initQuery();
  loadSessions();

  // Model download/loading status — displayed in dedicated panel
  const modelPanel = document.getElementById("model-status")!;
  await listen<{ model: string; progress: number; status: string }>("model-download", (event) => {
    const { model, progress, status } = event.payload;

    modelPanel.classList.add("visible");

    const lineId = `dl-${model.replace(/[^a-z0-9]/gi, "-")}`;
    let line = document.getElementById(lineId);
    if (!line) {
      line = document.createElement("p");
      line.id = lineId;
      modelPanel.appendChild(line);
    }

    if (status === "downloading") {
      const pct = progress.toFixed(1);
      const filled = Math.round(progress / 5);
      const bar = "█".repeat(filled) + "░".repeat(20 - filled);
      line.textContent = `⬇ Downloading ${model}: ${bar} ${pct}%`;
    } else if (status === "loading") {
      line.textContent = `⏳ Loading ${model}...`;
    } else if (status === "checking") {
      line.textContent = `🔍 Checking ${model}...`;
    } else if (status === "starting") {
      line.textContent = `⬇ Starting download: ${model}...`;
    } else if (status === "complete") {
      line.innerHTML = `<span style="font-size:1.1em">✓</span> ${model} ready`;
      // Hide panel after a short delay if all models are complete
      setTimeout(() => {
        const allDone = modelPanel.querySelectorAll("p");
        const allComplete = Array.from(allDone).every(p => p.innerHTML.includes("ready"));
        if (allComplete) modelPanel.classList.remove("visible");
      }, 2000);
    }
  });

  // Initialize Foundry Local SDK (downloads models if needed)
  modelPanel.classList.add("visible");
  const initLine = document.createElement("p");
  initLine.id = "dl-foundry-init";
  initLine.textContent = "⏳ Initializing Foundry Local...";
  modelPanel.appendChild(initLine);
  try {
    const msg = await invoke("init_foundry") as string;
    initLine.innerHTML = `<span style="font-size:1.1em">✓</span> ${msg}`;
    setTimeout(() => {
      const allDone = modelPanel.querySelectorAll("p");
      const allComplete = Array.from(allDone).every(p => p.innerHTML.includes("✓"));
      if (allComplete) modelPanel.classList.remove("visible");
    }, 2000);
  } catch (e) {
    initLine.textContent = `⚠ Init error: ${e}`;
  }
});
