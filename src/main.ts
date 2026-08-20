import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface StageMetric {
  name: string;
  duration_ms: number;
}

interface OperationMetric {
  operation_id: string;
  kind: "embedding" | "rag_query" | "speech_transcription";
  model?: string;
  started_at: string;
  duration_ms: number;
  success: boolean;
  error?: string;
  stages: StageMetric[];
  measurements: Record<string, number>;
  counts: Record<string, number>;
}

interface MetricsSnapshot {
  process?: {
    sampled_at: string;
    resident_bytes: number;
    resident_mib: number;
    peak_resident_bytes: number;
    peak_resident_mib: number;
  };
  in_memory_chunk_count: number;
  transcription_count: number;
  latest_retrieved_chunks: {
    rank: number;
    session_name: string;
    score: number;
    text: string;
  }[];
  recent_operations: OperationMetric[];
  diagnostics_error?: string;
}

let diagnosticsTimer: number | undefined;

function setDiagnosticsPolling(active: boolean) {
  if (diagnosticsTimer !== undefined) {
    window.clearInterval(diagnosticsTimer);
    diagnosticsTimer = undefined;
  }
  if (active) {
    void loadMetrics();
    diagnosticsTimer = window.setInterval(() => void loadMetrics(), 2000);
  }
}

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
      setDiagnosticsPolling(tab.getAttribute("data-tab") === "diagnostics");
    });
  });
}

// Recording state
let isRecording = false;
let recordingTimer: number | undefined;

function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  const minuteSeconds = `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
  return hours > 0 ? `${hours}:${minuteSeconds}` : minuteSeconds;
}

function startRecordingTimer() {
  const timer = document.getElementById("recording-timer")!;
  const startedAt = performance.now();
  timer.textContent = "00:00";
  recordingTimer = window.setInterval(() => {
    timer.textContent = formatDuration((performance.now() - startedAt) / 1000);
  }, 250);
}

function stopRecordingTimer() {
  if (recordingTimer !== undefined) {
    window.clearInterval(recordingTimer);
    recordingTimer = undefined;
  }
}

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

    btnRecord.disabled = true;
    btnStop.disabled = true;
    output.innerHTML = "";

    // Call Rust backend to start transcription
    try {
      await invoke("start_transcription", { sessionName });
      isRecording = true;
      btnStop.disabled = false;
      indicator.classList.remove("hidden");
      startRecordingTimer();
      pollTranscription(output);
    } catch (e) {
      btnRecord.disabled = false;
      output.innerHTML = `<p class="placeholder">Error: ${e}</p>`;
    }
  });

  btnStop.addEventListener("click", async () => {
    isRecording = false;
    btnRecord.disabled = false;
    btnStop.disabled = true;
    indicator.classList.add("hidden");
    stopRecordingTimer();

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
    const sessions = await invoke("list_sessions") as Array<{
      name: string;
      date: string;
      transcript: string;
      duration_seconds: number;
    }>;
    if (sessions.length === 0) {
      list.innerHTML = '<p class="placeholder">No sessions recorded yet.</p>';
      return;
    }
    list.innerHTML = sessions
      .map((s) => `
        <div class="session-item" data-session="${s.name}">
          <div class="session-header">
            <span class="name">${s.name}</span>
            <span class="session-meta">
              <span class="duration">${s.duration_seconds > 0 ? formatDuration(s.duration_seconds) : "Duration unavailable"}</span>
              <span class="date">${s.date}</span>
            </span>
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

function formatMilliseconds(value: number | undefined): string {
  if (value === undefined) return "—";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function operationLabel(kind: OperationMetric["kind"]): string {
  switch (kind) {
    case "rag_query": return "RAG query";
    case "speech_transcription": return "Speech";
    case "embedding": return "Embedding";
  }
}

function setText(id: string, value: string) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderMetrics(snapshot: MetricsSnapshot) {
  setText("diagnostics-status", `Updated ${new Date().toLocaleTimeString()}`);

  if (snapshot.process) {
    setText("metric-memory", `${snapshot.process.resident_mib.toFixed(1)} MiB`);
    setText("metric-memory-peak", `Peak ${snapshot.process.peak_resident_mib.toFixed(1)} MiB`);
  } else {
    setText("metric-memory", "Unavailable");
    setText("metric-memory-peak", "Peak unavailable");
  }
  setText("metric-total-chunks", snapshot.in_memory_chunk_count.toLocaleString());
  setText("metric-total-transcriptions", snapshot.transcription_count.toLocaleString());

  const operations = snapshot.recent_operations;
  const latestRag = operations.find((operation) => operation.kind === "rag_query" && operation.success);
  const latestEmbedding = operations.find((operation) => operation.kind === "embedding");
  const latestSpeech = operations.find((operation) => operation.kind === "speech_transcription");
  const latestModelLoad = operations.find((operation) =>
    operation.stages.some((stage) => stage.name.includes("model_load"))
  );
  const failures = operations.filter((operation) => !operation.success);

  setText("metric-ttft", formatMilliseconds(latestRag?.measurements.time_to_first_token_ms));
  const exactThroughput = latestRag?.measurements.tokens_per_second;
  const estimatedThroughput = latestRag?.measurements.estimated_tokens_per_second;
  const throughput = exactThroughput ?? estimatedThroughput;
  setText("metric-throughput", throughput === undefined ? "—" : `${throughput.toFixed(1)} tok/s`);
  setText(
    "metric-throughput-detail",
    throughput === undefined
      ? "No chat completion yet"
      : exactThroughput === undefined
        ? "Estimated tokens/second"
        : "API-reported tokens/second"
  );
  const promptTokens = latestRag?.counts.prompt_tokens;
  const completionTokens = latestRag?.counts.completion_tokens;
  const totalTokens = latestRag?.counts.total_tokens;
  setText(
    "metric-token-usage",
    promptTokens === undefined || completionTokens === undefined
      ? "—"
      : `${promptTokens.toLocaleString()} in / ${completionTokens.toLocaleString()} out`
  );
  setText(
    "metric-token-total",
    totalTokens === undefined
      ? "API usage unavailable"
      : `${totalTokens.toLocaleString()} total tokens`
  );
  setText("metric-rag-latency", formatMilliseconds(latestRag?.duration_ms));

  const loadStage = latestModelLoad?.stages
    .slice()
    .reverse()
    .find((stage) => stage.name.includes("model_load"));
  setText("metric-model-load", formatMilliseconds(loadStage?.duration_ms));
  setText("metric-model-name", latestModelLoad?.model ?? "No model operation yet");

  setText("metric-embedding", formatMilliseconds(latestEmbedding?.duration_ms));
  setText(
    "metric-embedding-detail",
    latestEmbedding
      ? `${latestEmbedding.counts.chunk_count ?? 0} chunks`
      : "No embedding operation yet"
  );

  setText(
    "metric-speech",
    formatMilliseconds(latestSpeech?.measurements.time_to_first_result_ms)
  );
  setText(
    "metric-speech-detail",
    latestSpeech
      ? `${latestSpeech.counts.transcription_result_count ?? 0} results · ${formatMilliseconds(latestSpeech.duration_ms)} total`
      : "No transcription operation yet"
  );
  setText("metric-failures", failures.length.toString());

  const retrievedChunks = document.getElementById("diagnostics-retrieved-chunks")!;
  retrievedChunks.replaceChildren();
  if (snapshot.latest_retrieved_chunks.length === 0) {
    const placeholder = document.createElement("p");
    placeholder.className = "placeholder";
    placeholder.textContent = "Ask a question to see the chunks selected for its answer.";
    retrievedChunks.appendChild(placeholder);
  } else {
    for (const chunk of snapshot.latest_retrieved_chunks) {
      const item = document.createElement("article");
      item.className = "retrieved-chunk";

      const heading = document.createElement("div");
      heading.className = "retrieved-chunk-heading";
      const title = document.createElement("strong");
      title.textContent = `${chunk.rank}. ${chunk.session_name}`;
      const score = document.createElement("span");
      score.textContent = `Score ${chunk.score.toFixed(3)}`;
      heading.append(title, score);

      const text = document.createElement("p");
      text.textContent = chunk.text;
      item.append(heading, text);
      retrievedChunks.appendChild(item);
    }
  }

  const diagnosticsError = document.getElementById("diagnostics-error")!;
  if (snapshot.diagnostics_error) {
    diagnosticsError.textContent = snapshot.diagnostics_error;
    diagnosticsError.classList.remove("hidden");
  } else {
    diagnosticsError.textContent = "";
    diagnosticsError.classList.add("hidden");
  }

  const recent = document.getElementById("metrics-recent")!;
  recent.replaceChildren();
  if (operations.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "placeholder";
    cell.textContent = "No instrumented operations yet.";
    row.appendChild(cell);
    recent.appendChild(row);
    return;
  }

  for (const operation of operations.slice(0, 10)) {
    const row = document.createElement("tr");
    const values = [
      operationLabel(operation.kind),
      operation.model ?? "—",
      formatMilliseconds(operation.duration_ms),
      operation.success ? "Success" : `Failed: ${operation.error ?? "Unknown error"}`,
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    row.classList.toggle("metric-failed", !operation.success);
    recent.appendChild(row);
  }
}

async function loadMetrics() {
  try {
    const snapshot = await invoke<MetricsSnapshot>("get_metrics_snapshot");
    renderMetrics(snapshot);
  } catch (error) {
    setText("diagnostics-status", "Metrics unavailable");
    const diagnosticsError = document.getElementById("diagnostics-error");
    if (diagnosticsError) {
      diagnosticsError.textContent = String(error);
      diagnosticsError.classList.remove("hidden");
    }
  }
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
