# InSession

A desktop app for therapists and coaches to record, transcribe, and query their client sessions — entirely on-device using local AI models.

InSession records audio from your microphone, transcribes it in real-time, stores the transcript, and lets you search across all sessions using natural language. No audio or text ever leaves your machine.

## Features

- 🎙️ **Real-time transcription** — live speech-to-text powered by NVIDIA Nemotron Speech
- 💾 **Session storage** — transcripts saved locally as JSON
- 🔍 **Semantic search** — query across all sessions using natural language (RAG with local embeddings and LLM)
- 📊 **Local diagnostics** — inspect process memory and model/RAG performance without sending telemetry off-device
- 🔒 **Fully local** — all AI inference runs on-device via [Foundry Local](https://github.com/microsoft/foundry-local)

## Models

| Purpose | Model |
|---|---|
| Speech recognition | `nemotron-speech-streaming-en-0.6b` |
| Embeddings (RAG) | `qwen3-embedding-0.6b` |
| Chat / Q&A | `qwen2.5-1.5b` |

Models are downloaded automatically on first launch via Foundry Local.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [Node.js](https://nodejs.org/) (v18+)

## Getting Started

```bash
npm install
npm run tauri dev
```

To build a distributable app:

```bash
npm run tauri build
```

## How It Works

1. **Record** — Enter a client name and click Record. The app captures your microphone and streams audio to the Nemotron speech model for live transcription.
2. **Stop** — Click Stop to end the session. The transcript is saved and automatically embedded for search.
3. **Search** — Switch to the Search tab and ask a question. The app finds the most relevant session excerpts and uses the local LLM to generate an answer with source citations.
4. **Diagnose** — Open the Diagnostics tab to view current memory, model load times, RAG latency, chat time to first token (TTFT), API-reported token usage and throughput, embedding timings, speech startup, and recent failures.

## Performance Instrumentation

Instrumentation is local and works through portable Rust and Tauri APIs on Windows, macOS, and Linux.

- **Memory** is the resident memory of the InSession process. Peak memory is the highest sample observed during the current app session.
- **TTFT** measures from the start of the streaming chat request to the first non-empty generated content.
- **Token usage** requests `stream_options: { include_usage: true }` and records the final stream chunk's prompt, completion, and total token counts. Tokens/second uses the reported completion count; if an older runtime omits usage, the UI falls back to a generated-text estimate and labels it as estimated.
- **RAG latency** includes query embedding, in-memory retrieval, chat model loading, and streamed generation. The individual stages are retained in the metric event.
- **Embedding and speech metrics** include model/session startup, inference durations, workload counts, and failures.

The Diagnostics tab retains the latest 100 completed operations and the chunks selected for the most recent question in memory for the current app session. Metric events are also appended as JSON Lines to:

```text
<Tauri app data directory>/logs/metrics.jsonl
```

The platform-specific app-data root is resolved by Tauri. JSONL metric events contain timings, counts, model aliases, statuses, and sanitized technical errors only. Prompts, transcripts, retrieved excerpts, generated answers, client/session names, and audio content are never written to the metrics log.

Cross-platform validation should include `npm run build`, Rust tests/checks, and a diagnostics snapshot plus JSONL write on Windows, macOS, and Linux.

## Tech Stack

- [Foundry Local SDK](https://github.com/microsoft/foundry-local) — local AI model management and inference
- [Tauri v2](https://tauri.app/) — Rust + WebView desktop app framework
- [cpal](https://github.com/RustAudio/cpal) — cross-platform audio capture
- TypeScript + Vite (frontend)

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## License

MIT — see [LICENSE](LICENSE).
