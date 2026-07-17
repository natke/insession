# InSession

A desktop app for therapists and coaches to record, transcribe, and query their client sessions — entirely on-device using local AI models.

InSession records audio from your microphone, transcribes it in real-time, stores the transcript, and lets you search across all sessions using natural language. No audio or text ever leaves your machine.

## Features

- 🎙️ **Real-time transcription** — live speech-to-text powered by NVIDIA Nemotron Speech
- 💾 **Session storage** — transcripts saved locally as JSON
- 🔍 **Semantic search** — query across all sessions using natural language (RAG with local embeddings and LLM)
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

## Tech Stack

- [Foundry Local SDK](https://github.com/microsoft/foundry-local) — local AI model management and inference
- [Tauri v2](https://tauri.app/) — Rust + WebView desktop app framework
- [cpal](https://github.com/RustAudio/cpal) — cross-platform audio capture
- TypeScript + Vite (frontend)

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## License

MIT — see [LICENSE](LICENSE).
