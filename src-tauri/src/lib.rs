use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tauri::{AppHandle, Emitter, Manager, State};
use foundry_local_sdk::{
    FoundryLocalConfig, FoundryLocalManager,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
};
use tokio_stream::StreamExt;

const SPEECH_MODEL: &str = "nemotron-speech-streaming-en-0.6b";
const EMBEDDING_MODEL: &str = "qwen3-embedding-0.6b";
const CHAT_MODEL: &str = "qwen2.5-1.5b";

struct AppState {
    is_recording: Arc<AtomicBool>,
    current_session: String,
    transcription_buffer: String,
    sessions: Vec<Session>,
    // Shared audio sample buffer (f32 PCM, 16kHz mono)
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    // Full session audio accumulator
    session_audio: Vec<f32>,
    // Handle to stop the capture thread
    capture_handle: Option<std::thread::JoinHandle<()>>,
    // Foundry Local manager (initialized once — static singleton)
    manager: Option<&'static FoundryLocalManager>,
    // Live transcription: audio sender and accumulated transcript
    audio_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
    live_transcript: Arc<Mutex<String>>,
    transcript_task: Option<tokio::task::JoinHandle<()>>,
    forward_task: Option<tokio::task::JoinHandle<()>>,
    // Embedding store for RAG
    embeddings: Vec<EmbeddingEntry>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Session {
    name: String,
    date: String,
    transcript: String,
}

#[derive(Clone)]
struct EmbeddingEntry {
    session_name: String,
    chunk: String,
    embedding: Vec<f32>,
}

#[derive(Clone, serde::Serialize)]
struct QueryResult {
    answer: String,
    sources: Vec<String>,
}

/// Get the sessions directory path (inside app data)
fn sessions_dir(app_handle: &AppHandle) -> Result<std::path::PathBuf, String> {
    let app_data = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {e}"))?;
    let dir = app_data.join("sessions");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create sessions dir: {e}"))?;
    Ok(dir)
}

/// Save a session to disk as JSON
fn save_session_to_disk(app_handle: &AppHandle, session: &Session) -> Result<(), String> {
    let dir = sessions_dir(app_handle)?;
    let slug = session.name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' })
        .collect::<String>();
    let path = dir.join(format!("{}.json", slug));
    let json = serde_json::to_string_pretty(session)
        .map_err(|e| format!("Serialize failed: {e}"))?;
    // Atomic write: write to temp then rename
    let tmp = dir.join(format!(".{}.tmp", slug));
    std::fs::write(&tmp, &json).map_err(|e| format!("Write failed: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("Rename failed: {e}"))?;
    eprintln!("[persist] Saved session to {}", path.display());
    Ok(())
}

/// Load all sessions from disk
fn load_sessions_from_disk(app_handle: &AppHandle) -> Vec<Session> {
    let dir = match sessions_dir(app_handle) {
        Ok(d) => d,
        Err(e) => { eprintln!("[persist] {}", e); return vec![]; }
    };
    let mut sessions = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            match std::fs::read_to_string(&path) {
                Ok(content) => match serde_json::from_str::<Session>(&content) {
                    Ok(session) => sessions.push(session),
                    Err(e) => eprintln!("[persist] Skipping {}: {}", path.display(), e),
                },
                Err(e) => eprintln!("[persist] Read error {}: {}", path.display(), e),
            }
        }
    }
    sessions.sort_by(|a, b| a.date.cmp(&b.date));
    eprintln!("[persist] Loaded {} sessions from disk", sessions.len());
    sessions
}

/// Copy seed sessions to app data on first run
fn seed_sessions_if_needed(app_handle: &AppHandle) {
    let dir = match sessions_dir(app_handle) {
        Ok(d) => d,
        Err(_) => return,
    };
    let marker = dir.join(".seeded");
    if marker.exists() {
        return;
    }
    // Resolve bundled seed-sessions from the resource directory
    let resource_dir = match app_handle.path().resource_dir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let seed_dir = resource_dir.join("seed-sessions");
    if !seed_dir.exists() {
        eprintln!("[seed] No seed-sessions dir at {}", seed_dir.display());
        return;
    }
    if let Ok(entries) = std::fs::read_dir(&seed_dir) {
        for entry in entries.flatten() {
            let src = entry.path();
            if src.extension().map(|e| e == "json").unwrap_or(false) {
                let dest = dir.join(entry.file_name());
                if !dest.exists() {
                    if let Err(e) = std::fs::copy(&src, &dest) {
                        eprintln!("[seed] Failed to copy {}: {}", src.display(), e);
                    } else {
                        eprintln!("[seed] Copied {}", entry.file_name().to_string_lossy());
                    }
                }
            }
        }
    }
    // Write marker
    let _ = std::fs::write(&marker, "seeded");
    eprintln!("[seed] Seeding complete");
}

/// Initialize the Foundry Local manager
#[tauri::command]
async fn init_foundry(app_handle: AppHandle, state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let has_manager = {
        let app = state.lock().map_err(|e| e.to_string())?;
        app.manager.is_some()
    };

    if has_manager {
        return Ok("Already initialized".into());
    }

    // Use app support dir for all SDK data (avoid Documents folder prompt)
    let app_data = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {e}"))?;
    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&app_data).ok();
    let app_data_str = app_data.to_string_lossy().to_string();
    let cache_dir = app_data.join("models").to_string_lossy().to_string();
    let logs_dir = app_data.join("logs").to_string_lossy().to_string();

    let manager = FoundryLocalManager::create(
        FoundryLocalConfig::new("insession")
            .app_data_dir(&app_data_str)
            .model_cache_dir(&cache_dir)
            .logs_dir(&logs_dir)
            .additional_setting("AzureCatalogFilter", "'',test")
    ).map_err(|e| format!("Failed to create manager: {e}"))?;

    // Discover and register GPU execution providers (WebGPU etc.)
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": "GPU acceleration", "progress": 0.0, "status": "checking"
    }));
    let eps = manager.discover_eps().unwrap_or_default();
    let ep_names: Vec<String> = eps.iter().map(|ep| format!("{} (registered={})", ep.name, ep.is_registered)).collect();
    eprintln!("[init] Available EPs: {:?}", ep_names);

    let unregistered: Vec<&str> = eps.iter()
        .filter(|ep| !ep.is_registered)
        .map(|ep| ep.name.as_str())
        .collect();

    if !unregistered.is_empty() {
        eprintln!("[init] Downloading GPU acceleration: {:?}", unregistered);
        let handle = app_handle.clone();
        let result = manager.download_and_register_eps_with_progress(
            Some(&unregistered),
            move |_ep_name: &str, percent: f64| {
                let _ = handle.emit("model-download", serde_json::json!({
                    "model": "GPU acceleration", "progress": percent, "status": "downloading"
                }));
            }
        ).await.map_err(|e| format!("EP registration failed: {e}"))?;
        eprintln!("[init] GPU acceleration: registered={:?}, failed={:?}", result.registered_eps, result.failed_eps);
        let _ = app_handle.emit("model-download", serde_json::json!({
            "model": "GPU acceleration", "progress": 100.0, "status": "complete"
        }));
    } else {
        eprintln!("[init] All EPs already registered");
        let _ = app_handle.emit("model-download", serde_json::json!({
            "model": "GPU acceleration", "progress": 100.0, "status": "complete"
        }));
    }

    // Download the speech model if not cached
    let model = manager.catalog().get_model(SPEECH_MODEL).await
        .map_err(|e| format!("Speech model not found: {e}"))?;

    if !model.is_cached().await.unwrap_or(false) {
        let handle = app_handle.clone();
        let model_name = SPEECH_MODEL.to_string();
        let _ = handle.emit("model-download", serde_json::json!({
            "model": model_name, "progress": 0.0, "status": "starting"
        }));
        model.download(Some(move |progress: f64| {
            let _ = handle.emit("model-download", serde_json::json!({
                "model": model_name, "progress": progress, "status": "downloading"
            }));
        })).await.map_err(|e| format!("Download failed: {e}"))?;
        let _ = app_handle.emit("model-download", serde_json::json!({
            "model": SPEECH_MODEL, "progress": 100.0, "status": "complete"
        }));
    }

    {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.manager = Some(manager);
    }

    Ok("Foundry Local initialized".into())
}

/// Convert f32 audio samples to 16-bit LE PCM bytes for the transcription session
fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Start mic capture and live transcription session
#[tauri::command]
async fn start_transcription(session_name: String, state: State<'_, Mutex<AppState>>) -> Result<(), String> {
    let (recording_flag, audio_buf, manager, live_transcript) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        if app.is_recording.load(Ordering::SeqCst) {
            return Err("Already recording".into());
        }

        app.current_session = session_name;
        app.transcription_buffer = String::new();
        app.session_audio.clear();
        *app.live_transcript.lock().unwrap() = String::new();

        let manager = app.manager.clone()
            .ok_or("Foundry not initialized — call init_foundry first")?;

        (
            app.is_recording.clone(),
            app.audio_buffer.clone(),
            manager,
            app.live_transcript.clone(),
        )
    };

    // Load the speech model
    let model = manager.catalog().get_model(SPEECH_MODEL).await
        .map_err(|e| format!("Model error: {e}"))?;
    model.load().await.map_err(|e| format!("Load failed: {e}"))?;

    // Create live transcription session
    let audio_client = model.create_audio_client();
    let session = Arc::new(audio_client.create_live_transcription_session());
    session.start(None).await.map_err(|e| format!("Session start failed: {e}"))?;

    // Channel for sending PCM audio to the transcription session
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(100);

    // Task: forward audio chunks to transcription session
    let session_fwd = Arc::clone(&session);
    let forward_task = tokio::spawn(async move {
        let mut chunk_count = 0u64;
        let mut total_bytes = 0u64;
        while let Some(bytes) = audio_rx.recv().await {
            total_bytes += bytes.len() as u64;
            chunk_count += 1;
            if chunk_count % 100 == 1 {
                eprintln!("[audio] chunk #{chunk_count}, total {total_bytes} bytes sent to transcriber");
            }
            if let Err(e) = session_fwd.append(&bytes, None).await {
                eprintln!("Append error: {e}");
                break;
            }
        }
        eprintln!("[audio] forwarding stopped after {chunk_count} chunks, {total_bytes} bytes total");
    });

    // Task: read transcription results
    let transcript_ref = live_transcript.clone();
    let mut stream = session.get_stream().await
        .map_err(|e| format!("Stream error: {e}"))?;
    let transcript_task = tokio::spawn(async move {
        eprintln!("[transcribe] listening for results...");
        while let Some(result) = stream.next().await {
            match result {
                Ok(r) => {
                    if let Some(content) = r.content.first() {
                        let text = &content.text;
                        eprintln!("[transcribe] got text: '{}' (final={})", text, r.is_final);
                        if !text.is_empty() {
                            if let Ok(mut t) = transcript_ref.lock() {
                                t.push_str(text);
                                if r.is_final {
                                    t.push('\n');
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Transcription error: {e}");
                    break;
                }
            }
        }
        eprintln!("[transcribe] stream ended");
    });

    // Spawn audio capture on a dedicated thread (cpal::Stream isn't Send)
    let recording_flag_thread = recording_flag.clone();
    let audio_buf_thread = audio_buf.clone();
    let audio_tx_thread = audio_tx.clone();
    let handle = std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => { eprintln!("No input device"); return; }
        };

        // Use the device's default config (macOS typically 48kHz stereo)
        let default_config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => { eprintln!("No default input config: {}", e); return; }
        };
        let native_rate = default_config.sample_rate().0;
        let native_channels = default_config.channels() as usize;
        eprintln!("Audio input: {}Hz, {} channels", native_rate, native_channels);

        let config = cpal::StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let buf = audio_buf_thread.clone();
        let tx = audio_tx_thread.clone();
        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix to mono if stereo
                let mono: Vec<f32> = if native_channels > 1 {
                    data.chunks(native_channels)
                        .map(|frame| frame.iter().sum::<f32>() / native_channels as f32)
                        .collect()
                } else {
                    data.to_vec()
                };

                // Downsample to 16kHz if needed
                let target_rate = 16000u32;
                let resampled: Vec<f32> = if native_rate != target_rate {
                    let ratio = native_rate as f64 / target_rate as f64;
                    let out_len = (mono.len() as f64 / ratio) as usize;
                    (0..out_len)
                        .map(|i| {
                            let src_idx = i as f64 * ratio;
                            let idx = src_idx as usize;
                            let frac = src_idx - idx as f64;
                            if idx + 1 < mono.len() {
                                mono[idx] * (1.0 - frac as f32) + mono[idx + 1] * frac as f32
                            } else if idx < mono.len() {
                                mono[idx]
                            } else {
                                0.0
                            }
                        })
                        .collect()
                } else {
                    mono.clone()
                };

                // Store resampled f32 for level metering
                if let Ok(mut b) = buf.lock() {
                    b.extend_from_slice(&resampled);
                }
                // Convert to PCM16 and send to transcription
                let pcm_bytes = f32_to_pcm16_bytes(&resampled);
                let _ = tx.try_send(pcm_bytes);
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        ) {
            Ok(s) => s,
            Err(e) => { eprintln!("Stream build failed: {}", e); return; }
        };

        if let Err(e) = stream.play() {
            eprintln!("Stream play failed: {}", e);
            return;
        }

        while recording_flag_thread.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });

    // Store handles and set recording flag
    {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.is_recording.store(true, Ordering::SeqCst);
        app.capture_handle = Some(handle);
        app.audio_tx = Some(audio_tx);
        app.transcript_task = Some(transcript_task);
        app.forward_task = Some(forward_task);
        app.transcription_buffer = "🎙️ Recording... (live transcription active)\n".into();
    }

    Ok(())
}

/// Stop recording and save session with transcript
#[tauri::command]
async fn stop_transcription(app_handle: AppHandle, state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let (recording_flag, manager) = {
        let app = state.lock().map_err(|e| e.to_string())?;
        if !app.is_recording.load(Ordering::SeqCst) {
            return Err("Not recording".into());
        }
        (app.is_recording.clone(), app.manager.clone())
    };

    // Signal capture thread to stop
    recording_flag.store(false, Ordering::SeqCst);

    // Take handles out of state
    let (capture_handle, audio_tx, transcript_task, forward_task) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        (
            app.capture_handle.take(),
            app.audio_tx.take(),
            app.transcript_task.take(),
            app.forward_task.take(),
        )
    };

    // Wait for capture thread
    if let Some(handle) = capture_handle {
        let _ = handle.join();
    }

    // Drop the sender to close the channel, then wait for forwarding to finish
    drop(audio_tx);
    if let Some(task) = forward_task {
        let _ = task.await;
    }

    // Give transcription a moment to finish, then collect results
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let (_transcript, session_name) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;

        // Drain remaining audio buffer
        let remaining: Vec<f32> = {
            if let Ok(mut buf) = app.audio_buffer.lock() {
                buf.drain(..).collect()
            } else {
                vec![]
            }
        };
        app.session_audio.extend(&remaining);

        let transcript = app.live_transcript.lock().unwrap().clone();
        let duration_secs = app.session_audio.len() as f32 / 16000.0;

        let final_transcript = if transcript.is_empty() {
            format!("Recording: {:.1}s captured. No speech detected.", duration_secs)
        } else {
            transcript.clone()
        };

        let session = Session {
            name: app.current_session.clone(),
            date: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
            transcript: final_transcript.clone(),
        };
        let name = session.name.clone();
        // Save to disk
        let _ = save_session_to_disk(&app_handle, &session);
        app.sessions.push(session);
        app.transcription_buffer = final_transcript.clone();

        (final_transcript, name)
    };

    // Cancel transcript reader task
    if let Some(task) = transcript_task {
        task.abort();
    }

    // Unload speech model to free memory
    if let Some(mgr) = manager {
        if let Ok(model) = mgr.catalog().get_model(SPEECH_MODEL).await {
            let _ = model.unload().await;
        }
    }

    Ok(session_name)
}

/// Poll transcription buffer — drains new audio, reports level + live transcript
#[tauri::command]
fn get_transcription_buffer(state: State<Mutex<AppState>>) -> Result<String, String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    if app.is_recording.load(Ordering::SeqCst) {
        // Drain samples from shared buffer
        let new_samples: Vec<f32> = {
            if let Ok(mut buf) = app.audio_buffer.lock() {
                buf.drain(..).collect()
            } else {
                vec![]
            }
        };

        if !new_samples.is_empty() {
            let rms = (new_samples.iter().map(|s| s * s).sum::<f32>()
                / new_samples.len() as f32).sqrt();
            let db = if rms > 0.0 { 20.0 * rms.log10() } else { -60.0 };
            app.session_audio.extend(&new_samples);

            let duration = app.session_audio.len() as f32 / 16000.0;
            let bars = "█".repeat(((rms * 50.0).min(20.0)) as usize);

            // Include live transcript text
            let live_text = app.live_transcript.lock().unwrap().clone();
            app.transcription_buffer = format!(
                "🎙️ Recording... {:.1}s | Level: {:.0} dB |{}|\n\n{}",
                duration, db, bars, live_text
            );
        }
    }

    Ok(app.transcription_buffer.clone())
}

/// Generate embeddings for a session transcript and store for RAG
#[tauri::command]
async fn embed_session(session_name: String, app_handle: AppHandle, state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let (manager, transcript, already_embedded) = {
        let app = state.lock().map_err(|e| e.to_string())?;
        let manager = app.manager.clone()
            .ok_or("Foundry not initialized")?;
        let session = app.sessions.iter().find(|s| s.name == session_name)
            .ok_or("Session not found")?;
        let already = app.embeddings.iter().any(|e| e.session_name == session_name);
        (manager, session.transcript.clone(), already)
    };

    if already_embedded {
        return Ok(format!("'{}' already embedded", session_name));
    }

    if transcript.is_empty() {
        return Err("No transcript to embed".into());
    }

    eprintln!("[embed] Starting for session '{}', transcript len={}", session_name, transcript.len());

    // Load embedding model
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": EMBEDDING_MODEL, "progress": 0.0, "status": "checking"
    }));
    let model = manager.catalog().get_model(EMBEDDING_MODEL).await
        .map_err(|e| format!("Embedding model error: {e}"))?;
    if !model.is_cached().await.unwrap_or(false) {
        let handle = app_handle.clone();
        let model_name = EMBEDDING_MODEL.to_string();
        model.download(Some(move |progress: f64| {
            let _ = handle.emit("model-download", serde_json::json!({
                "model": model_name, "progress": progress, "status": "downloading"
            }));
        })).await.map_err(|e| format!("Download failed: {e}"))?;
    }
    model.load().await.ok(); // may already be loaded
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": EMBEDDING_MODEL, "progress": 100.0, "status": "complete"
    }));

    // Prepend session metadata so name/date queries work in RAG
    let metadata_prefix = format!("[Session: {} | Date: {}] ", session_name, 
        session_name.split("—").nth(1).unwrap_or("").trim());

    // Chunk the transcript into ~200-word windows with overlap
    let words: Vec<&str> = transcript.split_whitespace().collect();
    let chunk_size = 200;
    let overlap = 50;
    let chunks: Vec<String> = if words.len() <= chunk_size {
        vec![format!("{}{}", metadata_prefix, words.join(" "))]
    } else {
        let mut result = Vec::new();
        let mut start = 0;
        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            result.push(format!("{}{}", metadata_prefix, words[start..end].join(" ")));
            start += chunk_size - overlap;
        }
        result
    };
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    eprintln!("[embed] {} words → {} chunks (window={}, overlap={})", words.len(), chunk_refs.len(), chunk_size, overlap);

    if chunk_refs.is_empty() {
        model.unload().await.map_err(|e| e.to_string())?;
        return Err("No content to embed".into());
    }

    let client = model.create_embedding_client();
    eprintln!("[embed] Generating embeddings for {} chunks...", chunk_refs.len());
    let response = client.generate_embeddings(&chunk_refs).await
        .map_err(|e| format!("Embedding failed: {e}"))?;

    let entries: Vec<EmbeddingEntry> = chunk_refs.iter().zip(response.data.iter()).map(|(&chunk, data)| {
        EmbeddingEntry {
            session_name: session_name.clone(),
            chunk: chunk.to_string(),
            embedding: data.embedding.clone(),
        }
    }).collect();

    let count = entries.len();
    {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.embeddings.extend(entries);
    }

    // Keep model loaded for faster subsequent queries
    // model.unload().await.map_err(|e| e.to_string())?;

    Ok(format!("Embedded {} chunks from '{}'", count, session_name))
}

#[tauri::command]
fn list_sessions(state: State<Mutex<AppState>>) -> Result<Vec<Session>, String> {
    let app = state.lock().map_err(|e| e.to_string())?;
    Ok(app.sessions.clone())
}

/// RAG query: embed query → cosine search → pass context to chat LLM
#[tauri::command]
async fn query_sessions(query: String, app_handle: AppHandle, state: State<'_, Mutex<AppState>>) -> Result<QueryResult, String> {
    let (manager, embeddings_empty, sessions_empty) = {
        let app = state.lock().map_err(|e| e.to_string())?;
        (
            app.manager.clone(),
            app.embeddings.is_empty(),
            app.sessions.is_empty(),
        )
    };

    if sessions_empty {
        return Ok(QueryResult {
            answer: "No sessions available. Record a session first.".into(),
            sources: vec![],
        });
    }

    let manager = manager.ok_or("Foundry not initialized")?;

    if embeddings_empty {
        return Ok(QueryResult {
            answer: "No embeddings available. Use 'Embed' on a session first.".into(),
            sources: vec![],
        });
    }

    // 1. Embed the query
    let emb_model = manager.catalog().get_model(EMBEDDING_MODEL).await
        .map_err(|e| format!("Embedding model error: {e}"))?;
    emb_model.load().await.ok(); // may already be loaded

    let emb_client = emb_model.create_embedding_client();
    let query_response = emb_client.generate_embedding(&query).await
        .map_err(|e| format!("Query embedding failed: {e}"))?;
    let query_emb = &query_response.data[0].embedding;

    // 2. Cosine similarity search against stored embeddings
    let (top_chunks, top_sources, session_directory) = {
        let app = state.lock().map_err(|e| e.to_string())?;

        // Build a directory of all sessions for the system prompt
        let directory: String = app.sessions.iter()
            .map(|s| format!("- {} ({})", s.name, s.date))
            .collect::<Vec<_>>()
            .join("\n");

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(f32, &EmbeddingEntry)> = app.embeddings.iter()
            .map(|entry| {
                let cosine = cosine_similarity(query_emb, &entry.embedding);
                // Keyword boost: if chunk contains query terms (e.g. client name), boost score
                let chunk_lower = entry.chunk.to_lowercase();
                let keyword_boost: f32 = query_words.iter()
                    .filter(|w| w.len() > 2 && chunk_lower.contains(*w))
                    .count() as f32 * 0.05;
                (cosine + keyword_boost, entry)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<_> = scored.into_iter().take(5).collect();

        let chunks: Vec<String> = top.iter().map(|(_, e)| e.chunk.clone()).collect();
        let sources: Vec<String> = top.iter().map(|(score, e)| {
            format!("{} ({:.2})", e.session_name, score)
        }).collect();
        (chunks, sources, directory)
    };

    // Keep embedding model loaded for faster subsequent queries
    // emb_model.unload().await.map_err(|e| e.to_string())?;

    // 3. Pass context to chat LLM
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": CHAT_MODEL, "progress": 0.0, "status": "checking"
    }));
    let chat_model = manager.catalog().get_model(CHAT_MODEL).await
        .map_err(|e| format!("Chat model error: {e}"))?;
    if !chat_model.is_cached().await.unwrap_or(false) {
        let handle = app_handle.clone();
        let model_name = CHAT_MODEL.to_string();
        chat_model.download(Some(move |progress: f64| {
            let _ = handle.emit("model-download", serde_json::json!({
                "model": model_name, "progress": progress, "status": "downloading"
            }));
        })).await.map_err(|e| format!("Download failed: {e}"))?;
    }
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": CHAT_MODEL, "progress": 100.0, "status": "loading"
    }));
    chat_model.load().await.ok(); // may already be loaded
    let _ = app_handle.emit("model-download", serde_json::json!({
        "model": CHAT_MODEL, "progress": 100.0, "status": "complete"
    }));

    let context = top_chunks.join("\n\n");
    let chat_client = chat_model.create_chat_client()
        .temperature(0.3)
        .max_tokens(512);

    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessage::from(format!(
            "You are a helpful assistant for a therapist. Answer questions based on the session transcripts provided. Be concise and professional.\n\nAvailable sessions:\n{}", session_directory
        )).into(),
        ChatCompletionRequestUserMessage::from(format!(
            "Based on these therapy session excerpts:\n\n{}\n\nAnswer this question: {}",
            context, query
        )).into(),
    ];

    let response = chat_client.complete_chat(&messages, None).await
        .map_err(|e| format!("Chat failed: {e}"))?;

    let answer = response.choices.first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("No response generated")
        .to_string();

    // Keep chat model loaded for faster subsequent queries
    // chat_model.unload().await.map_err(|e| e.to_string())?;

    Ok(QueryResult { answer, sources: top_sources })
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .manage(Mutex::new(AppState {
            is_recording: Arc::new(AtomicBool::new(false)),
            current_session: String::new(),
            transcription_buffer: String::new(),
            sessions: vec![],
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            session_audio: Vec::new(),
            capture_handle: None,
            manager: None,
            audio_tx: None,
            live_transcript: Arc::new(Mutex::new(String::new())),
            transcript_task: None,
            forward_task: None,
            embeddings: vec![],
        }))
        .setup(|app| {
            let handle = app.handle().clone();
            // Seed sample sessions on first run, then load from disk
            seed_sessions_if_needed(&handle);
            let sessions = load_sessions_from_disk(&handle);
            let state = handle.state::<Mutex<AppState>>();
            if let Ok(mut s) = state.lock() {
                s.sessions = sessions;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            init_foundry,
            start_transcription,
            stop_transcription,
            get_transcription_buffer,
            list_sessions,
            embed_session,
            query_sessions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
