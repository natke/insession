mod telemetry;

use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use foundry_local_sdk::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage, FoundryLocalConfig, FoundryLocalManager,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};
use telemetry::{
    duration_ms, MetricsSnapshot, OperationKind, OperationMetric, RetrievedChunk,
    StreamAccumulator, TelemetryStore,
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
    // Foundry Local manager (initialized once and shared across commands)
    manager: Option<Arc<FoundryLocalManager>>,
    // Live transcription: audio sender and accumulated transcript
    audio_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
    live_transcript: Arc<Mutex<String>>,
    transcript_task: Option<tokio::task::JoinHandle<()>>,
    forward_task: Option<tokio::task::JoinHandle<()>>,
    // Embedding store for RAG
    embeddings: Vec<EmbeddingEntry>,
    latest_retrieved_chunks: Vec<RetrievedChunk>,
    telemetry: Arc<Mutex<TelemetryStore>>,
    active_speech: Option<ActiveSpeechMetric>,
}

#[derive(Clone)]
struct SpeechCounters {
    audio_chunks: Arc<AtomicU64>,
    audio_bytes: Arc<AtomicU64>,
    transcription_results: Arc<AtomicU64>,
    first_result_ms: Arc<AtomicU64>,
    terminal_error: Arc<Mutex<Option<String>>>,
}

struct ActiveSpeechMetric {
    metric: OperationMetric,
    started: Instant,
    counters: SpeechCounters,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Session {
    name: String,
    date: String,
    transcript: String,
    #[serde(default)]
    duration_seconds: f64,
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

fn record_metric(telemetry: &Arc<Mutex<TelemetryStore>>, metric: OperationMetric) {
    if let Ok(mut store) = telemetry.lock() {
        store.record(metric);
    }
}

fn finish_metric(
    telemetry: &Arc<Mutex<TelemetryStore>>,
    mut metric: OperationMetric,
    started: Instant,
) {
    metric.duration_ms = duration_ms(started.elapsed());
    record_metric(telemetry, metric);
}

fn fail_metric(
    telemetry: &Arc<Mutex<TelemetryStore>>,
    mut metric: OperationMetric,
    started: Instant,
    error: String,
) -> String {
    metric.duration_ms = duration_ms(started.elapsed());
    let error_category = error
        .split_once(':')
        .map(|(category, _)| category)
        .unwrap_or("Operation failed");
    metric.fail(error_category);
    record_metric(telemetry, metric);
    error
}

#[tauri::command]
fn get_metrics_snapshot(state: State<Mutex<AppState>>) -> Result<MetricsSnapshot, String> {
    let (telemetry, in_memory_chunk_count, transcription_count, latest_retrieved_chunks) = {
        let app = state.lock().map_err(|error| error.to_string())?;
        (
            app.telemetry.clone(),
            app.embeddings.len(),
            app.sessions.len(),
            app.latest_retrieved_chunks.clone(),
        )
    };
    let mut store = telemetry.lock().map_err(|error| error.to_string())?;
    Ok(store.snapshot(
        in_memory_chunk_count,
        transcription_count,
        latest_retrieved_chunks,
    ))
}

/// Get the sessions directory path (inside app data)
fn sessions_dir(app_handle: &AppHandle) -> Result<std::path::PathBuf, String> {
    let app_data = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {e}"))?;
    let dir = app_data.join("sessions");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create sessions dir: {e}"))?;
    Ok(dir)
}

/// Save a session to disk as JSON
fn save_session_to_disk(app_handle: &AppHandle, session: &Session) -> Result<(), String> {
    let dir = sessions_dir(app_handle)?;
    let slug = session
        .name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>();
    let path = dir.join(format!("{}.json", slug));
    let json =
        serde_json::to_string_pretty(session).map_err(|e| format!("Serialize failed: {e}"))?;
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
        Err(e) => {
            eprintln!("[persist] {}", e);
            return vec![];
        }
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
async fn init_foundry(
    app_handle: AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<String, String> {
    let has_manager = {
        let app = state.lock().map_err(|e| e.to_string())?;
        app.manager.is_some()
    };

    if has_manager {
        return Ok("Already initialized".into());
    }

    // Use app support dir for all SDK data (avoid Documents folder prompt)
    let app_data = app_handle
        .path()
        .app_data_dir()
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
            .additional_setting("AzureCatalogFilter", "'',test"),
    )
    .map_err(|e| format!("Failed to create manager: {e}"))?;

    // Discover and register GPU execution providers (WebGPU etc.)
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": "GPU acceleration", "progress": 0.0, "status": "checking"
        }),
    );
    let eps = manager.discover_eps().unwrap_or_default();
    let ep_names: Vec<String> = eps
        .iter()
        .map(|ep| format!("{} (registered={})", ep.name, ep.is_registered))
        .collect();
    eprintln!("[init] Available EPs: {:?}", ep_names);

    let unregistered: Vec<&str> = eps
        .iter()
        .filter(|ep| !ep.is_registered)
        .map(|ep| ep.name.as_str())
        .collect();

    if !unregistered.is_empty() {
        eprintln!("[init] Downloading GPU acceleration: {:?}", unregistered);
        let handle = app_handle.clone();
        let result = manager
            .download_and_register_eps_with_progress(
                Some(&unregistered),
                move |_ep_name: &str, percent: f64| {
                    let _ = handle.emit("model-download", serde_json::json!({
                    "model": "GPU acceleration", "progress": percent, "status": "downloading"
                }));
                },
            )
            .await
            .map_err(|e| format!("EP registration failed: {e}"))?;
        eprintln!(
            "[init] GPU acceleration: registered={:?}, failed={:?}",
            result.registered_eps, result.failed_eps
        );
        let _ = app_handle.emit(
            "model-download",
            serde_json::json!({
                "model": "GPU acceleration", "progress": 100.0, "status": "complete"
            }),
        );
    } else {
        eprintln!("[init] All EPs already registered");
        let _ = app_handle.emit(
            "model-download",
            serde_json::json!({
                "model": "GPU acceleration", "progress": 100.0, "status": "complete"
            }),
        );
    }

    // Download the speech model if not cached
    let model = manager
        .catalog()
        .get_model(SPEECH_MODEL)
        .await
        .map_err(|e| format!("Speech model not found: {e}"))?;

    if !model.is_cached().await.unwrap_or(false) {
        let handle = app_handle.clone();
        let model_name = SPEECH_MODEL.to_string();
        let _ = handle.emit(
            "model-download",
            serde_json::json!({
                "model": model_name, "progress": 0.0, "status": "starting"
            }),
        );
        model
            .download(Some(move |progress: f64| {
                let _ = handle.emit(
                    "model-download",
                    serde_json::json!({
                        "model": model_name, "progress": progress, "status": "downloading"
                    }),
                );
            }))
            .await
            .map_err(|e| format!("Download failed: {e}"))?;
        let _ = app_handle.emit(
            "model-download",
            serde_json::json!({
                "model": SPEECH_MODEL, "progress": 100.0, "status": "complete"
            }),
        );
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
async fn start_transcription(
    session_name: String,
    state: State<'_, Mutex<AppState>>,
) -> Result<(), String> {
    let operation_started = Instant::now();
    let (recording_flag, audio_buf, manager, live_transcript, telemetry) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        if app.is_recording.load(Ordering::SeqCst) {
            return Err("Already recording".into());
        }

        app.current_session = session_name;
        app.transcription_buffer = String::new();
        app.session_audio.clear();
        *app.live_transcript.lock().unwrap() = String::new();

        let manager = app
            .manager
            .clone()
            .ok_or("Foundry not initialized — call init_foundry first")?;

        (
            app.is_recording.clone(),
            app.audio_buffer.clone(),
            manager,
            app.live_transcript.clone(),
            app.telemetry.clone(),
        )
    };
    let mut metric = OperationMetric::new(
        OperationKind::SpeechTranscription,
        Some(SPEECH_MODEL),
        Utc::now().to_rfc3339(),
    );

    // Load the speech model
    let model = match manager.catalog().get_model(SPEECH_MODEL).await {
        Ok(model) => model,
        Err(error) => {
            let error = format!("Model error: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    let model_load_started = Instant::now();
    if let Err(error) = model.load().await {
        let error = format!("Load failed: {error}");
        return Err(fail_metric(&telemetry, metric, operation_started, error));
    }
    metric.stage("model_load", model_load_started.elapsed());

    // Create live transcription session
    let audio_client = model.create_audio_client();
    let session = Arc::new(audio_client.create_live_transcription_session());
    let session_start_started = Instant::now();
    if let Err(error) = session.start(None).await {
        let error = format!("Session start failed: {error}");
        return Err(fail_metric(&telemetry, metric, operation_started, error));
    }
    metric.stage(
        "transcription_session_start",
        session_start_started.elapsed(),
    );

    let counters = SpeechCounters {
        audio_chunks: Arc::new(AtomicU64::new(0)),
        audio_bytes: Arc::new(AtomicU64::new(0)),
        transcription_results: Arc::new(AtomicU64::new(0)),
        first_result_ms: Arc::new(AtomicU64::new(u64::MAX)),
        terminal_error: Arc::new(Mutex::new(None)),
    };

    // Channel for sending PCM audio to the transcription session
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(100);

    // Task: forward audio chunks to transcription session
    let session_fwd = Arc::clone(&session);
    let forward_counters = counters.clone();
    let forward_task = tokio::spawn(async move {
        let mut chunk_count = 0u64;
        let mut total_bytes = 0u64;
        while let Some(bytes) = audio_rx.recv().await {
            total_bytes += bytes.len() as u64;
            chunk_count += 1;
            forward_counters
                .audio_chunks
                .fetch_add(1, Ordering::Relaxed);
            forward_counters
                .audio_bytes
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            if chunk_count % 100 == 1 {
                eprintln!(
                    "[audio] chunk #{chunk_count}, total {total_bytes} bytes sent to transcriber"
                );
            }
            if let Err(e) = session_fwd.append(&bytes, None).await {
                if let Ok(mut error) = forward_counters.terminal_error.lock() {
                    *error = Some(format!("Audio append failed: {e}"));
                }
                eprintln!("Append error: {e}");
                break;
            }
        }
        eprintln!(
            "[audio] forwarding stopped after {chunk_count} chunks, {total_bytes} bytes total"
        );
    });

    // Task: read transcription results
    let transcript_ref = live_transcript.clone();
    let transcript_counters = counters.clone();
    let mut stream = session.get_stream().await.map_err(|error| {
        fail_metric(
            &telemetry,
            metric.clone(),
            operation_started,
            format!("Stream error: {error}"),
        )
    })?;
    let transcript_task = tokio::spawn(async move {
        eprintln!("[transcribe] listening for results...");
        while let Some(result) = stream.next().await {
            match result {
                Ok(r) => {
                    if let Some(content) = r.content.first() {
                        let text = &content.text;
                        eprintln!("[transcribe] got text: '{}' (final={})", text, r.is_final);
                        if !text.is_empty() {
                            transcript_counters
                                .transcription_results
                                .fetch_add(1, Ordering::Relaxed);
                            let first_result_ms = operation_started.elapsed().as_millis() as u64;
                            let _ = transcript_counters.first_result_ms.compare_exchange(
                                u64::MAX,
                                first_result_ms,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            );
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
                    if let Ok(mut error) = transcript_counters.terminal_error.lock() {
                        *error = Some(format!("Transcription stream failed: {e}"));
                    }
                    eprintln!("Transcription error: {e}");
                    break;
                }
            }
        }
        eprintln!("[transcribe] stream ended");
    });

    // Spawn audio capture on a dedicated thread (cpal::Stream isn't Send)
    recording_flag.store(true, Ordering::SeqCst);
    let recording_flag_thread = recording_flag.clone();
    let audio_buf_thread = audio_buf.clone();
    let audio_tx_thread = audio_tx.clone();
    let capture_counters = counters.clone();
    let handle = std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                if let Ok(mut error) = capture_counters.terminal_error.lock() {
                    *error = Some("No input device".to_owned());
                }
                eprintln!("No input device");
                return;
            }
        };

        // Use the device's default config (macOS typically 48kHz stereo)
        let default_config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                if let Ok(mut error) = capture_counters.terminal_error.lock() {
                    *error = Some(format!("No default input config: {e}"));
                }
                eprintln!("No default input config: {}", e);
                return;
            }
        };
        let native_rate = default_config.sample_rate().0;
        let native_channels = default_config.channels() as usize;
        eprintln!(
            "Audio input: {}Hz, {} channels",
            native_rate, native_channels
        );

        let config = cpal::StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let buf = audio_buf_thread.clone();
        let tx = audio_tx_thread.clone();
        let callback_counters = capture_counters.clone();
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
            move |err| {
                if let Ok(mut error) = callback_counters.terminal_error.lock() {
                    *error = Some(format!("Audio error: {err}"));
                }
                eprintln!("Audio error: {}", err);
            },
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                if let Ok(mut error) = capture_counters.terminal_error.lock() {
                    *error = Some(format!("Stream build failed: {e}"));
                }
                eprintln!("Stream build failed: {}", e);
                return;
            }
        };

        if let Err(e) = stream.play() {
            if let Ok(mut error) = capture_counters.terminal_error.lock() {
                *error = Some(format!("Stream play failed: {e}"));
            }
            eprintln!("Stream play failed: {}", e);
            return;
        }

        while recording_flag_thread.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });

    // Store handles
    {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.capture_handle = Some(handle);
        app.audio_tx = Some(audio_tx);
        app.transcript_task = Some(transcript_task);
        app.forward_task = Some(forward_task);
        app.transcription_buffer = "🎙️ Recording... (live transcription active)\n".into();
        app.active_speech = Some(ActiveSpeechMetric {
            metric,
            started: operation_started,
            counters,
        });
    }

    Ok(())
}

/// Stop recording and save session with transcript
#[tauri::command]
async fn stop_transcription(
    app_handle: AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<String, String> {
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
    let (capture_handle, audio_tx, transcript_task, forward_task, active_speech) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        (
            app.capture_handle.take(),
            app.audio_tx.take(),
            app.transcript_task.take(),
            app.forward_task.take(),
            app.active_speech.take(),
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
            format!(
                "Recording: {:.1}s captured. No speech detected.",
                duration_secs
            )
        } else {
            transcript.clone()
        };

        let session = Session {
            name: app.current_session.clone(),
            date: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
            transcript: final_transcript.clone(),
            duration_seconds: duration_secs as f64,
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

    if let Some(mut active) = active_speech {
        active.metric.count(
            "audio_chunk_count",
            active.counters.audio_chunks.load(Ordering::Relaxed),
        );
        active.metric.count(
            "audio_byte_count",
            active.counters.audio_bytes.load(Ordering::Relaxed),
        );
        active.metric.count(
            "transcription_result_count",
            active
                .counters
                .transcription_results
                .load(Ordering::Relaxed),
        );
        let first_result_ms = active.counters.first_result_ms.load(Ordering::Relaxed);
        if first_result_ms != u64::MAX {
            active
                .metric
                .measurement("time_to_first_result_ms", first_result_ms as f64);
        }
        if let Ok(error) = active.counters.terminal_error.lock() {
            if let Some(error) = error.as_deref() {
                active.metric.fail(error);
            }
        }
        let telemetry = {
            let app = state.lock().map_err(|error| error.to_string())?;
            app.telemetry.clone()
        };
        finish_metric(&telemetry, active.metric, active.started);
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
            let rms =
                (new_samples.iter().map(|s| s * s).sum::<f32>() / new_samples.len() as f32).sqrt();
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
async fn embed_session(
    session_name: String,
    app_handle: AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<String, String> {
    let operation_started = Instant::now();
    let (manager, transcript, already_embedded, telemetry) = {
        let app = state.lock().map_err(|e| e.to_string())?;
        let manager = app.manager.clone().ok_or("Foundry not initialized")?;
        let session = app
            .sessions
            .iter()
            .find(|s| s.name == session_name)
            .ok_or("Session not found")?;
        let already = app
            .embeddings
            .iter()
            .any(|e| e.session_name == session_name);
        (
            manager,
            session.transcript.clone(),
            already,
            app.telemetry.clone(),
        )
    };

    if already_embedded {
        return Ok(format!("'{}' already embedded", session_name));
    }

    if transcript.is_empty() {
        return Err("No transcript to embed".into());
    }

    let mut metric = OperationMetric::new(
        OperationKind::Embedding,
        Some(EMBEDDING_MODEL),
        Utc::now().to_rfc3339(),
    );
    eprintln!(
        "[embed] Starting for session '{}', transcript len={}",
        session_name,
        transcript.len()
    );

    // Load embedding model
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": EMBEDDING_MODEL, "progress": 0.0, "status": "checking"
        }),
    );
    let model = match manager.catalog().get_model(EMBEDDING_MODEL).await {
        Ok(model) => model,
        Err(error) => {
            let error = format!("Embedding model error: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    if !model.is_cached().await.unwrap_or(false) {
        let download_started = Instant::now();
        let handle = app_handle.clone();
        let model_name = EMBEDDING_MODEL.to_string();
        if let Err(error) = model
            .download(Some(move |progress: f64| {
                let _ = handle.emit(
                    "model-download",
                    serde_json::json!({
                        "model": model_name, "progress": progress, "status": "downloading"
                    }),
                );
            }))
            .await
        {
            let error = format!("Download failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
        metric.stage("model_download", download_started.elapsed());
    }
    let load_started = Instant::now();
    match model.is_loaded().await {
        Ok(true) => metric.count("model_already_loaded", 1),
        Ok(false) => {
            if let Err(error) = model.load().await {
                let error = format!("Embedding model load failed: {error}");
                return Err(fail_metric(&telemetry, metric, operation_started, error));
            }
        }
        Err(error) => {
            let error = format!("Embedding model state check failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    }
    metric.stage("model_load", load_started.elapsed());
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": EMBEDDING_MODEL, "progress": 100.0, "status": "complete"
        }),
    );

    // Prepend session metadata so name/date queries work in RAG
    let metadata_prefix = format!(
        "[Session: {} | Date: {}] ",
        session_name,
        session_name.split("—").nth(1).unwrap_or("").trim()
    );

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
            result.push(format!(
                "{}{}",
                metadata_prefix,
                words[start..end].join(" ")
            ));
            start += chunk_size - overlap;
        }
        result
    };
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    eprintln!(
        "[embed] {} words → {} chunks (window={}, overlap={})",
        words.len(),
        chunk_refs.len(),
        chunk_size,
        overlap
    );

    if chunk_refs.is_empty() {
        let error = "No content to embed".to_owned();
        return Err(fail_metric(&telemetry, metric, operation_started, error));
    }

    let client = model.create_embedding_client();
    eprintln!(
        "[embed] Generating embeddings for {} chunks...",
        chunk_refs.len()
    );
    let inference_started = Instant::now();
    let response = match client.generate_embeddings(&chunk_refs).await {
        Ok(response) => response,
        Err(error) => {
            let error = format!("Embedding failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    metric.stage("embedding_generation", inference_started.elapsed());

    let entries: Vec<EmbeddingEntry> = chunk_refs
        .iter()
        .zip(response.data.iter())
        .map(|(&chunk, data)| EmbeddingEntry {
            session_name: session_name.clone(),
            chunk: chunk.to_string(),
            embedding: data.embedding.clone(),
        })
        .collect();

    let count = entries.len();
    {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.embeddings.extend(entries);
    }

    // Keep model loaded for faster subsequent queries
    // model.unload().await.map_err(|e| e.to_string())?;

    metric.count("chunk_count", count as u64);
    metric.count("input_character_count", transcript.chars().count() as u64);
    finish_metric(&telemetry, metric, operation_started);

    Ok(format!("Embedded {} chunks from '{}'", count, session_name))
}

#[tauri::command]
fn list_sessions(state: State<Mutex<AppState>>) -> Result<Vec<Session>, String> {
    let app = state.lock().map_err(|e| e.to_string())?;
    Ok(app.sessions.clone())
}

/// RAG query: embed query → cosine search → pass context to chat LLM
#[tauri::command]
async fn query_sessions(
    query: String,
    app_handle: AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<QueryResult, String> {
    let operation_started = Instant::now();
    let (manager, embeddings_empty, sessions_empty, telemetry) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;
        app.latest_retrieved_chunks.clear();
        (
            app.manager.clone(),
            app.embeddings.is_empty(),
            app.sessions.is_empty(),
            app.telemetry.clone(),
        )
    };
    let mut metric = OperationMetric::new(
        OperationKind::RagQuery,
        Some(CHAT_MODEL),
        Utc::now().to_rfc3339(),
    );

    if sessions_empty {
        metric.count("retrieved_chunk_count", 0);
        finish_metric(&telemetry, metric, operation_started);
        return Ok(QueryResult {
            answer: "No sessions available. Record a session first.".into(),
            sources: vec![],
        });
    }

    let manager = match manager {
        Some(manager) => manager,
        None => {
            let error = "Foundry not initialized".to_owned();
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };

    if embeddings_empty {
        metric.count("retrieved_chunk_count", 0);
        finish_metric(&telemetry, metric, operation_started);
        return Ok(QueryResult {
            answer: "No embeddings available. Use 'Embed' on a session first.".into(),
            sources: vec![],
        });
    }

    // 1. Embed the query
    let emb_model = match manager.catalog().get_model(EMBEDDING_MODEL).await {
        Ok(model) => model,
        Err(error) => {
            let error = format!("Embedding model error: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    let embedding_load_started = Instant::now();
    match emb_model.is_loaded().await {
        Ok(true) => metric.count("query_embedding_model_already_loaded", 1),
        Ok(false) => {
            if let Err(error) = emb_model.load().await {
                let error = format!("Embedding model load failed: {error}");
                return Err(fail_metric(&telemetry, metric, operation_started, error));
            }
        }
        Err(error) => {
            let error = format!("Embedding model state check failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    }
    metric.stage(
        "query_embedding_model_load",
        embedding_load_started.elapsed(),
    );

    let emb_client = emb_model.create_embedding_client();
    let query_embedding_started = Instant::now();
    let query_response = match emb_client.generate_embedding(&query).await {
        Ok(response) => response,
        Err(error) => {
            let error = format!("Query embedding failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    metric.stage("query_embedding", query_embedding_started.elapsed());
    let query_emb = &query_response.data[0].embedding;

    // 2. Cosine similarity search against stored embeddings
    let retrieval_started = Instant::now();
    let (top_chunks, top_sources, session_directory) = {
        let mut app = state.lock().map_err(|e| e.to_string())?;

        // Build a directory of all sessions for the system prompt
        let directory: String = app
            .sessions
            .iter()
            .map(|s| format!("- {} ({})", s.name, s.date))
            .collect::<Vec<_>>()
            .join("\n");

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(f32, &EmbeddingEntry)> = app
            .embeddings
            .iter()
            .map(|entry| {
                let cosine = cosine_similarity(query_emb, &entry.embedding);
                // Keyword boost: if chunk contains query terms (e.g. client name), boost score
                let chunk_lower = entry.chunk.to_lowercase();
                let keyword_boost: f32 = query_words
                    .iter()
                    .filter(|w| w.len() > 2 && chunk_lower.contains(*w))
                    .count() as f32
                    * 0.05;
                (cosine + keyword_boost, entry)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<_> = scored.into_iter().take(5).collect();

        let (chunks, sources, retrieved_chunks) = {
            let chunks: Vec<String> = top.iter().map(|(_, e)| e.chunk.clone()).collect();
            let sources: Vec<String> = top
                .iter()
                .map(|(score, e)| format!("{} ({:.2})", e.session_name, score))
                .collect();
            let retrieved_chunks = top
                .iter()
                .enumerate()
                .map(|(index, (score, entry))| RetrievedChunk {
                    rank: index + 1,
                    session_name: entry.session_name.clone(),
                    score: *score,
                    text: entry.chunk.clone(),
                })
                .collect();
            (chunks, sources, retrieved_chunks)
        };
        app.latest_retrieved_chunks = retrieved_chunks;
        (chunks, sources, directory)
    };
    metric.stage("retrieval", retrieval_started.elapsed());
    metric.count("retrieved_chunk_count", top_chunks.len() as u64);

    // Keep embedding model loaded for faster subsequent queries
    // emb_model.unload().await.map_err(|e| e.to_string())?;

    // 3. Pass context to chat LLM
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": CHAT_MODEL, "progress": 0.0, "status": "checking"
        }),
    );
    let chat_model = match manager.catalog().get_model(CHAT_MODEL).await {
        Ok(model) => model,
        Err(error) => {
            let error = format!("Chat model error: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    if !chat_model.is_cached().await.unwrap_or(false) {
        let download_started = Instant::now();
        let handle = app_handle.clone();
        let model_name = CHAT_MODEL.to_string();
        if let Err(error) = chat_model
            .download(Some(move |progress: f64| {
                let _ = handle.emit(
                    "model-download",
                    serde_json::json!({
                        "model": model_name, "progress": progress, "status": "downloading"
                    }),
                );
            }))
            .await
        {
            let error = format!("Download failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
        metric.stage("chat_model_download", download_started.elapsed());
    }
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": CHAT_MODEL, "progress": 100.0, "status": "loading"
        }),
    );
    let chat_load_started = Instant::now();
    match chat_model.is_loaded().await {
        Ok(true) => metric.count("chat_model_already_loaded", 1),
        Ok(false) => {
            if let Err(error) = chat_model.load().await {
                let error = format!("Chat model load failed: {error}");
                return Err(fail_metric(&telemetry, metric, operation_started, error));
            }
        }
        Err(error) => {
            let error = format!("Chat model state check failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    }
    metric.stage("chat_model_load", chat_load_started.elapsed());
    let _ = app_handle.emit(
        "model-download",
        serde_json::json!({
            "model": CHAT_MODEL, "progress": 100.0, "status": "complete"
        }),
    );

    let context = top_chunks.join("\n\n");
    let chat_client = chat_model
        .create_chat_client()
        .temperature(0.3)
        .include_usage(true)
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

    let inference_started = Instant::now();
    let mut stream = match chat_client.complete_streaming_chat(&messages, None).await {
        Ok(stream) => stream,
        Err(error) => {
            let error = format!("Chat failed: {error}");
            return Err(fail_metric(&telemetry, metric, operation_started, error));
        }
    };
    let mut accumulator = StreamAccumulator::new(inference_started);
    let mut token_usage = None;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                let error = format!("Chat stream failed: {error}");
                return Err(fail_metric(&telemetry, metric, operation_started, error));
            }
        };
        if let Some(content) = chunk
            .choices
            .first()
            .and_then(|choice| choice.delta.content.as_deref())
        {
            accumulator.push(content, Instant::now());
        }
        if let Some(usage) = chunk.usage {
            token_usage = Some(usage);
        }
    }
    let stream_summary = accumulator.finish(Instant::now());
    metric.stage("chat_inference", inference_started.elapsed());
    if let Some(ttft) = stream_summary.time_to_first_token {
        metric.measurement("time_to_first_token_ms", duration_ms(ttft));
    }
    if let Some(generation_duration) = stream_summary.generation_duration {
        metric.measurement("generation_duration_ms", duration_ms(generation_duration));
    }
    if let Some(usage) = token_usage {
        metric.count("prompt_tokens", u64::from(usage.prompt_tokens));
        metric.count("completion_tokens", u64::from(usage.completion_tokens));
        metric.count("total_tokens", u64::from(usage.total_tokens));
        if let Some(generation_duration) = stream_summary.generation_duration {
            let seconds = generation_duration.as_secs_f64();
            if seconds > 0.0 {
                metric.measurement(
                    "tokens_per_second",
                    f64::from(usage.completion_tokens) / seconds,
                );
            }
        }
    } else {
        if let Some(tokens_per_second) = stream_summary.estimated_tokens_per_second {
            metric.measurement("estimated_tokens_per_second", tokens_per_second);
        }
        metric.count("estimated_output_tokens", stream_summary.estimated_tokens);
    }
    metric.count(
        "generated_character_count",
        stream_summary.content.chars().count() as u64,
    );
    let answer = if stream_summary.content.is_empty() {
        "No response generated".to_owned()
    } else {
        stream_summary.content
    };

    // Keep chat model loaded for faster subsequent queries
    // chat_model.unload().await.map_err(|e| e.to_string())?;

    finish_metric(&telemetry, metric, operation_started);
    Ok(QueryResult {
        answer,
        sources: top_sources,
    })
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
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
            latest_retrieved_chunks: vec![],
            telemetry: Arc::new(Mutex::new(TelemetryStore::new())),
            active_speech: None,
        }))
        .setup(|app| {
            let handle = app.handle().clone();
            // Seed sample sessions on first run, then load from disk
            seed_sessions_if_needed(&handle);
            let sessions = load_sessions_from_disk(&handle);
            let state = handle.state::<Mutex<AppState>>();
            if let Ok(mut s) = state.lock() {
                s.sessions = sessions;
                if let Ok(mut telemetry) = s.telemetry.lock() {
                    match handle.path().app_data_dir() {
                        Ok(app_data) => {
                            let logs_dir = app_data.join("logs");
                            match std::fs::create_dir_all(&logs_dir) {
                                Ok(()) => {
                                    telemetry.configure_log_path(logs_dir.join("metrics.jsonl"));
                                }
                                Err(error) => telemetry.report_error(&format!(
                                    "Metric log directory creation failed: {error}"
                                )),
                            }
                        }
                        Err(error) => telemetry.report_error(&format!(
                            "Metric app-data directory lookup failed: {error}"
                        )),
                    }
                }
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
            get_metrics_snapshot,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::Session;

    #[test]
    fn loads_sessions_saved_before_duration_was_added() {
        let session: Session = serde_json::from_str(
            r#"{"name":"Existing session","date":"2026-01-01","transcript":"Text"}"#,
        )
        .unwrap();

        assert_eq!(session.duration_seconds, 0.0);
    }
}
