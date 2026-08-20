use std::{
    collections::{BTreeMap, VecDeque},
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use chrono::Utc;
use serde::Serialize;
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

const METRIC_SCHEMA_VERSION: u8 = 1;
const MAX_RECENT_EVENTS: usize = 100;
static NEXT_OPERATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Embedding,
    RagQuery,
    SpeechTranscription,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageMetric {
    pub name: String,
    pub duration_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OperationMetric {
    pub schema_version: u8,
    pub operation_id: String,
    pub kind: OperationKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub started_at: String,
    pub duration_ms: f64,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub stages: Vec<StageMetric>,
    pub measurements: BTreeMap<String, f64>,
    pub counts: BTreeMap<String, u64>,
}

impl OperationMetric {
    pub fn new(kind: OperationKind, model: Option<&str>, started_at: String) -> Self {
        Self {
            schema_version: METRIC_SCHEMA_VERSION,
            operation_id: format!(
                "{}-{}",
                Utc::now().timestamp_millis(),
                NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed)
            ),
            kind,
            model: model.map(str::to_owned),
            started_at,
            duration_ms: 0.0,
            success: true,
            error: None,
            stages: Vec::new(),
            measurements: BTreeMap::new(),
            counts: BTreeMap::new(),
        }
    }

    pub fn stage(&mut self, name: &str, duration: Duration) {
        self.stages.push(StageMetric {
            name: name.to_owned(),
            duration_ms: duration_ms(duration),
        });
    }

    pub fn measurement(&mut self, name: &str, value: f64) {
        if value.is_finite() {
            self.measurements.insert(name.to_owned(), value);
        }
    }

    pub fn count(&mut self, name: &str, value: u64) {
        self.counts.insert(name.to_owned(), value);
    }

    pub fn fail(&mut self, error: &str) {
        self.success = false;
        self.error = Some(sanitize_error(error));
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProcessMemorySample {
    pub sampled_at: String,
    pub resident_bytes: u64,
    pub resident_mib: f64,
    pub peak_resident_bytes: u64,
    pub peak_resident_mib: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievedChunk {
    pub rank: usize,
    pub session_name: String,
    pub score: f32,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process: Option<ProcessMemorySample>,
    pub in_memory_chunk_count: usize,
    pub transcription_count: usize,
    pub latest_retrieved_chunks: Vec<RetrievedChunk>,
    pub recent_operations: Vec<OperationMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics_error: Option<String>,
}

pub struct TelemetryStore {
    recent: VecDeque<OperationMetric>,
    peak_resident_bytes: u64,
    log_path: Option<PathBuf>,
    diagnostics_error: Option<String>,
}

impl TelemetryStore {
    pub fn new() -> Self {
        Self {
            recent: VecDeque::with_capacity(MAX_RECENT_EVENTS),
            peak_resident_bytes: 0,
            log_path: None,
            diagnostics_error: None,
        }
    }

    pub fn configure_log_path(&mut self, log_path: PathBuf) {
        self.log_path = Some(log_path);
    }

    pub fn report_error(&mut self, error: &str) {
        self.diagnostics_error = Some(sanitize_error(error));
    }

    pub fn record(&mut self, metric: OperationMetric) {
        if let Some(path) = &self.log_path {
            let write_result = (|| -> Result<(), String> {
                let line = serde_json::to_string(&metric)
                    .map_err(|error| format!("Metric serialization failed: {error}"))?;
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|error| format!("Metric log open failed: {error}"))?;
                writeln!(file, "{line}")
                    .map_err(|error| format!("Metric log write failed: {error}"))
            })();

            if let Err(error) = write_result {
                self.diagnostics_error = Some(sanitize_error(&error));
            }
        }

        if self.recent.len() == MAX_RECENT_EVENTS {
            self.recent.pop_front();
        }
        self.recent.push_back(metric);
    }

    pub fn snapshot(
        &mut self,
        in_memory_chunk_count: usize,
        transcription_count: usize,
        latest_retrieved_chunks: Vec<RetrievedChunk>,
    ) -> MetricsSnapshot {
        let process = match current_process_memory() {
            Ok(resident_bytes) => {
                self.peak_resident_bytes = self.peak_resident_bytes.max(resident_bytes);
                Some(ProcessMemorySample {
                    sampled_at: Utc::now().to_rfc3339(),
                    resident_bytes,
                    resident_mib: bytes_to_mib(resident_bytes),
                    peak_resident_bytes: self.peak_resident_bytes,
                    peak_resident_mib: bytes_to_mib(self.peak_resident_bytes),
                })
            }
            Err(error) => {
                self.diagnostics_error = Some(sanitize_error(&error));
                None
            }
        };

        MetricsSnapshot {
            process,
            in_memory_chunk_count,
            transcription_count,
            latest_retrieved_chunks,
            recent_operations: self.recent.iter().rev().cloned().collect(),
            diagnostics_error: self.diagnostics_error.clone(),
        }
    }
}

pub struct StreamAccumulator {
    started: Instant,
    first_content: Option<Instant>,
    content: String,
}

pub struct StreamSummary {
    pub content: String,
    pub time_to_first_token: Option<Duration>,
    pub generation_duration: Option<Duration>,
    pub estimated_tokens: u64,
    pub estimated_tokens_per_second: Option<f64>,
}

impl StreamAccumulator {
    pub fn new(started: Instant) -> Self {
        Self {
            started,
            first_content: None,
            content: String::new(),
        }
    }

    pub fn push(&mut self, content: &str, received_at: Instant) {
        if content.is_empty() {
            return;
        }
        self.first_content.get_or_insert(received_at);
        self.content.push_str(content);
    }

    pub fn finish(self, finished_at: Instant) -> StreamSummary {
        let estimated_tokens = estimate_tokens(&self.content);
        let time_to_first_token = self
            .first_content
            .map(|first_content| first_content.duration_since(self.started));
        let generation_duration = self
            .first_content
            .map(|first_content| finished_at.duration_since(first_content));
        let estimated_tokens_per_second = generation_duration.and_then(|duration| {
            let seconds = duration.as_secs_f64();
            (seconds > 0.0).then_some(estimated_tokens as f64 / seconds)
        });

        StreamSummary {
            content: self.content,
            time_to_first_token,
            generation_duration,
            estimated_tokens,
            estimated_tokens_per_second,
        }
    }
}

pub fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }

    let character_count = text.chars().count();
    let character_estimate = (character_count + 3) / 4;
    let word_floor = text.split_whitespace().count();
    character_estimate.max(word_floor) as u64
}

pub fn sanitize_error(error: &str) -> String {
    let single_line = error.split_whitespace().collect::<Vec<_>>().join(" ");
    single_line.chars().take(240).collect()
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn current_process_memory() -> Result<u64, String> {
    let pid = get_current_pid().map_err(|error| format!("Process ID unavailable: {error}"))?;
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system
        .process(pid)
        .map(|process| process.memory())
        .ok_or_else(|| "Current process memory unavailable".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_uses_character_estimate_with_word_floor() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("one two three"), 4);
        assert_eq!(estimate_tokens("a b c d e"), 5);
    }

    #[test]
    fn stream_summary_tracks_first_content_and_throughput() {
        let started = Instant::now();
        let mut stream = StreamAccumulator::new(started);
        stream.push("", started + Duration::from_millis(50));
        stream.push("hello ", started + Duration::from_millis(100));
        stream.push("world", started + Duration::from_millis(300));

        let summary = stream.finish(started + Duration::from_millis(600));
        assert_eq!(summary.content, "hello world");
        assert_eq!(
            summary.time_to_first_token,
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            summary.generation_duration,
            Some(Duration::from_millis(500))
        );
        assert_eq!(summary.estimated_tokens, 3);
        assert_eq!(summary.estimated_tokens_per_second, Some(6.0));
    }

    #[test]
    fn store_retains_only_the_latest_events() {
        let mut store = TelemetryStore::new();
        for index in 0..(MAX_RECENT_EVENTS + 5) {
            let mut metric =
                OperationMetric::new(OperationKind::Embedding, None, Utc::now().to_rfc3339());
            metric.count("index", index as u64);
            store.record(metric);
        }

        let snapshot = store.snapshot(0, 0, vec![]);
        assert_eq!(snapshot.recent_operations.len(), MAX_RECENT_EVENTS);
        assert_eq!(
            snapshot.recent_operations.first().unwrap().counts["index"],
            (MAX_RECENT_EVENTS + 4) as u64
        );
    }

    #[test]
    fn serialized_metric_cannot_contain_content_fields() {
        let metric = OperationMetric::new(
            OperationKind::RagQuery,
            Some("model"),
            Utc::now().to_rfc3339(),
        );
        let serialized = serde_json::to_string(&metric).unwrap();

        for forbidden in [
            "prompt",
            "transcript",
            "answer",
            "session_name",
            "client_name",
            "retrieved_chunk",
        ] {
            assert!(!serialized.contains(forbidden));
        }
    }

    #[test]
    fn writes_one_json_object_per_metric_event() {
        let path = std::env::temp_dir().join(format!(
            "insession-metrics-{}-{}.jsonl",
            std::process::id(),
            NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut store = TelemetryStore::new();
        store.configure_log_path(path.clone());
        store.record(OperationMetric::new(
            OperationKind::Embedding,
            Some("test-model"),
            Utc::now().to_rfc3339(),
        ));

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        let value: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["model"], "test-model");

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn sanitizes_multiline_errors_and_limits_length() {
        let error = format!("first\nsecond {}", "x".repeat(300));
        let sanitized = sanitize_error(&error);
        assert!(!sanitized.contains('\n'));
        assert_eq!(sanitized.chars().count(), 240);
    }
}
