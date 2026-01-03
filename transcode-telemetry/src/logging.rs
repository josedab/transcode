//! Structured logging for the transcode system.
//!
//! This module provides structured JSON logging with support for log levels,
//! contextual fields, and integration with the tracing ecosystem.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tracing::{SpanContext, TraceId, SpanId};

/// Log level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Level {
    /// Trace level - most verbose.
    Trace = 0,
    /// Debug level.
    Debug = 1,
    /// Info level.
    #[default]
    Info = 2,
    /// Warning level.
    Warn = 3,
    /// Error level.
    Error = 4,
}

impl Level {
    /// Get the level name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    /// Parse from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(Level::Trace),
            "DEBUG" => Some(Level::Debug),
            "INFO" => Some(Level::Info),
            "WARN" | "WARNING" => Some(Level::Warn),
            "ERROR" => Some(Level::Error),
            _ => None,
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A field value in a log record.
#[derive(Debug, Clone)]
pub enum FieldValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Unsigned integer value.
    Uint(u64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Null value.
    Null,
    /// Array of field values.
    Array(Vec<FieldValue>),
    /// Object/map of field values.
    Object(HashMap<String, FieldValue>),
}

impl FieldValue {
    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        match self {
            FieldValue::String(s) => format!("\"{}\"", escape_json(s)),
            FieldValue::Int(n) => n.to_string(),
            FieldValue::Uint(n) => n.to_string(),
            FieldValue::Float(n) => {
                if n.is_finite() {
                    n.to_string()
                } else {
                    "null".to_string()
                }
            }
            FieldValue::Bool(b) => b.to_string(),
            FieldValue::Null => "null".to_string(),
            FieldValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json()).collect();
                format!("[{}]", items.join(","))
            }
            FieldValue::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json(k), v.to_json()))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

impl From<&str> for FieldValue {
    fn from(s: &str) -> Self {
        FieldValue::String(s.to_string())
    }
}

impl From<String> for FieldValue {
    fn from(s: String) -> Self {
        FieldValue::String(s)
    }
}

impl From<i64> for FieldValue {
    fn from(n: i64) -> Self {
        FieldValue::Int(n)
    }
}

impl From<i32> for FieldValue {
    fn from(n: i32) -> Self {
        FieldValue::Int(n as i64)
    }
}

impl From<u64> for FieldValue {
    fn from(n: u64) -> Self {
        FieldValue::Uint(n)
    }
}

impl From<u32> for FieldValue {
    fn from(n: u32) -> Self {
        FieldValue::Uint(n as u64)
    }
}

impl From<f64> for FieldValue {
    fn from(n: f64) -> Self {
        FieldValue::Float(n)
    }
}

impl From<f32> for FieldValue {
    fn from(n: f32) -> Self {
        FieldValue::Float(n as f64)
    }
}

impl From<bool> for FieldValue {
    fn from(b: bool) -> Self {
        FieldValue::Bool(b)
    }
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

/// A structured log record.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Timestamp.
    pub timestamp: SystemTime,
    /// Log level.
    pub level: Level,
    /// Log message.
    pub message: String,
    /// Target (module path).
    pub target: Option<String>,
    /// File name.
    pub file: Option<String>,
    /// Line number.
    pub line: Option<u32>,
    /// Trace ID for correlation.
    pub trace_id: Option<TraceId>,
    /// Span ID for correlation.
    pub span_id: Option<SpanId>,
    /// Additional fields.
    pub fields: HashMap<String, FieldValue>,
}

impl LogRecord {
    /// Create a new log record.
    pub fn new(level: Level, message: impl Into<String>) -> Self {
        Self {
            timestamp: SystemTime::now(),
            level,
            message: message.into(),
            target: None,
            file: None,
            line: None,
            trace_id: None,
            span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set the target.
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set the file and line.
    pub fn location(mut self, file: impl Into<String>, line: u32) -> Self {
        self.file = Some(file.into());
        self.line = Some(line);
        self
    }

    /// Set the span context.
    pub fn span_context(mut self, context: &SpanContext) -> Self {
        self.trace_id = Some(context.trace_id);
        self.span_id = Some(context.span_id);
        self
    }

    /// Add a field.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Add multiple fields.
    pub fn fields(mut self, fields: impl IntoIterator<Item = (String, FieldValue)>) -> Self {
        for (k, v) in fields {
            self.fields.insert(k, v);
        }
        self
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();

        // Timestamp in ISO 8601 format
        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = timestamp.as_secs();
        let millis = timestamp.subsec_millis();
        parts.push(format!("\"timestamp\":\"{}\"", format_timestamp(secs, millis)));

        // Level
        parts.push(format!("\"level\":\"{}\"", self.level.as_str()));

        // Message
        parts.push(format!("\"message\":\"{}\"", escape_json(&self.message)));

        // Target
        if let Some(ref target) = self.target {
            parts.push(format!("\"target\":\"{}\"", escape_json(target)));
        }

        // Location
        if let Some(ref file) = self.file {
            parts.push(format!("\"file\":\"{}\"", escape_json(file)));
        }
        if let Some(line) = self.line {
            parts.push(format!("\"line\":{}", line));
        }

        // Trace context
        if let Some(ref trace_id) = self.trace_id {
            parts.push(format!("\"trace_id\":\"{}\"", trace_id.to_hex()));
        }
        if let Some(ref span_id) = self.span_id {
            parts.push(format!("\"span_id\":\"{}\"", span_id.to_hex()));
        }

        // Additional fields
        for (key, value) in &self.fields {
            parts.push(format!("\"{}\":{}", escape_json(key), value.to_json()));
        }

        format!("{{{}}}", parts.join(","))
    }
}

/// Format a Unix timestamp as ISO 8601.
fn format_timestamp(secs: u64, millis: u32) -> String {
    // Simple timestamp formatting (UTC)
    let days_since_epoch = secs / 86400;
    let secs_in_day = secs % 86400;
    let hours = secs_in_day / 3600;
    let minutes = (secs_in_day % 3600) / 60;
    let seconds = secs_in_day % 60;

    // Calculate year, month, day (simplified algorithm)
    let mut days = days_since_epoch as i64;
    let mut year = 1970;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &dim in &days_in_months {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }

    let day = days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Log output target.
pub trait LogOutput: Send + Sync {
    /// Write a log record.
    fn write(&self, record: &LogRecord) -> std::io::Result<()>;

    /// Flush any buffered output.
    fn flush(&self) -> std::io::Result<()>;
}

/// Standard output logger.
#[derive(Debug)]
pub struct StdoutOutput {
    /// Whether to use JSON format.
    json: bool,
}

impl StdoutOutput {
    /// Create a new stdout output.
    pub fn new() -> Self {
        Self { json: true }
    }

    /// Set JSON output mode.
    pub fn json(mut self, enabled: bool) -> Self {
        self.json = enabled;
        self
    }
}

impl Default for StdoutOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl LogOutput for StdoutOutput {
    fn write(&self, record: &LogRecord) -> std::io::Result<()> {
        let output = if self.json {
            record.to_json()
        } else {
            format!(
                "{} {} {}",
                format_timestamp(
                    record.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                    record.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().subsec_millis()
                ),
                record.level,
                record.message
            )
        };
        println!("{}", output);
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

/// File logger.
pub struct FileOutput {
    /// File path.
    path: String,
    /// File handle.
    file: RwLock<std::fs::File>,
    /// Whether to use JSON format.
    json: bool,
}

impl FileOutput {
    /// Create a new file output.
    pub fn new(path: impl Into<String>) -> std::io::Result<Self> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        Ok(Self {
            path,
            file: RwLock::new(file),
            json: true,
        })
    }

    /// Set JSON output mode.
    pub fn json(mut self, enabled: bool) -> Self {
        self.json = enabled;
        self
    }

    /// Get the file path.
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl LogOutput for FileOutput {
    fn write(&self, record: &LogRecord) -> std::io::Result<()> {
        let output = if self.json {
            format!("{}\n", record.to_json())
        } else {
            format!(
                "{} {} {}\n",
                format_timestamp(
                    record.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                    record.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().subsec_millis()
                ),
                record.level,
                record.message
            )
        };

        let mut file = self.file.write().unwrap();
        file.write_all(output.as_bytes())
    }

    fn flush(&self) -> std::io::Result<()> {
        self.file.write().unwrap().flush()
    }
}

/// In-memory log buffer for testing.
#[derive(Debug, Default)]
pub struct MemoryOutput {
    records: RwLock<Vec<LogRecord>>,
}

impl MemoryOutput {
    /// Create a new memory output.
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
        }
    }

    /// Get all records.
    pub fn records(&self) -> Vec<LogRecord> {
        self.records.read().unwrap().clone()
    }

    /// Clear all records.
    pub fn clear(&self) {
        self.records.write().unwrap().clear();
    }

    /// Get record count.
    pub fn len(&self) -> usize {
        self.records.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.records.read().unwrap().is_empty()
    }
}

impl LogOutput for MemoryOutput {
    fn write(&self, record: &LogRecord) -> std::io::Result<()> {
        self.records.write().unwrap().push(record.clone());
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Logger configuration.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Minimum log level.
    pub level: Level,
    /// Service name.
    pub service_name: String,
    /// Service version.
    pub service_version: Option<String>,
    /// Environment (e.g., "production", "development").
    pub environment: Option<String>,
    /// Default fields to add to all logs.
    pub default_fields: HashMap<String, FieldValue>,
}

impl LoggerConfig {
    /// Create a new logger configuration.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            level: Level::Info,
            service_name: service_name.into(),
            service_version: None,
            environment: None,
            default_fields: HashMap::new(),
        }
    }

    /// Set the minimum log level.
    pub fn level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Set the service version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.service_version = Some(version.into());
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.environment = Some(env.into());
        self
    }

    /// Add a default field.
    pub fn default_field(mut self, key: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.default_fields.insert(key.into(), value.into());
        self
    }
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self::new("transcode")
    }
}

/// The main logger.
pub struct Logger {
    /// Logger configuration.
    config: LoggerConfig,
    /// Log outputs.
    outputs: Vec<Arc<dyn LogOutput>>,
    /// Current span context for trace correlation.
    span_context: RwLock<Option<SpanContext>>,
}

impl Logger {
    /// Create a new logger.
    pub fn new(config: LoggerConfig) -> Self {
        Self {
            config,
            outputs: Vec::new(),
            span_context: RwLock::new(None),
        }
    }

    /// Add an output.
    pub fn add_output(&mut self, output: Arc<dyn LogOutput>) {
        self.outputs.push(output);
    }

    /// Set the current span context for trace correlation.
    pub fn set_span_context(&self, context: Option<SpanContext>) {
        *self.span_context.write().unwrap() = context;
    }

    /// Get the current span context.
    pub fn span_context(&self) -> Option<SpanContext> {
        self.span_context.read().unwrap().clone()
    }

    /// Check if a level is enabled.
    pub fn is_enabled(&self, level: Level) -> bool {
        level >= self.config.level
    }

    /// Log a record.
    pub fn log(&self, mut record: LogRecord) {
        if !self.is_enabled(record.level) {
            return;
        }

        // Add service info
        record.fields.insert(
            "service.name".to_string(),
            FieldValue::String(self.config.service_name.clone()),
        );

        if let Some(ref version) = self.config.service_version {
            record.fields.insert(
                "service.version".to_string(),
                FieldValue::String(version.clone()),
            );
        }

        if let Some(ref env) = self.config.environment {
            record.fields.insert(
                "environment".to_string(),
                FieldValue::String(env.clone()),
            );
        }

        // Add default fields
        for (k, v) in &self.config.default_fields {
            record.fields.entry(k.clone()).or_insert_with(|| v.clone());
        }

        // Add span context if not already set
        if record.trace_id.is_none() {
            if let Some(ref ctx) = *self.span_context.read().unwrap() {
                record.trace_id = Some(ctx.trace_id);
                record.span_id = Some(ctx.span_id);
            }
        }

        // Write to all outputs
        for output in &self.outputs {
            let _ = output.write(&record);
        }
    }

    /// Log at trace level.
    pub fn trace(&self, message: impl Into<String>) {
        self.log(LogRecord::new(Level::Trace, message));
    }

    /// Log at debug level.
    pub fn debug(&self, message: impl Into<String>) {
        self.log(LogRecord::new(Level::Debug, message));
    }

    /// Log at info level.
    pub fn info(&self, message: impl Into<String>) {
        self.log(LogRecord::new(Level::Info, message));
    }

    /// Log at warn level.
    pub fn warn(&self, message: impl Into<String>) {
        self.log(LogRecord::new(Level::Warn, message));
    }

    /// Log at error level.
    pub fn error(&self, message: impl Into<String>) {
        self.log(LogRecord::new(Level::Error, message));
    }

    /// Create a log record builder at trace level.
    pub fn trace_builder(&self, message: impl Into<String>) -> LogRecordBuilder<'_> {
        LogRecordBuilder::new(self, Level::Trace, message)
    }

    /// Create a log record builder at debug level.
    pub fn debug_builder(&self, message: impl Into<String>) -> LogRecordBuilder<'_> {
        LogRecordBuilder::new(self, Level::Debug, message)
    }

    /// Create a log record builder at info level.
    pub fn info_builder(&self, message: impl Into<String>) -> LogRecordBuilder<'_> {
        LogRecordBuilder::new(self, Level::Info, message)
    }

    /// Create a log record builder at warn level.
    pub fn warn_builder(&self, message: impl Into<String>) -> LogRecordBuilder<'_> {
        LogRecordBuilder::new(self, Level::Warn, message)
    }

    /// Create a log record builder at error level.
    pub fn error_builder(&self, message: impl Into<String>) -> LogRecordBuilder<'_> {
        LogRecordBuilder::new(self, Level::Error, message)
    }

    /// Flush all outputs.
    pub fn flush(&self) {
        for output in &self.outputs {
            let _ = output.flush();
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &LoggerConfig {
        &self.config
    }
}

/// Builder for log records.
pub struct LogRecordBuilder<'a> {
    logger: &'a Logger,
    record: LogRecord,
}

impl<'a> LogRecordBuilder<'a> {
    /// Create a new builder.
    pub fn new(logger: &'a Logger, level: Level, message: impl Into<String>) -> Self {
        Self {
            logger,
            record: LogRecord::new(level, message),
        }
    }

    /// Set the target.
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.record.target = Some(target.into());
        self
    }

    /// Set the location.
    pub fn location(mut self, file: impl Into<String>, line: u32) -> Self {
        self.record.file = Some(file.into());
        self.record.line = Some(line);
        self
    }

    /// Set the span context.
    pub fn span_context(mut self, context: &SpanContext) -> Self {
        self.record.trace_id = Some(context.trace_id);
        self.record.span_id = Some(context.span_id);
        self
    }

    /// Add a field.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.record.fields.insert(key.into(), value.into());
        self
    }

    /// Add a job ID field.
    pub fn job_id(self, job_id: impl Into<String>) -> Self {
        self.field("job_id", job_id.into())
    }

    /// Add a codec field.
    pub fn codec(self, codec: impl Into<String>) -> Self {
        self.field("codec", codec.into())
    }

    /// Add a frame number field.
    pub fn frame(self, frame: u64) -> Self {
        self.field("frame", frame)
    }

    /// Add an error field.
    pub fn error(self, err: &dyn std::error::Error) -> Self {
        self.field("error", err.to_string())
    }

    /// Log the record.
    pub fn log(self) {
        self.logger.log(self.record);
    }
}

/// Transcode-specific logging helpers.
pub struct TranscodeLogger {
    logger: Arc<Logger>,
}

impl TranscodeLogger {
    /// Create a new transcode logger.
    pub fn new(logger: Arc<Logger>) -> Self {
        Self { logger }
    }

    /// Log job start.
    pub fn log_job_start(&self, job_id: &str, input_path: &str, output_path: &str) {
        self.logger
            .info_builder("Transcode job started")
            .job_id(job_id)
            .field("input_path", input_path)
            .field("output_path", output_path)
            .log();
    }

    /// Log job complete.
    pub fn log_job_complete(&self, job_id: &str, duration_secs: f64, frames: u64) {
        self.logger
            .info_builder("Transcode job completed")
            .job_id(job_id)
            .field("duration_secs", duration_secs)
            .field("frames", frames)
            .log();
    }

    /// Log job failure.
    pub fn log_job_failure(&self, job_id: &str, error: &dyn std::error::Error) {
        self.logger
            .error_builder("Transcode job failed")
            .job_id(job_id)
            .error(error)
            .log();
    }

    /// Log frame processed.
    pub fn log_frame(&self, job_id: &str, frame: u64, fps: f64) {
        self.logger
            .debug_builder("Frame processed")
            .job_id(job_id)
            .frame(frame)
            .field("fps", fps)
            .log();
    }

    /// Log encode event.
    pub fn log_encode(&self, job_id: &str, codec: &str, frame: u64, duration_ms: f64) {
        self.logger
            .trace_builder("Frame encoded")
            .job_id(job_id)
            .codec(codec)
            .frame(frame)
            .field("duration_ms", duration_ms)
            .log();
    }

    /// Log decode event.
    pub fn log_decode(&self, job_id: &str, codec: &str, frame: u64, duration_ms: f64) {
        self.logger
            .trace_builder("Frame decoded")
            .job_id(job_id)
            .codec(codec)
            .frame(frame)
            .field("duration_ms", duration_ms)
            .log();
    }

    /// Log progress update.
    pub fn log_progress(&self, job_id: &str, percentage: f64) {
        self.logger
            .debug_builder("Progress update")
            .job_id(job_id)
            .field("progress_percent", percentage)
            .log();
    }
}

/// Macro for structured info-level logging.
///
/// # Example
/// ```ignore
/// log_info!(logger, "Job started", job_id = "123", status = "running");
/// ```
#[macro_export]
macro_rules! log_info {
    ($logger:expr, $msg:expr $(, $key:ident = $value:expr)*) => {{
        let record = $crate::logging::LogRecord::new($crate::logging::Level::Info, $msg)
            $(.field(stringify!($key), $value))*;
        $logger.log(record);
    }};
}

/// Macro for structured error-level logging.
///
/// # Example
/// ```ignore
/// log_error!(logger, "Job failed", job_id = "123", error = "timeout");
/// ```
#[macro_export]
macro_rules! log_error {
    ($logger:expr, $msg:expr $(, $key:ident = $value:expr)*) => {{
        let record = $crate::logging::LogRecord::new($crate::logging::Level::Error, $msg)
            $(.field(stringify!($key), $value))*;
        $logger.log(record);
    }};
}

/// Macro for structured debug-level logging.
///
/// # Example
/// ```ignore
/// log_debug!(logger, "Processing frame", frame = 100, fps = 30.0);
/// ```
#[macro_export]
macro_rules! log_debug {
    ($logger:expr, $msg:expr $(, $key:ident = $value:expr)*) => {{
        let record = $crate::logging::LogRecord::new($crate::logging::Level::Debug, $msg)
            $(.field(stringify!($key), $value))*;
        $logger.log(record);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_ordering() {
        assert!(Level::Trace < Level::Debug);
        assert!(Level::Debug < Level::Info);
        assert!(Level::Info < Level::Warn);
        assert!(Level::Warn < Level::Error);
    }

    #[test]
    fn test_level_parsing() {
        assert_eq!(Level::parse("INFO"), Some(Level::Info));
        assert_eq!(Level::parse("info"), Some(Level::Info));
        assert_eq!(Level::parse("WARNING"), Some(Level::Warn));
        assert_eq!(Level::parse("UNKNOWN"), None);
    }

    #[test]
    fn test_field_value_json() {
        assert_eq!(FieldValue::String("test".to_string()).to_json(), "\"test\"");
        assert_eq!(FieldValue::Int(42).to_json(), "42");
        assert_eq!(FieldValue::Float(3.14).to_json(), "3.14");
        assert_eq!(FieldValue::Bool(true).to_json(), "true");
        assert_eq!(FieldValue::Null.to_json(), "null");
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn test_log_record_to_json() {
        let record = LogRecord::new(Level::Info, "test message")
            .field("key1", "value1")
            .field("key2", 42i64);

        let json = record.to_json();
        assert!(json.contains("\"level\":\"INFO\""));
        assert!(json.contains("\"message\":\"test message\""));
        assert!(json.contains("\"key1\":\"value1\""));
        assert!(json.contains("\"key2\":42"));
    }

    #[test]
    fn test_memory_output() {
        let output = Arc::new(MemoryOutput::new());

        let record = LogRecord::new(Level::Info, "test");
        output.write(&record).unwrap();

        assert_eq!(output.len(), 1);
        assert!(!output.is_empty());

        let records = output.records();
        assert_eq!(records[0].message, "test");

        output.clear();
        assert!(output.is_empty());
    }

    #[test]
    fn test_logger() {
        let config = LoggerConfig::new("test-service")
            .level(Level::Debug)
            .version("1.0.0")
            .environment("test");

        let mut logger = Logger::new(config);
        let output = Arc::new(MemoryOutput::new());
        logger.add_output(output.clone());

        // Should be logged
        logger.info("info message");
        logger.debug("debug message");

        // Should NOT be logged (below minimum level)
        logger.trace("trace message");

        let records = output.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].level, Level::Info);
        assert_eq!(records[1].level, Level::Debug);
    }

    #[test]
    fn test_logger_with_span_context() {
        let config = LoggerConfig::new("test-service");
        let mut logger = Logger::new(config);
        let output = Arc::new(MemoryOutput::new());
        logger.add_output(output.clone());

        let ctx = SpanContext::new_root();
        logger.set_span_context(Some(ctx.clone()));

        logger.info("test message");

        let records = output.records();
        assert!(records[0].trace_id.is_some());
        assert!(records[0].span_id.is_some());
    }

    #[test]
    fn test_log_record_builder() {
        let config = LoggerConfig::new("test-service");
        let mut logger = Logger::new(config);
        let output = Arc::new(MemoryOutput::new());
        logger.add_output(output.clone());

        logger
            .info_builder("Job started")
            .job_id("job-123")
            .codec("h264")
            .field("custom", "value")
            .log();

        let records = output.records();
        assert_eq!(records.len(), 1);

        let record = &records[0];
        assert!(record.fields.contains_key("job_id"));
        assert!(record.fields.contains_key("codec"));
        assert!(record.fields.contains_key("custom"));
    }

    #[test]
    fn test_transcode_logger() {
        let config = LoggerConfig::new("test-service").level(Level::Trace);
        let mut logger = Logger::new(config);
        let output = Arc::new(MemoryOutput::new());
        logger.add_output(output.clone());

        let transcode_logger = TranscodeLogger::new(Arc::new(logger));

        transcode_logger.log_job_start("job-123", "/input.mp4", "/output.mp4");
        transcode_logger.log_progress("job-123", 50.0);
        transcode_logger.log_job_complete("job-123", 60.0, 1800);

        let records = output.records();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_logger_config_default_fields() {
        let config = LoggerConfig::new("test-service")
            .default_field("host", "server1")
            .default_field("region", "us-east-1");

        let mut logger = Logger::new(config);
        let output = Arc::new(MemoryOutput::new());
        logger.add_output(output.clone());

        logger.info("test message");

        let records = output.records();
        assert!(records[0].fields.contains_key("host"));
        assert!(records[0].fields.contains_key("region"));
    }

    #[test]
    fn test_stdout_output() {
        let output = StdoutOutput::new().json(true);
        let record = LogRecord::new(Level::Info, "test");
        // Just verify it doesn't panic
        assert!(output.write(&record).is_ok());
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2023));
    }
}
