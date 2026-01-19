//! Distributed tracing for the transcode system.
//!
//! This module provides distributed tracing capabilities using OpenTelemetry-compatible
//! spans for tracking operations across the transcode pipeline stages.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// Error types available but not used in this standalone implementation
#[allow(unused_imports)]
use crate::error::{Result, TracingError};

/// Trace ID (128-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

impl TraceId {
    /// Create a new random trace ID.
    pub fn new() -> Self {
        let mut bytes = [0u8; 16];
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let random: u64 = rand_simple();

        bytes[0..8].copy_from_slice(&now.to_be_bytes()[8..16]);
        bytes[8..16].copy_from_slice(&random.to_be_bytes());

        Self(bytes)
    }

    /// Create a trace ID from bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Get the bytes of the trace ID.
    pub fn to_bytes(&self) -> [u8; 16] {
        self.0
    }

    /// Convert to hex string.
    #[allow(clippy::format_collect)]
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 {
            return None;
        }

        let mut bytes = [0u8; 16];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let s = std::str::from_utf8(chunk).ok()?;
            bytes[i] = u8::from_str_radix(s, 16).ok()?;
        }

        Some(Self(bytes))
    }

    /// Check if the trace ID is valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.0.iter().any(|&b| b != 0)
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Span ID (64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// Create a new random span ID.
    pub fn new() -> Self {
        let random: u64 = rand_simple();
        Self(random.to_be_bytes())
    }

    /// Create a span ID from bytes.
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Get the bytes of the span ID.
    pub fn to_bytes(&self) -> [u8; 8] {
        self.0
    }

    /// Convert to hex string.
    #[allow(clippy::format_collect)]
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 16 {
            return None;
        }

        let mut bytes = [0u8; 8];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let s = std::str::from_utf8(chunk).ok()?;
            bytes[i] = u8::from_str_radix(s, 16).ok()?;
        }

        Some(Self(bytes))
    }

    /// Check if the span ID is valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.0.iter().any(|&b| b != 0)
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Simple random number generator for IDs.
fn rand_simple() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    time.wrapping_mul(6364136223846793005).wrapping_add(counter)
}

/// Span context for propagation.
#[derive(Debug, Clone)]
pub struct SpanContext {
    /// Trace ID.
    pub trace_id: TraceId,
    /// Span ID.
    pub span_id: SpanId,
    /// Parent span ID.
    pub parent_span_id: Option<SpanId>,
    /// Trace flags (sampled, etc.).
    pub trace_flags: u8,
    /// Trace state for vendor-specific data.
    pub trace_state: HashMap<String, String>,
}

impl SpanContext {
    /// Create a new root span context.
    pub fn new_root() -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            trace_flags: 0x01, // sampled by default
            trace_state: HashMap::new(),
        }
    }

    /// Create a child span context.
    pub fn new_child(parent: &SpanContext) -> Self {
        Self {
            trace_id: parent.trace_id,
            span_id: SpanId::new(),
            parent_span_id: Some(parent.span_id),
            trace_flags: parent.trace_flags,
            trace_state: parent.trace_state.clone(),
        }
    }

    /// Check if the span is sampled.
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    /// Set sampled flag.
    pub fn set_sampled(&mut self, sampled: bool) {
        if sampled {
            self.trace_flags |= 0x01;
        } else {
            self.trace_flags &= !0x01;
        }
    }

    /// Generate W3C Trace Context traceparent header.
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id.to_hex(),
            self.span_id.to_hex(),
            self.trace_flags
        )
    }

    /// Parse W3C Trace Context traceparent header.
    pub fn from_traceparent(traceparent: &str) -> Option<Self> {
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 || parts[0] != "00" {
            return None;
        }

        let trace_id = TraceId::from_hex(parts[1])?;
        let span_id = SpanId::from_hex(parts[2])?;
        let trace_flags = u8::from_str_radix(parts[3], 16).ok()?;

        Some(Self {
            trace_id,
            span_id,
            parent_span_id: None,
            trace_flags,
            trace_state: HashMap::new(),
        })
    }
}

/// Span status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpanStatus {
    /// Unset status.
    #[default]
    Unset,
    /// Success.
    Ok,
    /// Error.
    Error,
}

/// Span kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpanKind {
    /// Internal operation.
    #[default]
    Internal,
    /// Server handling a request.
    Server,
    /// Client making a request.
    Client,
    /// Producer sending a message.
    Producer,
    /// Consumer receiving a message.
    Consumer,
}

/// Span attribute value.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// String array.
    StringArray(Vec<String>),
    /// Integer array.
    IntArray(Vec<i64>),
    /// Float array.
    FloatArray(Vec<f64>),
    /// Boolean array.
    BoolArray(Vec<bool>),
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        AttributeValue::String(s.to_string())
    }
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        AttributeValue::String(s)
    }
}

impl From<i64> for AttributeValue {
    fn from(n: i64) -> Self {
        AttributeValue::Int(n)
    }
}

impl From<i32> for AttributeValue {
    fn from(n: i32) -> Self {
        AttributeValue::Int(n as i64)
    }
}

impl From<f64> for AttributeValue {
    fn from(n: f64) -> Self {
        AttributeValue::Float(n)
    }
}

impl From<bool> for AttributeValue {
    fn from(b: bool) -> Self {
        AttributeValue::Bool(b)
    }
}

/// Span event.
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// Event timestamp.
    pub timestamp: SystemTime,
    /// Event attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

impl SpanEvent {
    /// Create a new span event.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: SystemTime::now(),
            attributes: HashMap::new(),
        }
    }

    /// Create an event with attributes.
    pub fn with_attributes(
        name: impl Into<String>,
        attributes: HashMap<String, AttributeValue>,
    ) -> Self {
        Self {
            name: name.into(),
            timestamp: SystemTime::now(),
            attributes,
        }
    }
}

/// Span link to another trace.
#[derive(Debug, Clone)]
pub struct SpanLink {
    /// Linked span context.
    pub context: SpanContext,
    /// Link attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

/// A tracing span.
#[derive(Debug)]
pub struct Span {
    /// Span name.
    name: String,
    /// Span context.
    context: SpanContext,
    /// Span kind.
    kind: SpanKind,
    /// Start time.
    start_time: Instant,
    /// Start time as SystemTime for export.
    start_system_time: SystemTime,
    /// End time.
    end_time: RwLock<Option<Instant>>,
    /// Span status.
    status: RwLock<SpanStatus>,
    /// Status message.
    status_message: RwLock<Option<String>>,
    /// Span attributes.
    attributes: RwLock<HashMap<String, AttributeValue>>,
    /// Span events.
    events: RwLock<Vec<SpanEvent>>,
    /// Span links.
    links: RwLock<Vec<SpanLink>>,
}

impl Span {
    /// Create a new root span.
    pub fn new_root(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: SpanContext::new_root(),
            kind: SpanKind::Internal,
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
            end_time: RwLock::new(None),
            status: RwLock::new(SpanStatus::Unset),
            status_message: RwLock::new(None),
            attributes: RwLock::new(HashMap::new()),
            events: RwLock::new(Vec::new()),
            links: RwLock::new(Vec::new()),
        }
    }

    /// Create a child span.
    pub fn new_child(name: impl Into<String>, parent: &Span) -> Self {
        Self {
            name: name.into(),
            context: SpanContext::new_child(&parent.context),
            kind: SpanKind::Internal,
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
            end_time: RwLock::new(None),
            status: RwLock::new(SpanStatus::Unset),
            status_message: RwLock::new(None),
            attributes: RwLock::new(HashMap::new()),
            events: RwLock::new(Vec::new()),
            links: RwLock::new(Vec::new()),
        }
    }

    /// Create a child span from context.
    pub fn new_child_from_context(name: impl Into<String>, parent_context: &SpanContext) -> Self {
        Self {
            name: name.into(),
            context: SpanContext::new_child(parent_context),
            kind: SpanKind::Internal,
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
            end_time: RwLock::new(None),
            status: RwLock::new(SpanStatus::Unset),
            status_message: RwLock::new(None),
            attributes: RwLock::new(HashMap::new()),
            events: RwLock::new(Vec::new()),
            links: RwLock::new(Vec::new()),
        }
    }

    /// Get the span name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the span context.
    pub fn context(&self) -> &SpanContext {
        &self.context
    }

    /// Get the span kind.
    pub fn kind(&self) -> SpanKind {
        self.kind
    }

    /// Set the span kind.
    pub fn set_kind(&mut self, kind: SpanKind) {
        self.kind = kind;
    }

    /// Get the duration of the span.
    pub fn duration(&self) -> Duration {
        self.end_time
            .read()
            .unwrap()
            .map(|end| end.duration_since(self.start_time))
            .unwrap_or_else(|| self.start_time.elapsed())
    }

    /// Check if the span has ended.
    pub fn is_ended(&self) -> bool {
        self.end_time.read().unwrap().is_some()
    }

    /// End the span.
    pub fn end(&self) {
        let mut end_time = self.end_time.write().unwrap();
        if end_time.is_none() {
            *end_time = Some(Instant::now());
        }
    }

    /// Set the span status.
    pub fn set_status(&self, status: SpanStatus, message: Option<String>) {
        *self.status.write().unwrap() = status;
        *self.status_message.write().unwrap() = message;
    }

    /// Get the span status.
    pub fn status(&self) -> SpanStatus {
        *self.status.read().unwrap()
    }

    /// Set an attribute.
    pub fn set_attribute(&self, key: impl Into<String>, value: impl Into<AttributeValue>) {
        self.attributes
            .write()
            .unwrap()
            .insert(key.into(), value.into());
    }

    /// Set multiple attributes.
    pub fn set_attributes(&self, attrs: impl IntoIterator<Item = (String, AttributeValue)>) {
        let mut attributes = self.attributes.write().unwrap();
        for (k, v) in attrs {
            attributes.insert(k, v);
        }
    }

    /// Get an attribute.
    pub fn get_attribute(&self, key: &str) -> Option<AttributeValue> {
        self.attributes.read().unwrap().get(key).cloned()
    }

    /// Get all attributes.
    pub fn attributes(&self) -> HashMap<String, AttributeValue> {
        self.attributes.read().unwrap().clone()
    }

    /// Add an event.
    pub fn add_event(&self, event: SpanEvent) {
        self.events.write().unwrap().push(event);
    }

    /// Add a named event.
    pub fn add_event_with_name(&self, name: impl Into<String>) {
        self.add_event(SpanEvent::new(name));
    }

    /// Get all events.
    pub fn events(&self) -> Vec<SpanEvent> {
        self.events.read().unwrap().clone()
    }

    /// Add a link.
    pub fn add_link(&self, link: SpanLink) {
        self.links.write().unwrap().push(link);
    }

    /// Get all links.
    pub fn links(&self) -> Vec<SpanLink> {
        self.links.read().unwrap().clone()
    }

    /// Record an exception.
    pub fn record_exception(&self, error: &dyn std::error::Error) {
        let mut attrs = HashMap::new();
        attrs.insert(
            "exception.type".to_string(),
            AttributeValue::String(std::any::type_name_of_val(error).to_string()),
        );
        attrs.insert(
            "exception.message".to_string(),
            AttributeValue::String(error.to_string()),
        );

        self.add_event(SpanEvent::with_attributes("exception", attrs));
        self.set_status(SpanStatus::Error, Some(error.to_string()));
    }

    /// Get start time as SystemTime.
    pub fn start_system_time(&self) -> SystemTime {
        self.start_system_time
    }

    /// Get end time as SystemTime.
    pub fn end_system_time(&self) -> Option<SystemTime> {
        self.end_time.read().unwrap().map(|end| {
            let duration = end.duration_since(self.start_time);
            self.start_system_time + duration
        })
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.end();
    }
}

/// Transcode pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Input stage (reading source).
    Input,
    /// Demux stage (container parsing).
    Demux,
    /// Decode stage (video/audio decoding).
    Decode,
    /// Filter stage (processing filters).
    Filter,
    /// Encode stage (video/audio encoding).
    Encode,
    /// Mux stage (container writing).
    Mux,
    /// Output stage (writing destination).
    Output,
}

impl PipelineStage {
    /// Get the stage name.
    pub fn name(&self) -> &'static str {
        match self {
            PipelineStage::Input => "input",
            PipelineStage::Demux => "demux",
            PipelineStage::Decode => "decode",
            PipelineStage::Filter => "filter",
            PipelineStage::Encode => "encode",
            PipelineStage::Mux => "mux",
            PipelineStage::Output => "output",
        }
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Builder for creating spans with common attributes.
#[derive(Debug)]
pub struct SpanBuilder {
    name: String,
    kind: SpanKind,
    attributes: HashMap<String, AttributeValue>,
    parent: Option<SpanContext>,
}

impl SpanBuilder {
    /// Create a new span builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: SpanKind::Internal,
            attributes: HashMap::new(),
            parent: None,
        }
    }

    /// Set the span kind.
    pub fn kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the parent context.
    pub fn parent(mut self, parent: &SpanContext) -> Self {
        self.parent = Some(parent.clone());
        self
    }

    /// Set an attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<AttributeValue>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Set the job ID attribute.
    pub fn job_id(self, job_id: impl Into<String>) -> Self {
        self.attribute("transcode.job.id", job_id.into())
    }

    /// Set the codec attribute.
    pub fn codec(self, codec: impl Into<String>) -> Self {
        self.attribute("transcode.codec", codec.into())
    }

    /// Set the pipeline stage attribute.
    pub fn stage(self, stage: PipelineStage) -> Self {
        self.attribute("transcode.pipeline.stage", stage.name())
    }

    /// Set the frame number attribute.
    pub fn frame(self, frame: u64) -> Self {
        self.attribute("transcode.frame", frame as i64)
    }

    /// Build the span.
    pub fn build(mut self) -> Span {
        let mut span = match &self.parent {
            Some(parent) => Span::new_child_from_context(&self.name, parent),
            None => Span::new_root(&self.name),
        };

        span.set_kind(self.kind);

        for (k, v) in self.attributes.drain() {
            span.set_attribute(k, v);
        }

        span
    }
}

/// Tracer for creating and managing spans.
#[derive(Debug)]
pub struct Tracer {
    /// Service name.
    service_name: String,
    /// Service version.
    service_version: Option<String>,
    /// Active spans.
    active_spans: RwLock<HashMap<SpanId, Arc<Span>>>,
    /// Completed spans for export.
    completed_spans: RwLock<Vec<Arc<Span>>>,
    /// Maximum completed spans to buffer.
    max_completed_spans: usize,
}

impl Tracer {
    /// Create a new tracer.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            service_version: None,
            active_spans: RwLock::new(HashMap::new()),
            completed_spans: RwLock::new(Vec::new()),
            max_completed_spans: 1000,
        }
    }

    /// Set the service version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = Some(version.into());
        self
    }

    /// Set the maximum completed spans buffer size.
    pub fn with_max_spans(mut self, max: usize) -> Self {
        self.max_completed_spans = max;
        self
    }

    /// Get the service name.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Get the service version.
    pub fn service_version(&self) -> Option<&str> {
        self.service_version.as_deref()
    }

    /// Start a new root span.
    pub fn start_span(&self, name: impl Into<String>) -> Arc<Span> {
        let span = Arc::new(Span::new_root(name));
        self.active_spans
            .write()
            .unwrap()
            .insert(span.context.span_id, span.clone());
        span
    }

    /// Start a new child span.
    pub fn start_child_span(&self, name: impl Into<String>, parent: &Span) -> Arc<Span> {
        let span = Arc::new(Span::new_child(name, parent));
        self.active_spans
            .write()
            .unwrap()
            .insert(span.context.span_id, span.clone());
        span
    }

    /// Start a span using a builder.
    pub fn start_span_with_builder(&self, builder: SpanBuilder) -> Arc<Span> {
        let span = Arc::new(builder.build());
        self.active_spans
            .write()
            .unwrap()
            .insert(span.context.span_id, span.clone());
        span
    }

    /// End a span.
    pub fn end_span(&self, span: &Arc<Span>) {
        span.end();

        // Move from active to completed
        let span_id = span.context.span_id;
        self.active_spans.write().unwrap().remove(&span_id);

        let mut completed = self.completed_spans.write().unwrap();
        completed.push(span.clone());

        // Trim if over limit
        while completed.len() > self.max_completed_spans {
            completed.remove(0);
        }
    }

    /// Get an active span by ID.
    pub fn get_span(&self, span_id: &SpanId) -> Option<Arc<Span>> {
        self.active_spans.read().unwrap().get(span_id).cloned()
    }

    /// Get all active spans.
    pub fn active_spans(&self) -> Vec<Arc<Span>> {
        self.active_spans.read().unwrap().values().cloned().collect()
    }

    /// Drain completed spans for export.
    pub fn drain_completed(&self) -> Vec<Arc<Span>> {
        std::mem::take(&mut *self.completed_spans.write().unwrap())
    }

    /// Get completed spans count.
    pub fn completed_count(&self) -> usize {
        self.completed_spans.read().unwrap().len()
    }

    /// Create a pipeline stage span.
    pub fn start_pipeline_span(
        &self,
        stage: PipelineStage,
        job_id: &str,
        parent: Option<&Span>,
    ) -> Arc<Span> {
        let mut builder = SpanBuilder::new(format!("transcode.{}", stage.name()))
            .stage(stage)
            .job_id(job_id);

        if let Some(p) = parent {
            builder = builder.parent(&p.context);
        }

        self.start_span_with_builder(builder)
    }

    /// Create a codec operation span.
    pub fn start_codec_span(
        &self,
        operation: &str,
        codec: &str,
        parent: Option<&Span>,
    ) -> Arc<Span> {
        let mut builder = SpanBuilder::new(format!("codec.{}", operation)).codec(codec);

        if let Some(p) = parent {
            builder = builder.parent(&p.context);
        }

        self.start_span_with_builder(builder)
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new("transcode")
    }
}

/// Context propagation for distributed tracing.
pub struct ContextPropagator;

impl ContextPropagator {
    /// Extract span context from headers.
    pub fn extract(headers: &HashMap<String, String>) -> Option<SpanContext> {
        headers
            .get("traceparent")
            .and_then(|tp| SpanContext::from_traceparent(tp))
    }

    /// Inject span context into headers.
    pub fn inject(context: &SpanContext, headers: &mut HashMap<String, String>) {
        headers.insert("traceparent".to_string(), context.to_traceparent());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id() {
        let trace_id = TraceId::new();
        assert!(trace_id.is_valid());

        let hex = trace_id.to_hex();
        assert_eq!(hex.len(), 32);

        let parsed = TraceId::from_hex(&hex).unwrap();
        assert_eq!(trace_id.to_bytes(), parsed.to_bytes());
    }

    #[test]
    fn test_span_id() {
        let span_id = SpanId::new();
        assert!(span_id.is_valid());

        let hex = span_id.to_hex();
        assert_eq!(hex.len(), 16);

        let parsed = SpanId::from_hex(&hex).unwrap();
        assert_eq!(span_id.to_bytes(), parsed.to_bytes());
    }

    #[test]
    fn test_span_context() {
        let ctx = SpanContext::new_root();
        assert!(ctx.trace_id.is_valid());
        assert!(ctx.span_id.is_valid());
        assert!(ctx.parent_span_id.is_none());
        assert!(ctx.is_sampled());

        let child = SpanContext::new_child(&ctx);
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_eq!(child.parent_span_id, Some(ctx.span_id));
    }

    #[test]
    fn test_traceparent() {
        let ctx = SpanContext::new_root();
        let traceparent = ctx.to_traceparent();

        let parts: Vec<&str> = traceparent.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "00");

        let parsed = SpanContext::from_traceparent(&traceparent).unwrap();
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
    }

    #[test]
    fn test_span_basic() {
        let span = Span::new_root("test_operation");
        assert_eq!(span.name(), "test_operation");
        assert!(!span.is_ended());

        span.set_attribute("key", "value");
        assert!(matches!(
            span.get_attribute("key"),
            Some(AttributeValue::String(s)) if s == "value"
        ));

        span.end();
        assert!(span.is_ended());
    }

    #[test]
    fn test_span_child() {
        let parent = Span::new_root("parent");
        let child = Span::new_child("child", &parent);

        assert_eq!(child.context.trace_id, parent.context.trace_id);
        assert_eq!(child.context.parent_span_id, Some(parent.context.span_id));
    }

    #[test]
    fn test_span_events() {
        let span = Span::new_root("test");

        span.add_event_with_name("event1");
        span.add_event(SpanEvent::new("event2"));

        let events = span.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "event1");
        assert_eq!(events[1].name, "event2");
    }

    #[test]
    fn test_span_status() {
        let span = Span::new_root("test");
        assert_eq!(span.status(), SpanStatus::Unset);

        span.set_status(SpanStatus::Ok, None);
        assert_eq!(span.status(), SpanStatus::Ok);

        span.set_status(SpanStatus::Error, Some("error message".to_string()));
        assert_eq!(span.status(), SpanStatus::Error);
    }

    #[test]
    fn test_span_builder() {
        let span = SpanBuilder::new("test_span")
            .kind(SpanKind::Server)
            .job_id("job-123")
            .codec("h264")
            .stage(PipelineStage::Encode)
            .build();

        assert_eq!(span.name(), "test_span");
        assert_eq!(span.kind(), SpanKind::Server);
        assert!(matches!(
            span.get_attribute("transcode.job.id"),
            Some(AttributeValue::String(s)) if s == "job-123"
        ));
    }

    #[test]
    fn test_tracer() {
        let tracer = Tracer::new("test-service").with_version("1.0.0");
        assert_eq!(tracer.service_name(), "test-service");
        assert_eq!(tracer.service_version(), Some("1.0.0"));

        let span = tracer.start_span("operation");
        assert_eq!(tracer.active_spans().len(), 1);

        tracer.end_span(&span);
        assert_eq!(tracer.active_spans().len(), 0);
        assert_eq!(tracer.completed_count(), 1);
    }

    #[test]
    fn test_pipeline_span() {
        let tracer = Tracer::new("transcode");

        let span = tracer.start_pipeline_span(PipelineStage::Encode, "job-456", None);
        assert!(matches!(
            span.get_attribute("transcode.pipeline.stage"),
            Some(AttributeValue::String(s)) if s == "encode"
        ));
    }

    #[test]
    fn test_context_propagation() {
        let ctx = SpanContext::new_root();
        let mut headers = HashMap::new();

        ContextPropagator::inject(&ctx, &mut headers);
        assert!(headers.contains_key("traceparent"));

        let extracted = ContextPropagator::extract(&headers).unwrap();
        assert_eq!(extracted.trace_id, ctx.trace_id);
        assert_eq!(extracted.span_id, ctx.span_id);
    }

    #[test]
    fn test_pipeline_stages() {
        assert_eq!(PipelineStage::Input.name(), "input");
        assert_eq!(PipelineStage::Decode.name(), "decode");
        assert_eq!(PipelineStage::Encode.name(), "encode");
        assert_eq!(PipelineStage::Output.name(), "output");
    }

    #[test]
    fn test_attribute_values() {
        let span = Span::new_root("test");

        span.set_attribute("string", "value");
        span.set_attribute("int", 42i64);
        span.set_attribute("float", 3.14f64);
        span.set_attribute("bool", true);

        assert!(matches!(
            span.get_attribute("string"),
            Some(AttributeValue::String(_))
        ));
        assert!(matches!(
            span.get_attribute("int"),
            Some(AttributeValue::Int(42))
        ));
        assert!(matches!(
            span.get_attribute("float"),
            Some(AttributeValue::Float(_))
        ));
        assert!(matches!(
            span.get_attribute("bool"),
            Some(AttributeValue::Bool(true))
        ));
    }
}
