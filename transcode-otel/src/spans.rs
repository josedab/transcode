//! Tracing spans for transcoding operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Span context for distributed tracing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub baggage: HashMap<String, String>,
}

impl SpanContext {
    /// Create a new root context.
    pub fn new_root() -> Self {
        Self {
            trace_id: generate_id(),
            span_id: generate_id(),
            parent_span_id: None,
            baggage: HashMap::new(),
        }
    }

    /// Create a child context.
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: generate_id(),
            parent_span_id: Some(self.span_id.clone()),
            baggage: self.baggage.clone(),
        }
    }

    /// Add baggage item.
    pub fn with_baggage(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.baggage.insert(key.into(), value.into());
        self
    }

    /// Serialize to W3C traceparent header format.
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-01",
            &self.trace_id[..32.min(self.trace_id.len())],
            &self.span_id[..16.min(self.span_id.len())]
        )
    }
}

/// Span for a transcoding operation.
pub struct TranscodeSpan {
    name: String,
    context: SpanContext,
    start_time: Instant,
    attributes: HashMap<String, String>,
    events: Vec<SpanEvent>,
    ended: bool,
}

impl TranscodeSpan {
    /// Create a new span.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: SpanContext::new_root(),
            start_time: Instant::now(),
            attributes: HashMap::new(),
            events: Vec::new(),
            ended: false,
        }
    }

    /// Create a child span.
    pub fn child(&self, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: self.context.child(),
            start_time: Instant::now(),
            attributes: HashMap::new(),
            events: Vec::new(),
            ended: false,
        }
    }

    /// Set input file attribute.
    pub fn input_file(mut self, path: impl Into<String>) -> Self {
        self.attributes.insert("input.file".to_string(), path.into());
        self
    }

    /// Set output codec attribute.
    pub fn output_codec(mut self, codec: impl Into<String>) -> Self {
        self.attributes.insert("output.codec".to_string(), codec.into());
        self
    }

    /// Set resolution attribute.
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.attributes
            .insert("video.resolution".to_string(), format!("{}x{}", width, height));
        self
    }

    /// Add a custom attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Start the span (returns self for chaining).
    pub fn start(self) -> Self {
        tracing::info!(
            span_name = %self.name,
            trace_id = %self.context.trace_id,
            "Starting transcode span"
        );
        self
    }

    /// Record an event.
    pub fn event(&mut self, name: impl Into<String>) {
        self.events.push(SpanEvent {
            name: name.into(),
            timestamp: self.start_time.elapsed(),
            attributes: HashMap::new(),
        });
    }

    /// Record frame processing.
    pub fn record_frame(&mut self, frame_number: u64, width: u32, height: u32, encode_time_ms: f64) {
        let mut attrs = HashMap::new();
        attrs.insert("frame.number".to_string(), frame_number.to_string());
        attrs.insert("frame.width".to_string(), width.to_string());
        attrs.insert("frame.height".to_string(), height.to_string());
        attrs.insert("frame.encode_time_ms".to_string(), format!("{:.2}", encode_time_ms));

        self.events.push(SpanEvent {
            name: "frame_encoded".to_string(),
            timestamp: self.start_time.elapsed(),
            attributes: attrs,
        });
    }

    /// Set error on span.
    pub fn error(&mut self, message: impl Into<String>) {
        self.attributes.insert("error".to_string(), "true".to_string());
        self.attributes.insert("error.message".to_string(), message.into());
    }

    /// Get the span context.
    pub fn context(&self) -> &SpanContext {
        &self.context
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Finish the span.
    pub fn finish(mut self) -> SpanResult {
        self.ended = true;
        let duration = self.start_time.elapsed();

        tracing::info!(
            span_name = %self.name,
            trace_id = %self.context.trace_id,
            duration_ms = duration.as_millis(),
            "Finished transcode span"
        );

        SpanResult {
            name: std::mem::take(&mut self.name),
            context: std::mem::take(&mut self.context),
            duration,
            attributes: std::mem::take(&mut self.attributes),
            events: std::mem::take(&mut self.events),
        }
    }
}

impl Drop for TranscodeSpan {
    fn drop(&mut self) {
        if !self.ended {
            tracing::warn!(
                span_name = %self.name,
                "Span dropped without being finished"
            );
        }
    }
}

/// Result of a completed span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanResult {
    pub name: String,
    pub context: SpanContext,
    pub duration: Duration,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
}

/// Event within a span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub name: String,
    pub timestamp: Duration,
    pub attributes: HashMap<String, String>,
}

/// Span for a transcoding job.
pub struct JobSpan {
    inner: TranscodeSpan,
    job_id: String,
    frames_processed: u64,
}

impl JobSpan {
    /// Create a new job span.
    pub fn new(job_id: impl Into<String>) -> Self {
        let job_id = job_id.into();
        let span = TranscodeSpan::new(format!("job:{}", job_id))
            .attribute("job.id", &job_id)
            .start();

        Self {
            inner: span,
            job_id,
            frames_processed: 0,
        }
    }

    /// Set input file.
    pub fn input(mut self, path: impl Into<String>) -> Self {
        self.inner = self.inner.input_file(path);
        self
    }

    /// Set output codec.
    pub fn output_codec(mut self, codec: impl Into<String>) -> Self {
        self.inner = self.inner.output_codec(codec);
        self
    }

    /// Record frame encoded.
    pub fn frame_encoded(&mut self, frame_number: u64, _size_bytes: u64) {
        self.frames_processed += 1;
        self.inner.event(format!("frame:{}", frame_number));
    }

    /// Set error.
    pub fn error(&mut self, message: impl Into<String>) {
        self.inner.error(message);
    }

    /// Get job ID.
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Get frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Finish the job span.
    pub fn finish(mut self) -> SpanResult {
        self.inner = self.inner.attribute("job.frames_processed", self.frames_processed.to_string());
        self.inner.finish()
    }
}

/// Span for a single frame.
pub struct FrameSpan {
    inner: TranscodeSpan,
    frame_number: u64,
}

impl FrameSpan {
    /// Create a new frame span.
    pub fn new(frame_number: u64, width: u32, height: u32) -> Self {
        let span = TranscodeSpan::new(format!("frame:{}", frame_number))
            .attribute("frame.number", frame_number.to_string())
            .resolution(width, height)
            .start();

        Self {
            inner: span,
            frame_number,
        }
    }

    /// Set frame type.
    pub fn frame_type(mut self, frame_type: impl Into<String>) -> Self {
        self.inner = self.inner.attribute("frame.type", frame_type);
        self
    }

    /// Record encode complete.
    pub fn encode_complete(&mut self, size_bytes: u64, quality: Option<f64>) {
        self.inner = std::mem::replace(&mut self.inner, TranscodeSpan::new("temp"))
            .attribute("frame.size_bytes", size_bytes.to_string());

        if let Some(q) = quality {
            self.inner = std::mem::replace(&mut self.inner, TranscodeSpan::new("temp"))
                .attribute("frame.quality", format!("{:.2}", q));
        }

        self.inner.event("encode_complete");
    }

    /// Finish the frame span.
    pub fn finish(self) -> SpanResult {
        self.inner.finish()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:032x}", ts ^ rand::random::<u128>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_context() {
        let ctx = SpanContext::new_root();
        assert!(!ctx.trace_id.is_empty());
        assert!(!ctx.span_id.is_empty());
        assert!(ctx.parent_span_id.is_none());

        let child = ctx.child();
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_ne!(child.span_id, ctx.span_id);
        assert_eq!(child.parent_span_id, Some(ctx.span_id.clone()));
    }

    #[test]
    fn test_transcode_span() {
        let mut span = TranscodeSpan::new("test_transcode")
            .input_file("input.mp4")
            .output_codec("h264")
            .resolution(1920, 1080)
            .start();

        span.record_frame(1, 1920, 1080, 5.5);
        span.event("keyframe_inserted");

        let result = span.finish();
        assert_eq!(result.name, "test_transcode");
        assert!(!result.events.is_empty());
    }

    #[test]
    fn test_job_span() {
        let mut span = JobSpan::new("job-123")
            .input("input.mp4")
            .output_codec("h264");

        span.frame_encoded(1, 50000);
        span.frame_encoded(2, 5000);

        assert_eq!(span.frames_processed(), 2);

        let result = span.finish();
        assert!(result.attributes.contains_key("job.frames_processed"));
    }

    #[test]
    fn test_traceparent() {
        let ctx = SpanContext::new_root();
        let traceparent = ctx.to_traceparent();

        assert!(traceparent.starts_with("00-"));
        assert!(traceparent.ends_with("-01"));
    }
}
