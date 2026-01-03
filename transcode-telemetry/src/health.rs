//! Health check endpoints for the transcode system.
//!
//! This module provides health check functionality including liveness and readiness
//! probes for Kubernetes-style deployments, as well as detailed component health status.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{HealthError, Result};

/// Health status of a component or the overall system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealthStatus {
    /// Component is healthy and operational.
    Healthy,
    /// Component is degraded but still functional.
    Degraded,
    /// Component is unhealthy.
    Unhealthy,
    /// Health status is unknown.
    #[default]
    Unknown,
}

impl HealthStatus {
    /// Get the status as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
            HealthStatus::Unknown => "unknown",
        }
    }

    /// Check if the status indicates the component is up.
    pub fn is_up(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Check if the status is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Get HTTP status code for this health status.
    pub fn http_status_code(&self) -> u16 {
        match self {
            HealthStatus::Healthy => 200,
            HealthStatus::Degraded => 200,
            HealthStatus::Unhealthy => 503,
            HealthStatus::Unknown => 503,
        }
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Health check result for a component.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Component name.
    pub name: String,
    /// Health status.
    pub status: HealthStatus,
    /// Optional message.
    pub message: Option<String>,
    /// Check duration.
    pub duration: Duration,
    /// Timestamp of the check.
    pub timestamp: SystemTime,
    /// Additional details.
    pub details: HashMap<String, String>,
}

impl HealthCheckResult {
    /// Create a healthy result.
    pub fn healthy(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            message: None,
            duration,
            timestamp: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Create a degraded result.
    pub fn degraded(name: impl Into<String>, message: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded,
            message: Some(message.into()),
            duration,
            timestamp: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Create an unhealthy result.
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy,
            message: Some(message.into()),
            duration,
            timestamp: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    /// Add a detail.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!("\"name\":\"{}\"", escape_json(&self.name)));
        parts.push(format!("\"status\":\"{}\"", self.status.as_str()));

        if let Some(ref msg) = self.message {
            parts.push(format!("\"message\":\"{}\"", escape_json(msg)));
        }

        parts.push(format!("\"duration_ms\":{}", self.duration.as_millis()));

        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        parts.push(format!("\"timestamp\":{}", timestamp));

        if !self.details.is_empty() {
            let details: Vec<String> = self
                .details
                .iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
                .collect();
            parts.push(format!("\"details\":{{{}}}", details.join(",")));
        }

        format!("{{{}}}", parts.join(","))
    }
}

/// A health check that can be performed.
pub trait HealthCheck: Send + Sync {
    /// Get the name of this health check.
    fn name(&self) -> &str;

    /// Perform the health check.
    fn check(&self) -> HealthCheckResult;

    /// Get the timeout for this check.
    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

/// A simple health check implementation.
pub struct SimpleHealthCheck {
    name: String,
    check_fn: Box<dyn Fn() -> (HealthStatus, Option<String>) + Send + Sync>,
    timeout: Duration,
}

impl SimpleHealthCheck {
    /// Create a new simple health check.
    pub fn new<F>(name: impl Into<String>, check_fn: F) -> Self
    where
        F: Fn() -> (HealthStatus, Option<String>) + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            check_fn: Box::new(check_fn),
            timeout: Duration::from_secs(5),
        }
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl HealthCheck for SimpleHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let (status, message) = (self.check_fn)();
        let duration = start.elapsed();

        HealthCheckResult {
            name: self.name.clone(),
            status,
            message,
            duration,
            timestamp: SystemTime::now(),
            details: HashMap::new(),
        }
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Liveness check - indicates if the process is running.
pub struct LivenessCheck {
    start_time: Instant,
}

impl LivenessCheck {
    /// Create a new liveness check.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    /// Get the uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for LivenessCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthCheck for LivenessCheck {
    fn name(&self) -> &str {
        "liveness"
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let uptime = self.uptime();

        HealthCheckResult::healthy("liveness", start.elapsed())
            .with_detail("uptime_seconds", uptime.as_secs().to_string())
    }
}

/// Readiness check - indicates if the service is ready to accept traffic.
pub struct ReadinessCheck {
    ready: Arc<RwLock<bool>>,
    reason: Arc<RwLock<Option<String>>>,
}

impl ReadinessCheck {
    /// Create a new readiness check.
    pub fn new() -> Self {
        Self {
            ready: Arc::new(RwLock::new(false)),
            reason: Arc::new(RwLock::new(None)),
        }
    }

    /// Set ready status.
    pub fn set_ready(&self, ready: bool) {
        *self.ready.write().unwrap() = ready;
        if ready {
            *self.reason.write().unwrap() = None;
        }
    }

    /// Set not ready with reason.
    pub fn set_not_ready(&self, reason: impl Into<String>) {
        *self.ready.write().unwrap() = false;
        *self.reason.write().unwrap() = Some(reason.into());
    }

    /// Check if ready.
    pub fn is_ready(&self) -> bool {
        *self.ready.read().unwrap()
    }
}

impl Default for ReadinessCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthCheck for ReadinessCheck {
    fn name(&self) -> &str {
        "readiness"
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let is_ready = self.is_ready();
        let reason = self.reason.read().unwrap().clone();
        let duration = start.elapsed();

        if is_ready {
            HealthCheckResult::healthy("readiness", duration)
        } else {
            HealthCheckResult::unhealthy(
                "readiness",
                reason.unwrap_or_else(|| "Service not ready".to_string()),
                duration,
            )
        }
    }
}

/// Transcode-specific health checks.
pub struct TranscodeHealthCheck {
    name: String,
    max_queue_depth: u64,
    max_error_rate: f64,
    current_queue_depth: Arc<RwLock<u64>>,
    current_error_rate: Arc<RwLock<f64>>,
    codec_available: Arc<RwLock<bool>>,
}

impl TranscodeHealthCheck {
    /// Create a new transcode health check.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_queue_depth: 100,
            max_error_rate: 0.1, // 10%
            current_queue_depth: Arc::new(RwLock::new(0)),
            current_error_rate: Arc::new(RwLock::new(0.0)),
            codec_available: Arc::new(RwLock::new(true)),
        }
    }

    /// Set maximum queue depth threshold.
    pub fn with_max_queue_depth(mut self, max: u64) -> Self {
        self.max_queue_depth = max;
        self
    }

    /// Set maximum error rate threshold.
    pub fn with_max_error_rate(mut self, max: f64) -> Self {
        self.max_error_rate = max;
        self
    }

    /// Update current queue depth.
    pub fn update_queue_depth(&self, depth: u64) {
        *self.current_queue_depth.write().unwrap() = depth;
    }

    /// Update current error rate.
    pub fn update_error_rate(&self, rate: f64) {
        *self.current_error_rate.write().unwrap() = rate;
    }

    /// Set codec availability.
    pub fn set_codec_available(&self, available: bool) {
        *self.codec_available.write().unwrap() = available;
    }
}

impl HealthCheck for TranscodeHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();

        let queue_depth = *self.current_queue_depth.read().unwrap();
        let error_rate = *self.current_error_rate.read().unwrap();
        let codec_available = *self.codec_available.read().unwrap();

        let duration = start.elapsed();

        // Check for unhealthy conditions
        if !codec_available {
            return HealthCheckResult::unhealthy(&self.name, "Codec not available", duration)
                .with_detail("codec_available", "false");
        }

        if error_rate > self.max_error_rate {
            return HealthCheckResult::unhealthy(
                &self.name,
                format!("Error rate too high: {:.2}%", error_rate * 100.0),
                duration,
            )
            .with_detail("error_rate", format!("{:.4}", error_rate))
            .with_detail("max_error_rate", format!("{:.4}", self.max_error_rate));
        }

        // Check for degraded conditions
        if queue_depth > self.max_queue_depth {
            return HealthCheckResult::degraded(
                &self.name,
                format!("Queue depth high: {}", queue_depth),
                duration,
            )
            .with_detail("queue_depth", queue_depth.to_string())
            .with_detail("max_queue_depth", self.max_queue_depth.to_string());
        }

        HealthCheckResult::healthy(&self.name, duration)
            .with_detail("queue_depth", queue_depth.to_string())
            .with_detail("error_rate", format!("{:.4}", error_rate))
            .with_detail("codec_available", "true")
    }
}

/// Aggregate health check result.
#[derive(Debug, Clone)]
pub struct AggregateHealthResult {
    /// Overall status.
    pub status: HealthStatus,
    /// Individual check results.
    pub checks: Vec<HealthCheckResult>,
    /// Total check duration.
    pub total_duration: Duration,
    /// Timestamp.
    pub timestamp: SystemTime,
}

impl AggregateHealthResult {
    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        let checks_json: Vec<String> = self.checks.iter().map(|c| c.to_json()).collect();

        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        format!(
            "{{\"status\":\"{}\",\"checks\":[{}],\"total_duration_ms\":{},\"timestamp\":{}}}",
            self.status.as_str(),
            checks_json.join(","),
            self.total_duration.as_millis(),
            timestamp
        )
    }
}

/// Health checker that manages multiple health checks.
pub struct HealthChecker {
    checks: RwLock<Vec<Arc<dyn HealthCheck>>>,
    liveness: Arc<LivenessCheck>,
    readiness: Arc<ReadinessCheck>,
}

impl HealthChecker {
    /// Create a new health checker.
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(Vec::new()),
            liveness: Arc::new(LivenessCheck::new()),
            readiness: Arc::new(ReadinessCheck::new()),
        }
    }

    /// Get the liveness check.
    pub fn liveness(&self) -> Arc<LivenessCheck> {
        self.liveness.clone()
    }

    /// Get the readiness check.
    pub fn readiness(&self) -> Arc<ReadinessCheck> {
        self.readiness.clone()
    }

    /// Add a health check.
    pub fn add_check(&self, check: Arc<dyn HealthCheck>) {
        self.checks.write().unwrap().push(check);
    }

    /// Run the liveness check.
    pub fn check_liveness(&self) -> HealthCheckResult {
        self.liveness.check()
    }

    /// Run the readiness check.
    pub fn check_readiness(&self) -> HealthCheckResult {
        self.readiness.check()
    }

    /// Run all health checks.
    pub fn check_all(&self) -> AggregateHealthResult {
        let start = Instant::now();
        let mut results = Vec::new();
        let mut overall_status = HealthStatus::Healthy;

        // Run liveness check
        results.push(self.liveness.check());

        // Run readiness check
        let readiness_result = self.readiness.check();
        if !readiness_result.status.is_healthy() {
            overall_status = readiness_result.status;
        }
        results.push(readiness_result);

        // Run all registered checks
        for check in self.checks.read().unwrap().iter() {
            let result = check.check();

            // Update overall status
            match (&overall_status, &result.status) {
                (HealthStatus::Healthy, HealthStatus::Degraded) => {
                    overall_status = HealthStatus::Degraded;
                }
                (_, HealthStatus::Unhealthy) => {
                    overall_status = HealthStatus::Unhealthy;
                }
                _ => {}
            }

            results.push(result);
        }

        AggregateHealthResult {
            status: overall_status,
            checks: results,
            total_duration: start.elapsed(),
            timestamp: SystemTime::now(),
        }
    }

    /// Run a specific check by name.
    pub fn check_by_name(&self, name: &str) -> Option<HealthCheckResult> {
        if name == "liveness" {
            return Some(self.liveness.check());
        }
        if name == "readiness" {
            return Some(self.readiness.check());
        }

        for check in self.checks.read().unwrap().iter() {
            if check.name() == name {
                return Some(check.check());
            }
        }

        None
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the health check server.
#[derive(Debug, Clone)]
pub struct HealthServerConfig {
    /// Address to bind to.
    pub address: String,
    /// Port to listen on.
    pub port: u16,
    /// Liveness path.
    pub liveness_path: String,
    /// Readiness path.
    pub readiness_path: String,
    /// Full health path.
    pub health_path: String,
}

impl HealthServerConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self {
            address: "0.0.0.0".to_string(),
            port: 8080,
            liveness_path: "/health/live".to_string(),
            readiness_path: "/health/ready".to_string(),
            health_path: "/health".to_string(),
        }
    }

    /// Set the address.
    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = address.into();
        self
    }

    /// Set the port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Get the bind address.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

impl Default for HealthServerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check HTTP server.
pub struct HealthServer {
    config: HealthServerConfig,
    checker: Arc<HealthChecker>,
    running: Arc<RwLock<bool>>,
    listener: Option<TcpListener>,
}

impl HealthServer {
    /// Create a new health server.
    pub fn new(config: HealthServerConfig, checker: Arc<HealthChecker>) -> Self {
        Self {
            config,
            checker,
            running: Arc::new(RwLock::new(false)),
            listener: None,
        }
    }

    /// Start the server.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr).map_err(|e| HealthError::CheckFailed {
            check_name: "server".to_string(),
            reason: format!("Failed to bind to {}: {}", addr, e),
        })?;

        listener.set_nonblocking(true).map_err(|e| HealthError::CheckFailed {
            check_name: "server".to_string(),
            reason: e.to_string(),
        })?;

        self.listener = Some(listener);
        *self.running.write().unwrap() = true;

        Ok(())
    }

    /// Stop the server.
    pub fn stop(&mut self) {
        *self.running.write().unwrap() = false;
        self.listener = None;
    }

    /// Check if the server is running.
    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }

    /// Process incoming connections (non-blocking).
    pub fn poll(&self) -> Result<()> {
        if !self.is_running() {
            return Ok(());
        }

        let listener = match &self.listener {
            Some(l) => l,
            None => return Ok(()),
        };

        match listener.accept() {
            Ok((stream, _addr)) => {
                self.handle_connection(stream)?;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No connections waiting
            }
            Err(e) => {
                return Err(HealthError::CheckFailed {
                    check_name: "server".to_string(),
                    reason: e.to_string(),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Handle a client connection.
    fn handle_connection(&self, mut stream: TcpStream) -> Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

        // Read request
        let mut buffer = [0u8; 4096];
        let n = stream.read(&mut buffer).map_err(|e| HealthError::CheckFailed {
            check_name: "server".to_string(),
            reason: e.to_string(),
        })?;

        let request = String::from_utf8_lossy(&buffer[..n]);
        let request_line = request.lines().next().unwrap_or("");
        let parts: Vec<&str> = request_line.split_whitespace().collect();

        if parts.len() < 2 {
            self.send_response(&mut stream, 400, "Bad Request", "{\"error\":\"Invalid request\"}")?;
            return Ok(());
        }

        let method = parts[0];
        let path = parts[1].split('?').next().unwrap_or(parts[1]);

        // Handle request
        match (method, path) {
            ("GET", p) if p == self.config.liveness_path => {
                let result = self.checker.check_liveness();
                let status_code = result.status.http_status_code();
                self.send_json_response(&mut stream, status_code, &result.to_json())?;
            }
            ("GET", p) if p == self.config.readiness_path => {
                let result = self.checker.check_readiness();
                let status_code = result.status.http_status_code();
                self.send_json_response(&mut stream, status_code, &result.to_json())?;
            }
            ("GET", p) if p == self.config.health_path => {
                let result = self.checker.check_all();
                let status_code = result.status.http_status_code();
                self.send_json_response(&mut stream, status_code, &result.to_json())?;
            }
            ("GET", "/") => {
                let body = format!(
                    "<html><body><h1>Health Endpoints</h1>\
                    <ul>\
                    <li><a href=\"{}\">{}</a></li>\
                    <li><a href=\"{}\">{}</a></li>\
                    <li><a href=\"{}\">{}</a></li>\
                    </ul></body></html>",
                    self.config.liveness_path, self.config.liveness_path,
                    self.config.readiness_path, self.config.readiness_path,
                    self.config.health_path, self.config.health_path
                );
                self.send_response(&mut stream, 200, "OK", &body)?;
            }
            _ => {
                self.send_response(&mut stream, 404, "Not Found", "{\"error\":\"Not found\"}")?;
            }
        }

        Ok(())
    }

    /// Send an HTTP response.
    fn send_response(
        &self,
        stream: &mut TcpStream,
        status: u16,
        status_text: &str,
        body: &str,
    ) -> Result<()> {
        let response = format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            status,
            status_text,
            body.len(),
            body
        );

        stream.write_all(response.as_bytes()).map_err(|e| HealthError::CheckFailed {
            check_name: "server".to_string(),
            reason: e.to_string(),
        })?;

        Ok(())
    }

    /// Send a JSON response.
    fn send_json_response(&self, stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
        let status_text = match status {
            200 => "OK",
            503 => "Service Unavailable",
            _ => "Unknown",
        };

        let response = format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: application/json; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            status,
            status_text,
            body.len(),
            body
        );

        stream.write_all(response.as_bytes()).map_err(|e| HealthError::CheckFailed {
            check_name: "server".to_string(),
            reason: e.to_string(),
        })?;

        Ok(())
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

/// Build health check response for use with other HTTP frameworks.
pub fn build_liveness_response(checker: &HealthChecker) -> (u16, String, String) {
    let result = checker.check_liveness();
    let status = result.status.http_status_code();
    let content_type = "application/json".to_string();
    let body = result.to_json();
    (status, content_type, body)
}

/// Build readiness response for use with other HTTP frameworks.
pub fn build_readiness_response(checker: &HealthChecker) -> (u16, String, String) {
    let result = checker.check_readiness();
    let status = result.status.http_status_code();
    let content_type = "application/json".to_string();
    let body = result.to_json();
    (status, content_type, body)
}

/// Build full health response for use with other HTTP frameworks.
pub fn build_health_response(checker: &HealthChecker) -> (u16, String, String) {
    let result = checker.check_all();
    let status = result.status.http_status_code();
    let content_type = "application/json".to_string();
    let body = result.to_json();
    (status, content_type, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(HealthStatus::Healthy.is_up());
        assert!(HealthStatus::Degraded.is_up());
        assert!(!HealthStatus::Degraded.is_healthy());
        assert!(!HealthStatus::Unhealthy.is_up());
        assert!(!HealthStatus::Unknown.is_up());
    }

    #[test]
    fn test_health_status_http_codes() {
        assert_eq!(HealthStatus::Healthy.http_status_code(), 200);
        assert_eq!(HealthStatus::Degraded.http_status_code(), 200);
        assert_eq!(HealthStatus::Unhealthy.http_status_code(), 503);
        assert_eq!(HealthStatus::Unknown.http_status_code(), 503);
    }

    #[test]
    fn test_health_check_result_healthy() {
        let result = HealthCheckResult::healthy("test", Duration::from_millis(10));
        assert_eq!(result.name, "test");
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_health_check_result_unhealthy() {
        let result = HealthCheckResult::unhealthy("test", "error message", Duration::from_millis(10));
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert_eq!(result.message, Some("error message".to_string()));
    }

    #[test]
    fn test_health_check_result_with_details() {
        let result = HealthCheckResult::healthy("test", Duration::from_millis(10))
            .with_detail("key1", "value1")
            .with_detail("key2", "value2");

        assert_eq!(result.details.len(), 2);
        assert_eq!(result.details.get("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_health_check_result_json() {
        let result = HealthCheckResult::healthy("test", Duration::from_millis(10))
            .with_detail("version", "1.0.0");

        let json = result.to_json();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"duration_ms\":"));
    }

    #[test]
    fn test_liveness_check() {
        let check = LivenessCheck::new();
        std::thread::sleep(Duration::from_millis(10));

        let result = check.check();
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.details.contains_key("uptime_seconds"));
    }

    #[test]
    fn test_readiness_check() {
        let check = ReadinessCheck::new();

        // Initially not ready
        assert!(!check.is_ready());
        let result = check.check();
        assert_eq!(result.status, HealthStatus::Unhealthy);

        // Set ready
        check.set_ready(true);
        assert!(check.is_ready());
        let result = check.check();
        assert_eq!(result.status, HealthStatus::Healthy);

        // Set not ready with reason
        check.set_not_ready("Warming up");
        assert!(!check.is_ready());
        let result = check.check();
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert!(result.message.unwrap().contains("Warming up"));
    }

    #[test]
    fn test_simple_health_check() {
        let check = SimpleHealthCheck::new("custom", || {
            (HealthStatus::Healthy, None)
        });

        let result = check.check();
        assert_eq!(result.name, "custom");
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_transcode_health_check_healthy() {
        let check = TranscodeHealthCheck::new("transcode")
            .with_max_queue_depth(100)
            .with_max_error_rate(0.1);

        check.update_queue_depth(50);
        check.update_error_rate(0.01);
        check.set_codec_available(true);

        let result = check.check();
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_transcode_health_check_degraded() {
        let check = TranscodeHealthCheck::new("transcode")
            .with_max_queue_depth(100);

        check.update_queue_depth(150);
        check.update_error_rate(0.01);
        check.set_codec_available(true);

        let result = check.check();
        assert_eq!(result.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_transcode_health_check_unhealthy_error_rate() {
        let check = TranscodeHealthCheck::new("transcode")
            .with_max_error_rate(0.1);

        check.update_queue_depth(50);
        check.update_error_rate(0.2);
        check.set_codec_available(true);

        let result = check.check();
        assert_eq!(result.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_transcode_health_check_unhealthy_codec() {
        let check = TranscodeHealthCheck::new("transcode");

        check.set_codec_available(false);

        let result = check.check();
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert!(result.message.unwrap().contains("Codec not available"));
    }

    #[test]
    fn test_health_checker() {
        let checker = HealthChecker::new();

        // Add a custom check
        let custom_check = Arc::new(SimpleHealthCheck::new("custom", || {
            (HealthStatus::Healthy, None)
        }));
        checker.add_check(custom_check);

        // Set ready
        checker.readiness().set_ready(true);

        // Check all
        let result = checker.check_all();
        assert_eq!(result.status, HealthStatus::Healthy);
        assert_eq!(result.checks.len(), 3); // liveness, readiness, custom
    }

    #[test]
    fn test_health_checker_with_failing_check() {
        let checker = HealthChecker::new();

        let failing_check = Arc::new(SimpleHealthCheck::new("failing", || {
            (HealthStatus::Unhealthy, Some("Always fails".to_string()))
        }));
        checker.add_check(failing_check);
        checker.readiness().set_ready(true);

        let result = checker.check_all();
        assert_eq!(result.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_checker_check_by_name() {
        let checker = HealthChecker::new();

        let result = checker.check_by_name("liveness");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "liveness");

        let result = checker.check_by_name("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_aggregate_health_result_json() {
        let result = AggregateHealthResult {
            status: HealthStatus::Healthy,
            checks: vec![
                HealthCheckResult::healthy("check1", Duration::from_millis(5)),
                HealthCheckResult::healthy("check2", Duration::from_millis(3)),
            ],
            total_duration: Duration::from_millis(10),
            timestamp: SystemTime::now(),
        };

        let json = result.to_json();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"checks\":["));
        assert!(json.contains("\"check1\""));
        assert!(json.contains("\"check2\""));
    }

    #[test]
    fn test_health_server_config() {
        let config = HealthServerConfig::new()
            .address("127.0.0.1")
            .port(9090);

        assert_eq!(config.address, "127.0.0.1");
        assert_eq!(config.port, 9090);
        assert_eq!(config.bind_address(), "127.0.0.1:9090");
    }

    #[test]
    fn test_build_liveness_response() {
        let checker = HealthChecker::new();
        let (status, content_type, body) = build_liveness_response(&checker);

        assert_eq!(status, 200);
        assert!(content_type.contains("application/json"));
        assert!(body.contains("\"status\":\"healthy\""));
    }

    #[test]
    fn test_build_readiness_response() {
        let checker = HealthChecker::new();

        // Not ready
        let (status, _, body) = build_readiness_response(&checker);
        assert_eq!(status, 503);
        assert!(body.contains("\"status\":\"unhealthy\""));

        // Ready
        checker.readiness().set_ready(true);
        let (status, _, body) = build_readiness_response(&checker);
        assert_eq!(status, 200);
        assert!(body.contains("\"status\":\"healthy\""));
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("simple"), "simple");
        assert_eq!(escape_json("with\"quote"), "with\\\"quote");
        assert_eq!(escape_json("with\nnewline"), "with\\nnewline");
    }
}
