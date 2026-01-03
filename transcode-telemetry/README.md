# transcode-telemetry

OpenTelemetry-based observability for the transcode system.

## Overview

This crate provides comprehensive observability features including metrics, distributed tracing, structured logging, Prometheus export, and Kubernetes-compatible health checks.

## Features

- **Metrics**: Counters, gauges, and histograms for transcode jobs, codecs, resources, and errors
- **Tracing**: Distributed tracing with W3C Trace Context propagation
- **Logging**: Structured JSON logging with trace correlation
- **Prometheus**: `/metrics` endpoint with text exposition format
- **Health Checks**: Liveness and readiness probes (`/health/live`, `/health/ready`)

## Quick Start

```rust
use transcode_telemetry::{TelemetryConfig, Telemetry, logging::Level};

// Initialize telemetry
let config = TelemetryConfig::new("transcode-service")
    .version("1.0.0")
    .environment("production")
    .log_level(Level::Info)
    .prometheus_port(9090)
    .health_port(8080);

let mut telemetry = Telemetry::new(config);
telemetry.start_servers().unwrap();

// Record metrics
telemetry.metrics().transcode.record_job_start();
telemetry.metrics().transcode.update_fps(30.0);
telemetry.metrics().codec.record_encode(0.033, true);

// Start a trace span
let span = telemetry.tracer().start_span("process_frame");
span.set_attribute("frame", 100i64);

// Log with trace correlation
telemetry.logger().info("Transcode started");

// Mark service as ready
telemetry.set_ready(true);
```

## Key Types

### Metrics

| Type | Description |
|------|-------------|
| `MetricsRegistry` | Central registry with pre-defined metric groups |
| `TranscodeMetrics` | Job counts, FPS, bitrate, progress, queue depth |
| `CodecMetrics` | Encode/decode time, GOP size, keyframe counts |
| `ResourceMetrics` | CPU, memory, GPU utilization |
| `ErrorMetrics` | Error counts by type |
| `Counter`, `Gauge`, `Histogram` | Metric primitives |

### Tracing

| Type | Description |
|------|-------------|
| `Tracer` | Creates and manages spans |
| `Span` | Represents a unit of work with attributes and events |
| `SpanContext` | Trace/span IDs for propagation |
| `PipelineStage` | Transcode stages (Input, Decode, Encode, Output, etc.) |

### Health

| Type | Description |
|------|-------------|
| `HealthChecker` | Manages liveness and readiness checks |
| `HealthStatus` | Healthy, Degraded, Unhealthy, Unknown |
| `TranscodeHealthCheck` | Queue depth and error rate monitoring |

## Optional Features

```toml
[dependencies]
transcode-telemetry = { version = "0.1", features = ["full"] }
```

| Feature | Description |
|---------|-------------|
| `otlp` | OpenTelemetry SDK with OTLP exporter |
| `prometheus-client` | Use prometheus crate instead of built-in |
| `tracing-integration` | Integration with `tracing` crate ecosystem |
| `async-server` | Tokio-based async server support |
| `full` | All features enabled |

## Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`:

```
# Transcode metrics
transcode_jobs_started_total
transcode_jobs_completed_total
transcode_current_fps
transcode_duration_seconds

# Codec metrics
codec_encode_time_seconds
codec_frames_encoded_total

# Resource metrics
resource_cpu_usage_percent
resource_memory_usage_bytes
```

## License

See workspace root for license information.
