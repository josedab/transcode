---
sidebar_position: 20
title: Real-Time Analytics & Alerting
description: Stream quality metrics, define alert rules, and generate chart data
---

# Real-Time Analytics & Alerting

The `transcode-dashboard` crate provides real-time metric streaming, configurable alert rules, and chart data generation for monitoring transcoding quality.

## Overview

Monitor encoding quality in real time with:

- **Metric streaming** — Push frame-by-frame quality data
- **Alert engine** — Configurable rules with cooldown periods
- **Chart data** — Time-series data with windowed averages

```rust
use transcode_dashboard::alerts::{
    MetricStream, StreamMetrics, AlertEngine, AlertRule,
    AlertMetricType, AlertCondition, AlertSeverity,
};

// Create alert engine with rules
let mut engine = AlertEngine::new();
engine.add_rule(AlertRule {
    id: "low-psnr".to_string(),
    name: "Low PSNR Alert".to_string(),
    metric: AlertMetricType::Psnr,
    condition: AlertCondition::Below(30.0),
    severity: AlertSeverity::Warning,
    enabled: true,
    cooldown_frames: 30,
});

// Create metric stream with alerting
let mut stream = MetricStream::with_alert_engine(engine);

// Push metrics as frames are encoded
stream.push_metrics(1, StreamMetrics {
    psnr: Some(42.5),
    ssim: Some(0.98),
    vmaf: Some(92.0),
    bitrate_kbps: Some(5000.0),
    encoding_fps: Some(60.0),
});
```

## Quick Start

```toml
[dependencies]
transcode-dashboard = "1.0"
```

### Defining Alert Rules

```rust
use transcode_dashboard::alerts::*;

let mut engine = AlertEngine::new();

// Alert when PSNR drops below 30 dB
engine.add_rule(AlertRule {
    id: "low-psnr".into(),
    name: "Low PSNR".into(),
    metric: AlertMetricType::Psnr,
    condition: AlertCondition::Below(30.0),
    severity: AlertSeverity::Warning,
    enabled: true,
    cooldown_frames: 30, // Don't re-fire for 30 frames
});

// Alert when bitrate exceeds target
engine.add_rule(AlertRule {
    id: "high-bitrate".into(),
    name: "Bitrate Exceeded".into(),
    metric: AlertMetricType::Bitrate,
    condition: AlertCondition::Above(8000.0),
    severity: AlertSeverity::Critical,
    enabled: true,
    cooldown_frames: 60,
});

// Alert when VMAF is outside acceptable range
engine.add_rule(AlertRule {
    id: "vmaf-range".into(),
    name: "VMAF Out of Range".into(),
    metric: AlertMetricType::Vmaf,
    condition: AlertCondition::OutOfRange(80.0, 100.0),
    severity: AlertSeverity::Info,
    enabled: true,
    cooldown_frames: 10,
});
```

### Streaming Metrics

```rust
let mut stream = MetricStream::with_alert_engine(engine);

// Simulate encoding loop
for frame in 0..1000 {
    stream.push_metrics(frame, StreamMetrics {
        psnr: Some(35.0 + (frame as f64 * 0.01).sin() * 10.0),
        ssim: Some(0.95),
        vmaf: Some(88.0),
        bitrate_kbps: Some(5000.0),
        encoding_fps: Some(60.0),
    });
}

// Check for alerts
let latest = stream.latest_update().unwrap();
println!("Frame {}: {} alerts", latest.frame_number, latest.alerts.len());

// Poll for updates since a specific frame
let updates = stream.updates_since(990);
println!("Updates since frame 990: {}", updates.len());
```

### Chart Data

Generate time-series data for visualization:

```rust
use transcode_dashboard::alerts::{ChartData, DataPoint};

// Build chart from stream data
let chart = ChartData::from_stream(&stream);

// Available series: "psnr", "ssim", "vmaf", "bitrate_kbps", "encoding_fps"
for name in chart.series_names() {
    println!("Series: {}", name);
}

// Smoothed data with moving average
let smoothed = chart.windowed_average("psnr", 30); // 30-frame window
for point in &smoothed[..5] {
    println!("  Frame {:.0}: {:.2} dB", point.x, point.y);
}

// Export to JSON for frontend charts
let json = chart.to_json();
println!("{}", json);
```

### JSON Export for Dashboards

```rust
// Stream updates serialize to JSON for WebSocket delivery
let json = stream.to_json();
// Send to connected dashboard clients via WebSocket
```

## Alert Conditions

| Condition | Syntax | Description |
|-----------|--------|-------------|
| Below threshold | `AlertCondition::Below(30.0)` | Fires when metric < value |
| Above threshold | `AlertCondition::Above(8000.0)` | Fires when metric > value |
| Out of range | `AlertCondition::OutOfRange(80.0, 100.0)` | Fires when metric outside bounds |

## Metrics Available

| Metric | Type | Typical Range |
|--------|------|---------------|
| PSNR | `AlertMetricType::Psnr` | 20-50 dB |
| SSIM | `AlertMetricType::Ssim` | 0.0-1.0 |
| VMAF | `AlertMetricType::Vmaf` | 0-100 |
| Bitrate | `AlertMetricType::Bitrate` | kbps |
| Frame Rate | `AlertMetricType::FrameRate` | fps |
| Encoding Speed | `AlertMetricType::EncodingSpeed` | fps |

## API Reference

| Type | Description |
|------|-------------|
| `AlertEngine` | Evaluates rules against frame metrics |
| `AlertRule` | Configurable alert with metric, condition, severity |
| `Alert` | Fired alert with context |
| `MetricStream` | Real-time metric push with alerting |
| `StreamUpdate` | Single frame update with metrics and alerts |
| `StreamMetrics` | Per-frame quality measurements |
| `ChartData` | Time-series data for visualization |
| `DataPoint` | Single (x, y) chart point |

## Next Steps

- [Quality Metrics](/docs/guides/quality-metrics) — PSNR, SSIM, VMAF measurement
- [Distributed Processing](/docs/guides/distributed-processing) — Monitor distributed workers
