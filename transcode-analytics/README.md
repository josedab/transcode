# transcode-analytics

Real-time quality analytics and monitoring for video transcoding pipelines.

## Overview

`transcode-analytics` provides comprehensive metrics collection, time series analysis, and quality monitoring for the transcode ecosystem. It enables real-time tracking of encoding performance, bitrate statistics, and video quality metrics (PSNR, SSIM).

## Features

- **Real-time Statistics**: FPS, bitrate, encode time, buffer fullness, dropped frames
- **Quality Metrics**: PSNR and SSIM tracking with min/max/average aggregation
- **Frame Analysis**: Frame type distribution (I/P/B frames) and per-frame statistics
- **Time Series**: Rolling buffers with automatic expiration and statistical functions
- **Moving Averages**: Configurable window-based smoothing for metrics
- **Bitrate Monitoring**: Target tracking, variance calculation, and tolerance checks

## Key Types

| Type | Description |
|------|-------------|
| `AnalyticsCollector` | Main collector for frame statistics and metrics |
| `AnalyticsConfig` | Configuration for window size, sampling, and metric options |
| `FrameStats` | Per-frame data: size, encode time, type, QP, PSNR, SSIM |
| `RealtimeStats` | Current FPS, bitrate, encode time, quality score |
| `AnalyticsReport` | Final summary with all aggregated metrics |
| `TimeSeries<T>` | Generic rolling time series buffer |
| `MovingAverage` | Window-based average calculator |
| `EncodingMetrics` | Encoding performance aggregation |
| `QualityMetrics` | PSNR/SSIM statistics with min/max tracking |
| `BitrateStats` | Bitrate monitoring with target compliance |

## Usage

### Basic Analytics Collection

```rust
use transcode_analytics::{AnalyticsCollector, AnalyticsConfig, FrameStats, FrameType};
use std::time::Duration;

let config = AnalyticsConfig::default();
let mut collector = AnalyticsCollector::new(config);

// Record frame statistics during encoding
collector.record_frame(FrameStats {
    frame_number: 0,
    size_bytes: 15000,
    encode_time: Duration::from_millis(8),
    frame_type: FrameType::I,
    qp: Some(23.0),
    psnr: Some(42.5),
    ssim: Some(0.97),
});

// Get real-time statistics
let stats = collector.get_realtime_stats();
println!("FPS: {:.2}, Bitrate: {:.0} kbps", stats.fps, stats.bitrate_kbps);

// Get frame distribution
let dist = collector.get_frame_distribution();
println!("I: {}, P: {}, B: {}", dist.i_frames, dist.p_frames, dist.b_frames);

// Generate final report
let report = collector.generate_report();
```

### Time Series Analysis

```rust
use transcode_analytics::TimeSeries;
use std::time::Duration;

let mut bitrate_ts: TimeSeries<f64> = TimeSeries::new(
    Duration::from_secs(60),  // Keep last 60 seconds
    1000,                      // Max 1000 data points
);

bitrate_ts.push(2500.0);
bitrate_ts.push(2600.0);
bitrate_ts.push(2450.0);

println!("Average: {:?}", bitrate_ts.average());
println!("Std Dev: {:?}", bitrate_ts.std_dev());
println!("Rate of Change: {:?}", bitrate_ts.rate_of_change());
```

### Moving Average

```rust
use transcode_analytics::MovingAverage;

let mut ma = MovingAverage::new(10); // 10-sample window
let smoothed = ma.add(2500.0);
println!("Smoothed bitrate: {}", smoothed);
```

## Configuration

```rust
use transcode_analytics::AnalyticsConfig;
use std::time::Duration;

let config = AnalyticsConfig {
    window_size: 100,                        // Samples for moving averages
    sample_interval: Duration::from_millis(100),
    detailed_metrics: true,                  // Enable detailed per-frame metrics
    quality_metrics: true,                   // Enable PSNR/SSIM tracking
};
```

## License

See the workspace root for license information.
