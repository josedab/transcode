---
slug: next-gen-features-2026
title: "10 Next-Gen Features: Real-Time Streaming, ML Encoding, and More"
authors: [transcode-team]
tags: [release, technical]
---

Transcode just shipped 10 major feature additions spanning real-time streaming, machine learning, cloud orchestration, and developer tooling. Here's what's new and how to use it.

<!-- truncate -->

## What's New

These features fill critical gaps for production video platforms. Each one is fully tested with comprehensive documentation.

### 🔴 WHIP/WHEP Real-Time Streaming

The `transcode-whip` crate now includes a **media relay** for WebRTC pub/sub forwarding. Publishers ingest via WHIP, and subscribers receive via WHEP—with sub-second latency.

```rust
use transcode_whip::relay::{MediaRelay, RelayPacket, MediaType};

let mut relay = MediaRelay::new();
relay.create_stream("live-camera");

let (_, mut rx) = relay.subscribe("live-camera").unwrap();
relay.publish("live-camera", RelayPacket {
    media_type: MediaType::Video,
    payload: frame_data,
    timestamp: 90000,
    sequence: 1,
}).unwrap();
```

[Read the guide →](/docs/guides/whip-whep-streaming)

### ☸️ Kubernetes Manifest Generation

Generate production-ready K8s manifests directly from Rust—Deployments, HPAs, Services, and Helm values:

```rust
use transcode_k8s::manifest::generate_worker_deployment;

let yaml = generate_worker_deployment(
    "video-workers", "ghcr.io/transcode/worker:latest",
    5, "1", "2Gi", "4", "8Gi",
);
```

[Read the guide →](/docs/guides/kubernetes-manifests)

### 🧠 ML Per-Title Encoding

Train bitrate prediction models on your content library. OLS regression with L2 regularization, cross-validation, and convex hull optimization for Pareto-optimal bitrate ladders.

```rust
use transcode_pertitle_ml::training::{TrainingDataset, TrainingConfig, train_with_cv};

let dataset = TrainingDataset::generate_synthetic(1000);
let result = train_with_cv(&dataset, &TrainingConfig::default(), 5);
println!("Cross-validated MSE: {:.4}", result.mean_mse);
```

[Read the guide →](/docs/guides/ml-per-title-encoding)

### 🔌 Plugin Runtime

Hot-reloadable plugins with lifecycle management (Loaded → Ready → Processing → Shutdown) and sandboxed execution:

```rust
use transcode_plugin::runtime::PluginRuntime;

let mut runtime = PluginRuntime::new();
runtime.load_plugin("denoise", "/plugins/denoise.wasm")?;
runtime.initialize_all()?;
let output = runtime.process("denoise", &frame_data)?;
```

[Read the guide →](/docs/guides/plugin-runtime)

### 🔄 FFmpeg Filter Compatibility

Parse and translate FFmpeg `-vf`/`-af` filter graphs to native Transcode filters. Supports scale, fps, crop, deinterlace, color adjustment, loudness normalization, and more:

```rust
use transcode_ffcompat::filter::FilterGraph;

let graph = FilterGraph::parse("scale=1920:1080,yadif,eq=brightness=0.1");
let filters = graph.translate();
println!("Support ratio: {:.0}%", graph.support_ratio() * 100.0);
```

[Read the guide →](/docs/guides/ffmpeg-filter-compat)

### 🤖 GenAI Inference Engine

Model session management with LRU caching, batch inference, and PSNR-based quality validation:

```rust
use transcode_genai::inference::{ModelCache, BatchInference};

let mut cache = ModelCache::new(4 * 1024 * 1024 * 1024);
cache.load_model("esrgan-4x", ModelBackend::ONNX, ModelPrecision::Fp16, 512_000_000)?;
```

[Read the guide →](/docs/guides/genai-inference)

### ☁️ Multi-Cloud Health Monitoring

Circuit breaker pattern with health probes and cost estimation across cloud providers:

```rust
use transcode_multicloud::health::{CircuitBreaker, CostEstimator};

let mut breaker = CircuitBreaker::new(5, 3, Duration::from_secs(30));
// Automatically routes around failed providers
```

[Read the guide →](/docs/guides/multicloud-health)

### 🔐 Confidential Computing Audit

Tamper-proof hash-chained audit logs and policy-based key rotation for TEE environments:

```rust
use transcode_confidential::audit::{AuditLog, KeyRotationManager};

let mut log = AuditLog::new();
log.log(AuditEventType::KeyGenerated, Some("enclave-1".into()), ...);
assert!(log.verify_integrity()); // Detects any tampering
```

[Read the guide →](/docs/guides/confidential-audit)

### 🎬 EDL Import & Export

Parse CMX 3600 and FCP XML edit decision lists. Round-trip between professional editing formats:

```rust
use transcode_edit::import::{parse_cmx3600, export_fcp_xml};

let parsed = parse_cmx3600(&edl_text)?;
let xml = export_fcp_xml(&edl, 30.0);
```

[Read the guide →](/docs/guides/edl-import-export)

### 📊 Real-Time Analytics & Alerting

Configurable alert rules with cooldown, metric streaming, and chart data generation:

```rust
use transcode_dashboard::alerts::{AlertEngine, AlertRule, MetricStream};

let mut engine = AlertEngine::new();
engine.add_rule(AlertRule {
    id: "low-psnr".into(),
    metric: AlertMetricType::Psnr,
    condition: AlertCondition::Below(30.0),
    severity: AlertSeverity::Warning,
    ..
});
```

[Read the guide →](/docs/guides/analytics-alerting)

## Getting Started

All features are available now. Install or update:

```bash
cargo add transcode
```

Check the [documentation](/docs/getting-started/installation) for detailed setup, or explore the [full guide index](/docs/guides/basic-transcoding) to find the feature you need.

## What's Next

We're working on VVC/H.266 codec support, live streaming input, and audio enhancement. Follow us on [GitHub](https://github.com/transcode/transcode) for updates.
