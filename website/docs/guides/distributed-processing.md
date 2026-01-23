---
sidebar_position: 6
title: Distributed Processing
description: Scale transcoding across multiple workers
---

# Distributed Processing

The `transcode-distributed` crate enables scaling transcoding workloads across multiple machines.

## Overview

Distributed transcoding uses a coordinator/worker architecture:

```
                    ┌─────────────┐
                    │ Coordinator │
                    │             │
                    │ - Job queue │
                    │ - Scheduling│
                    │ - Monitoring│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │ Worker  │        │ Worker  │        │ Worker  │
   │  Node 1 │        │  Node 2 │        │  Node 3 │
   └─────────┘        └─────────┘        └─────────┘
```

## Setup

```toml
[dependencies]
transcode = { version = "1.0", features = ["distributed"] }
transcode-distributed = "1.0"
```

## Coordinator

The coordinator manages jobs and distributes tasks to workers.

### Starting a Coordinator

```rust
use transcode_distributed::{Coordinator, CoordinatorConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CoordinatorConfig {
        bind_address: "0.0.0.0:8080".parse()?,
        max_workers: 100,
        task_timeout: Duration::from_secs(300),
        ..Default::default()
    };

    let coordinator = Coordinator::new(config).await?;
    coordinator.run().await?;

    Ok(())
}
```

### Submitting Jobs

```rust
use transcode_distributed::{Client, Job, JobParams};

let client = Client::connect("http://coordinator:8080").await?;

let job = Job::new("transcode_video")
    .input("s3://bucket/input.mp4")
    .output("s3://bucket/output.mp4")
    .params(JobParams {
        video_codec: "h264".to_string(),
        video_bitrate: 5_000_000,
        audio_codec: "aac".to_string(),
        audio_bitrate: 128_000,
        ..Default::default()
    });

let job_id = client.submit(job).await?;
println!("Submitted job: {}", job_id);
```

### Monitoring Jobs

```rust
// Get job status
let status = client.get_status(&job_id).await?;
println!("Job status: {:?}", status.state);
println!("Progress: {:.1}%", status.progress);

// Wait for completion
let result = client.wait_for_completion(&job_id).await?;
match result {
    JobResult::Completed(stats) => {
        println!("Job completed!");
        println!("Frames: {}", stats.frames_encoded);
    }
    JobResult::Failed(error) => {
        println!("Job failed: {}", error);
    }
}
```

## Worker

Workers process tasks assigned by the coordinator.

### Starting a Worker

```rust
use transcode_distributed::{Worker, WorkerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = WorkerConfig {
        coordinator_url: "http://coordinator:8080".to_string(),
        worker_id: "worker-1".to_string(),
        capabilities: WorkerCapabilities {
            max_concurrent_tasks: 2,
            supported_codecs: vec!["h264", "h265", "av1"],
            gpu_available: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let worker = Worker::new(config).await?;
    worker.run().await?;

    Ok(())
}
```

### Worker Capabilities

Workers advertise their capabilities:

```rust
let capabilities = WorkerCapabilities {
    // Processing capacity
    max_concurrent_tasks: 4,
    cpu_cores: 16,
    memory_gb: 64,

    // Codec support
    supported_codecs: vec!["h264", "h265", "av1", "vp9"],

    // Hardware
    gpu_available: true,
    gpu_memory_gb: Some(8),
    hardware_encoders: vec!["nvenc", "qsv"],

    // Storage
    temp_storage_gb: 500,
};
```

## Segment-Based Processing

Large videos are split into segments for parallel processing:

```rust
use transcode_distributed::{SegmentConfig, SplitStrategy};

let job = Job::new("transcode_video")
    .input("s3://bucket/long_video.mp4")
    .output("s3://bucket/output.mp4")
    .segment_config(SegmentConfig {
        strategy: SplitStrategy::Duration(Duration::from_secs(30)),
        overlap: Duration::from_secs(1),  // For seamless joins
        ..Default::default()
    });
```

### Processing Flow

```
Input Video
    │
    ▼
┌──────────────┐
│   Split      │ ─── Divide into segments at keyframes
└──────────────┘
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
┌────────┐    ┌────────┐    ┌────────┐
│ Seg 1  │    │ Seg 2  │    │ Seg 3  │  ─── Process in parallel
└────────┘    └────────┘    └────────┘
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
           ┌──────────────┐
           │    Merge     │ ─── Combine into final output
           └──────────────┘
                   │
                   ▼
            Output Video
```

## Storage Integration

### S3 Storage

```rust
use transcode_distributed::storage::S3Storage;

let storage = S3Storage::new(S3Config {
    bucket: "my-bucket".to_string(),
    region: "us-east-1".to_string(),
    credentials: AwsCredentials::from_env(),
})?;

let job = Job::new("transcode_video")
    .storage(storage)
    .input("s3://my-bucket/input.mp4")
    .output("s3://my-bucket/output.mp4");
```

### Google Cloud Storage

```rust
use transcode_distributed::storage::GcsStorage;

let storage = GcsStorage::new(GcsConfig {
    bucket: "my-bucket".to_string(),
    credentials_path: "/path/to/credentials.json".to_string(),
})?;
```

### Local/NFS Storage

```rust
use transcode_distributed::storage::FileStorage;

let storage = FileStorage::new("/mnt/shared/media");
```

## Fault Tolerance

### Automatic Retries

```rust
let config = CoordinatorConfig {
    retry_policy: RetryPolicy {
        max_retries: 3,
        retry_delay: Duration::from_secs(10),
        backoff_multiplier: 2.0,
    },
    ..Default::default()
};
```

### Worker Health Checks

```rust
let config = CoordinatorConfig {
    health_check_interval: Duration::from_secs(30),
    worker_timeout: Duration::from_secs(60),
    ..Default::default()
};
```

### Task Reassignment

If a worker fails, tasks are automatically reassigned:

```rust
let config = CoordinatorConfig {
    task_timeout: Duration::from_secs(300),
    reassign_on_timeout: true,
    ..Default::default()
};
```

## Scaling

### Auto-Scaling Workers

```rust
use transcode_distributed::autoscale::{AutoScaler, ScalePolicy};

let scaler = AutoScaler::new(ScalePolicy {
    min_workers: 2,
    max_workers: 20,
    scale_up_threshold: 0.8,    // 80% utilization
    scale_down_threshold: 0.3,  // 30% utilization
    cooldown: Duration::from_secs(300),
});

coordinator.set_autoscaler(scaler);
```

### Kubernetes Deployment

```yaml
# coordinator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transcode-coordinator
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: coordinator
        image: transcode/coordinator:latest
        ports:
        - containerPort: 8080
---
# worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transcode-worker
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: worker
        image: transcode/worker:latest
        env:
        - name: COORDINATOR_URL
          value: "http://transcode-coordinator:8080"
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
```

## Monitoring

### Prometheus Metrics

```rust
let config = CoordinatorConfig {
    metrics_endpoint: "/metrics".to_string(),
    ..Default::default()
};
```

Exposed metrics:
- `transcode_jobs_total` - Total jobs submitted
- `transcode_jobs_completed` - Completed jobs
- `transcode_jobs_failed` - Failed jobs
- `transcode_tasks_active` - Currently processing tasks
- `transcode_worker_count` - Active workers
- `transcode_queue_depth` - Pending tasks

### Dashboard Integration

```rust
use transcode_distributed::dashboard::Dashboard;

let dashboard = Dashboard::new(DashboardConfig {
    bind_address: "0.0.0.0:3000".parse()?,
    ..Default::default()
});

// Access at http://localhost:3000
```

## Priority Queues

```rust
let job = Job::new("urgent_transcode")
    .priority(Priority::High)  // High, Normal, Low
    .input("s3://bucket/urgent.mp4")
    .output("s3://bucket/output.mp4");
```

## Event Webhooks

```rust
let job = Job::new("transcode_video")
    .webhook(WebhookConfig {
        url: "https://api.example.com/transcode/callback".to_string(),
        events: vec![Event::Completed, Event::Failed],
        secret: Some("webhook_secret".to_string()),
    })
    .input("s3://bucket/input.mp4")
    .output("s3://bucket/output.mp4");
```

Webhook payload:
```json
{
  "event": "completed",
  "job_id": "job-123",
  "timestamp": "2024-01-15T10:30:00Z",
  "result": {
    "frames_encoded": 3600,
    "duration_seconds": 120,
    "output_size_bytes": 50000000
  }
}
```

## CLI Usage

```bash
# Start coordinator
transcode-coordinator --bind 0.0.0.0:8080

# Start worker
transcode-worker --coordinator http://localhost:8080 --id worker-1

# Submit job
transcode-cli distributed submit \
  --input s3://bucket/input.mp4 \
  --output s3://bucket/output.mp4 \
  --video-bitrate 5000

# Check status
transcode-cli distributed status job-123

# List workers
transcode-cli distributed workers
```

## Next Steps

- [Docker Deployment](/docs/guides/docker-deployment) - Container orchestration
- [Hardware Acceleration](/docs/advanced/hardware-acceleration) - GPU workers
- [Configuration Reference](/docs/reference/configuration) - All distributed options
