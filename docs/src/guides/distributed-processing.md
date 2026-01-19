# Distributed Processing

This guide covers distributed transcoding using the `transcode-distributed` crate.

## Overview

Transcode supports distributed processing for large-scale video transcoding:

- **Horizontal scaling** - Add workers to increase throughput
- **Fault tolerance** - Automatic retry and failover
- **Load balancing** - Intelligent task distribution
- **Progress tracking** - Real-time job monitoring

## Architecture

```
┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ Coordinator │
└─────────────┘     └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ Worker 1 │    │ Worker 2 │    │ Worker 3 │
     └──────────┘    └──────────┘    └──────────┘
```

## Coordinator Setup

```rust
use transcode_distributed::{Coordinator, CoordinatorConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = CoordinatorConfig::default()
        .bind_address("0.0.0.0:8080")
        .max_workers(10)
        .task_timeout(Duration::from_secs(300));

    let coordinator = Coordinator::new(config).await?;

    // Start accepting connections
    coordinator.run().await?;

    Ok(())
}
```

### Configuration

```rust
let config = CoordinatorConfig::default()
    .bind_address("0.0.0.0:8080")     // Listen address
    .max_workers(100)                   // Maximum workers
    .task_timeout(Duration::from_secs(300))  // Task timeout
    .retry_attempts(3)                  // Retry failed tasks
    .heartbeat_interval(Duration::from_secs(10))
    .dead_worker_timeout(Duration::from_secs(30));
```

## Worker Setup

```rust
use transcode_distributed::{Worker, WorkerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = WorkerConfig::default()
        .coordinator_url("http://coordinator:8080")
        .worker_id("worker-1")
        .capabilities(WorkerCapabilities {
            max_concurrent: 4,
            gpu_available: true,
            codecs: vec!["h264", "hevc", "av1"],
        });

    let worker = Worker::new(config).await?;

    // Start processing tasks
    worker.run().await?;

    Ok(())
}
```

## Job Submission

### Simple Job

```rust
use transcode_distributed::{Coordinator, Job, TranscodeParams};

let params = TranscodeParams::default()
    .video_codec("h264")
    .video_bitrate(5_000_000)
    .audio_codec("aac")
    .audio_bitrate(128_000);

let job = coordinator.submit_job(
    "input.mp4",
    "output.mp4",
    params,
).await?;

println!("Job ID: {}", job.id);
```

### Segmented Job

For large files, split into segments for parallel processing:

```rust
use transcode_distributed::{Coordinator, SegmentedJob};

// Split video into segments
let segments = split_video("input.mp4", Duration::from_secs(10))?;

// Submit as segmented job
let job = coordinator.submit_segmented_job(
    segments,
    params,
).await?;
```

## Progress Monitoring

### Event Stream

```rust
use transcode_distributed::CoordinatorEvent;

while let Some(event) = coordinator.next_event().await {
    match event {
        CoordinatorEvent::TaskStarted { task_id, worker_id } => {
            println!("Task {} started on worker {}", task_id, worker_id);
        }
        CoordinatorEvent::TaskProgress { task_id, progress } => {
            println!("Task {}: {:.1}%", task_id, progress * 100.0);
        }
        CoordinatorEvent::TaskCompleted { task_id, result } => {
            println!("Task {} completed", task_id);
        }
        CoordinatorEvent::TaskFailed { task_id, error } => {
            eprintln!("Task {} failed: {}", task_id, error);
        }
        CoordinatorEvent::JobCompleted { job_id, stats } => {
            println!("Job {} finished!", job_id);
            println!("  Total time: {:?}", stats.total_time);
            println!("  Tasks: {}", stats.task_count);
            break;
        }
        _ => {}
    }
}
```

### Job Status

```rust
let status = coordinator.job_status(&job.id).await?;

println!("Job: {}", status.id);
println!("State: {:?}", status.state);
println!("Progress: {:.1}%", status.progress * 100.0);
println!("Tasks: {}/{}", status.completed_tasks, status.total_tasks);
```

## Fault Tolerance

### Automatic Retry

```rust
let config = CoordinatorConfig::default()
    .retry_attempts(3)                    // Retry up to 3 times
    .retry_delay(Duration::from_secs(5))  // Wait between retries
    .task_timeout(Duration::from_secs(300));
```

### Worker Health

```rust
// Get worker status
let workers = coordinator.list_workers().await?;

for worker in workers {
    println!("Worker: {}", worker.id);
    println!("  Status: {:?}", worker.status);
    println!("  Load: {}/{}", worker.active_tasks, worker.max_concurrent);
    println!("  Last heartbeat: {:?} ago", worker.last_heartbeat.elapsed());
}
```

### Graceful Shutdown

```rust
use transcode_distributed::ShutdownMode;

// Graceful: wait for current tasks to complete
coordinator.shutdown(ShutdownMode::Graceful).await?;

// Immediate: cancel all tasks
coordinator.shutdown(ShutdownMode::Immediate).await?;

// Drain: stop accepting new tasks, finish current ones
coordinator.shutdown(ShutdownMode::Drain).await?;
```

## Load Balancing

### Strategies

```rust
use transcode_distributed::LoadBalanceStrategy;

let config = CoordinatorConfig::default()
    .load_balance(LoadBalanceStrategy::LeastLoaded);  // Default

// Available strategies:
// - LeastLoaded: Prefer workers with fewer active tasks
// - RoundRobin: Distribute evenly
// - Capabilities: Match task requirements to worker capabilities
// - Affinity: Prefer same worker for related tasks
```

### Task Affinity

```rust
// Keep related tasks on same worker
let job = coordinator.submit_segmented_job(segments, params)
    .with_affinity(TaskAffinity::SameWorker)
    .await?;
```

## Scaling

### Dynamic Scaling

```rust
use transcode_distributed::ScalingPolicy;

let config = CoordinatorConfig::default()
    .scaling_policy(ScalingPolicy::Auto {
        min_workers: 2,
        max_workers: 10,
        scale_up_threshold: 0.8,   // Scale up at 80% utilization
        scale_down_threshold: 0.3, // Scale down at 30% utilization
    });
```

### Manual Scaling

```rust
// Add worker dynamically
coordinator.add_worker("worker-4", worker_config).await?;

// Remove worker (gracefully)
coordinator.remove_worker("worker-2").await?;
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  coordinator:
    image: transcode:latest
    command: coordinator
    ports:
      - "8080:8080"
    environment:
      - MAX_WORKERS=10

  worker:
    image: transcode:latest
    command: worker
    deploy:
      replicas: 3
    environment:
      - COORDINATOR_URL=http://coordinator:8080
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transcode-worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: transcode-worker
  template:
    spec:
      containers:
        - name: worker
          image: transcode:latest
          env:
            - name: COORDINATOR_URL
              value: "http://transcode-coordinator:8080"
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
```

## Monitoring

### Metrics

```rust
let metrics = coordinator.metrics().await?;

println!("Active jobs: {}", metrics.active_jobs);
println!("Pending tasks: {}", metrics.pending_tasks);
println!("Active workers: {}", metrics.active_workers);
println!("Tasks/second: {:.2}", metrics.throughput);
```

### Prometheus Integration

```rust
let config = CoordinatorConfig::default()
    .metrics_endpoint("/metrics")
    .metrics_port(9090);
```
