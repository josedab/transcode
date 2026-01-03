# transcode-distributed

Distributed video transcoding system with coordinator-worker architecture for parallel processing across multiple nodes.

## Architecture

```
                    +--------------+
                    | Coordinator  |
                    |  - Scheduler |
                    |  - Health    |
                    +------+-------+
                           |
         +-----------------+-----------------+
         |                 |                 |
   +-----v-----+     +-----v-----+     +-----v-----+
   |  Worker 1 |     |  Worker 2 |     |  Worker N |
   |  (GPU)    |     |  (CPU)    |     |  (CPU)    |
   +-----------+     +-----------+     +-----------+
```

**Coordinator**: Central node managing job scheduling, task distribution, and worker health monitoring.

**Workers**: Processing nodes that execute transcoding tasks. Support GPU acceleration and codec-specific routing.

**Segment-based Processing**: Videos are split into segments (default 10s) for parallel transcoding across workers.

## Key Types

| Type | Description |
|------|-------------|
| `Coordinator` | Central job scheduler and worker manager |
| `WorkerPool` | Manages worker registration and task assignment |
| `WorkerInfo` | Worker metadata, capabilities, and health status |
| `Job` | Complete transcoding job (collection of tasks) |
| `Task` | Individual segment processing unit |
| `TranscodeParams` | Codec, resolution, bitrate settings |

## Usage

### Coordinator Setup

```rust
use transcode_distributed::{Coordinator, CoordinatorConfig, WorkerInfo, WorkerCapabilities};

#[tokio::main]
async fn main() {
    let coordinator = Coordinator::new(CoordinatorConfig::default());
    coordinator.start().await;

    // Register a worker
    let worker = WorkerInfo::new(
        "worker-1".into(),
        "192.168.1.10:8080".into(),
        WorkerCapabilities::default(),
    );
    coordinator.register_worker(worker).await.unwrap();

    // Submit a job
    let job_id = coordinator.submit_job(
        "my-video".into(),
        "/input/video.mp4".into(),
        "/output/video.mp4".into(),
        TranscodeParams::default(),
        120.0, // duration in seconds
    ).await.unwrap();
}
```

### Event Monitoring

```rust
let mut events = coordinator.subscribe();
while let Ok(event) = events.recv().await {
    match event {
        CoordinatorEvent::Progress { job_id, progress } => {
            println!("Job {}: {:.1}%", job_id, progress * 100.0);
        }
        CoordinatorEvent::JobCompleted { job_id } => break,
        _ => {}
    }
}
```

## Features

- **Automatic segmentation**: Videos split into parallel segments
- **Smart worker selection**: Routes AV1/HEVC to GPU workers when available
- **Fault tolerance**: Automatic task retries and worker failover
- **Health monitoring**: Heartbeat-based worker status tracking
- **Priority scheduling**: Low, Normal, High, Critical task priorities

## Optional Features

```toml
[dependencies]
transcode-distributed = { version = "0.1", features = ["grpc"] }
```

| Feature | Description |
|---------|-------------|
| `grpc` | gRPC transport via tonic/prost |
| `rabbitmq` | RabbitMQ message queue support |
