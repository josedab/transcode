# ADR-0008: Coordinator-Worker Distributed Architecture

## Status

Accepted

## Date

2024-05 (inferred from module structure)

## Context

Large-scale transcoding workloads require horizontal scaling across multiple machines:

1. **Long videos** (hours) take too long on a single machine
2. **Batch processing** many files simultaneously
3. **Redundancy** for fault tolerance
4. **Resource utilization** across heterogeneous hardware

We need to support:

- Automatic work distribution
- Fault tolerance with retries
- Progress tracking and monitoring
- Heterogeneous worker capabilities
- Graceful scaling up and down

The challenge is coordinating work across machines while handling failures gracefully.

## Decision

Implement a **coordinator-worker architecture** with **segment-based job splitting** and a **task state machine** for reliable execution.

### 1. Coordinator-Worker Model

Central coordinator manages jobs and workers:

```rust
pub struct Coordinator {
    jobs: HashMap<Uuid, Job>,
    tasks: HashMap<Uuid, Task>,
    workers: WorkerPool,
    config: CoordinatorConfig,
}

impl Coordinator {
    pub async fn submit_job(
        &mut self,
        name: String,
        source: String,
        output: String,
        params: TranscodeParams,
        duration: f64,
    ) -> Result<Uuid> {
        let job = Job::new(name, source.clone(), output, params.clone());
        let job_id = job.id;

        // Split video into segments
        let splitter = SegmentSplitter::new(self.config.segment_duration);
        let segments = splitter.split(&source, duration);

        // Create task for each segment
        for segment in segments {
            let task = Task::new(job_id, segment, params.clone());
            self.tasks.insert(task.id, task);
        }

        self.jobs.insert(job_id, job);
        Ok(job_id)
    }
}
```

### 2. Segment-Based Processing

Videos are split into independent segments for parallel processing:

```rust
pub struct SegmentSplitter {
    segment_duration: f64,  // Default: 10 seconds
}

impl SegmentSplitter {
    pub fn split(&self, source: &str, duration: f64) -> Vec<VideoSegment> {
        let num_segments = (duration / self.segment_duration).ceil() as usize;
        let mut segments = Vec::with_capacity(num_segments);

        for i in 0..num_segments {
            let start = i as f64 * self.segment_duration;
            let end = ((i + 1) as f64 * self.segment_duration).min(duration);

            segments.push(VideoSegment::new(
                source.to_string(),
                start,
                end,
                i,
                num_segments,
            ));
        }

        segments
    }
}
```

### 3. Task State Machine

Tasks follow a well-defined state machine with transitions:

```rust
pub enum TaskState {
    Pending,    // Waiting to be picked up
    Queued,     // Assigned to a worker
    Running,    // Currently processing
    Completed,  // Successfully finished
    Failed,     // Processing failed
    Cancelled,  // Manually cancelled
    Retrying,   // Being retried after failure
}

impl TaskState {
    pub fn can_transition_to(&self, next: TaskState) -> bool {
        match (self, next) {
            (Pending, Queued) => true,
            (Pending, Cancelled) => true,
            (Queued, Running) => true,
            (Queued, Cancelled) => true,
            (Queued, Pending) => true,  // Re-queue
            (Running, Completed) => true,
            (Running, Failed) => true,
            (Running, Cancelled) => true,
            (Running, Retrying) => true,
            (Failed, Retrying) => true,
            (Failed, Pending) => true,  // Manual retry
            (Retrying, Pending) => true,
            (Retrying, Cancelled) => true,
            _ => false,
        }
    }
}
```

State transition diagram:

```
         ┌──────────────────────────────────────────┐
         ▼                                          │
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Pending │───▶│ Queued  │───▶│ Running │───▶│Completed│
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
         ▲              │              │
         │              │              ▼
         │              │         ┌─────────┐
         │              └────────▶│ Failed  │
         │                        └─────────┘
         │                             │
         │         ┌─────────┐         │
         └─────────│Retrying │◀────────┘
                   └─────────┘
```

### 4. Worker Management

Workers report health and receive task assignments:

```rust
pub struct WorkerInfo {
    pub id: String,
    pub address: String,
    pub capabilities: WorkerCapabilities,
    pub health: WorkerHealth,
    pub load: WorkerLoad,
}

pub struct WorkerCapabilities {
    pub codecs: Vec<String>,
    pub max_concurrent_tasks: usize,
    pub has_gpu: bool,
    pub available_memory: u64,
}

pub struct WorkerPool {
    workers: HashMap<String, WorkerInfo>,
}

impl WorkerPool {
    pub fn select_worker(&self, task: &Task) -> Option<&WorkerInfo> {
        self.workers
            .values()
            .filter(|w| w.health.is_healthy())
            .filter(|w| w.capabilities.supports(&task.params))
            .min_by_key(|w| w.load.score())
    }
}
```

### 5. Retry Logic

Automatic retries with backoff:

```rust
pub struct Task {
    pub retry_count: u32,
    pub max_retries: u32,  // Default: 3
    // ...
}

impl Task {
    pub fn retry(&mut self) -> Result<bool> {
        if self.retry_count >= self.max_retries {
            return Ok(false);
        }

        self.transition_to(TaskState::Retrying)?;
        self.transition_to(TaskState::Pending)?;
        self.worker_id = None;
        self.progress = 0.0;
        self.retry_count += 1;

        Ok(true)
    }
}
```

### 6. Event-Driven Progress

Subscribe to job events for monitoring:

```rust
pub enum CoordinatorEvent {
    JobSubmitted { job_id: Uuid },
    TaskQueued { job_id: Uuid, task_id: Uuid, worker_id: String },
    Progress { job_id: Uuid, progress: f64 },
    TaskCompleted { job_id: Uuid, task_id: Uuid },
    TaskFailed { job_id: Uuid, task_id: Uuid, error: String },
    JobCompleted { job_id: Uuid },
    WorkerJoined { worker_id: String },
    WorkerLeft { worker_id: String },
}
```

## Consequences

### Positive

1. **Horizontal scaling**: Add workers to increase throughput linearly

2. **Fault tolerance**: Failed tasks automatically retry on other workers

3. **Heterogeneous support**: Workers with different capabilities (GPU vs CPU)

4. **Progress visibility**: Real-time monitoring of job and task progress

5. **Graceful degradation**: System continues with remaining workers if some fail

6. **Priority scheduling**: Critical jobs processed before background work

### Negative

1. **Coordination overhead**: Network communication adds latency

2. **State management complexity**: Distributed state harder to debug

3. **Segment boundary artifacts**: May need post-processing to smooth transitions

4. **Infrastructure requirements**: Need to deploy coordinator and workers

### Mitigations

1. **Local mode**: Single-machine mode for development and small jobs

2. **Persistence**: Optional PostgreSQL/Redis for durable state

```rust
#[cfg(feature = "postgres")]
pub use persistence::{PostgresStore, JobStore, TaskStore};

#[cfg(feature = "redis")]
pub use redis::{RedisCoordinator, DistributedLock};
```

3. **Segment overlap**: Small overlap at boundaries for seamless stitching

4. **Health monitoring**: Heartbeats and automatic worker failover

## Alternatives Considered

### Alternative 1: Message Queue (RabbitMQ/Kafka)

Use a message queue for task distribution.

Rejected because:
- Additional infrastructure dependency
- More complex operational model
- Less control over task assignment logic

### Alternative 2: Peer-to-Peer Work Stealing

Decentralized work distribution.

Rejected because:
- Complex consensus algorithms
- Harder to implement priorities
- Less predictable behavior

### Alternative 3: Kubernetes Jobs

Use Kubernetes native job scheduling.

Rejected because:
- Kubernetes-only deployment
- Less fine-grained control
- Heavyweight for simple deployments

### Alternative 4: Single-Machine Threading

Use only local threading/async.

Rejected because:
- Cannot scale beyond single machine
- No fault tolerance for hardware failures
- Underutilizes cluster resources

## References

- [MapReduce: Simplified Data Processing](https://research.google/pubs/pub62/)
- [Celery distributed task queue](https://docs.celeryq.dev/)
- [FFmpeg segment-based encoding](https://trac.ffmpeg.org/wiki/Encode/Streaming)
- [Video encoding at scale](https://netflixtechblog.com/high-quality-video-encoding-at-scale-d159db052746)
