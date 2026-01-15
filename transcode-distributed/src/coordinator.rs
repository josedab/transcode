//! Coordinator for distributed transcoding.

use crate::error::{DistributedError, Result};
use crate::task::{Job, Task, TaskState, TranscodeParams, VideoSegment};
use crate::worker::{WorkerInfo, WorkerLoad, WorkerPool};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Coordinator configuration.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Segment duration in seconds.
    pub segment_duration: f64,
    /// Maximum concurrent jobs.
    pub max_concurrent_jobs: usize,
    /// Worker heartbeat timeout.
    pub heartbeat_timeout: Duration,
    /// Task retry delay.
    pub retry_delay: Duration,
    /// Health check interval.
    pub health_check_interval: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            segment_duration: 10.0,
            max_concurrent_jobs: 10,
            heartbeat_timeout: Duration::from_secs(30),
            retry_delay: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(10),
        }
    }
}

/// Event types for the coordinator.
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// Job submitted.
    JobSubmitted { job_id: Uuid },
    /// Job completed.
    JobCompleted { job_id: Uuid },
    /// Job failed.
    JobFailed { job_id: Uuid, error: String },
    /// Task started.
    TaskStarted { task_id: Uuid, worker_id: String },
    /// Task completed.
    TaskCompleted { task_id: Uuid },
    /// Task failed.
    TaskFailed { task_id: Uuid, error: String },
    /// Worker registered.
    WorkerRegistered { worker_id: String },
    /// Worker unregistered.
    WorkerUnregistered { worker_id: String },
    /// Progress update.
    Progress { job_id: Uuid, progress: f64 },
}


/// Coordinator for distributed transcoding.
pub struct Coordinator {
    /// Configuration.
    config: CoordinatorConfig,
    /// Job registry.
    jobs: DashMap<Uuid, Job>,
    /// Task registry.
    tasks: DashMap<Uuid, Task>,
    /// Worker pool.
    workers: Arc<WorkerPool>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<CoordinatorEvent>,
    /// Running flag.
    running: Arc<RwLock<bool>>,
}

impl Coordinator {
    /// Create a new coordinator.
    pub fn new(config: CoordinatorConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        let workers = Arc::new(WorkerPool::new(config.heartbeat_timeout));

        Self {
            config,
            jobs: DashMap::new(),
            tasks: DashMap::new(),
            workers,
            event_tx,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Subscribe to coordinator events.
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinatorEvent> {
        self.event_tx.subscribe()
    }

    /// Start the coordinator.
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            return;
        }
        *running = true;
        drop(running);

        info!("Coordinator started");

        // Start background tasks
        self.start_scheduler().await;
        self.start_health_checker().await;
    }

    /// Stop the coordinator immediately without draining tasks.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Coordinator stopped");
    }

    /// Stop the coordinator gracefully, waiting for in-flight tasks to complete.
    ///
    /// This method will:
    /// 1. Stop accepting new jobs
    /// 2. Wait for running tasks to complete (up to timeout)
    /// 3. Cancel any remaining pending tasks
    /// 4. Return the number of tasks that were forcefully cancelled
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for tasks to complete
    ///
    /// # Returns
    /// The number of tasks that were cancelled due to timeout
    pub async fn stop_graceful(&self, timeout: std::time::Duration) -> usize {
        info!(timeout_secs = timeout.as_secs(), "Initiating graceful shutdown");

        // Signal that we're stopping (prevents new task scheduling)
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        let deadline = tokio::time::Instant::now() + timeout;

        // Wait for running tasks to complete
        loop {
            let running_count = self
                .tasks
                .iter()
                .filter(|t| t.state == TaskState::Running || t.state == TaskState::Queued)
                .count();

            if running_count == 0 {
                info!("All tasks completed, shutdown complete");
                return 0;
            }

            if tokio::time::Instant::now() >= deadline {
                warn!(
                    running_tasks = running_count,
                    "Graceful shutdown timeout reached"
                );
                break;
            }

            debug!(
                running_tasks = running_count,
                "Waiting for tasks to complete"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Cancel remaining tasks
        let mut cancelled_count = 0;
        for mut task in self.tasks.iter_mut() {
            if !task.state.is_terminal() && task.cancel().is_ok() {
                cancelled_count += 1;
            }
        }

        // Update job states
        for job in self.jobs.iter() {
            let job_id = job.id;
            drop(job);
            self.check_job_completion(job_id).await;
        }

        if cancelled_count > 0 {
            warn!(cancelled_tasks = cancelled_count, "Force cancelled tasks during shutdown");
        }

        info!(cancelled_count = cancelled_count, "Coordinator shutdown complete");
        cancelled_count
    }

    /// Get count of active (non-terminal) tasks.
    pub fn active_task_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| !t.state.is_terminal())
            .count()
    }

    /// Get count of pending tasks.
    pub fn pending_task_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.state == TaskState::Pending)
            .count()
    }

    /// Get count of running tasks.
    pub fn running_task_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.state == TaskState::Running)
            .count()
    }

    /// Check if coordinator is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Submit a new job.
    pub async fn submit_job(
        &self,
        name: String,
        source: String,
        output: String,
        params: TranscodeParams,
        duration: f64,
    ) -> Result<Uuid> {
        let mut job = Job::new(name, source.clone(), output, params.clone());
        let job_id = job.id;

        // Create tasks for each segment
        let num_segments = (duration / self.config.segment_duration).ceil() as usize;

        // Use Arc to share params across all tasks (avoids O(n) clones)
        let shared_params = Arc::new(params);
        // Use Arc<str> to share source path (avoids O(n) string clones)
        let shared_source: Arc<str> = source.into();

        for i in 0..num_segments {
            let start = i as f64 * self.config.segment_duration;
            let end = ((i + 1) as f64 * self.config.segment_duration).min(duration);

            let segment = VideoSegment::new_shared(Arc::clone(&shared_source), start, end, i, num_segments);
            let task = Task::new_shared(job_id, segment, Arc::clone(&shared_params));

            job.add_task(task.id);
            self.tasks.insert(task.id, task);
        }

        self.jobs.insert(job_id, job);

        info!(job_id = %job_id, task_count = num_segments, "Job submitted");
        self.emit(CoordinatorEvent::JobSubmitted { job_id });

        Ok(job_id)
    }

    /// Cancel a job.
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<()> {
        let mut job = self
            .jobs
            .get_mut(&job_id)
            .ok_or_else(|| DistributedError::TaskNotFound(job_id.to_string()))?;

        job.state = TaskState::Cancelled;

        // Cancel all pending/running tasks
        for task_id in &job.task_ids {
            if let Some(mut task) = self.tasks.get_mut(task_id) {
                if !task.state.is_terminal() {
                    let _ = task.cancel();
                }
            }
        }

        info!(job_id = %job_id, "Job cancelled");
        Ok(())
    }

    /// Get job status.
    pub fn get_job(&self, job_id: Uuid) -> Option<Job> {
        self.jobs.get(&job_id).map(|j| j.clone())
    }

    /// Get task status.
    pub fn get_task(&self, task_id: Uuid) -> Option<Task> {
        self.tasks.get(&task_id).map(|t| t.clone())
    }

    /// Get all jobs.
    pub fn all_jobs(&self) -> Vec<Job> {
        self.jobs.iter().map(|e| e.value().clone()).collect()
    }

    /// Get all tasks for a job.
    pub fn job_tasks(&self, job_id: Uuid) -> Vec<Task> {
        self.jobs
            .get(&job_id)
            .map(|job| {
                job.task_ids
                    .iter()
                    .filter_map(|id| self.tasks.get(id).map(|t| t.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Register a worker.
    pub async fn register_worker(&self, worker: WorkerInfo) -> Result<()> {
        let worker_id = worker.id.clone();
        self.workers.register(worker).await?;

        info!(worker_id = %worker_id, "Worker registered");
        self.emit(CoordinatorEvent::WorkerRegistered { worker_id });

        Ok(())
    }

    /// Unregister a worker.
    pub async fn unregister_worker(&self, worker_id: &str) -> Result<()> {
        // Fail tasks assigned to this worker
        for mut task in self.tasks.iter_mut() {
            if task.worker_id.as_deref() == Some(worker_id) && task.state == TaskState::Running {
                let _ = task.fail("Worker disconnected".into());
            }
        }

        self.workers.unregister(worker_id).await?;

        info!("Worker {} unregistered", worker_id);
        self.emit(CoordinatorEvent::WorkerUnregistered {
            worker_id: worker_id.to_string(),
        });

        Ok(())
    }

    /// Worker heartbeat.
    pub async fn worker_heartbeat(&self, worker_id: &str, load: WorkerLoad) -> Result<()> {
        self.workers.heartbeat(worker_id, load).await
    }

    /// Get worker pool.
    pub fn worker_pool(&self) -> Arc<WorkerPool> {
        Arc::clone(&self.workers)
    }

    /// Report task progress.
    pub fn report_progress(&self, task_id: Uuid, progress: f64) {
        if let Some(mut task) = self.tasks.get_mut(&task_id) {
            task.update_progress(progress);

            // Update job progress
            if let Some(mut job) = self.jobs.get_mut(&task.job_id) {
                let tasks: Vec<_> = job
                    .task_ids
                    .iter()
                    .filter_map(|id| self.tasks.get(id).map(|t| t.clone()))
                    .collect();
                job.update_progress(&tasks);

                self.emit(CoordinatorEvent::Progress {
                    job_id: job.id,
                    progress: job.progress,
                });
            }
        }
    }

    /// Report task completion.
    pub async fn report_task_complete(&self, task_id: Uuid, output: String) -> Result<()> {
        let job_id = {
            let mut task = self
                .tasks
                .get_mut(&task_id)
                .ok_or_else(|| DistributedError::TaskNotFound(task_id.to_string()))?;

            task.complete(output)?;

            if let Some(ref worker_id) = task.worker_id {
                self.workers.complete_task(worker_id, task_id).await?;
            }

            task.job_id
        };

        info!("Task {} completed", task_id);
        self.emit(CoordinatorEvent::TaskCompleted { task_id });

        // Check if job is complete
        self.check_job_completion(job_id).await;

        Ok(())
    }

    /// Report task failure.
    pub async fn report_task_failed(&self, task_id: Uuid, error: String) -> Result<()> {
        let (job_id, can_retry) = {
            let mut task = self
                .tasks
                .get_mut(&task_id)
                .ok_or_else(|| DistributedError::TaskNotFound(task_id.to_string()))?;

            task.fail(error.clone())?;

            if let Some(ref worker_id) = task.worker_id {
                let _ = self.workers.complete_task(worker_id, task_id).await;
            }

            (task.job_id, task.can_retry())
        };

        warn!("Task {} failed: {}", task_id, error);
        self.emit(CoordinatorEvent::TaskFailed {
            task_id,
            error: error.clone(),
        });

        // Try to retry
        if can_retry {
            if let Some(mut task) = self.tasks.get_mut(&task_id) {
                if task.retry()? {
                    info!("Task {} scheduled for retry", task_id);
                }
            }
        }

        // Check if job failed
        self.check_job_completion(job_id).await;

        Ok(())
    }

    /// Start the task scheduler.
    async fn start_scheduler(&self) {
        let tasks = self.tasks.clone();
        let workers = Arc::clone(&self.workers);
        let _event_tx = self.event_tx.clone();
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                if !*running.read().await {
                    break;
                }

                // Find pending tasks
                let pending: Vec<_> = tasks
                    .iter()
                    .filter(|t| t.state == TaskState::Pending)
                    .map(|t| t.clone())
                    .collect();

                for task in pending {
                    // Try to find a worker
                    match workers.select_for_task(&task).await {
                        Ok(worker) => {
                            // Assign task to worker
                            if let Some(mut t) = tasks.get_mut(&task.id) {
                                if t.queue_for(worker.id.clone()).is_ok()
                                    && workers.assign_task(&worker.id, task.id).await.is_ok() {
                                        debug!("Task {} assigned to worker {}", task.id, worker.id);
                                    }
                            }
                        }
                        Err(DistributedError::NoWorkersAvailable) => {
                            // No workers, wait for next cycle
                            break;
                        }
                        Err(e) => {
                            warn!("Failed to assign task {}: {}", task.id, e);
                        }
                    }
                }
            }
        });
    }

    /// Start the health checker.
    async fn start_health_checker(&self) {
        let workers = Arc::clone(&self.workers);
        let tasks = self.tasks.clone();
        let running = Arc::clone(&self.running);
        let interval_duration = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                if !*running.read().await {
                    break;
                }

                // Check worker health
                workers.check_health().await;

                // Re-queue tasks from offline workers
                for mut task in tasks.iter_mut() {
                    if task.state == TaskState::Queued || task.state == TaskState::Running {
                        if let Some(ref worker_id) = task.worker_id {
                            if let Some(worker) = workers.get(worker_id).await {
                                if worker.health == crate::worker::WorkerHealth::Offline {
                                    warn!(
                                        "Re-queuing task {} from offline worker {}",
                                        task.id, worker_id
                                    );
                                    task.state = TaskState::Pending;
                                    task.worker_id = None;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Check if a job is complete.
    async fn check_job_completion(&self, job_id: Uuid) {
        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            let tasks: Vec<_> = job
                .task_ids
                .iter()
                .filter_map(|id| self.tasks.get(id).map(|t| t.clone()))
                .collect();

            job.update_progress(&tasks);

            match job.state {
                TaskState::Completed => {
                    info!("Job {} completed", job_id);
                    self.emit(CoordinatorEvent::JobCompleted { job_id });
                }
                TaskState::Failed => {
                    let error = tasks
                        .iter()
                        .find_map(|t| t.error.clone())
                        .unwrap_or_else(|| "Unknown error".to_string());

                    error!("Job {} failed: {}", job_id, error);
                    self.emit(CoordinatorEvent::JobFailed { job_id, error });
                }
                _ => {}
            }
        }
    }

    /// Emit an event.
    fn emit(&self, event: CoordinatorEvent) {
        let _ = self.event_tx.send(event);
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new(CoordinatorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_job_submission() {
        let coordinator = Coordinator::default();

        let job_id = coordinator
            .submit_job(
                "test-job".into(),
                "/input/video.mp4".into(),
                "/output/video.mp4".into(),
                TranscodeParams::default(),
                30.0, // 30 seconds
            )
            .await
            .unwrap();

        let job = coordinator.get_job(job_id).unwrap();
        assert_eq!(job.name, "test-job");
        assert_eq!(job.task_ids.len(), 3); // 30s / 10s segments = 3 tasks
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let coordinator = Coordinator::default();

        let worker = WorkerInfo::new(
            "worker-1".into(),
            "localhost:8080".into(),
            crate::worker::WorkerCapabilities::default(),
        );

        coordinator.register_worker(worker.clone()).await.unwrap();

        let workers = coordinator.workers.workers().await;
        assert_eq!(workers.len(), 1);
    }
}
