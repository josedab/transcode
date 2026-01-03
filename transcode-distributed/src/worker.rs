//! Worker management for distributed transcoding.

use crate::error::{DistributedError, Result};
use crate::task::Task;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Worker health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WorkerHealth {
    /// Worker is healthy and accepting tasks.
    #[default]
    Healthy,
    /// Worker is degraded but still operational.
    Degraded,
    /// Worker is unhealthy and should not receive tasks.
    Unhealthy,
    /// Worker is offline.
    Offline,
}

impl std::fmt::Display for WorkerHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Offline => write!(f, "offline"),
        }
    }
}

/// Worker capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Supported codecs.
    pub codecs: Vec<String>,
    /// Maximum concurrent tasks.
    pub max_concurrent: usize,
    /// GPU available.
    pub has_gpu: bool,
    /// GPU model (if available).
    pub gpu_model: Option<String>,
    /// Available memory in bytes.
    pub memory_bytes: u64,
    /// Number of CPU cores.
    pub cpu_cores: usize,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            codecs: vec!["h264".into(), "h265".into(), "av1".into()],
            max_concurrent: 2,
            has_gpu: false,
            gpu_model: None,
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            cpu_cores: 4,
        }
    }
}

/// Worker load information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLoad {
    /// Number of active tasks.
    pub active_tasks: usize,
    /// CPU usage (0.0 - 1.0).
    pub cpu_usage: f64,
    /// Memory usage (0.0 - 1.0).
    pub memory_usage: f64,
    /// GPU usage (0.0 - 1.0), if applicable.
    pub gpu_usage: Option<f64>,
}

impl Default for WorkerLoad {
    fn default() -> Self {
        Self {
            active_tasks: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: None,
        }
    }
}

impl WorkerLoad {
    /// Calculate overall load score (0.0 - 1.0).
    pub fn score(&self) -> f64 {
        let cpu_weight = 0.4;
        let mem_weight = 0.3;
        let task_weight = 0.3;

        let task_ratio = self.active_tasks as f64 / 4.0; // Assume max 4 tasks
        let task_score = task_ratio.min(1.0);

        self.cpu_usage * cpu_weight + self.memory_usage * mem_weight + task_score * task_weight
    }
}

/// Worker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique worker identifier.
    pub id: String,
    /// Worker name.
    pub name: String,
    /// Worker address (e.g., "192.168.1.10:8080").
    pub address: String,
    /// Worker capabilities.
    pub capabilities: WorkerCapabilities,
    /// Current load.
    pub load: WorkerLoad,
    /// Health status.
    pub health: WorkerHealth,
    /// Registration timestamp.
    pub registered_at: DateTime<Utc>,
    /// Last heartbeat timestamp.
    pub last_heartbeat: DateTime<Utc>,
    /// Active task IDs.
    pub active_tasks: HashSet<Uuid>,
}

impl WorkerInfo {
    /// Create a new worker info.
    pub fn new(name: String, address: String, capabilities: WorkerCapabilities) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            address,
            capabilities,
            load: WorkerLoad::default(),
            health: WorkerHealth::Healthy,
            registered_at: now,
            last_heartbeat: now,
            active_tasks: HashSet::new(),
        }
    }

    /// Check if worker can accept more tasks.
    pub fn can_accept_task(&self) -> bool {
        self.health == WorkerHealth::Healthy
            && self.active_tasks.len() < self.capabilities.max_concurrent
    }

    /// Check if worker supports a codec.
    pub fn supports_codec(&self, codec: &str) -> bool {
        self.capabilities
            .codecs
            .iter()
            .any(|c| c.eq_ignore_ascii_case(codec))
    }

    /// Update heartbeat.
    pub fn heartbeat(&mut self, load: WorkerLoad) {
        self.last_heartbeat = Utc::now();
        self.load = load;
    }

    /// Check if worker is stale (no heartbeat in timeout period).
    pub fn is_stale(&self, timeout: Duration) -> bool {
        let elapsed = Utc::now() - self.last_heartbeat;
        elapsed.to_std().unwrap_or_default() > timeout
    }

    /// Add a task to this worker.
    pub fn add_task(&mut self, task_id: Uuid) {
        self.active_tasks.insert(task_id);
        self.load.active_tasks = self.active_tasks.len();
    }

    /// Remove a task from this worker.
    pub fn remove_task(&mut self, task_id: Uuid) {
        self.active_tasks.remove(&task_id);
        self.load.active_tasks = self.active_tasks.len();
    }
}

/// Worker pool for managing multiple workers.
#[derive(Debug)]
pub struct WorkerPool {
    /// Registered workers.
    workers: Arc<RwLock<Vec<WorkerInfo>>>,
    /// Heartbeat timeout.
    heartbeat_timeout: Duration,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(heartbeat_timeout: Duration) -> Self {
        Self {
            workers: Arc::new(RwLock::new(Vec::new())),
            heartbeat_timeout,
        }
    }

    /// Register a worker.
    pub async fn register(&self, worker: WorkerInfo) -> Result<()> {
        let mut workers = self.workers.write().await;

        // Check for duplicate
        if workers.iter().any(|w| w.id == worker.id) {
            return Err(DistributedError::Internal(format!(
                "Worker {} already registered",
                worker.id
            )));
        }

        workers.push(worker);
        Ok(())
    }

    /// Unregister a worker.
    pub async fn unregister(&self, worker_id: &str) -> Result<()> {
        let mut workers = self.workers.write().await;

        let pos = workers
            .iter()
            .position(|w| w.id == worker_id)
            .ok_or_else(|| DistributedError::WorkerNotFound(worker_id.to_string()))?;

        workers.remove(pos);
        Ok(())
    }

    /// Update worker heartbeat.
    pub async fn heartbeat(&self, worker_id: &str, load: WorkerLoad) -> Result<()> {
        let mut workers = self.workers.write().await;

        let worker = workers
            .iter_mut()
            .find(|w| w.id == worker_id)
            .ok_or_else(|| DistributedError::WorkerNotFound(worker_id.to_string()))?;

        worker.heartbeat(load);
        Ok(())
    }

    /// Get all workers.
    pub async fn workers(&self) -> Vec<WorkerInfo> {
        self.workers.read().await.clone()
    }

    /// Get healthy workers.
    pub async fn healthy_workers(&self) -> Vec<WorkerInfo> {
        self.workers
            .read()
            .await
            .iter()
            .filter(|w| w.health == WorkerHealth::Healthy && !w.is_stale(self.heartbeat_timeout))
            .cloned()
            .collect()
    }

    /// Get available workers (healthy and can accept tasks).
    pub async fn available_workers(&self) -> Vec<WorkerInfo> {
        self.workers
            .read()
            .await
            .iter()
            .filter(|w| w.can_accept_task() && !w.is_stale(self.heartbeat_timeout))
            .cloned()
            .collect()
    }

    /// Get a worker by ID.
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo> {
        self.workers
            .read()
            .await
            .iter()
            .find(|w| w.id == worker_id)
            .cloned()
    }

    /// Select best worker for a task.
    pub async fn select_for_task(&self, task: &Task) -> Result<WorkerInfo> {
        let available = self.available_workers().await;

        if available.is_empty() {
            return Err(DistributedError::NoWorkersAvailable);
        }

        // Filter by codec support
        let codec = &task.params.codec;
        let mut candidates: Vec<_> = available
            .into_iter()
            .filter(|w| w.supports_codec(codec))
            .collect();

        if candidates.is_empty() {
            return Err(DistributedError::NoWorkersAvailable);
        }

        // Sort by load (lowest first)
        candidates.sort_by(|a, b| {
            a.load
                .score()
                .partial_cmp(&b.load.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Prefer GPU workers for certain codecs
        if matches!(codec.as_str(), "av1" | "hevc" | "h265") {
            if let Some(gpu_worker) = candidates.iter().find(|w| w.capabilities.has_gpu) {
                return Ok(gpu_worker.clone());
            }
        }

        Ok(candidates.remove(0))
    }

    /// Assign a task to a worker.
    pub async fn assign_task(&self, worker_id: &str, task_id: Uuid) -> Result<()> {
        let mut workers = self.workers.write().await;

        let worker = workers
            .iter_mut()
            .find(|w| w.id == worker_id)
            .ok_or_else(|| DistributedError::WorkerNotFound(worker_id.to_string()))?;

        if !worker.can_accept_task() {
            return Err(DistributedError::Internal(
                "Worker cannot accept more tasks".to_string(),
            ));
        }

        worker.add_task(task_id);
        Ok(())
    }

    /// Complete a task on a worker.
    pub async fn complete_task(&self, worker_id: &str, task_id: Uuid) -> Result<()> {
        let mut workers = self.workers.write().await;

        let worker = workers
            .iter_mut()
            .find(|w| w.id == worker_id)
            .ok_or_else(|| DistributedError::WorkerNotFound(worker_id.to_string()))?;

        worker.remove_task(task_id);
        Ok(())
    }

    /// Mark stale workers as unhealthy.
    pub async fn check_health(&self) {
        let mut workers = self.workers.write().await;

        for worker in workers.iter_mut() {
            if worker.is_stale(self.heartbeat_timeout) {
                worker.health = WorkerHealth::Offline;
            }
        }
    }

    /// Get worker count.
    pub async fn count(&self) -> usize {
        self.workers.read().await.len()
    }

    /// Get healthy worker count.
    pub async fn healthy_count(&self) -> usize {
        self.healthy_workers().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_registration() {
        let pool = WorkerPool::default();

        let worker = WorkerInfo::new(
            "worker-1".into(),
            "localhost:8080".into(),
            WorkerCapabilities::default(),
        );

        pool.register(worker.clone()).await.unwrap();
        assert_eq!(pool.count().await, 1);

        let retrieved = pool.get(&worker.id).await.unwrap();
        assert_eq!(retrieved.name, "worker-1");
    }

    #[tokio::test]
    async fn test_worker_selection() {
        let pool = WorkerPool::default();

        let worker1 = WorkerInfo::new(
            "worker-1".into(),
            "localhost:8080".into(),
            WorkerCapabilities::default(),
        );

        let mut worker2 = WorkerInfo::new(
            "worker-2".into(),
            "localhost:8081".into(),
            WorkerCapabilities {
                has_gpu: true,
                ..Default::default()
            },
        );
        worker2.load.cpu_usage = 0.2;

        pool.register(worker1).await.unwrap();
        pool.register(worker2).await.unwrap();

        let task = Task::new(
            Uuid::new_v4(),
            crate::task::VideoSegment::new("test.mp4".into(), 0.0, 10.0, 0, 1),
            crate::task::TranscodeParams {
                codec: "av1".into(),
                ..Default::default()
            },
        );

        // Should prefer GPU worker for AV1
        let selected = pool.select_for_task(&task).await.unwrap();
        assert!(selected.capabilities.has_gpu);
    }
}
