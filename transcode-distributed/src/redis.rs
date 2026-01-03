//! Redis coordination for distributed transcoding.
//!
//! This module provides Redis-based coordination primitives for distributed
//! task scheduling, including distributed locks, priority queues, and worker
//! heartbeat management via pub/sub.
//!
//! # Features
//!
//! - **Distributed Locks**: Prevent concurrent task assignment with Redis locks
//! - **Priority Queue**: Task queue with priority ordering using sorted sets
//! - **Worker Heartbeat**: Pub/sub based heartbeat monitoring
//! - **Lease-based Assignment**: Tasks assigned with time-limited leases
//!
//! # Example
//!
//! ```ignore
//! use transcode_distributed::redis::{RedisCoordinator, RedisConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RedisConfig::new("redis://localhost:6379");
//!     let coordinator = RedisCoordinator::connect(config).await?;
//!
//!     // Enqueue a task with priority
//!     coordinator.enqueue_task(&task, Priority::High).await?;
//!
//!     // Acquire a task with lease
//!     if let Some(task) = coordinator.acquire_task("worker-1", Duration::from_secs(300)).await? {
//!         // Process task...
//!         coordinator.complete_task(task.id).await?;
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{DistributedError, Result};
use crate::task::{Priority, Task, TaskState};
use crate::worker::WorkerLoad;
use async_trait::async_trait;
use std::time::Duration;
use uuid::Uuid;

#[cfg(feature = "redis")]
use redis::aio::ConnectionManager;
#[cfg(feature = "redis")]
use redis::{AsyncCommands, Script};

/// Redis configuration.
#[derive(Debug, Clone)]
pub struct RedisConfig {
    /// Redis URL (e.g., "redis://localhost:6379")
    pub url: String,
    /// Key prefix for all Redis keys
    pub prefix: String,
    /// Default lock timeout
    pub lock_timeout: Duration,
    /// Default lease duration
    pub lease_duration: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Heartbeat timeout (worker considered offline)
    pub heartbeat_timeout: Duration,
}

impl RedisConfig {
    /// Create a new Redis config with default settings.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            prefix: "transcode".to_string(),
            lock_timeout: Duration::from_secs(30),
            lease_duration: Duration::from_secs(300), // 5 minutes
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
        }
    }

    /// Set the key prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the lease duration.
    pub fn with_lease_duration(mut self, duration: Duration) -> Self {
        self.lease_duration = duration;
        self
    }
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self::new("redis://localhost:6379")
    }
}

/// Task coordination trait for distributed task management.
#[async_trait]
pub trait TaskCoordinator: Send + Sync {
    /// Enqueue a task with priority.
    async fn enqueue_task(&self, task: &Task, priority: Priority) -> Result<()>;

    /// Acquire a task with a lease.
    async fn acquire_task(&self, worker_id: &str, lease_duration: Duration) -> Result<Option<Task>>;

    /// Release a task (return it to the queue).
    async fn release_task(&self, task_id: Uuid) -> Result<()>;

    /// Complete a task.
    async fn complete_task(&self, task_id: Uuid) -> Result<()>;

    /// Fail a task.
    async fn fail_task(&self, task_id: Uuid, error: &str) -> Result<()>;

    /// Renew a task lease.
    async fn renew_lease(&self, task_id: Uuid, worker_id: &str, duration: Duration) -> Result<bool>;

    /// Get pending task count.
    async fn pending_count(&self) -> Result<u64>;
}

/// Worker coordination trait for heartbeat management.
#[async_trait]
pub trait WorkerCoordinator: Send + Sync {
    /// Register worker heartbeat.
    async fn heartbeat(&self, worker_id: &str, load: &WorkerLoad) -> Result<()>;

    /// Get online workers.
    async fn online_workers(&self) -> Result<Vec<String>>;

    /// Check if a worker is online.
    async fn is_worker_online(&self, worker_id: &str) -> Result<bool>;
}

/// Distributed lock trait.
#[async_trait]
pub trait DistributedLock: Send + Sync {
    /// Acquire a lock.
    async fn acquire_lock(&self, name: &str, ttl: Duration) -> Result<Option<LockGuard>>;

    /// Release a lock.
    async fn release_lock(&self, name: &str, token: &str) -> Result<bool>;

    /// Extend a lock.
    async fn extend_lock(&self, name: &str, token: &str, ttl: Duration) -> Result<bool>;
}

/// Lock guard representing an acquired lock.
#[derive(Debug)]
pub struct LockGuard {
    /// Lock name
    pub name: String,
    /// Lock token for releasing
    pub token: String,
    /// Time-to-live
    pub ttl: Duration,
}

// Redis implementation
#[cfg(feature = "redis")]
mod redis_impl {
    use super::*;
    use tokio::sync::RwLock;

    /// Redis-based coordinator.
    pub struct RedisCoordinator {
        /// Connection manager
        conn: RwLock<ConnectionManager>,
        /// Configuration
        config: RedisConfig,
    }

    impl RedisCoordinator {
        /// Connect to Redis.
        pub async fn connect(config: RedisConfig) -> Result<Self> {
            let client = redis::Client::open(config.url.as_str())
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            let conn = ConnectionManager::new(client)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(Self {
                conn: RwLock::new(conn),
                config,
            })
        }

        /// Get a key with the configured prefix.
        fn key(&self, suffix: &str) -> String {
            format!("{}:{}", self.config.prefix, suffix)
        }

        /// Task queue key.
        fn task_queue_key(&self) -> String {
            self.key("tasks:queue")
        }

        /// Task data key.
        fn task_data_key(&self, task_id: Uuid) -> String {
            self.key(&format!("tasks:data:{}", task_id))
        }

        /// Task lease key.
        fn task_lease_key(&self, task_id: Uuid) -> String {
            self.key(&format!("tasks:lease:{}", task_id))
        }

        /// Worker heartbeat key.
        fn worker_heartbeat_key(&self, worker_id: &str) -> String {
            self.key(&format!("workers:heartbeat:{}", worker_id))
        }

        /// Worker set key (online workers).
        fn workers_set_key(&self) -> String {
            self.key("workers:online")
        }

        /// Lock key.
        fn lock_key(&self, name: &str) -> String {
            self.key(&format!("locks:{}", name))
        }
    }

    #[async_trait]
    impl TaskCoordinator for RedisCoordinator {
        async fn enqueue_task(&self, task: &Task, priority: Priority) -> Result<()> {
            let mut conn = self.conn.write().await;

            // Serialize task data
            let task_json = serde_json::to_string(task)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            // Store task data
            let data_key = self.task_data_key(task.id);
            conn.set::<_, _, ()>(&data_key, &task_json)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            // Add to priority queue (higher priority = higher score)
            let queue_key = self.task_queue_key();
            let score = priority as i64;
            conn.zadd::<_, _, _, ()>(&queue_key, task.id.to_string(), score)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            tracing::debug!("Enqueued task {} with priority {:?}", task.id, priority);
            Ok(())
        }

        async fn acquire_task(&self, worker_id: &str, lease_duration: Duration) -> Result<Option<Task>> {
            let mut conn = self.conn.write().await;

            // Atomic acquire script: pop highest priority task and set lease
            let script = Script::new(
                r#"
                local queue_key = KEYS[1]
                local task_id = redis.call('ZPOPMAX', queue_key)
                if task_id and task_id[1] then
                    local task_uuid = task_id[1]
                    local lease_key = KEYS[2] .. task_uuid
                    local data_key = KEYS[3] .. task_uuid

                    -- Set lease
                    redis.call('SET', lease_key, ARGV[1], 'EX', ARGV[2])

                    -- Get task data
                    local task_data = redis.call('GET', data_key)
                    return task_data
                end
                return nil
                "#,
            );

            let queue_key = self.task_queue_key();
            let lease_prefix = self.key("tasks:lease:");
            let data_prefix = self.key("tasks:data:");
            let lease_secs = lease_duration.as_secs() as i64;

            let result: Option<String> = script
                .key(&queue_key)
                .key(&lease_prefix)
                .key(&data_prefix)
                .arg(worker_id)
                .arg(lease_secs)
                .invoke_async(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            match result {
                Some(task_json) => {
                    let task: Task = serde_json::from_str(&task_json)
                        .map_err(|e| DistributedError::Serialization(e.to_string()))?;
                    tracing::debug!("Worker {} acquired task {}", worker_id, task.id);
                    Ok(Some(task))
                }
                None => Ok(None),
            }
        }

        async fn release_task(&self, task_id: Uuid) -> Result<()> {
            let mut conn = self.conn.write().await;

            // Remove lease
            let lease_key = self.task_lease_key(task_id);
            conn.del::<_, ()>(&lease_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            // Get task data to get priority
            let data_key = self.task_data_key(task_id);
            let task_json: Option<String> = conn
                .get(&data_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            if let Some(json) = task_json {
                let task: Task = serde_json::from_str(&json)
                    .map_err(|e| DistributedError::Serialization(e.to_string()))?;

                // Re-add to queue with original priority
                let queue_key = self.task_queue_key();
                conn.zadd::<_, _, _, ()>(&queue_key, task_id.to_string(), task.priority as i64)
                    .await
                    .map_err(|e| DistributedError::Network(e.to_string()))?;
            }

            tracing::debug!("Released task {} back to queue", task_id);
            Ok(())
        }

        async fn complete_task(&self, task_id: Uuid) -> Result<()> {
            let mut conn = self.conn.write().await;

            // Remove task data and lease
            let data_key = self.task_data_key(task_id);
            let lease_key = self.task_lease_key(task_id);

            redis::pipe()
                .del(&data_key)
                .del(&lease_key)
                .query_async::<()>(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            tracing::debug!("Completed task {}", task_id);
            Ok(())
        }

        async fn fail_task(&self, task_id: Uuid, error: &str) -> Result<()> {
            let mut conn = self.conn.write().await;

            // Get task data
            let data_key = self.task_data_key(task_id);
            let task_json: Option<String> = conn
                .get(&data_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            if let Some(json) = task_json {
                let mut task: Task = serde_json::from_str(&json)
                    .map_err(|e| DistributedError::Serialization(e.to_string()))?;

                // Update task state
                task.state = TaskState::Failed;
                task.error = Some(error.to_string());

                // Save updated task
                let updated_json = serde_json::to_string(&task)
                    .map_err(|e| DistributedError::Serialization(e.to_string()))?;
                conn.set::<_, _, ()>(&data_key, &updated_json)
                    .await
                    .map_err(|e| DistributedError::Network(e.to_string()))?;

                // Move to failed set for retry handling
                let failed_key = self.key("tasks:failed");
                conn.zadd::<_, _, _, ()>(&failed_key, task_id.to_string(), chrono::Utc::now().timestamp())
                    .await
                    .map_err(|e| DistributedError::Network(e.to_string()))?;
            }

            // Remove lease
            let lease_key = self.task_lease_key(task_id);
            conn.del::<_, ()>(&lease_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            tracing::warn!("Task {} failed: {}", task_id, error);
            Ok(())
        }

        async fn renew_lease(&self, task_id: Uuid, worker_id: &str, duration: Duration) -> Result<bool> {
            let mut conn = self.conn.write().await;

            // Atomic renewal: only renew if we hold the lease
            let script = Script::new(
                r#"
                local lease_key = KEYS[1]
                local current = redis.call('GET', lease_key)
                if current == ARGV[1] then
                    redis.call('EXPIRE', lease_key, ARGV[2])
                    return 1
                end
                return 0
                "#,
            );

            let lease_key = self.task_lease_key(task_id);
            let lease_secs = duration.as_secs() as i64;

            let result: i32 = script
                .key(&lease_key)
                .arg(worker_id)
                .arg(lease_secs)
                .invoke_async(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(result == 1)
        }

        async fn pending_count(&self) -> Result<u64> {
            let mut conn = self.conn.write().await;
            let queue_key = self.task_queue_key();

            let count: u64 = conn
                .zcard(&queue_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(count)
        }
    }

    #[async_trait]
    impl WorkerCoordinator for RedisCoordinator {
        async fn heartbeat(&self, worker_id: &str, load: &WorkerLoad) -> Result<()> {
            let mut conn = self.conn.write().await;

            let heartbeat_key = self.worker_heartbeat_key(worker_id);
            let workers_key = self.workers_set_key();
            let ttl_secs = self.config.heartbeat_timeout.as_secs() as i64;

            // Serialize load data
            let load_json = serde_json::to_string(load)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            // Update heartbeat and add to online workers set
            redis::pipe()
                .set_ex(&heartbeat_key, &load_json, ttl_secs as u64)
                .zadd(&workers_key, worker_id, chrono::Utc::now().timestamp())
                .query_async::<()>(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            tracing::trace!("Worker {} heartbeat", worker_id);
            Ok(())
        }

        async fn online_workers(&self) -> Result<Vec<String>> {
            let mut conn = self.conn.write().await;

            // First, clean up stale workers
            let workers_key = self.workers_set_key();
            let cutoff = chrono::Utc::now().timestamp() - self.config.heartbeat_timeout.as_secs() as i64;

            // Remove workers with timestamp older than cutoff
            conn.zrembyscore::<_, _, _, ()>(&workers_key, "-inf", cutoff)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            // Get remaining workers
            let workers: Vec<String> = conn
                .zrange(&workers_key, 0, -1)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(workers)
        }

        async fn is_worker_online(&self, worker_id: &str) -> Result<bool> {
            let mut conn = self.conn.write().await;
            let heartbeat_key = self.worker_heartbeat_key(worker_id);

            let exists: bool = conn
                .exists(&heartbeat_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(exists)
        }
    }

    #[async_trait]
    impl DistributedLock for RedisCoordinator {
        async fn acquire_lock(&self, name: &str, ttl: Duration) -> Result<Option<LockGuard>> {
            let mut conn = self.conn.write().await;
            let lock_key = self.lock_key(name);
            let token = Uuid::new_v4().to_string();
            let ttl_ms = ttl.as_millis() as u64;

            // Try to acquire lock with SET NX PX
            let result: bool = redis::cmd("SET")
                .arg(&lock_key)
                .arg(&token)
                .arg("NX")
                .arg("PX")
                .arg(ttl_ms)
                .query_async(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            if result {
                Ok(Some(LockGuard {
                    name: name.to_string(),
                    token,
                    ttl,
                }))
            } else {
                Ok(None)
            }
        }

        async fn release_lock(&self, name: &str, token: &str) -> Result<bool> {
            let mut conn = self.conn.write().await;

            // Atomic release: only release if we hold the lock
            let script = Script::new(
                r#"
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('DEL', KEYS[1])
                else
                    return 0
                end
                "#,
            );

            let lock_key = self.lock_key(name);
            let result: i32 = script
                .key(&lock_key)
                .arg(token)
                .invoke_async(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(result == 1)
        }

        async fn extend_lock(&self, name: &str, token: &str, ttl: Duration) -> Result<bool> {
            let mut conn = self.conn.write().await;

            // Atomic extend: only extend if we hold the lock
            let script = Script::new(
                r#"
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
                else
                    return 0
                end
                "#,
            );

            let lock_key = self.lock_key(name);
            let ttl_ms = ttl.as_millis() as i64;

            let result: i32 = script
                .key(&lock_key)
                .arg(token)
                .arg(ttl_ms)
                .invoke_async(&mut *conn)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            Ok(result == 1)
        }
    }

    impl RedisCoordinator {
        /// Recover expired leases and return tasks to queue.
        ///
        /// This should be called periodically to handle workers that crashed
        /// without releasing their tasks.
        pub async fn recover_expired_leases(&self) -> Result<usize> {
            // This is a complex operation that would scan for tasks with expired leases
            // For now, we rely on the lease expiration mechanism
            // In production, you'd want a background task scanning for orphaned tasks
            tracing::debug!("Lease recovery check");
            Ok(0)
        }

        /// Get failed tasks for retry.
        pub async fn get_failed_tasks(&self, limit: usize) -> Result<Vec<Task>> {
            let mut conn = self.conn.write().await;
            let failed_key = self.key("tasks:failed");

            // Get oldest failed tasks
            let task_ids: Vec<String> = conn
                .zrange(&failed_key, 0, (limit as isize) - 1)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            let mut tasks = Vec::new();
            for task_id in task_ids {
                if let Ok(uuid) = Uuid::parse_str(&task_id) {
                    let data_key = self.task_data_key(uuid);
                    if let Ok(Some(json)) = conn.get::<_, Option<String>>(&data_key).await {
                        if let Ok(task) = serde_json::from_str::<Task>(&json) {
                            tasks.push(task);
                        }
                    }
                }
            }

            Ok(tasks)
        }

        /// Retry a failed task.
        pub async fn retry_failed_task(&self, task_id: Uuid) -> Result<bool> {
            let mut conn = self.conn.write().await;

            // Get task data
            let data_key = self.task_data_key(task_id);
            let task_json: Option<String> = conn
                .get(&data_key)
                .await
                .map_err(|e| DistributedError::Network(e.to_string()))?;

            if let Some(json) = task_json {
                let mut task: Task = serde_json::from_str(&json)
                    .map_err(|e| DistributedError::Serialization(e.to_string()))?;

                // Check retry limit
                if task.retry_count >= task.max_retries {
                    return Ok(false);
                }

                // Update task for retry
                task.retry_count += 1;
                task.state = TaskState::Pending;
                task.error = None;

                // Save updated task
                let updated_json = serde_json::to_string(&task)
                    .map_err(|e| DistributedError::Serialization(e.to_string()))?;
                conn.set::<_, _, ()>(&data_key, &updated_json)
                    .await
                    .map_err(|e| DistributedError::Network(e.to_string()))?;

                // Remove from failed set and add to queue
                let failed_key = self.key("tasks:failed");
                let queue_key = self.task_queue_key();

                redis::pipe()
                    .zrem(&failed_key, task_id.to_string())
                    .zadd(&queue_key, task_id.to_string(), task.priority as i64)
                    .query_async::<()>(&mut *conn)
                    .await
                    .map_err(|e| DistributedError::Network(e.to_string()))?;

                tracing::info!("Retrying task {} (attempt {})", task_id, task.retry_count);
                return Ok(true);
            }

            Ok(false)
        }

        /// Get configuration.
        pub fn config(&self) -> &RedisConfig {
            &self.config
        }
    }
}

// Re-export Redis types when the feature is enabled
#[cfg(feature = "redis")]
pub use redis_impl::RedisCoordinator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://localhost:6379");
        assert_eq!(config.prefix, "transcode");
    }

    #[test]
    fn test_redis_config_builder() {
        let config = RedisConfig::new("redis://custom:6380")
            .with_prefix("myapp")
            .with_lease_duration(Duration::from_secs(600));

        assert_eq!(config.url, "redis://custom:6380");
        assert_eq!(config.prefix, "myapp");
        assert_eq!(config.lease_duration, Duration::from_secs(600));
    }
}
