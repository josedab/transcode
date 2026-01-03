//! Persistence layer for distributed transcoding.
//!
//! This module provides database-backed storage for jobs, tasks, and workers,
//! enabling fault tolerance and recovery across coordinator restarts.
//!
//! # Features
//!
//! - PostgreSQL support via SQLx (requires `postgres` feature)
//! - Transactional job and task operations
//! - Worker registry with heartbeat tracking
//! - Efficient queries for pending tasks and job status
//!
//! # Example
//!
//! ```ignore
//! use transcode_distributed::persistence::{PostgresStore, JobStore};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = PostgresStore::connect("postgres://localhost/transcode").await?;
//!     store.migrate().await?;
//!
//!     // Store and retrieve jobs
//!     let job = Job::new("video".into(), "input.mp4".into(), "output.mp4".into(), Default::default());
//!     store.create_job(&job).await?;
//!
//!     Ok(())
//! }
//! ```

use crate::error::{DistributedError, Result};
use crate::task::{Job, Priority, Task, TaskState, TranscodeParams, VideoSegment};
use crate::worker::{WorkerHealth, WorkerInfo, WorkerLoad};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Job storage trait.
#[async_trait]
pub trait JobStore: Send + Sync {
    /// Create a new job.
    async fn create_job(&self, job: &Job) -> Result<()>;

    /// Get a job by ID.
    async fn get_job(&self, id: Uuid) -> Result<Option<Job>>;

    /// Update a job.
    async fn update_job(&self, job: &Job) -> Result<()>;

    /// Delete a job.
    async fn delete_job(&self, id: Uuid) -> Result<()>;

    /// List all jobs.
    async fn list_jobs(&self) -> Result<Vec<Job>>;

    /// List jobs by state.
    async fn list_jobs_by_state(&self, state: TaskState) -> Result<Vec<Job>>;

    /// Count jobs by state.
    async fn count_jobs_by_state(&self, state: TaskState) -> Result<i64>;
}

/// Task storage trait.
#[async_trait]
pub trait TaskStore: Send + Sync {
    /// Create a new task.
    async fn create_task(&self, task: &Task) -> Result<()>;

    /// Get a task by ID.
    async fn get_task(&self, id: Uuid) -> Result<Option<Task>>;

    /// Update a task.
    async fn update_task(&self, task: &Task) -> Result<()>;

    /// Delete a task.
    async fn delete_task(&self, id: Uuid) -> Result<()>;

    /// List tasks for a job.
    async fn list_tasks_for_job(&self, job_id: Uuid) -> Result<Vec<Task>>;

    /// List pending tasks (ready to be assigned).
    async fn list_pending_tasks(&self, limit: i64) -> Result<Vec<Task>>;

    /// List tasks assigned to a worker.
    async fn list_tasks_for_worker(&self, worker_id: &str) -> Result<Vec<Task>>;

    /// Assign task to worker.
    async fn assign_task(&self, task_id: Uuid, worker_id: &str) -> Result<()>;

    /// Update task progress.
    async fn update_task_progress(&self, task_id: Uuid, progress: f64) -> Result<()>;

    /// Mark task as completed.
    async fn complete_task(&self, task_id: Uuid, output: &str) -> Result<()>;

    /// Mark task as failed.
    async fn fail_task(&self, task_id: Uuid, error: &str) -> Result<()>;

    /// Count tasks by state for a job.
    async fn count_tasks_by_state(&self, job_id: Uuid, state: TaskState) -> Result<i64>;
}

/// Worker storage trait.
#[async_trait]
pub trait WorkerStore: Send + Sync {
    /// Register a worker.
    async fn register_worker(&self, worker: &WorkerInfo) -> Result<()>;

    /// Get a worker by ID.
    async fn get_worker(&self, id: &str) -> Result<Option<WorkerInfo>>;

    /// Update worker.
    async fn update_worker(&self, worker: &WorkerInfo) -> Result<()>;

    /// Unregister a worker.
    async fn unregister_worker(&self, id: &str) -> Result<()>;

    /// List all workers.
    async fn list_workers(&self) -> Result<Vec<WorkerInfo>>;

    /// List workers by health status.
    async fn list_workers_by_health(&self, health: WorkerHealth) -> Result<Vec<WorkerInfo>>;

    /// Update worker heartbeat.
    async fn heartbeat(&self, worker_id: &str, load: &WorkerLoad) -> Result<()>;

    /// Get stale workers (no heartbeat within timeout).
    async fn get_stale_workers(&self, timeout_secs: i64) -> Result<Vec<WorkerInfo>>;
}

/// Combined persistence store.
#[async_trait]
pub trait PersistenceStore: JobStore + TaskStore + WorkerStore {
    /// Run database migrations.
    async fn migrate(&self) -> Result<()>;

    /// Check if database is healthy.
    async fn health_check(&self) -> Result<bool>;

    /// Close the connection pool.
    async fn close(&self);
}

// PostgreSQL implementation
#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
    use sqlx::Row;

    /// PostgreSQL-backed persistence store.
    pub struct PostgresStore {
        pool: PgPool,
    }

    impl PostgresStore {
        /// Connect to PostgreSQL database.
        pub async fn connect(database_url: &str) -> Result<Self> {
            let pool = PgPoolOptions::new()
                .max_connections(20)
                .connect(database_url)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(Self { pool })
        }

        /// Connect with custom pool options.
        pub async fn connect_with_options(
            database_url: &str,
            max_connections: u32,
        ) -> Result<Self> {
            let pool = PgPoolOptions::new()
                .max_connections(max_connections)
                .connect(database_url)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(Self { pool })
        }

        /// Get database pool for custom queries.
        pub fn pool(&self) -> &PgPool {
            &self.pool
        }
    }

    #[async_trait]
    impl JobStore for PostgresStore {
        async fn create_job(&self, job: &Job) -> Result<()> {
            let params_json = serde_json::to_value(&job.params)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let task_ids: Vec<String> = job.task_ids.iter().map(|id| id.to_string()).collect();

            sqlx::query(
                r#"
                INSERT INTO jobs (id, name, source, output, params, task_ids, state, created_at, completed_at, progress)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                "#,
            )
            .bind(job.id)
            .bind(&job.name)
            .bind(&job.source)
            .bind(&job.output)
            .bind(&params_json)
            .bind(&task_ids)
            .bind(job.state.to_string())
            .bind(job.created_at)
            .bind(job.completed_at)
            .bind(job.progress)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                if e.to_string().contains("duplicate key") {
                    DistributedError::Duplicate(job.id.to_string())
                } else {
                    DistributedError::Database(e.to_string())
                }
            })?;

            Ok(())
        }

        async fn get_job(&self, id: Uuid) -> Result<Option<Job>> {
            let row = sqlx::query(
                r#"
                SELECT id, name, source, output, params, task_ids, state, created_at, completed_at, progress
                FROM jobs
                WHERE id = $1
                "#,
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            match row {
                Some(row) => Ok(Some(row_to_job(&row)?)),
                None => Ok(None),
            }
        }

        async fn update_job(&self, job: &Job) -> Result<()> {
            let params_json = serde_json::to_value(&job.params)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let task_ids: Vec<String> = job.task_ids.iter().map(|id| id.to_string()).collect();

            let result = sqlx::query(
                r#"
                UPDATE jobs
                SET name = $2, source = $3, output = $4, params = $5, task_ids = $6,
                    state = $7, completed_at = $8, progress = $9
                WHERE id = $1
                "#,
            )
            .bind(job.id)
            .bind(&job.name)
            .bind(&job.source)
            .bind(&job.output)
            .bind(&params_json)
            .bind(&task_ids)
            .bind(job.state.to_string())
            .bind(job.completed_at)
            .bind(job.progress)
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(job.id.to_string()));
            }

            Ok(())
        }

        async fn delete_job(&self, id: Uuid) -> Result<()> {
            sqlx::query("DELETE FROM jobs WHERE id = $1")
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn list_jobs(&self) -> Result<Vec<Job>> {
            let rows = sqlx::query(
                r#"
                SELECT id, name, source, output, params, task_ids, state, created_at, completed_at, progress
                FROM jobs
                ORDER BY created_at DESC
                "#,
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_job).collect()
        }

        async fn list_jobs_by_state(&self, state: TaskState) -> Result<Vec<Job>> {
            let rows = sqlx::query(
                r#"
                SELECT id, name, source, output, params, task_ids, state, created_at, completed_at, progress
                FROM jobs
                WHERE state = $1
                ORDER BY created_at DESC
                "#,
            )
            .bind(state.to_string())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_job).collect()
        }

        async fn count_jobs_by_state(&self, state: TaskState) -> Result<i64> {
            let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM jobs WHERE state = $1")
                .bind(state.to_string())
                .fetch_one(&self.pool)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(row.0)
        }
    }

    #[async_trait]
    impl TaskStore for PostgresStore {
        async fn create_task(&self, task: &Task) -> Result<()> {
            let segment_json = serde_json::to_value(&task.segment)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let params_json = serde_json::to_value(&task.params)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            sqlx::query(
                r#"
                INSERT INTO tasks (
                    id, job_id, segment, params, priority, state, worker_id,
                    created_at, updated_at, started_at, completed_at,
                    retry_count, max_retries, error, progress, output
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                "#,
            )
            .bind(task.id)
            .bind(task.job_id)
            .bind(&segment_json)
            .bind(&params_json)
            .bind(task.priority as i32)
            .bind(task.state.to_string())
            .bind(&task.worker_id)
            .bind(task.created_at)
            .bind(task.updated_at)
            .bind(task.started_at)
            .bind(task.completed_at)
            .bind(task.retry_count as i32)
            .bind(task.max_retries as i32)
            .bind(&task.error)
            .bind(task.progress)
            .bind(&task.output)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                if e.to_string().contains("duplicate key") {
                    DistributedError::Duplicate(task.id.to_string())
                } else {
                    DistributedError::Database(e.to_string())
                }
            })?;

            Ok(())
        }

        async fn get_task(&self, id: Uuid) -> Result<Option<Task>> {
            let row = sqlx::query(
                r#"
                SELECT id, job_id, segment, params, priority, state, worker_id,
                       created_at, updated_at, started_at, completed_at,
                       retry_count, max_retries, error, progress, output
                FROM tasks
                WHERE id = $1
                "#,
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            match row {
                Some(row) => Ok(Some(row_to_task(&row)?)),
                None => Ok(None),
            }
        }

        async fn update_task(&self, task: &Task) -> Result<()> {
            let segment_json = serde_json::to_value(&task.segment)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let params_json = serde_json::to_value(&task.params)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            let result = sqlx::query(
                r#"
                UPDATE tasks
                SET segment = $2, params = $3, priority = $4, state = $5, worker_id = $6,
                    updated_at = $7, started_at = $8, completed_at = $9,
                    retry_count = $10, max_retries = $11, error = $12, progress = $13, output = $14
                WHERE id = $1
                "#,
            )
            .bind(task.id)
            .bind(&segment_json)
            .bind(&params_json)
            .bind(task.priority as i32)
            .bind(task.state.to_string())
            .bind(&task.worker_id)
            .bind(task.updated_at)
            .bind(task.started_at)
            .bind(task.completed_at)
            .bind(task.retry_count as i32)
            .bind(task.max_retries as i32)
            .bind(&task.error)
            .bind(task.progress)
            .bind(&task.output)
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(task.id.to_string()));
            }

            Ok(())
        }

        async fn delete_task(&self, id: Uuid) -> Result<()> {
            sqlx::query("DELETE FROM tasks WHERE id = $1")
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn list_tasks_for_job(&self, job_id: Uuid) -> Result<Vec<Task>> {
            let rows = sqlx::query(
                r#"
                SELECT id, job_id, segment, params, priority, state, worker_id,
                       created_at, updated_at, started_at, completed_at,
                       retry_count, max_retries, error, progress, output
                FROM tasks
                WHERE job_id = $1
                ORDER BY (segment->>'index')::int ASC
                "#,
            )
            .bind(job_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_task).collect()
        }

        async fn list_pending_tasks(&self, limit: i64) -> Result<Vec<Task>> {
            let rows = sqlx::query(
                r#"
                SELECT id, job_id, segment, params, priority, state, worker_id,
                       created_at, updated_at, started_at, completed_at,
                       retry_count, max_retries, error, progress, output
                FROM tasks
                WHERE state = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT $1
                "#,
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_task).collect()
        }

        async fn list_tasks_for_worker(&self, worker_id: &str) -> Result<Vec<Task>> {
            let rows = sqlx::query(
                r#"
                SELECT id, job_id, segment, params, priority, state, worker_id,
                       created_at, updated_at, started_at, completed_at,
                       retry_count, max_retries, error, progress, output
                FROM tasks
                WHERE worker_id = $1 AND state IN ('queued', 'running')
                ORDER BY priority DESC, created_at ASC
                "#,
            )
            .bind(worker_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_task).collect()
        }

        async fn assign_task(&self, task_id: Uuid, worker_id: &str) -> Result<()> {
            let result = sqlx::query(
                r#"
                UPDATE tasks
                SET worker_id = $2, state = 'queued', updated_at = $3
                WHERE id = $1 AND state = 'pending'
                "#,
            )
            .bind(task_id)
            .bind(worker_id)
            .bind(Utc::now())
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(task_id.to_string()));
            }

            Ok(())
        }

        async fn update_task_progress(&self, task_id: Uuid, progress: f64) -> Result<()> {
            sqlx::query(
                r#"
                UPDATE tasks
                SET progress = $2, updated_at = $3
                WHERE id = $1
                "#,
            )
            .bind(task_id)
            .bind(progress)
            .bind(Utc::now())
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn complete_task(&self, task_id: Uuid, output: &str) -> Result<()> {
            let now = Utc::now();
            let result = sqlx::query(
                r#"
                UPDATE tasks
                SET state = 'completed', output = $2, progress = 1.0,
                    completed_at = $3, updated_at = $3
                WHERE id = $1
                "#,
            )
            .bind(task_id)
            .bind(output)
            .bind(now)
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(task_id.to_string()));
            }

            Ok(())
        }

        async fn fail_task(&self, task_id: Uuid, error: &str) -> Result<()> {
            let now = Utc::now();
            let result = sqlx::query(
                r#"
                UPDATE tasks
                SET state = 'failed', error = $2, completed_at = $3, updated_at = $3
                WHERE id = $1
                "#,
            )
            .bind(task_id)
            .bind(error)
            .bind(now)
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(task_id.to_string()));
            }

            Ok(())
        }

        async fn count_tasks_by_state(&self, job_id: Uuid, state: TaskState) -> Result<i64> {
            let row: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM tasks WHERE job_id = $1 AND state = $2",
            )
            .bind(job_id)
            .bind(state.to_string())
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(row.0)
        }
    }

    #[async_trait]
    impl WorkerStore for PostgresStore {
        async fn register_worker(&self, worker: &WorkerInfo) -> Result<()> {
            let capabilities_json = serde_json::to_value(&worker.capabilities)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let load_json = serde_json::to_value(&worker.load)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            sqlx::query(
                r#"
                INSERT INTO workers (id, name, address, capabilities, health, load, registered_at, last_heartbeat)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                ON CONFLICT (id) DO UPDATE
                SET name = $2, address = $3, capabilities = $4, health = $5, load = $6, last_heartbeat = $7
                "#,
            )
            .bind(&worker.id)
            .bind(&worker.name)
            .bind(&worker.address)
            .bind(&capabilities_json)
            .bind(worker.health.to_string())
            .bind(&load_json)
            .bind(Utc::now())
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn get_worker(&self, id: &str) -> Result<Option<WorkerInfo>> {
            let row = sqlx::query(
                r#"
                SELECT id, name, address, capabilities, health, load, registered_at, last_heartbeat
                FROM workers
                WHERE id = $1
                "#,
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            match row {
                Some(row) => Ok(Some(row_to_worker(&row)?)),
                None => Ok(None),
            }
        }

        async fn update_worker(&self, worker: &WorkerInfo) -> Result<()> {
            let capabilities_json = serde_json::to_value(&worker.capabilities)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;
            let load_json = serde_json::to_value(&worker.load)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            let result = sqlx::query(
                r#"
                UPDATE workers
                SET name = $2, address = $3, capabilities = $4, health = $5, load = $6, last_heartbeat = $7
                WHERE id = $1
                "#,
            )
            .bind(&worker.id)
            .bind(&worker.name)
            .bind(&worker.address)
            .bind(&capabilities_json)
            .bind(worker.health.to_string())
            .bind(&load_json)
            .bind(Utc::now())
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(worker.id.clone()));
            }

            Ok(())
        }

        async fn unregister_worker(&self, id: &str) -> Result<()> {
            sqlx::query("DELETE FROM workers WHERE id = $1")
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn list_workers(&self) -> Result<Vec<WorkerInfo>> {
            let rows = sqlx::query(
                r#"
                SELECT id, name, address, capabilities, health, load, registered_at, last_heartbeat
                FROM workers
                ORDER BY registered_at DESC
                "#,
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_worker).collect()
        }

        async fn list_workers_by_health(&self, health: WorkerHealth) -> Result<Vec<WorkerInfo>> {
            let rows = sqlx::query(
                r#"
                SELECT id, name, address, capabilities, health, load, registered_at, last_heartbeat
                FROM workers
                WHERE health = $1
                ORDER BY registered_at DESC
                "#,
            )
            .bind(health.to_string())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_worker).collect()
        }

        async fn heartbeat(&self, worker_id: &str, load: &WorkerLoad) -> Result<()> {
            let load_json = serde_json::to_value(load)
                .map_err(|e| DistributedError::Serialization(e.to_string()))?;

            let result = sqlx::query(
                r#"
                UPDATE workers
                SET load = $2, last_heartbeat = $3, health = 'healthy'
                WHERE id = $1
                "#,
            )
            .bind(worker_id)
            .bind(&load_json)
            .bind(Utc::now())
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            if result.rows_affected() == 0 {
                return Err(DistributedError::NotFound(worker_id.to_string()));
            }

            Ok(())
        }

        async fn get_stale_workers(&self, timeout_secs: i64) -> Result<Vec<WorkerInfo>> {
            let cutoff = Utc::now() - chrono::Duration::seconds(timeout_secs);

            let rows = sqlx::query(
                r#"
                SELECT id, name, address, capabilities, health, load, registered_at, last_heartbeat
                FROM workers
                WHERE last_heartbeat < $1 AND health != 'offline'
                "#,
            )
            .bind(cutoff)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            rows.iter().map(row_to_worker).collect()
        }
    }

    #[async_trait]
    impl PersistenceStore for PostgresStore {
        async fn migrate(&self) -> Result<()> {
            // Create tables if they don't exist
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS jobs (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    source TEXT NOT NULL,
                    output TEXT NOT NULL,
                    params JSONB NOT NULL,
                    task_ids TEXT[] NOT NULL DEFAULT '{}',
                    state VARCHAR(32) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    progress DOUBLE PRECISION NOT NULL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
                "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS tasks (
                    id UUID PRIMARY KEY,
                    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    segment JSONB NOT NULL,
                    params JSONB NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 50,
                    state VARCHAR(32) NOT NULL DEFAULT 'pending',
                    worker_id VARCHAR(255),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    error TEXT,
                    progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    output TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
                CREATE INDEX IF NOT EXISTS idx_tasks_worker_id ON tasks(worker_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority_created ON tasks(priority DESC, created_at ASC);
                "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS workers (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    address VARCHAR(255) NOT NULL,
                    capabilities JSONB NOT NULL,
                    health VARCHAR(32) NOT NULL DEFAULT 'healthy',
                    load JSONB NOT NULL,
                    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_workers_health ON workers(health);
                CREATE INDEX IF NOT EXISTS idx_workers_last_heartbeat ON workers(last_heartbeat);
                "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(())
        }

        async fn health_check(&self) -> Result<bool> {
            let result: (i32,) = sqlx::query_as("SELECT 1")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| DistributedError::Database(e.to_string()))?;

            Ok(result.0 == 1)
        }

        async fn close(&self) {
            self.pool.close().await;
        }
    }

    // Helper functions to convert rows to types
    fn row_to_job(row: &PgRow) -> Result<Job> {
        let id: Uuid = row.get("id");
        let name: String = row.get("name");
        let source: String = row.get("source");
        let output: String = row.get("output");
        let params_json: serde_json::Value = row.get("params");
        let task_ids_str: Vec<String> = row.get("task_ids");
        let state_str: String = row.get("state");
        let created_at: DateTime<Utc> = row.get("created_at");
        let completed_at: Option<DateTime<Utc>> = row.get("completed_at");
        let progress: f64 = row.get("progress");

        let params: TranscodeParams = serde_json::from_value(params_json)
            .map_err(|e| DistributedError::Serialization(e.to_string()))?;

        let task_ids: Vec<Uuid> = task_ids_str
            .iter()
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        let state = parse_task_state(&state_str);

        Ok(Job {
            id,
            name,
            source,
            output,
            params,
            task_ids,
            state,
            created_at,
            completed_at,
            progress,
        })
    }

    fn row_to_task(row: &PgRow) -> Result<Task> {
        let id: Uuid = row.get("id");
        let job_id: Uuid = row.get("job_id");
        let segment_json: serde_json::Value = row.get("segment");
        let params_json: serde_json::Value = row.get("params");
        let priority_int: i32 = row.get("priority");
        let state_str: String = row.get("state");
        let worker_id: Option<String> = row.get("worker_id");
        let created_at: DateTime<Utc> = row.get("created_at");
        let updated_at: DateTime<Utc> = row.get("updated_at");
        let started_at: Option<DateTime<Utc>> = row.get("started_at");
        let completed_at: Option<DateTime<Utc>> = row.get("completed_at");
        let retry_count: i32 = row.get("retry_count");
        let max_retries: i32 = row.get("max_retries");
        let error: Option<String> = row.get("error");
        let progress: f64 = row.get("progress");
        let output: Option<String> = row.get("output");

        let segment: VideoSegment = serde_json::from_value(segment_json)
            .map_err(|e| DistributedError::Serialization(e.to_string()))?;
        let params: TranscodeParams = serde_json::from_value(params_json)
            .map_err(|e| DistributedError::Serialization(e.to_string()))?;

        let priority = match priority_int {
            0 => Priority::Low,
            75 => Priority::High,
            100 => Priority::Critical,
            _ => Priority::Normal,
        };

        let state = parse_task_state(&state_str);

        Ok(Task {
            id,
            job_id,
            segment,
            params,
            priority,
            state,
            worker_id,
            created_at,
            updated_at,
            started_at,
            completed_at,
            retry_count: retry_count as u32,
            max_retries: max_retries as u32,
            error,
            progress,
            output,
        })
    }

    fn row_to_worker(row: &PgRow) -> Result<WorkerInfo> {
        use crate::worker::WorkerCapabilities;
        use std::collections::HashSet;

        let id: String = row.get("id");
        let name: String = row.get("name");
        let address: String = row.get("address");
        let capabilities_json: serde_json::Value = row.get("capabilities");
        let health_str: String = row.get("health");
        let load_json: serde_json::Value = row.get("load");
        let registered_at: DateTime<Utc> = row.get("registered_at");
        let last_heartbeat: DateTime<Utc> = row.get("last_heartbeat");

        let capabilities: WorkerCapabilities = serde_json::from_value(capabilities_json)
            .map_err(|e| DistributedError::Serialization(e.to_string()))?;
        let load: WorkerLoad = serde_json::from_value(load_json)
            .map_err(|e| DistributedError::Serialization(e.to_string()))?;

        let health = match health_str.as_str() {
            "healthy" => WorkerHealth::Healthy,
            "degraded" => WorkerHealth::Degraded,
            "unhealthy" => WorkerHealth::Unhealthy,
            "offline" => WorkerHealth::Offline,
            _ => WorkerHealth::Healthy,
        };

        Ok(WorkerInfo {
            id,
            name,
            address,
            capabilities,
            health,
            load,
            registered_at,
            last_heartbeat,
            active_tasks: HashSet::new(), // Runtime state, not persisted
        })
    }

    fn parse_task_state(s: &str) -> TaskState {
        match s {
            "pending" => TaskState::Pending,
            "queued" => TaskState::Queued,
            "running" => TaskState::Running,
            "completed" => TaskState::Completed,
            "failed" => TaskState::Failed,
            "cancelled" => TaskState::Cancelled,
            "retrying" => TaskState::Retrying,
            _ => TaskState::Pending,
        }
    }
}

// Re-export PostgresStore when the feature is enabled
#[cfg(feature = "postgres")]
pub use postgres::PostgresStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state_display() {
        assert_eq!(TaskState::Pending.to_string(), "pending");
        assert_eq!(TaskState::Completed.to_string(), "completed");
    }
}
