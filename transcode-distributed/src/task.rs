//! Distributed task types and management.

use crate::error::{DistributedError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Low priority (background jobs).
    Low = 0,
    /// Normal priority (default).
    #[default]
    Normal = 50,
    /// High priority (user-initiated).
    High = 75,
    /// Critical priority (time-sensitive).
    Critical = 100,
}

/// Task state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskState {
    /// Task is waiting to be picked up.
    Pending,
    /// Task is queued for a specific worker.
    Queued,
    /// Task is currently being processed.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
    /// Task is being retried.
    Retrying,
}

impl std::fmt::Display for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Retrying => write!(f, "retrying"),
        }
    }
}

impl TaskState {
    /// Check if this state can transition to another state.
    pub fn can_transition_to(&self, next: TaskState) -> bool {
        match (self, next) {
            // From Pending
            (Self::Pending, Self::Queued) => true,
            (Self::Pending, Self::Cancelled) => true,
            // From Queued
            (Self::Queued, Self::Running) => true,
            (Self::Queued, Self::Cancelled) => true,
            (Self::Queued, Self::Pending) => true, // Re-queue
            // From Running
            (Self::Running, Self::Completed) => true,
            (Self::Running, Self::Failed) => true,
            (Self::Running, Self::Cancelled) => true,
            (Self::Running, Self::Retrying) => true,
            // From Failed
            (Self::Failed, Self::Retrying) => true,
            (Self::Failed, Self::Pending) => true, // Manual retry
            // From Retrying
            (Self::Retrying, Self::Pending) => true,
            (Self::Retrying, Self::Cancelled) => true,
            // No other transitions allowed
            _ => false,
        }
    }

    /// Check if this is a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Video segment to process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSegment {
    /// Segment identifier.
    pub id: String,
    /// Source file path or URL.
    pub source: String,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Segment index (for ordering).
    pub index: usize,
    /// Total number of segments.
    pub total_segments: usize,
}

impl VideoSegment {
    /// Create a new video segment.
    pub fn new(
        source: String,
        start_time: f64,
        end_time: f64,
        index: usize,
        total_segments: usize,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source,
            start_time,
            end_time,
            index,
            total_segments,
        }
    }

    /// Get segment duration.
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Transcoding parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeParams {
    /// Output codec (e.g., "h264", "av1", "hevc").
    pub codec: String,
    /// Output container format (e.g., "mp4", "webm").
    pub container: String,
    /// Output resolution (width, height) or None for same as source.
    pub resolution: Option<(u32, u32)>,
    /// Output bitrate in bits per second, or None for auto.
    pub bitrate: Option<u64>,
    /// Frame rate or None for same as source.
    pub frame_rate: Option<f64>,
    /// Additional encoder options.
    pub options: HashMap<String, String>,
}

impl Default for TranscodeParams {
    fn default() -> Self {
        Self {
            codec: "h264".to_string(),
            container: "mp4".to_string(),
            resolution: None,
            bitrate: None,
            frame_rate: None,
            options: HashMap::new(),
        }
    }
}

/// A transcoding task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    pub id: Uuid,
    /// Parent job identifier.
    pub job_id: Uuid,
    /// Video segment to process.
    pub segment: VideoSegment,
    /// Transcoding parameters.
    pub params: TranscodeParams,
    /// Task priority.
    pub priority: Priority,
    /// Current state.
    pub state: TaskState,
    /// Worker assigned to this task (if any).
    pub worker_id: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Start processing timestamp.
    pub started_at: Option<DateTime<Utc>>,
    /// Completion timestamp.
    pub completed_at: Option<DateTime<Utc>>,
    /// Number of retry attempts.
    pub retry_count: u32,
    /// Maximum retries allowed.
    pub max_retries: u32,
    /// Error message if failed.
    pub error: Option<String>,
    /// Progress (0.0 - 1.0).
    pub progress: f64,
    /// Output file path (after completion).
    pub output: Option<String>,
}

impl Task {
    /// Create a new task.
    pub fn new(job_id: Uuid, segment: VideoSegment, params: TranscodeParams) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            job_id,
            segment,
            params,
            priority: Priority::default(),
            state: TaskState::Pending,
            worker_id: None,
            created_at: now,
            updated_at: now,
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            error: None,
            progress: 0.0,
            output: None,
        }
    }

    /// Set task priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set maximum retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Transition to a new state.
    pub fn transition_to(&mut self, new_state: TaskState) -> Result<()> {
        if !self.state.can_transition_to(new_state) {
            return Err(DistributedError::InvalidStateTransition {
                from: self.state.to_string(),
                to: new_state.to_string(),
            });
        }

        self.state = new_state;
        self.updated_at = Utc::now();

        match new_state {
            TaskState::Running => {
                self.started_at = Some(Utc::now());
            }
            TaskState::Completed | TaskState::Failed | TaskState::Cancelled => {
                self.completed_at = Some(Utc::now());
            }
            TaskState::Retrying => {
                self.retry_count += 1;
                self.error = None;
            }
            _ => {}
        }

        Ok(())
    }

    /// Mark task as queued for a worker.
    pub fn queue_for(&mut self, worker_id: String) -> Result<()> {
        self.transition_to(TaskState::Queued)?;
        self.worker_id = Some(worker_id);
        Ok(())
    }

    /// Mark task as started.
    pub fn start(&mut self) -> Result<()> {
        self.transition_to(TaskState::Running)
    }

    /// Mark task as completed.
    pub fn complete(&mut self, output: String) -> Result<()> {
        self.transition_to(TaskState::Completed)?;
        self.output = Some(output);
        self.progress = 1.0;
        Ok(())
    }

    /// Mark task as failed.
    pub fn fail(&mut self, error: String) -> Result<()> {
        self.transition_to(TaskState::Failed)?;
        self.error = Some(error);
        Ok(())
    }

    /// Cancel the task.
    pub fn cancel(&mut self) -> Result<()> {
        self.transition_to(TaskState::Cancelled)
    }

    /// Retry the task.
    pub fn retry(&mut self) -> Result<bool> {
        if self.retry_count >= self.max_retries {
            return Ok(false);
        }

        self.transition_to(TaskState::Retrying)?;
        self.transition_to(TaskState::Pending)?;
        self.worker_id = None;
        self.progress = 0.0;

        Ok(true)
    }

    /// Update progress.
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Check if task can be retried.
    pub fn can_retry(&self) -> bool {
        self.state == TaskState::Failed && self.retry_count < self.max_retries
    }

    /// Get elapsed processing time.
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|start| {
            let end = self.completed_at.unwrap_or_else(Utc::now);
            (end - start).to_std().unwrap_or_default()
        })
    }
}

/// A transcoding job (collection of tasks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique job identifier.
    pub id: Uuid,
    /// Job name/description.
    pub name: String,
    /// Source file path or URL.
    pub source: String,
    /// Output destination.
    pub output: String,
    /// Transcoding parameters.
    pub params: TranscodeParams,
    /// Task IDs belonging to this job.
    pub task_ids: Vec<Uuid>,
    /// Job state.
    pub state: TaskState,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Completion timestamp.
    pub completed_at: Option<DateTime<Utc>>,
    /// Overall progress (0.0 - 1.0).
    pub progress: f64,
}

impl Job {
    /// Create a new job.
    pub fn new(name: String, source: String, output: String, params: TranscodeParams) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            source,
            output,
            params,
            task_ids: Vec::new(),
            state: TaskState::Pending,
            created_at: Utc::now(),
            completed_at: None,
            progress: 0.0,
        }
    }

    /// Add a task ID to this job.
    pub fn add_task(&mut self, task_id: Uuid) {
        self.task_ids.push(task_id);
    }

    /// Update job progress based on task states.
    pub fn update_progress(&mut self, tasks: &[Task]) {
        if tasks.is_empty() {
            return;
        }

        let total_progress: f64 = tasks.iter().map(|t| t.progress).sum();
        self.progress = total_progress / tasks.len() as f64;

        // Determine overall state
        let all_completed = tasks.iter().all(|t| t.state == TaskState::Completed);
        let any_failed = tasks.iter().any(|t| t.state == TaskState::Failed);
        let any_running = tasks.iter().any(|t| t.state == TaskState::Running);
        let all_cancelled = tasks
            .iter()
            .all(|t| t.state == TaskState::Cancelled);

        if all_completed {
            self.state = TaskState::Completed;
            self.completed_at = Some(Utc::now());
        } else if all_cancelled {
            self.state = TaskState::Cancelled;
            self.completed_at = Some(Utc::now());
        } else if any_failed && !any_running {
            self.state = TaskState::Failed;
        } else if any_running {
            self.state = TaskState::Running;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state_transitions() {
        let mut task = Task::new(
            Uuid::new_v4(),
            VideoSegment::new("test.mp4".into(), 0.0, 10.0, 0, 1),
            TranscodeParams::default(),
        );

        assert_eq!(task.state, TaskState::Pending);

        task.queue_for("worker-1".into()).unwrap();
        assert_eq!(task.state, TaskState::Queued);

        task.start().unwrap();
        assert_eq!(task.state, TaskState::Running);

        task.complete("/output/test_0.mp4".into()).unwrap();
        assert_eq!(task.state, TaskState::Completed);
    }

    #[test]
    fn test_task_retry() {
        let mut task = Task::new(
            Uuid::new_v4(),
            VideoSegment::new("test.mp4".into(), 0.0, 10.0, 0, 1),
            TranscodeParams::default(),
        );

        task.queue_for("worker-1".into()).unwrap();
        task.start().unwrap();
        task.fail("Encoding error".into()).unwrap();

        assert!(task.can_retry());
        assert!(task.retry().unwrap());
        assert_eq!(task.state, TaskState::Pending);
        assert_eq!(task.retry_count, 1);
    }

    #[test]
    fn test_invalid_state_transition() {
        let mut task = Task::new(
            Uuid::new_v4(),
            VideoSegment::new("test.mp4".into(), 0.0, 10.0, 0, 1),
            TranscodeParams::default(),
        );

        // Can't go directly from Pending to Completed
        let result = task.transition_to(TaskState::Completed);
        assert!(result.is_err());
    }
}
