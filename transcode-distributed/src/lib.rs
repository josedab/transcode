//! Distributed transcoding system for transcode.
//!
//! This crate provides a distributed transcoding system that allows
//! video processing to be distributed across multiple worker nodes.
//!
//! # Architecture
//!
//! The system follows a coordinator-worker architecture:
//!
//! - **Coordinator**: Central node that manages jobs, tasks, and workers
//! - **Workers**: Processing nodes that execute transcoding tasks
//! - **Tasks**: Individual video segments to be processed
//! - **Jobs**: Collections of tasks representing a complete transcoding job
//!
//! # Example
//!
//! ```ignore
//! use transcode_distributed::{Coordinator, CoordinatorConfig, WorkerInfo};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create coordinator
//!     let coordinator = Coordinator::new(CoordinatorConfig::default());
//!     coordinator.start().await;
//!
//!     // Register workers
//!     let worker = WorkerInfo::new(
//!         "worker-1".into(),
//!         "192.168.1.10:8080".into(),
//!         WorkerCapabilities::default(),
//!     );
//!     coordinator.register_worker(worker).await.unwrap();
//!
//!     // Submit a job
//!     let job_id = coordinator.submit_job(
//!         "my-video".into(),
//!         "/input/video.mp4".into(),
//!         "/output/video.mp4".into(),
//!         TranscodeParams::default(),
//!         120.0, // duration in seconds
//!     ).await.unwrap();
//!
//!     // Monitor progress
//!     let mut events = coordinator.subscribe();
//!     while let Ok(event) = events.recv().await {
//!         match event {
//!             CoordinatorEvent::Progress { job_id, progress } => {
//!                 println!("Job {} progress: {:.1}%", job_id, progress * 100.0);
//!             }
//!             CoordinatorEvent::JobCompleted { job_id } => {
//!                 println!("Job {} completed!", job_id);
//!                 break;
//!             }
//!             _ => {}
//!         }
//!     }
//! }
//! ```
//!
//! # Segment-based Processing
//!
//! Videos are automatically split into segments for parallel processing:
//!
//! 1. Job submitted with video duration
//! 2. Coordinator creates tasks for each segment
//! 3. Tasks distributed to available workers
//! 4. Workers process segments in parallel
//! 5. Results aggregated for final output
//!
//! # Fault Tolerance
//!
//! The system handles failures gracefully:
//!
//! - **Task retries**: Failed tasks are automatically retried
//! - **Worker failover**: Tasks from offline workers are reassigned
//! - **Health monitoring**: Workers report heartbeats to coordinator

pub mod coordinator;
pub mod error;
#[cfg(feature = "postgres")]
pub mod persistence;
#[cfg(feature = "redis")]
pub mod redis;
pub mod task;
pub mod worker;

pub use coordinator::{Coordinator, CoordinatorConfig, CoordinatorEvent};
pub use error::{DistributedError, Result};
#[cfg(feature = "postgres")]
pub use persistence::{JobStore, PersistenceStore, PostgresStore, TaskStore, WorkerStore};
#[cfg(feature = "redis")]
pub use redis::{
    DistributedLock, LockGuard, RedisConfig, RedisCoordinator, TaskCoordinator, WorkerCoordinator,
};
pub use task::{Job, Priority, Task, TaskState, TranscodeParams, VideoSegment};
pub use worker::{WorkerCapabilities, WorkerHealth, WorkerInfo, WorkerLoad, WorkerPool};

/// Protocol version for compatibility checking.
pub const PROTOCOL_VERSION: &str = "1.0.0";

/// Default segment duration in seconds.
pub const DEFAULT_SEGMENT_DURATION: f64 = 10.0;

/// Client for communicating with a coordinator.
#[derive(Debug)]
pub struct CoordinatorClient {
    /// Coordinator address.
    address: String,
}

impl CoordinatorClient {
    /// Create a new coordinator client.
    pub fn new(address: String) -> Self {
        Self { address }
    }

    /// Get coordinator address.
    pub fn address(&self) -> &str {
        &self.address
    }
}

/// Worker runner for processing tasks.
#[derive(Debug)]
pub struct WorkerRunner {
    /// Worker information.
    info: WorkerInfo,
    /// Coordinator client.
    #[allow(dead_code)]
    coordinator: CoordinatorClient,
    /// Running flag.
    running: bool,
}

impl WorkerRunner {
    /// Create a new worker runner.
    pub fn new(info: WorkerInfo, coordinator_address: String) -> Self {
        Self {
            info,
            coordinator: CoordinatorClient::new(coordinator_address),
            running: false,
        }
    }

    /// Get worker info.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Start the worker.
    pub async fn start(&mut self) -> Result<()> {
        self.running = true;
        tracing::info!("Worker {} started", self.info.id);
        Ok(())
    }

    /// Stop the worker.
    pub async fn stop(&mut self) -> Result<()> {
        self.running = false;
        tracing::info!("Worker {} stopped", self.info.id);
        Ok(())
    }

    /// Check if worker is running.
    pub fn is_running(&self) -> bool {
        self.running
    }
}

/// Segment splitter for dividing videos.
#[derive(Debug, Clone)]
pub struct SegmentSplitter {
    /// Segment duration in seconds.
    segment_duration: f64,
}

impl Default for SegmentSplitter {
    fn default() -> Self {
        Self {
            segment_duration: DEFAULT_SEGMENT_DURATION,
        }
    }
}

impl SegmentSplitter {
    /// Create a new segment splitter.
    pub fn new(segment_duration: f64) -> Self {
        Self {
            segment_duration: segment_duration.max(1.0),
        }
    }

    /// Split a video into segments.
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

    /// Estimate segment count for a duration.
    pub fn estimate_segments(&self, duration: f64) -> usize {
        (duration / self.segment_duration).ceil() as usize
    }
}

/// Statistics for the distributed system.
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total jobs submitted.
    pub total_jobs: u64,
    /// Completed jobs.
    pub completed_jobs: u64,
    /// Failed jobs.
    pub failed_jobs: u64,
    /// Total tasks.
    pub total_tasks: u64,
    /// Completed tasks.
    pub completed_tasks: u64,
    /// Failed tasks.
    pub failed_tasks: u64,
    /// Active workers.
    pub active_workers: usize,
    /// Total processing time (seconds).
    pub total_processing_time: f64,
}

impl DistributedStats {
    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_jobs == 0 {
            0.0
        } else {
            self.completed_jobs as f64 / self.total_jobs as f64
        }
    }

    /// Calculate average processing time per job.
    pub fn avg_processing_time(&self) -> f64 {
        if self.completed_jobs == 0 {
            0.0
        } else {
            self.total_processing_time / self.completed_jobs as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_splitter() {
        let splitter = SegmentSplitter::new(10.0);
        let segments = splitter.split("/video.mp4", 25.0);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].start_time, 0.0);
        assert_eq!(segments[0].end_time, 10.0);
        assert_eq!(segments[1].start_time, 10.0);
        assert_eq!(segments[1].end_time, 20.0);
        assert_eq!(segments[2].start_time, 20.0);
        assert_eq!(segments[2].end_time, 25.0);
    }

    #[test]
    fn test_segment_splitter_short_video() {
        let splitter = SegmentSplitter::new(10.0);
        let segments = splitter.split("/video.mp4", 5.0);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].duration(), 5.0);
    }

    #[test]
    fn test_distributed_stats() {
        let stats = DistributedStats {
            total_jobs: 100,
            completed_jobs: 90,
            failed_jobs: 10,
            total_processing_time: 9000.0,
            ..Default::default()
        };

        assert_eq!(stats.success_rate(), 0.9);
        assert_eq!(stats.avg_processing_time(), 100.0);
    }
}
