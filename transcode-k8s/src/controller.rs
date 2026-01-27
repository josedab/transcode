//! Controller logic for the Transcode Kubernetes operator.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::crd::{JobPhase, TranscodeJob, TranscodeJobStatus, TranscodeWorkerPool};
use crate::error::{Error, Result};
use crate::metrics::MetricsCollector;
use crate::scaling::AutoScaler;

/// Configuration for the operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    pub namespace: String,
    pub reconcile_interval_secs: u64,
    pub metrics_enabled: bool,
    pub max_concurrent_reconciles: usize,
    pub leader_election_enabled: bool,
    pub leader_election_lease_name: String,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            namespace: "default".into(),
            reconcile_interval_secs: 30,
            metrics_enabled: true,
            max_concurrent_reconciles: 10,
            leader_election_enabled: true,
            leader_election_lease_name: "transcode-operator-lease".into(),
        }
    }
}

/// Action returned by the reconciler indicating what to do next.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileAction {
    /// No action needed — resource is in desired state.
    NoOp,
    /// Requeue after the specified number of seconds.
    Requeue { after_secs: u64 },
    /// Assign the job to a specific worker.
    AssignToWorker { job_name: String, worker_name: String },
    /// Scale the worker pool to the target replicas.
    ScaleWorkerPool { pool_name: String, target: u32 },
    /// Update the job status.
    UpdateStatus { job_name: String, status: TranscodeJobStatus },
}

/// The main operator that watches CRDs and reconciles state.
pub struct Operator {
    config: OperatorConfig,
    jobs: HashMap<String, TranscodeJob>,
    worker_pools: HashMap<String, TranscodeWorkerPool>,
    queue: VecDeque<String>,
    metrics: MetricsCollector,
    scaler: Option<AutoScaler>,
}

impl Operator {
    pub fn new(config: OperatorConfig) -> Result<Self> {
        info!(
            namespace = %config.namespace,
            "Initializing transcode operator"
        );
        Ok(Self {
            config,
            jobs: HashMap::new(),
            worker_pools: HashMap::new(),
            queue: VecDeque::new(),
            metrics: MetricsCollector::new(),
            scaler: None,
        })
    }

    /// Run the operator's main reconciliation loop.
    pub async fn run(&self) -> Result<()> {
        info!("Starting transcode operator");
        // In a real implementation, this would watch K8s API resources.
        // Here we provide the control loop structure.
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(
                self.config.reconcile_interval_secs,
            ))
            .await;
            debug!("Reconciliation tick");
        }
    }

    /// Submit a new job to the operator.
    pub fn submit_job(&mut self, job: TranscodeJob) -> Result<String> {
        let name = job.metadata.name.clone();
        if self.jobs.contains_key(&name) {
            return Err(Error::Reconcile {
                resource: name,
                message: "Job already exists".into(),
            });
        }
        self.metrics.record_job_submitted(&name);
        info!(job = %name, "Job submitted");
        self.queue.push_back(name.clone());
        self.jobs.insert(name.clone(), job);
        Ok(name)
    }

    /// Register a worker pool.
    pub fn register_worker_pool(&mut self, pool: TranscodeWorkerPool) {
        let name = pool.metadata.name.clone();
        if let Some(ref scaling) = pool.spec.scaling {
            self.scaler = Some(AutoScaler::new(scaling.clone()));
        }
        info!(pool = %name, replicas = pool.spec.replicas, "Worker pool registered");
        self.worker_pools.insert(name, pool);
    }

    /// Reconcile a single job — the core operator logic.
    pub fn reconcile_job(&mut self, job_name: &str) -> Result<ReconcileAction> {
        let job = self.jobs.get(job_name).ok_or_else(|| Error::NotFound {
            kind: "TranscodeJob".into(),
            name: job_name.into(),
            namespace: self.config.namespace.clone(),
        })?;

        let phase = job
            .status
            .as_ref()
            .map(|s| s.phase)
            .unwrap_or(JobPhase::Pending);

        match phase {
            JobPhase::Pending => {
                // Move to queued
                let status = TranscodeJobStatus {
                    phase: JobPhase::Queued,
                    ..Default::default()
                };
                Ok(ReconcileAction::UpdateStatus {
                    job_name: job_name.into(),
                    status,
                })
            }
            JobPhase::Queued => {
                // Try to assign to an available worker
                if let Some(worker_name) = self.find_available_worker(job) {
                    Ok(ReconcileAction::AssignToWorker {
                        job_name: job_name.into(),
                        worker_name,
                    })
                } else {
                    // Check if we should scale up
                    if let Some(ref scaler) = self.scaler {
                        let queue_depth = self.queue.len() as u32;
                        let active = self.count_active_workers();
                        let current_replicas = self.total_replicas();
                        let decision = scaler.evaluate(queue_depth, active, current_replicas);
                        if let crate::scaling::ScalingDecision::ScaleUp { target } = decision {
                            if let Some(pool_name) = self.worker_pools.keys().next().cloned() {
                                return Ok(ReconcileAction::ScaleWorkerPool {
                                    pool_name,
                                    target,
                                });
                            }
                        }
                    }
                    Ok(ReconcileAction::Requeue { after_secs: 10 })
                }
            }
            JobPhase::Running => {
                // Check for timeout
                Ok(ReconcileAction::Requeue { after_secs: 30 })
            }
            JobPhase::Succeeded | JobPhase::Failed | JobPhase::Cancelled => {
                Ok(ReconcileAction::NoOp)
            }
            JobPhase::Assigned => Ok(ReconcileAction::Requeue { after_secs: 5 }),
        }
    }

    /// Apply a reconcile action (update internal state).
    pub fn apply_action(&mut self, action: &ReconcileAction) -> Result<()> {
        match action {
            ReconcileAction::UpdateStatus { job_name, status } => {
                if let Some(job) = self.jobs.get_mut(job_name) {
                    job.status = Some(status.clone());
                    debug!(job = %job_name, phase = ?status.phase, "Status updated");
                }
            }
            ReconcileAction::AssignToWorker {
                job_name,
                worker_name,
            } => {
                if let Some(job) = self.jobs.get_mut(job_name) {
                    let now = chrono::Utc::now().to_rfc3339();
                    job.status = Some(TranscodeJobStatus {
                        phase: JobPhase::Assigned,
                        worker_name: Some(worker_name.clone()),
                        start_time: Some(now),
                        ..Default::default()
                    });
                    info!(job = %job_name, worker = %worker_name, "Job assigned");
                }
            }
            ReconcileAction::ScaleWorkerPool { pool_name, target } => {
                if let Some(pool) = self.worker_pools.get_mut(pool_name) {
                    let old = pool.spec.replicas;
                    pool.spec.replicas = *target;
                    info!(pool = %pool_name, from = old, to = target, "Scaled worker pool");
                }
            }
            ReconcileAction::NoOp | ReconcileAction::Requeue { .. } => {}
        }
        Ok(())
    }

    /// Mark a job as completed.
    pub fn complete_job(&mut self, job_name: &str, output_size: u64, frames: u64) -> Result<()> {
        let job = self.jobs.get_mut(job_name).ok_or_else(|| Error::NotFound {
            kind: "TranscodeJob".into(),
            name: job_name.into(),
            namespace: self.config.namespace.clone(),
        })?;

        let now = chrono::Utc::now().to_rfc3339();
        let duration = job
            .status
            .as_ref()
            .and_then(|s| s.start_time.as_ref())
            .map(|_| 45.0) // placeholder duration
            .unwrap_or(0.0);

        job.status = Some(TranscodeJobStatus {
            phase: JobPhase::Succeeded,
            completion_time: Some(now),
            progress: 1.0,
            frames_processed: frames,
            output_size_bytes: output_size,
            ..job.status.clone().unwrap_or_default()
        });

        self.metrics.record_job_completed(job_name, duration);
        self.metrics.record_frames_processed(frames);
        info!(job = %job_name, frames = frames, "Job completed");
        Ok(())
    }

    /// Fail a job with a message.
    pub fn fail_job(&mut self, job_name: &str, message: &str) -> Result<()> {
        let job = self.jobs.get_mut(job_name).ok_or_else(|| Error::NotFound {
            kind: "TranscodeJob".into(),
            name: job_name.into(),
            namespace: self.config.namespace.clone(),
        })?;

        let retries = job.status.as_ref().map(|s| s.retries).unwrap_or(0);
        if retries < job.spec.max_retries {
            job.status = Some(TranscodeJobStatus {
                phase: JobPhase::Queued,
                retries: retries + 1,
                message: Some(format!("Retry {}: {}", retries + 1, message)),
                worker_name: None,
                ..job.status.clone().unwrap_or_default()
            });
            warn!(job = %job_name, retry = retries + 1, "Job retrying");
            self.queue.push_back(job_name.into());
        } else {
            job.status = Some(TranscodeJobStatus {
                phase: JobPhase::Failed,
                retries,
                message: Some(format!("Max retries exceeded: {}", message)),
                ..job.status.clone().unwrap_or_default()
            });
            self.metrics.record_job_failed(job_name);
            warn!(job = %job_name, "Job failed permanently");
        }
        Ok(())
    }

    pub fn get_job(&self, name: &str) -> Option<&TranscodeJob> {
        self.jobs.get(name)
    }

    pub fn pending_jobs(&self) -> usize {
        self.queue.len()
    }

    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics
    }

    fn find_available_worker(&self, job: &TranscodeJob) -> Option<String> {
        for (name, pool) in &self.worker_pools {
            let active = pool.status.as_ref().map(|s| s.active_jobs).unwrap_or(0);
            if active < pool.spec.replicas {
                // Check GPU affinity
                if job.spec.gpu_required && pool.spec.gpu_type.is_none() {
                    continue;
                }
                return Some(name.clone());
            }
        }
        None
    }

    fn count_active_workers(&self) -> u32 {
        self.worker_pools
            .values()
            .filter_map(|p| p.status.as_ref())
            .map(|s| s.active_jobs)
            .sum()
    }

    fn total_replicas(&self) -> u32 {
        self.worker_pools.values().map(|p| p.spec.replicas).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::*;
    use crate::scaling::ScalingPolicy;

    fn make_job(name: &str, gpu: bool) -> TranscodeJob {
        TranscodeJob::new(
            name,
            "default",
            TranscodeJobSpec {
                input: "s3://in.mp4".into(),
                output: "s3://out.mp4".into(),
                codec: "h264".into(),
                preset: Preset::Fast,
                priority: Priority::Normal,
                gpu_required: gpu,
                max_retries: 3,
                timeout_secs: 600,
                labels: HashMap::new(),
            },
        )
    }

    fn make_pool(name: &str, replicas: u32, gpu: bool) -> TranscodeWorkerPool {
        TranscodeWorkerPool::new(
            name,
            "default",
            WorkerPoolSpec {
                replicas,
                gpu_type: if gpu { Some(GpuType::NvidiaT4) } else { None },
                scaling: Some(ScalingPolicy::default()),
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_submit_and_reconcile() {
        let mut op = Operator::new(OperatorConfig::default()).unwrap();
        op.register_worker_pool(make_pool("pool-1", 2, false));
        let name = op.submit_job(make_job("job-1", false)).unwrap();

        // First reconcile: Pending -> Queued
        let action = op.reconcile_job(&name).unwrap();
        assert!(matches!(action, ReconcileAction::UpdateStatus { .. }));
        op.apply_action(&action).unwrap();

        // Second reconcile: Queued -> Assigned
        let action = op.reconcile_job(&name).unwrap();
        assert!(matches!(action, ReconcileAction::AssignToWorker { .. }));
        op.apply_action(&action).unwrap();

        let job = op.get_job(&name).unwrap();
        assert_eq!(job.status.as_ref().unwrap().phase, JobPhase::Assigned);
    }

    #[test]
    fn test_gpu_affinity() {
        let mut op = Operator::new(OperatorConfig::default()).unwrap();
        op.register_worker_pool(make_pool("cpu-pool", 2, false));

        op.submit_job(make_job("gpu-job", true)).unwrap();
        let action = op.reconcile_job("gpu-job").unwrap();
        op.apply_action(&action).unwrap();

        // GPU job should not be assigned to CPU pool
        let action = op.reconcile_job("gpu-job").unwrap();
        assert!(matches!(
            action,
            ReconcileAction::ScaleWorkerPool { .. } | ReconcileAction::Requeue { .. }
        ));
    }

    #[test]
    fn test_job_completion() {
        let mut op = Operator::new(OperatorConfig::default()).unwrap();
        op.submit_job(make_job("job-1", false)).unwrap();
        op.complete_job("job-1", 1024 * 1024, 300).unwrap();

        let job = op.get_job("job-1").unwrap();
        assert_eq!(job.status.as_ref().unwrap().phase, JobPhase::Succeeded);
        assert_eq!(job.status.as_ref().unwrap().frames_processed, 300);
    }

    #[test]
    fn test_job_retry_and_failure() {
        let mut op = Operator::new(OperatorConfig::default()).unwrap();
        op.submit_job(make_job("job-1", false)).unwrap();

        // Fail 3 times (max retries)
        for i in 0..3 {
            op.fail_job("job-1", &format!("error {}", i)).unwrap();
        }
        // 4th failure should be permanent
        op.fail_job("job-1", "final error").unwrap();
        let job = op.get_job("job-1").unwrap();
        assert_eq!(job.status.as_ref().unwrap().phase, JobPhase::Failed);
    }

    #[test]
    fn test_duplicate_job_rejected() {
        let mut op = Operator::new(OperatorConfig::default()).unwrap();
        op.submit_job(make_job("job-1", false)).unwrap();
        assert!(op.submit_job(make_job("job-1", false)).is_err());
    }
}
