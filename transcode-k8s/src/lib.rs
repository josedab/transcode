//! Kubernetes operator for distributed transcoding.
//!
//! This crate provides a Kubernetes-native operator for managing distributed
//! transcoding workloads with auto-scaling, GPU node affinity, and Prometheus
//! metrics integration.
//!
//! # Architecture
//!
//! The operator manages three Custom Resource Definitions (CRDs):
//!
//! - **TranscodeJob**: A transcoding job with input/output, codec settings, and priority
//! - **TranscodeWorkerPool**: A pool of transcoding workers with scaling policies
//! - **TranscodeCluster**: Cluster-wide configuration and resource quotas
//!
//! # Example
//!
//! ```yaml
//! apiVersion: transcode.io/v1alpha1
//! kind: TranscodeJob
//! metadata:
//!   name: encode-video-001
//! spec:
//!   input: s3://bucket/input.mp4
//!   output: s3://bucket/output.mp4
//!   codec: h264
//!   preset: medium
//!   priority: high
//! ```
//!
//! ```no_run
//! use transcode_k8s::{Operator, OperatorConfig};
//!
//! #[tokio::main]
//! async fn main() -> transcode_k8s::Result<()> {
//!     let config = OperatorConfig::default();
//!     let operator = Operator::new(config)?;
//!     operator.run().await
//! }
//! ```

#![allow(dead_code)]

mod crd;
mod controller;
mod error;
pub mod manifest;
mod metrics;
mod scaling;

pub use crd::{
    TranscodeJob, TranscodeJobSpec, TranscodeJobStatus, JobPhase,
    TranscodeWorkerPool, WorkerPoolSpec, WorkerPoolStatus,
    TranscodeCluster, ClusterSpec,
};
pub use controller::{Operator, OperatorConfig, ReconcileAction};
pub use error::{Error, Result};
pub use manifest::{
    ImageConfig, ProbeConfig, ResourceSpec,
    generate_worker_deployment, generate_hpa, generate_service,
    generate_service_monitor, generate_helm_values,
};
pub use metrics::MetricsCollector;
pub use scaling::{ScalingPolicy, ScalingDecision, AutoScaler};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::{Preset, Priority};

    #[test]
    fn test_operator_config_defaults() {
        let config = OperatorConfig::default();
        assert_eq!(config.namespace, "default");
        assert_eq!(config.reconcile_interval_secs, 30);
        assert!(config.metrics_enabled);
    }

    #[test]
    fn test_job_spec_creation() {
        let spec = TranscodeJobSpec {
            input: "s3://bucket/input.mp4".into(),
            output: "s3://bucket/output.mp4".into(),
            codec: "h264".into(),
            preset: Preset::Medium,
            priority: Priority::Normal,
            gpu_required: false,
            max_retries: 3,
            timeout_secs: 3600,
            labels: std::collections::HashMap::new(),
        };
        assert_eq!(spec.codec, "h264");
        assert_eq!(spec.max_retries, 3);
    }

    #[test]
    fn test_scaling_decision() {
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 10,
            target_queue_depth_per_worker: 2,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_secs: 300,
        };
        let scaler = AutoScaler::new(policy);
        let decision = scaler.evaluate(20, 7, 8);
        assert!(matches!(decision, ScalingDecision::ScaleUp { .. }));
    }

    #[test]
    fn test_worker_pool_spec() {
        let spec = WorkerPoolSpec {
            replicas: 3,
            gpu_type: None,
            cpu_request: "2".into(),
            memory_request: "4Gi".into(),
            scaling: Some(ScalingPolicy::default()),
            tolerations: vec![],
            node_selector: std::collections::HashMap::new(),
        };
        assert_eq!(spec.replicas, 3);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted("encode-001");
        collector.record_job_completed("encode-001", 45.0);
        collector.record_frames_processed(1000);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.jobs_submitted, 1);
        assert_eq!(snapshot.jobs_completed, 1);
        assert_eq!(snapshot.total_frames_processed, 1000);
    }
}
