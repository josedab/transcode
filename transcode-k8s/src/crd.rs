//! Custom Resource Definitions for the Transcode Kubernetes operator.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TranscodeJob CRD
// ---------------------------------------------------------------------------

/// A transcoding job to be executed by the worker pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJob {
    pub api_version: String,
    pub kind: String,
    pub metadata: ObjectMeta,
    pub spec: TranscodeJobSpec,
    #[serde(default)]
    pub status: Option<TranscodeJobStatus>,
}

impl TranscodeJob {
    pub fn new(name: &str, namespace: &str, spec: TranscodeJobSpec) -> Self {
        Self {
            api_version: "transcode.io/v1alpha1".into(),
            kind: "TranscodeJob".into(),
            metadata: ObjectMeta {
                name: name.into(),
                namespace: namespace.into(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
                uid: uuid::Uuid::new_v4().to_string(),
                resource_version: String::new(),
            },
            spec,
            status: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJobSpec {
    pub input: String,
    pub output: String,
    pub codec: String,
    pub preset: Preset,
    pub priority: Priority,
    #[serde(default)]
    pub gpu_required: bool,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

fn default_max_retries() -> u32 {
    3
}
fn default_timeout() -> u64 {
    3600
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscodeJobStatus {
    pub phase: JobPhase,
    pub worker_name: Option<String>,
    pub start_time: Option<String>,
    pub completion_time: Option<String>,
    pub progress: f64,
    pub retries: u32,
    pub message: Option<String>,
    pub frames_processed: u64,
    pub output_size_bytes: u64,
}

impl Default for TranscodeJobStatus {
    fn default() -> Self {
        Self {
            phase: JobPhase::Pending,
            worker_name: None,
            start_time: None,
            completion_time: None,
            progress: 0.0,
            retries: 0,
            message: None,
            frames_processed: 0,
            output_size_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum JobPhase {
    Pending,
    Queued,
    Assigned,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    #[default]
    Medium,
    Slow,
    Slower,
    Veryslow,
    Placebo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

impl Priority {
    pub fn weight(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 5,
            Self::High => 10,
            Self::Critical => 100,
        }
    }
}

// ---------------------------------------------------------------------------
// TranscodeWorkerPool CRD
// ---------------------------------------------------------------------------

/// A pool of transcoding workers with scaling policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeWorkerPool {
    pub api_version: String,
    pub kind: String,
    pub metadata: ObjectMeta,
    pub spec: WorkerPoolSpec,
    #[serde(default)]
    pub status: Option<WorkerPoolStatus>,
}

impl TranscodeWorkerPool {
    pub fn new(name: &str, namespace: &str, spec: WorkerPoolSpec) -> Self {
        Self {
            api_version: "transcode.io/v1alpha1".into(),
            kind: "TranscodeWorkerPool".into(),
            metadata: ObjectMeta {
                name: name.into(),
                namespace: namespace.into(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
                uid: uuid::Uuid::new_v4().to_string(),
                resource_version: String::new(),
            },
            spec,
            status: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPoolSpec {
    pub replicas: u32,
    pub gpu_type: Option<GpuType>,
    pub cpu_request: String,
    pub memory_request: String,
    pub scaling: Option<crate::scaling::ScalingPolicy>,
    #[serde(default)]
    pub tolerations: Vec<Toleration>,
    #[serde(default)]
    pub node_selector: HashMap<String, String>,
}

impl Default for WorkerPoolSpec {
    fn default() -> Self {
        Self {
            replicas: 1,
            gpu_type: None,
            cpu_request: "2".into(),
            memory_request: "4Gi".into(),
            scaling: None,
            tolerations: vec![],
            node_selector: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPoolStatus {
    pub ready_replicas: u32,
    pub active_jobs: u32,
    pub total_completed: u64,
    pub total_failed: u64,
    pub avg_job_duration_secs: f64,
}

// ---------------------------------------------------------------------------
// TranscodeCluster CRD
// ---------------------------------------------------------------------------

/// Cluster-wide transcoding configuration and resource quotas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeCluster {
    pub api_version: String,
    pub kind: String,
    pub metadata: ObjectMeta,
    pub spec: ClusterSpec,
}

impl TranscodeCluster {
    pub fn new(name: &str, namespace: &str, spec: ClusterSpec) -> Self {
        Self {
            api_version: "transcode.io/v1alpha1".into(),
            kind: "TranscodeCluster".into(),
            metadata: ObjectMeta {
                name: name.into(),
                namespace: namespace.into(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
                uid: uuid::Uuid::new_v4().to_string(),
                resource_version: String::new(),
            },
            spec,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSpec {
    pub max_concurrent_jobs: u32,
    pub default_priority: Priority,
    pub storage_class: String,
    pub gpu_scheduling_enabled: bool,
    pub metrics_enabled: bool,
    pub job_history_limit: u32,
    pub default_timeout_secs: u64,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 100,
            default_priority: Priority::Normal,
            storage_class: "standard".into(),
            gpu_scheduling_enabled: false,
            metrics_enabled: true,
            job_history_limit: 1000,
            default_timeout_secs: 3600,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObjectMeta {
    pub name: String,
    pub namespace: String,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
    #[serde(default)]
    pub uid: String,
    #[serde(default)]
    pub resource_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuType {
    NvidiaTesla,
    NvidiaA100,
    NvidiaT4,
    NvidiaV100,
    AmdMi250,
    IntelArc,
}

impl GpuType {
    pub fn resource_key(&self) -> &'static str {
        match self {
            Self::NvidiaTesla | Self::NvidiaA100 | Self::NvidiaT4 | Self::NvidiaV100 => {
                "nvidia.com/gpu"
            }
            Self::AmdMi250 => "amd.com/gpu",
            Self::IntelArc => "gpu.intel.com/i915",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    pub key: String,
    pub operator: String,
    pub value: Option<String>,
    pub effect: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_serialization_roundtrip() {
        let spec = TranscodeJobSpec {
            input: "s3://bucket/input.mp4".into(),
            output: "s3://bucket/output.mp4".into(),
            codec: "h264".into(),
            preset: Preset::Medium,
            priority: Priority::High,
            gpu_required: true,
            max_retries: 5,
            timeout_secs: 7200,
            labels: HashMap::from([("team".into(), "video".into())]),
        };
        let job = TranscodeJob::new("test-job", "production", spec);
        let json = serde_json::to_string(&job).unwrap();
        let deserialized: TranscodeJob = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.metadata.name, "test-job");
        assert_eq!(deserialized.spec.priority, Priority::High);
    }

    #[test]
    fn test_priority_weights() {
        assert!(Priority::Critical.weight() > Priority::High.weight());
        assert!(Priority::High.weight() > Priority::Normal.weight());
        assert!(Priority::Normal.weight() > Priority::Low.weight());
    }

    #[test]
    fn test_gpu_resource_keys() {
        assert_eq!(GpuType::NvidiaA100.resource_key(), "nvidia.com/gpu");
        assert_eq!(GpuType::AmdMi250.resource_key(), "amd.com/gpu");
        assert_eq!(GpuType::IntelArc.resource_key(), "gpu.intel.com/i915");
    }

    #[test]
    fn test_worker_pool_defaults() {
        let spec = WorkerPoolSpec::default();
        assert_eq!(spec.replicas, 1);
        assert!(spec.gpu_type.is_none());
        assert_eq!(spec.cpu_request, "2");
    }

    #[test]
    fn test_cluster_defaults() {
        let spec = ClusterSpec::default();
        assert_eq!(spec.max_concurrent_jobs, 100);
        assert!(spec.metrics_enabled);
    }
}
