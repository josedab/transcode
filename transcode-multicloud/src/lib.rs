//! Multi-cloud orchestration SDK for the transcode library.
//!
//! This crate provides a unified API for orchestrating transcoding jobs across
//! multiple cloud providers (AWS MediaConvert, GCP Transcoder API, Azure Media Services).
//!
//! # Features
//!
//! - **Unified API**: Single interface for all cloud providers
//! - **Job Abstraction**: Provider-agnostic job definitions
//! - **Cost Optimization**: Route jobs to cheapest available provider
//! - **Failover**: Automatic failover between providers
//! - **Load Balancing**: Distribute jobs across providers
//!
//! # Example
//!
//! ```ignore
//! use transcode_multicloud::{CloudOrchestrator, TranscodeJob, Provider, RoutingStrategy};
//!
//! // Create orchestrator with multiple providers
//! let orchestrator = CloudOrchestrator::builder()
//!     .aws("us-east-1")
//!     .gcp("us-central1")
//!     .azure("westus2")
//!     .strategy(RoutingStrategy::CostOptimized)
//!     .with_failover()
//!     .build();
//!
//! // Create a transcoding job
//! let job = TranscodeJob::builder()
//!     .input("s3://bucket/input.mp4")
//!     .output_uri("s3://bucket/output/")
//!     .preset(Preset::hls_1080p())
//!     .priority(100)
//!     .max_cost(500)
//!     .build()?;
//!
//! // Submit the job
//! let job_id = orchestrator.submit(job).await?;
//!
//! // Check status
//! let status = orchestrator.get_status(&job_id)?;
//!
//! // Get result when complete
//! let result = orchestrator.get_result(&job_id)?;
//! ```
//!
//! # Routing Strategies
//!
//! The orchestrator supports multiple routing strategies:
//!
//! - `CostOptimized`: Route to the cheapest available provider
//! - `LatencyOptimized`: Route to the fastest available provider
//! - `RoundRobin`: Distribute jobs evenly across providers
//! - `LoadBalanced`: Route based on current provider load
//! - `Preferred`: Use a preferred provider with fallback
//!
//! # Provider Configuration
//!
//! ```ignore
//! use transcode_multicloud::{Provider, ProviderConfig, CredentialsConfig};
//!
//! // AWS with custom settings
//! let aws = Provider::new(
//!     ProviderConfig::aws("us-east-1")
//!         .credentials(CredentialsConfig::InstanceProfile)
//! );
//!
//! // GCP with service account
//! let gcp = Provider::new(
//!     ProviderConfig::gcp("us-central1")
//!         .credentials(CredentialsConfig::ServiceAccount {
//!             path: "/path/to/service-account.json".into()
//!         })
//! );
//! ```

#![allow(dead_code)]

mod error;
mod health;
mod job;
mod orchestrator;
mod provider;

pub use error::{MultiCloudError, Result};
pub use health::{
    CircuitBreaker, CircuitState, CostBreakdown, CostEstimate, CostEstimator, HealthMonitor,
    HealthProbe, ProviderPricing, ProviderStatus,
};
pub use job::{
    AudioCodec, AudioOutput, JobResult, JobStats, JobStatus, JobTimestamps, OutputConfig,
    OutputFormat, Preset, QualityPreset, TranscodeJob, TranscodeJobBuilder, VideoCodec,
    VideoOutput,
};
pub use orchestrator::{
    CloudOrchestrator, CloudOrchestratorBuilder, OrchestratorConfig, OrchestratorConfigBuilder,
    OrchestratorStats, RoutingStrategy,
};
pub use provider::{
    AccelerationMode, AwsSettings, AzureSettings, CredentialsConfig, GcpSettings, LocalSettings,
    Provider, ProviderCapabilities, ProviderConfig, ProviderHealth, ProviderSettings, ProviderType,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[tokio::test]
    async fn test_end_to_end() {
        // Create orchestrator
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .strategy(RoutingStrategy::CostOptimized)
            .build();

        // Create job
        let job = TranscodeJob::builder()
            .input("s3://bucket/input.mp4")
            .output_uri("s3://bucket/output/")
            .priority(100)
            .build()
            .unwrap();

        // Submit job
        let job_id = orchestrator.submit(job).await.unwrap();
        assert!(!job_id.is_empty());

        // Check status
        let status = orchestrator.get_status(&job_id).unwrap();
        assert!(status.is_active());

        // Check stats
        let stats = orchestrator.stats();
        assert_eq!(stats.total_providers, 2);
        assert!(stats.active_jobs > 0 || stats.total_jobs > 0);
    }

    #[test]
    fn test_presets() {
        let hls = Preset::hls_1080p();
        assert_eq!(hls.video.width, Some(1920));
        assert_eq!(hls.video.height, Some(1080));
        assert_eq!(hls.format, OutputFormat::Hls);

        let archive = Preset::archive();
        assert_eq!(archive.video.codec, VideoCodec::H265);
        assert_eq!(archive.audio.codec, AudioCodec::Flac);
    }
}
