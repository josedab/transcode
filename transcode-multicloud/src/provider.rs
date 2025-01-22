//! Cloud provider definitions and capabilities.

use crate::job::{OutputFormat, TranscodeJob, VideoCodec};
use crate::{MultiCloudError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Supported cloud provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// AWS Elemental MediaConvert.
    #[default]
    Aws,
    /// Google Cloud Transcoder API.
    Gcp,
    /// Azure Media Services.
    Azure,
    /// Local transcoding (fallback).
    Local,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Aws => write!(f, "AWS"),
            ProviderType::Gcp => write!(f, "GCP"),
            ProviderType::Azure => write!(f, "Azure"),
            ProviderType::Local => write!(f, "Local"),
        }
    }
}

/// Cloud provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type.
    pub provider_type: ProviderType,
    /// Region/location.
    pub region: String,
    /// API endpoint (optional, for custom endpoints).
    pub endpoint: Option<String>,
    /// Credentials configuration.
    pub credentials: CredentialsConfig,
    /// Provider-specific settings.
    pub settings: ProviderSettings,
}

impl ProviderConfig {
    /// Create AWS provider configuration.
    pub fn aws(region: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Aws,
            region: region.into(),
            endpoint: None,
            credentials: CredentialsConfig::default(),
            settings: ProviderSettings::Aws(AwsSettings::default()),
        }
    }

    /// Create GCP provider configuration.
    pub fn gcp(region: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Gcp,
            region: region.into(),
            endpoint: None,
            credentials: CredentialsConfig::default(),
            settings: ProviderSettings::Gcp(GcpSettings::default()),
        }
    }

    /// Create Azure provider configuration.
    pub fn azure(region: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Azure,
            region: region.into(),
            endpoint: None,
            credentials: CredentialsConfig::default(),
            settings: ProviderSettings::Azure(AzureSettings::default()),
        }
    }

    /// Create local provider configuration.
    pub fn local() -> Self {
        Self {
            provider_type: ProviderType::Local,
            region: "local".to_string(),
            endpoint: None,
            credentials: CredentialsConfig::None,
            settings: ProviderSettings::Local(LocalSettings::default()),
        }
    }

    /// Set custom endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set credentials.
    pub fn credentials(mut self, credentials: CredentialsConfig) -> Self {
        self.credentials = credentials;
        self
    }
}

/// Credentials configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CredentialsConfig {
    /// No credentials (for local provider).
    None,
    /// Use environment variables.
    Environment,
    /// Use instance metadata/IAM roles.
    InstanceProfile,
    /// Explicit credentials.
    Explicit {
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
    },
    /// Service account file (GCP).
    ServiceAccount { path: String },
}

impl Default for CredentialsConfig {
    fn default() -> Self {
        Self::Environment
    }
}

/// Provider-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum ProviderSettings {
    Aws(AwsSettings),
    Gcp(GcpSettings),
    Azure(AzureSettings),
    Local(LocalSettings),
}

/// AWS MediaConvert settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AwsSettings {
    /// MediaConvert queue ARN.
    pub queue_arn: Option<String>,
    /// IAM role ARN for MediaConvert.
    pub role_arn: Option<String>,
    /// Acceleration mode.
    pub acceleration: Option<AccelerationMode>,
    /// Priority (-50 to 50).
    pub priority: Option<i32>,
}

/// GCP Transcoder API settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcpSettings {
    /// GCP project ID.
    pub project_id: Option<String>,
    /// Job template name.
    pub template_name: Option<String>,
    /// TTL for completed jobs.
    pub ttl_days: Option<u32>,
}

/// Azure Media Services settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AzureSettings {
    /// Subscription ID.
    pub subscription_id: Option<String>,
    /// Resource group name.
    pub resource_group: Option<String>,
    /// Media Services account name.
    pub account_name: Option<String>,
    /// Transform name.
    pub transform_name: Option<String>,
}

/// Local transcoding settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSettings {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Maximum concurrent jobs.
    pub max_concurrent_jobs: usize,
    /// Temporary directory.
    pub temp_dir: Option<String>,
}

impl Default for LocalSettings {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            max_concurrent_jobs: 4,
            temp_dir: None,
        }
    }
}

/// Hardware acceleration mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationMode {
    /// No acceleration.
    Disabled,
    /// Prefer acceleration when available.
    Preferred,
    /// Require acceleration.
    Required,
}

/// Provider capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Supported video codecs.
    pub video_codecs: HashSet<VideoCodec>,
    /// Supported output formats.
    pub output_formats: HashSet<OutputFormat>,
    /// Maximum input file size in bytes.
    pub max_input_size_bytes: u64,
    /// Maximum output resolution.
    pub max_resolution: (u32, u32),
    /// Supports hardware acceleration.
    pub hardware_acceleration: bool,
    /// Supports HDR.
    pub hdr_support: bool,
    /// Supports DRM.
    pub drm_support: bool,
    /// Average cost per minute of output (cents).
    pub cost_per_minute_cents: f64,
    /// Average latency multiplier (1.0 = realtime).
    pub latency_multiplier: f64,
}

impl ProviderCapabilities {
    /// Get capabilities for a provider type.
    pub fn for_provider(provider_type: ProviderType) -> Self {
        match provider_type {
            ProviderType::Aws => Self::aws_capabilities(),
            ProviderType::Gcp => Self::gcp_capabilities(),
            ProviderType::Azure => Self::azure_capabilities(),
            ProviderType::Local => Self::local_capabilities(),
        }
    }

    fn aws_capabilities() -> Self {
        Self {
            video_codecs: [VideoCodec::H264, VideoCodec::H265, VideoCodec::Vp9, VideoCodec::Av1]
                .into_iter()
                .collect(),
            output_formats: [
                OutputFormat::Mp4,
                OutputFormat::Hls,
                OutputFormat::Dash,
                OutputFormat::Webm,
                OutputFormat::Mov,
            ]
            .into_iter()
            .collect(),
            max_input_size_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
            max_resolution: (8192, 4320),                   // 8K
            hardware_acceleration: true,
            hdr_support: true,
            drm_support: true,
            cost_per_minute_cents: 1.5,
            latency_multiplier: 0.5,
        }
    }

    fn gcp_capabilities() -> Self {
        Self {
            video_codecs: [VideoCodec::H264, VideoCodec::H265, VideoCodec::Vp9]
                .into_iter()
                .collect(),
            output_formats: [OutputFormat::Mp4, OutputFormat::Hls, OutputFormat::Dash]
                .into_iter()
                .collect(),
            max_input_size_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
            max_resolution: (4096, 2160),                   // 4K
            hardware_acceleration: true,
            hdr_support: true,
            drm_support: true,
            cost_per_minute_cents: 1.2,
            latency_multiplier: 0.6,
        }
    }

    fn azure_capabilities() -> Self {
        Self {
            video_codecs: [VideoCodec::H264, VideoCodec::H265]
                .into_iter()
                .collect(),
            output_formats: [OutputFormat::Mp4, OutputFormat::Hls, OutputFormat::Dash]
                .into_iter()
                .collect(),
            max_input_size_bytes: 200 * 1024 * 1024 * 1024, // 200 GB
            max_resolution: (4096, 2160),                    // 4K
            hardware_acceleration: true,
            hdr_support: true,
            drm_support: true,
            cost_per_minute_cents: 1.3,
            latency_multiplier: 0.7,
        }
    }

    fn local_capabilities() -> Self {
        Self {
            video_codecs: [VideoCodec::H264, VideoCodec::H265, VideoCodec::Vp9, VideoCodec::Av1]
                .into_iter()
                .collect(),
            output_formats: [
                OutputFormat::Mp4,
                OutputFormat::Hls,
                OutputFormat::Dash,
                OutputFormat::Webm,
                OutputFormat::Mkv,
                OutputFormat::Mov,
            ]
            .into_iter()
            .collect(),
            max_input_size_bytes: u64::MAX,
            max_resolution: (8192, 4320),
            hardware_acceleration: false,
            hdr_support: true,
            drm_support: false,
            cost_per_minute_cents: 0.0,
            latency_multiplier: 2.0,
        }
    }

    /// Check if provider can handle a job.
    pub fn can_handle(&self, job: &TranscodeJob) -> bool {
        let codec = job.preset.video.codec;
        let format = job.preset.format;

        self.video_codecs.contains(&codec) && self.output_formats.contains(&format)
    }

    /// Estimate cost for a job (in cents).
    pub fn estimate_cost(&self, duration_minutes: f64) -> f64 {
        duration_minutes * self.cost_per_minute_cents
    }

    /// Estimate processing time (in minutes).
    pub fn estimate_time(&self, input_duration_minutes: f64) -> f64 {
        input_duration_minutes * self.latency_multiplier
    }
}

/// A configured cloud provider.
#[derive(Debug, Clone)]
pub struct Provider {
    /// Provider configuration.
    pub config: ProviderConfig,
    /// Provider capabilities.
    pub capabilities: ProviderCapabilities,
    /// Provider health status.
    pub health: ProviderHealth,
}

impl Provider {
    /// Create a new provider from configuration.
    pub fn new(config: ProviderConfig) -> Self {
        let capabilities = ProviderCapabilities::for_provider(config.provider_type);
        Self {
            config,
            capabilities,
            health: ProviderHealth::default(),
        }
    }

    /// Create AWS provider.
    pub fn aws(region: impl Into<String>) -> Self {
        Self::new(ProviderConfig::aws(region))
    }

    /// Create GCP provider.
    pub fn gcp(region: impl Into<String>) -> Self {
        Self::new(ProviderConfig::gcp(region))
    }

    /// Create Azure provider.
    pub fn azure(region: impl Into<String>) -> Self {
        Self::new(ProviderConfig::azure(region))
    }

    /// Create local provider.
    pub fn local() -> Self {
        Self::new(ProviderConfig::local())
    }

    /// Get provider type.
    pub fn provider_type(&self) -> ProviderType {
        self.config.provider_type
    }

    /// Get provider region.
    pub fn region(&self) -> &str {
        &self.config.region
    }

    /// Get provider name.
    pub fn name(&self) -> String {
        format!("{}:{}", self.config.provider_type, self.config.region)
    }

    /// Check if provider is healthy.
    pub fn is_healthy(&self) -> bool {
        self.health.is_healthy()
    }

    /// Check if provider can handle a job.
    pub fn can_handle(&self, job: &TranscodeJob) -> bool {
        self.is_healthy() && self.capabilities.can_handle(job)
    }

    /// Estimate cost for a job.
    pub fn estimate_cost(&self, duration_minutes: f64) -> f64 {
        self.capabilities.estimate_cost(duration_minutes)
    }

    /// Estimate processing time.
    pub fn estimate_time(&self, input_duration_minutes: f64) -> f64 {
        self.capabilities.estimate_time(input_duration_minutes)
    }

    /// Submit a job to this provider.
    pub async fn submit(&self, job: &TranscodeJob) -> Result<String> {
        if !self.is_healthy() {
            return Err(MultiCloudError::ProviderError {
                provider: self.name(),
                message: "Provider is not healthy".into(),
            });
        }

        if !self.capabilities.can_handle(job) {
            return Err(MultiCloudError::ProviderError {
                provider: self.name(),
                message: "Provider cannot handle this job configuration".into(),
            });
        }

        // Simulate job submission
        tracing::info!(
            provider = %self.name(),
            job_id = %job.id,
            "Submitting job to provider"
        );

        // In a real implementation, this would call the provider's API
        let provider_job_id = format!("{}-{}", self.config.provider_type, uuid::Uuid::new_v4());
        Ok(provider_job_id)
    }
}

/// Provider health status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderHealth {
    /// Whether the provider is available.
    pub available: bool,
    /// Current active jobs.
    pub active_jobs: u32,
    /// Maximum concurrent jobs.
    pub max_jobs: u32,
    /// Recent error rate (0.0 - 1.0).
    pub error_rate: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Last health check timestamp.
    pub last_check: Option<chrono::DateTime<chrono::Utc>>,
}

impl ProviderHealth {
    /// Check if provider is considered healthy.
    pub fn is_healthy(&self) -> bool {
        self.available && self.error_rate < 0.5 && self.active_jobs < self.max_jobs
    }

    /// Create a healthy status.
    pub fn healthy() -> Self {
        Self {
            available: true,
            active_jobs: 0,
            max_jobs: 100,
            error_rate: 0.0,
            avg_latency_ms: 100.0,
            last_check: Some(chrono::Utc::now()),
        }
    }

    /// Create an unhealthy status.
    pub fn unhealthy(reason: &str) -> Self {
        tracing::warn!(reason = reason, "Provider marked unhealthy");
        Self {
            available: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = Provider::aws("us-east-1");
        assert_eq!(provider.provider_type(), ProviderType::Aws);
        assert_eq!(provider.region(), "us-east-1");
        assert_eq!(provider.name(), "AWS:us-east-1");
    }

    #[test]
    fn test_provider_capabilities() {
        let aws_caps = ProviderCapabilities::for_provider(ProviderType::Aws);
        assert!(aws_caps.video_codecs.contains(&VideoCodec::H264));
        assert!(aws_caps.video_codecs.contains(&VideoCodec::Av1));
        assert!(aws_caps.hardware_acceleration);

        let local_caps = ProviderCapabilities::for_provider(ProviderType::Local);
        assert!(!local_caps.hardware_acceleration);
        assert_eq!(local_caps.cost_per_minute_cents, 0.0);
    }

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::aws("us-west-2")
            .endpoint("https://custom.endpoint.com")
            .credentials(CredentialsConfig::InstanceProfile);

        assert_eq!(config.region, "us-west-2");
        assert_eq!(config.endpoint, Some("https://custom.endpoint.com".into()));
        assert!(matches!(config.credentials, CredentialsConfig::InstanceProfile));
    }

    #[test]
    fn test_provider_health() {
        let healthy = ProviderHealth::healthy();
        assert!(healthy.is_healthy());

        let unhealthy = ProviderHealth::unhealthy("test");
        assert!(!unhealthy.is_healthy());

        let overloaded = ProviderHealth {
            available: true,
            active_jobs: 100,
            max_jobs: 100,
            error_rate: 0.0,
            avg_latency_ms: 100.0,
            last_check: None,
        };
        assert!(!overloaded.is_healthy());
    }

    #[test]
    fn test_cost_estimation() {
        let provider = Provider::aws("us-east-1");
        let cost = provider.estimate_cost(10.0); // 10 minutes
        assert!(cost > 0.0);
    }
}
