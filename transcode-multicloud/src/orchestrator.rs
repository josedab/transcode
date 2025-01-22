//! Cloud orchestrator for job routing and management.

use crate::job::{JobResult, JobStats, JobStatus, JobTimestamps, TranscodeJob};
use crate::provider::{Provider, ProviderHealth, ProviderType};
use crate::{MultiCloudError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Routing strategy for job distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Route to the cheapest available provider.
    CostOptimized,
    /// Route to the fastest available provider.
    LatencyOptimized,
    /// Route in round-robin fashion.
    RoundRobin,
    /// Route based on provider load.
    LoadBalanced,
    /// Route to preferred provider, fallback to others.
    Preferred {
        #[serde(default)]
        provider: ProviderType,
    },
    /// Use custom routing logic.
    Custom,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::CostOptimized
    }
}

/// Orchestrator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Routing strategy.
    pub strategy: RoutingStrategy,
    /// Enable automatic failover.
    pub enable_failover: bool,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Retry delay base (exponential backoff).
    pub retry_delay_ms: u64,
    /// Health check interval.
    pub health_check_interval: Duration,
    /// Job timeout.
    pub job_timeout: Duration,
    /// Enable cost tracking.
    pub track_costs: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::CostOptimized,
            enable_failover: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            health_check_interval: Duration::from_secs(30),
            job_timeout: Duration::from_secs(3600),
            track_costs: true,
        }
    }
}

impl OrchestratorConfig {
    /// Create a new configuration builder.
    pub fn builder() -> OrchestratorConfigBuilder {
        OrchestratorConfigBuilder::default()
    }
}

/// Builder for OrchestratorConfig.
#[derive(Default)]
pub struct OrchestratorConfigBuilder {
    strategy: Option<RoutingStrategy>,
    enable_failover: Option<bool>,
    max_retries: Option<u32>,
    retry_delay_ms: Option<u64>,
    health_check_interval: Option<Duration>,
    job_timeout: Option<Duration>,
    track_costs: Option<bool>,
}

impl OrchestratorConfigBuilder {
    /// Set routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Enable/disable failover.
    pub fn failover(mut self, enable: bool) -> Self {
        self.enable_failover = Some(enable);
        self
    }

    /// Set max retries.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set retry delay.
    pub fn retry_delay(mut self, delay_ms: u64) -> Self {
        self.retry_delay_ms = Some(delay_ms);
        self
    }

    /// Set health check interval.
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = Some(interval);
        self
    }

    /// Set job timeout.
    pub fn job_timeout(mut self, timeout: Duration) -> Self {
        self.job_timeout = Some(timeout);
        self
    }

    /// Enable/disable cost tracking.
    pub fn track_costs(mut self, enable: bool) -> Self {
        self.track_costs = Some(enable);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> OrchestratorConfig {
        OrchestratorConfig {
            strategy: self.strategy.unwrap_or_default(),
            enable_failover: self.enable_failover.unwrap_or(true),
            max_retries: self.max_retries.unwrap_or(3),
            retry_delay_ms: self.retry_delay_ms.unwrap_or(1000),
            health_check_interval: self.health_check_interval.unwrap_or(Duration::from_secs(30)),
            job_timeout: self.job_timeout.unwrap_or(Duration::from_secs(3600)),
            track_costs: self.track_costs.unwrap_or(true),
        }
    }
}

/// Internal state for tracking jobs.
#[derive(Debug)]
struct JobState {
    job: TranscodeJob,
    status: JobStatus,
    provider: Option<String>,
    provider_job_id: Option<String>,
    attempts: u32,
    timestamps: JobTimestamps,
}

/// Cloud orchestrator for multi-provider job management.
pub struct CloudOrchestrator {
    config: OrchestratorConfig,
    providers: RwLock<Vec<Provider>>,
    jobs: RwLock<HashMap<String, JobState>>,
    round_robin_index: RwLock<usize>,
}

impl CloudOrchestrator {
    /// Create a new orchestrator builder.
    pub fn builder() -> CloudOrchestratorBuilder {
        CloudOrchestratorBuilder::default()
    }

    /// Create a new orchestrator with default config.
    pub fn new() -> Self {
        Self {
            config: OrchestratorConfig::default(),
            providers: RwLock::new(Vec::new()),
            jobs: RwLock::new(HashMap::new()),
            round_robin_index: RwLock::new(0),
        }
    }

    /// Create a new orchestrator with config.
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            config,
            providers: RwLock::new(Vec::new()),
            jobs: RwLock::new(HashMap::new()),
            round_robin_index: RwLock::new(0),
        }
    }

    /// Add a provider.
    pub fn add_provider(&self, provider: Provider) {
        let mut providers = self.providers.write();
        tracing::info!(
            provider = %provider.name(),
            "Adding provider to orchestrator"
        );
        providers.push(provider);
    }

    /// Remove a provider by type and region.
    pub fn remove_provider(&self, provider_type: ProviderType, region: &str) -> bool {
        let mut providers = self.providers.write();
        let initial_len = providers.len();
        providers.retain(|p| !(p.provider_type() == provider_type && p.region() == region));
        providers.len() < initial_len
    }

    /// Get all providers.
    pub fn providers(&self) -> Vec<Provider> {
        self.providers.read().clone()
    }

    /// Get healthy providers.
    pub fn healthy_providers(&self) -> Vec<Provider> {
        self.providers
            .read()
            .iter()
            .filter(|p| p.is_healthy())
            .cloned()
            .collect()
    }

    /// Update provider health.
    pub fn update_provider_health(&self, provider_type: ProviderType, region: &str, health: ProviderHealth) {
        let mut providers = self.providers.write();
        if let Some(provider) = providers
            .iter_mut()
            .find(|p| p.provider_type() == provider_type && p.region() == region)
        {
            provider.health = health;
        }
    }

    /// Submit a job for processing.
    pub async fn submit(&self, job: TranscodeJob) -> Result<String> {
        let job_id = job.id.clone();

        // Find a suitable provider
        let provider = self.select_provider(&job)?;

        tracing::info!(
            job_id = %job_id,
            provider = %provider.name(),
            strategy = ?self.config.strategy,
            "Selected provider for job"
        );

        // Submit to provider
        let provider_job_id = provider.submit(&job).await?;

        // Track job state
        let state = JobState {
            job,
            status: JobStatus::Submitted,
            provider: Some(provider.name()),
            provider_job_id: Some(provider_job_id),
            attempts: 1,
            timestamps: JobTimestamps {
                created_at: Utc::now(),
                submitted_at: Some(Utc::now()),
                started_at: None,
                completed_at: None,
            },
        };

        self.jobs.write().insert(job_id.clone(), state);

        Ok(job_id)
    }

    /// Get job status.
    pub fn get_status(&self, job_id: &str) -> Result<JobStatus> {
        let jobs = self.jobs.read();
        jobs.get(job_id)
            .map(|s| s.status.clone())
            .ok_or_else(|| MultiCloudError::JobNotFound(job_id.to_string()))
    }

    /// Get job result (for completed jobs).
    pub fn get_result(&self, job_id: &str) -> Result<JobResult> {
        let jobs = self.jobs.read();
        let state = jobs
            .get(job_id)
            .ok_or_else(|| MultiCloudError::JobNotFound(job_id.to_string()))?;

        if !state.status.is_terminal() {
            return Err(MultiCloudError::InvalidInput(
                "Job is not yet complete".into(),
            ));
        }

        Ok(JobResult {
            job_id: job_id.to_string(),
            status: state.status.clone(),
            provider: state.provider.clone().unwrap_or_default(),
            outputs: vec![state.job.output.uri.clone()],
            duration: state
                .timestamps
                .completed_at
                .and_then(|c| {
                    state
                        .timestamps
                        .started_at
                        .map(|s| (c - s).to_std().unwrap_or_default())
                })
                .unwrap_or_default(),
            cost_cents: None,
            stats: JobStats::default(),
            timestamps: state.timestamps.clone(),
        })
    }

    /// Cancel a job.
    pub async fn cancel(&self, job_id: &str) -> Result<()> {
        let mut jobs = self.jobs.write();
        let state = jobs
            .get_mut(job_id)
            .ok_or_else(|| MultiCloudError::JobNotFound(job_id.to_string()))?;

        if state.status.is_terminal() {
            return Err(MultiCloudError::InvalidInput(
                "Cannot cancel a completed job".into(),
            ));
        }

        state.status = JobStatus::Cancelled;
        state.timestamps.completed_at = Some(Utc::now());

        tracing::info!(job_id = %job_id, "Job cancelled");
        Ok(())
    }

    /// List all jobs.
    pub fn list_jobs(&self) -> Vec<(String, JobStatus)> {
        self.jobs
            .read()
            .iter()
            .map(|(id, state)| (id.clone(), state.status.clone()))
            .collect()
    }

    /// List active jobs.
    pub fn active_jobs(&self) -> Vec<String> {
        self.jobs
            .read()
            .iter()
            .filter(|(_, state)| state.status.is_active())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get orchestrator statistics.
    pub fn stats(&self) -> OrchestratorStats {
        let jobs = self.jobs.read();
        let providers = self.providers.read();

        let total_jobs = jobs.len();
        let active_jobs = jobs.values().filter(|s| s.status.is_active()).count();
        let completed_jobs = jobs
            .values()
            .filter(|s| matches!(s.status, JobStatus::Completed))
            .count();
        let failed_jobs = jobs
            .values()
            .filter(|s| matches!(s.status, JobStatus::Failed { .. }))
            .count();

        OrchestratorStats {
            total_providers: providers.len(),
            healthy_providers: providers.iter().filter(|p| p.is_healthy()).count(),
            total_jobs,
            active_jobs,
            completed_jobs,
            failed_jobs,
            total_cost_cents: 0, // Would be tracked in production
        }
    }

    /// Select a provider for a job based on routing strategy.
    fn select_provider(&self, job: &TranscodeJob) -> Result<Provider> {
        let providers = self.providers.read();
        let eligible: Vec<_> = providers.iter().filter(|p| p.can_handle(job)).collect();

        if eligible.is_empty() {
            return Err(MultiCloudError::NoProvidersAvailable);
        }

        let selected = match self.config.strategy {
            RoutingStrategy::CostOptimized => {
                // Select cheapest provider
                eligible
                    .iter()
                    .min_by(|a, b| {
                        a.capabilities
                            .cost_per_minute_cents
                            .partial_cmp(&b.capabilities.cost_per_minute_cents)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|p| (*p).clone())
            }
            RoutingStrategy::LatencyOptimized => {
                // Select fastest provider
                eligible
                    .iter()
                    .min_by(|a, b| {
                        a.capabilities
                            .latency_multiplier
                            .partial_cmp(&b.capabilities.latency_multiplier)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|p| (*p).clone())
            }
            RoutingStrategy::RoundRobin => {
                // Round-robin selection
                let mut index = self.round_robin_index.write();
                let provider = eligible[*index % eligible.len()].clone();
                *index = (*index + 1) % eligible.len();
                Some(provider)
            }
            RoutingStrategy::LoadBalanced => {
                // Select provider with lowest load
                eligible
                    .iter()
                    .min_by_key(|p| p.health.active_jobs)
                    .map(|p| (*p).clone())
            }
            RoutingStrategy::Preferred { provider } => {
                // Try preferred provider first
                eligible
                    .iter()
                    .find(|p| p.provider_type() == provider)
                    .or_else(|| eligible.first())
                    .map(|p| (*p).clone())
            }
            RoutingStrategy::Custom => {
                // Default to first available for custom
                eligible.first().map(|p| (*p).clone())
            }
        };

        selected.ok_or(MultiCloudError::NoProvidersAvailable)
    }
}

impl Default for CloudOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for CloudOrchestrator.
#[derive(Default)]
pub struct CloudOrchestratorBuilder {
    config: Option<OrchestratorConfig>,
    providers: Vec<Provider>,
}

impl CloudOrchestratorBuilder {
    /// Set configuration.
    pub fn config(mut self, config: OrchestratorConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        let config = self.config.get_or_insert_with(OrchestratorConfig::default);
        config.strategy = strategy;
        self
    }

    /// Add a provider.
    pub fn add_provider(mut self, provider: Provider) -> Self {
        self.providers.push(provider);
        self
    }

    /// Add AWS provider.
    pub fn aws(self, region: impl Into<String>) -> Self {
        self.add_provider(Provider::aws(region))
    }

    /// Add GCP provider.
    pub fn gcp(self, region: impl Into<String>) -> Self {
        self.add_provider(Provider::gcp(region))
    }

    /// Add Azure provider.
    pub fn azure(self, region: impl Into<String>) -> Self {
        self.add_provider(Provider::azure(region))
    }

    /// Add local provider.
    pub fn local(self) -> Self {
        self.add_provider(Provider::local())
    }

    /// Enable failover.
    pub fn with_failover(mut self) -> Self {
        let config = self.config.get_or_insert_with(OrchestratorConfig::default);
        config.enable_failover = true;
        self
    }

    /// Build the orchestrator.
    pub fn build(self) -> CloudOrchestrator {
        let orchestrator = CloudOrchestrator::with_config(self.config.unwrap_or_default());

        // Initialize providers with healthy status for simulation
        for mut provider in self.providers {
            provider.health = ProviderHealth::healthy();
            orchestrator.add_provider(provider);
        }

        orchestrator
    }
}

/// Orchestrator statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorStats {
    pub total_providers: usize,
    pub healthy_providers: usize,
    pub total_jobs: usize,
    pub active_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub total_cost_cents: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::TranscodeJob;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .strategy(RoutingStrategy::CostOptimized)
            .build();

        assert_eq!(orchestrator.providers().len(), 2);
    }

    #[test]
    fn test_provider_selection_cost() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .strategy(RoutingStrategy::CostOptimized)
            .build();

        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");
        let provider = orchestrator.select_provider(&job).unwrap();

        // GCP is cheaper in our mock capabilities
        assert_eq!(provider.provider_type(), ProviderType::Gcp);
    }

    #[test]
    fn test_provider_selection_latency() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .strategy(RoutingStrategy::LatencyOptimized)
            .build();

        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");
        let provider = orchestrator.select_provider(&job).unwrap();

        // AWS is faster in our mock capabilities
        assert_eq!(provider.provider_type(), ProviderType::Aws);
    }

    #[test]
    fn test_round_robin() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .strategy(RoutingStrategy::RoundRobin)
            .build();

        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");

        let first = orchestrator.select_provider(&job).unwrap();
        let second = orchestrator.select_provider(&job).unwrap();

        assert_ne!(first.provider_type(), second.provider_type());
    }

    #[test]
    fn test_no_providers() {
        let orchestrator = CloudOrchestrator::new();
        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");

        let result = orchestrator.select_provider(&job);
        assert!(matches!(result, Err(MultiCloudError::NoProvidersAvailable)));
    }

    #[tokio::test]
    async fn test_job_submission() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .build();

        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");
        let job_id = orchestrator.submit(job).await.unwrap();

        assert!(!job_id.is_empty());

        let status = orchestrator.get_status(&job_id).unwrap();
        assert!(matches!(status, JobStatus::Submitted));
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .build();

        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");
        let job_id = orchestrator.submit(job).await.unwrap();

        orchestrator.cancel(&job_id).await.unwrap();

        let status = orchestrator.get_status(&job_id).unwrap();
        assert!(matches!(status, JobStatus::Cancelled));
    }

    #[test]
    fn test_stats() {
        let orchestrator = CloudOrchestrator::builder()
            .aws("us-east-1")
            .gcp("us-central1")
            .build();

        let stats = orchestrator.stats();
        assert_eq!(stats.total_providers, 2);
        assert_eq!(stats.healthy_providers, 2);
        assert_eq!(stats.total_jobs, 0);
    }
}
