//! Health monitoring with circuit breaker pattern and cost estimation.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Circuit Breaker
// ---------------------------------------------------------------------------

/// State of a circuit breaker.
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Normal operation – requests flow through.
    Closed,
    /// Requests are blocked until `since + open_duration` elapses.
    Open { since: Instant },
    /// A limited number of probe requests are allowed.
    HalfOpen { since: Instant },
}

/// Circuit breaker that trips after repeated failures.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
    open_duration: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: u32, success_threshold: u32, open_duration: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
            open_duration,
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen { .. } => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Open { .. } => {}
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open {
                        since: Instant::now(),
                    };
                }
            }
            CircuitState::HalfOpen { .. } => {
                self.state = CircuitState::Open {
                    since: Instant::now(),
                };
                self.success_count = 0;
            }
            CircuitState::Open { .. } => {}
        }
    }

    /// Returns `true` when the breaker allows requests.
    pub fn is_available(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open { since } => {
                if since.elapsed() >= self.open_duration {
                    self.state = CircuitState::HalfOpen {
                        since: Instant::now(),
                    };
                    self.success_count = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen { .. } => true,
        }
    }

    /// Current state.
    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    /// Reset to initial closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 3, Duration::from_secs(30))
    }
}

// ---------------------------------------------------------------------------
// Health Probe
// ---------------------------------------------------------------------------

/// Per-provider health probe that tracks latency, failures, and circuit state.
#[derive(Debug, Clone)]
pub struct HealthProbe {
    pub provider_id: String,
    pub last_check: Option<Instant>,
    pub last_latency_ms: Option<u64>,
    pub consecutive_failures: u32,
    pub total_checks: u64,
    pub total_failures: u64,
    total_latency_ms: u64,
    pub circuit_breaker: CircuitBreaker,
}

impl HealthProbe {
    /// Create a new probe for the given provider.
    pub fn new(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            last_check: None,
            last_latency_ms: None,
            consecutive_failures: 0,
            total_checks: 0,
            total_failures: 0,
            total_latency_ms: 0,
            circuit_breaker: CircuitBreaker::default(),
        }
    }

    /// Record the result of a health check.
    pub fn record_check(&mut self, success: bool, latency_ms: u64) {
        self.last_check = Some(Instant::now());
        self.last_latency_ms = Some(latency_ms);
        self.total_checks += 1;
        self.total_latency_ms += latency_ms;

        if success {
            self.consecutive_failures = 0;
            self.circuit_breaker.record_success();
        } else {
            self.consecutive_failures += 1;
            self.total_failures += 1;
            self.circuit_breaker.record_failure();
        }
    }

    /// Whether the provider is considered healthy (delegates to circuit breaker).
    pub fn is_healthy(&mut self) -> bool {
        self.circuit_breaker.is_available()
    }

    /// Availability as a percentage (0.0–100.0).
    pub fn availability_percent(&self) -> f64 {
        if self.total_checks == 0 {
            return 100.0;
        }
        let successes = self.total_checks - self.total_failures;
        (successes as f64 / self.total_checks as f64) * 100.0
    }

    /// Average observed latency, or `None` if no checks recorded.
    pub fn avg_latency_ms(&self) -> Option<u64> {
        if self.total_checks == 0 {
            return None;
        }
        Some(self.total_latency_ms / self.total_checks)
    }
}

// ---------------------------------------------------------------------------
// Health Monitor
// ---------------------------------------------------------------------------

/// Snapshot of a single provider's health status.
#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub provider_id: String,
    pub healthy: bool,
    pub availability: f64,
    pub latency_ms: Option<u64>,
    pub circuit_state: String,
}

/// Aggregates health probes for all registered providers.
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    probes: HashMap<String, HealthProbe>,
    pub check_interval: Duration,
}

impl HealthMonitor {
    /// Create a new health monitor with the given check interval.
    pub fn new(check_interval: Duration) -> Self {
        Self {
            probes: HashMap::new(),
            check_interval,
        }
    }

    /// Register a provider for monitoring.
    pub fn register_provider(&mut self, provider_id: &str) {
        self.probes
            .entry(provider_id.to_string())
            .or_insert_with(|| HealthProbe::new(provider_id));
    }

    /// Record a check result for a provider.
    pub fn record_result(&mut self, provider_id: &str, success: bool, latency_ms: u64) {
        if let Some(probe) = self.probes.get_mut(provider_id) {
            probe.record_check(success, latency_ms);
        }
    }

    /// Return ids of all currently healthy providers.
    pub fn healthy_providers(&mut self) -> Vec<&str> {
        // First pass: trigger any pending state transitions.
        let keys: Vec<String> = self.probes.keys().cloned().collect();
        for key in &keys {
            if let Some(probe) = self.probes.get_mut(key) {
                let _ = probe.is_healthy();
            }
        }
        // Second pass: collect healthy providers immutably.
        self.probes
            .iter()
            .filter(|(_, p)| {
                matches!(
                    p.circuit_breaker.state(),
                    CircuitState::Closed | CircuitState::HalfOpen { .. }
                )
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get the status snapshot for a single provider.
    pub fn provider_status(&mut self, provider_id: &str) -> Option<ProviderStatus> {
        self.probes.get_mut(provider_id).map(|probe| {
            let healthy = probe.is_healthy();
            ProviderStatus {
                provider_id: probe.provider_id.clone(),
                healthy,
                availability: probe.availability_percent(),
                latency_ms: probe.avg_latency_ms(),
                circuit_state: circuit_state_label(probe.circuit_breaker.state()),
            }
        })
    }

    /// Status snapshots for every registered provider.
    pub fn all_status(&mut self) -> Vec<ProviderStatus> {
        self.probes
            .values_mut()
            .map(|probe| {
                let healthy = probe.is_healthy();
                ProviderStatus {
                    provider_id: probe.provider_id.clone(),
                    healthy,
                    availability: probe.availability_percent(),
                    latency_ms: probe.avg_latency_ms(),
                    circuit_state: circuit_state_label(probe.circuit_breaker.state()),
                }
            })
            .collect()
    }
}

fn circuit_state_label(state: &CircuitState) -> String {
    match state {
        CircuitState::Closed => "closed".to_string(),
        CircuitState::Open { .. } => "open".to_string(),
        CircuitState::HalfOpen { .. } => "half_open".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Cost Estimator
// ---------------------------------------------------------------------------

/// Pricing model for a single provider.
#[derive(Debug, Clone)]
pub struct ProviderPricing {
    pub per_minute_video: f64,
    pub per_gb_storage: f64,
    pub per_gb_transfer: f64,
}

/// Itemised cost breakdown.
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub compute: f64,
    pub storage: f64,
    pub transfer: f64,
}

/// A cost estimate for a single provider.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub provider_id: String,
    pub total_cost: f64,
    pub breakdown: CostBreakdown,
}

/// Compares costs across providers.
#[derive(Debug, Clone, Default)]
pub struct CostEstimator {
    pricing: HashMap<String, ProviderPricing>,
}

impl CostEstimator {
    /// Create an empty estimator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set pricing for a provider.
    pub fn set_pricing(&mut self, provider_id: &str, pricing: ProviderPricing) {
        self.pricing.insert(provider_id.to_string(), pricing);
    }

    /// Estimate total cost for a provider.
    pub fn estimate_cost(
        &self,
        provider_id: &str,
        duration_minutes: f64,
        storage_gb: f64,
        transfer_gb: f64,
    ) -> Option<f64> {
        self.pricing.get(provider_id).map(|p| {
            p.per_minute_video * duration_minutes
                + p.per_gb_storage * storage_gb
                + p.per_gb_transfer * transfer_gb
        })
    }

    /// Return the cheapest provider and its cost.
    pub fn cheapest_provider(
        &self,
        duration_minutes: f64,
        storage_gb: f64,
        transfer_gb: f64,
    ) -> Option<(String, f64)> {
        self.pricing
            .iter()
            .map(|(id, p)| {
                let cost = p.per_minute_video * duration_minutes
                    + p.per_gb_storage * storage_gb
                    + p.per_gb_transfer * transfer_gb;
                (id.clone(), cost)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compare costs across all providers, sorted cheapest-first.
    pub fn compare_all(
        &self,
        duration_minutes: f64,
        storage_gb: f64,
        transfer_gb: f64,
    ) -> Vec<CostEstimate> {
        let mut estimates: Vec<CostEstimate> = self
            .pricing
            .iter()
            .map(|(id, p)| {
                let compute = p.per_minute_video * duration_minutes;
                let storage = p.per_gb_storage * storage_gb;
                let transfer = p.per_gb_transfer * transfer_gb;
                CostEstimate {
                    provider_id: id.clone(),
                    total_cost: compute + storage + transfer,
                    breakdown: CostBreakdown {
                        compute,
                        storage,
                        transfer,
                    },
                }
            })
            .collect();
        estimates.sort_by(|a, b| {
            a.total_cost
                .partial_cmp(&b.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        estimates
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Circuit Breaker tests --

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::default();
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 2, Duration::from_secs(30));
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
        assert!(!cb.is_available());
    }

    #[test]
    fn test_circuit_breaker_success_resets_failures() {
        let mut cb = CircuitBreaker::new(3, 2, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // resets failure_count
        cb.record_failure();
        // Only 1 failure since last success → still closed
        assert!(matches!(cb.state(), CircuitState::Closed));
        assert!(cb.is_available());
    }

    #[test]
    fn test_circuit_breaker_half_open_transitions() {
        let mut cb = CircuitBreaker::new(2, 2, Duration::from_millis(0));
        cb.record_failure();
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
        // Duration is 0 ms so is_available should transition to HalfOpen
        assert!(cb.is_available());
        assert!(matches!(cb.state(), CircuitState::HalfOpen { .. }));
        // Successes in half-open close the breaker
        cb.record_success();
        cb.record_success();
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let mut cb = CircuitBreaker::new(2, 2, Duration::from_millis(0));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_available()); // transitions to half-open
        cb.record_failure(); // should reopen
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(2, 2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
        cb.reset();
        assert!(matches!(cb.state(), CircuitState::Closed));
        assert!(cb.is_available());
    }

    // -- HealthProbe tests --

    #[test]
    fn test_health_probe_initial_state() {
        let mut probe = HealthProbe::new("aws");
        assert!(probe.is_healthy());
        assert_eq!(probe.availability_percent(), 100.0);
        assert_eq!(probe.avg_latency_ms(), None);
    }

    #[test]
    fn test_health_probe_records_checks() {
        let mut probe = HealthProbe::new("gcp");
        probe.record_check(true, 50);
        probe.record_check(true, 100);
        probe.record_check(false, 200);
        assert_eq!(probe.total_checks, 3);
        assert_eq!(probe.total_failures, 1);
        assert_eq!(probe.consecutive_failures, 1);
        // avg latency = (50+100+200)/3 = 116
        assert_eq!(probe.avg_latency_ms(), Some(116));
        // availability = 2/3 ≈ 66.67%
        let avail = probe.availability_percent();
        assert!((avail - 66.666).abs() < 1.0);
    }

    // -- HealthMonitor tests --

    #[test]
    fn test_health_monitor_register_and_status() {
        let mut mon = HealthMonitor::new(Duration::from_secs(10));
        mon.register_provider("aws");
        mon.register_provider("gcp");
        mon.record_result("aws", true, 42);
        let status = mon.provider_status("aws").unwrap();
        assert!(status.healthy);
        assert_eq!(status.circuit_state, "closed");
        assert_eq!(status.latency_ms, Some(42));
    }

    #[test]
    fn test_health_monitor_healthy_providers() {
        let mut mon = HealthMonitor::new(Duration::from_secs(10));
        mon.register_provider("aws");
        mon.register_provider("gcp");
        // Trip gcp circuit breaker (default threshold = 5)
        for _ in 0..5 {
            mon.record_result("gcp", false, 500);
        }
        let healthy = mon.healthy_providers();
        assert!(healthy.contains(&"aws"));
        assert!(!healthy.contains(&"gcp"));
    }

    #[test]
    fn test_health_monitor_all_status() {
        let mut mon = HealthMonitor::new(Duration::from_secs(10));
        mon.register_provider("a");
        mon.register_provider("b");
        let statuses = mon.all_status();
        assert_eq!(statuses.len(), 2);
    }

    // -- CostEstimator tests --

    #[test]
    fn test_cost_estimator_basic() {
        let mut est = CostEstimator::new();
        est.set_pricing(
            "aws",
            ProviderPricing {
                per_minute_video: 0.10,
                per_gb_storage: 0.02,
                per_gb_transfer: 0.05,
            },
        );
        let cost = est.estimate_cost("aws", 10.0, 5.0, 2.0).unwrap();
        // 10*0.10 + 5*0.02 + 2*0.05 = 1.0 + 0.1 + 0.1 = 1.2
        assert!((cost - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_cost_estimator_cheapest() {
        let mut est = CostEstimator::new();
        est.set_pricing(
            "aws",
            ProviderPricing {
                per_minute_video: 0.10,
                per_gb_storage: 0.02,
                per_gb_transfer: 0.05,
            },
        );
        est.set_pricing(
            "gcp",
            ProviderPricing {
                per_minute_video: 0.08,
                per_gb_storage: 0.01,
                per_gb_transfer: 0.04,
            },
        );
        let (id, _cost) = est.cheapest_provider(10.0, 5.0, 2.0).unwrap();
        assert_eq!(id, "gcp");
    }

    #[test]
    fn test_cost_estimator_compare_all() {
        let mut est = CostEstimator::new();
        est.set_pricing(
            "aws",
            ProviderPricing {
                per_minute_video: 0.10,
                per_gb_storage: 0.02,
                per_gb_transfer: 0.05,
            },
        );
        est.set_pricing(
            "gcp",
            ProviderPricing {
                per_minute_video: 0.08,
                per_gb_storage: 0.01,
                per_gb_transfer: 0.04,
            },
        );
        let all = est.compare_all(10.0, 5.0, 2.0);
        assert_eq!(all.len(), 2);
        // Sorted cheapest first
        assert!(all[0].total_cost <= all[1].total_cost);
        assert_eq!(all[0].provider_id, "gcp");
    }
}
