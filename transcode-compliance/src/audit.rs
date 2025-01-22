//! Audit logging and compliance reporting.

use crate::regulation::Regulation;
use crate::{ComplianceError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Audit event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// Job started.
    JobStarted,
    /// Job completed.
    JobCompleted,
    /// Job failed.
    JobFailed,
    /// Compliance check performed.
    ComplianceCheck,
    /// Compliance violation detected.
    ComplianceViolation,
    /// Data access.
    DataAccess,
    /// Data modification.
    DataModification,
    /// Data deletion.
    DataDeletion,
    /// Configuration change.
    ConfigChange,
    /// User action.
    UserAction,
    /// System event.
    SystemEvent,
}

/// An audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID.
    pub id: String,
    /// Event type.
    pub event_type: AuditEventType,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Actor (user or system component).
    pub actor: String,
    /// Resource being acted upon.
    pub resource: String,
    /// Action performed.
    pub action: String,
    /// Outcome (success/failure).
    pub outcome: AuditOutcome,
    /// Additional details.
    pub details: HashMap<String, String>,
    /// Associated job ID.
    pub job_id: Option<String>,
    /// Client IP address.
    pub ip_address: Option<String>,
    /// User agent.
    pub user_agent: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry.
    pub fn new(
        event_type: AuditEventType,
        actor: impl Into<String>,
        resource: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            actor: actor.into(),
            resource: resource.into(),
            action: action.into(),
            outcome: AuditOutcome::Success,
            details: HashMap::new(),
            job_id: None,
            ip_address: None,
            user_agent: None,
        }
    }

    /// Set outcome.
    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Add detail.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Set job ID.
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set IP address.
    pub fn with_ip(mut self, ip: impl Into<String>) -> Self {
        self.ip_address = Some(ip.into());
        self
    }

    /// Set user agent.
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }
}

/// Audit outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    Success,
    Failure,
    Partial,
    Denied,
}

/// Audit logger for compliance tracking.
#[derive(Clone)]
pub struct AuditLogger {
    inner: Arc<AuditLoggerInner>,
}

struct AuditLoggerInner {
    entries: RwLock<Vec<AuditEntry>>,
    max_entries: usize,
    retention_days: u32,
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AuditLoggerInner {
                entries: RwLock::new(Vec::new()),
                max_entries: 100_000,
                retention_days: 90,
            }),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(max_entries: usize, retention_days: u32) -> Self {
        Self {
            inner: Arc::new(AuditLoggerInner {
                entries: RwLock::new(Vec::new()),
                max_entries,
                retention_days,
            }),
        }
    }

    /// Log an entry.
    pub fn log(&self, entry: AuditEntry) {
        let mut entries = self.inner.entries.write();

        tracing::info!(
            event_type = ?entry.event_type,
            actor = %entry.actor,
            resource = %entry.resource,
            action = %entry.action,
            outcome = ?entry.outcome,
            "Audit log entry"
        );

        entries.push(entry);

        // Trim old entries if needed
        if entries.len() > self.inner.max_entries {
            let drain_count = entries.len() - self.inner.max_entries;
            entries.drain(0..drain_count);
        }
    }

    /// Log a job start event.
    pub fn log_job_started(&self, job_id: &str, actor: &str, input: &str) {
        self.log(
            AuditEntry::new(
                AuditEventType::JobStarted,
                actor,
                format!("job:{}", job_id),
                "start_job",
            )
            .with_job_id(job_id)
            .with_detail("input", input),
        );
    }

    /// Log a job completion event.
    pub fn log_job_completed(&self, job_id: &str, actor: &str, output: &str, duration_secs: f64) {
        self.log(
            AuditEntry::new(
                AuditEventType::JobCompleted,
                actor,
                format!("job:{}", job_id),
                "complete_job",
            )
            .with_job_id(job_id)
            .with_detail("output", output)
            .with_detail("duration_secs", duration_secs.to_string()),
        );
    }

    /// Log a job failure event.
    pub fn log_job_failed(&self, job_id: &str, actor: &str, error: &str) {
        self.log(
            AuditEntry::new(
                AuditEventType::JobFailed,
                actor,
                format!("job:{}", job_id),
                "job_failed",
            )
            .with_outcome(AuditOutcome::Failure)
            .with_job_id(job_id)
            .with_detail("error", error),
        );
    }

    /// Log a compliance check.
    pub fn log_compliance_check(&self, job_id: &str, regulations: &[Regulation], passed: bool) {
        let reg_names: Vec<_> = regulations.iter().map(|r| r.name()).collect();

        self.log(
            AuditEntry::new(
                AuditEventType::ComplianceCheck,
                "system",
                format!("job:{}", job_id),
                "compliance_check",
            )
            .with_outcome(if passed {
                AuditOutcome::Success
            } else {
                AuditOutcome::Failure
            })
            .with_job_id(job_id)
            .with_detail("regulations", reg_names.join(", "))
            .with_detail("passed", passed.to_string()),
        );
    }

    /// Log a compliance violation.
    pub fn log_violation(
        &self,
        job_id: &str,
        rule_id: &str,
        regulation: Regulation,
        message: &str,
    ) {
        self.log(
            AuditEntry::new(
                AuditEventType::ComplianceViolation,
                "system",
                format!("job:{}", job_id),
                "violation_detected",
            )
            .with_outcome(AuditOutcome::Failure)
            .with_job_id(job_id)
            .with_detail("rule_id", rule_id)
            .with_detail("regulation", regulation.name())
            .with_detail("message", message),
        );
    }

    /// Log data access.
    pub fn log_data_access(&self, actor: &str, resource: &str, access_type: &str) {
        self.log(
            AuditEntry::new(AuditEventType::DataAccess, actor, resource, access_type)
                .with_detail("access_type", access_type),
        );
    }

    /// Get all entries.
    pub fn entries(&self) -> Vec<AuditEntry> {
        self.inner.entries.read().clone()
    }

    /// Get entries by event type.
    pub fn entries_by_type(&self, event_type: AuditEventType) -> Vec<AuditEntry> {
        self.inner
            .entries
            .read()
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get entries for a job.
    pub fn entries_for_job(&self, job_id: &str) -> Vec<AuditEntry> {
        self.inner
            .entries
            .read()
            .iter()
            .filter(|e| e.job_id.as_deref() == Some(job_id))
            .cloned()
            .collect()
    }

    /// Get entries in a time range.
    pub fn entries_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AuditEntry> {
        self.inner
            .entries
            .read()
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Export entries as JSON.
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.entries())
            .map_err(|e| ComplianceError::EncodingError(e.to_string()))
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.inner.entries.write().clear();
    }

    /// Get entry count.
    pub fn count(&self) -> usize {
        self.inner.entries.read().len()
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Compliance report generator.
pub struct ComplianceReporter {
    audit_logger: AuditLogger,
}

impl ComplianceReporter {
    /// Create a new reporter.
    pub fn new(audit_logger: AuditLogger) -> Self {
        Self { audit_logger }
    }

    /// Generate a compliance report for a time period.
    pub fn generate_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        regulations: &[Regulation],
    ) -> ComplianceReport {
        let entries = self.audit_logger.entries_in_range(start, end);

        let total_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobStarted)
            .count();

        let completed_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobCompleted)
            .count();

        let failed_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobFailed)
            .count();

        let compliance_checks = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::ComplianceCheck)
            .count();

        let violations = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::ComplianceViolation)
            .count();

        let compliance_rate = if compliance_checks > 0 {
            1.0 - (violations as f64 / compliance_checks as f64)
        } else {
            1.0
        };

        ComplianceReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            generated_at: Utc::now(),
            period_start: start,
            period_end: end,
            regulations: regulations.to_vec(),
            summary: ReportSummary {
                total_jobs,
                completed_jobs,
                failed_jobs,
                compliance_checks,
                violations,
                compliance_rate,
            },
            entries,
        }
    }

    /// Generate a summary report.
    pub fn generate_summary(&self, _regulations: &[Regulation]) -> ReportSummary {
        let entries = self.audit_logger.entries();

        let total_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobStarted)
            .count();

        let completed_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobCompleted)
            .count();

        let failed_jobs = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::JobFailed)
            .count();

        let compliance_checks = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::ComplianceCheck)
            .count();

        let violations = entries
            .iter()
            .filter(|e| e.event_type == AuditEventType::ComplianceViolation)
            .count();

        let compliance_rate = if compliance_checks > 0 {
            1.0 - (violations as f64 / compliance_checks as f64)
        } else {
            1.0
        };

        ReportSummary {
            total_jobs,
            completed_jobs,
            failed_jobs,
            compliance_checks,
            violations,
            compliance_rate,
        }
    }
}

/// Full compliance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report ID.
    pub report_id: String,
    /// Generation timestamp.
    pub generated_at: DateTime<Utc>,
    /// Period start.
    pub period_start: DateTime<Utc>,
    /// Period end.
    pub period_end: DateTime<Utc>,
    /// Regulations covered.
    pub regulations: Vec<Regulation>,
    /// Summary statistics.
    pub summary: ReportSummary,
    /// Detailed entries.
    pub entries: Vec<AuditEntry>,
}

/// Report summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total jobs processed.
    pub total_jobs: usize,
    /// Successfully completed jobs.
    pub completed_jobs: usize,
    /// Failed jobs.
    pub failed_jobs: usize,
    /// Number of compliance checks.
    pub compliance_checks: usize,
    /// Number of violations.
    pub violations: usize,
    /// Overall compliance rate (0.0 - 1.0).
    pub compliance_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(
            AuditEventType::JobStarted,
            "user@example.com",
            "job:123",
            "start_job",
        )
        .with_detail("input", "s3://bucket/video.mp4")
        .with_job_id("123");

        assert_eq!(entry.event_type, AuditEventType::JobStarted);
        assert_eq!(entry.actor, "user@example.com");
        assert_eq!(entry.job_id, Some("123".to_string()));
    }

    #[test]
    fn test_audit_logger() {
        let logger = AuditLogger::new();

        logger.log_job_started("job-1", "user", "input.mp4");
        logger.log_job_completed("job-1", "user", "output.mp4", 120.5);

        assert_eq!(logger.count(), 2);

        let job_entries = logger.entries_for_job("job-1");
        assert_eq!(job_entries.len(), 2);
    }

    #[test]
    fn test_compliance_logging() {
        let logger = AuditLogger::new();

        logger.log_compliance_check("job-1", &[Regulation::Gdpr], true);
        logger.log_violation("job-2", "GDPR-001", Regulation::Gdpr, "Missing consent");

        let checks = logger.entries_by_type(AuditEventType::ComplianceCheck);
        assert_eq!(checks.len(), 1);

        let violations = logger.entries_by_type(AuditEventType::ComplianceViolation);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_report_generation() {
        let logger = AuditLogger::new();

        logger.log_job_started("job-1", "user", "input.mp4");
        logger.log_compliance_check("job-1", &[Regulation::Gdpr], true);
        logger.log_job_completed("job-1", "user", "output.mp4", 60.0);

        let reporter = ComplianceReporter::new(logger);
        let summary = reporter.generate_summary(&[Regulation::Gdpr]);

        assert_eq!(summary.total_jobs, 1);
        assert_eq!(summary.completed_jobs, 1);
        assert_eq!(summary.compliance_checks, 1);
        assert_eq!(summary.compliance_rate, 1.0);
    }

    #[test]
    fn test_entry_limit() {
        let logger = AuditLogger::with_config(10, 30);

        for i in 0..20 {
            logger.log_job_started(&format!("job-{}", i), "user", "input.mp4");
        }

        assert_eq!(logger.count(), 10);
    }

    #[test]
    fn test_json_export() {
        let logger = AuditLogger::new();
        logger.log_job_started("job-1", "user", "input.mp4");

        let json = logger.export_json().unwrap();
        assert!(json.contains("job-1"));
        // Event type is serialized as snake_case per serde rename_all
        assert!(json.contains("job_started"));
    }
}
