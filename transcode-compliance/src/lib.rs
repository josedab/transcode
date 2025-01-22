//! Compliance automation suite for video transcoding workflows.
//!
//! This crate provides comprehensive compliance tools including:
//!
//! - **Caption Processing**: Parse, convert, and validate caption formats
//! - **Accessibility Checking**: Validate against WCAG, FCC, and other standards
//! - **Regulatory Compliance**: GDPR, CCPA, COPPA, FCC validation
//! - **Audit Logging**: Complete audit trail for compliance reporting
//!
//! # Example
//!
//! ```ignore
//! use transcode_compliance::{
//!     AccessibilityChecker, AccessibilityStandard,
//!     ComplianceValidator, Regulation,
//!     CaptionTrack, CaptionFormat, CaptionCue,
//!     AuditLogger,
//! };
//! use std::time::Duration;
//!
//! // Create caption track
//! let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);
//! track.add_cue(CaptionCue::new(
//!     Duration::from_secs(0),
//!     Duration::from_secs(3),
//!     "Hello, welcome to our video.",
//! ));
//!
//! // Check accessibility compliance
//! let checker = AccessibilityChecker::wcag21_aa();
//! let report = checker.check_captions(&track);
//!
//! if report.passed {
//!     println!("Captions meet WCAG 2.1 AA requirements");
//! } else {
//!     for issue in &report.issues {
//!         println!("Issue: {} - {}", issue.code, issue.message);
//!     }
//! }
//!
//! // Validate regulatory compliance
//! let validator = ComplianceValidator::gdpr();
//! let context = ValidationContext::new()
//!     .with("consent_obtained", "true")
//!     .with("retention_period_days", "30")
//!     .with("processing_purpose", "transcoding");
//!
//! let result = validator.validate(&context);
//! assert!(result.passed);
//!
//! // Audit logging
//! let logger = AuditLogger::new();
//! logger.log_job_started("job-123", "user@example.com", "input.mp4");
//! logger.log_compliance_check("job-123", &[Regulation::Gdpr], true);
//! logger.log_job_completed("job-123", "user@example.com", "output.mp4", 120.5);
//! ```
//!
//! # Caption Formats
//!
//! The crate supports multiple caption formats:
//!
//! - WebVTT
//! - SRT (SubRip)
//! - CEA-608/708
//! - TTML/IMSC
//! - EBU-TT
//!
//! # Accessibility Standards
//!
//! - WCAG 2.0/2.1 (A, AA, AAA)
//! - FCC Closed Captioning Requirements
//! - Section 508
//! - CVAA
//! - EN 301 549
//!
//! # Regulatory Frameworks
//!
//! - GDPR (EU)
//! - CCPA (California)
//! - COPPA (Children's Privacy)
//! - HIPAA (Healthcare)
//! - FCC/EBU/ATSC/DVB Broadcast Standards

#![allow(dead_code)]

mod accessibility;
mod audit;
mod captions;
mod error;
mod regulation;

pub use accessibility::{
    AccessibilityChecker, AccessibilityIssue, AccessibilityReport, AccessibilityStandard,
    IssueSeverity, VideoAccessibilityRequirements,
};
pub use audit::{
    AuditEntry, AuditEventType, AuditLogger, AuditOutcome, ComplianceReport, ComplianceReporter,
    ReportSummary,
};
pub use captions::{
    CaptionAlign, CaptionConverter, CaptionCue, CaptionFormat, CaptionPosition, CaptionStyle,
    CaptionTrack, CaptionType, TextDirection,
};
pub use error::{ComplianceError, Result};
pub use regulation::{
    ComplianceRule, ComplianceValidator, ConditionOperator, Regulation, RuleCategory,
    RuleCondition, RuleSeverity, ValidationContext, ValidationResult, Violation,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_end_to_end_compliance() {
        // Create a caption track
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);
        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(3),
            "Welcome to our video presentation.",
        ));
        track.add_cue(CaptionCue::new(
            Duration::from_secs(3),
            Duration::from_secs(6),
            "Today we will discuss compliance.",
        ));

        // Check accessibility
        let checker = AccessibilityChecker::wcag21_aa();
        let accessibility_report = checker.check_captions(&track);
        assert!(accessibility_report.passed);

        // Check regulatory compliance
        let validator = ComplianceValidator::fcc();
        let context = ValidationContext::new()
            .with("caption_accuracy", "0.995")
            .with("caption_sync_offset_ms", "50")
            .with("content_rating", "TV-PG");

        let validation_result = validator.validate(&context);
        assert!(validation_result.passed);

        // Log the compliance check
        let logger = AuditLogger::new();
        logger.log_compliance_check("test-job", &[Regulation::Fcc], validation_result.passed);

        // Generate report
        let reporter = ComplianceReporter::new(logger);
        let summary = reporter.generate_summary(&[Regulation::Fcc]);
        assert_eq!(summary.compliance_checks, 1);
        assert_eq!(summary.compliance_rate, 1.0);
    }

    #[test]
    fn test_caption_conversion() {
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);
        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(2),
            "Hello",
        ));

        // Convert to different formats
        let vtt = CaptionConverter::convert(&track, CaptionFormat::WebVtt).unwrap();
        assert!(vtt.contains("WEBVTT"));

        let srt = CaptionConverter::convert(&track, CaptionFormat::Srt).unwrap();
        assert!(srt.contains("1"));
        assert!(srt.contains("Hello"));
    }
}
