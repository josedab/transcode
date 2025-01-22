//! Accessibility compliance checking and validation.

use crate::captions::CaptionTrack;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Accessibility standard to check against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessibilityStandard {
    /// WCAG 2.0 Level A.
    Wcag20A,
    /// WCAG 2.0 Level AA.
    Wcag20Aa,
    /// WCAG 2.0 Level AAA.
    Wcag20Aaa,
    /// WCAG 2.1 Level A.
    Wcag21A,
    /// WCAG 2.1 Level AA.
    Wcag21Aa,
    /// WCAG 2.1 Level AAA.
    Wcag21Aaa,
    /// Section 508 (US federal).
    Section508,
    /// FCC closed captioning requirements.
    FccCaptions,
    /// CVAA (21st Century Communications).
    Cvaa,
    /// EN 301 549 (EU).
    En301549,
}

impl AccessibilityStandard {
    /// Get the caption accuracy threshold for this standard.
    pub fn caption_accuracy_threshold(&self) -> f64 {
        match self {
            Self::FccCaptions | Self::Cvaa => 0.99,
            Self::Section508 | Self::En301549 => 0.98,
            Self::Wcag20Aaa | Self::Wcag21Aaa => 0.99,
            Self::Wcag20Aa | Self::Wcag21Aa => 0.95,
            Self::Wcag20A | Self::Wcag21A => 0.90,
        }
    }

    /// Get the maximum sync offset allowed (in milliseconds).
    pub fn max_sync_offset_ms(&self) -> i64 {
        match self {
            Self::FccCaptions | Self::Cvaa => 100,
            Self::Section508 | Self::En301549 => 200,
            _ => 500,
        }
    }

    /// Get the minimum caption display time (in seconds).
    pub fn min_display_time_secs(&self) -> f64 {
        match self {
            Self::Wcag21Aaa => 2.0,
            Self::Wcag21Aa | Self::Wcag20Aaa => 1.5,
            _ => 1.0,
        }
    }

    /// Get the maximum characters per second for readability.
    pub fn max_chars_per_second(&self) -> f64 {
        match self {
            Self::Wcag21Aaa => 15.0,
            Self::Wcag21Aa | Self::Wcag20Aaa => 20.0,
            _ => 25.0,
        }
    }

    /// Get the maximum words per minute.
    pub fn max_words_per_minute(&self) -> f64 {
        match self {
            Self::Wcag21Aaa => 140.0,
            Self::Wcag21Aa | Self::Wcag20Aaa => 160.0,
            _ => 180.0,
        }
    }

    /// Check if audio description is required.
    pub fn requires_audio_description(&self) -> bool {
        matches!(
            self,
            Self::Wcag20Aa
                | Self::Wcag20Aaa
                | Self::Wcag21Aa
                | Self::Wcag21Aaa
                | Self::Section508
                | Self::En301549
        )
    }
}

/// Accessibility checker for video content.
pub struct AccessibilityChecker {
    standards: Vec<AccessibilityStandard>,
}

impl AccessibilityChecker {
    /// Create a new checker with the specified standards.
    pub fn new(standards: impl IntoIterator<Item = AccessibilityStandard>) -> Self {
        Self {
            standards: standards.into_iter().collect(),
        }
    }

    /// Create a checker for WCAG 2.1 AA (most common).
    pub fn wcag21_aa() -> Self {
        Self::new([AccessibilityStandard::Wcag21Aa])
    }

    /// Create a checker for FCC compliance.
    pub fn fcc() -> Self {
        Self::new([AccessibilityStandard::FccCaptions])
    }

    /// Check a single caption track for compliance.
    pub fn check_captions(&self, track: &CaptionTrack) -> AccessibilityReport {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        for standard in &self.standards {
            // Check each cue
            for (i, cue) in track.cues.iter().enumerate() {
                // Check display duration
                let duration_secs = cue.duration().as_secs_f64();
                let min_duration = standard.min_display_time_secs();
                if duration_secs < min_duration {
                    warnings.push(AccessibilityIssue {
                        standard: *standard,
                        severity: IssueSeverity::Warning,
                        code: "CAPTION_DURATION_SHORT".to_string(),
                        message: format!(
                            "Cue {} duration ({:.2}s) is below minimum ({:.2}s)",
                            i + 1,
                            duration_secs,
                            min_duration
                        ),
                        cue_index: Some(i),
                        suggestion: Some("Consider extending caption display time".to_string()),
                    });
                }

                // Check reading speed
                let cps = cue.chars_per_second();
                let max_cps = standard.max_chars_per_second();
                if cps > max_cps {
                    issues.push(AccessibilityIssue {
                        standard: *standard,
                        severity: IssueSeverity::Error,
                        code: "CAPTION_SPEED_HIGH".to_string(),
                        message: format!(
                            "Cue {} reading speed ({:.1} chars/s) exceeds maximum ({:.1})",
                            i + 1,
                            cps,
                            max_cps
                        ),
                        cue_index: Some(i),
                        suggestion: Some(
                            "Consider splitting caption into multiple cues".to_string(),
                        ),
                    });
                }

                // Check WPM
                let wpm = cue.words_per_minute();
                let max_wpm = standard.max_words_per_minute();
                if wpm > max_wpm {
                    warnings.push(AccessibilityIssue {
                        standard: *standard,
                        severity: IssueSeverity::Warning,
                        code: "CAPTION_WPM_HIGH".to_string(),
                        message: format!(
                            "Cue {} words per minute ({:.0}) exceeds recommended ({:.0})",
                            i + 1,
                            wpm,
                            max_wpm
                        ),
                        cue_index: Some(i),
                        suggestion: Some("Consider simplifying text or extending duration".to_string()),
                    });
                }

                // Check for empty captions
                if cue.text.trim().is_empty() {
                    issues.push(AccessibilityIssue {
                        standard: *standard,
                        severity: IssueSeverity::Error,
                        code: "CAPTION_EMPTY".to_string(),
                        message: format!("Cue {} is empty", i + 1),
                        cue_index: Some(i),
                        suggestion: Some("Remove empty cue or add text".to_string()),
                    });
                }
            }

            // Check for gaps in captions
            for i in 0..track.cues.len().saturating_sub(1) {
                let gap = track.cues[i + 1]
                    .start
                    .saturating_sub(track.cues[i].end)
                    .as_secs_f64();

                if gap > 5.0 {
                    warnings.push(AccessibilityIssue {
                        standard: *standard,
                        severity: IssueSeverity::Warning,
                        code: "CAPTION_GAP_LARGE".to_string(),
                        message: format!(
                            "Large gap ({:.1}s) between cues {} and {}",
                            gap,
                            i + 1,
                            i + 2
                        ),
                        cue_index: Some(i),
                        suggestion: Some(
                            "Verify no audio content is missing captions".to_string(),
                        ),
                    });
                }
            }
        }

        issues.extend(warnings);
        let passed = !issues.iter().any(|i| i.severity == IssueSeverity::Error);

        AccessibilityReport {
            passed,
            standards_checked: self.standards.clone(),
            issues,
            metadata: HashMap::new(),
        }
    }

    /// Check full video accessibility.
    pub fn check_video(&self, requirements: &VideoAccessibilityRequirements) -> AccessibilityReport {
        let mut issues = Vec::new();

        for standard in &self.standards {
            // Check for captions
            if !requirements.has_captions {
                issues.push(AccessibilityIssue {
                    standard: *standard,
                    severity: IssueSeverity::Error,
                    code: "NO_CAPTIONS".to_string(),
                    message: "Video has no captions".to_string(),
                    cue_index: None,
                    suggestion: Some("Add closed captions for all spoken content".to_string()),
                });
            }

            // Check for audio description
            if standard.requires_audio_description() && !requirements.has_audio_description {
                issues.push(AccessibilityIssue {
                    standard: *standard,
                    severity: IssueSeverity::Warning,
                    code: "NO_AUDIO_DESCRIPTION".to_string(),
                    message: "Video has no audio description track".to_string(),
                    cue_index: None,
                    suggestion: Some(
                        "Add audio description for visual content not described in audio".to_string(),
                    ),
                });
            }

            // Check for transcript
            if !requirements.has_transcript {
                let severity = match standard {
                    AccessibilityStandard::Wcag21Aaa | AccessibilityStandard::Wcag20Aaa => {
                        IssueSeverity::Error
                    }
                    _ => IssueSeverity::Warning,
                };

                issues.push(AccessibilityIssue {
                    standard: *standard,
                    severity,
                    code: "NO_TRANSCRIPT".to_string(),
                    message: "No transcript provided".to_string(),
                    cue_index: None,
                    suggestion: Some("Provide a full text transcript".to_string()),
                });
            }

            // Check for sign language
            if requirements.requires_sign_language && !requirements.has_sign_language {
                issues.push(AccessibilityIssue {
                    standard: *standard,
                    severity: IssueSeverity::Warning,
                    code: "NO_SIGN_LANGUAGE".to_string(),
                    message: "Sign language interpretation required but not provided".to_string(),
                    cue_index: None,
                    suggestion: Some("Add sign language interpretation track".to_string()),
                });
            }
        }

        let passed = !issues.iter().any(|i| i.severity == IssueSeverity::Error);

        AccessibilityReport {
            passed,
            standards_checked: self.standards.clone(),
            issues,
            metadata: HashMap::new(),
        }
    }
}

/// Video accessibility requirements.
#[derive(Debug, Clone, Default)]
pub struct VideoAccessibilityRequirements {
    /// Video has captions.
    pub has_captions: bool,
    /// Video has audio description.
    pub has_audio_description: bool,
    /// Video has transcript.
    pub has_transcript: bool,
    /// Video has sign language.
    pub has_sign_language: bool,
    /// Sign language is required.
    pub requires_sign_language: bool,
    /// Caption tracks available.
    pub caption_tracks: Vec<String>,
    /// Audio description tracks available.
    pub audio_description_tracks: Vec<String>,
}

/// Accessibility check report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityReport {
    /// Whether all checks passed.
    pub passed: bool,
    /// Standards that were checked.
    pub standards_checked: Vec<AccessibilityStandard>,
    /// Issues found.
    pub issues: Vec<AccessibilityIssue>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl AccessibilityReport {
    /// Get error count.
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Get warning count.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }

    /// Get issues by severity.
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&AccessibilityIssue> {
        self.issues.iter().filter(|i| i.severity == severity).collect()
    }

    /// Get issues by code.
    pub fn issues_by_code(&self, code: &str) -> Vec<&AccessibilityIssue> {
        self.issues.iter().filter(|i| i.code == code).collect()
    }
}

/// An accessibility issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityIssue {
    /// Standard that was violated.
    pub standard: AccessibilityStandard,
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Issue code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Caption cue index (if applicable).
    pub cue_index: Option<usize>,
    /// Suggested fix.
    pub suggestion: Option<String>,
}

/// Issue severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    /// Informational only.
    Info,
    /// Warning - should be fixed.
    Warning,
    /// Error - must be fixed for compliance.
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::captions::{CaptionCue, CaptionFormat};
    use std::time::Duration;

    #[test]
    fn test_checker_creation() {
        let checker = AccessibilityChecker::wcag21_aa();
        assert_eq!(checker.standards.len(), 1);
    }

    #[test]
    fn test_caption_check() {
        let checker = AccessibilityChecker::wcag21_aa();
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);

        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(2),
            "Hello, this is a test caption.",
        ));

        let report = checker.check_captions(&track);
        assert!(report.passed);
    }

    #[test]
    fn test_empty_caption() {
        let checker = AccessibilityChecker::wcag21_aa();
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);

        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(2),
            "",
        ));

        let report = checker.check_captions(&track);
        assert!(!report.passed);
        assert!(report.issues.iter().any(|i| i.code == "CAPTION_EMPTY"));
    }

    #[test]
    fn test_fast_caption() {
        let checker = AccessibilityChecker::new([AccessibilityStandard::Wcag21Aaa]);
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);

        // Very fast caption - 100 chars in 1 second = 100 cps
        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(1),
            "This is a very long caption that contains way too many characters to be read in just one second of display time.",
        ));

        let report = checker.check_captions(&track);
        assert!(!report.passed);
        assert!(report.issues.iter().any(|i| i.code == "CAPTION_SPEED_HIGH"));
    }

    #[test]
    fn test_video_requirements() {
        let checker = AccessibilityChecker::wcag21_aa();
        let requirements = VideoAccessibilityRequirements {
            has_captions: false,
            ..Default::default()
        };

        let report = checker.check_video(&requirements);
        assert!(!report.passed);
        assert!(report.issues.iter().any(|i| i.code == "NO_CAPTIONS"));
    }

    #[test]
    fn test_standard_thresholds() {
        let fcc = AccessibilityStandard::FccCaptions;
        assert_eq!(fcc.caption_accuracy_threshold(), 0.99);
        assert_eq!(fcc.max_sync_offset_ms(), 100);

        let wcag = AccessibilityStandard::Wcag21Aa;
        assert!(wcag.requires_audio_description());
    }
}
