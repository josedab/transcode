//! Conformance test report generation
//!
//! Generates detailed reports of conformance test results in various formats.

use crate::{ConformanceConfig, TestResult, TestStatus, TestStream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Conformance test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceReport {
    /// Report title
    pub title: String,
    /// Report generation timestamp
    pub timestamp: String,
    /// Configuration used
    pub config_summary: ConfigSummary,
    /// Overall summary
    pub summary: ReportSummary,
    /// Results by profile
    pub by_profile: HashMap<String, ProfileResults>,
    /// Individual test results
    pub results: Vec<TestResult>,
    /// System information
    pub system_info: SystemInfo,
}

/// Configuration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    /// Profiles tested
    pub profiles: Vec<String>,
    /// Whether downloads were allowed
    pub allow_download: bool,
    /// Cache directory
    pub cache_dir: String,
    /// Timeout per test
    pub timeout_secs: u64,
}

/// Overall report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total tests run
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Tests skipped
    pub skipped: usize,
    /// Tests with errors
    pub errors: usize,
    /// Pass rate percentage
    pub pass_rate: f64,
    /// Total duration in milliseconds
    pub total_duration_ms: u64,
}

impl ReportSummary {
    /// Calculate summary from test results
    pub fn from_results(results: &[TestResult]) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed = results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let skipped = results.iter().filter(|r| r.status == TestStatus::Skipped).count();
        let errors = results.iter().filter(|r| r.status == TestStatus::Error).count();
        let total_duration_ms: u64 = results.iter().map(|r| r.duration_ms).sum();

        let pass_rate = if total > 0 {
            (passed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Self {
            total,
            passed,
            failed,
            skipped,
            errors,
            pass_rate,
            total_duration_ms,
        }
    }
}

/// Results for a specific profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResults {
    /// Profile name
    pub profile: String,
    /// Number of tests
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Pass rate
    pub pass_rate: f64,
    /// List of failed test IDs
    pub failed_tests: Vec<String>,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Rust version (if available)
    pub rust_version: Option<String>,
    /// Transcode version
    pub transcode_version: String,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            rust_version: option_env!("RUSTC_VERSION").map(String::from),
            transcode_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Report format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON format
    Json,
    /// Human-readable text
    Text,
    /// HTML format
    Html,
    /// Markdown format
    Markdown,
}

/// Report generator
pub struct ReportGenerator {
    format: ReportFormat,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new(ReportFormat::Json)
    }
}

impl ReportGenerator {
    /// Create a new report generator
    pub fn new(format: ReportFormat) -> Self {
        Self { format }
    }

    /// Generate a report from test results
    pub fn generate(
        &self,
        title: &str,
        config: &ConformanceConfig,
        streams: &[TestStream],
        results: &[TestResult],
    ) -> ConformanceReport {
        let summary = ReportSummary::from_results(results);

        let mut by_profile: HashMap<String, ProfileResults> = HashMap::new();

        for profile in &config.profiles {
            let profile_name = profile.name().to_string();
            let profile_results: Vec<_> = results
                .iter()
                .filter(|r| {
                    streams
                        .iter()
                        .any(|s| s.id == r.stream_id && s.profile == *profile)
                })
                .collect();

            let passed = profile_results
                .iter()
                .filter(|r| r.status == TestStatus::Passed)
                .count();
            let failed = profile_results
                .iter()
                .filter(|r| r.status == TestStatus::Failed)
                .count();
            let total = profile_results.len();

            let failed_tests: Vec<_> = profile_results
                .iter()
                .filter(|r| r.status == TestStatus::Failed)
                .map(|r| r.stream_id.clone())
                .collect();

            by_profile.insert(
                profile_name.clone(),
                ProfileResults {
                    profile: profile_name,
                    total,
                    passed,
                    failed,
                    pass_rate: if total > 0 {
                        (passed as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    },
                    failed_tests,
                },
            );
        }

        ConformanceReport {
            title: title.to_string(),
            timestamp: chrono_lite::now(),
            config_summary: ConfigSummary {
                profiles: config.profiles.iter().map(|p| p.name().to_string()).collect(),
                allow_download: config.allow_download,
                cache_dir: config.cache_dir.display().to_string(),
                timeout_secs: config.timeout_secs,
            },
            summary,
            by_profile,
            results: results.to_vec(),
            system_info: SystemInfo::default(),
        }
    }

    /// Write report to file
    pub fn write_to_file(&self, report: &ConformanceReport, path: &Path) -> std::io::Result<()> {
        let content = match self.format {
            ReportFormat::Json => self.to_json(report),
            ReportFormat::Text => self.to_text(report),
            ReportFormat::Html => self.to_html(report),
            ReportFormat::Markdown => self.to_markdown(report),
        };

        let mut file = fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    /// Generate JSON output
    fn to_json(&self, report: &ConformanceReport) -> String {
        serde_json::to_string_pretty(report).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Generate text output
    fn to_text(&self, report: &ConformanceReport) -> String {
        let mut output = String::new();

        output.push_str("═══════════════════════════════════════════════════════\n");
        output.push_str(&format!("  {}\n", report.title));
        output.push_str(&format!("  Generated: {}\n", report.timestamp));
        output.push_str("═══════════════════════════════════════════════════════\n\n");

        output.push_str("SUMMARY\n");
        output.push_str("───────────────────────────────────────────────────────\n");
        output.push_str(&format!("  Total Tests:   {}\n", report.summary.total));
        output.push_str(&format!("  Passed:        {} ({:.1}%)\n", report.summary.passed, report.summary.pass_rate));
        output.push_str(&format!("  Failed:        {}\n", report.summary.failed));
        output.push_str(&format!("  Skipped:       {}\n", report.summary.skipped));
        output.push_str(&format!("  Errors:        {}\n", report.summary.errors));
        output.push_str(&format!("  Duration:      {}ms\n\n", report.summary.total_duration_ms));

        output.push_str("RESULTS BY PROFILE\n");
        output.push_str("───────────────────────────────────────────────────────\n");

        for (name, profile) in &report.by_profile {
            output.push_str(&format!("\n  {} Profile:\n", name));
            output.push_str(&format!("    Total:   {}\n", profile.total));
            output.push_str(&format!("    Passed:  {} ({:.1}%)\n", profile.passed, profile.pass_rate));
            output.push_str(&format!("    Failed:  {}\n", profile.failed));

            if !profile.failed_tests.is_empty() {
                output.push_str("    Failed tests:\n");
                for test in &profile.failed_tests {
                    output.push_str(&format!("      - {}\n", test));
                }
            }
        }

        output.push_str("\nDETAILED RESULTS\n");
        output.push_str("───────────────────────────────────────────────────────\n");

        for result in &report.results {
            let status_icon = match result.status {
                TestStatus::Passed => "[PASS]",
                TestStatus::Failed => "[FAIL]",
                TestStatus::Skipped => "[SKIP]",
                TestStatus::Error => "[ERR] ",
            };

            output.push_str(&format!("\n  {} {}\n", status_icon, result.stream_id));
            output.push_str(&format!("    Test: {}\n", result.test_name));
            output.push_str(&format!("    Duration: {}ms\n", result.duration_ms));

            if let Some(frames) = result.decoded_frames {
                output.push_str(&format!("    Decoded frames: {}\n", frames));
            }

            if let Some(error) = &result.error_message {
                output.push_str(&format!("    Error: {}\n", error));
            }

            if !result.checksum_results.is_empty() {
                let passed = result.checksum_results.iter().filter(|c| c.matches).count();
                let total = result.checksum_results.len();
                output.push_str(&format!("    Checksums: {}/{} passed\n", passed, total));
            }
        }

        output.push_str("\n═══════════════════════════════════════════════════════\n");
        output.push_str(&format!("  System: {} {}\n", report.system_info.os, report.system_info.arch));
        output.push_str(&format!("  Transcode version: {}\n", report.system_info.transcode_version));
        output.push_str("═══════════════════════════════════════════════════════\n");

        output
    }

    /// Generate HTML output
    fn to_html(&self, report: &ConformanceReport) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.title));
        html.push_str("<style>\n");
        html.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }\n");
        html.push_str("h1 { color: #333; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str(".pass { color: #28a745; font-weight: bold; }\n");
        html.push_str(".fail { color: #dc3545; font-weight: bold; }\n");
        html.push_str(".skip { color: #6c757d; }\n");
        html.push_str(".error { color: #fd7e14; font-weight: bold; }\n");
        html.push_str(".summary { background-color: #e9ecef; padding: 20px; border-radius: 5px; }\n");
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str(&format!("<h1>{}</h1>\n", report.title));
        html.push_str(&format!("<p>Generated: {}</p>\n", report.timestamp));

        html.push_str("<div class='summary'>\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str(&format!("<p>Total: {} | ", report.summary.total));
        html.push_str(&format!("<span class='pass'>Passed: {} ({:.1}%)</span> | ", report.summary.passed, report.summary.pass_rate));
        html.push_str(&format!("<span class='fail'>Failed: {}</span> | ", report.summary.failed));
        html.push_str(&format!("<span class='skip'>Skipped: {}</span> | ", report.summary.skipped));
        html.push_str(&format!("Duration: {}ms</p>\n", report.summary.total_duration_ms));
        html.push_str("</div>\n");

        html.push_str("<h2>Results by Profile</h2>\n");
        html.push_str("<table>\n<tr><th>Profile</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr>\n");
        for (name, profile) in &report.by_profile {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class='pass'>{}</td><td class='fail'>{}</td><td>{:.1}%</td></tr>\n",
                name, profile.total, profile.passed, profile.failed, profile.pass_rate
            ));
        }
        html.push_str("</table>\n");

        html.push_str("<h2>Detailed Results</h2>\n");
        html.push_str("<table>\n<tr><th>Status</th><th>Stream ID</th><th>Test Name</th><th>Duration</th><th>Details</th></tr>\n");
        for result in &report.results {
            let (status_class, status_text) = match result.status {
                TestStatus::Passed => ("pass", "PASS"),
                TestStatus::Failed => ("fail", "FAIL"),
                TestStatus::Skipped => ("skip", "SKIP"),
                TestStatus::Error => ("error", "ERROR"),
            };

            let details = result.error_message.clone().unwrap_or_else(|| {
                result.decoded_frames.map(|f| format!("{} frames", f)).unwrap_or_default()
            });

            html.push_str(&format!(
                "<tr><td class='{}'>{}</td><td>{}</td><td>{}</td><td>{}ms</td><td>{}</td></tr>\n",
                status_class, status_text, result.stream_id, result.test_name, result.duration_ms, details
            ));
        }
        html.push_str("</table>\n");

        html.push_str(&format!("<footer><p>System: {} {} | Transcode {}</p></footer>\n",
            report.system_info.os, report.system_info.arch, report.system_info.transcode_version));

        html.push_str("</body>\n</html>\n");

        html
    }

    /// Generate Markdown output
    fn to_markdown(&self, report: &ConformanceReport) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", report.title));
        md.push_str(&format!("Generated: {}\n\n", report.timestamp));

        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total Tests | {} |\n", report.summary.total));
        md.push_str(&format!("| Passed | {} ({:.1}%) |\n", report.summary.passed, report.summary.pass_rate));
        md.push_str(&format!("| Failed | {} |\n", report.summary.failed));
        md.push_str(&format!("| Skipped | {} |\n", report.summary.skipped));
        md.push_str(&format!("| Errors | {} |\n", report.summary.errors));
        md.push_str(&format!("| Duration | {}ms |\n\n", report.summary.total_duration_ms));

        md.push_str("## Results by Profile\n\n");
        md.push_str("| Profile | Total | Passed | Failed | Pass Rate |\n");
        md.push_str("|---------|-------|--------|--------|----------|\n");
        for (name, profile) in &report.by_profile {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:.1}% |\n",
                name, profile.total, profile.passed, profile.failed, profile.pass_rate
            ));
        }
        md.push('\n');

        md.push_str("## Detailed Results\n\n");
        md.push_str("| Status | Stream ID | Test Name | Duration |\n");
        md.push_str("|--------|-----------|-----------|----------|\n");
        for result in &report.results {
            let status = match result.status {
                TestStatus::Passed => "PASS",
                TestStatus::Failed => "FAIL",
                TestStatus::Skipped => "SKIP",
                TestStatus::Error => "ERROR",
            };
            md.push_str(&format!(
                "| {} | {} | {} | {}ms |\n",
                status, result.stream_id, result.test_name, result.duration_ms
            ));
        }
        md.push('\n');

        md.push_str("---\n\n");
        md.push_str(&format!("*System: {} {} | Transcode {}*\n",
            report.system_info.os, report.system_info.arch, report.system_info.transcode_version));

        md
    }
}

/// Simple timestamp generation (avoiding full chrono dependency)
mod chrono_lite {
    pub fn now() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        // Simple ISO-8601 like format
        let secs = duration.as_secs();
        let days = secs / 86400;
        let remaining = secs % 86400;
        let hours = remaining / 3600;
        let minutes = (remaining % 3600) / 60;
        let seconds = remaining % 60;

        // Approximate date calculation from days since epoch
        let years = 1970 + days / 365;
        let day_of_year = days % 365;
        let month = day_of_year / 30 + 1;
        let day = day_of_year % 30 + 1;

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            years, month, day, hours, minutes, seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_from_results() {
        let results = vec![
            TestResult {
                stream_id: "test1".to_string(),
                test_name: "Test 1".to_string(),
                status: TestStatus::Passed,
                duration_ms: 100,
                error_message: None,
                decoded_frames: Some(10),
                checksum_results: vec![],
                notes: vec![],
            },
            TestResult {
                stream_id: "test2".to_string(),
                test_name: "Test 2".to_string(),
                status: TestStatus::Failed,
                duration_ms: 200,
                error_message: Some("error".to_string()),
                decoded_frames: None,
                checksum_results: vec![],
                notes: vec![],
            },
        ];

        let summary = ReportSummary::from_results(&results);

        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.total_duration_ms, 300);
        assert!((summary.pass_rate - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_report_formats() {
        let config = ConformanceConfig::baseline_only();
        let streams = vec![];
        let results = vec![];

        let generator = ReportGenerator::new(ReportFormat::Json);
        let report = generator.generate("Test Report", &config, &streams, &results);

        let json = generator.to_json(&report);
        assert!(json.contains("\"title\""));

        let generator = ReportGenerator::new(ReportFormat::Text);
        let text = generator.to_text(&report);
        assert!(text.contains("Test Report"));

        let generator = ReportGenerator::new(ReportFormat::Html);
        let html = generator.to_html(&report);
        assert!(html.contains("<html>"));

        let generator = ReportGenerator::new(ReportFormat::Markdown);
        let md = generator.to_markdown(&report);
        assert!(md.contains("# Test Report"));
    }
}
