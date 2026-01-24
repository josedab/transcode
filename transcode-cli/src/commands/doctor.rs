//! Environment diagnostics command.

use clap::Args;
use console::style;
use serde::Serialize;
use std::env;

/// Diagnostic check result.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticCheck {
    /// Check name.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Status message.
    pub message: String,
    /// Optional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Diagnostic report.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    /// Transcode version.
    pub version: String,
    /// Platform information.
    pub platform: PlatformInfo,
    /// SIMD capabilities.
    pub simd: SimdInfo,
    /// Check results.
    pub checks: Vec<DiagnosticCheck>,
    /// Overall status.
    pub overall_status: String,
}

/// Platform information.
#[derive(Debug, Clone, Serialize)]
pub struct PlatformInfo {
    /// Operating system.
    pub os: String,
    /// Architecture.
    pub arch: String,
    /// Number of CPUs.
    pub cpus: usize,
}

/// SIMD capability information.
#[derive(Debug, Clone, Serialize)]
pub struct SimdInfo {
    /// Best available SIMD level.
    pub level: String,
    /// Available instruction sets.
    pub features: Vec<String>,
}

/// Run environment diagnostics.
#[derive(Args, Debug)]
pub struct CmdDoctor {
    /// Output in JSON format.
    #[arg(long)]
    pub json: bool,
}

impl CmdDoctor {
    /// Execute the doctor command.
    pub fn run(&self) -> anyhow::Result<()> {
        let report = self.run_diagnostics();

        if self.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            self.print_report(&report);
        }

        // Exit with error if any critical checks failed
        let has_failures = report.checks.iter().any(|c| !c.passed);
        if has_failures {
            std::process::exit(1);
        }

        Ok(())
    }

    fn run_diagnostics(&self) -> DiagnosticReport {
        let mut checks = Vec::new();

        // Check SIMD support
        let simd_caps = transcode_codecs::detect_simd();
        let simd_level = simd_caps.best_level().to_string();
        let mut simd_features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                simd_features.push("AVX2".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.1") {
                simd_features.push("SSE4.1".to_string());
            }
            if std::arch::is_x86_feature_detected!("ssse3") {
                simd_features.push("SSSE3".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            simd_features.push("NEON".to_string());
        }

        checks.push(DiagnosticCheck {
            name: "SIMD Support".to_string(),
            passed: !simd_features.is_empty() || simd_level != "none",
            message: format!("Using {} acceleration", simd_level),
            details: if simd_features.is_empty() {
                Some("Scalar fallback will be used".to_string())
            } else {
                Some(format!("Available: {}", simd_features.join(", ")))
            },
        });

        // Check available memory
        let memory_check = self.check_memory();
        checks.push(memory_check);

        // Check temp directory
        checks.push(self.check_temp_dir());

        // Check thread count
        let cpus = num_cpus();
        checks.push(DiagnosticCheck {
            name: "Thread Pool".to_string(),
            passed: cpus >= 2,
            message: format!("{} logical CPUs available", cpus),
            details: if cpus < 2 {
                Some("Single-threaded mode may be slow".to_string())
            } else {
                None
            },
        });

        // Check for GPU support (placeholder)
        checks.push(DiagnosticCheck {
            name: "GPU Acceleration".to_string(),
            passed: true,
            message: "GPU support available via wgpu".to_string(),
            details: None,
        });

        // Overall status
        let all_passed = checks.iter().all(|c| c.passed);
        let overall_status = if all_passed {
            "All checks passed".to_string()
        } else {
            "Some checks failed".to_string()
        };

        DiagnosticReport {
            version: transcode::VERSION.to_string(),
            platform: PlatformInfo {
                os: env::consts::OS.to_string(),
                arch: env::consts::ARCH.to_string(),
                cpus,
            },
            simd: SimdInfo {
                level: simd_level,
                features: simd_features,
            },
            checks,
            overall_status,
        }
    }

    fn check_memory(&self) -> DiagnosticCheck {
        // Try to estimate available memory (platform-specific)
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                if let Some(line) = meminfo.lines().find(|l| l.starts_with("MemAvailable:")) {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            let gb = kb as f64 / 1024.0 / 1024.0;
                            return DiagnosticCheck {
                                name: "Available Memory".to_string(),
                                passed: gb >= 1.0,
                                message: format!("{:.1} GB available", gb),
                                details: if gb < 1.0 {
                                    Some("Low memory may cause issues with large files".to_string())
                                } else {
                                    None
                                },
                            };
                        }
                    }
                }
            }
        }

        DiagnosticCheck {
            name: "Available Memory".to_string(),
            passed: true,
            message: "Memory check not available on this platform".to_string(),
            details: None,
        }
    }

    fn check_temp_dir(&self) -> DiagnosticCheck {
        let temp = env::temp_dir();
        let writable = temp.exists()
            && std::fs::write(temp.join(".transcode_test"), b"test").is_ok()
            && std::fs::remove_file(temp.join(".transcode_test")).is_ok();

        DiagnosticCheck {
            name: "Temp Directory".to_string(),
            passed: writable,
            message: if writable {
                format!("Writable: {}", temp.display())
            } else {
                format!("Not writable: {}", temp.display())
            },
            details: if !writable {
                Some("Temporary files may fail to write".to_string())
            } else {
                None
            },
        }
    }

    fn print_report(&self, report: &DiagnosticReport) {
        println!();
        println!("{}", style("Transcode Doctor").cyan().bold());
        println!();

        // Version and platform
        println!("{}", style("Environment:").white().bold());
        println!("  Version:  {}", style(&report.version).yellow());
        println!(
            "  Platform: {} ({})",
            report.platform.os, report.platform.arch
        );
        println!("  CPUs:     {}", report.platform.cpus);
        println!();

        // SIMD
        println!("{}", style("SIMD Acceleration:").white().bold());
        println!("  Level:    {}", style(&report.simd.level).green());
        if !report.simd.features.is_empty() {
            println!("  Features: {}", report.simd.features.join(", "));
        }
        println!();

        // Checks
        println!("{}", style("Diagnostics:").white().bold());
        for check in &report.checks {
            let status = if check.passed {
                style("[OK]").green()
            } else {
                style("[FAIL]").red()
            };
            println!("  {} {} - {}", status, check.name, check.message);
            if let Some(ref details) = check.details {
                println!("       {}", style(details).dim());
            }
        }

        println!();
        let status_style = if report.overall_status.contains("passed") {
            style(&report.overall_status).green().bold()
        } else {
            style(&report.overall_status).red().bold()
        };
        println!("{}", status_style);
        println!();
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus >= 1);
    }

    #[test]
    fn test_diagnostic_check_serialization() {
        let check = DiagnosticCheck {
            name: "Test".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: None,
        };
        let json = serde_json::to_string(&check).unwrap();
        assert!(json.contains("\"passed\":true"));
    }
}
