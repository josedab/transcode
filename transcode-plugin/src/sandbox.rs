//! Sandbox configuration for WASM plugin isolation.

use serde::{Deserialize, Serialize};

/// Security policy for sandboxed plugin execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Maximum memory a plugin can allocate (bytes).
    pub max_memory_bytes: u64,
    /// Maximum CPU time per invocation (milliseconds).
    pub max_cpu_time_ms: u64,
    /// Whether the plugin can access the filesystem.
    pub allow_filesystem: bool,
    /// Whether the plugin can make network requests.
    pub allow_network: bool,
    /// Allowed filesystem paths (if filesystem access is enabled).
    pub allowed_paths: Vec<String>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            max_cpu_time_ms: 30_000,              // 30 seconds
            allow_filesystem: false,
            allow_network: false,
            allowed_paths: Vec::new(),
        }
    }
}

/// Configuration for the plugin sandbox environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub policy: SandboxPolicy,
    /// Whether to trust native plugins (skip sandbox).
    pub trust_native: bool,
    /// Whether WASM plugins run in sandbox.
    pub sandbox_wasm: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            policy: SandboxPolicy::default(),
            trust_native: false,
            sandbox_wasm: true,
        }
    }
}

impl SandboxConfig {
    /// Permissive config for development.
    pub fn development() -> Self {
        Self {
            trust_native: true,
            sandbox_wasm: true,
            policy: SandboxPolicy {
                allow_filesystem: true,
                allow_network: true,
                ..Default::default()
            },
        }
    }

    /// Restrictive config for production.
    pub fn production() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sandbox_is_restrictive() {
        let config = SandboxConfig::default();
        assert!(!config.trust_native);
        assert!(config.sandbox_wasm);
        assert!(!config.policy.allow_filesystem);
        assert!(!config.policy.allow_network);
    }

    #[test]
    fn test_development_sandbox() {
        let config = SandboxConfig::development();
        assert!(config.trust_native);
        assert!(config.policy.allow_filesystem);
    }

    #[test]
    fn test_production_sandbox() {
        let config = SandboxConfig::production();
        assert!(!config.trust_native);
    }
}
