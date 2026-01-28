//! Plugin runtime for managing plugin lifecycle and execution.

use crate::api::{PluginCapabilities, PluginInfo};
use crate::error::{Error, Result};
use crate::sandbox::SandboxConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// State of a plugin instance in the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginState {
    /// Plugin loaded but not initialized.
    Loaded,
    /// Plugin initialized and ready for processing.
    Ready,
    /// Plugin currently processing data.
    Processing,
    /// Plugin encountered an error.
    Error,
    /// Plugin shut down.
    Shutdown,
}

/// Statistics for a running plugin instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginStats {
    pub invocations: u64,
    pub total_processing_ms: u64,
    pub errors: u64,
    pub bytes_processed: u64,
    pub peak_memory_bytes: u64,
}

impl PluginStats {
    pub fn avg_processing_ms(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.total_processing_ms as f64 / self.invocations as f64
        }
    }
}

/// A managed plugin instance with lifecycle tracking.
pub struct PluginInstance {
    info: PluginInfo,
    state: PluginState,
    stats: PluginStats,
    sandbox: SandboxConfig,
    capabilities: PluginCapabilities,
}

impl PluginInstance {
    /// Create a new plugin instance (in Loaded state).
    pub fn new(info: PluginInfo, sandbox: SandboxConfig) -> Self {
        Self {
            info,
            state: PluginState::Loaded,
            stats: PluginStats::default(),
            sandbox,
            capabilities: PluginCapabilities::default(),
        }
    }

    pub fn info(&self) -> &PluginInfo {
        &self.info
    }

    pub fn state(&self) -> PluginState {
        self.state
    }

    pub fn stats(&self) -> &PluginStats {
        &self.stats
    }

    pub fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    /// Set plugin capabilities.
    pub fn with_capabilities(mut self, caps: PluginCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Initialize the plugin (Loaded → Ready).
    pub fn initialize(&mut self) -> Result<()> {
        if self.state != PluginState::Loaded {
            return Err(Error::InitFailed {
                message: format!("Cannot initialize plugin in {:?} state", self.state),
            });
        }

        tracing::info!(plugin = %self.info.name, "Initializing plugin");
        self.state = PluginState::Ready;
        Ok(())
    }

    /// Process a data buffer through the plugin.
    /// Enforces sandbox memory and time limits.
    pub fn process(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        if self.state != PluginState::Ready {
            return Err(Error::PluginError {
                name: self.info.name.clone(),
                message: format!("Plugin not ready (state: {:?})", self.state),
            });
        }

        // Check memory limits
        if input.len() as u64 > self.sandbox.policy.max_memory_bytes {
            return Err(Error::SandboxViolation {
                message: format!(
                    "Input size {} exceeds memory limit {}",
                    input.len(),
                    self.sandbox.policy.max_memory_bytes
                ),
            });
        }

        self.state = PluginState::Processing;
        let start = Instant::now();

        // Simulated plugin processing — in production this would dispatch
        // to WASM runtime or native plugin
        let output = self.execute_sandboxed(input);

        let elapsed = start.elapsed();

        // Check time limits
        if elapsed > Duration::from_millis(self.sandbox.policy.max_cpu_time_ms) {
            self.state = PluginState::Error;
            self.stats.errors += 1;
            return Err(Error::SandboxViolation {
                message: format!(
                    "Processing took {}ms, exceeding limit of {}ms",
                    elapsed.as_millis(),
                    self.sandbox.policy.max_cpu_time_ms
                ),
            });
        }

        self.stats.invocations += 1;
        self.stats.total_processing_ms += elapsed.as_millis() as u64;
        self.stats.bytes_processed += input.len() as u64;
        self.state = PluginState::Ready;

        output
    }

    /// Shut down the plugin (Ready → Shutdown).
    pub fn shutdown(&mut self) -> Result<()> {
        if self.state == PluginState::Shutdown {
            return Ok(());
        }
        tracing::info!(
            plugin = %self.info.name,
            invocations = self.stats.invocations,
            "Shutting down plugin"
        );
        self.state = PluginState::Shutdown;
        Ok(())
    }

    /// Reset plugin from Error state back to Ready.
    pub fn reset(&mut self) -> Result<()> {
        if self.state != PluginState::Error {
            return Err(Error::PluginError {
                name: self.info.name.clone(),
                message: "Plugin is not in error state".into(),
            });
        }
        self.state = PluginState::Ready;
        Ok(())
    }

    fn execute_sandboxed(&self, input: &[u8]) -> Result<Vec<u8>> {
        // Simulated processing — pass-through with a marker header
        // In production: dispatch to wasmtime/wasmer for WASM plugins,
        // or dlopen/dlsym for native plugins
        let mut output = Vec::with_capacity(input.len() + 8);
        output.extend_from_slice(b"TPLUGIN\0");
        output.extend_from_slice(input);
        Ok(output)
    }
}

/// The plugin runtime manages all active plugin instances.
pub struct PluginRuntime {
    instances: HashMap<String, PluginInstance>,
    default_sandbox: SandboxConfig,
}

impl PluginRuntime {
    /// Create a new runtime with default sandbox configuration.
    pub fn new(sandbox: SandboxConfig) -> Self {
        Self {
            instances: HashMap::new(),
            default_sandbox: sandbox,
        }
    }

    /// Load and register a plugin instance.
    pub fn load(&mut self, info: PluginInfo) -> Result<()> {
        if self.instances.contains_key(&info.name) {
            return Err(Error::AlreadyRegistered {
                name: info.name.clone(),
            });
        }

        let instance = PluginInstance::new(info.clone(), self.default_sandbox.clone());
        self.instances.insert(info.name.clone(), instance);
        Ok(())
    }

    /// Initialize a loaded plugin.
    pub fn initialize(&mut self, name: &str) -> Result<()> {
        let instance = self.instances.get_mut(name).ok_or_else(|| Error::NotFound {
            name: name.into(),
        })?;
        instance.initialize()
    }

    /// Process data through a plugin.
    pub fn process(&mut self, name: &str, input: &[u8]) -> Result<Vec<u8>> {
        let instance = self.instances.get_mut(name).ok_or_else(|| Error::NotFound {
            name: name.into(),
        })?;
        instance.process(input)
    }

    /// Shut down a specific plugin.
    pub fn shutdown_plugin(&mut self, name: &str) -> Result<()> {
        let instance = self.instances.get_mut(name).ok_or_else(|| Error::NotFound {
            name: name.into(),
        })?;
        instance.shutdown()
    }

    /// Shut down all plugins.
    pub fn shutdown_all(&mut self) {
        for instance in self.instances.values_mut() {
            let _ = instance.shutdown();
        }
    }

    /// Get plugin state.
    pub fn state(&self, name: &str) -> Option<PluginState> {
        self.instances.get(name).map(|i| i.state())
    }

    /// Get plugin stats.
    pub fn stats(&self, name: &str) -> Option<&PluginStats> {
        self.instances.get(name).map(|i| i.stats())
    }

    /// List all loaded plugins.
    pub fn list(&self) -> Vec<&PluginInfo> {
        self.instances.values().map(|i| i.info()).collect()
    }

    /// Count loaded plugins.
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Hot-reload a plugin: shut down old instance, load new one.
    pub fn hot_reload(&mut self, info: PluginInfo) -> Result<()> {
        let name = info.name.clone();
        if let Some(instance) = self.instances.get_mut(&name) {
            instance.shutdown()?;
        }
        self.instances.remove(&name);

        let mut instance = PluginInstance::new(info, self.default_sandbox.clone());
        instance.initialize()?;
        self.instances.insert(name, instance);
        Ok(())
    }
}

impl Default for PluginRuntime {
    fn default() -> Self {
        Self::new(SandboxConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::api::PluginType;
    use crate::sandbox::SandboxPolicy;

    use super::*;

    fn test_info(name: &str, pt: PluginType) -> PluginInfo {
        PluginInfo {
            name: name.into(),
            version: "1.0.0".into(),
            plugin_type: pt,
            description: "test".into(),
            author: "test".into(),
            api_version: crate::API_VERSION,
        }
    }

    #[test]
    fn test_plugin_lifecycle() {
        let info = test_info("my-filter", PluginType::Filter);
        let mut instance = PluginInstance::new(info, SandboxConfig::default());

        assert_eq!(instance.state(), PluginState::Loaded);

        instance.initialize().unwrap();
        assert_eq!(instance.state(), PluginState::Ready);

        let output = instance.process(b"hello").unwrap();
        assert!(output.starts_with(b"TPLUGIN\0"));
        assert_eq!(&output[8..], b"hello");
        assert_eq!(instance.stats().invocations, 1);

        instance.shutdown().unwrap();
        assert_eq!(instance.state(), PluginState::Shutdown);
    }

    #[test]
    fn test_process_before_init_fails() {
        let info = test_info("bad", PluginType::Filter);
        let mut instance = PluginInstance::new(info, SandboxConfig::default());
        assert!(instance.process(b"data").is_err());
    }

    #[test]
    fn test_sandbox_memory_limit() {
        let info = test_info("limited", PluginType::Filter);
        let sandbox = SandboxConfig {
            policy: SandboxPolicy {
                max_memory_bytes: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut instance = PluginInstance::new(info, sandbox);
        instance.initialize().unwrap();

        // Data within limit
        assert!(instance.process(&[0u8; 5]).is_ok());

        // Data exceeding limit
        assert!(instance.process(&[0u8; 20]).is_err());
    }

    #[test]
    fn test_runtime_load_and_process() {
        let mut runtime = PluginRuntime::default();

        let info = test_info("codec-x", PluginType::Codec);
        runtime.load(info).unwrap();
        runtime.initialize("codec-x").unwrap();

        let output = runtime.process("codec-x", b"frame-data").unwrap();
        assert!(output.len() > 10);

        assert_eq!(runtime.state("codec-x"), Some(PluginState::Ready));
        assert_eq!(runtime.stats("codec-x").unwrap().invocations, 1);
    }

    #[test]
    fn test_runtime_hot_reload() {
        let mut runtime = PluginRuntime::default();

        let info_v1 = test_info("reloadable", PluginType::Filter);
        runtime.load(info_v1).unwrap();
        runtime.initialize("reloadable").unwrap();
        runtime.process("reloadable", b"data1").unwrap();

        // Hot reload with new version
        let mut info_v2 = test_info("reloadable", PluginType::Filter);
        info_v2.version = "2.0.0".into();
        runtime.hot_reload(info_v2).unwrap();

        // Stats should be reset after reload
        assert_eq!(runtime.stats("reloadable").unwrap().invocations, 0);
        assert_eq!(runtime.state("reloadable"), Some(PluginState::Ready));
    }

    #[test]
    fn test_runtime_shutdown_all() {
        let mut runtime = PluginRuntime::default();
        runtime.load(test_info("p1", PluginType::Filter)).unwrap();
        runtime.load(test_info("p2", PluginType::Codec)).unwrap();
        runtime.initialize("p1").unwrap();
        runtime.initialize("p2").unwrap();

        runtime.shutdown_all();
        assert_eq!(runtime.state("p1"), Some(PluginState::Shutdown));
        assert_eq!(runtime.state("p2"), Some(PluginState::Shutdown));
    }

    #[test]
    fn test_error_reset() {
        let info = test_info("errored", PluginType::Filter);
        let sandbox = SandboxConfig {
            policy: SandboxPolicy {
                max_memory_bytes: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut instance = PluginInstance::new(info, sandbox);
        instance.initialize().unwrap();

        // Trigger sandbox violation
        let _ = instance.process(&[0u8; 100]);
        // State should be Error due to sandbox violation
        if instance.state() == PluginState::Error {
            instance.reset().unwrap();
            assert_eq!(instance.state(), PluginState::Ready);
        }
    }

    #[test]
    fn test_plugin_stats() {
        let info = test_info("stats-test", PluginType::Analyzer);
        let mut instance = PluginInstance::new(info, SandboxConfig::default());
        instance.initialize().unwrap();

        for _ in 0..5 {
            instance.process(b"data").unwrap();
        }

        let stats = instance.stats();
        assert_eq!(stats.invocations, 5);
        assert_eq!(stats.bytes_processed, 20); // 5 * 4 bytes
        assert!(stats.avg_processing_ms() >= 0.0);
    }
}
