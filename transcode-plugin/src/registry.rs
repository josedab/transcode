//! Plugin registry for managing loaded plugins.

use std::collections::HashMap;
use tracing::info;

use crate::api::{PluginInfo, PluginType};
use crate::error::{Error, Result};
use crate::API_VERSION;

/// Central registry that tracks all loaded plugins.
pub struct PluginRegistry {
    plugins: HashMap<String, RegisteredPlugin>,
}

struct RegisteredPlugin {
    info: PluginInfo,
    enabled: bool,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin by its info. Returns error if name is duplicate or API mismatched.
    pub fn register(&mut self, info: PluginInfo) -> Result<()> {
        if info.api_version != API_VERSION {
            return Err(Error::ApiVersionMismatch {
                plugin: info.api_version,
                expected: API_VERSION,
            });
        }

        if self.plugins.contains_key(&info.name) {
            return Err(Error::AlreadyRegistered {
                name: info.name.clone(),
            });
        }

        info!(
            name = %info.name,
            version = %info.version,
            plugin_type = ?info.plugin_type,
            "Plugin registered"
        );

        self.plugins.insert(
            info.name.clone(),
            RegisteredPlugin {
                info,
                enabled: true,
            },
        );
        Ok(())
    }

    /// Unregister a plugin by name.
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        self.plugins.remove(name).ok_or_else(|| Error::NotFound {
            name: name.into(),
        })?;
        Ok(())
    }

    /// Get plugin info by name.
    pub fn get(&self, name: &str) -> Option<&PluginInfo> {
        self.plugins.get(name).map(|p| &p.info)
    }

    /// List all registered plugins.
    pub fn list_all(&self) -> Vec<&PluginInfo> {
        self.plugins.values().map(|p| &p.info).collect()
    }

    /// List plugins filtered by type.
    pub fn list_by_type(&self, plugin_type: PluginType) -> Vec<&PluginInfo> {
        self.plugins
            .values()
            .filter(|p| p.info.plugin_type == plugin_type)
            .map(|p| &p.info)
            .collect()
    }

    /// Enable or disable a plugin.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<()> {
        let plugin = self.plugins.get_mut(name).ok_or_else(|| Error::NotFound {
            name: name.into(),
        })?;
        plugin.enabled = enabled;
        Ok(())
    }

    /// Check if a plugin is enabled.
    pub fn is_enabled(&self, name: &str) -> bool {
        self.plugins
            .get(name)
            .map(|p| p.enabled)
            .unwrap_or(false)
    }

    /// Return the count of registered plugins.
    pub fn count(&self) -> usize {
        self.plugins.len()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
