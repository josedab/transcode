//! Plugin trait and type definitions.

use serde::{Deserialize, Serialize};

/// The core plugin trait that all plugins must implement.
pub trait Plugin: Send + Sync {
    /// Return metadata about this plugin.
    fn info(&self) -> PluginInfo;

    /// Initialize the plugin. Called once after loading.
    fn initialize(&mut self) -> crate::Result<()>;

    /// Shut down the plugin. Called before unloading.
    fn shutdown(&mut self) -> crate::Result<()>;
}

/// Metadata describing a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub plugin_type: PluginType,
    pub description: String,
    pub author: String,
    pub api_version: u32,
}

/// The type of functionality a plugin provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginType {
    Codec,
    Filter,
    Muxer,
    Demuxer,
    Analyzer,
}

/// Capabilities declared by a plugin.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginCapabilities {
    pub input_formats: Vec<String>,
    pub output_formats: Vec<String>,
    pub supports_gpu: bool,
    pub supports_async: bool,
    pub max_parallel: Option<usize>,
}

/// C-ABI compatible plugin API for native plugins.
///
/// Native plugins export a function with this signature:
/// ```c
/// extern "C" TranscodePluginApi* transcode_plugin_register();
/// ```
#[repr(C)]
pub struct PluginApi {
    pub magic: u64,
    pub api_version: u32,
    pub name: *const std::ffi::c_char,
    pub version: *const std::ffi::c_char,
    pub plugin_type: u32,
    pub init: Option<unsafe extern "C" fn() -> i32>,
    pub shutdown: Option<unsafe extern "C" fn() -> i32>,
}

// Safety: PluginApi is only used for FFI boundary definition
unsafe impl Send for PluginApi {}
unsafe impl Sync for PluginApi {}
