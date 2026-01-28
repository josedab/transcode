//! Plugin SDK for extending transcode with dynamic codec, filter, and muxer plugins.
//!
//! This crate provides the infrastructure for loading and managing plugins
//! that extend the transcode ecosystem. Plugins can provide:
//!
//! - **Codecs**: Custom video/audio encoders and decoders
//! - **Filters**: Video/audio processing filters
//! - **Muxers**: Container format muxers/demuxers
//!
//! # Plugin Types
//!
//! - **Native plugins**: Shared libraries (.so/.dylib/.dll) loaded at runtime
//! - **WASM plugins**: WebAssembly modules running in a sandboxed environment
//!
//! # Example
//!
//! ```
//! use transcode_plugin::{PluginRegistry, PluginInfo, PluginType};
//!
//! let mut registry = PluginRegistry::new();
//!
//! // Discover and load plugins from a directory
//! // registry.load_directory("/usr/lib/transcode/plugins").unwrap();
//!
//! // Query available plugins
//! let codecs = registry.list_by_type(PluginType::Codec);
//! for info in codecs {
//!     println!("Codec plugin: {} v{}", info.name, info.version);
//! }
//! ```
//!
//! # Writing a Plugin
//!
//! Plugins implement the `Plugin` trait and export a C-ABI registration function:
//!
//! ```ignore
//! use transcode_plugin::{Plugin, PluginInfo, PluginType, PluginApi};
//!
//! struct MyFilter;
//!
//! impl Plugin for MyFilter {
//!     fn info(&self) -> PluginInfo {
//!         PluginInfo {
//!             name: "my-filter".into(),
//!             version: "1.0.0".into(),
//!             plugin_type: PluginType::Filter,
//!             description: "My custom video filter".into(),
//!             author: "Author".into(),
//!             api_version: transcode_plugin::API_VERSION,
//!         }
//!     }
//!
//!     fn initialize(&mut self) -> transcode_plugin::Result<()> {
//!         Ok(())
//!     }
//!
//!     fn shutdown(&mut self) -> transcode_plugin::Result<()> {
//!         Ok(())
//!     }
//! }
//! ```

#![allow(dead_code)]

mod error;
mod registry;
mod api;
mod loader;
mod runtime;
mod sandbox;

pub use error::{Error, Result};
pub use registry::PluginRegistry;
pub use api::{Plugin, PluginInfo, PluginType, PluginCapabilities, PluginApi};
pub use loader::{PluginLoader, LoadedPlugin, PluginSource};
pub use runtime::{PluginInstance, PluginRuntime, PluginState, PluginStats};
pub use sandbox::{SandboxConfig, SandboxPolicy};

/// Current plugin API version. Plugins must match this to be loaded.
pub const API_VERSION: u32 = 1;

/// Plugin ABI magic number for native plugin validation.
pub const ABI_MAGIC: u64 = 0x5452_414E_5343_4F44; // "TRANSCOD" in ASCII

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_operations() {
        let mut registry = PluginRegistry::new();
        assert!(registry.list_all().is_empty());

        let info = PluginInfo {
            name: "test-codec".into(),
            version: "1.0.0".into(),
            plugin_type: PluginType::Codec,
            description: "Test codec plugin".into(),
            author: "Test".into(),
            api_version: API_VERSION,
        };

        registry.register(info.clone()).unwrap();
        assert_eq!(registry.list_all().len(), 1);
        assert_eq!(registry.list_by_type(PluginType::Codec).len(), 1);
        assert_eq!(registry.list_by_type(PluginType::Filter).len(), 0);
    }

    #[test]
    fn test_duplicate_registration_rejected() {
        let mut registry = PluginRegistry::new();
        let info = PluginInfo {
            name: "dup-plugin".into(),
            version: "1.0.0".into(),
            plugin_type: PluginType::Filter,
            description: "".into(),
            author: "".into(),
            api_version: API_VERSION,
        };
        registry.register(info.clone()).unwrap();
        assert!(registry.register(info).is_err());
    }

    #[test]
    fn test_api_version_mismatch() {
        let mut registry = PluginRegistry::new();
        let info = PluginInfo {
            name: "old-plugin".into(),
            version: "0.1.0".into(),
            plugin_type: PluginType::Codec,
            description: "".into(),
            author: "".into(),
            api_version: 0, // old version
        };
        assert!(registry.register(info).is_err());
    }

    #[test]
    fn test_plugin_lookup() {
        let mut registry = PluginRegistry::new();
        let info = PluginInfo {
            name: "my-filter".into(),
            version: "2.0.0".into(),
            plugin_type: PluginType::Filter,
            description: "A great filter".into(),
            author: "Dev".into(),
            api_version: API_VERSION,
        };
        registry.register(info).unwrap();

        assert!(registry.get("my-filter").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
