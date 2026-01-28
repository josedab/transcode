//! Plugin loading from filesystem and WASM.

use std::path::{Path, PathBuf};
use tracing::{debug, warn};

use crate::api::PluginInfo;
use crate::error::{Error, Result};

/// Source of a loaded plugin.
#[derive(Debug, Clone)]
pub enum PluginSource {
    /// Native shared library (.so/.dylib/.dll).
    Native(PathBuf),
    /// WASM module (.wasm).
    Wasm(PathBuf),
    /// Built-in (compiled into the binary).
    Builtin,
}

/// A plugin that has been loaded into memory.
pub struct LoadedPlugin {
    pub info: PluginInfo,
    pub source: PluginSource,
}

/// Handles discovery and loading of plugins.
pub struct PluginLoader {
    search_paths: Vec<PathBuf>,
}

impl PluginLoader {
    pub fn new() -> Self {
        Self {
            search_paths: Vec::new(),
        }
    }

    /// Add a directory to search for plugins.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Discover all plugin files in search paths.
    pub fn discover(&self) -> Result<Vec<PathBuf>> {
        let mut found = Vec::new();
        for dir in &self.search_paths {
            if !dir.exists() {
                debug!(path = %dir.display(), "Plugin directory does not exist, skipping");
                continue;
            }
            match std::fs::read_dir(dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if Self::is_plugin_file(&path) {
                            found.push(path);
                        }
                    }
                }
                Err(e) => {
                    warn!(path = %dir.display(), error = %e, "Failed to read plugin directory");
                }
            }
        }
        Ok(found)
    }

    /// Validate a plugin file without loading it.
    pub fn validate(&self, path: &Path) -> Result<PluginSource> {
        if !path.exists() {
            return Err(Error::LoadFailed {
                path: path.display().to_string(),
                message: "File not found".into(),
            });
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "so" | "dylib" | "dll" => Ok(PluginSource::Native(path.to_path_buf())),
            "wasm" => Ok(PluginSource::Wasm(path.to_path_buf())),
            _ => Err(Error::InvalidBinary {
                message: format!("Unsupported plugin extension: .{}", ext),
            }),
        }
    }

    fn is_plugin_file(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| matches!(ext, "so" | "dylib" | "dll" | "wasm"))
            .unwrap_or(false)
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_discover_empty() {
        let loader = PluginLoader::new();
        let found = loader.discover().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn test_discover_finds_plugins() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("codec.so"), b"fake").unwrap();
        fs::write(dir.path().join("filter.wasm"), b"fake").unwrap();
        fs::write(dir.path().join("readme.txt"), b"not a plugin").unwrap();

        let mut loader = PluginLoader::new();
        loader.add_search_path(dir.path());
        let found = loader.discover().unwrap();
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_validate_native() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.dylib");
        fs::write(&path, b"fake").unwrap();

        let loader = PluginLoader::new();
        let source = loader.validate(&path).unwrap();
        assert!(matches!(source, PluginSource::Native(_)));
    }

    #[test]
    fn test_validate_wasm() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wasm");
        fs::write(&path, b"fake").unwrap();

        let loader = PluginLoader::new();
        let source = loader.validate(&path).unwrap();
        assert!(matches!(source, PluginSource::Wasm(_)));
    }

    #[test]
    fn test_validate_invalid_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, b"fake").unwrap();

        let loader = PluginLoader::new();
        assert!(loader.validate(&path).is_err());
    }
}
