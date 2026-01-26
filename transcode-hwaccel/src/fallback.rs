//! Automatic hardware encoder fallback chain.
//!
//! Provides a [`FallbackEncoder`] that tries hardware encoders in priority order,
//! automatically falling back to the next option when one fails. The default
//! chain is: NVENC → QSV → VA-API → VideoToolbox → Software.

use tracing::{info, warn};

use crate::encoder::{HwEncoder, HwEncoderConfig, HwPacket};
use crate::error::{HwAccelError, Result};
use crate::types::HwFrame;
use crate::HwAccelType;

/// Defines the order in which hardware encoders are tried.
#[derive(Debug, Clone)]
pub struct FallbackChain {
    backends: Vec<HwAccelType>,
}

impl FallbackChain {
    /// Create a chain with the given priority order.
    pub fn new(backends: Vec<HwAccelType>) -> Self {
        Self { backends }
    }

    /// Default platform-aware chain.
    #[allow(clippy::vec_init_then_push)]
    pub fn platform_default() -> Self {
        let mut backends = Vec::new();

        // GPU encoders first (highest throughput)
        backends.push(HwAccelType::Nvenc);
        backends.push(HwAccelType::Qsv);

        // Platform-specific
        #[cfg(target_os = "linux")]
        backends.push(HwAccelType::Vaapi);

        #[cfg(target_os = "macos")]
        backends.push(HwAccelType::VideoToolbox);

        // Always end with software
        backends.push(HwAccelType::Software);

        Self { backends }
    }

    /// Filter to only backends available on this system.
    pub fn available_only(&self) -> Vec<HwAccelType> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .copied()
            .collect()
    }

    pub fn backends(&self) -> &[HwAccelType] {
        &self.backends
    }
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self::platform_default()
    }
}

/// An encoder that automatically falls back through a chain of hardware backends.
pub struct FallbackEncoder {
    chain: FallbackChain,
    config: HwEncoderConfig,
    active_encoder: Option<HwEncoder>,
    active_backend: Option<HwAccelType>,
    tried_backends: Vec<(HwAccelType, String)>,
}

impl FallbackEncoder {
    /// Create a new fallback encoder with the platform-default chain.
    pub fn new(config: HwEncoderConfig) -> Self {
        Self {
            chain: FallbackChain::platform_default(),
            config,
            active_encoder: None,
            active_backend: None,
            tried_backends: Vec::new(),
        }
    }

    /// Create with a custom fallback chain.
    pub fn with_chain(config: HwEncoderConfig, chain: FallbackChain) -> Self {
        Self {
            chain,
            config,
            active_encoder: None,
            active_backend: None,
            tried_backends: Vec::new(),
        }
    }

    /// Initialize — tries each backend in the chain until one succeeds.
    pub fn init(&mut self) -> Result<HwAccelType> {
        let available = self.chain.available_only();

        if available.is_empty() {
            return Err(HwAccelError::NotSupported(
                "No hardware encoders available on this system".into(),
            ));
        }

        for backend in &available {
            info!(backend = backend.name(), "Trying hardware encoder");

            match HwEncoder::new(*backend, self.config.clone()) {
                Ok(mut encoder) => match encoder.init() {
                    Ok(()) => {
                        info!(
                            backend = backend.name(),
                            "Hardware encoder initialized successfully"
                        );
                        self.active_encoder = Some(encoder);
                        self.active_backend = Some(*backend);
                        return Ok(*backend);
                    }
                    Err(e) => {
                        let msg = format!("Init failed: {}", e);
                        warn!(backend = backend.name(), error = %msg, "Falling back");
                        self.tried_backends.push((*backend, msg));
                    }
                },
                Err(e) => {
                    let msg = format!("Creation failed: {}", e);
                    warn!(backend = backend.name(), error = %msg, "Falling back");
                    self.tried_backends.push((*backend, msg));
                }
            }
        }

        Err(HwAccelError::NotSupported(format!(
            "All hardware encoders failed. Tried: {}",
            self.tried_backends
                .iter()
                .map(|(b, e)| format!("{}: {}", b.name(), e))
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }

    /// Encode a frame using the active encoder.
    pub fn encode(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        let encoder = self.active_encoder.as_mut().ok_or_else(|| {
            HwAccelError::Config("Encoder not initialized — call init() first".into())
        })?;
        encoder.encode(frame)
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<HwPacket>> {
        match self.active_encoder.as_mut() {
            Some(encoder) => encoder.flush(),
            None => Ok(Vec::new()),
        }
    }

    /// Which backend is currently active.
    pub fn active_backend(&self) -> Option<HwAccelType> {
        self.active_backend
    }

    /// List of backends that were tried and failed, with error messages.
    pub fn tried_backends(&self) -> &[(HwAccelType, String)] {
        &self.tried_backends
    }

    /// Whether the encoder is initialized.
    pub fn is_initialized(&self) -> bool {
        self.active_encoder.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HwSurfaceFormat;

    #[test]
    fn test_fallback_chain_default() {
        let chain = FallbackChain::platform_default();
        let backends = chain.backends();
        // Should always end with Software
        assert_eq!(*backends.last().unwrap(), HwAccelType::Software);
    }

    #[test]
    fn test_fallback_chain_available() {
        let chain = FallbackChain::platform_default();
        let available = chain.available_only();
        // Software should always be available
        assert!(available.contains(&HwAccelType::Software));
    }

    #[test]
    fn test_fallback_encoder_initializes() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = FallbackEncoder::new(config);
        let backend = encoder.init().unwrap();
        // Should succeed (at least software fallback)
        assert!(encoder.is_initialized());
        assert_eq!(encoder.active_backend(), Some(backend));
    }

    #[test]
    fn test_fallback_encoder_encode() {
        let config = HwEncoderConfig::h264(640, 480, 1_000_000);
        let mut encoder = FallbackEncoder::new(config);
        encoder.init().unwrap();

        let frame = HwFrame::new_cpu(
            vec![0u8; 640 * 480 * 3 / 2],
            640,
            480,
            HwSurfaceFormat::Nv12,
        );
        let packet = encoder.encode(&frame).unwrap();
        assert!(packet.is_some());
    }

    #[test]
    fn test_fallback_custom_chain() {
        let chain = FallbackChain::new(vec![HwAccelType::Software]);
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = FallbackEncoder::with_chain(config, chain);
        let backend = encoder.init().unwrap();
        assert_eq!(backend, HwAccelType::Software);
    }

    #[test]
    fn test_fallback_encode_without_init() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = FallbackEncoder::new(config);
        let frame = HwFrame::new_cpu(
            vec![0u8; 1920 * 1080 * 3 / 2],
            1920,
            1080,
            HwSurfaceFormat::Nv12,
        );
        assert!(encoder.encode(&frame).is_err());
    }

    #[test]
    fn test_fallback_empty_chain_fails() {
        let chain = FallbackChain::new(vec![]);
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = FallbackEncoder::with_chain(config, chain);
        assert!(encoder.init().is_err());
    }
}
