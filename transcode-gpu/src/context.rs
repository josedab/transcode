//! GPU context and device management.

use crate::error::{GpuError, Result};
use crate::GpuCapabilities;
use std::sync::Arc;
use tracing::{debug, info};

/// GPU context holding device and queue.
pub struct GpuContext {
    /// wgpu instance.
    instance: wgpu::Instance,
    /// GPU adapter.
    adapter: wgpu::Adapter,
    /// GPU device.
    device: Arc<wgpu::Device>,
    /// Command queue.
    queue: Arc<wgpu::Queue>,
    /// Device capabilities.
    capabilities: GpuCapabilities,
}

impl GpuContext {
    /// Create a new GPU context with default settings.
    pub async fn new() -> Result<Self> {
        Self::with_config(GpuContextConfig::default()).await
    }

    /// Create a GPU context with custom configuration.
    pub async fn with_config(config: GpuContextConfig) -> Result<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: config.backends,
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: config.force_fallback,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        info!(
            "Using GPU: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        // Request device with compute capabilities
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Transcode GPU Device"),
                    required_features: config.required_features,
                    required_limits: config.required_limits.unwrap_or_else(|| {
                        // Use downlevel defaults for maximum compatibility
                        wgpu::Limits::downlevel_defaults()
                    }),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let limits = device.limits();
        let capabilities = GpuCapabilities::from_adapter(&adapter_info, &limits);

        debug!("GPU capabilities: {:?}", capabilities);

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        })
    }

    /// Get the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a clone of the device Arc.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Get the command queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a clone of the queue Arc.
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }

    /// Get device capabilities.
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Check if a texture size is valid.
    pub fn validate_texture_size(&self, width: u32, height: u32) -> Result<()> {
        let max = self.capabilities.max_texture_dimension;
        if width > max || height > max {
            return Err(GpuError::InvalidDimensions {
                width,
                height,
                max_dimension: max,
            });
        }
        Ok(())
    }

    /// Create a compute command encoder.
    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            })
    }

    /// Submit commands and wait for completion.
    pub fn submit(&self, commands: Vec<wgpu::CommandBuffer>) {
        self.queue.submit(commands);
    }

    /// Poll the device for completion.
    pub fn poll(&self, maintain: wgpu::Maintain) -> wgpu::MaintainResult {
        self.device.poll(maintain)
    }

    /// Create a staging buffer for reading back results.
    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a GPU buffer with initial data.
    pub fn create_buffer_init(&self, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }
}

/// Configuration for GPU context creation.
#[derive(Debug, Clone)]
pub struct GpuContextConfig {
    /// Backends to use (Vulkan, Metal, DX12, etc.)
    pub backends: wgpu::Backends,
    /// Power preference.
    pub power_preference: wgpu::PowerPreference,
    /// Required features.
    pub required_features: wgpu::Features,
    /// Required limits.
    pub required_limits: Option<wgpu::Limits>,
    /// Force software fallback.
    pub force_fallback: bool,
}

impl Default for GpuContextConfig {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: None,
            force_fallback: false,
        }
    }
}

impl GpuContextConfig {
    /// Create config preferring high performance GPU.
    pub fn high_performance() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }
    }

    /// Create config preferring low power GPU.
    pub fn low_power() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::LowPower,
            ..Default::default()
        }
    }

    /// Set specific backend.
    pub fn with_backend(mut self, backend: wgpu::Backends) -> Self {
        self.backends = backend;
        self
    }

    /// Add required features.
    pub fn with_features(mut self, features: wgpu::Features) -> Self {
        self.required_features |= features;
        self
    }
}
