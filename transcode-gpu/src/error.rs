//! GPU error types.

use thiserror::Error;

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;

/// GPU-specific errors.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to initialize GPU.
    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    /// No compatible GPU adapter found.
    #[error("No compatible GPU adapter found")]
    NoAdapter,

    /// Failed to create device.
    #[error("Device creation failed: {0}")]
    DeviceCreationFailed(String),

    /// Shader compilation error.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    /// Pipeline creation error.
    #[error("Pipeline creation failed: {0}")]
    PipelineCreationFailed(String),

    /// Buffer creation error.
    #[error("Buffer creation failed: {0}")]
    BufferCreationFailed(String),

    /// Texture creation error.
    #[error("Texture creation failed: {0}")]
    TextureCreationFailed(String),

    /// Invalid texture dimensions.
    #[error("Invalid dimensions: {width}x{height} (max: {max_dimension})")]
    InvalidDimensions {
        width: u32,
        height: u32,
        max_dimension: u32,
    },

    /// Unsupported pixel format.
    #[error("Unsupported pixel format: {0}")]
    UnsupportedFormat(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// GPU command submission failed.
    #[error("Command submission failed: {0}")]
    SubmissionFailed(String),

    /// Buffer mapping failed.
    #[error("Buffer mapping failed: {0}")]
    MappingFailed(String),

    /// Operation timed out.
    #[error("Operation timed out after {0}ms")]
    Timeout(u64),

    /// Device lost.
    #[error("GPU device lost")]
    DeviceLost,

    /// Out of GPU memory.
    #[error("Out of GPU memory")]
    OutOfMemory,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        GpuError::DeviceCreationFailed(err.to_string())
    }
}

impl From<wgpu::CreateSurfaceError> for GpuError {
    fn from(err: wgpu::CreateSurfaceError) -> Self {
        GpuError::InitializationFailed(err.to_string())
    }
}

impl From<GpuError> for transcode_core::Error {
    fn from(err: GpuError) -> Self {
        transcode_core::Error::Unsupported(err.to_string())
    }
}
