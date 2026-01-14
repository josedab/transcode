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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_initialization_failed() {
        let err = GpuError::InitializationFailed("test reason".to_string());
        assert_eq!(err.to_string(), "GPU initialization failed: test reason");
    }

    #[test]
    fn test_error_display_no_adapter() {
        let err = GpuError::NoAdapter;
        assert_eq!(err.to_string(), "No compatible GPU adapter found");
    }

    #[test]
    fn test_error_display_device_creation_failed() {
        let err = GpuError::DeviceCreationFailed("device error".to_string());
        assert_eq!(err.to_string(), "Device creation failed: device error");
    }

    #[test]
    fn test_error_display_shader_compilation_failed() {
        let err = GpuError::ShaderCompilationFailed("syntax error".to_string());
        assert_eq!(err.to_string(), "Shader compilation failed: syntax error");
    }

    #[test]
    fn test_error_display_pipeline_creation_failed() {
        let err = GpuError::PipelineCreationFailed("pipeline error".to_string());
        assert_eq!(err.to_string(), "Pipeline creation failed: pipeline error");
    }

    #[test]
    fn test_error_display_buffer_creation_failed() {
        let err = GpuError::BufferCreationFailed("buffer error".to_string());
        assert_eq!(err.to_string(), "Buffer creation failed: buffer error");
    }

    #[test]
    fn test_error_display_texture_creation_failed() {
        let err = GpuError::TextureCreationFailed("texture error".to_string());
        assert_eq!(err.to_string(), "Texture creation failed: texture error");
    }

    #[test]
    fn test_error_display_invalid_dimensions() {
        let err = GpuError::InvalidDimensions {
            width: 20000,
            height: 10000,
            max_dimension: 16384,
        };
        assert_eq!(
            err.to_string(),
            "Invalid dimensions: 20000x10000 (max: 16384)"
        );
    }

    #[test]
    fn test_error_display_unsupported_format() {
        let err = GpuError::UnsupportedFormat("YUV420".to_string());
        assert_eq!(err.to_string(), "Unsupported pixel format: YUV420");
    }

    #[test]
    fn test_error_display_buffer_size_mismatch() {
        let err = GpuError::BufferSizeMismatch {
            expected: 1024,
            actual: 512,
        };
        assert_eq!(
            err.to_string(),
            "Buffer size mismatch: expected 1024, got 512"
        );
    }

    #[test]
    fn test_error_display_submission_failed() {
        let err = GpuError::SubmissionFailed("queue error".to_string());
        assert_eq!(err.to_string(), "Command submission failed: queue error");
    }

    #[test]
    fn test_error_display_mapping_failed() {
        let err = GpuError::MappingFailed("map error".to_string());
        assert_eq!(err.to_string(), "Buffer mapping failed: map error");
    }

    #[test]
    fn test_error_display_timeout() {
        let err = GpuError::Timeout(5000);
        assert_eq!(err.to_string(), "Operation timed out after 5000ms");
    }

    #[test]
    fn test_error_display_device_lost() {
        let err = GpuError::DeviceLost;
        assert_eq!(err.to_string(), "GPU device lost");
    }

    #[test]
    fn test_error_display_out_of_memory() {
        let err = GpuError::OutOfMemory;
        assert_eq!(err.to_string(), "Out of GPU memory");
    }

    #[test]
    fn test_error_display_invalid_config() {
        let err = GpuError::InvalidConfig("bad config".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: bad config");
    }

    #[test]
    fn test_error_into_transcode_core_error() {
        let gpu_err = GpuError::NoAdapter;
        let core_err: transcode_core::Error = gpu_err.into();
        assert!(core_err.to_string().contains("No compatible GPU adapter found"));
    }
}
