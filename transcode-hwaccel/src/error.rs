//! Error types for hardware acceleration.

use thiserror::Error;

/// Hardware acceleration error.
#[derive(Error, Debug)]
pub enum HwAccelError {
    /// No hardware accelerator available.
    #[error("No hardware accelerator available")]
    NoAccelerator,

    /// Accelerator not supported on this platform.
    #[error("Accelerator {0} not supported on this platform")]
    NotSupported(String),

    /// Codec not supported by hardware.
    #[error("Codec {0} not supported by hardware accelerator")]
    CodecNotSupported(String),

    /// Device initialization failed.
    #[error("Failed to initialize device: {0}")]
    DeviceInit(String),

    /// Encoding error.
    #[error("Hardware encoding failed: {0}")]
    Encode(String),

    /// Decoding error.
    #[error("Hardware decoding failed: {0}")]
    Decode(String),

    /// Buffer allocation error.
    #[error("Failed to allocate hardware buffer: {0}")]
    BufferAlloc(String),

    /// Memory transfer error.
    #[error("Failed to transfer data: {0}")]
    Transfer(String),

    /// Configuration error.
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Driver error.
    #[error("Driver error: {0}")]
    Driver(String),

    /// Feature not available.
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Resource exhausted.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Device error.
    #[error("Device error: {0}")]
    DeviceError(String),

    /// CUDA error.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// NVENC API error.
    #[error("NVENC error: {0}")]
    NvencError(String),
}

/// Result type for hardware acceleration operations.
pub type Result<T> = std::result::Result<T, HwAccelError>;
