//! WASM error types.

#![allow(dead_code)]

use wasm_bindgen::prelude::*;

/// Error type for WASM operations.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmError {
    message: String,
    code: ErrorCode,
}

/// Error codes for categorizing errors.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Unknown error.
    Unknown = 0,
    /// Invalid input data.
    InvalidInput = 1,
    /// Codec error during encoding/decoding.
    CodecError = 2,
    /// Container format error.
    ContainerError = 3,
    /// I/O error.
    IoError = 4,
    /// Out of memory.
    OutOfMemory = 5,
    /// Feature not supported.
    NotSupported = 6,
    /// Invalid configuration.
    InvalidConfig = 7,
    /// Operation cancelled.
    Cancelled = 8,
    /// Browser feature not available.
    BrowserNotSupported = 9,
}

#[wasm_bindgen]
impl WasmError {
    /// Create a new error with message and code.
    #[wasm_bindgen(constructor)]
    pub fn new(message: &str, code: ErrorCode) -> Self {
        Self {
            message: message.to_string(),
            code,
        }
    }

    /// Get the error message.
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    /// Get the error code.
    #[wasm_bindgen(getter)]
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Create an invalid input error.
    pub fn invalid_input(msg: &str) -> Self {
        Self::new(msg, ErrorCode::InvalidInput)
    }

    /// Create a codec error.
    pub fn codec_error(msg: &str) -> Self {
        Self::new(msg, ErrorCode::CodecError)
    }

    /// Create a not supported error.
    pub fn not_supported(msg: &str) -> Self {
        Self::new(msg, ErrorCode::NotSupported)
    }

    /// Create an invalid config error.
    pub fn invalid_config(msg: &str) -> Self {
        Self::new(msg, ErrorCode::InvalidConfig)
    }

    /// Create a browser not supported error.
    pub fn browser_not_supported(msg: &str) -> Self {
        Self::new(msg, ErrorCode::BrowserNotSupported)
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

impl std::error::Error for WasmError {}

impl From<transcode_core::Error> for WasmError {
    fn from(err: transcode_core::Error) -> Self {
        match err {
            transcode_core::Error::Io(e) => Self::new(&e.to_string(), ErrorCode::IoError),
            transcode_core::Error::Codec(e) => Self::new(&e.to_string(), ErrorCode::CodecError),
            transcode_core::Error::Container(e) => {
                Self::new(&e.to_string(), ErrorCode::ContainerError)
            }
            transcode_core::Error::Config(msg) => Self::new(&msg, ErrorCode::InvalidConfig),
            _ => Self::new(&err.to_string(), ErrorCode::Unknown),
        }
    }
}

/// Result type for WASM operations.
pub type WasmResult<T> = Result<T, WasmError>;

/// Convert a WasmResult to a JsValue (for returning to JavaScript).
pub fn to_js_result<T: Into<JsValue>>(result: WasmResult<T>) -> Result<JsValue, JsValue> {
    match result {
        Ok(value) => Ok(value.into()),
        Err(err) => Err(JsValue::from_str(&err.to_string())),
    }
}
