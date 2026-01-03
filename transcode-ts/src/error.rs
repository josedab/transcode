//! MPEG Transport Stream error types.
//!
//! This module provides error types specific to MPEG-TS parsing and muxing.

use thiserror::Error;

/// MPEG-TS specific errors.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TsError {
    /// Invalid sync byte (expected 0x47).
    #[error("Invalid sync byte: expected 0x47, got 0x{0:02X}")]
    InvalidSyncByte(u8),

    /// Packet too short.
    #[error("Packet too short: expected 188 bytes, got {0}")]
    PacketTooShort(usize),

    /// Invalid PID value.
    #[error("Invalid PID: {0}")]
    InvalidPid(u16),

    /// Continuity counter error.
    #[error("Continuity counter error on PID {pid}: expected {expected}, got {actual}")]
    ContinuityError {
        /// The PID with the error.
        pid: u16,
        /// Expected continuity counter.
        expected: u8,
        /// Actual continuity counter.
        actual: u8,
    },

    /// Transport error indicator set.
    #[error("Transport error indicator set on PID {0}")]
    TransportError(u16),

    /// Invalid adaptation field.
    #[error("Invalid adaptation field: {0}")]
    InvalidAdaptationField(String),

    /// Invalid PSI table.
    #[error("Invalid PSI table: {0}")]
    InvalidPsi(String),

    /// Invalid PAT (Program Association Table).
    #[error("Invalid PAT: {0}")]
    InvalidPat(String),

    /// Invalid PMT (Program Map Table).
    #[error("Invalid PMT: {0}")]
    InvalidPmt(String),

    /// Invalid PES packet.
    #[error("Invalid PES packet: {0}")]
    InvalidPes(String),

    /// CRC mismatch.
    #[error("CRC mismatch: expected 0x{expected:08X}, got 0x{actual:08X}")]
    CrcMismatch {
        /// Expected CRC value.
        expected: u32,
        /// Actual CRC value.
        actual: u32,
    },

    /// Program not found.
    #[error("Program {0} not found")]
    ProgramNotFound(u16),

    /// Stream not found.
    #[error("Stream with PID {0} not found")]
    StreamNotFound(u16),

    /// Unsupported stream type.
    #[error("Unsupported stream type: 0x{0:02X}")]
    UnsupportedStreamType(u8),

    /// Buffer overflow.
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),

    /// Timestamp error.
    #[error("Timestamp error: {0}")]
    TimestampError(String),

    /// PCR error.
    #[error("PCR error: {0}")]
    PcrError(String),

    /// Missing required data.
    #[error("Missing required data: {0}")]
    MissingData(String),
}

impl TsError {
    /// Create an invalid PSI error.
    pub fn invalid_psi(msg: impl Into<String>) -> Self {
        TsError::InvalidPsi(msg.into())
    }

    /// Create an invalid PAT error.
    pub fn invalid_pat(msg: impl Into<String>) -> Self {
        TsError::InvalidPat(msg.into())
    }

    /// Create an invalid PMT error.
    pub fn invalid_pmt(msg: impl Into<String>) -> Self {
        TsError::InvalidPmt(msg.into())
    }

    /// Create an invalid PES error.
    pub fn invalid_pes(msg: impl Into<String>) -> Self {
        TsError::InvalidPes(msg.into())
    }

    /// Create an invalid adaptation field error.
    pub fn invalid_adaptation_field(msg: impl Into<String>) -> Self {
        TsError::InvalidAdaptationField(msg.into())
    }
}

impl From<TsError> for transcode_core::error::Error {
    fn from(err: TsError) -> Self {
        transcode_core::error::Error::Container(
            transcode_core::error::ContainerError::InvalidStructure(err.to_string()),
        )
    }
}

/// Result type for MPEG-TS operations.
pub type Result<T> = std::result::Result<T, TsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TsError::InvalidSyncByte(0xFF);
        assert_eq!(err.to_string(), "Invalid sync byte: expected 0x47, got 0xFF");
    }

    #[test]
    fn test_continuity_error() {
        let err = TsError::ContinuityError {
            pid: 256,
            expected: 5,
            actual: 7,
        };
        assert_eq!(
            err.to_string(),
            "Continuity counter error on PID 256: expected 5, got 7"
        );
    }

    #[test]
    fn test_crc_mismatch() {
        let err = TsError::CrcMismatch {
            expected: 0xDEADBEEF,
            actual: 0xCAFEBABE,
        };
        assert_eq!(
            err.to_string(),
            "CRC mismatch: expected 0xDEADBEEF, got 0xCAFEBABE"
        );
    }

    #[test]
    fn test_conversion_to_core_error() {
        let ts_err = TsError::InvalidSyncByte(0xFF);
        let core_err: transcode_core::error::Error = ts_err.into();
        assert!(matches!(
            core_err,
            transcode_core::error::Error::Container(_)
        ));
    }
}
