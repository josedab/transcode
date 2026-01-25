//! Confidential computing support for secure transcoding in TEEs.
//!
//! This crate provides abstractions for processing encrypted video content
//! within Trusted Execution Environments (TEEs) like Intel SGX, AMD SEV,
//! ARM TrustZone, and AWS Nitro Enclaves.
//!
//! # Features
//!
//! - **Remote Attestation**: Verify enclave identity before key provisioning
//! - **Key Management**: Secure key derivation, wrapping, and sealing
//! - **Encrypted Processing**: Process content without exposing plaintext
//! - **Multi-TEE Support**: Abstraction layer for different TEE platforms
//!
//! # Security Model
//!
//! The confidential transcoding model ensures:
//!
//! 1. Content keys are only accessible inside the TEE
//! 2. Decrypted video frames never leave the enclave
//! 3. Remote parties can verify the enclave before provisioning keys
//! 4. Keys can be sealed for persistence across enclave restarts
//!
//! # Example
//!
//! ```rust
//! use transcode_confidential::{
//!     EnclaveBuilder, AttestationVerifier, ContentKey, TeeType,
//!     SealPolicy, generate_nonce,
//! };
//!
//! // Create an enclave
//! let enclave = EnclaveBuilder::new()
//!     .tee_type(TeeType::Simulated)
//!     .build()
//!     .unwrap();
//!
//! // Generate attestation for remote verification
//! let nonce = generate_nonce();
//! let report = enclave.attest(&nonce).unwrap();
//!
//! // Verify the attestation (on key server side)
//! let verifier = AttestationVerifier::new();
//! let result = verifier.verify(&report).unwrap();
//! assert!(result.overall_valid);
//!
//! // Generate and seal a content key
//! let key = ContentKey::generate().unwrap();
//! let sealed = enclave.seal(&key, SealPolicy::SignerIdentity).unwrap();
//!
//! // Later: unseal the key
//! let key_id = key.key_id().to_string();
//! let restored = enclave.unseal(&key_id, &sealed).unwrap();
//! ```
//!
//! # Integration with DRM
//!
//! This crate is designed to integrate with DRM systems:
//!
//! ```ignore
//! use transcode_confidential::{EnclaveBuilder, AttestationVerifier};
//! use transcode_drm::widevine::WidevineClient;
//!
//! // Create attested enclave
//! let enclave = EnclaveBuilder::new().build()?;
//! let report = enclave.attest(&nonce)?;
//!
//! // Provision keys via Widevine (after attestation verification)
//! let license = widevine.get_license(&report)?;
//! let key = enclave.import_key(&license.key)?;
//!
//! // Process encrypted content
//! for frame in encrypted_stream {
//!     let output = enclave.process_frame(&frame, &key, &nonce)?;
//!     // Output is re-encrypted, never plaintext outside enclave
//! }
//! ```
//!
//! # Platform Support
//!
//! | Platform | Status | Notes |
//! |----------|--------|-------|
//! | Intel SGX | Planned | Requires SGX-enabled CPU and SDK |
//! | AMD SEV | Planned | Requires SEV-enabled CPU |
//! | ARM TrustZone | Planned | Requires OP-TEE |
//! | AWS Nitro | Planned | Requires Nitro Enclaves |
//! | Simulated | Available | For development/testing |

#![allow(dead_code)]

mod attestation;
mod audit;
mod enclave;
mod error;
mod keys;

pub use attestation::{
    generate_nonce, generate_simulated_report, AttestationReport, AttestationVerifier,
    PlatformInfo, TeeType, VerificationResult,
};
pub use audit::{
    AuditEvent, AuditEventType, AuditLog, AuditSeverity, KeyRotationManager, KeyRotationPolicy,
    ManagedKey,
};
pub use enclave::{
    EnclaveBuilder, EnclaveCapabilities, EnclaveConfig, EnclaveState, EnclaveStats, SecureEnclave,
};
pub use error::{ConfidentialError, Result};
pub use keys::{ContentKey, KeyDerivation, KeyWrapKey, SealPolicy, SealedData, WrappedKey};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if hardware TEE is available on this system.
pub fn is_tee_available() -> TeeAvailability {
    TeeAvailability {
        sgx: check_sgx_available(),
        sev: check_sev_available(),
        trustzone: check_trustzone_available(),
        nitro: check_nitro_available(),
    }
}

/// TEE availability on the current system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeAvailability {
    pub sgx: bool,
    pub sev: bool,
    pub trustzone: bool,
    pub nitro: bool,
}

impl TeeAvailability {
    /// Check if any hardware TEE is available.
    pub fn any_available(&self) -> bool {
        self.sgx || self.sev || self.trustzone || self.nitro
    }

    /// Get the best available TEE type.
    pub fn best_available(&self) -> Option<TeeType> {
        if self.sgx {
            Some(TeeType::IntelSgx)
        } else if self.sev {
            Some(TeeType::AmdSev)
        } else if self.nitro {
            Some(TeeType::AwsNitro)
        } else if self.trustzone {
            Some(TeeType::ArmTrustZone)
        } else {
            None
        }
    }
}

use serde::{Deserialize, Serialize};

fn check_sgx_available() -> bool {
    // Would check /dev/sgx_enclave or cpuid
    false
}

fn check_sev_available() -> bool {
    // Would check /dev/sev or /sys/devices
    false
}

fn check_trustzone_available() -> bool {
    // Would check OP-TEE availability
    false
}

fn check_nitro_available() -> bool {
    // Would check /dev/nitro_enclaves
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_availability() {
        let avail = is_tee_available();
        // In test environment, no hardware TEE expected
        assert!(!avail.any_available());
    }

    #[test]
    fn test_end_to_end() {
        // Create enclave
        let enclave = EnclaveBuilder::new()
            .tee_type(TeeType::Simulated)
            .build()
            .unwrap();

        // Attest
        let nonce = generate_nonce();
        let report = enclave.attest(&nonce).unwrap();

        // Verify
        let verifier = AttestationVerifier::new();
        let result = verifier.verify(&report).unwrap();
        assert!(result.overall_valid);

        // Generate key
        let key = ContentKey::generate().unwrap();
        let key_id = key.key_id().to_string();

        // Seal
        let sealed = enclave.seal(&key, SealPolicy::SignerIdentity).unwrap();

        // Unseal
        let restored = enclave.unseal(&key_id, &sealed).unwrap();
        assert_eq!(restored.as_bytes(), key.as_bytes());

        // Process encrypted data
        let plaintext = b"video frame data";
        let nonce = [0u8; 12];
        let encrypted = key.encrypt(plaintext, &nonce).unwrap();

        let output = enclave.process_frame(&encrypted, &key, &nonce).unwrap();
        assert!(!output.is_empty());
    }
}
