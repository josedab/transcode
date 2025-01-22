//! Enclave abstraction for TEE-based processing.

use crate::attestation::{generate_simulated_report, AttestationReport, TeeType};
use crate::error::{ConfidentialError, Result};
use crate::keys::{ContentKey, KeyWrapKey, SealPolicy, SealedData, WrappedKey};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Enclave capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveCapabilities {
    /// Maximum memory available.
    pub max_memory_mb: usize,
    /// Supported encryption algorithms.
    pub supported_algorithms: Vec<String>,
    /// Supports key sealing.
    pub supports_sealing: bool,
    /// Supports remote attestation.
    pub supports_attestation: bool,
    /// Maximum concurrent operations.
    pub max_concurrent_ops: usize,
}

impl Default for EnclaveCapabilities {
    fn default() -> Self {
        Self {
            max_memory_mb: 256,
            supported_algorithms: vec![
                "AES-256-GCM".to_string(),
                "ChaCha20-Poly1305".to_string(),
            ],
            supports_sealing: true,
            supports_attestation: true,
            max_concurrent_ops: 4,
        }
    }
}

/// Enclave state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnclaveState {
    /// Enclave not initialized.
    Uninitialized,
    /// Enclave initialized and ready.
    Ready,
    /// Enclave is processing.
    Processing,
    /// Enclave is shutting down.
    ShuttingDown,
    /// Enclave has terminated.
    Terminated,
}

/// Statistics from enclave operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnclaveStats {
    pub bytes_encrypted: u64,
    pub bytes_decrypted: u64,
    pub frames_processed: u64,
    pub attestations_performed: u32,
    pub keys_sealed: u32,
    pub keys_unsealed: u32,
}

/// Simulated secure enclave for development and testing.
///
/// In production, this would be replaced with actual TEE implementations
/// (Intel SGX SDK, AWS Nitro Enclaves SDK, etc.)
pub struct SecureEnclave {
    tee_type: TeeType,
    state: RwLock<EnclaveState>,
    capabilities: EnclaveCapabilities,
    /// Sealed keys (key_id -> sealed data)
    sealed_keys: RwLock<HashMap<String, SealedData>>,
    /// Sealing key (derived from enclave identity)
    sealing_key: KeyWrapKey,
    /// Statistics
    stats: RwLock<EnclaveStats>,
}

impl SecureEnclave {
    /// Create a new simulated enclave.
    pub fn new_simulated() -> Result<Arc<Self>> {
        let sealing_key = KeyWrapKey::generate()?;

        Ok(Arc::new(Self {
            tee_type: TeeType::Simulated,
            state: RwLock::new(EnclaveState::Ready),
            capabilities: EnclaveCapabilities::default(),
            sealed_keys: RwLock::new(HashMap::new()),
            sealing_key,
            stats: RwLock::new(EnclaveStats::default()),
        }))
    }

    /// Get TEE type.
    pub fn tee_type(&self) -> TeeType {
        self.tee_type
    }

    /// Get enclave state.
    pub fn state(&self) -> EnclaveState {
        *self.state.read()
    }

    /// Get capabilities.
    pub fn capabilities(&self) -> &EnclaveCapabilities {
        &self.capabilities
    }

    /// Get statistics.
    pub fn stats(&self) -> EnclaveStats {
        self.stats.read().clone()
    }

    /// Generate attestation report.
    pub fn attest(&self, nonce: &[u8]) -> Result<AttestationReport> {
        if *self.state.read() != EnclaveState::Ready {
            return Err(ConfidentialError::TeeNotAvailable(
                "Enclave not ready".to_string(),
            ));
        }

        self.stats.write().attestations_performed += 1;
        Ok(generate_simulated_report(nonce))
    }

    /// Seal data for persistence.
    pub fn seal(&self, key: &ContentKey, policy: SealPolicy) -> Result<SealedData> {
        let wrapped = self.sealing_key.wrap(key)?;

        let sealed = SealedData {
            ciphertext: wrapped.wrapped_key,
            nonce: wrapped.nonce,
            tag: vec![], // Tag is included in ciphertext for AES-GCM
            seal_policy: policy,
        };

        // Store in memory for later unsealing
        self.sealed_keys
            .write()
            .insert(key.key_id().to_string(), sealed.clone());

        self.stats.write().keys_sealed += 1;
        Ok(sealed)
    }

    /// Unseal data.
    pub fn unseal(&self, key_id: &str, sealed: &SealedData) -> Result<ContentKey> {
        let wrapped = WrappedKey {
            key_id: key_id.to_string(),
            wrapped_key: sealed.ciphertext.clone(),
            nonce: sealed.nonce.clone(),
            algorithm: "AES-256-GCM-WRAP".to_string(),
        };

        let key = self.sealing_key.unwrap(&wrapped)?;
        self.stats.write().keys_unsealed += 1;
        Ok(key)
    }

    /// Process encrypted video frame inside enclave.
    ///
    /// This simulates processing encrypted content without exposing plaintext
    /// outside the enclave boundary.
    pub fn process_frame(
        &self,
        encrypted_frame: &[u8],
        key: &ContentKey,
        nonce: &[u8; 12],
    ) -> Result<Vec<u8>> {
        if *self.state.read() != EnclaveState::Ready {
            return Err(ConfidentialError::TeeNotAvailable(
                "Enclave not ready".to_string(),
            ));
        }

        // Decrypt inside enclave
        let plaintext = key.decrypt(encrypted_frame, nonce)?;
        self.stats.write().bytes_decrypted += encrypted_frame.len() as u64;

        // Process frame (in reality, would do transcoding here)
        // For simulation, just return as-is
        let processed = plaintext;

        // Re-encrypt for output
        let output_nonce = increment_nonce(nonce);
        let output = key.encrypt(&processed, &output_nonce)?;
        self.stats.write().bytes_encrypted += output.len() as u64;
        self.stats.write().frames_processed += 1;

        // Prepend nonce to output
        let mut result = output_nonce.to_vec();
        result.extend(output);

        Ok(result)
    }

    /// Shutdown the enclave securely.
    pub fn shutdown(&self) -> Result<()> {
        let mut state = self.state.write();
        *state = EnclaveState::ShuttingDown;

        // Clear sensitive data
        self.sealed_keys.write().clear();

        *state = EnclaveState::Terminated;
        tracing::info!("Enclave shutdown complete");
        Ok(())
    }
}

/// Enclave configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveConfig {
    /// TEE type to use.
    pub tee_type: TeeType,
    /// Maximum memory allocation.
    pub max_memory_mb: usize,
    /// Enable debug mode (less secure, more logging).
    pub debug_mode: bool,
    /// Key rotation interval in seconds.
    pub key_rotation_secs: u64,
}

impl Default for EnclaveConfig {
    fn default() -> Self {
        Self {
            tee_type: TeeType::Simulated,
            max_memory_mb: 256,
            debug_mode: false,
            key_rotation_secs: 3600,
        }
    }
}

/// Builder for creating enclaves.
pub struct EnclaveBuilder {
    config: EnclaveConfig,
}

impl Default for EnclaveBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EnclaveBuilder {
    pub fn new() -> Self {
        Self {
            config: EnclaveConfig::default(),
        }
    }

    pub fn tee_type(mut self, tee_type: TeeType) -> Self {
        self.config.tee_type = tee_type;
        self
    }

    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.config.max_memory_mb = mb;
        self
    }

    pub fn debug_mode(mut self, enabled: bool) -> Self {
        self.config.debug_mode = enabled;
        self
    }

    pub fn build(self) -> Result<Arc<SecureEnclave>> {
        match self.config.tee_type {
            TeeType::Simulated => SecureEnclave::new_simulated(),
            _ => {
                // In production, would initialize actual TEE
                // For now, fall back to simulation
                tracing::warn!(
                    "TEE type {:?} not available, using simulation",
                    self.config.tee_type
                );
                SecureEnclave::new_simulated()
            }
        }
    }
}

fn increment_nonce(nonce: &[u8; 12]) -> [u8; 12] {
    let mut result = *nonce;
    for i in (0..12).rev() {
        result[i] = result[i].wrapping_add(1);
        if result[i] != 0 {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enclave_creation() {
        let enclave = SecureEnclave::new_simulated().unwrap();
        assert_eq!(enclave.tee_type(), TeeType::Simulated);
        assert_eq!(enclave.state(), EnclaveState::Ready);
    }

    #[test]
    fn test_enclave_attestation() {
        let enclave = SecureEnclave::new_simulated().unwrap();
        let nonce = vec![1, 2, 3, 4];

        let report = enclave.attest(&nonce).unwrap();
        assert_eq!(report.tee_type, TeeType::Simulated);
        assert_eq!(report.report_data, nonce);
    }

    #[test]
    fn test_enclave_seal_unseal() {
        let enclave = SecureEnclave::new_simulated().unwrap();
        let key = ContentKey::generate().unwrap();
        let key_id = key.key_id().to_string();
        let original_bytes = *key.as_bytes();

        let sealed = enclave.seal(&key, SealPolicy::SignerIdentity).unwrap();
        let unsealed = enclave.unseal(&key_id, &sealed).unwrap();

        assert_eq!(unsealed.as_bytes(), &original_bytes);
    }

    #[test]
    fn test_enclave_process_frame() {
        let enclave = SecureEnclave::new_simulated().unwrap();
        let key = ContentKey::generate().unwrap();
        let nonce = [0u8; 12];

        let plaintext = b"frame data here";
        let encrypted = key.encrypt(plaintext, &nonce).unwrap();

        let result = enclave.process_frame(&encrypted, &key, &nonce).unwrap();

        // Result should have nonce prepended
        assert!(result.len() > encrypted.len());
    }

    #[test]
    fn test_enclave_builder() {
        let enclave = EnclaveBuilder::new()
            .tee_type(TeeType::Simulated)
            .max_memory_mb(512)
            .debug_mode(true)
            .build()
            .unwrap();

        assert_eq!(enclave.state(), EnclaveState::Ready);
    }

    #[test]
    fn test_enclave_stats() {
        let enclave = SecureEnclave::new_simulated().unwrap();

        // Perform some operations
        let _ = enclave.attest(&[1, 2, 3]).unwrap();
        let key = ContentKey::generate().unwrap();
        let _ = enclave.seal(&key, SealPolicy::SignerIdentity).unwrap();

        let stats = enclave.stats();
        assert_eq!(stats.attestations_performed, 1);
        assert_eq!(stats.keys_sealed, 1);
    }
}
