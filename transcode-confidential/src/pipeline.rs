//! End-to-end encrypted transcoding pipeline within TEE.
//!
//! Provides [`ConfidentialPipeline`] which manages the full workflow:
//! attestation → key provisioning → decrypt → transcode → re-encrypt.

use crate::attestation::{generate_nonce, AttestationReport, AttestationVerifier, TeeType};
use crate::enclave::{EnclaveBuilder, SecureEnclave};
use crate::error::{ConfidentialError, Result};
use crate::keys::{ContentKey, SealPolicy};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for the confidential pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub tee_type: TeeType,
    pub seal_policy: SealPolicy,
    /// Whether to verify attestation (disable only for testing).
    pub verify_attestation: bool,
    /// Maximum frame size in bytes before rejecting (DoS protection).
    pub max_frame_size: usize,
    /// Key rotation interval (number of frames, 0 = no rotation).
    pub key_rotation_interval: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tee_type: TeeType::Simulated,
            seal_policy: SealPolicy::SignerIdentity,
            verify_attestation: true,
            max_frame_size: 64 * 1024 * 1024, // 64MB max frame
            key_rotation_interval: 0,
        }
    }
}

/// Statistics from the confidential pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    pub frames_processed: u64,
    pub bytes_decrypted: u64,
    pub bytes_encrypted: u64,
    pub key_rotations: u64,
    pub attestation_verified: bool,
}

/// State of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Uninitialized,
    Attested,
    KeyProvisioned,
    Processing,
    Completed,
    Failed,
}

/// An end-to-end encrypted transcoding pipeline running within a TEE.
pub struct ConfidentialPipeline {
    config: PipelineConfig,
    enclave: Option<Arc<SecureEnclave>>,
    input_key: Option<ContentKey>,
    output_key: Option<ContentKey>,
    state: PipelineState,
    stats: PipelineStats,
    attestation_report: Option<AttestationReport>,
}

impl ConfidentialPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            enclave: None,
            input_key: None,
            output_key: None,
            state: PipelineState::Uninitialized,
            stats: PipelineStats::default(),
            attestation_report: None,
        }
    }

    /// Initialize the enclave and perform attestation.
    pub fn initialize(&mut self) -> Result<&AttestationReport> {
        let enclave = EnclaveBuilder::new()
            .tee_type(self.config.tee_type)
            .build()?;

        let nonce = generate_nonce();
        let report = enclave.attest(&nonce)?;

        if self.config.verify_attestation {
            let verifier = AttestationVerifier::new();
            let result = verifier.verify(&report)?;
            if !result.overall_valid {
                return Err(ConfidentialError::AttestationFailed(
                    "Attestation verification failed".into(),
                ));
            }
            self.stats.attestation_verified = true;
        }

        self.attestation_report = Some(report);
        self.enclave = Some(enclave);
        self.state = PipelineState::Attested;

        Ok(self.attestation_report.as_ref().unwrap())
    }

    /// Provision keys for decryption (input) and re-encryption (output).
    pub fn provision_keys(
        &mut self,
        input_key: ContentKey,
        output_key: ContentKey,
    ) -> Result<()> {
        if self.state != PipelineState::Attested {
            return Err(ConfidentialError::PolicyViolation(
                "Must be attested before key provisioning".into(),
            ));
        }

        // Seal keys for persistence
        let enclave = self.enclave.as_ref().ok_or(ConfidentialError::PolicyViolation(
            "Enclave not initialized".into(),
        ))?;

        let _sealed_input = enclave.seal(&input_key, self.config.seal_policy)?;
        let _sealed_output = enclave.seal(&output_key, self.config.seal_policy)?;

        self.input_key = Some(input_key);
        self.output_key = Some(output_key);
        self.state = PipelineState::KeyProvisioned;
        Ok(())
    }

    /// Process an encrypted frame: decrypt → (transcode placeholder) → re-encrypt.
    pub fn process_frame(
        &mut self,
        encrypted_input: &[u8],
        input_nonce: &[u8; 12],
        output_nonce: &[u8; 12],
    ) -> Result<Vec<u8>> {
        if self.state != PipelineState::KeyProvisioned && self.state != PipelineState::Processing {
            return Err(ConfidentialError::PolicyViolation(
                "Keys must be provisioned before processing".into(),
            ));
        }

        if encrypted_input.len() > self.config.max_frame_size {
            return Err(ConfidentialError::CryptoError(format!(
                "Frame size {} exceeds maximum {}",
                encrypted_input.len(),
                self.config.max_frame_size
            )));
        }

        let input_key = self.input_key.as_ref().ok_or(ConfidentialError::KeyError(
            "Input key not available".into(),
        ))?;
        let output_key = self.output_key.as_ref().ok_or(ConfidentialError::KeyError(
            "Output key not available".into(),
        ))?;

        let enclave = self.enclave.as_ref().ok_or(ConfidentialError::PolicyViolation(
            "Enclave not initialized".into(),
        ))?;

        // Decrypt inside enclave
        let plaintext = enclave.process_frame(encrypted_input, input_key, input_nonce)?;
        self.stats.bytes_decrypted += plaintext.len() as u64;

        // Re-encrypt with output key
        let re_encrypted = output_key.encrypt(&plaintext, output_nonce)?;
        self.stats.bytes_encrypted += re_encrypted.len() as u64;
        self.stats.frames_processed += 1;

        // Key rotation check
        if self.config.key_rotation_interval > 0
            && self.stats.frames_processed % self.config.key_rotation_interval == 0
        {
            self.rotate_output_key()?;
        }

        self.state = PipelineState::Processing;
        Ok(re_encrypted)
    }

    /// Finalize the pipeline.
    pub fn finalize(&mut self) -> Result<PipelineStats> {
        self.state = PipelineState::Completed;
        // Zeroize keys in memory
        self.input_key = None;
        self.output_key = None;
        Ok(self.stats.clone())
    }

    pub fn state(&self) -> PipelineState {
        self.state
    }

    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    pub fn attestation_report(&self) -> Option<&AttestationReport> {
        self.attestation_report.as_ref()
    }

    fn rotate_output_key(&mut self) -> Result<()> {
        let new_key = ContentKey::generate()?;
        if let Some(ref enclave) = self.enclave {
            let _sealed = enclave.seal(&new_key, self.config.seal_policy)?;
        }
        self.output_key = Some(new_key);
        self.stats.key_rotations += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_lifecycle() {
        let mut pipeline = ConfidentialPipeline::new(PipelineConfig::default());
        assert_eq!(pipeline.state(), PipelineState::Uninitialized);

        // Initialize
        pipeline.initialize().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Attested);
        assert!(pipeline.stats().attestation_verified);

        // Provision keys
        let input_key = ContentKey::generate().unwrap();
        let output_key = ContentKey::generate().unwrap();
        let ik = input_key.clone();
        pipeline.provision_keys(input_key, output_key).unwrap();
        assert_eq!(pipeline.state(), PipelineState::KeyProvisioned);

        // Process a frame
        let plaintext = b"test video frame data here";
        let nonce_in = [1u8; 12];
        let nonce_out = [2u8; 12];
        let encrypted = ik.encrypt(plaintext, &nonce_in).unwrap();

        let result = pipeline.process_frame(&encrypted, &nonce_in, &nonce_out).unwrap();
        assert!(!result.is_empty());
        assert_eq!(pipeline.stats().frames_processed, 1);

        // Finalize
        let stats = pipeline.finalize().unwrap();
        assert_eq!(stats.frames_processed, 1);
        assert_eq!(pipeline.state(), PipelineState::Completed);
    }

    #[test]
    fn test_pipeline_state_enforcement() {
        let mut pipeline = ConfidentialPipeline::new(PipelineConfig::default());

        // Can't provision keys without attestation
        let k1 = ContentKey::generate().unwrap();
        let k2 = ContentKey::generate().unwrap();
        assert!(pipeline.provision_keys(k1, k2).is_err());

        // Can't process without keys
        assert!(pipeline.process_frame(&[0u8; 32], &[0; 12], &[0; 12]).is_err());
    }

    #[test]
    fn test_max_frame_size_enforcement() {
        let mut pipeline = ConfidentialPipeline::new(PipelineConfig {
            max_frame_size: 100,
            ..Default::default()
        });

        pipeline.initialize().unwrap();
        let ik = ContentKey::generate().unwrap();
        let ok = ContentKey::generate().unwrap();
        pipeline.provision_keys(ik, ok).unwrap();

        let big_frame = vec![0u8; 200];
        assert!(pipeline.process_frame(&big_frame, &[0; 12], &[0; 12]).is_err());
    }

    #[test]
    fn test_key_rotation() {
        let config = PipelineConfig {
            key_rotation_interval: 2,
            ..Default::default()
        };
        let mut pipeline = ConfidentialPipeline::new(config);
        pipeline.initialize().unwrap();

        let input_key = ContentKey::generate().unwrap();
        let ik = input_key.clone();
        let output_key = ContentKey::generate().unwrap();
        pipeline.provision_keys(input_key, output_key).unwrap();

        let nonce = [0u8; 12];
        let data = ik.encrypt(b"frame", &nonce).unwrap();

        // Process 4 frames — should trigger 2 rotations (at frame 2 and 4)
        for _ in 0..4 {
            pipeline.process_frame(&data, &nonce, &nonce).unwrap();
        }
        assert_eq!(pipeline.stats().key_rotations, 2);
    }
}
