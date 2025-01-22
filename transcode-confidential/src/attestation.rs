//! Remote attestation for TEE verification.

use crate::error::{ConfidentialError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Attestation report from a TEE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    /// TEE type.
    pub tee_type: TeeType,
    /// Measurement of enclave code.
    pub measurement: Vec<u8>,
    /// Report data (user-provided nonce).
    pub report_data: Vec<u8>,
    /// Platform certificate chain.
    pub cert_chain: Vec<Vec<u8>>,
    /// Signature over the report.
    pub signature: Vec<u8>,
    /// Timestamp of report generation.
    pub timestamp: i64,
    /// Additional platform info.
    pub platform_info: PlatformInfo,
}

impl AttestationReport {
    /// Verify the report signature.
    pub fn verify(&self) -> Result<bool> {
        // In production, would verify against vendor root certificates
        // For simulation, just check basic structure
        if self.measurement.is_empty() {
            return Err(ConfidentialError::AttestationFailed(
                "Empty measurement".to_string(),
            ));
        }
        if self.signature.is_empty() {
            return Err(ConfidentialError::AttestationFailed(
                "Missing signature".to_string(),
            ));
        }
        Ok(true) // Simulation: always pass
    }

    /// Get the enclave identity hash.
    pub fn enclave_identity(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(&self.measurement);
        hasher.update(self.platform_info.cpu_svn.to_le_bytes());
        hasher.finalize().to_vec()
    }
}

/// TEE type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeeType {
    /// Intel SGX.
    IntelSgx,
    /// AMD SEV.
    AmdSev,
    /// ARM TrustZone.
    ArmTrustZone,
    /// AWS Nitro Enclaves.
    AwsNitro,
    /// Simulated TEE (for development).
    Simulated,
}

impl std::fmt::Display for TeeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeeType::IntelSgx => write!(f, "Intel SGX"),
            TeeType::AmdSev => write!(f, "AMD SEV"),
            TeeType::ArmTrustZone => write!(f, "ARM TrustZone"),
            TeeType::AwsNitro => write!(f, "AWS Nitro"),
            TeeType::Simulated => write!(f, "Simulated"),
        }
    }
}

/// Platform security information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// CPU security version number.
    pub cpu_svn: u32,
    /// Enclave security version.
    pub isv_svn: u16,
    /// Product ID.
    pub isv_prod_id: u16,
    /// Enclave attributes.
    pub attributes: u64,
    /// Miscellaneous select.
    pub misc_select: u32,
}

/// Attestation verifier for validating TEE reports.
pub struct AttestationVerifier {
    /// Allowed enclave measurements.
    allowed_measurements: Vec<Vec<u8>>,
    /// Minimum CPU SVN.
    min_cpu_svn: u32,
    /// Minimum ISV SVN.
    min_isv_svn: u16,
    /// Allowed TEE types.
    allowed_tee_types: Vec<TeeType>,
}

impl Default for AttestationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl AttestationVerifier {
    /// Create a new verifier with permissive defaults.
    pub fn new() -> Self {
        Self {
            allowed_measurements: Vec::new(),
            min_cpu_svn: 0,
            min_isv_svn: 0,
            allowed_tee_types: vec![
                TeeType::IntelSgx,
                TeeType::AmdSev,
                TeeType::ArmTrustZone,
                TeeType::AwsNitro,
                TeeType::Simulated,
            ],
        }
    }

    /// Add an allowed enclave measurement.
    pub fn allow_measurement(mut self, measurement: Vec<u8>) -> Self {
        self.allowed_measurements.push(measurement);
        self
    }

    /// Set minimum CPU SVN.
    pub fn min_cpu_svn(mut self, svn: u32) -> Self {
        self.min_cpu_svn = svn;
        self
    }

    /// Set minimum ISV SVN.
    pub fn min_isv_svn(mut self, svn: u16) -> Self {
        self.min_isv_svn = svn;
        self
    }

    /// Allow only specific TEE types.
    pub fn allow_tee_types(mut self, types: Vec<TeeType>) -> Self {
        self.allowed_tee_types = types;
        self
    }

    /// Verify an attestation report.
    pub fn verify(&self, report: &AttestationReport) -> Result<VerificationResult> {
        let mut result = VerificationResult::default();

        // Check TEE type
        if !self.allowed_tee_types.contains(&report.tee_type) {
            return Err(ConfidentialError::AttestationFailed(format!(
                "TEE type {} not allowed",
                report.tee_type
            )));
        }
        result.tee_type_valid = true;

        // Verify signature
        result.signature_valid = report.verify()?;

        // Check measurement if allowlist is set
        if !self.allowed_measurements.is_empty() {
            result.measurement_valid = self
                .allowed_measurements
                .iter()
                .any(|m| m == &report.measurement);
            if !result.measurement_valid {
                return Err(ConfidentialError::AttestationFailed(
                    "Enclave measurement not in allowlist".to_string(),
                ));
            }
        } else {
            result.measurement_valid = true; // No allowlist = accept all
        }

        // Check SVN
        result.svn_valid = report.platform_info.cpu_svn >= self.min_cpu_svn
            && report.platform_info.isv_svn >= self.min_isv_svn;
        if !result.svn_valid {
            return Err(ConfidentialError::AttestationFailed(
                "Security version too old".to_string(),
            ));
        }

        result.overall_valid = result.tee_type_valid
            && result.signature_valid
            && result.measurement_valid
            && result.svn_valid;

        Ok(result)
    }
}

/// Result of attestation verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationResult {
    pub overall_valid: bool,
    pub tee_type_valid: bool,
    pub signature_valid: bool,
    pub measurement_valid: bool,
    pub svn_valid: bool,
}

/// Generate a nonce for attestation challenge.
pub fn generate_nonce() -> Vec<u8> {
    use ring::rand::{SecureRandom, SystemRandom};
    let rng = SystemRandom::new();
    let mut nonce = vec![0u8; 32];
    rng.fill(&mut nonce).expect("Failed to generate nonce");
    nonce
}

/// Generate a simulated attestation report for development.
pub fn generate_simulated_report(report_data: &[u8]) -> AttestationReport {
    let mut hasher = Sha256::new();
    hasher.update(b"simulated_enclave_v1");
    let measurement = hasher.finalize().to_vec();

    let mut sig_hasher = Sha256::new();
    sig_hasher.update(&measurement);
    sig_hasher.update(report_data);
    let signature = sig_hasher.finalize().to_vec();

    AttestationReport {
        tee_type: TeeType::Simulated,
        measurement,
        report_data: report_data.to_vec(),
        cert_chain: vec![vec![0u8; 32]], // Placeholder
        signature,
        timestamp: chrono::Utc::now().timestamp(),
        platform_info: PlatformInfo {
            cpu_svn: 1,
            isv_svn: 1,
            isv_prod_id: 1,
            attributes: 0,
            misc_select: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_nonce() {
        let nonce1 = generate_nonce();
        let nonce2 = generate_nonce();
        assert_eq!(nonce1.len(), 32);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_simulated_report() {
        let nonce = generate_nonce();
        let report = generate_simulated_report(&nonce);

        assert_eq!(report.tee_type, TeeType::Simulated);
        assert!(!report.measurement.is_empty());
        assert!(!report.signature.is_empty());
    }

    #[test]
    fn test_verifier() {
        let nonce = generate_nonce();
        let report = generate_simulated_report(&nonce);

        let verifier = AttestationVerifier::new();
        let result = verifier.verify(&report).unwrap();

        assert!(result.overall_valid);
    }

    #[test]
    fn test_verifier_tee_type_restriction() {
        let nonce = generate_nonce();
        let report = generate_simulated_report(&nonce);

        let verifier =
            AttestationVerifier::new().allow_tee_types(vec![TeeType::IntelSgx]);

        let result = verifier.verify(&report);
        assert!(result.is_err());
    }
}
