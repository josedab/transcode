//! Audit logging and key rotation for confidential transcoding.
//!
//! Provides tamper-proof event logging with hash-chain integrity
//! and key rotation management with configurable policies.

#![allow(dead_code)]

use crate::error::{ConfidentialError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Audit event types and severity
// ---------------------------------------------------------------------------

/// Types of auditable events in the confidential transcoding system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    EnclaveCreated,
    EnclaveDestroyed,
    AttestationRequested,
    AttestationVerified,
    AttestationFailed,
    KeyGenerated,
    KeyRotated,
    KeyRevoked,
    KeyAccessed,
    ContentProcessed,
    ContentDecrypted,
    ContentEncrypted,
    PolicyViolation,
    UnauthorizedAccess,
    SecureChannelEstablished,
    SecureChannelClosed,
}

/// Severity level for an audit event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditSeverity {
    Info,
    Warning,
    Critical,
}

// ---------------------------------------------------------------------------
// Audit event & log
// ---------------------------------------------------------------------------

/// A single audit event with chain-hash for tamper detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: String,
    pub timestamp: u64,
    pub event_type: AuditEventType,
    pub enclave_id: Option<String>,
    pub key_id: Option<String>,
    pub details: String,
    pub severity: AuditSeverity,
    pub hash: String,
}

/// Tamper-proof audit log backed by a hash chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    events: Vec<AuditEvent>,
    event_counter: u64,
    chain_hash: String,
}

const INITIAL_CHAIN_HASH: &str = "0000000000000000";

/// Compute a simple FNV-1a-style 64-bit hash, returned as hex.
fn compute_hash(data: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

impl AuditLog {
    /// Create a new, empty audit log.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            event_counter: 0,
            chain_hash: INITIAL_CHAIN_HASH.to_string(),
        }
    }

    /// Record an event, extending the hash chain.
    pub fn log(
        &mut self,
        event_type: AuditEventType,
        enclave_id: Option<String>,
        key_id: Option<String>,
        details: impl Into<String>,
        severity: AuditSeverity,
    ) {
        self.event_counter += 1;
        let details = details.into();
        let timestamp = self.event_counter; // monotonic stand-in for epoch millis

        let id = format!("evt-{}-{}", timestamp, self.event_counter);

        let hash_input = format!(
            "{}|{}|{:?}|{:?}|{:?}|{}|{:?}",
            self.chain_hash, timestamp, event_type, enclave_id, key_id, details, severity
        );
        let hash = compute_hash(&hash_input);
        self.chain_hash = hash.clone();

        self.events.push(AuditEvent {
            id,
            timestamp,
            event_type,
            enclave_id,
            key_id,
            details,
            severity,
            hash,
        });
    }

    /// Verify the integrity of the entire hash chain.
    pub fn verify_integrity(&self) -> bool {
        let mut running_hash = INITIAL_CHAIN_HASH.to_string();
        for event in &self.events {
            let hash_input = format!(
                "{}|{}|{:?}|{:?}|{:?}|{}|{:?}",
                running_hash,
                event.timestamp,
                event.event_type,
                event.enclave_id,
                event.key_id,
                event.details,
                event.severity
            );
            let expected = compute_hash(&hash_input);
            if event.hash != expected {
                return false;
            }
            running_hash = expected;
        }
        true
    }

    /// Return all events matching the given type.
    pub fn events_by_type(&self, event_type: &AuditEventType) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Return all events with a timestamp >= the given value.
    pub fn events_since(&self, timestamp: u64) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= timestamp)
            .collect()
    }

    /// Return all events with Critical severity.
    pub fn critical_events(&self) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.severity == AuditSeverity::Critical)
            .collect()
    }

    /// Number of events recorded.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Key rotation policy & managed keys
// ---------------------------------------------------------------------------

/// Policy controlling when key rotation is required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationPolicy {
    /// Maximum key age in seconds before rotation is required.
    pub max_age_secs: u64,
    /// Maximum number of uses before rotation is required.
    pub max_uses: u64,
    /// Whether rotation requires re-attestation.
    pub require_attestation: bool,
}

/// A single version of a managed key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedKey {
    pub key_id: String,
    pub version: u32,
    pub created_at: u64,
    pub use_count: u64,
    pub active: bool,
    pub key_data: Vec<u8>,
}

/// Manages key lifecycle including generation, rotation, and revocation.
pub struct KeyRotationManager {
    keys: HashMap<String, Vec<ManagedKey>>,
    policy: KeyRotationPolicy,
    audit_log: AuditLog,
}

impl KeyRotationManager {
    /// Create a new manager with the given rotation policy.
    pub fn new(policy: KeyRotationPolicy) -> Self {
        Self {
            keys: HashMap::new(),
            policy,
            audit_log: AuditLog::new(),
        }
    }

    /// Generate a brand-new key with version 1.
    pub fn generate_key(&mut self, key_id: &str) -> Result<()> {
        if self.keys.contains_key(key_id) {
            return Err(ConfidentialError::KeyError(format!(
                "Key '{}' already exists",
                key_id
            )));
        }

        let key = ManagedKey {
            key_id: key_id.to_string(),
            version: 1,
            created_at: self.current_time(),
            use_count: 0,
            active: true,
            key_data: Self::generate_key_data(key_id, 1),
        };

        self.keys.insert(key_id.to_string(), vec![key]);
        self.audit_log.log(
            AuditEventType::KeyGenerated,
            None,
            Some(key_id.to_string()),
            format!("Generated key '{}' version 1", key_id),
            AuditSeverity::Info,
        );
        Ok(())
    }

    /// Rotate a key: deactivate the current active version and create a new one.
    /// Returns the new version number.
    pub fn rotate_key(&mut self, key_id: &str) -> Result<u32> {
        let created_at = self.current_time();
        let versions = self.keys.get_mut(key_id).ok_or_else(|| {
            ConfidentialError::KeyError(format!("Key '{}' not found", key_id))
        })?;

        let current_version = versions.iter().map(|k| k.version).max().unwrap_or(0);
        let new_version = current_version + 1;

        // Deactivate all existing versions.
        for v in versions.iter_mut() {
            v.active = false;
        }

        versions.push(ManagedKey {
            key_id: key_id.to_string(),
            version: new_version,
            created_at,
            use_count: 0,
            active: true,
            key_data: Self::generate_key_data(key_id, new_version),
        });

        self.audit_log.log(
            AuditEventType::KeyRotated,
            None,
            Some(key_id.to_string()),
            format!(
                "Rotated key '{}' from version {} to {}",
                key_id, current_version, new_version
            ),
            AuditSeverity::Info,
        );
        Ok(new_version)
    }

    /// Check whether a key needs rotation based on the configured policy.
    pub fn needs_rotation(&self, key_id: &str, current_time: u64) -> bool {
        let active = match self.get_active_key(key_id) {
            Some(k) => k,
            None => return false,
        };

        let age = current_time.saturating_sub(active.created_at);
        age >= self.policy.max_age_secs || active.use_count >= self.policy.max_uses
    }

    /// Return the latest active version for a key.
    pub fn get_active_key(&self, key_id: &str) -> Option<&ManagedKey> {
        self.keys
            .get(key_id)?
            .iter()
            .rev()
            .find(|k| k.active)
    }

    /// Revoke all versions of a key.
    pub fn revoke_key(&mut self, key_id: &str) -> Result<()> {
        let versions = self.keys.get_mut(key_id).ok_or_else(|| {
            ConfidentialError::KeyError(format!("Key '{}' not found", key_id))
        })?;

        for v in versions.iter_mut() {
            v.active = false;
        }

        self.audit_log.log(
            AuditEventType::KeyRevoked,
            None,
            Some(key_id.to_string()),
            format!("Revoked all versions of key '{}'", key_id),
            AuditSeverity::Warning,
        );
        Ok(())
    }

    /// Return all versions for a key.
    pub fn key_versions(&self, key_id: &str) -> Vec<&ManagedKey> {
        self.keys
            .get(key_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Access the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        &self.audit_log
    }

    // -- internal helpers ---------------------------------------------------

    fn current_time(&self) -> u64 {
        self.audit_log.event_counter + 1
    }

    fn generate_key_data(key_id: &str, version: u32) -> Vec<u8> {
        // Deterministic simulated key material for testing.
        let seed = format!("key-material-{}-v{}", key_id, version);
        compute_hash(&seed).into_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AuditLog tests -----------------------------------------------------

    #[test]
    fn test_audit_log_new_is_empty() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.verify_integrity());
    }

    #[test]
    fn test_audit_log_records_events() {
        let mut log = AuditLog::new();
        log.log(
            AuditEventType::EnclaveCreated,
            Some("enc-1".into()),
            None,
            "Enclave created",
            AuditSeverity::Info,
        );
        log.log(
            AuditEventType::KeyGenerated,
            None,
            Some("key-1".into()),
            "Key generated",
            AuditSeverity::Info,
        );
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_audit_log_integrity_valid() {
        let mut log = AuditLog::new();
        for i in 0..5 {
            log.log(
                AuditEventType::ContentProcessed,
                None,
                None,
                format!("Frame {}", i),
                AuditSeverity::Info,
            );
        }
        assert!(log.verify_integrity());
    }

    #[test]
    fn test_audit_log_tamper_detection() {
        let mut log = AuditLog::new();
        log.log(
            AuditEventType::EnclaveCreated,
            Some("enc-1".into()),
            None,
            "Created",
            AuditSeverity::Info,
        );
        log.log(
            AuditEventType::KeyGenerated,
            None,
            Some("k-1".into()),
            "Generated",
            AuditSeverity::Info,
        );
        assert!(log.verify_integrity());

        // Tamper with the first event's details.
        log.events[0].details = "TAMPERED".to_string();
        assert!(!log.verify_integrity());
    }

    #[test]
    fn test_audit_log_tamper_detection_hash() {
        let mut log = AuditLog::new();
        log.log(
            AuditEventType::EnclaveCreated,
            None,
            None,
            "ok",
            AuditSeverity::Info,
        );
        assert!(log.verify_integrity());

        // Tamper with the stored hash directly.
        log.events[0].hash = "ffffffffffffffff".to_string();
        assert!(!log.verify_integrity());
    }

    #[test]
    fn test_events_by_type() {
        let mut log = AuditLog::new();
        log.log(AuditEventType::KeyGenerated, None, Some("k1".into()), "g1", AuditSeverity::Info);
        log.log(AuditEventType::ContentProcessed, None, None, "p1", AuditSeverity::Info);
        log.log(AuditEventType::KeyGenerated, None, Some("k2".into()), "g2", AuditSeverity::Info);

        let key_events = log.events_by_type(&AuditEventType::KeyGenerated);
        assert_eq!(key_events.len(), 2);
    }

    #[test]
    fn test_events_since() {
        let mut log = AuditLog::new();
        for _ in 0..5 {
            log.log(AuditEventType::ContentProcessed, None, None, "frame", AuditSeverity::Info);
        }
        let recent = log.events_since(4);
        assert_eq!(recent.len(), 2); // events at timestamps 4 and 5
    }

    #[test]
    fn test_critical_events() {
        let mut log = AuditLog::new();
        log.log(AuditEventType::EnclaveCreated, None, None, "ok", AuditSeverity::Info);
        log.log(
            AuditEventType::PolicyViolation,
            None,
            None,
            "violation!",
            AuditSeverity::Critical,
        );
        log.log(
            AuditEventType::UnauthorizedAccess,
            None,
            None,
            "intruder!",
            AuditSeverity::Critical,
        );
        assert_eq!(log.critical_events().len(), 2);
    }

    // -- KeyRotationManager tests -------------------------------------------

    fn default_policy() -> KeyRotationPolicy {
        KeyRotationPolicy {
            max_age_secs: 3600,
            max_uses: 1000,
            require_attestation: false,
        }
    }

    #[test]
    fn test_generate_key() {
        let mut mgr = KeyRotationManager::new(default_policy());
        mgr.generate_key("master").unwrap();

        let active = mgr.get_active_key("master").unwrap();
        assert_eq!(active.version, 1);
        assert!(active.active);
        assert_eq!(mgr.audit_log().len(), 1);
    }

    #[test]
    fn test_generate_key_duplicate_fails() {
        let mut mgr = KeyRotationManager::new(default_policy());
        mgr.generate_key("master").unwrap();
        assert!(mgr.generate_key("master").is_err());
    }

    #[test]
    fn test_rotate_key() {
        let mut mgr = KeyRotationManager::new(default_policy());
        mgr.generate_key("master").unwrap();

        let new_ver = mgr.rotate_key("master").unwrap();
        assert_eq!(new_ver, 2);

        let active = mgr.get_active_key("master").unwrap();
        assert_eq!(active.version, 2);

        let versions = mgr.key_versions("master");
        assert_eq!(versions.len(), 2);
        assert!(!versions[0].active); // v1 deactivated
        assert!(versions[1].active); // v2 active
    }

    #[test]
    fn test_rotate_nonexistent_key_fails() {
        let mut mgr = KeyRotationManager::new(default_policy());
        assert!(mgr.rotate_key("nope").is_err());
    }

    #[test]
    fn test_needs_rotation_by_age() {
        let policy = KeyRotationPolicy {
            max_age_secs: 100,
            max_uses: u64::MAX,
            require_attestation: false,
        };
        let mut mgr = KeyRotationManager::new(policy);
        mgr.generate_key("k1").unwrap();

        // Key just created – created_at is 1, so age at t=50 is 49 < 100.
        assert!(!mgr.needs_rotation("k1", 50));
        // At t=200 age is 199 >= 100.
        assert!(mgr.needs_rotation("k1", 200));
    }

    #[test]
    fn test_needs_rotation_by_uses() {
        let policy = KeyRotationPolicy {
            max_age_secs: u64::MAX,
            max_uses: 5,
            require_attestation: false,
        };
        let mut mgr = KeyRotationManager::new(policy);
        mgr.generate_key("k1").unwrap();

        assert!(!mgr.needs_rotation("k1", 0));

        // Simulate usage by mutating use_count directly.
        if let Some(versions) = mgr.keys.get_mut("k1") {
            versions.last_mut().unwrap().use_count = 5;
        }
        assert!(mgr.needs_rotation("k1", 0));
    }

    #[test]
    fn test_revoke_key() {
        let mut mgr = KeyRotationManager::new(default_policy());
        mgr.generate_key("k1").unwrap();
        mgr.rotate_key("k1").unwrap();

        mgr.revoke_key("k1").unwrap();

        assert!(mgr.get_active_key("k1").is_none());
        let versions = mgr.key_versions("k1");
        assert!(versions.iter().all(|v| !v.active));

        // Audit should contain a KeyRevoked event.
        let revoked = mgr
            .audit_log()
            .events_by_type(&AuditEventType::KeyRevoked);
        assert_eq!(revoked.len(), 1);
    }

    #[test]
    fn test_revoke_nonexistent_key_fails() {
        let mut mgr = KeyRotationManager::new(default_policy());
        assert!(mgr.revoke_key("nope").is_err());
    }

    #[test]
    fn test_key_rotation_manager_audit_integrity() {
        let mut mgr = KeyRotationManager::new(default_policy());
        mgr.generate_key("a").unwrap();
        mgr.generate_key("b").unwrap();
        mgr.rotate_key("a").unwrap();
        mgr.revoke_key("b").unwrap();

        assert!(mgr.audit_log().verify_integrity());
    }
}
