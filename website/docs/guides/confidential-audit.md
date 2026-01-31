---
sidebar_position: 18
title: Confidential Computing Audit
description: Tamper-proof audit logging and key rotation for trusted execution environments
---

# Confidential Computing Audit

The `transcode-confidential` crate provides tamper-proof audit logging and key rotation management for transcoding in Trusted Execution Environments (TEEs).

## Overview

When processing sensitive media content (DRM-protected, medical, legal), you need:

- **Tamper-proof audit logs** — Hash-chained event records that detect modification
- **Key rotation** — Automatic rotation based on age or usage policies
- **Compliance tracking** — Record every key access, attestation, and policy violation

## Quick Start

```toml
[dependencies]
transcode-confidential = "1.0"
```

### Audit Logging

```rust
use transcode_confidential::audit::{
    AuditLog, AuditEventType, AuditSeverity,
};

let mut log = AuditLog::new();

// Record events
log.log(
    AuditEventType::EnclaveCreated,
    Some("enclave-001".to_string()),
    None,
    "SGX enclave initialized".to_string(),
    AuditSeverity::Info,
);

log.log(
    AuditEventType::KeyGenerated,
    Some("enclave-001".to_string()),
    Some("content-key-1".to_string()),
    "AES-256-GCM key generated for content encryption".to_string(),
    AuditSeverity::Info,
);

log.log(
    AuditEventType::ContentProcessed,
    Some("enclave-001".to_string()),
    Some("content-key-1".to_string()),
    "Processed 1080p frame batch (240 frames)".to_string(),
    AuditSeverity::Info,
);

println!("Events logged: {}", log.len());

// Verify integrity — detect any tampering
assert!(log.verify_integrity());
```

### Tamper Detection

The hash chain detects any modification to the log:

```rust
// Each event's hash depends on the previous event
// If any event is modified, verify_integrity() returns false
assert!(log.verify_integrity());

// Query specific event types
let key_events = log.events_by_type(&AuditEventType::KeyGenerated);
println!("Key generation events: {}", key_events.len());

// Find critical security events
let critical = log.critical_events();
for event in critical {
    println!("[CRITICAL] {}: {}", event.event_type, event.details);
}
```

### Key Rotation

Manage encryption keys with automatic rotation policies:

```rust
use transcode_confidential::audit::{
    KeyRotationManager, KeyRotationPolicy,
};

let policy = KeyRotationPolicy {
    max_age_secs: 3600,      // Rotate every hour
    max_uses: 10_000,         // Or after 10K uses
    require_attestation: true, // Re-attest on rotation
};

let mut manager = KeyRotationManager::new(policy);

// Generate initial key
manager.generate_key("content-key-1")?;

// Use the key
let key = manager.get_active_key("content-key-1").unwrap();
println!("Key version: {}", key.version);

// Rotate when policy requires it
if manager.needs_rotation("content-key-1", current_time_secs()) {
    let new_version = manager.rotate_key("content-key-1")?;
    println!("Rotated to version {}", new_version);
}

// All operations are audit-logged
let audit = manager.audit_log();
println!("Audit events: {}", audit.len());
assert!(audit.verify_integrity());
```

### Key Revocation

Immediately revoke compromised keys:

```rust
manager.revoke_key("content-key-1")?;

// All versions are deactivated
let versions = manager.key_versions("content-key-1");
for key in versions {
    assert!(!key.active);
}
```

## Event Types

| Event Type | Severity | Description |
|------------|----------|-------------|
| `EnclaveCreated` | Info | TEE enclave initialized |
| `AttestationVerified` | Info | Remote attestation passed |
| `AttestationFailed` | Critical | Attestation failure |
| `KeyGenerated` | Info | New encryption key created |
| `KeyRotated` | Info | Key rotated to new version |
| `KeyRevoked` | Warning | Key revoked (all versions) |
| `ContentProcessed` | Info | Media processed in enclave |
| `PolicyViolation` | Critical | Security policy violated |
| `UnauthorizedAccess` | Critical | Unauthorized access attempt |

## API Reference

| Type | Description |
|------|-------------|
| `AuditLog` | Hash-chained event log with integrity verification |
| `AuditEvent` | Single event with type, severity, and chain hash |
| `AuditEventType` | Categorized event types |
| `AuditSeverity` | Info, Warning, Critical |
| `KeyRotationManager` | Policy-based key lifecycle management |
| `KeyRotationPolicy` | Rotation triggers (age, usage, attestation) |
| `ManagedKey` | Versioned key with usage tracking |

## Next Steps

- [Security Policy](/docs/advanced/security) — Security practices and reporting
- [Error Handling](/docs/guides/error-handling) — Handling security errors
