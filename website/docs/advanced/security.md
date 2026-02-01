---
sidebar_position: 4
title: Security Policy
description: Security practices, vulnerability reporting, and safety guarantees
---

# Security Policy

Transcode processes untrusted media files, making security a first-class concern. This page covers our security practices, how to report vulnerabilities, and what safety guarantees the library provides.

## Memory Safety Guarantees

As a pure Rust library, Transcode eliminates entire classes of vulnerabilities:

| Vulnerability Class | C/C++ Codecs | Transcode |
|---------------------|-------------|-----------|
| Buffer overflow | Common | **Impossible** in safe code |
| Use-after-free | Common | **Impossible** in safe code |
| Data races | Common | **Impossible** (enforced by compiler) |
| Null pointer dereference | Common | **Impossible** (`Option` type) |
| Integer overflow | Silent | **Checked** (panics or wraps explicitly) |

## Input Validation

All bitstream parsers validate data bounds before access:

```rust
use transcode_core::bitstream::BitReader;

// BitReader checks bounds on every read
let mut reader = BitReader::new(data);
let value = reader.read_bits(8)?; // Returns Err if insufficient data
```

Key protections:
- **Bitstream parsing** — All readers validate data bounds
- **Buffer sizes** — Strict limits on allocation sizes
- **Integer overflow** — Checked arithmetic for size calculations
- **Fuzzing** — Critical parsers are fuzz-tested

## Unsafe Code

Limited `unsafe` blocks exist for performance-critical paths:

- **SIMD intrinsics** — AVX2, NEON operations (audited, tested with Miri)
- **FFI bindings** — Validated at boundaries with safe wrappers
- **Performance-critical paths** — Minimized and documented

All unsafe code follows these rules:
1. Documented with safety invariants
2. Wrapped in safe abstractions
3. Tested with Miri where applicable
4. Reviewed in every PR

## Reporting Vulnerabilities

:::danger Important
**Do NOT** open a public GitHub issue for security vulnerabilities.
:::

### How to Report

1. Email security concerns to the maintainers directly
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

| Severity | Acknowledgment | Resolution |
|----------|---------------|------------|
| Critical | 48 hours | 7 days |
| High | 48 hours | 14 days |
| Medium | 48 hours | 30 days |
| Low | 48 hours | 90 days |

### Disclosure Policy

- We follow coordinated disclosure practices
- Credit is given to reporters (unless anonymity is requested)
- Public disclosure occurs after a fix is available

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x | ✅ Active support |

## Dependencies

- Dependencies are reviewed before inclusion
- `cargo audit` is run in CI
- Minimal dependency policy — only what's needed
- `cargo deny` configured for license and advisory checks

## Best Practices for Users

1. **Keep Updated** — Use the latest version for security patches
2. **Set Resource Limits** — Configure maximum file size, resolution, and processing timeout
3. **Sandbox Processing** — Run in containers or sandboxed environments for untrusted input
4. **Validate Input** — Check file headers before processing

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("untrusted_file.mp4")
    .output("output.mp4")
    .max_file_size(500 * 1024 * 1024)  // 500MB limit
    .timeout(std::time::Duration::from_secs(300));  // 5 min timeout
```

## Known Limitations

- DRM features should not be relied upon for high-security applications
- Hardware acceleration inherits platform security properties
- Not formally audited (contributions toward professional auditing are welcome)

## Confidential Computing

For processing sensitive content in Trusted Execution Environments, see the [Confidential Computing guide](/docs/guides/confidential-audit) which covers:

- Tamper-proof audit logging with hash chains
- Key rotation with configurable policies
- Remote attestation verification
