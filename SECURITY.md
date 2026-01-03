# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### Disclosure Policy

- We follow coordinated disclosure practices
- Credit will be given to reporters (unless anonymity is requested)
- Public disclosure occurs after a fix is available

## Security Considerations

### Input Validation

Transcode processes untrusted media files. Key security measures:

- **Bitstream Parsing**: All bitstream readers validate data bounds
- **Buffer Sizes**: Strict limits on allocation sizes
- **Integer Overflow**: Checked arithmetic for size calculations
- **Fuzzing**: Critical parsers are fuzz-tested

### Memory Safety

As a Rust library, Transcode benefits from:

- No buffer overflows in safe code
- No use-after-free bugs
- No data races
- Automatic memory management

### Unsafe Code

Limited `unsafe` blocks exist for:
- SIMD intrinsics (audited)
- FFI bindings (validated at boundaries)
- Performance-critical paths (minimized)

All unsafe code is:
- Documented with safety invariants
- Reviewed carefully
- Tested with Miri where applicable

### Dependencies

- Dependencies are reviewed before inclusion
- `cargo audit` is run regularly
- Minimal dependency policy

## Best Practices for Users

1. **Keep Updated**: Use the latest version
2. **Resource Limits**: Set appropriate limits for:
   - Maximum file size
   - Maximum resolution
   - Processing timeout
3. **Sandboxing**: Consider running in a sandboxed environment
4. **Input Validation**: Validate files before processing

## Known Limitations

- Not designed for adversarial input in all code paths
- DRM features should not be relied upon for high-security applications
- Hardware acceleration inherits platform security properties

## Security Audits

This project has not undergone a formal security audit. Contributions toward professional auditing are welcome.

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities.
