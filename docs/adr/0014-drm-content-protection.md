# ADR-0014: DRM and Content Protection

## Status

Accepted

## Date

2024-06

## Context

Professional video workflows require content protection to satisfy licensing agreements with content owners. The streaming industry has standardized on:

1. **Common Encryption (CENC)**: ISO/IEC 23001-7 defines how to encrypt media samples
2. **Multi-DRM**: Content must work with multiple DRM systems simultaneously
   - Widevine (Google) - Android, Chrome, smart TVs
   - PlayReady (Microsoft) - Windows, Xbox, smart TVs
   - FairPlay (Apple) - iOS, macOS, Safari, Apple TV

The challenge is implementing encryption correctly while supporting all major DRM systems without requiring separate encoding passes.

## Decision

Implement comprehensive DRM support in `transcode-drm` with:

### 1. CENC Encryption Engine

Support all four CENC encryption schemes:

```rust
pub enum CencScheme {
    Cenc,  // AES-CTR, full sample (DASH default)
    Cbc1,  // AES-CBC, full sample (legacy)
    Cens,  // AES-CTR with pattern (subsample)
    Cbcs,  // AES-CBC with pattern (HLS/FairPlay)
}
```

### 2. Unified Key Management

```rust
pub struct KeyId([u8; 16]);      // 128-bit key identifier
pub struct ContentKey([u8; 16]); // 128-bit AES key
pub struct Iv([u8; 16]);         // Initialization vector

impl ContentKey {
    pub fn generate() -> Self;
    pub fn from_hex(s: &str) -> Result<Self>;
    pub fn to_hex(&self) -> String;
}
```

### 3. Multi-DRM Builder

```rust
pub struct MultiDrm {
    key_id: KeyId,
    widevine: Option<WidevineConfig>,
    playready: Option<PlayReadyConfig>,
    fairplay: Option<FairPlayConfig>,
}

impl MultiDrm {
    pub fn new(key_id: KeyId) -> Self;

    pub fn with_widevine<F>(self, f: F) -> Self
    where F: FnOnce(WidevineBuilder) -> WidevineBuilder;

    pub fn with_playready<F>(self, f: F) -> Self;
    pub fn with_fairplay<F>(self, f: F) -> Self;

    pub fn pssh_boxes(&self) -> Result<Vec<PsshBox>>;
}
```

### 4. Subsample Encryption

NAL unit-aware encryption for video:

```rust
pub struct SubsampleEncryptor {
    scheme: CencScheme,
    // Tracks NAL unit boundaries
    // Encrypts only slice data, not headers
}

impl SubsampleEncryptor {
    pub fn encrypt_sample(&mut self, data: &mut [u8]) -> Result<SampleEncryptionInfo>;
}
```

### 5. Clear Lead Support

Configurable unencrypted initial segments:

```rust
pub struct CencConfig {
    pub clear_lead_seconds: f64, // Unencrypted duration at start
    pub key_rotation: Option<KeyRotationConfig>,
}
```

## Consequences

### Positive

1. **Single encode, multi-DRM**: One encrypted file works with all DRM systems

2. **Spec compliance**: Full ISO/IEC 23001-7 implementation passes compliance tests

3. **Flexible key management**: Supports static keys, key rotation, and external KMS

4. **Subsample encryption**: Video headers remain readable for seeking/indexing

5. **Pattern encryption**: Efficient CBCS for constrained devices (mobile, STB)

### Negative

1. **Complexity**: DRM specs are intricate with many edge cases

2. **No license server**: Library encrypts content but doesn't handle license delivery

3. **Testing difficulty**: Full testing requires actual DRM license servers

4. **Legal considerations**: Some jurisdictions restrict encryption software

### Mitigations

1. **Extensive test vectors**: Validate against known-good encrypted content

2. **Clear documentation**: Document which configurations work with which players

3. **Example integrations**: Provide examples for common license server APIs

## Implementation Details

### PSSH Box Generation

Protection System Specific Header boxes for each DRM:

```rust
// Widevine PSSH
pub struct WidevinePssh {
    pub system_id: [u8; 16], // edef8ba9-79d6-4ace-a3c8-27dcd51d21ed
    pub key_ids: Vec<KeyId>,
    pub provider: Option<String>,
    pub content_id: Option<Vec<u8>>,
}

// PlayReady PSSH
pub struct PlayReadyPssh {
    pub system_id: [u8; 16], // 9a04f079-9840-4286-ab92-e65be0885f95
    pub pro: PlayReadyObject, // XML-based rights object
}

// FairPlay (uses different signaling)
pub struct FairPlayInfo {
    pub key_uri: String,     // skd://...
    pub iv: Iv,
}
```

### Encryption Patterns

CBCS pattern for HLS compatibility:

```rust
pub struct EncryptionPattern {
    pub crypt_byte_block: u8, // Encrypted blocks (typically 1)
    pub skip_byte_block: u8,  // Clear blocks (typically 9)
}

// 1:9 pattern = encrypt 1 block, skip 9 blocks
// Reduces CPU load on mobile devices
```

### Key Rotation

For live streaming with periodic key changes:

```rust
pub struct KeyRotationConfig {
    pub rotation_period: Duration,
    pub key_derivation: KeyDerivation,
}

pub enum KeyDerivation {
    Static(ContentKey),
    Derived { master_key: ContentKey, salt: [u8; 16] },
    External(Box<dyn KeyProvider>),
}
```

## Security Considerations

1. **Key handling**: ContentKey implements `Zeroize` to clear memory on drop

2. **No key logging**: Debug output redacts key material

3. **Constant-time operations**: AES operations use constant-time implementations

4. **Input validation**: Reject malformed key IDs and IVs early

## Alternatives Considered

### Alternative 1: Bento4 Integration

Use Bento4's mp4encrypt for DRM.

Rejected because:
- External binary dependency
- Less control over encryption process
- Can't integrate with streaming pipeline

### Alternative 2: DRM-Specific Libraries

Separate libraries for each DRM system.

Rejected because:
- Code duplication (CENC core is shared)
- Harder to ensure consistency
- Users need multiple dependencies

### Alternative 3: Shaka Packager Integration

Use Google's Shaka Packager.

Rejected because:
- Large C++ dependency
- Doesn't fit pure-Rust architecture
- Overkill for encryption-only needs

## References

- [ISO/IEC 23001-7 (CENC)](https://www.iso.org/standard/68042.html)
- [Widevine DRM](https://www.widevine.com/)
- [PlayReady Documentation](https://docs.microsoft.com/en-us/playready/)
- [FairPlay Streaming](https://developer.apple.com/streaming/fps/)
- [DASH-IF CENC Guidelines](https://dashif.org/guidelines/)
