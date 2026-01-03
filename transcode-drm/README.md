# transcode-drm

DRM and encryption support for the Transcode library. Provides content protection with industry-standard systems including Widevine, PlayReady, and FairPlay Streaming.

## Features

- **Common Encryption (CENC)**: ISO/IEC 23001-7 with all encryption schemes (cenc, cbc1, cens, cbcs)
- **AES Encryption**: AES-128-CTR and AES-128-CBC modes
- **Pattern Encryption**: CBCS pattern encryption for HLS/FairPlay
- **Multi-DRM**: Widevine, PlayReady, and FairPlay Streaming support
- **PSSH Generation**: Protection System Specific Header boxes for all DRM systems
- **Subsample Encryption**: NAL unit-aware encryption for video streams
- **Key Management**: Key stores with derivation support

## Encryption Schemes

| Scheme | Mode | Pattern | Use Case |
|--------|------|---------|----------|
| `cenc` | CTR | No | DASH (Widevine, PlayReady) |
| `cbc1` | CBC | No | Legacy systems |
| `cens` | CTR | Yes | Pattern-based CTR |
| `cbcs` | CBC | Yes | HLS (FairPlay) |

## Key Types

- `KeyId` - UUID-based key identifier (16 bytes)
- `ContentKey` - AES-128 content encryption key (16 bytes)
- `Iv` - Initialization vector with counter support (16 bytes)
- `KeyPair` - Combined key ID and content key
- `KeyStore` - Key storage with optional master key derivation

## Usage

### Basic CENC Encryption (DASH)

```rust
use transcode_drm::prelude::*;

let key_id = KeyId::generate();
let key = ContentKey::generate();
let iv = Iv::generate();

let config = CencConfig::for_dash(key_id, key, iv);
let mut encryptor = CencEncryptor::new(config)?;

let mut sample = vec![0u8; 1024];
let info = encryptor.encrypt_sample(&mut sample)?;
```

### Multi-DRM PSSH Generation

```rust
use transcode_drm::prelude::*;

let key_id = KeyId::generate();

let multi_drm = MultiDrm::new(key_id)
    .with_widevine(|w| w.with_provider("MyService"))
    .with_playready(|p| p.with_la_url("https://license.example.com/"))
    .with_fairplay(|f| f.with_key_server_url("https://fps.example.com/"));

let psshs = multi_drm.pssh_boxes()?;
```

### HLS with FairPlay

```rust
use transcode_drm::prelude::*;

let config = FairPlayConfig::new(
    KeyId::generate(),
    ContentKey::generate(),
    Iv::generate(),
    "https://fps.example.com/",
);

let hls_info = HlsDrmInfo::from_config(&config);
println!("EXT-X-KEY: {}", hls_info.ext_x_key);
```

### Key Store with Derivation

```rust
use transcode_drm::prelude::*;

let master_key = ContentKey::generate();
let mut store = KeyStore::with_master_key(master_key);

// Keys are derived automatically from master key
let key_id = KeyId::generate();
let derived_key = store.get_or_derive(&key_id)?;
```

## Feature Flags

- `widevine` - Widevine DRM support
- `fairplay` - FairPlay Streaming support
- `playready` - PlayReady DRM support

## License

See the workspace root for license information.
