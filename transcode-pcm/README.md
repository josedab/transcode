# transcode-pcm

PCM (Pulse Code Modulation) audio codec support for the transcode library.

## Features

- **Multiple Formats**: Signed/unsigned integer, floating point
- **Bit Depths**: 8, 16, 20, 24, 32-bit support
- **Endianness**: Little and big endian variants
- **Compressed PCM**: A-law and mu-law (G.711) support
- **Format Conversion**: Efficient conversion between PCM formats

## Supported Formats

| Format | Type | Bits | Endian |
|--------|------|------|--------|
| S8 | Signed | 8 | N/A |
| U8 | Unsigned | 8 | N/A |
| S16Le/Be | Signed | 16 | LE/BE |
| S24Le/Be | Signed | 24 | LE/BE |
| S32Le/Be | Signed | 32 | LE/BE |
| F32Le/Be | Float | 32 | LE/BE |
| F64Le/Be | Float | 64 | LE/BE |
| Alaw | Compressed | 8 | N/A |
| Mulaw | Compressed | 8 | N/A |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-pcm = { path = "../transcode-pcm" }
```

### Decoding

```rust
use transcode_pcm::{PcmDecoder, PcmFormat};

// Create a decoder for 16-bit signed little-endian PCM
let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);

// Decode raw PCM bytes to samples
let samples = decoder.decode(&raw_bytes)?;
```

### Encoding

```rust
use transcode_pcm::{PcmEncoder, PcmFormat};

// Create an encoder for 32-bit float PCM
let mut encoder = PcmEncoder::new(PcmFormat::F32Le, 48000, 2);

// Encode samples to raw PCM bytes
let bytes = encoder.encode(&samples)?;
```

### Format Information

```rust
use transcode_pcm::{codec_info, PcmFormat};

let info = codec_info(PcmFormat::S16Le);
println!("Codec: {}", info.name);           // "pcm_s16le"
println!("Bits: {}", info.bits_per_sample); // 16
println!("Signed: {}", info.is_signed);     // true
println!("Float: {}", info.is_float);       // false
```

## Common Use Cases

- **WAV files**: S16Le (CD quality), S24Le (high-resolution)
- **AIFF files**: S16Be, S24Be
- **Professional audio**: F32Le (DAW standard)
- **Telephony**: Alaw/Mulaw (G.711)

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
