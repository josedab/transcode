# transcode-watermark

Forensic watermarking library for invisible video watermarks, designed for piracy tracking and content protection.

## Features

- **Invisible Watermarking**: Embed imperceptible watermarks in video frames
- **DCT Domain Embedding**: Frequency-domain watermarking for robustness against compression
- **Spatial Domain Embedding**: Direct pixel modification for faster processing
- **Spread Spectrum**: Chip-based spreading for improved security
- **Error Correction**: Multiple redundancy levels (None, Low, Medium, High)
- **Multi-Frame Extraction**: Majority voting across frames for robust detection
- **SHA-256 Hashing**: Cryptographic payload verification

## Key Types

### Configuration

- `WatermarkConfig` - Embedding parameters (strength, DCT mode, block size, redundancy)
- `ErrorCorrectionLevel` - Redundancy level for error resilience

### Core

- `Watermarker` - Main interface for embedding and extracting watermarks
- `WatermarkPayload` - Payload container with ID, timestamp, custom data, and hash

### Embedding

- `PositionSelector` - Pseudo-random position generator using keyed hashing
- `SpreadSpectrumEmbedder` - Spread spectrum sequence generation and correlation

### Extraction

- `WatermarkDetector` - Threshold-based watermark detection
- `MultiFrameExtractor` - Cross-frame consensus extraction with confidence scoring

### Errors

- `WatermarkError` - Error types: InvalidPayload, Embedding, Extraction, NotFound, Corrupted

## Usage

### Embedding a Watermark

```rust
use transcode_watermark::{Watermarker, WatermarkConfig, WatermarkPayload};

// Create watermarker with default config
let config = WatermarkConfig::default();
let watermarker = Watermarker::new(config);

// Create payload with user ID and custom data
let payload = WatermarkPayload::new("user-12345", b"session-data");

// Embed in Y plane of video frame
let mut frame_data = vec![128u8; 1920 * 1080];
watermarker.embed(&mut frame_data, 1920, 1080, &payload)?;
```

### Extracting a Watermark

```rust
use transcode_watermark::{Watermarker, WatermarkConfig};

// Use same key for extraction
let watermarker = Watermarker::with_key(config, key);

// Extract from frame
let payload = watermarker.extract(&frame_data, 1920, 1080)?;
println!("User ID: {}", payload.id);
println!("Timestamp: {}", payload.timestamp);
```

### Custom Configuration

```rust
use transcode_watermark::{WatermarkConfig, ErrorCorrectionLevel};

let config = WatermarkConfig {
    strength: 0.15,                           // Higher = more visible but robust
    use_dct: true,                            // DCT domain embedding
    block_size: 8,                            // 8x8 DCT blocks
    redundancy: 5,                            // Repeat watermark 5 times
    error_correction: ErrorCorrectionLevel::High,
};
```

### Multi-Frame Detection

```rust
use transcode_watermark::{MultiFrameExtractor, WatermarkDetector};

let mut extractor = MultiFrameExtractor::new();
let detector = WatermarkDetector::new(0.5);

// Add bits from multiple frames
for frame_bits in detected_bits_per_frame {
    extractor.add_frame(frame_bits);
}

// Get consensus with confidence
let bits = extractor.get_consensus()?;
let confidence = extractor.confidence();
```

## License

See workspace root for license information.
