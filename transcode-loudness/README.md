# transcode-loudness

EBU R128 loudness normalization and measurement for the Transcode codec library.

## Overview

This crate provides comprehensive loudness processing following international broadcast standards:

- **EBU R128** - European broadcast standard (Target: -23 LUFS)
- **ITU-R BS.1770** - Measurement algorithm for K-weighting and gating
- **ATSC A/85** - US broadcast standard (Target: -24 LKFS)

## Features

- **Integrated loudness (I)** - Program loudness with gating
- **Momentary loudness (M)** - 400ms sliding window
- **Short-term loudness (S)** - 3s sliding window
- **Loudness range (LRA)** - Dynamic range in LU
- **True peak (TP)** - Inter-sample peak detection in dBTP
- **K-weighting filter** - ITU-R BS.1770 pre-filter
- **True peak limiter** - Prevents inter-sample peaks

## Key Types

| Type | Description |
|------|-------------|
| `LoudnessMeter` | EBU R128 loudness measurement |
| `TwoPassNormalizer` | Analyze then normalize workflow |
| `RealtimeNormalizer` | Live loudness adjustment |
| `TruePeakLimiter` | Peak limiting with oversampling |
| `NormalizationMode` | Target presets (EbuR128, AtscA85, Streaming, etc.) |
| `LoudnessResults` | Measurement results (integrated, momentary, range, peak) |

## Normalization Targets

| Standard | Target | Use Case |
|----------|--------|----------|
| EBU R128 | -23 LUFS | European broadcast |
| ATSC A/85 | -24 LKFS | US broadcast |
| Streaming | -14 LUFS | Spotify, YouTube |
| Apple Music | -16 LUFS | iTunes, Apple Music |
| Podcast | -16 LUFS | Podcast platforms |

## Usage

### Measuring Loudness

```rust
use transcode_loudness::{LoudnessMeter, targets};

let mut meter = LoudnessMeter::new(48000, 2)?;
meter.process_interleaved_f64(&samples);

let results = meter.results();
println!("Integrated: {:.1} LUFS", results.integrated);
println!("True peak: {:.1} dBTP", results.true_peak);
println!("Range: {:.1} LU", results.range);
```

### Two-Pass Normalization

```rust
use transcode_loudness::{TwoPassNormalizer, NormalizationMode, NormalizerConfig};

let config = NormalizerConfig::new(48000, 2)
    .with_mode(NormalizationMode::Streaming)  // -14 LUFS
    .with_limiter(true, -1.0);                // -1 dBTP ceiling

let mut normalizer = TwoPassNormalizer::with_config(config)?;

// First pass: analyze
normalizer.analyze_f64(&samples);
let analysis = normalizer.finish_analysis()?;
println!("Gain: {:.1} dB", analysis.gain_db);

// Second pass: apply normalization
normalizer.normalize_f64(&mut output)?;
```

### Using Presets

```rust
use transcode_loudness::presets;

// European broadcast (-23 LUFS, -1 dBTP)
let mut normalizer = presets::broadcast_europe(48000, 2)?;

// US broadcast (-24 LKFS, -2 dBTP)
let mut normalizer = presets::broadcast_us(48000, 2)?;

// Streaming platforms (-14 LUFS)
let mut normalizer = presets::streaming(48000, 2)?;

// Real-time live streaming
let mut realtime = presets::live_streaming(48000, 2)?;
realtime.process_f64(&mut live_buffer);
```

### True Peak Limiting

```rust
use transcode_loudness::{TruePeakLimiter, LimiterConfig};

let config = LimiterConfig::new(48000, 2)
    .with_ceiling(-1.0)   // -1 dBTP
    .with_attack(5.0)     // 5ms attack
    .with_release(100.0); // 100ms release

let mut limiter = TruePeakLimiter::with_config(config)?;
limiter.process_interleaved(&mut samples);

println!("Max reduction: {:.1} dB", limiter.max_gain_reduction_db());
```

## License

See the workspace root for license information.
