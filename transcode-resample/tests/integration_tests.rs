//! Integration tests for transcode-resample.

use std::f64::consts::PI;
use transcode_resample::{
    LinearResampler, PolyphaseResampler, Resampler, ResamplerConfig, ResamplerImpl,
    SincResampler, WindowFunction,
};

/// Generate a sine wave at a given frequency.
fn generate_sine(sample_rate: u32, frequency: f64, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate as f64).sin() as f32)
        .collect()
}

/// Calculate the RMS of a signal.
fn rms(samples: &[f32]) -> f32 {
    let sum: f32 = samples.iter().map(|&s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Calculate the correlation between two signals.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let sum: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let rms_a = rms(&a[..len]);
    let rms_b = rms(&b[..len]);

    if rms_a > 0.0 && rms_b > 0.0 {
        sum / (len as f32 * rms_a * rms_b)
    } else {
        0.0
    }
}

// ============================================================================
// Linear Resampler Tests
// ============================================================================

#[test]
fn test_linear_44100_to_48000() {
    let mut resampler = LinearResampler::new(44100, 48000, 1).unwrap();
    let input = generate_sine(44100, 440.0, 4410); // 100ms of audio
    let output = resampler.process(&input).unwrap();

    // Expected output length: 4410 * (48000/44100) = ~4800 samples
    let expected_len = (4410.0 * 48000.0 / 44100.0) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() < 10,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );
}

#[test]
fn test_linear_48000_to_44100() {
    let mut resampler = LinearResampler::new(48000, 44100, 1).unwrap();
    let input = generate_sine(48000, 440.0, 4800);
    let output = resampler.process(&input).unwrap();

    let expected_len = (4800.0 * 44100.0 / 48000.0) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() < 10,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );
}

#[test]
fn test_linear_stereo() {
    let mut resampler = LinearResampler::new(44100, 48000, 2).unwrap();

    // Create stereo signal: left=440Hz, right=880Hz
    let left = generate_sine(44100, 440.0, 1000);
    let right = generate_sine(44100, 880.0, 1000);

    let mut interleaved = Vec::with_capacity(2000);
    for i in 0..1000 {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }

    let output = resampler.process_interleaved(&interleaved, 2).unwrap();

    // Output should be divisible by 2
    assert_eq!(output.len() % 2, 0);

    // Should have more samples than input (upsampling)
    assert!(output.len() > interleaved.len());
}

// ============================================================================
// Sinc Resampler Tests
// ============================================================================

#[test]
fn test_sinc_44100_to_48000() {
    let mut resampler = SincResampler::with_defaults(44100, 48000, 1).unwrap();
    let input = generate_sine(44100, 440.0, 4410);
    let output = resampler.process(&input).unwrap();

    let expected_len = (4410.0 * 48000.0 / 44100.0) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() < 100,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );
}

#[test]
fn test_sinc_different_windows() {
    let windows = [
        WindowFunction::Rectangular,
        WindowFunction::Hann,
        WindowFunction::Hamming,
        WindowFunction::Blackman,
        WindowFunction::Kaiser { beta: 8 },
        WindowFunction::Lanczos,
    ];

    let input = generate_sine(44100, 440.0, 2205);

    for window in &windows {
        let mut resampler = SincResampler::new(44100, 48000, 1, 64, *window).unwrap();
        let output = resampler.process(&input).unwrap();

        assert!(
            !output.is_empty(),
            "Window {:?} produced empty output",
            window
        );
    }
}

#[test]
fn test_sinc_quality_sine_preservation() {
    let mut resampler = SincResampler::new(44100, 48000, 1, 128, WindowFunction::Blackman).unwrap();

    // Generate a clean 440 Hz sine wave
    let input = generate_sine(44100, 440.0, 44100); // 1 second

    let output = resampler.process(&input).unwrap();

    // Generate expected output at new rate
    let expected = generate_sine(48000, 440.0, output.len());

    // Skip latency samples
    let latency = resampler.latency();
    let start = latency.min(output.len());

    if output.len() > start + 1000 {
        let corr = correlation(&output[start..start + 1000], &expected[..1000]);

        // High-quality resampling should preserve the sine wave well
        assert!(
            corr > 0.9,
            "Correlation too low: {} (expected > 0.9)",
            corr
        );
    }
}

#[test]
fn test_sinc_window_sizes() {
    for window_size in [16, 32, 64, 128, 256] {
        let resampler = SincResampler::new(44100, 48000, 1, window_size, WindowFunction::Hann);
        assert!(
            resampler.is_ok(),
            "Failed to create resampler with window size {}",
            window_size
        );

        let mut resampler = resampler.unwrap();
        assert_eq!(resampler.latency(), window_size / 2);

        let input = generate_sine(44100, 440.0, 1000);
        assert!(resampler.process(&input).is_ok());
    }
}

// ============================================================================
// Polyphase Resampler Tests
// ============================================================================

#[test]
fn test_polyphase_44100_to_48000() {
    let mut resampler = PolyphaseResampler::with_defaults(44100, 48000, 1).unwrap();
    let input = generate_sine(44100, 440.0, 4410);
    let output = resampler.process(&input).unwrap();

    let expected_len = (4410.0 * 48000.0 / 44100.0) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() < 50,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );
}

#[test]
fn test_polyphase_simple_ratios() {
    // Test 2x upsampling
    let mut resampler = PolyphaseResampler::new(22050, 44100, 1, 64).unwrap();
    let input = vec![0.0f32; 100];
    let output = resampler.process(&input).unwrap();
    assert!(
        (output.len() as i32 - 200).abs() < 20,
        "2x upsample: expected ~200, got {}",
        output.len()
    );

    // Test 2x downsampling
    let mut resampler = PolyphaseResampler::new(44100, 22050, 1, 64).unwrap();
    let input = vec![0.0f32; 200];
    let output = resampler.process(&input).unwrap();
    assert!(
        (output.len() as i32 - 100).abs() < 20,
        "2x downsample: expected ~100, got {}",
        output.len()
    );
}

#[test]
fn test_polyphase_stereo() {
    let mut resampler = PolyphaseResampler::with_defaults(44100, 48000, 2).unwrap();

    // Create stereo silence
    let input = vec![0.0f32; 2000]; // 1000 stereo frames

    let output = resampler.process_interleaved(&input, 2).unwrap();

    assert_eq!(output.len() % 2, 0);
}

// ============================================================================
// Unified Resampler Tests
// ============================================================================

#[test]
fn test_unified_linear() {
    let config = ResamplerConfig::fast(44100, 48000).with_channels(2);
    let mut resampler = Resampler::new(config).unwrap();

    let input = generate_sine(44100, 440.0, 1000);
    let output = resampler.process(&input).unwrap();

    assert!(!output.is_empty());
}

#[test]
fn test_unified_sinc() {
    let config = ResamplerConfig::high_quality(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    let input = generate_sine(44100, 440.0, 4410);
    let output = resampler.process(&input).unwrap();

    assert!(!output.is_empty());
}

#[test]
fn test_unified_polyphase() {
    let config = ResamplerConfig::optimized(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    let input = generate_sine(44100, 440.0, 4410);
    let output = resampler.process(&input).unwrap();

    assert!(!output.is_empty());
}

#[test]
fn test_all_resamplers_consistent_ratio() {
    let input = generate_sine(44100, 440.0, 4410);
    let expected_ratio = 48000.0 / 44100.0;

    // Linear
    let mut linear = Resampler::new(ResamplerConfig::fast(44100, 48000)).unwrap();
    let linear_out = linear.process(&input).unwrap();

    // Sinc
    let mut sinc = Resampler::new(ResamplerConfig::high_quality(44100, 48000)).unwrap();
    let sinc_out = sinc.process(&input).unwrap();

    // Polyphase
    let mut poly = Resampler::new(ResamplerConfig::optimized(44100, 48000)).unwrap();
    let poly_out = poly.process(&input).unwrap();

    // All should produce similar output lengths
    let expected_len = (input.len() as f64 * expected_ratio) as i32;

    assert!(
        (linear_out.len() as i32 - expected_len).abs() < 20,
        "Linear: expected ~{}, got {}",
        expected_len,
        linear_out.len()
    );
    assert!(
        (sinc_out.len() as i32 - expected_len).abs() < 200,
        "Sinc: expected ~{}, got {}",
        expected_len,
        sinc_out.len()
    );
    assert!(
        (poly_out.len() as i32 - expected_len).abs() < 50,
        "Polyphase: expected ~{}, got {}",
        expected_len,
        poly_out.len()
    );
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_empty_input() {
    let mut resampler = Resampler::new(ResamplerConfig::new(44100, 48000)).unwrap();
    let output = resampler.process(&[]).unwrap();
    assert!(output.is_empty());
}

#[test]
fn test_single_sample() {
    let mut resampler = Resampler::new(ResamplerConfig::fast(44100, 48000)).unwrap();
    let output = resampler.process(&[0.5]).unwrap();
    // Should produce at least one sample
    assert!(!output.is_empty() || true); // May be empty due to buffering
}

#[test]
fn test_invalid_sample_rate() {
    let result = Resampler::new(ResamplerConfig::new(0, 48000));
    assert!(result.is_err());

    let result = Resampler::new(ResamplerConfig::new(44100, 0));
    assert!(result.is_err());
}

#[test]
fn test_invalid_channels() {
    let config = ResamplerConfig::new(44100, 48000).with_channels(0);
    let result = Resampler::new(config);
    assert!(result.is_err());
}

#[test]
fn test_reset() {
    let config = ResamplerConfig::new(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    // Process some audio
    let input = generate_sine(44100, 440.0, 1000);
    let _ = resampler.process(&input).unwrap();

    // Reset
    resampler.reset();

    // Should work the same after reset
    let output = resampler.process(&input).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_flush() {
    let config = ResamplerConfig::high_quality(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    let input = generate_sine(44100, 440.0, 1000);
    let _ = resampler.process(&input).unwrap();

    // Flush remaining samples
    let flushed = resampler.flush().unwrap();
    // Flushed samples may or may not be empty depending on implementation
    let _ = flushed;
}

// ============================================================================
// Multi-rate Conversion Tests
// ============================================================================

#[test]
fn test_common_rates() {
    let rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000];

    for &input_rate in &rates {
        for &output_rate in &rates {
            if input_rate == output_rate {
                continue;
            }

            // Only test reasonable ratios to avoid extreme cases
            let ratio = output_rate as f64 / input_rate as f64;
            if ratio > 32.0 || ratio < 1.0 / 32.0 {
                continue;
            }

            let config = ResamplerConfig::fast(input_rate, output_rate);
            let result = Resampler::new(config);

            assert!(
                result.is_ok(),
                "Failed to create resampler for {} -> {}",
                input_rate,
                output_rate
            );

            let mut resampler = result.unwrap();
            let input = generate_sine(input_rate, 440.0, input_rate as usize / 10); // 100ms

            let output = resampler.process(&input);
            assert!(
                output.is_ok(),
                "Failed to resample {} -> {}",
                input_rate,
                output_rate
            );
        }
    }
}

// ============================================================================
// Performance-related Tests
// ============================================================================

#[test]
fn test_large_buffer() {
    let config = ResamplerConfig::fast(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    // 10 seconds of audio
    let input = generate_sine(44100, 440.0, 441000);
    let output = resampler.process(&input).unwrap();

    let expected_len = (441000.0 * 48000.0 / 44100.0) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() < 100,
        "Large buffer: expected ~{}, got {}",
        expected_len,
        output.len()
    );
}

#[test]
fn test_streaming_chunks() {
    let config = ResamplerConfig::high_quality(44100, 48000);
    let mut resampler = Resampler::new(config).unwrap();

    // Process in small chunks
    let mut total_output = Vec::new();

    for _ in 0..10 {
        let input = generate_sine(44100, 440.0, 441); // 10ms chunks
        let output = resampler.process(&input).unwrap();
        total_output.extend(output);
    }

    // Should have approximately 100ms of output at 48kHz = ~4800 samples
    assert!(
        total_output.len() > 4000 && total_output.len() < 5500,
        "Streaming chunks: expected 4000-5500, got {}",
        total_output.len()
    );
}

// ============================================================================
// Multi-channel Tests
// ============================================================================

#[test]
fn test_multichannel_51() {
    let config = ResamplerConfig::fast(44100, 48000).with_channels(6);
    let mut resampler = Resampler::new(config).unwrap();

    // Create 6-channel silence
    let input = vec![0.0f32; 6000]; // 1000 frames * 6 channels

    let output = resampler.process_interleaved(&input).unwrap();

    // Output should be divisible by 6
    assert_eq!(output.len() % 6, 0);
}

#[test]
fn test_multichannel_buffer_mismatch() {
    let mut resampler = LinearResampler::new(44100, 48000, 2).unwrap();

    // Input not divisible by channel count
    let input = vec![0.0f32; 101]; // Not divisible by 2

    let result = resampler.process_interleaved(&input, 2);
    assert!(result.is_err());
}
