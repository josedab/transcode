//! Tone mapping operators for HDR to SDR conversion.
//!
//! This module provides various tone mapping algorithms including:
//! - Reinhard (global and local)
//! - Hable/Uncharted 2 filmic
//! - ACES filmic
//! - BT.2390 EETF (reference)
//! - Mobius tone mapping
//!
//! All operators work on linear light values and support configurable
//! peak luminance and knee points.

use crate::colorspace::{luminance_bt709, luminance_bt2020};

/// Tone mapping algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneMapAlgorithm {
    /// Reinhard global operator
    #[default]
    Reinhard,
    /// Reinhard with local adaptation
    ReinhardLocal,
    /// Hable/Uncharted 2 filmic curve
    Hable,
    /// ACES filmic tone mapping
    Aces,
    /// BT.2390 EETF (reference implementation)
    Bt2390,
    /// Mobius tone mapping
    Mobius,
    /// Simple hard clip
    Clip,
    /// Linear scaling (no compression)
    Linear,
}

/// Configuration for tone mapping operations.
#[derive(Debug, Clone)]
pub struct ToneMapConfig {
    /// Source peak luminance (nits)
    pub source_peak: f64,
    /// Target peak luminance (nits)
    pub target_peak: f64,
    /// Source minimum luminance (nits)
    pub source_min: f64,
    /// Target minimum luminance (nits)
    pub target_min: f64,
    /// Knee point for soft roll-off (0.0-1.0)
    pub knee_point: f64,
    /// Knee width for transition smoothness
    pub knee_width: f64,
    /// Use BT.2020 luminance coefficients (vs BT.709)
    pub use_bt2020_luminance: bool,
    /// Desaturation factor for bright highlights (0.0 = none, 1.0 = full)
    pub highlight_desaturation: f64,
    /// Shadow detail preservation factor
    pub shadow_gain: f64,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            source_peak: 1000.0,
            target_peak: 100.0,
            source_min: 0.0,
            target_min: 0.0,
            knee_point: 0.75,
            knee_width: 0.5,
            use_bt2020_luminance: true,
            highlight_desaturation: 0.0,
            shadow_gain: 1.0,
        }
    }
}

impl ToneMapConfig {
    /// Create config for HDR10 to SDR conversion.
    pub fn hdr10_to_sdr() -> Self {
        Self {
            source_peak: 1000.0,
            target_peak: 100.0,
            source_min: 0.0001,
            target_min: 0.0,
            knee_point: 0.75,
            knee_width: 0.5,
            use_bt2020_luminance: true,
            highlight_desaturation: 0.3,
            shadow_gain: 1.0,
        }
    }

    /// Create config with custom peak luminances.
    pub fn with_peaks(source_peak: f64, target_peak: f64) -> Self {
        Self {
            source_peak,
            target_peak,
            ..Default::default()
        }
    }
}

/// Tone mapper with configurable algorithm and parameters.
#[derive(Debug, Clone)]
pub struct ToneMapper {
    /// Selected algorithm
    pub algorithm: ToneMapAlgorithm,
    /// Configuration
    pub config: ToneMapConfig,
}

impl ToneMapper {
    /// Create a new tone mapper with the given algorithm and configuration.
    pub fn new(algorithm: ToneMapAlgorithm, config: ToneMapConfig) -> Self {
        Self { algorithm, config }
    }

    /// Create a tone mapper with default configuration.
    pub fn with_algorithm(algorithm: ToneMapAlgorithm) -> Self {
        Self {
            algorithm,
            config: ToneMapConfig::default(),
        }
    }

    /// Create a simple tone mapper with peak luminance values.
    pub fn simple(algorithm: ToneMapAlgorithm, source_peak: f64, target_peak: f64) -> Self {
        Self {
            algorithm,
            config: ToneMapConfig::with_peaks(source_peak, target_peak),
        }
    }

    /// Apply tone mapping to a single luminance value.
    /// Input: linear light (0.0 to source_peak/10000)
    /// Output: linear light (0.0 to 1.0)
    #[inline]
    pub fn map_luminance(&self, l: f64) -> f64 {
        // Normalize to source peak
        let normalized = l * 10000.0 / self.config.source_peak;

        let mapped = match self.algorithm {
            ToneMapAlgorithm::Reinhard => reinhard_global(normalized),
            ToneMapAlgorithm::ReinhardLocal => reinhard_extended(normalized, 1.0),
            ToneMapAlgorithm::Hable => hable_filmic(normalized),
            ToneMapAlgorithm::Aces => aces_filmic(normalized),
            ToneMapAlgorithm::Bt2390 => {
                bt2390_eetf(normalized, self.config.knee_point, self.config.knee_width)
            }
            ToneMapAlgorithm::Mobius => {
                mobius(normalized, self.config.knee_point)
            }
            ToneMapAlgorithm::Clip => normalized.min(1.0),
            ToneMapAlgorithm::Linear => normalized * self.config.target_peak / self.config.source_peak,
        };

        mapped.clamp(0.0, 1.0)
    }

    /// Apply tone mapping to RGB triplet.
    /// Preserves color ratios using luminance-based mapping.
    pub fn map_rgb(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Calculate luminance using appropriate coefficients
        let lum = if self.config.use_bt2020_luminance {
            luminance_bt2020(r, g, b)
        } else {
            luminance_bt709(r, g, b)
        };

        if lum <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        // Map luminance
        let mapped_lum = self.map_luminance(lum);

        // Apply the luminance ratio to RGB
        let ratio = mapped_lum / lum;
        let mut new_r = r * ratio;
        let mut new_g = g * ratio;
        let mut new_b = b * ratio;

        // Apply highlight desaturation if configured
        if self.config.highlight_desaturation > 0.0 && mapped_lum > self.config.knee_point {
            let desat_factor = ((mapped_lum - self.config.knee_point) / (1.0 - self.config.knee_point))
                .clamp(0.0, 1.0)
                * self.config.highlight_desaturation;

            new_r = new_r * (1.0 - desat_factor) + mapped_lum * desat_factor;
            new_g = new_g * (1.0 - desat_factor) + mapped_lum * desat_factor;
            new_b = new_b * (1.0 - desat_factor) + mapped_lum * desat_factor;
        }

        (new_r.max(0.0), new_g.max(0.0), new_b.max(0.0))
    }

    /// Apply tone mapping to a slice of RGB triplets in place.
    pub fn apply_slice(&self, pixels: &mut [f64]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.map_rgb(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Apply tone mapping to a slice of f32 RGB triplets in place.
    pub fn apply_slice_f32(&self, pixels: &mut [f32]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.map_rgb(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64);
            chunk[0] = r as f32;
            chunk[1] = g as f32;
            chunk[2] = b as f32;
        }
    }
}

// ============================================================================
// Reinhard Tone Mapping
// ============================================================================

/// Reinhard global operator.
/// Simple: L / (1 + L)
#[inline]
pub fn reinhard_global(l: f64) -> f64 {
    l / (1.0 + l)
}

/// Reinhard extended operator with white point.
/// L * (1 + L/Lwhite^2) / (1 + L)
#[inline]
pub fn reinhard_extended(l: f64, l_white: f64) -> f64 {
    let l_white_sq = l_white * l_white;
    l * (1.0 + l / l_white_sq) / (1.0 + l)
}

/// Reinhard local operator with luminance adaptation.
/// Uses local luminance average for adaptation.
pub fn reinhard_local_pixel(l: f64, l_local_avg: f64, key_value: f64) -> f64 {
    let l_scaled = key_value * l / l_local_avg;
    l_scaled / (1.0 + l_scaled)
}

// ============================================================================
// Hable/Uncharted 2 Filmic
// ============================================================================

/// Hable filmic tone mapping (Uncharted 2).
/// Uses shoulder strength, linear strength, linear angle, toe strength,
/// toe numerator, toe denominator parameters.
#[inline]
pub fn hable_filmic(x: f64) -> f64 {
    // Standard Uncharted 2 parameters
    const A: f64 = 0.15; // Shoulder strength
    const B: f64 = 0.50; // Linear strength
    const C: f64 = 0.10; // Linear angle
    const D: f64 = 0.20; // Toe strength
    const E: f64 = 0.02; // Toe numerator
    const F: f64 = 0.30; // Toe denominator

    fn hable_partial(x: f64) -> f64 {
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }

    // White point (what maps to 1.0)
    const W: f64 = 11.2;
    let white_scale = 1.0 / hable_partial(W);

    hable_partial(x) * white_scale
}

/// Hable filmic with custom parameters.
#[allow(clippy::too_many_arguments)]
pub fn hable_filmic_custom(
    x: f64,
    shoulder_strength: f64,
    linear_strength: f64,
    linear_angle: f64,
    toe_strength: f64,
    toe_numerator: f64,
    toe_denominator: f64,
    white_point: f64,
) -> f64 {
    fn hable_partial_custom(
        x: f64,
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
    ) -> f64 {
        ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
    }

    let white_scale = 1.0
        / hable_partial_custom(
            white_point,
            shoulder_strength,
            linear_strength,
            linear_angle,
            toe_strength,
            toe_numerator,
            toe_denominator,
        );

    hable_partial_custom(
        x,
        shoulder_strength,
        linear_strength,
        linear_angle,
        toe_strength,
        toe_numerator,
        toe_denominator,
    ) * white_scale
}

// ============================================================================
// ACES Filmic Tone Mapping
// ============================================================================

/// ACES filmic tone mapping (simplified RRT + ODT).
/// This is the approximation commonly used in games/real-time.
#[inline]
pub fn aces_filmic(x: f64) -> f64 {
    const A: f64 = 2.51;
    const B: f64 = 0.03;
    const C: f64 = 2.43;
    const D: f64 = 0.59;
    const E: f64 = 0.14;

    let numerator = x * (A * x + B);
    let denominator = x * (C * x + D) + E;

    (numerator / denominator).clamp(0.0, 1.0)
}

/// ACES filmic with exposure adjustment.
#[inline]
pub fn aces_filmic_exposed(x: f64, exposure: f64) -> f64 {
    aces_filmic(x * exposure)
}

// ============================================================================
// BT.2390 EETF (Electro-Electrical Transfer Function)
// ============================================================================

/// BT.2390 EETF implementation.
/// Reference tone mapping algorithm for HDR to SDR conversion.
#[inline]
pub fn bt2390_eetf(e: f64, knee_point: f64, knee_width: f64) -> f64 {
    if e <= knee_point - knee_width / 2.0 {
        // Linear region below knee
        e
    } else if e >= knee_point + knee_width / 2.0 {
        // Compressed region above knee
        let t = (e - knee_point) / (1.0 - knee_point);
        knee_point + (1.0 - knee_point) * hermite_spline(t)
    } else {
        // Transition region (smooth blend)
        let t = (e - (knee_point - knee_width / 2.0)) / knee_width;
        let linear = e;
        let compressed = knee_point + (1.0 - knee_point) * hermite_spline((e - knee_point) / (1.0 - knee_point));
        linear * (1.0 - smoothstep(t)) + compressed * smoothstep(t)
    }
}

/// Full BT.2390 EETF with configurable parameters.
pub fn bt2390_eetf_full(
    e: f64,
    source_min: f64,
    source_max: f64,
    target_min: f64,
    target_max: f64,
) -> f64 {
    // Normalize input
    let e_normalized = (e - source_min) / (source_max - source_min);

    // Calculate knee point based on target range
    let knee_start = target_max / source_max;

    // Apply EETF
    let mapped = if e_normalized <= knee_start {
        e_normalized
    } else {
        let t = (e_normalized - knee_start) / (1.0 - knee_start);
        let compressed = 1.0 - (1.0 - knee_start) * (1.0 - t).powi(2);
        knee_start + (compressed - knee_start) * t
    };

    // Scale to target range
    mapped * (target_max - target_min) + target_min
}

/// Hermite spline for smooth interpolation.
#[inline]
fn hermite_spline(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Smoothstep function.
#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ============================================================================
// Mobius Tone Mapping
// ============================================================================

/// Mobius tone mapping.
/// Provides smooth roll-off with configurable transition point.
#[inline]
pub fn mobius(x: f64, transition: f64) -> f64 {
    if x <= transition {
        x
    } else {
        let a = -transition * transition * (1.0 - transition) / (transition * transition - 2.0 * transition + 1.0);
        let b = (transition * transition - 2.0 * transition * transition * transition + transition * transition * transition * transition) / (transition * transition - 2.0 * transition + 1.0);
        (b * x + a) / (x + b + a - 1.0)
    }
}

/// Mobius with linear section.
pub fn mobius_linear(x: f64, linear_end: f64, transition: f64) -> f64 {
    if x <= linear_end {
        x
    } else {
        let adjusted_x = (x - linear_end) / (1.0 - linear_end);
        let adjusted_transition = (transition - linear_end) / (1.0 - linear_end);
        linear_end + (1.0 - linear_end) * mobius(adjusted_x, adjusted_transition)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate key value (geometric mean luminance) for a frame.
pub fn calculate_key_value(luminances: &[f64], delta: f64) -> f64 {
    if luminances.is_empty() {
        return 0.18; // Default key value
    }

    let log_sum: f64 = luminances.iter().map(|&l| (l + delta).ln()).sum();
    (log_sum / luminances.len() as f64).exp()
}

/// Simple exposure adjustment based on average luminance.
pub fn auto_exposure(avg_luminance: f64, target_mid_gray: f64) -> f64 {
    if avg_luminance <= 0.0 {
        return 1.0;
    }
    target_mid_gray / avg_luminance
}

/// Apply exposure to linear light value.
#[inline]
pub fn apply_exposure(l: f64, exposure: f64) -> f64 {
    l * exposure
}

/// Compute histogram-based percentile luminance.
pub fn percentile_luminance(luminances: &[f64], percentile: f64) -> f64 {
    if luminances.is_empty() {
        return 0.0;
    }

    let mut sorted = luminances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((sorted.len() - 1) as f64 * percentile / 100.0).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_reinhard_global() {
        // At 0, output should be 0
        assert!((reinhard_global(0.0)).abs() < EPSILON);
        // At 1, output should be 0.5
        assert!((reinhard_global(1.0) - 0.5).abs() < EPSILON);
        // Should approach 1 asymptotically
        assert!(reinhard_global(100.0) < 1.0);
        assert!(reinhard_global(100.0) > 0.99);
    }

    #[test]
    fn test_reinhard_extended() {
        // With white point at infinity, should behave like global Reinhard
        // The extended formula: L * (1 + L/Lw^2) / (1 + L)
        // As Lw -> infinity, this becomes L / (1 + L)
        let result_large_white = reinhard_extended(1.0, 1000.0);
        assert!((result_large_white - reinhard_global(1.0)).abs() < 0.01);

        // Higher white point should preserve more highlights
        let result_high = reinhard_extended(0.5, 2.0);
        let result_low = reinhard_extended(0.5, 0.5);
        assert!(result_high < result_low); // Lower white point compresses more
    }

    #[test]
    fn test_hable_filmic() {
        // Should be monotonically increasing
        let v1 = hable_filmic(0.1);
        let v2 = hable_filmic(0.5);
        let v3 = hable_filmic(1.0);
        assert!(v1 < v2);
        assert!(v2 < v3);

        // At the white point (11.2), output should be close to 1.0
        let at_white = hable_filmic(11.2);
        assert!((at_white - 1.0).abs() < 0.01);

        // Should approach asymptote for large values
        let large = hable_filmic(100.0);
        assert!(large > 1.0); // Hable overshoots slightly beyond white point
        assert!(large < 1.5); // But not by too much
    }

    #[test]
    fn test_aces_filmic() {
        // At 0, output should be 0
        assert!((aces_filmic(0.0)).abs() < EPSILON);

        // Should be bounded
        assert!(aces_filmic(10.0) <= 1.0);

        // Monotonically increasing
        let v1 = aces_filmic(0.1);
        let v2 = aces_filmic(0.5);
        assert!(v1 < v2);
    }

    #[test]
    fn test_bt2390_eetf() {
        // Below knee should be linear
        let knee = 0.75;
        let below_knee = bt2390_eetf(0.3, knee, 0.1);
        assert!((below_knee - 0.3).abs() < 0.1);

        // Should be monotonically increasing
        let v1 = bt2390_eetf(0.5, knee, 0.1);
        let v2 = bt2390_eetf(0.9, knee, 0.1);
        assert!(v1 < v2);
    }

    #[test]
    fn test_mobius() {
        // Below transition should be linear
        let transition = 0.5;
        assert!((mobius(0.3, transition) - 0.3).abs() < EPSILON);

        // At transition, should be continuous
        let below = mobius(transition - 0.001, transition);
        let above = mobius(transition + 0.001, transition);
        assert!((above - below).abs() < 0.01);
    }

    #[test]
    fn test_tone_mapper_rgb() {
        let mapper = ToneMapper::simple(ToneMapAlgorithm::Reinhard, 1000.0, 100.0);

        // White should remain white (normalized)
        let (r, g, b) = mapper.map_rgb(0.1, 0.1, 0.1);
        assert!((r - g).abs() < EPSILON);
        assert!((g - b).abs() < EPSILON);

        // Black should remain black
        let (r, g, b) = mapper.map_rgb(0.0, 0.0, 0.0);
        assert!(r.abs() < EPSILON);
        assert!(g.abs() < EPSILON);
        assert!(b.abs() < EPSILON);
    }

    #[test]
    fn test_tone_mapper_slice() {
        let mapper = ToneMapper::simple(ToneMapAlgorithm::Aces, 1000.0, 100.0);
        let mut pixels = vec![0.1, 0.05, 0.2, 0.5, 0.3, 0.1];

        mapper.apply_slice(&mut pixels);

        // All values should be finite and non-negative
        assert!(pixels.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_calculate_key_value() {
        let luminances = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let key = calculate_key_value(&luminances, 0.0001);
        assert!(key > 0.0);
        assert!(key < 1.0);
    }

    #[test]
    fn test_percentile_luminance() {
        let luminances = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // 0th percentile should be minimum
        let p0 = percentile_luminance(&luminances, 0.0);
        assert!((p0 - 0.1).abs() < EPSILON);

        // 100th percentile should be maximum
        let p100 = percentile_luminance(&luminances, 100.0);
        assert!((p100 - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep() {
        // At 0, output should be 0
        assert!((smoothstep(0.0)).abs() < EPSILON);
        // At 1, output should be 1
        assert!((smoothstep(1.0) - 1.0).abs() < EPSILON);
        // At 0.5, output should be 0.5
        assert!((smoothstep(0.5) - 0.5).abs() < EPSILON);
    }
}
