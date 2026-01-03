//! Gamut mapping for HDR processing.
//!
//! This module provides algorithms for handling out-of-gamut colors when converting
//! between different color spaces, particularly from wide gamut (BT.2020) to
//! standard gamut (BT.709/sRGB).
//!
//! Key features:
//! - BT.2020 to BT.709 gamut mapping
//! - Soft clipping for out-of-gamut colors
//! - Multiple gamut compression algorithms
//! - Perceptual gamut mapping

use crate::colorspace::{luminance_bt709, ColorSpaceConverter};
use crate::ColorSpace;

/// Gamut mapping algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GamutMappingAlgorithm {
    /// Hard clip to gamut boundary (fast but can cause color shifts)
    Clip,
    /// Soft clip with smooth roll-off near boundary
    #[default]
    SoftClip,
    /// Preserve luminance, desaturate toward white
    DesaturateToWhite,
    /// Preserve luminance, desaturate toward neutral axis
    DesaturateToNeutral,
    /// ACES-style gamut compression
    AcesGamutCompress,
    /// BT.2407 gamut mapping (ITU recommendation)
    Bt2407,
    /// Cusp-based gamut mapping
    CuspMapping,
}

/// Configuration for gamut mapping operations.
#[derive(Debug, Clone)]
pub struct GamutMapConfig {
    /// Source color space
    pub source: ColorSpace,
    /// Target color space
    pub target: ColorSpace,
    /// Mapping algorithm
    pub algorithm: GamutMappingAlgorithm,
    /// Threshold for soft clipping (0.0 - 1.0)
    pub soft_clip_threshold: f64,
    /// Knee width for soft clipping
    pub soft_clip_knee: f64,
    /// Maximum desaturation (0.0 - 1.0)
    pub max_desaturation: f64,
    /// Preserve luminance during mapping
    pub preserve_luminance: bool,
}

impl Default for GamutMapConfig {
    fn default() -> Self {
        Self {
            source: ColorSpace::Bt2020,
            target: ColorSpace::Bt709,
            algorithm: GamutMappingAlgorithm::SoftClip,
            soft_clip_threshold: 0.8,
            soft_clip_knee: 0.5,
            max_desaturation: 1.0,
            preserve_luminance: true,
        }
    }
}

impl GamutMapConfig {
    /// Create config for BT.2020 to BT.709 conversion.
    pub fn bt2020_to_bt709() -> Self {
        Self::default()
    }

    /// Create config with specific algorithm.
    pub fn with_algorithm(algorithm: GamutMappingAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }
}

/// Gamut mapper with configurable algorithm and parameters.
#[derive(Debug, Clone)]
pub struct GamutMapper {
    /// Configuration
    pub config: GamutMapConfig,
    /// Color space converter
    converter: ColorSpaceConverter,
}

impl GamutMapper {
    /// Create a new gamut mapper with the given configuration.
    pub fn new(config: GamutMapConfig) -> Self {
        let converter = ColorSpaceConverter::new(config.source, config.target);
        Self { config, converter }
    }

    /// Create a gamut mapper for BT.2020 to BT.709 with default settings.
    pub fn bt2020_to_bt709() -> Self {
        Self::new(GamutMapConfig::bt2020_to_bt709())
    }

    /// Map a single RGB value from source to target gamut.
    /// Input values should be in linear light, source color space.
    pub fn map(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // First, convert to target color space
        let (tr, tg, tb) = self.converter.convert(r, g, b);

        // Check if in gamut
        if is_in_gamut(tr, tg, tb) {
            return (tr, tg, tb);
        }

        // Apply gamut mapping based on algorithm
        match self.config.algorithm {
            GamutMappingAlgorithm::Clip => clip_to_gamut(tr, tg, tb),
            GamutMappingAlgorithm::SoftClip => {
                soft_clip(tr, tg, tb, self.config.soft_clip_threshold, self.config.soft_clip_knee)
            }
            GamutMappingAlgorithm::DesaturateToWhite => {
                desaturate_to_white(tr, tg, tb, self.config.preserve_luminance)
            }
            GamutMappingAlgorithm::DesaturateToNeutral => {
                desaturate_to_neutral(tr, tg, tb, self.config.preserve_luminance)
            }
            GamutMappingAlgorithm::AcesGamutCompress => aces_gamut_compress(tr, tg, tb),
            GamutMappingAlgorithm::Bt2407 => {
                bt2407_gamut_map(tr, tg, tb, self.config.max_desaturation)
            }
            GamutMappingAlgorithm::CuspMapping => cusp_gamut_map(tr, tg, tb),
        }
    }

    /// Map RGB values (f32 version).
    #[inline]
    pub fn map_f32(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let (nr, ng, nb) = self.map(r as f64, g as f64, b as f64);
        (nr as f32, ng as f32, nb as f32)
    }

    /// Map a slice of RGB triplets in place.
    pub fn map_slice(&self, pixels: &mut [f64]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.map(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Map a slice of f32 RGB triplets in place.
    pub fn map_slice_f32(&self, pixels: &mut [f32]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.map_f32(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Check if a color is within the target gamut after conversion.
    pub fn is_in_gamut(&self, r: f64, g: f64, b: f64) -> bool {
        let (tr, tg, tb) = self.converter.convert(r, g, b);
        is_in_gamut(tr, tg, tb)
    }
}

// ============================================================================
// Gamut Checking Utilities
// ============================================================================

/// Check if RGB values are within the 0-1 gamut.
#[inline]
pub fn is_in_gamut(r: f64, g: f64, b: f64) -> bool {
    (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b)
}

/// Check if RGB values are within the 0-1 gamut with tolerance.
#[inline]
pub fn is_in_gamut_with_tolerance(r: f64, g: f64, b: f64, tolerance: f64) -> bool {
    r >= -tolerance
        && r <= 1.0 + tolerance
        && g >= -tolerance
        && g <= 1.0 + tolerance
        && b >= -tolerance
        && b <= 1.0 + tolerance
}

/// Calculate how far outside the gamut a color is.
/// Returns 0.0 if in gamut, positive value if out of gamut.
pub fn gamut_distance(r: f64, g: f64, b: f64) -> f64 {
    let dr = if r < 0.0 { -r } else if r > 1.0 { r - 1.0 } else { 0.0 };
    let dg = if g < 0.0 { -g } else if g > 1.0 { g - 1.0 } else { 0.0 };
    let db = if b < 0.0 { -b } else if b > 1.0 { b - 1.0 } else { 0.0 };

    (dr * dr + dg * dg + db * db).sqrt()
}

// ============================================================================
// Clipping Methods
// ============================================================================

/// Hard clip to gamut boundary.
#[inline]
pub fn clip_to_gamut(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Soft clip with smooth roll-off.
pub fn soft_clip(r: f64, g: f64, b: f64, threshold: f64, knee: f64) -> (f64, f64, f64) {
    let soft_clip_channel = |v: f64| -> f64 {
        if v < 0.0 {
            // Below zero: soft clip to zero
            let t = -v / knee;
            -knee * soft_knee(t)
        } else if v > threshold {
            // Above threshold: soft clip toward 1.0
            let excess = v - threshold;
            let remaining = 1.0 - threshold;
            let t = excess / (remaining * 2.0);
            threshold + remaining * soft_knee(t)
        } else {
            v
        }
    };

    (
        soft_clip_channel(r).clamp(0.0, 1.0),
        soft_clip_channel(g).clamp(0.0, 1.0),
        soft_clip_channel(b).clamp(0.0, 1.0),
    )
}

/// Soft knee function for smooth roll-off.
#[inline]
fn soft_knee(t: f64) -> f64 {
    if t <= 0.0 {
        0.0
    } else if t >= 1.0 {
        1.0
    } else {
        t * t * (3.0 - 2.0 * t)
    }
}

// ============================================================================
// Desaturation Methods
// ============================================================================

/// Desaturate toward white to bring color into gamut.
pub fn desaturate_to_white(r: f64, g: f64, b: f64, preserve_luminance: bool) -> (f64, f64, f64) {
    let luminance = luminance_bt709(r, g, b);

    // Binary search for the saturation level that brings us into gamut
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..16 {
        let mid = (low + high) / 2.0;
        let nr = r * mid + luminance * (1.0 - mid);
        let ng = g * mid + luminance * (1.0 - mid);
        let nb = b * mid + luminance * (1.0 - mid);

        if is_in_gamut(nr, ng, nb) {
            low = mid;
        } else {
            high = mid;
        }
    }

    let saturation = low;
    let mut nr = r * saturation + luminance * (1.0 - saturation);
    let mut ng = g * saturation + luminance * (1.0 - saturation);
    let mut nb = b * saturation + luminance * (1.0 - saturation);

    // Preserve original luminance if requested
    if preserve_luminance && luminance > 0.0 {
        let new_luminance = luminance_bt709(nr, ng, nb);
        if new_luminance > 0.0 {
            let scale = luminance / new_luminance;
            nr *= scale;
            ng *= scale;
            nb *= scale;
        }
    }

    clip_to_gamut(nr, ng, nb)
}

/// Desaturate toward neutral axis (gray) to bring color into gamut.
pub fn desaturate_to_neutral(r: f64, g: f64, b: f64, _preserve_luminance: bool) -> (f64, f64, f64) {
    let luminance = luminance_bt709(r, g, b);
    let neutral = luminance; // Neutral axis at same luminance

    // Binary search for the saturation level
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..16 {
        let mid = (low + high) / 2.0;
        let nr = r * mid + neutral * (1.0 - mid);
        let ng = g * mid + neutral * (1.0 - mid);
        let nb = b * mid + neutral * (1.0 - mid);

        if is_in_gamut(nr, ng, nb) {
            low = mid;
        } else {
            high = mid;
        }
    }

    let saturation = low;
    let nr = r * saturation + neutral * (1.0 - saturation);
    let ng = g * saturation + neutral * (1.0 - saturation);
    let nb = b * saturation + neutral * (1.0 - saturation);

    clip_to_gamut(nr, ng, nb)
}

// ============================================================================
// Advanced Gamut Mapping Algorithms
// ============================================================================

/// ACES-style gamut compression.
/// Compresses out-of-gamut colors while preserving in-gamut colors.
pub fn aces_gamut_compress(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // ACES gamut compress parameters
    const THRESHOLD_CYAN: f64 = 0.815;
    const THRESHOLD_MAGENTA: f64 = 0.803;
    const THRESHOLD_YELLOW: f64 = 0.880;
    const LIMIT: f64 = 1.2;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);

    if max <= 1.0 && min >= 0.0 {
        return (r, g, b);
    }

    // Calculate distance from achromatic
    let achromatic = (r + g + b) / 3.0;
    let dist_r = (r - achromatic).abs();
    let dist_g = (g - achromatic).abs();
    let dist_b = (b - achromatic).abs();

    // Compress each channel
    let compress = |v: f64, dist: f64, threshold: f64| -> f64 {
        if dist <= threshold {
            v
        } else {
            let compressed_dist = threshold + (dist - threshold) / (1.0 + (dist - threshold) / (LIMIT - threshold));
            achromatic + (v - achromatic).signum() * compressed_dist
        }
    };

    let nr = compress(r, dist_r, THRESHOLD_CYAN);
    let ng = compress(g, dist_g, THRESHOLD_MAGENTA);
    let nb = compress(b, dist_b, THRESHOLD_YELLOW);

    clip_to_gamut(nr, ng, nb)
}

/// BT.2407 gamut mapping (ITU-R BT.2407).
pub fn bt2407_gamut_map(r: f64, g: f64, b: f64, max_desaturation: f64) -> (f64, f64, f64) {
    let luminance = luminance_bt709(r, g, b);

    // Check if in gamut
    if is_in_gamut(r, g, b) {
        return (r, g, b);
    }

    // Calculate saturation
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;

    if chroma <= 0.0 {
        return clip_to_gamut(r, g, b);
    }

    // Find the maximum saturation that keeps us in gamut
    let mut scale = 1.0;
    for _ in 0..16 {
        let nr = luminance + (r - luminance) * scale;
        let ng = luminance + (g - luminance) * scale;
        let nb = luminance + (b - luminance) * scale;

        if is_in_gamut(nr, ng, nb) {
            break;
        }
        scale *= 0.9;
    }

    // Apply desaturation with limit
    scale = scale.max(1.0 - max_desaturation);

    let nr = luminance + (r - luminance) * scale;
    let ng = luminance + (g - luminance) * scale;
    let nb = luminance + (b - luminance) * scale;

    clip_to_gamut(nr, ng, nb)
}

/// Cusp-based gamut mapping.
/// Maps colors toward the cusp of the gamut boundary.
pub fn cusp_gamut_map(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let luminance = luminance_bt709(r, g, b);

    // Check if in gamut
    if is_in_gamut(r, g, b) {
        return (r, g, b);
    }

    // Find the dominant hue
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;

    if chroma <= 0.0 {
        return (luminance.clamp(0.0, 1.0), luminance.clamp(0.0, 1.0), luminance.clamp(0.0, 1.0));
    }

    // Calculate hue (for future per-hue cusp calculation)
    let _hue = if max == r {
        ((g - b) / chroma) % 6.0
    } else if max == g {
        (b - r) / chroma + 2.0
    } else {
        (r - g) / chroma + 4.0
    };

    // Find cusp luminance for this hue (simplified)
    let cusp_luminance = 0.5; // Simplified - real implementation would calculate per-hue

    // Map toward cusp
    let target_luminance = luminance.clamp(0.0, 1.0);
    let factor = if luminance > cusp_luminance {
        (1.0 - target_luminance) / (1.0 - cusp_luminance).max(0.001)
    } else {
        target_luminance / cusp_luminance.max(0.001)
    };

    let scale = factor.min(1.0);

    let nr = target_luminance + (r - luminance) * scale;
    let ng = target_luminance + (g - luminance) * scale;
    let nb = target_luminance + (b - luminance) * scale;

    clip_to_gamut(nr, ng, nb)
}

// ============================================================================
// Gamut Analysis
// ============================================================================

/// Analyze gamut coverage for a set of colors.
pub struct GamutAnalysis {
    /// Total number of pixels
    pub total_pixels: usize,
    /// Number of in-gamut pixels
    pub in_gamut_pixels: usize,
    /// Number of out-of-gamut pixels
    pub out_of_gamut_pixels: usize,
    /// Maximum gamut distance
    pub max_distance: f64,
    /// Average gamut distance for out-of-gamut pixels
    pub avg_out_of_gamut_distance: f64,
}

impl GamutAnalysis {
    /// Calculate the percentage of in-gamut pixels.
    pub fn in_gamut_percentage(&self) -> f64 {
        if self.total_pixels == 0 {
            return 100.0;
        }
        (self.in_gamut_pixels as f64 / self.total_pixels as f64) * 100.0
    }
}

/// Analyze gamut coverage for a frame.
pub fn analyze_gamut(pixels: &[f64], source: ColorSpace, target: ColorSpace) -> GamutAnalysis {
    let converter = ColorSpaceConverter::new(source, target);

    let mut total = 0;
    let mut in_gamut = 0;
    let mut max_distance = 0.0f64;
    let mut total_distance = 0.0;

    for chunk in pixels.chunks_exact(3) {
        total += 1;
        let (r, g, b) = converter.convert(chunk[0], chunk[1], chunk[2]);

        if is_in_gamut(r, g, b) {
            in_gamut += 1;
        } else {
            let dist = gamut_distance(r, g, b);
            max_distance = max_distance.max(dist);
            total_distance += dist;
        }
    }

    let out_of_gamut = total - in_gamut;
    let avg_distance = if out_of_gamut > 0 {
        total_distance / out_of_gamut as f64
    } else {
        0.0
    };

    GamutAnalysis {
        total_pixels: total,
        in_gamut_pixels: in_gamut,
        out_of_gamut_pixels: out_of_gamut,
        max_distance,
        avg_out_of_gamut_distance: avg_distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_is_in_gamut() {
        assert!(is_in_gamut(0.5, 0.5, 0.5));
        assert!(is_in_gamut(0.0, 0.0, 0.0));
        assert!(is_in_gamut(1.0, 1.0, 1.0));
        assert!(!is_in_gamut(-0.1, 0.5, 0.5));
        assert!(!is_in_gamut(0.5, 1.1, 0.5));
    }

    #[test]
    fn test_clip_to_gamut() {
        let (r, g, b) = clip_to_gamut(-0.1, 1.2, 0.5);
        assert_eq!(r, 0.0);
        assert_eq!(g, 1.0);
        assert_eq!(b, 0.5);
    }

    #[test]
    fn test_soft_clip() {
        // In-gamut values should pass through
        let (r, g, b) = soft_clip(0.5, 0.5, 0.5, 0.8, 0.5);
        assert!((r - 0.5).abs() < EPSILON);
        assert!((g - 0.5).abs() < EPSILON);
        assert!((b - 0.5).abs() < EPSILON);

        // Out-of-gamut values should be compressed
        let (r, _g, b) = soft_clip(1.2, 0.5, -0.1, 0.8, 0.5);
        assert!(r <= 1.0);
        assert!(r > 0.8); // Should be compressed, not clipped
        assert!(b >= 0.0);
    }

    #[test]
    fn test_gamut_distance() {
        // In-gamut should have distance 0
        assert!((gamut_distance(0.5, 0.5, 0.5)).abs() < EPSILON);

        // Out of gamut should have positive distance
        assert!(gamut_distance(1.1, 0.5, 0.5) > 0.0);
        assert!(gamut_distance(-0.1, 0.5, 0.5) > 0.0);
    }

    #[test]
    fn test_desaturate_to_white() {
        // Out-of-gamut saturated color
        let (r, g, b) = desaturate_to_white(1.5, 0.0, 0.0, true);
        assert!(is_in_gamut(r, g, b));
        // Should be reddish but desaturated
        assert!(r > g);
        assert!(r > b);
    }

    #[test]
    fn test_desaturate_to_neutral() {
        let (r, g, b) = desaturate_to_neutral(1.3, -0.1, 0.2, true);
        assert!(is_in_gamut(r, g, b));
    }

    #[test]
    fn test_aces_gamut_compress() {
        // In-gamut should pass through
        let (r, g, b) = aces_gamut_compress(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.3).abs() < 0.01);
        assert!((b - 0.7).abs() < 0.01);

        // Out-of-gamut should be compressed
        let (r, g, b) = aces_gamut_compress(1.2, -0.1, 0.5);
        assert!(is_in_gamut(r, g, b));
    }

    #[test]
    fn test_gamut_mapper() {
        let mapper = GamutMapper::bt2020_to_bt709();

        // In-gamut color should pass through (approximately)
        let (r, g, b) = mapper.map(0.5, 0.5, 0.5);
        assert!(is_in_gamut(r, g, b));

        // Saturated BT.2020 color should be mapped to gamut
        let (r, g, b) = mapper.map(0.0, 1.0, 0.0);
        assert!(is_in_gamut(r, g, b));
    }

    #[test]
    fn test_gamut_analysis() {
        let pixels = vec![
            0.5, 0.5, 0.5,  // In gamut
            0.1, 0.9, 0.2,  // In gamut
            1.5, 0.0, 0.0,  // Out of gamut (already in target space for simplicity)
        ];

        let analysis = analyze_gamut(&pixels, ColorSpace::Bt709, ColorSpace::Bt709);
        assert_eq!(analysis.total_pixels, 3);
        assert_eq!(analysis.in_gamut_pixels, 2);
        assert_eq!(analysis.out_of_gamut_pixels, 1);
    }

    #[test]
    fn test_bt2407_gamut_map() {
        let (r, g, b) = bt2407_gamut_map(1.2, 0.0, -0.1, 0.5);
        assert!(is_in_gamut(r, g, b));
    }

    #[test]
    fn test_cusp_gamut_map() {
        let (r, g, b) = cusp_gamut_map(1.3, 0.5, -0.2);
        assert!(is_in_gamut(r, g, b));
    }

    #[test]
    fn test_gamut_mapper_slice() {
        let mapper = GamutMapper::bt2020_to_bt709();
        let mut pixels = vec![0.5, 0.5, 0.5, 0.0, 1.0, 0.0];

        mapper.map_slice(&mut pixels);

        // All pixels should be in gamut after mapping
        for chunk in pixels.chunks_exact(3) {
            assert!(is_in_gamut(chunk[0], chunk[1], chunk[2]));
        }
    }
}
