//! Transfer function implementations for HDR processing.
//!
//! This module provides implementations of electro-optical transfer functions (EOTFs)
//! and opto-electronic transfer functions (OETFs) for various HDR and SDR standards.

use crate::TransferFunction;

/// PQ (Perceptual Quantizer / SMPTE ST.2084) constants
pub mod pq {
    /// PQ constant m1 (1305/8192)
    pub const M1: f64 = 0.1593017578125;
    /// PQ constant m2 (2523/32)
    pub const M2: f64 = 78.84375;
    /// PQ constant c1 (107/128)
    pub const C1: f64 = 0.8359375;
    /// PQ constant c2 (2413/128)
    pub const C2: f64 = 18.8515625;
    /// PQ constant c3 (2392/128)
    pub const C3: f64 = 18.6875;
    /// PQ reference white luminance (nits)
    pub const REFERENCE_WHITE: f64 = 10000.0;
}

/// HLG (Hybrid Log-Gamma / ARIB STD-B67) constants
pub mod hlg {
    /// HLG constant a
    pub const A: f64 = 0.17883277;
    /// HLG constant b (1 - 4a)
    pub const B: f64 = 0.28466892;
    /// HLG constant c (0.5 - a * ln(4a))
    pub const C: f64 = 0.55991073;
    /// HLG reference white luminance (nits)
    pub const REFERENCE_WHITE: f64 = 1000.0;
}

/// BT.1886 constants for SDR display
pub mod bt1886 {
    /// BT.1886 gamma exponent
    pub const GAMMA: f64 = 2.4;
    /// Black luminance for reference display
    pub const L_B: f64 = 0.0;
    /// White luminance for reference display (nits)
    pub const L_W: f64 = 100.0;
}

/// sRGB transfer function constants
pub mod srgb {
    /// Linear threshold
    pub const THRESHOLD: f64 = 0.04045;
    /// Inverse linear threshold
    pub const INV_THRESHOLD: f64 = 0.0031308;
    /// Linear slope
    pub const SLOPE: f64 = 12.92;
    /// Gamma exponent
    pub const GAMMA: f64 = 2.4;
    /// Offset
    pub const OFFSET: f64 = 0.055;
}

/// Transfer function converter for applying EOTFs and OETFs.
#[derive(Debug, Clone, Copy)]
pub struct TransferConverter {
    /// Source transfer function
    pub source: TransferFunction,
    /// Target transfer function
    pub target: TransferFunction,
    /// Reference luminance for normalization (nits)
    pub reference_luminance: f64,
}

impl Default for TransferConverter {
    fn default() -> Self {
        Self {
            source: TransferFunction::Pq,
            target: TransferFunction::Gamma22,
            reference_luminance: 100.0,
        }
    }
}

impl TransferConverter {
    /// Create a new transfer converter.
    pub fn new(source: TransferFunction, target: TransferFunction) -> Self {
        Self {
            source,
            target,
            reference_luminance: 100.0,
        }
    }

    /// Set reference luminance for normalization.
    pub fn with_reference_luminance(mut self, nits: f64) -> Self {
        self.reference_luminance = nits;
        self
    }

    /// Convert a single value from source to target transfer function.
    /// Input and output are in normalized range [0, 1].
    pub fn convert(&self, value: f64) -> f64 {
        // First convert to linear light
        let linear = self.to_linear(value);
        // Then apply target transfer function
        self.from_linear(linear)
    }

    /// Convert value to linear light using source EOTF.
    pub fn to_linear(&self, value: f64) -> f64 {
        match self.source {
            TransferFunction::Pq => pq_eotf(value),
            TransferFunction::Hlg => hlg_eotf(value),
            TransferFunction::Gamma22 => gamma_eotf(value, 2.2),
            TransferFunction::Gamma24 => gamma_eotf(value, 2.4),
            TransferFunction::Srgb => srgb_eotf(value),
            TransferFunction::Linear => value,
            TransferFunction::Bt1886 => bt1886_eotf(value),
        }
    }

    /// Convert linear light to encoded value using target OETF.
    pub fn from_linear(&self, linear: f64) -> f64 {
        match self.target {
            TransferFunction::Pq => pq_oetf(linear),
            TransferFunction::Hlg => hlg_oetf(linear),
            TransferFunction::Gamma22 => gamma_oetf(linear, 2.2),
            TransferFunction::Gamma24 => gamma_oetf(linear, 2.4),
            TransferFunction::Srgb => srgb_oetf(linear),
            TransferFunction::Linear => linear,
            TransferFunction::Bt1886 => bt1886_oetf(linear),
        }
    }

    /// Convert a slice of values in place.
    pub fn convert_slice(&self, values: &mut [f64]) {
        for v in values.iter_mut() {
            *v = self.convert(*v);
        }
    }

    /// Convert a slice of f32 values in place.
    pub fn convert_slice_f32(&self, values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = self.convert(*v as f64) as f32;
        }
    }
}

// ============================================================================
// PQ (ST.2084) Transfer Functions
// ============================================================================

/// PQ EOTF (Electro-Optical Transfer Function).
/// Converts PQ-encoded signal (0-1) to linear light (0-10000 nits normalized to 0-1).
#[inline]
pub fn pq_eotf(e: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }

    let e_pow = e.powf(1.0 / pq::M2);
    let numerator = (e_pow - pq::C1).max(0.0);
    let denominator = pq::C2 - pq::C3 * e_pow;

    if denominator <= 0.0 {
        return 0.0;
    }

    (numerator / denominator).powf(1.0 / pq::M1)
}

/// PQ EOTF returning absolute luminance in nits.
#[inline]
pub fn pq_eotf_nits(e: f64) -> f64 {
    pq_eotf(e) * pq::REFERENCE_WHITE
}

/// PQ OETF (Opto-Electronic Transfer Function).
/// Converts linear light (normalized 0-1, where 1.0 = 10000 nits) to PQ signal.
#[inline]
pub fn pq_oetf(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }

    let y_m1 = y.powf(pq::M1);
    let numerator = pq::C1 + pq::C2 * y_m1;
    let denominator = 1.0 + pq::C3 * y_m1;

    (numerator / denominator).powf(pq::M2)
}

/// PQ OETF from absolute luminance in nits.
#[inline]
pub fn pq_oetf_nits(nits: f64) -> f64 {
    pq_oetf(nits / pq::REFERENCE_WHITE)
}

/// PQ inverse EOTF (same as OETF for round-trip).
#[inline]
pub fn pq_inverse_eotf(y: f64) -> f64 {
    pq_oetf(y)
}

// ============================================================================
// HLG (ARIB STD-B67) Transfer Functions
// ============================================================================

/// HLG OETF (scene-referred).
/// Converts linear scene light (0-1) to HLG signal.
#[inline]
pub fn hlg_oetf(e: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }

    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        hlg::A * (12.0 * e - hlg::B).ln() + hlg::C
    }
}

/// HLG inverse OETF (EOTF for scene light).
#[inline]
pub fn hlg_inverse_oetf(e_prime: f64) -> f64 {
    if e_prime <= 0.0 {
        return 0.0;
    }

    if e_prime <= 0.5 {
        (e_prime * e_prime) / 3.0
    } else {
        (((e_prime - hlg::C) / hlg::A).exp() + hlg::B) / 12.0
    }
}

/// HLG EOTF (display-referred).
/// Converts HLG signal to display light.
#[inline]
pub fn hlg_eotf(e_prime: f64) -> f64 {
    hlg_inverse_oetf(e_prime)
}

/// HLG OOTF (Opto-Optical Transfer Function).
/// Applies system gamma for display rendering.
#[inline]
pub fn hlg_ootf(y: f64, gamma: f64) -> f64 {
    y.powf(gamma - 1.0) * y
}

/// HLG full EOTF with OOTF.
/// Converts HLG signal to display light with system gamma.
#[inline]
pub fn hlg_eotf_full(e_prime: f64, system_gamma: f64, l_w: f64) -> f64 {
    let scene_light = hlg_inverse_oetf(e_prime);
    let display_light = hlg_ootf(scene_light, system_gamma);
    display_light * l_w
}

/// Calculate HLG system gamma based on peak luminance.
/// BT.2100 defines gamma = 1.2 * 1.111^(log2(Lw/1000)).
#[inline]
pub fn hlg_system_gamma(peak_luminance: f64) -> f64 {
    1.2 * (1.111_f64).powf((peak_luminance / 1000.0).log2())
}

// ============================================================================
// SDR Gamma Transfer Functions
// ============================================================================

/// Simple power-law EOTF.
#[inline]
pub fn gamma_eotf(v: f64, gamma: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    v.powf(gamma)
}

/// Simple power-law OETF.
#[inline]
pub fn gamma_oetf(l: f64, gamma: f64) -> f64 {
    if l <= 0.0 {
        return 0.0;
    }
    l.powf(1.0 / gamma)
}

// ============================================================================
// BT.1886 Transfer Functions (Reference SDR Display)
// ============================================================================

/// BT.1886 EOTF for reference SDR display.
/// V is normalized video signal (0-1), returns luminance.
#[inline]
pub fn bt1886_eotf(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }

    // For ideal display with Lb = 0, this simplifies to power law
    v.powf(bt1886::GAMMA)
}

/// BT.1886 EOTF returning absolute luminance.
#[inline]
pub fn bt1886_eotf_nits(v: f64, l_w: f64, l_b: f64) -> f64 {
    if v <= 0.0 {
        return l_b;
    }

    let a = (l_w.powf(1.0 / bt1886::GAMMA) - l_b.powf(1.0 / bt1886::GAMMA)).powf(bt1886::GAMMA);
    let b = l_b.powf(1.0 / bt1886::GAMMA)
        / (l_w.powf(1.0 / bt1886::GAMMA) - l_b.powf(1.0 / bt1886::GAMMA));

    a * (v + b).max(0.0).powf(bt1886::GAMMA)
}

/// BT.1886 inverse EOTF (OETF).
#[inline]
pub fn bt1886_oetf(l: f64) -> f64 {
    if l <= 0.0 {
        return 0.0;
    }
    l.powf(1.0 / bt1886::GAMMA)
}

/// BT.1886 OETF from absolute luminance.
#[inline]
pub fn bt1886_oetf_nits(nits: f64, l_w: f64, l_b: f64) -> f64 {
    if nits <= l_b {
        return 0.0;
    }

    let a = (l_w.powf(1.0 / bt1886::GAMMA) - l_b.powf(1.0 / bt1886::GAMMA)).powf(bt1886::GAMMA);
    let b = l_b.powf(1.0 / bt1886::GAMMA)
        / (l_w.powf(1.0 / bt1886::GAMMA) - l_b.powf(1.0 / bt1886::GAMMA));

    (nits / a).powf(1.0 / bt1886::GAMMA) - b
}

// ============================================================================
// sRGB Transfer Functions
// ============================================================================

/// sRGB EOTF (decoding).
#[inline]
pub fn srgb_eotf(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }

    if v <= srgb::THRESHOLD {
        v / srgb::SLOPE
    } else {
        ((v + srgb::OFFSET) / (1.0 + srgb::OFFSET)).powf(srgb::GAMMA)
    }
}

/// sRGB OETF (encoding).
#[inline]
pub fn srgb_oetf(l: f64) -> f64 {
    if l <= 0.0 {
        return 0.0;
    }

    if l <= srgb::INV_THRESHOLD {
        srgb::SLOPE * l
    } else {
        (1.0 + srgb::OFFSET) * l.powf(1.0 / srgb::GAMMA) - srgb::OFFSET
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert from one transfer function to another for a single value.
#[inline]
pub fn convert_transfer(value: f64, from: TransferFunction, to: TransferFunction) -> f64 {
    if from == to {
        return value;
    }

    let converter = TransferConverter::new(from, to);
    converter.convert(value)
}

/// Convert luminance from nits to normalized PQ value.
#[inline]
pub fn nits_to_pq(nits: f64) -> f64 {
    pq_oetf_nits(nits)
}

/// Convert normalized PQ value to luminance in nits.
#[inline]
pub fn pq_to_nits(pq_value: f64) -> f64 {
    pq_eotf_nits(pq_value)
}

/// Get the reference white luminance for a transfer function.
pub fn reference_white(tf: TransferFunction) -> f64 {
    match tf {
        TransferFunction::Pq => pq::REFERENCE_WHITE,
        TransferFunction::Hlg => hlg::REFERENCE_WHITE,
        TransferFunction::Bt1886 => bt1886::L_W,
        TransferFunction::Gamma22 | TransferFunction::Gamma24 | TransferFunction::Srgb => 100.0,
        TransferFunction::Linear => 1.0,
    }
}

/// Check if a transfer function is HDR.
pub fn is_hdr(tf: TransferFunction) -> bool {
    matches!(tf, TransferFunction::Pq | TransferFunction::Hlg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_pq_roundtrip() {
        let test_values = [0.0, 0.1, 0.5, 0.75, 1.0];
        for &v in &test_values {
            let encoded = pq_oetf(v);
            let decoded = pq_eotf(encoded);
            assert!(
                (v - decoded).abs() < EPSILON,
                "PQ roundtrip failed for {}: got {}",
                v,
                decoded
            );
        }
    }

    #[test]
    fn test_pq_known_values() {
        // PQ(0) = 0
        assert!((pq_oetf(0.0)).abs() < EPSILON);
        // PQ(1) should be close to 1 (10000 nits maps to 1.0)
        assert!((pq_oetf(1.0) - 1.0).abs() < EPSILON);
        // 100 nits should map to approximately 0.508
        let nits_100 = pq_oetf_nits(100.0);
        assert!((nits_100 - 0.508).abs() < 0.01);
    }

    #[test]
    fn test_hlg_roundtrip() {
        let test_values = [0.0, 0.05, 0.1, 0.5, 1.0];
        for &v in &test_values {
            let encoded = hlg_oetf(v);
            let decoded = hlg_inverse_oetf(encoded);
            assert!(
                (v - decoded).abs() < EPSILON,
                "HLG roundtrip failed for {}: got {}",
                v,
                decoded
            );
        }
    }

    #[test]
    fn test_hlg_known_values() {
        // HLG OETF(0) = 0
        assert!((hlg_oetf(0.0)).abs() < EPSILON);
        // HLG OETF(1/12) should be 0.5 (boundary)
        let boundary = hlg_oetf(1.0 / 12.0);
        assert!((boundary - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_srgb_roundtrip() {
        let test_values = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0];
        for &v in &test_values {
            let encoded = srgb_oetf(v);
            let decoded = srgb_eotf(encoded);
            assert!(
                (v - decoded).abs() < EPSILON,
                "sRGB roundtrip failed for {}: got {}",
                v,
                decoded
            );
        }
    }

    #[test]
    fn test_bt1886_roundtrip() {
        let test_values = [0.0, 0.1, 0.5, 0.75, 1.0];
        for &v in &test_values {
            let linear = bt1886_eotf(v);
            let encoded = bt1886_oetf(linear);
            assert!(
                (v - encoded).abs() < EPSILON,
                "BT.1886 roundtrip failed for {}: got {}",
                v,
                encoded
            );
        }
    }

    #[test]
    fn test_gamma_roundtrip() {
        for gamma in [2.2, 2.4, 2.6] {
            for &v in &[0.0, 0.1, 0.5, 1.0] {
                let linear = gamma_eotf(v, gamma);
                let encoded = gamma_oetf(linear, gamma);
                assert!(
                    (v - encoded).abs() < EPSILON,
                    "Gamma {} roundtrip failed for {}: got {}",
                    gamma,
                    v,
                    encoded
                );
            }
        }
    }

    #[test]
    fn test_transfer_converter() {
        let converter = TransferConverter::new(TransferFunction::Pq, TransferFunction::Gamma22);
        let result = converter.convert(0.5);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_hlg_system_gamma() {
        // At 1000 nits, gamma should be 1.2
        let gamma = hlg_system_gamma(1000.0);
        assert!((gamma - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_is_hdr() {
        assert!(is_hdr(TransferFunction::Pq));
        assert!(is_hdr(TransferFunction::Hlg));
        assert!(!is_hdr(TransferFunction::Gamma22));
        assert!(!is_hdr(TransferFunction::Srgb));
    }
}
