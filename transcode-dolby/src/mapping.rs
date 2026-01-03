//! Tone and gamut mapping from RPU data.
//!
//! This module implements the tone mapping and gamut mapping algorithms
//! used in Dolby Vision, including polynomial coefficient processing,
//! MMR (Multi-resolution Mapping), NLQ (Non-Linear Quantization), and
//! pivot-based curve application.

use crate::error::{DolbyError, Result};
use crate::metadata::{L1Metadata, L2Metadata};
use crate::rpu::{MmrCoefficients, NlqData, Rpu, VdrData};

/// Tone mapping processor.
///
/// Processes video data using Dolby Vision RPU metadata to perform
/// dynamic tone mapping for different target displays.
#[derive(Debug)]
pub struct ToneMapper {
    /// Target display peak luminance in nits.
    pub target_peak_nits: f64,
    /// Target display minimum luminance in nits.
    pub target_min_nits: f64,
    /// Source peak luminance in nits (from content).
    pub source_peak_nits: f64,
    /// Polynomial processing coefficients.
    poly_processor: Option<PolynomialProcessor>,
    /// MMR processor.
    mmr_processor: Option<MmrProcessor>,
    /// NLQ processor.
    nlq_processor: Option<NlqProcessor>,
}

impl ToneMapper {
    /// Create a new tone mapper for a target display.
    pub fn new(target_peak_nits: f64, target_min_nits: f64) -> Self {
        ToneMapper {
            target_peak_nits,
            target_min_nits,
            source_peak_nits: 10000.0, // Default to full PQ range
            poly_processor: None,
            mmr_processor: None,
            nlq_processor: None,
        }
    }

    /// Create a tone mapper for a 1000 nit display.
    pub fn for_1000_nits() -> Self {
        ToneMapper::new(1000.0, 0.0001)
    }

    /// Create a tone mapper for a 100 nit SDR display.
    pub fn for_sdr() -> Self {
        ToneMapper::new(100.0, 0.0001)
    }

    /// Configure the tone mapper from RPU data.
    pub fn configure_from_rpu(&mut self, rpu: &Rpu) -> Result<()> {
        // Get source peak from L1 metadata
        if let Some(l1) = &rpu.metadata.l1 {
            self.source_peak_nits = l1.max_nits();
        }

        // Configure polynomial processor from VDR data
        if let Some(vdr) = &rpu.vdr {
            if !vdr.poly_coefficients.is_empty() {
                self.poly_processor = Some(PolynomialProcessor::from_vdr(vdr)?);
            }

            if let Some(mmr) = &vdr.mmr_coefficients {
                self.mmr_processor = Some(MmrProcessor::new(mmr.clone()));
            }

            if let Some(nlq) = &vdr.nlq_data {
                self.nlq_processor = Some(NlqProcessor::new(nlq.clone()));
            }
        }

        Ok(())
    }

    /// Configure the tone mapper from L2 trim metadata.
    pub fn configure_from_l2(&mut self, l2: &L2Metadata) {
        // Apply trim adjustments
        let slope = l2.slope();
        let offset = l2.offset();
        let power = l2.power();

        // Create a simple polynomial processor from trim values
        self.poly_processor = Some(PolynomialProcessor::from_trim(slope, offset, power));
    }

    /// Map a single luminance value (PQ code value 0-4095).
    pub fn map_luminance(&self, pq_input: u16) -> u16 {
        let normalized = pq_input as f64 / 4095.0;

        // Apply polynomial mapping if available
        let mapped = if let Some(poly) = &self.poly_processor {
            poly.process(normalized)
        } else {
            // Simple linear scaling based on peak luminance ratio
            self.simple_tone_map(normalized)
        };

        // Clamp and convert back to PQ code value
        (mapped.clamp(0.0, 1.0) * 4095.0).round() as u16
    }

    /// Map an RGB pixel (each component as PQ value 0-4095).
    pub fn map_pixel(&self, r: u16, g: u16, b: u16) -> (u16, u16, u16) {
        // For luminance-based tone mapping, we need to:
        // 1. Convert to linear light
        // 2. Calculate luminance
        // 3. Apply tone mapping to luminance
        // 4. Scale RGB accordingly

        let r_linear = pq_to_linear(r as f64 / 4095.0);
        let g_linear = pq_to_linear(g as f64 / 4095.0);
        let b_linear = pq_to_linear(b as f64 / 4095.0);

        // BT.2020 luminance coefficients
        let y = 0.2627 * r_linear + 0.6780 * g_linear + 0.0593 * b_linear;

        if y <= 0.0 {
            return (0, 0, 0);
        }

        // Tone map the luminance
        let y_pq = linear_to_pq(y);
        let y_mapped_pq = self.map_luminance((y_pq * 4095.0) as u16);
        let y_mapped = pq_to_linear(y_mapped_pq as f64 / 4095.0);

        // Scale RGB to maintain color ratios
        let scale = if y > 0.0 { y_mapped / y } else { 1.0 };

        let r_out = linear_to_pq(r_linear * scale);
        let g_out = linear_to_pq(g_linear * scale);
        let b_out = linear_to_pq(b_linear * scale);

        (
            (r_out.clamp(0.0, 1.0) * 4095.0).round() as u16,
            (g_out.clamp(0.0, 1.0) * 4095.0).round() as u16,
            (b_out.clamp(0.0, 1.0) * 4095.0).round() as u16,
        )
    }

    fn simple_tone_map(&self, normalized_pq: f64) -> f64 {
        // Convert to linear, scale, convert back
        let linear = pq_to_linear(normalized_pq) * 10000.0; // nits

        // Simple reinhard-like tone mapping
        let mapped_nits = if linear > self.target_peak_nits {
            // Soft roll-off for values above target peak
            let ratio = linear / self.target_peak_nits;
            self.target_peak_nits * (ratio.ln() + 1.0) / (ratio.ln() + ratio)
        } else {
            linear
        };

        linear_to_pq(mapped_nits / 10000.0)
    }
}

/// Polynomial coefficient processor.
#[derive(Debug)]
pub struct PolynomialProcessor {
    /// Pivot values (normalized 0-1).
    pivots: Vec<f64>,
    /// Coefficients for each piece.
    coefficients: Vec<Vec<f64>>,
    /// Trim slope.
    slope: f64,
    /// Trim offset.
    offset: f64,
    /// Trim power.
    power: f64,
}

impl PolynomialProcessor {
    /// Create from VDR data.
    pub fn from_vdr(vdr: &VdrData) -> Result<Self> {
        // Use the first component (Y/luma) for tone mapping
        let poly = vdr
            .poly_coefficients
            .first()
            .ok_or_else(|| DolbyError::PolynomialError {
                message: "No polynomial coefficients found".to_string(),
            })?;

        Ok(PolynomialProcessor {
            pivots: poly.pivots.clone(),
            coefficients: poly.coefficients.clone(),
            slope: 1.0,
            offset: 0.0,
            power: 1.0,
        })
    }

    /// Create from L2 trim values.
    pub fn from_trim(slope: f64, offset: f64, power: f64) -> Self {
        PolynomialProcessor {
            pivots: vec![0.0, 1.0],
            coefficients: vec![vec![offset, slope]], // Linear: y = slope*x + offset
            slope,
            offset,
            power,
        }
    }

    /// Process a normalized input value (0-1).
    pub fn process(&self, input: f64) -> f64 {
        // Find the piece for this input
        let piece_idx = self.find_piece(input);

        // Apply polynomial
        let poly_result = if piece_idx < self.coefficients.len() {
            self.evaluate_poly(input, &self.coefficients[piece_idx])
        } else {
            input
        };

        // Apply trim adjustments
        let trimmed = (poly_result * self.slope + self.offset).max(0.0);

        // Apply power curve
        if self.power != 1.0 && trimmed > 0.0 {
            trimmed.powf(self.power)
        } else {
            trimmed
        }
    }

    fn find_piece(&self, input: f64) -> usize {
        for (i, pivot) in self.pivots.iter().enumerate().skip(1) {
            if input < *pivot {
                return i - 1;
            }
        }
        self.pivots.len().saturating_sub(2)
    }

    fn evaluate_poly(&self, x: f64, coeffs: &[f64]) -> f64 {
        let mut result = 0.0;
        let mut x_power = 1.0;

        for coeff in coeffs {
            result += coeff * x_power;
            x_power *= x;
        }

        result
    }
}

/// MMR (Multi-resolution Mapping) processor.
#[derive(Debug)]
pub struct MmrProcessor {
    coefficients: MmrCoefficients,
}

impl MmrProcessor {
    /// Create a new MMR processor.
    pub fn new(coefficients: MmrCoefficients) -> Self {
        MmrProcessor { coefficients }
    }

    /// Process a value using MMR.
    pub fn process(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        // MMR uses cross-channel dependencies
        // Order 0: Simple offset
        // Order 1: Linear combination
        // Order 2: Quadratic terms
        // Order 3: Cubic terms

        match self.coefficients.order {
            0 => self.process_order0(y, cb, cr),
            1 => self.process_order1(y, cb, cr),
            2 => self.process_order2(y, cb, cr),
            _ => self.process_order3(y, cb, cr),
        }
    }

    fn process_order0(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        // Simple offset (if coefficients available)
        let offsets = if self.coefficients.coefficients.len() >= 3 {
            (
                self.coefficients.coefficients[0].first().copied().unwrap_or(0.0),
                self.coefficients.coefficients[1].first().copied().unwrap_or(0.0),
                self.coefficients.coefficients[2].first().copied().unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        (y + offsets.0, cb + offsets.1, cr + offsets.2)
    }

    fn process_order1(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        // Linear combination
        // Each output is: c0 + c1*Y + c2*Cb + c3*Cr
        let get_coeffs = |idx: usize| -> [f64; 4] {
            if idx < self.coefficients.coefficients.len() {
                let c = &self.coefficients.coefficients[idx];
                [
                    c.first().copied().unwrap_or(0.0),
                    c.get(1).copied().unwrap_or(1.0),
                    c.get(2).copied().unwrap_or(0.0),
                    c.get(3).copied().unwrap_or(0.0),
                ]
            } else {
                [0.0, 1.0, 0.0, 0.0]
            }
        };

        let cy = get_coeffs(0);
        let ccb = get_coeffs(1);
        let ccr = get_coeffs(2);

        let y_out = cy[0] + cy[1] * y + cy[2] * cb + cy[3] * cr;
        let cb_out = ccb[0] + ccb[1] * y + ccb[2] * cb + ccb[3] * cr;
        let cr_out = ccr[0] + ccr[1] * y + ccr[2] * cb + ccr[3] * cr;

        (y_out, cb_out, cr_out)
    }

    fn process_order2(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        // Quadratic - includes Y*Y, Y*Cb, Y*Cr, Cb*Cb, Cb*Cr, Cr*Cr terms
        // Simplified implementation
        let (y1, cb1, cr1) = self.process_order1(y, cb, cr);
        (y1, cb1, cr1)
    }

    fn process_order3(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        // Cubic - includes all terms up to order 3
        // Simplified implementation
        let (y2, cb2, cr2) = self.process_order2(y, cb, cr);
        (y2, cb2, cr2)
    }
}

/// NLQ (Non-Linear Quantization) processor.
#[derive(Debug)]
pub struct NlqProcessor {
    data: NlqData,
}

impl NlqProcessor {
    /// Create a new NLQ processor.
    pub fn new(data: NlqData) -> Self {
        NlqProcessor { data }
    }

    /// Process a value using NLQ.
    pub fn process(&self, input: f64, component: usize) -> f64 {
        // NLQ applies a non-linear quantization curve
        // Used for enhancement layer processing

        let offset = self
            .data
            .nlq_offset
            .get(component)
            .copied()
            .unwrap_or(0) as f64
            / 4095.0;

        let vdr_in_max = self
            .data
            .vdr_in_max
            .get(component)
            .copied()
            .unwrap_or(4095) as f64
            / 4095.0;

        let deadzone_slope = self
            .data
            .linear_deadzone_slope
            .get(component)
            .copied()
            .unwrap_or(0) as f64
            / 4095.0;

        let deadzone_threshold = self
            .data
            .linear_deadzone_threshold
            .get(component)
            .copied()
            .unwrap_or(0) as f64
            / 4095.0;

        // Apply deadzone
        let deadzoned = if input.abs() < deadzone_threshold {
            input * deadzone_slope
        } else {
            input
        };

        // Scale and offset
        let scaled = deadzoned * vdr_in_max + offset;

        scaled.clamp(0.0, 1.0)
    }

    /// Process YCbCr values.
    pub fn process_ycbcr(&self, y: f64, cb: f64, cr: f64) -> (f64, f64, f64) {
        (
            self.process(y, 0),
            self.process(cb, 1),
            self.process(cr, 2),
        )
    }
}

/// Gamut mapping processor for color space conversion.
#[derive(Debug)]
pub struct GamutMapper {
    /// Source color primaries matrix (to XYZ).
    source_to_xyz: [[f64; 3]; 3],
    /// Target color primaries matrix (from XYZ).
    xyz_to_target: [[f64; 3]; 3],
    /// Enable soft clipping for out-of-gamut colors.
    soft_clip: bool,
}

impl GamutMapper {
    /// Create a gamut mapper from BT.2020 to BT.709.
    pub fn bt2020_to_bt709() -> Self {
        // BT.2020 to XYZ matrix
        let source_to_xyz = [
            [0.6370, 0.1446, 0.1689],
            [0.2627, 0.6780, 0.0593],
            [0.0000, 0.0281, 1.0610],
        ];

        // XYZ to BT.709 matrix
        let xyz_to_target = [
            [3.2410, -1.5374, -0.4986],
            [-0.9692, 1.8760, 0.0416],
            [0.0556, -0.2040, 1.0570],
        ];

        GamutMapper {
            source_to_xyz,
            xyz_to_target,
            soft_clip: true,
        }
    }

    /// Create a gamut mapper from BT.2020 to P3-D65.
    pub fn bt2020_to_p3() -> Self {
        let source_to_xyz = [
            [0.6370, 0.1446, 0.1689],
            [0.2627, 0.6780, 0.0593],
            [0.0000, 0.0281, 1.0610],
        ];

        // XYZ to P3-D65 matrix
        let xyz_to_target = [
            [2.4935, -0.9314, -0.4027],
            [-0.8295, 1.7627, 0.0236],
            [0.0358, -0.0762, 0.9569],
        ];

        GamutMapper {
            source_to_xyz,
            xyz_to_target,
            soft_clip: true,
        }
    }

    /// Enable or disable soft clipping.
    pub fn set_soft_clip(&mut self, enabled: bool) {
        self.soft_clip = enabled;
    }

    /// Map RGB values from source to target gamut.
    pub fn map(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Convert to XYZ
        let x = self.source_to_xyz[0][0] * r
            + self.source_to_xyz[0][1] * g
            + self.source_to_xyz[0][2] * b;
        let y = self.source_to_xyz[1][0] * r
            + self.source_to_xyz[1][1] * g
            + self.source_to_xyz[1][2] * b;
        let z = self.source_to_xyz[2][0] * r
            + self.source_to_xyz[2][1] * g
            + self.source_to_xyz[2][2] * b;

        // Convert to target
        let r_out = self.xyz_to_target[0][0] * x
            + self.xyz_to_target[0][1] * y
            + self.xyz_to_target[0][2] * z;
        let g_out = self.xyz_to_target[1][0] * x
            + self.xyz_to_target[1][1] * y
            + self.xyz_to_target[1][2] * z;
        let b_out = self.xyz_to_target[2][0] * x
            + self.xyz_to_target[2][1] * y
            + self.xyz_to_target[2][2] * z;

        if self.soft_clip {
            self.soft_clip_rgb(r_out, g_out, b_out)
        } else {
            (
                r_out.clamp(0.0, 1.0),
                g_out.clamp(0.0, 1.0),
                b_out.clamp(0.0, 1.0),
            )
        }
    }

    fn soft_clip_rgb(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Soft clipping: reduce saturation for out-of-gamut colors
        let max_val = r.max(g).max(b);
        let min_val = r.min(g).min(b);

        if max_val <= 1.0 && min_val >= 0.0 {
            return (r, g, b);
        }

        // Calculate luminance
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Scale towards gray
        let scale = if max_val > 1.0 {
            (1.0 - y) / (max_val - y)
        } else if min_val < 0.0 {
            y / (y - min_val)
        } else {
            1.0
        };

        let scale = scale.clamp(0.0, 1.0);

        (
            (y + (r - y) * scale).clamp(0.0, 1.0),
            (y + (g - y) * scale).clamp(0.0, 1.0),
            (y + (b - y) * scale).clamp(0.0, 1.0),
        )
    }
}

/// Pivot-based curve application.
#[derive(Debug)]
pub struct PivotCurve {
    /// Pivot input values.
    pub pivots_in: Vec<f64>,
    /// Pivot output values.
    pub pivots_out: Vec<f64>,
}

impl PivotCurve {
    /// Create a new pivot curve.
    pub fn new(pivots_in: Vec<f64>, pivots_out: Vec<f64>) -> Result<Self> {
        if pivots_in.len() != pivots_out.len() {
            return Err(DolbyError::InvalidCoefficients {
                message: "Pivot input and output counts must match".to_string(),
            });
        }

        if pivots_in.len() < 2 {
            return Err(DolbyError::InvalidCoefficients {
                message: "At least 2 pivots required".to_string(),
            });
        }

        Ok(PivotCurve {
            pivots_in,
            pivots_out,
        })
    }

    /// Create a simple identity curve.
    pub fn identity() -> Self {
        PivotCurve {
            pivots_in: vec![0.0, 1.0],
            pivots_out: vec![0.0, 1.0],
        }
    }

    /// Apply the curve to an input value.
    pub fn apply(&self, input: f64) -> f64 {
        // Find the segment
        let mut segment = 0;
        for (i, pivot) in self.pivots_in.iter().enumerate().skip(1) {
            if input < *pivot {
                segment = i - 1;
                break;
            }
            segment = i - 1;
        }

        // Linear interpolation within segment
        let x0 = self.pivots_in[segment];
        let x1 = self.pivots_in.get(segment + 1).copied().unwrap_or(1.0);
        let y0 = self.pivots_out[segment];
        let y1 = self.pivots_out.get(segment + 1).copied().unwrap_or(1.0);

        if (x1 - x0).abs() < f64::EPSILON {
            return y0;
        }

        let t = (input - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}

/// Combined tone and gamut mapping processor.
#[derive(Debug)]
pub struct DolbyVisionMapper {
    /// Tone mapper.
    pub tone_mapper: ToneMapper,
    /// Gamut mapper (optional).
    pub gamut_mapper: Option<GamutMapper>,
    /// L1 metadata for current scene.
    pub l1: Option<L1Metadata>,
}

impl DolbyVisionMapper {
    /// Create a new Dolby Vision mapper.
    pub fn new(target_peak_nits: f64) -> Self {
        DolbyVisionMapper {
            tone_mapper: ToneMapper::new(target_peak_nits, 0.0001),
            gamut_mapper: None,
            l1: None,
        }
    }

    /// Configure from RPU.
    pub fn configure(&mut self, rpu: &Rpu) -> Result<()> {
        self.l1 = rpu.metadata.l1;
        self.tone_mapper.configure_from_rpu(rpu)?;

        // Find L2 for our target display
        let target_pq = nits_to_pq(self.tone_mapper.target_peak_nits);
        if let Some(l2) = rpu.metadata.get_l2_for_target(target_pq as u16) {
            self.tone_mapper.configure_from_l2(l2);
        }

        Ok(())
    }

    /// Enable gamut mapping to BT.709.
    pub fn enable_bt709_gamut(&mut self) {
        self.gamut_mapper = Some(GamutMapper::bt2020_to_bt709());
    }

    /// Enable gamut mapping to P3.
    pub fn enable_p3_gamut(&mut self) {
        self.gamut_mapper = Some(GamutMapper::bt2020_to_p3());
    }

    /// Process a pixel (PQ values 0-4095).
    pub fn process_pixel(&self, r: u16, g: u16, b: u16) -> (u16, u16, u16) {
        // Apply tone mapping
        let (r_tm, g_tm, b_tm) = self.tone_mapper.map_pixel(r, g, b);

        // Apply gamut mapping if enabled
        if let Some(gamut) = &self.gamut_mapper {
            let r_norm = r_tm as f64 / 4095.0;
            let g_norm = g_tm as f64 / 4095.0;
            let b_norm = b_tm as f64 / 4095.0;

            // Convert PQ to linear for gamut mapping
            let r_lin = pq_to_linear(r_norm);
            let g_lin = pq_to_linear(g_norm);
            let b_lin = pq_to_linear(b_norm);

            let (r_gm, g_gm, b_gm) = gamut.map(r_lin, g_lin, b_lin);

            // Convert back to PQ
            let r_out = linear_to_pq(r_gm);
            let g_out = linear_to_pq(g_gm);
            let b_out = linear_to_pq(b_gm);

            (
                (r_out * 4095.0).round() as u16,
                (g_out * 4095.0).round() as u16,
                (b_out * 4095.0).round() as u16,
            )
        } else {
            (r_tm, g_tm, b_tm)
        }
    }
}

// Helper functions for PQ conversion

/// PQ EOTF: Convert PQ signal (0-1) to linear light (0-1).
fn pq_to_linear(pq: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    if pq <= 0.0 {
        return 0.0;
    }

    let pq_pow = pq.powf(1.0 / M2);
    let numerator = (pq_pow - C1).max(0.0);
    let denominator = C2 - C3 * pq_pow;

    if denominator <= 0.0 {
        return 1.0;
    }

    (numerator / denominator).powf(1.0 / M1)
}

/// PQ inverse EOTF: Convert linear light (0-1) to PQ signal (0-1).
fn linear_to_pq(linear: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    if linear <= 0.0 {
        return 0.0;
    }

    let y_pow = linear.powf(M1);
    let numerator = C1 + C2 * y_pow;
    let denominator = 1.0 + C3 * y_pow;

    (numerator / denominator).powf(M2)
}

/// Convert nits to PQ code value.
fn nits_to_pq(nits: f64) -> f64 {
    linear_to_pq(nits / 10000.0) * 4095.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_roundtrip() {
        let values = [0.0, 0.1, 0.5, 0.9, 1.0];

        for &v in &values {
            let pq = linear_to_pq(v);
            let linear = pq_to_linear(pq);
            assert!((v - linear).abs() < 0.0001, "Roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_tone_mapper_for_1000_nits() {
        let mapper = ToneMapper::for_1000_nits();
        assert_eq!(mapper.target_peak_nits, 1000.0);

        // Map a value at 1000 nits (should be near target)
        let pq_1000 = 3079; // ~1000 nits
        let mapped = mapper.map_luminance(pq_1000);
        assert!(mapped <= pq_1000 + 100); // Should not significantly increase
    }

    #[test]
    fn test_gamut_mapper() {
        let mapper = GamutMapper::bt2020_to_bt709();

        // Pure white should stay white
        let (r, g, b) = mapper.map(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.1);
        assert!((g - 1.0).abs() < 0.1);
        assert!((b - 1.0).abs() < 0.1);

        // Black should stay black
        let (r, g, b) = mapper.map(0.0, 0.0, 0.0);
        assert!(r.abs() < 0.001);
        assert!(g.abs() < 0.001);
        assert!(b.abs() < 0.001);
    }

    #[test]
    fn test_pivot_curve() {
        let curve = PivotCurve::new(vec![0.0, 0.5, 1.0], vec![0.0, 0.3, 1.0]).unwrap();

        assert!((curve.apply(0.0) - 0.0).abs() < 0.001);
        assert!((curve.apply(0.5) - 0.3).abs() < 0.001);
        assert!((curve.apply(1.0) - 1.0).abs() < 0.001);

        // Interpolation
        let mid = curve.apply(0.25);
        assert!(mid > 0.0 && mid < 0.3);
    }

    #[test]
    fn test_polynomial_processor() {
        let processor = PolynomialProcessor::from_trim(1.0, 0.0, 1.0);

        // Identity mapping
        assert!((processor.process(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_nlq_processor() {
        let data = NlqData {
            nlq_offset: vec![0, 0, 0],
            vdr_in_max: vec![4095, 4095, 4095],
            linear_deadzone_slope: vec![0, 0, 0],
            linear_deadzone_threshold: vec![0, 0, 0],
        };

        let processor = NlqProcessor::new(data);
        let result = processor.process(0.5, 0);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mmr_processor_order0() {
        let coeffs = MmrCoefficients {
            order: 0,
            coefficients: vec![vec![0.0], vec![0.0], vec![0.0]],
        };

        let processor = MmrProcessor::new(coeffs);
        let (y, cb, cr) = processor.process(0.5, 0.5, 0.5);

        assert!((y - 0.5).abs() < 0.001);
        assert!((cb - 0.5).abs() < 0.001);
        assert!((cr - 0.5).abs() < 0.001);
    }
}
