//! Ambisonics support for spatial audio.
//!
//! This module provides comprehensive ambisonics processing including:
//! - First-order ambisonics (FOA) with 4 channels (WXYZ)
//! - Higher-order ambisonics (HOA) with ACN/SN3D ordering
//! - Ambisonics to binaural conversion
//! - Ambisonics to speaker array decoding
//! - B-format handling

use crate::channels::{ChannelLayout, ChannelPosition, StandardLayout};
use crate::error::{AmbisonicsError, Result, SpatialError};
use std::f32::consts::PI;

/// Ambisonics channel ordering conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AmbisonicsOrdering {
    /// Ambisonic Channel Number (ACN) - standard for HOA.
    /// Order: W, Y, Z, X, V, T, R, S, U, ...
    #[default]
    Acn,
    /// Furse-Malham (FuMa) - traditional B-format ordering.
    /// Order: W, X, Y, Z (for first order)
    FuMa,
    /// SID ordering (used in some older systems).
    Sid,
}

/// Ambisonics normalization conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AmbisonicsNormalization {
    /// SN3D (Schmidt semi-normalized) - standard for AmbiX.
    #[default]
    Sn3d,
    /// N3D (fully normalized).
    N3d,
    /// FuMa normalization (MaxN for W, SN3D-like for others).
    FuMa,
    /// MaxN normalization.
    MaxN,
}

/// Ambisonics format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AmbisonicsFormat {
    /// Ambisonics order (0 = mono, 1 = FOA, 2+ = HOA).
    pub order: u32,
    /// Channel ordering convention.
    pub ordering: AmbisonicsOrdering,
    /// Normalization convention.
    pub normalization: AmbisonicsNormalization,
}

impl AmbisonicsFormat {
    /// Create a new ambisonics format.
    pub fn new(order: u32, ordering: AmbisonicsOrdering, normalization: AmbisonicsNormalization) -> Self {
        Self {
            order,
            ordering,
            normalization,
        }
    }

    /// Create standard AmbiX format (ACN ordering, SN3D normalization).
    pub fn ambix(order: u32) -> Self {
        Self {
            order,
            ordering: AmbisonicsOrdering::Acn,
            normalization: AmbisonicsNormalization::Sn3d,
        }
    }

    /// Create traditional FuMa B-format.
    pub fn fuma(order: u32) -> Self {
        Self {
            order,
            ordering: AmbisonicsOrdering::FuMa,
            normalization: AmbisonicsNormalization::FuMa,
        }
    }

    /// Get the number of channels for this ambisonics order.
    /// Formula: (order + 1)^2
    pub fn channel_count(&self) -> u32 {
        (self.order + 1) * (self.order + 1)
    }

    /// Check if this is first-order ambisonics (4 channels).
    pub fn is_first_order(&self) -> bool {
        self.order == 1
    }

    /// Check if this is higher-order ambisonics (order >= 2).
    pub fn is_higher_order(&self) -> bool {
        self.order >= 2
    }
}

/// B-format representation for first-order ambisonics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BFormat {
    /// W - omnidirectional (pressure) component.
    pub w: f32,
    /// X - front-back component.
    pub x: f32,
    /// Y - left-right component.
    pub y: f32,
    /// Z - up-down component.
    pub z: f32,
}

impl BFormat {
    /// Create a new B-format sample.
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Create B-format from a mono source at a specific direction.
    /// Azimuth: 0 = front, positive = left (counter-clockwise).
    /// Elevation: 0 = horizontal, positive = up.
    pub fn from_direction(mono: f32, azimuth: f32, elevation: f32) -> Self {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        let cos_el = el_rad.cos();

        Self {
            w: mono * 0.707107, // 1/sqrt(2) for SN3D normalization
            x: mono * az_rad.cos() * cos_el,
            y: mono * az_rad.sin() * cos_el,
            z: mono * el_rad.sin(),
        }
    }

    /// Create silence (all zeros).
    pub fn silence() -> Self {
        Self {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Add another B-format signal.
    pub fn add(&self, other: &BFormat) -> Self {
        Self {
            w: self.w + other.w,
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Scale by a factor.
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            w: self.w * factor,
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    /// Rotate around the Z-axis (yaw).
    pub fn rotate_z(&self, angle: f32) -> Self {
        let rad = angle.to_radians();
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        Self {
            w: self.w,
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
            z: self.z,
        }
    }

    /// Convert to channel array in ACN order [W, Y, Z, X].
    pub fn to_acn_array(&self) -> [f32; 4] {
        [self.w, self.y, self.z, self.x]
    }

    /// Convert to channel array in FuMa order [W, X, Y, Z].
    pub fn to_fuma_array(&self) -> [f32; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Create from ACN ordered array [W, Y, Z, X].
    pub fn from_acn_array(arr: &[f32; 4]) -> Self {
        Self {
            w: arr[0],
            y: arr[1],
            z: arr[2],
            x: arr[3],
        }
    }

    /// Create from FuMa ordered array [W, X, Y, Z].
    pub fn from_fuma_array(arr: &[f32; 4]) -> Self {
        Self {
            w: arr[0],
            x: arr[1],
            y: arr[2],
            z: arr[3],
        }
    }
}

impl Default for BFormat {
    fn default() -> Self {
        Self::silence()
    }
}

/// Ambisonics encoder for placing mono sources in 3D space.
#[derive(Debug, Clone)]
pub struct AmbisonicsEncoder {
    format: AmbisonicsFormat,
    /// Encoding matrix for each ACN channel.
    encoding_matrix: Vec<f32>,
}

impl AmbisonicsEncoder {
    /// Create a new ambisonics encoder.
    pub fn new(format: AmbisonicsFormat) -> Self {
        let channel_count = format.channel_count() as usize;
        Self {
            format,
            encoding_matrix: vec![0.0; channel_count],
        }
    }

    /// Get the format.
    pub fn format(&self) -> &AmbisonicsFormat {
        &self.format
    }

    /// Encode a mono source at a specific direction.
    pub fn encode(&mut self, mono: f32, azimuth: f32, elevation: f32) -> Vec<f32> {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        let cos_el = el_rad.cos();
        let sin_el = el_rad.sin();
        let cos_az = az_rad.cos();
        let sin_az = az_rad.sin();

        let mut output = vec![0.0; self.format.channel_count() as usize];

        // ACN channel ordering with SN3D normalization
        // Order 0: W
        output[0] = mono; // W (SN3D normalized)

        if self.format.order >= 1 {
            // Order 1: Y, Z, X
            output[1] = mono * sin_az * cos_el; // Y
            output[2] = mono * sin_el; // Z
            output[3] = mono * cos_az * cos_el; // X
        }

        if self.format.order >= 2 {
            // Order 2: V, T, R, S, U
            let cos_2az = (2.0 * az_rad).cos();
            let sin_2az = (2.0 * az_rad).sin();
            let sin_2el = (2.0 * el_rad).sin();

            output[4] = mono * 1.732051 * sin_az * cos_az * cos_el * cos_el; // V
            output[5] = mono * 1.732051 * sin_az * sin_el * cos_el; // T
            output[6] = mono * 0.5 * (3.0 * sin_el * sin_el - 1.0); // R
            output[7] = mono * 1.732051 * cos_az * sin_el * cos_el; // S
            output[8] = mono * 0.866025 * cos_2az * cos_el * cos_el; // U
        }

        // Apply normalization conversion if needed
        if self.format.normalization == AmbisonicsNormalization::N3d {
            for (i, sample) in output.iter_mut().enumerate() {
                *sample *= sn3d_to_n3d_factor(i);
            }
        }

        output
    }

    /// Encode first-order B-format from direction.
    pub fn encode_foa(&self, mono: f32, azimuth: f32, elevation: f32) -> BFormat {
        BFormat::from_direction(mono, azimuth, elevation)
    }
}

/// Ambisonics decoder for rendering to speaker arrays.
#[derive(Debug, Clone)]
pub struct AmbisonicsDecoder {
    format: AmbisonicsFormat,
    speaker_layout: ChannelLayout,
    /// Decoding matrix: [speaker][ambisonic_channel].
    decode_matrix: Vec<Vec<f32>>,
}

impl AmbisonicsDecoder {
    /// Create a new ambisonics decoder for a speaker layout.
    pub fn new(format: AmbisonicsFormat, speaker_layout: ChannelLayout) -> Result<Self> {
        let decode_matrix = Self::compute_decode_matrix(&format, &speaker_layout)?;

        Ok(Self {
            format,
            speaker_layout,
            decode_matrix,
        })
    }

    /// Create decoder for stereo output.
    pub fn stereo(format: AmbisonicsFormat) -> Result<Self> {
        Self::new(format, ChannelLayout::from_standard(StandardLayout::Stereo))
    }

    /// Create decoder for 5.1 output.
    pub fn surround_51(format: AmbisonicsFormat) -> Result<Self> {
        Self::new(
            format,
            ChannelLayout::from_standard(StandardLayout::Surround51),
        )
    }

    /// Create decoder for 7.1 output.
    pub fn surround_71(format: AmbisonicsFormat) -> Result<Self> {
        Self::new(
            format,
            ChannelLayout::from_standard(StandardLayout::Surround71),
        )
    }

    /// Compute the decoding matrix.
    fn compute_decode_matrix(
        format: &AmbisonicsFormat,
        layout: &ChannelLayout,
    ) -> Result<Vec<Vec<f32>>> {
        let num_speakers = layout.channel_count() as usize;
        let num_channels = format.channel_count() as usize;

        let mut matrix = vec![vec![0.0; num_channels]; num_speakers];

        // Compute decoding coefficients for each speaker
        for (speaker_idx, position) in layout.positions().iter().enumerate() {
            // Skip LFE channels
            if position.is_lfe() {
                continue;
            }

            let (azimuth, elevation) = position.position_degrees();
            let az_rad = azimuth.to_radians();
            let el_rad = elevation.to_radians();

            let cos_el = el_rad.cos();
            let sin_el = el_rad.sin();
            let cos_az = az_rad.cos();
            let sin_az = az_rad.sin();

            // Order 0
            matrix[speaker_idx][0] = 1.0; // W

            if format.order >= 1 && num_channels >= 4 {
                // Order 1
                matrix[speaker_idx][1] = sin_az * cos_el; // Y
                matrix[speaker_idx][2] = sin_el; // Z
                matrix[speaker_idx][3] = cos_az * cos_el; // X
            }

            if format.order >= 2 && num_channels >= 9 {
                // Order 2
                let cos_2az = (2.0 * az_rad).cos();

                matrix[speaker_idx][4] = 1.732051 * sin_az * cos_az * cos_el * cos_el; // V
                matrix[speaker_idx][5] = 1.732051 * sin_az * sin_el * cos_el; // T
                matrix[speaker_idx][6] = 0.5 * (3.0 * sin_el * sin_el - 1.0); // R
                matrix[speaker_idx][7] = 1.732051 * cos_az * sin_el * cos_el; // S
                matrix[speaker_idx][8] = 0.866025 * cos_2az * cos_el * cos_el; // U
            }
        }

        // Normalize for speaker count
        let norm_factor = (num_speakers as f32).sqrt().recip();
        for row in &mut matrix {
            for coef in row {
                *coef *= norm_factor;
            }
        }

        Ok(matrix)
    }

    /// Decode ambisonics to speaker feeds.
    pub fn decode(&self, ambi_channels: &[f32]) -> Result<Vec<f32>> {
        if ambi_channels.len() < self.format.channel_count() as usize {
            return Err(SpatialError::Ambisonics(
                AmbisonicsError::ChannelCountMismatch {
                    count: ambi_channels.len() as u32,
                    order: self.format.order,
                    expected: self.format.channel_count(),
                },
            ));
        }

        let num_speakers = self.speaker_layout.channel_count() as usize;
        let mut output = vec![0.0; num_speakers];

        for (speaker_idx, coeffs) in self.decode_matrix.iter().enumerate() {
            for (ambi_idx, &coef) in coeffs.iter().enumerate() {
                if ambi_idx < ambi_channels.len() {
                    output[speaker_idx] += ambi_channels[ambi_idx] * coef;
                }
            }
        }

        Ok(output)
    }

    /// Decode B-format to speaker feeds.
    pub fn decode_bformat(&self, bformat: &BFormat) -> Result<Vec<f32>> {
        let channels = bformat.to_acn_array();
        self.decode(&channels)
    }

    /// Get the speaker layout.
    pub fn speaker_layout(&self) -> &ChannelLayout {
        &self.speaker_layout
    }

    /// Get the decoding matrix.
    pub fn decode_matrix(&self) -> &[Vec<f32>] {
        &self.decode_matrix
    }
}

/// Convert normalization from SN3D to N3D.
fn sn3d_to_n3d_factor(acn_index: usize) -> f32 {
    // N3D = SN3D * sqrt(2n + 1), where n is the order
    let order = ((acn_index as f32).sqrt()).floor() as u32;
    (2 * order + 1) as f32
}

/// Convert between ambisonics orderings.
pub fn convert_ordering(
    input: &[f32],
    from: AmbisonicsOrdering,
    to: AmbisonicsOrdering,
) -> Vec<f32> {
    if from == to || input.len() < 4 {
        return input.to_vec();
    }

    let mut output = input.to_vec();

    // Convert first-order channels
    match (from, to) {
        (AmbisonicsOrdering::FuMa, AmbisonicsOrdering::Acn) => {
            // FuMa [W, X, Y, Z] -> ACN [W, Y, Z, X]
            output[1] = input[2]; // Y
            output[2] = input[3]; // Z
            output[3] = input[1]; // X
        }
        (AmbisonicsOrdering::Acn, AmbisonicsOrdering::FuMa) => {
            // ACN [W, Y, Z, X] -> FuMa [W, X, Y, Z]
            output[1] = input[3]; // X
            output[2] = input[1]; // Y
            output[3] = input[2]; // Z
        }
        _ => {} // Other conversions not implemented
    }

    output
}

/// Convert between ambisonics normalizations.
pub fn convert_normalization(
    input: &[f32],
    from: AmbisonicsNormalization,
    to: AmbisonicsNormalization,
) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }

    let mut output = input.to_vec();

    match (from, to) {
        (AmbisonicsNormalization::Sn3d, AmbisonicsNormalization::N3d) => {
            for (i, sample) in output.iter_mut().enumerate() {
                *sample *= sn3d_to_n3d_factor(i).sqrt();
            }
        }
        (AmbisonicsNormalization::N3d, AmbisonicsNormalization::Sn3d) => {
            for (i, sample) in output.iter_mut().enumerate() {
                *sample /= sn3d_to_n3d_factor(i).sqrt();
            }
        }
        (AmbisonicsNormalization::FuMa, AmbisonicsNormalization::Sn3d) => {
            // FuMa W is scaled by 1/sqrt(2), convert to SN3D
            if !output.is_empty() {
                output[0] *= 1.414214; // sqrt(2)
            }
        }
        (AmbisonicsNormalization::Sn3d, AmbisonicsNormalization::FuMa) => {
            // SN3D to FuMa: scale W by 1/sqrt(2)
            if !output.is_empty() {
                output[0] *= 0.707107; // 1/sqrt(2)
            }
        }
        _ => {} // Other conversions
    }

    output
}

/// Virtual microphone pattern for ambisonics decoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VirtualMicPattern {
    /// Omnidirectional (just W).
    Omni,
    /// Cardioid (W + directional).
    Cardioid,
    /// Hypercardioid.
    Hypercardioid,
    /// Figure-8 (just directional).
    Figure8,
    /// Super-cardioid.
    SuperCardioid,
}

impl VirtualMicPattern {
    /// Get the W (omni) and directional weights.
    pub fn weights(&self) -> (f32, f32) {
        match self {
            Self::Omni => (1.0, 0.0),
            Self::Cardioid => (0.5, 0.5),
            Self::Hypercardioid => (0.25, 0.75),
            Self::Figure8 => (0.0, 1.0),
            Self::SuperCardioid => (0.366, 0.634),
        }
    }
}

/// Extract a virtual microphone from ambisonics.
pub fn virtual_microphone(
    bformat: &BFormat,
    azimuth: f32,
    elevation: f32,
    pattern: VirtualMicPattern,
) -> f32 {
    let az_rad = azimuth.to_radians();
    let el_rad = elevation.to_radians();

    let cos_el = el_rad.cos();
    let sin_el = el_rad.sin();
    let cos_az = az_rad.cos();
    let sin_az = az_rad.sin();

    // Directional component
    let directional = bformat.x * cos_az * cos_el + bformat.y * sin_az * cos_el + bformat.z * sin_el;

    let (w_weight, dir_weight) = pattern.weights();

    bformat.w * w_weight * 1.414214 + directional * dir_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambisonics_format() {
        let foa = AmbisonicsFormat::ambix(1);
        assert_eq!(foa.channel_count(), 4);
        assert!(foa.is_first_order());
        assert!(!foa.is_higher_order());

        let hoa2 = AmbisonicsFormat::ambix(2);
        assert_eq!(hoa2.channel_count(), 9);
        assert!(!hoa2.is_first_order());
        assert!(hoa2.is_higher_order());

        let hoa3 = AmbisonicsFormat::ambix(3);
        assert_eq!(hoa3.channel_count(), 16);
    }

    #[test]
    fn test_bformat_direction() {
        // Source directly in front
        let front = BFormat::from_direction(1.0, 0.0, 0.0);
        assert!(front.x > 0.9); // Strong X component
        assert!(front.y.abs() < 0.01); // No Y component

        // Source to the left
        let left = BFormat::from_direction(1.0, 90.0, 0.0);
        assert!(left.y > 0.9); // Strong Y component
        assert!(left.x.abs() < 0.01); // No X component

        // Source above
        let above = BFormat::from_direction(1.0, 0.0, 90.0);
        assert!(above.z > 0.9); // Strong Z component
    }

    #[test]
    fn test_bformat_rotation() {
        let front = BFormat::from_direction(1.0, 0.0, 0.0);
        let rotated = front.rotate_z(90.0);

        // After 90 degree rotation, front becomes left
        assert!((rotated.y - front.x).abs() < 0.01);
        assert!((rotated.x + front.y).abs() < 0.01);
    }

    #[test]
    fn test_encoder() {
        let format = AmbisonicsFormat::ambix(1);
        let mut encoder = AmbisonicsEncoder::new(format);

        let encoded = encoder.encode(1.0, 0.0, 0.0);
        assert_eq!(encoded.len(), 4);

        // X should be dominant for front source
        assert!(encoded[3] > 0.9); // X in ACN order
    }

    #[test]
    fn test_decoder_stereo() {
        let format = AmbisonicsFormat::ambix(1);
        let decoder = AmbisonicsDecoder::stereo(format).unwrap();

        // Front-center source should be equal in both speakers
        let bformat = BFormat::from_direction(1.0, 0.0, 0.0);
        let stereo = decoder.decode_bformat(&bformat).unwrap();

        assert_eq!(stereo.len(), 2);
        // Both channels should have similar levels for centered source
        assert!((stereo[0] - stereo[1]).abs() < 0.3);
    }

    #[test]
    fn test_convert_ordering() {
        let fuma = [1.0f32, 2.0, 3.0, 4.0]; // W, X, Y, Z
        let acn = convert_ordering(&fuma, AmbisonicsOrdering::FuMa, AmbisonicsOrdering::Acn);

        // ACN should be W, Y, Z, X
        assert!((acn[0] - 1.0).abs() < 0.001); // W unchanged
        assert!((acn[1] - 3.0).abs() < 0.001); // Y (was index 2)
        assert!((acn[2] - 4.0).abs() < 0.001); // Z (was index 3)
        assert!((acn[3] - 2.0).abs() < 0.001); // X (was index 1)
    }

    #[test]
    fn test_virtual_microphone() {
        let bformat = BFormat::from_direction(1.0, 0.0, 0.0);

        // Cardioid pointing forward should capture the source
        let forward = virtual_microphone(&bformat, 0.0, 0.0, VirtualMicPattern::Cardioid);
        assert!(forward > 0.8);

        // Cardioid pointing backward should reject the source
        let backward = virtual_microphone(&bformat, 180.0, 0.0, VirtualMicPattern::Cardioid);
        assert!(backward.abs() < 0.2);
    }
}
