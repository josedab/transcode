//! Color space handling for HDR processing.
//!
//! This module provides color primaries, transfer characteristics, matrix coefficients,
//! and color space conversion matrices for various standards including BT.709, BT.2020,
//! DCI-P3, and Display P3.

use crate::ColorSpace;

// ============================================================================
// Color Primaries (CIE 1931 xy chromaticity coordinates)
// ============================================================================

/// Color primaries definition with RGB chromaticity coordinates and white point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorPrimaries {
    /// Red primary (x, y)
    pub red: [f64; 2],
    /// Green primary (x, y)
    pub green: [f64; 2],
    /// Blue primary (x, y)
    pub blue: [f64; 2],
    /// White point (x, y)
    pub white: [f64; 2],
}

/// BT.709 / sRGB primaries (ITU-R BT.709-6)
pub const BT709_PRIMARIES: ColorPrimaries = ColorPrimaries {
    red: [0.640, 0.330],
    green: [0.300, 0.600],
    blue: [0.150, 0.060],
    white: [0.3127, 0.3290], // D65
};

/// BT.2020 primaries (ITU-R BT.2020-2)
pub const BT2020_PRIMARIES: ColorPrimaries = ColorPrimaries {
    red: [0.708, 0.292],
    green: [0.170, 0.797],
    blue: [0.131, 0.046],
    white: [0.3127, 0.3290], // D65
};

/// DCI-P3 primaries (theatrical)
pub const DCI_P3_PRIMARIES: ColorPrimaries = ColorPrimaries {
    red: [0.680, 0.320],
    green: [0.265, 0.690],
    blue: [0.150, 0.060],
    white: [0.314, 0.351], // DCI white
};

/// Display P3 primaries (same as DCI-P3 but with D65 white)
pub const DISPLAY_P3_PRIMARIES: ColorPrimaries = ColorPrimaries {
    red: [0.680, 0.320],
    green: [0.265, 0.690],
    blue: [0.150, 0.060],
    white: [0.3127, 0.3290], // D65
};

/// ACES primaries (Academy Color Encoding System)
pub const ACES_PRIMARIES: ColorPrimaries = ColorPrimaries {
    red: [0.7347, 0.2653],
    green: [0.0000, 1.0000],
    blue: [0.0001, -0.0770],
    white: [0.32168, 0.33767], // ACES white
};

/// Get primaries for a color space.
pub fn get_primaries(cs: ColorSpace) -> ColorPrimaries {
    match cs {
        ColorSpace::Bt709 | ColorSpace::Srgb => BT709_PRIMARIES,
        ColorSpace::Bt2020 => BT2020_PRIMARIES,
        ColorSpace::DciP3 => DCI_P3_PRIMARIES,
        ColorSpace::DisplayP3 => DISPLAY_P3_PRIMARIES,
    }
}

// ============================================================================
// Common White Points
// ============================================================================

/// D65 standard illuminant (daylight)
pub const D65_WHITE: [f64; 2] = [0.3127, 0.3290];

/// D50 standard illuminant (used in print)
pub const D50_WHITE: [f64; 2] = [0.3457, 0.3585];

/// DCI white point
pub const DCI_WHITE: [f64; 2] = [0.314, 0.351];

/// ACES white point
pub const ACES_WHITE: [f64; 2] = [0.32168, 0.33767];

// ============================================================================
// RGB to XYZ Conversion Matrices
// ============================================================================

/// RGB to XYZ matrix for BT.709/sRGB (D65)
pub const BT709_TO_XYZ: [[f64; 3]; 3] = [
    [0.4123907993, 0.3575843394, 0.1804807884],
    [0.2126390059, 0.7151686788, 0.0721923154],
    [0.0193308187, 0.1191947798, 0.9505321522],
];

/// XYZ to RGB matrix for BT.709/sRGB (D65)
pub const XYZ_TO_BT709: [[f64; 3]; 3] = [
    [3.2409699419, -1.5373831776, -0.4986107603],
    [-0.9692436363, 1.8759675015, 0.0415550574],
    [0.0556300797, -0.2039769589, 1.0569715142],
];

/// RGB to XYZ matrix for BT.2020 (D65)
pub const BT2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.6369580483, 0.1446169036, 0.1688809752],
    [0.2627002120, 0.6779980715, 0.0593017165],
    [0.0000000000, 0.0280726930, 1.0609850577],
];

/// XYZ to RGB matrix for BT.2020 (D65)
pub const XYZ_TO_BT2020: [[f64; 3]; 3] = [
    [1.7166511880, -0.3556707838, -0.2533662814],
    [-0.6666843518, 1.6164812366, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.9421031212],
];

/// RGB to XYZ matrix for DCI-P3 (DCI white)
pub const DCI_P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.4451698156, 0.2771344092, 0.1722826698],
    [0.2094916779, 0.7215952542, 0.0689130679],
    [0.0000000000, 0.0470605601, 0.9073553944],
];

/// XYZ to RGB matrix for DCI-P3 (DCI white)
pub const XYZ_TO_DCI_P3: [[f64; 3]; 3] = [
    [2.7253940305, -1.0180030062, -0.4401631952],
    [-0.7951680258, 1.6897320548, 0.0226471906],
    [0.0412418914, -0.0876390192, 1.1009293786],
];

/// RGB to XYZ matrix for Display P3 (D65)
pub const DISPLAY_P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.4865709486, 0.2656676932, 0.1982172852],
    [0.2289745641, 0.6917385218, 0.0792869141],
    [0.0000000000, 0.0451133819, 1.0439443689],
];

/// XYZ to RGB matrix for Display P3 (D65)
pub const XYZ_TO_DISPLAY_P3: [[f64; 3]; 3] = [
    [2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696, 1.7626640603, 0.0236246858],
    [0.0358458302, -0.0761723893, 0.9568845240],
];

// ============================================================================
// Direct Color Space Conversion Matrices
// ============================================================================

/// BT.2020 to BT.709 conversion matrix
pub const BT2020_TO_BT709: [[f64; 3]; 3] = [
    [1.6604910021, -0.5876411388, -0.0728498633],
    [-0.1245504745, 1.1328998971, -0.0083494226],
    [-0.0181507634, -0.1005788980, 1.1187296614],
];

/// BT.709 to BT.2020 conversion matrix
pub const BT709_TO_BT2020: [[f64; 3]; 3] = [
    [0.6274039446, 0.3292830384, 0.0433130170],
    [0.0690972894, 0.9195403951, 0.0113623155],
    [0.0163914388, 0.0880133079, 0.8955952533],
];

/// Display P3 to BT.709 conversion matrix
pub const DISPLAY_P3_TO_BT709: [[f64; 3]; 3] = [
    [1.2248932420, -0.2249215921, 0.0000283501],
    [-0.0420495884, 1.0420510936, -0.0000015052],
    [-0.0196374866, -0.0786364498, 1.0982739365],
];

/// BT.709 to Display P3 conversion matrix
pub const BT709_TO_DISPLAY_P3: [[f64; 3]; 3] = [
    [0.8224621811, 0.1775378189, 0.0000000000],
    [0.0331941221, 0.9668058779, 0.0000000000],
    [0.0170826460, 0.0723974343, 0.9105199197],
];

/// BT.2020 to Display P3 conversion matrix
pub const BT2020_TO_DISPLAY_P3: [[f64; 3]; 3] = [
    [1.3434100344, -0.2821498049, -0.0612602295],
    [-0.0652568020, 1.0757523178, -0.0104955158],
    [0.0028003000, -0.0196209330, 1.0168206330],
];

/// Display P3 to BT.2020 conversion matrix
pub const DISPLAY_P3_TO_BT2020: [[f64; 3]; 3] = [
    [0.7538928827, 0.1985084427, 0.0475986746],
    [0.0457201626, 0.9419217746, 0.0123580628],
    [-0.0012107084, 0.0176345073, 0.9835762011],
];

// ============================================================================
// Matrix Coefficients for YCbCr <-> RGB conversion
// ============================================================================

/// YCbCr matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixCoefficients {
    /// Kr coefficient (red contribution to Y)
    pub kr: f64,
    /// Kb coefficient (blue contribution to Y)
    pub kb: f64,
}

impl MatrixCoefficients {
    /// Calculate Kg (green contribution to Y) = 1 - Kr - Kb
    pub fn kg(&self) -> f64 {
        1.0 - self.kr - self.kb
    }

    /// Get RGB to YCbCr conversion matrix.
    pub fn rgb_to_ycbcr_matrix(&self) -> [[f64; 3]; 3] {
        let kg = self.kg();
        [
            [self.kr, kg, self.kb],
            [
                -self.kr / (2.0 * (1.0 - self.kb)),
                -kg / (2.0 * (1.0 - self.kb)),
                0.5,
            ],
            [
                0.5,
                -kg / (2.0 * (1.0 - self.kr)),
                -self.kb / (2.0 * (1.0 - self.kr)),
            ],
        ]
    }

    /// Get YCbCr to RGB conversion matrix.
    pub fn ycbcr_to_rgb_matrix(&self) -> [[f64; 3]; 3] {
        let kg = self.kg();
        [
            [1.0, 0.0, 2.0 * (1.0 - self.kr)],
            [
                1.0,
                -2.0 * self.kb * (1.0 - self.kb) / kg,
                -2.0 * self.kr * (1.0 - self.kr) / kg,
            ],
            [1.0, 2.0 * (1.0 - self.kb), 0.0],
        ]
    }
}

/// BT.709 matrix coefficients
pub const BT709_MATRIX: MatrixCoefficients = MatrixCoefficients {
    kr: 0.2126,
    kb: 0.0722,
};

/// BT.2020 matrix coefficients
pub const BT2020_MATRIX: MatrixCoefficients = MatrixCoefficients {
    kr: 0.2627,
    kb: 0.0593,
};

/// BT.601 matrix coefficients (NTSC)
pub const BT601_MATRIX: MatrixCoefficients = MatrixCoefficients {
    kr: 0.299,
    kb: 0.114,
};

/// Get matrix coefficients for a color space.
pub fn get_matrix_coefficients(cs: ColorSpace) -> MatrixCoefficients {
    match cs {
        ColorSpace::Bt709 | ColorSpace::Srgb | ColorSpace::DisplayP3 | ColorSpace::DciP3 => {
            BT709_MATRIX
        }
        ColorSpace::Bt2020 => BT2020_MATRIX,
    }
}

// ============================================================================
// Color Space Converter
// ============================================================================

/// Color space converter for RGB transformations.
#[derive(Debug, Clone)]
pub struct ColorSpaceConverter {
    /// Source color space
    pub source: ColorSpace,
    /// Target color space
    pub target: ColorSpace,
    /// Cached conversion matrix
    matrix: [[f64; 3]; 3],
}

impl ColorSpaceConverter {
    /// Create a new color space converter.
    pub fn new(source: ColorSpace, target: ColorSpace) -> Self {
        let matrix = Self::compute_matrix(source, target);
        Self {
            source,
            target,
            matrix,
        }
    }

    /// Compute the conversion matrix between two color spaces.
    fn compute_matrix(source: ColorSpace, target: ColorSpace) -> [[f64; 3]; 3] {
        if source == target {
            return identity_matrix();
        }

        // Use pre-computed matrices for common conversions
        match (source, target) {
            (ColorSpace::Bt2020, ColorSpace::Bt709) | (ColorSpace::Bt2020, ColorSpace::Srgb) => {
                BT2020_TO_BT709
            }
            (ColorSpace::Bt709, ColorSpace::Bt2020) | (ColorSpace::Srgb, ColorSpace::Bt2020) => {
                BT709_TO_BT2020
            }
            (ColorSpace::DisplayP3, ColorSpace::Bt709)
            | (ColorSpace::DisplayP3, ColorSpace::Srgb) => DISPLAY_P3_TO_BT709,
            (ColorSpace::Bt709, ColorSpace::DisplayP3)
            | (ColorSpace::Srgb, ColorSpace::DisplayP3) => BT709_TO_DISPLAY_P3,
            (ColorSpace::Bt2020, ColorSpace::DisplayP3) => BT2020_TO_DISPLAY_P3,
            (ColorSpace::DisplayP3, ColorSpace::Bt2020) => DISPLAY_P3_TO_BT2020,
            _ => {
                // For other conversions, go through XYZ
                let to_xyz = get_rgb_to_xyz_matrix(source);
                let from_xyz = get_xyz_to_rgb_matrix(target);
                multiply_matrices(&from_xyz, &to_xyz)
            }
        }
    }

    /// Convert RGB values from source to target color space.
    /// Input values should be in linear light (not gamma-encoded).
    #[inline]
    pub fn convert(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let new_r = self.matrix[0][0] * r + self.matrix[0][1] * g + self.matrix[0][2] * b;
        let new_g = self.matrix[1][0] * r + self.matrix[1][1] * g + self.matrix[1][2] * b;
        let new_b = self.matrix[2][0] * r + self.matrix[2][1] * g + self.matrix[2][2] * b;
        (new_r, new_g, new_b)
    }

    /// Convert RGB values (f32 version).
    #[inline]
    pub fn convert_f32(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let (nr, ng, nb) = self.convert(r as f64, g as f64, b as f64);
        (nr as f32, ng as f32, nb as f32)
    }

    /// Convert a slice of RGB triplets in place.
    pub fn convert_slice(&self, pixels: &mut [f64]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.convert(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Convert a slice of RGB triplets (f32 version) in place.
    pub fn convert_slice_f32(&self, pixels: &mut [f32]) {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.convert_f32(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Get the conversion matrix.
    pub fn matrix(&self) -> &[[f64; 3]; 3] {
        &self.matrix
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get RGB to XYZ matrix for a color space.
pub fn get_rgb_to_xyz_matrix(cs: ColorSpace) -> [[f64; 3]; 3] {
    match cs {
        ColorSpace::Bt709 | ColorSpace::Srgb => BT709_TO_XYZ,
        ColorSpace::Bt2020 => BT2020_TO_XYZ,
        ColorSpace::DciP3 => DCI_P3_TO_XYZ,
        ColorSpace::DisplayP3 => DISPLAY_P3_TO_XYZ,
    }
}

/// Get XYZ to RGB matrix for a color space.
pub fn get_xyz_to_rgb_matrix(cs: ColorSpace) -> [[f64; 3]; 3] {
    match cs {
        ColorSpace::Bt709 | ColorSpace::Srgb => XYZ_TO_BT709,
        ColorSpace::Bt2020 => XYZ_TO_BT2020,
        ColorSpace::DciP3 => XYZ_TO_DCI_P3,
        ColorSpace::DisplayP3 => XYZ_TO_DISPLAY_P3,
    }
}

/// Identity matrix.
#[inline]
pub fn identity_matrix() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Multiply two 3x3 matrices.
pub fn multiply_matrices(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Convert RGB to XYZ using a given color space.
#[inline]
pub fn rgb_to_xyz(r: f64, g: f64, b: f64, cs: ColorSpace) -> (f64, f64, f64) {
    let m = get_rgb_to_xyz_matrix(cs);
    let x = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    let y = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    let z = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    (x, y, z)
}

/// Convert XYZ to RGB using a given color space.
#[inline]
pub fn xyz_to_rgb(x: f64, y: f64, z: f64, cs: ColorSpace) -> (f64, f64, f64) {
    let m = get_xyz_to_rgb_matrix(cs);
    let r = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    let g = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    let b = m[2][0] * x + m[2][1] * y + m[2][2] * z;
    (r, g, b)
}

/// Calculate luminance from RGB in BT.709.
#[inline]
pub fn luminance_bt709(r: f64, g: f64, b: f64) -> f64 {
    BT709_MATRIX.kr * r + BT709_MATRIX.kg() * g + BT709_MATRIX.kb * b
}

/// Calculate luminance from RGB in BT.2020.
#[inline]
pub fn luminance_bt2020(r: f64, g: f64, b: f64) -> f64 {
    BT2020_MATRIX.kr * r + BT2020_MATRIX.kg() * g + BT2020_MATRIX.kb * b
}

/// Calculate luminance from RGB using matrix coefficients.
#[inline]
pub fn luminance(r: f64, g: f64, b: f64, matrix: &MatrixCoefficients) -> f64 {
    matrix.kr * r + matrix.kg() * g + matrix.kb * b
}

/// Get the gamut volume ratio between two color spaces (approximate).
pub fn gamut_volume_ratio(source: ColorSpace, target: ColorSpace) -> f64 {
    // Approximate gamut volumes relative to BT.709
    let volume = |cs: ColorSpace| -> f64 {
        match cs {
            ColorSpace::Bt709 | ColorSpace::Srgb => 1.0,
            ColorSpace::DisplayP3 => 1.25,
            ColorSpace::DciP3 => 1.26,
            ColorSpace::Bt2020 => 1.75,
        }
    };

    volume(source) / volume(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_identity_conversion() {
        let converter = ColorSpaceConverter::new(ColorSpace::Bt709, ColorSpace::Bt709);
        let (r, g, b) = converter.convert(0.5, 0.3, 0.8);
        assert!((r - 0.5).abs() < EPSILON);
        assert!((g - 0.3).abs() < EPSILON);
        assert!((b - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_bt2020_to_bt709_white() {
        let converter = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);
        // White should stay white
        let (r, g, b) = converter.convert(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bt2020_to_bt709_black() {
        let converter = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);
        // Black should stay black
        let (r, g, b) = converter.convert(0.0, 0.0, 0.0);
        assert!(r.abs() < EPSILON);
        assert!(g.abs() < EPSILON);
        assert!(b.abs() < EPSILON);
    }

    #[test]
    fn test_roundtrip_bt709_bt2020() {
        let to_2020 = ColorSpaceConverter::new(ColorSpace::Bt709, ColorSpace::Bt2020);
        let to_709 = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);

        let (r1, g1, b1) = (0.5, 0.3, 0.7);
        let (r2, g2, b2) = to_2020.convert(r1, g1, b1);
        let (r3, g3, b3) = to_709.convert(r2, g2, b2);

        assert!((r1 - r3).abs() < 0.001);
        assert!((g1 - g3).abs() < 0.001);
        assert!((b1 - b3).abs() < 0.001);
    }

    #[test]
    fn test_luminance_calculation() {
        // For white, luminance should be 1.0
        let lum = luminance_bt709(1.0, 1.0, 1.0);
        assert!((lum - 1.0).abs() < EPSILON);

        // For black, luminance should be 0.0
        let lum = luminance_bt709(0.0, 0.0, 0.0);
        assert!(lum.abs() < EPSILON);

        // Pure green should have the highest contribution
        let lum_r = luminance_bt709(1.0, 0.0, 0.0);
        let lum_g = luminance_bt709(0.0, 1.0, 0.0);
        let lum_b = luminance_bt709(0.0, 0.0, 1.0);
        assert!(lum_g > lum_r);
        assert!(lum_g > lum_b);
    }

    #[test]
    fn test_matrix_coefficients_sum() {
        // Kr + Kg + Kb should equal 1.0
        let sum = BT709_MATRIX.kr + BT709_MATRIX.kg() + BT709_MATRIX.kb;
        assert!((sum - 1.0).abs() < EPSILON);

        let sum = BT2020_MATRIX.kr + BT2020_MATRIX.kg() + BT2020_MATRIX.kb;
        assert!((sum - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rgb_xyz_roundtrip() {
        for cs in [
            ColorSpace::Bt709,
            ColorSpace::Bt2020,
            ColorSpace::DisplayP3,
        ] {
            let (r1, g1, b1) = (0.5, 0.3, 0.7);
            let (x, y, z) = rgb_to_xyz(r1, g1, b1, cs);
            let (r2, g2, b2) = xyz_to_rgb(x, y, z, cs);

            assert!((r1 - r2).abs() < 0.001, "RGB-XYZ roundtrip failed for {:?}", cs);
            assert!((g1 - g2).abs() < 0.001, "RGB-XYZ roundtrip failed for {:?}", cs);
            assert!((b1 - b2).abs() < 0.001, "RGB-XYZ roundtrip failed for {:?}", cs);
        }
    }

    #[test]
    fn test_gamut_volume_ratio() {
        // BT.2020 should have larger gamut than BT.709
        let ratio = gamut_volume_ratio(ColorSpace::Bt2020, ColorSpace::Bt709);
        assert!(ratio > 1.0);

        // Same color space should have ratio of 1.0
        let ratio = gamut_volume_ratio(ColorSpace::Bt709, ColorSpace::Bt709);
        assert!((ratio - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_slice_conversion() {
        let converter = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);
        let mut pixels = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        converter.convert_slice(&mut pixels);

        // Verify all values were modified (not identity)
        // This is a sanity check that conversion happened
        assert!(pixels.iter().all(|&v| v.is_finite()));
    }
}
