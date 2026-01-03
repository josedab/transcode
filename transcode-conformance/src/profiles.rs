//! Profile-specific conformance test definitions
//!
//! Defines test suites for Baseline, Main, and High profile conformance.

use crate::{ConformanceConfig, H264Level, H264Profile, TestStream};

/// Baseline profile conformance test suite
pub struct BaselineProfileTests {
    streams: Vec<TestStream>,
}

impl Default for BaselineProfileTests {
    fn default() -> Self {
        Self::new()
    }
}

impl BaselineProfileTests {
    /// Create new baseline profile test suite
    pub fn new() -> Self {
        Self {
            streams: Self::define_streams(),
        }
    }

    fn define_streams() -> Vec<TestStream> {
        vec![
            // Basic I-frame only
            TestStream::builder("BP_I_ONLY_01")
                .name("Baseline I-frame Only")
                .description("Tests basic I-frame decoding without inter prediction")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level1)
                .resolution(176, 144)
                .expected_frames(10)
                .categories(vec!["baseline", "intra", "basic"])
                .build(),

            // I and P frames
            TestStream::builder("BP_IP_01")
                .name("Baseline I and P frames")
                .description("Tests I and P frame decoding with forward prediction")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(30)
                .categories(vec!["baseline", "inter", "p_slice"])
                .build(),

            // Multiple slices
            TestStream::builder("BP_MULTI_SLICE_01")
                .name("Baseline Multiple Slices")
                .description("Tests frame with multiple slices")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(10)
                .categories(vec!["baseline", "slices"])
                .build(),

            // CAVLC entropy coding
            TestStream::builder("BP_CAVLC_01")
                .name("Baseline CAVLC")
                .description("Tests CAVLC entropy decoding")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(50)
                .categories(vec!["baseline", "cavlc", "entropy"])
                .build(),

            // Deblocking filter
            TestStream::builder("BP_DEBLOCK_01")
                .name("Baseline Deblocking Filter")
                .description("Tests loop filter operation")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(30)
                .categories(vec!["baseline", "deblock", "filter"])
                .build(),

            // Multiple reference frames
            TestStream::builder("BP_MULTI_REF_01")
                .name("Baseline Multiple References")
                .description("Tests multiple reference frame prediction")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(60)
                .categories(vec!["baseline", "multi_ref", "prediction"])
                .build(),

            // FMO (Flexible Macroblock Ordering) - if supported
            TestStream::builder("BP_FMO_01")
                .name("Baseline FMO")
                .description("Tests Flexible Macroblock Ordering")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(10)
                .categories(vec!["baseline", "fmo", "advanced"])
                .build(),

            // ASO (Arbitrary Slice Ordering) - if supported
            TestStream::builder("BP_ASO_01")
                .name("Baseline ASO")
                .description("Tests Arbitrary Slice Ordering")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(10)
                .categories(vec!["baseline", "aso", "advanced"])
                .build(),
        ]
    }

    /// Get all test streams
    pub fn streams(&self) -> &[TestStream] {
        &self.streams
    }

    /// Get streams by category
    pub fn by_category(&self, category: &str) -> Vec<&TestStream> {
        self.streams
            .iter()
            .filter(|s| s.categories.contains(&category.to_string()))
            .collect()
    }
}

/// Main profile conformance test suite
pub struct MainProfileTests {
    streams: Vec<TestStream>,
}

impl Default for MainProfileTests {
    fn default() -> Self {
        Self::new()
    }
}

impl MainProfileTests {
    /// Create new main profile test suite
    pub fn new() -> Self {
        Self {
            streams: Self::define_streams(),
        }
    }

    fn define_streams() -> Vec<TestStream> {
        vec![
            // B-frames basic
            TestStream::builder("MP_B_FRAME_01")
                .name("Main Profile B-frames")
                .description("Tests B-frame bidirectional prediction")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(60)
                .categories(vec!["main", "b_slice", "inter"])
                .build(),

            // CABAC entropy coding
            TestStream::builder("MP_CABAC_01")
                .name("Main Profile CABAC")
                .description("Tests CABAC entropy decoding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(50)
                .categories(vec!["main", "cabac", "entropy"])
                .build(),

            // Weighted prediction
            TestStream::builder("MP_WEIGHTED_PRED_01")
                .name("Main Profile Weighted Prediction")
                .description("Tests weighted prediction for P slices")
                .profile(H264Profile::Main)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .expected_frames(30)
                .categories(vec!["main", "weighted_pred", "prediction"])
                .build(),

            // Bi-weighted prediction
            TestStream::builder("MP_BIWEIGHTED_PRED_01")
                .name("Main Profile Bi-Weighted Prediction")
                .description("Tests weighted prediction for B slices")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(60)
                .categories(vec!["main", "weighted_pred", "b_slice"])
                .build(),

            // MBAFF (Macroblock-Adaptive Frame-Field)
            TestStream::builder("MP_MBAFF_01")
                .name("Main Profile MBAFF")
                .description("Tests Macroblock-Adaptive Frame-Field coding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 576)
                .expected_frames(50)
                .categories(vec!["main", "mbaff", "interlace"])
                .build(),

            // PAFF (Picture Adaptive Frame-Field)
            TestStream::builder("MP_PAFF_01")
                .name("Main Profile PAFF")
                .description("Tests Picture Adaptive Frame-Field coding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 576)
                .expected_frames(50)
                .categories(vec!["main", "paff", "interlace"])
                .build(),

            // Direct mode
            TestStream::builder("MP_DIRECT_01")
                .name("Main Profile Direct Mode")
                .description("Tests direct mode prediction in B slices")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(60)
                .categories(vec!["main", "direct", "b_slice"])
                .build(),

            // Long-term reference
            TestStream::builder("MP_LONG_TERM_REF_01")
                .name("Main Profile Long-Term Reference")
                .description("Tests long-term reference picture management")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .expected_frames(100)
                .categories(vec!["main", "long_term_ref", "dpb"])
                .build(),
        ]
    }

    /// Get all test streams
    pub fn streams(&self) -> &[TestStream] {
        &self.streams
    }

    /// Get streams by category
    pub fn by_category(&self, category: &str) -> Vec<&TestStream> {
        self.streams
            .iter()
            .filter(|s| s.categories.contains(&category.to_string()))
            .collect()
    }
}

/// High profile conformance test suite
pub struct HighProfileTests {
    streams: Vec<TestStream>,
}

impl Default for HighProfileTests {
    fn default() -> Self {
        Self::new()
    }
}

impl HighProfileTests {
    /// Create new high profile test suite
    pub fn new() -> Self {
        Self {
            streams: Self::define_streams(),
        }
    }

    fn define_streams() -> Vec<TestStream> {
        vec![
            // 8x8 transform
            TestStream::builder("HP_8X8_TRANSFORM_01")
                .name("High Profile 8x8 Transform")
                .description("Tests 8x8 integer transform")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .expected_frames(30)
                .categories(vec!["high", "8x8", "transform"])
                .build(),

            // 8x8 intra prediction
            TestStream::builder("HP_8X8_INTRA_01")
                .name("High Profile 8x8 Intra Prediction")
                .description("Tests 8x8 intra prediction modes")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .expected_frames(30)
                .categories(vec!["high", "8x8", "intra"])
                .build(),

            // Scaling matrices
            TestStream::builder("HP_SCALING_01")
                .name("High Profile Scaling Matrices")
                .description("Tests custom quantization scaling matrices")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1280, 720)
                .expected_frames(30)
                .categories(vec!["high", "scaling", "quantization"])
                .build(),

            // 4:2:0 at HD resolution
            TestStream::builder("HP_HD_01")
                .name("High Profile HD 1080p")
                .description("Tests HD 1080p decoding")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .expected_frames(60)
                .categories(vec!["high", "hd", "1080p"])
                .build(),

            // Level 4.1 (Blu-ray)
            TestStream::builder("HP_BLURAY_01")
                .name("High Profile Blu-ray Level")
                .description("Tests Level 4.1 Blu-ray compliant stream")
                .profile(H264Profile::High)
                .level(H264Level::Level41)
                .resolution(1920, 1080)
                .expected_frames(60)
                .categories(vec!["high", "bluray", "level41"])
                .build(),

            // Level 5.1 (4K reference)
            TestStream::builder("HP_4K_REF_01")
                .name("High Profile 4K Reference")
                .description("Tests Level 5.1 4K reference decoding")
                .profile(H264Profile::High)
                .level(H264Level::Level51)
                .resolution(3840, 2160)
                .expected_frames(30)
                .categories(vec!["high", "4k", "level51"])
                .build(),

            // CABAC with High profile features
            TestStream::builder("HP_CABAC_FULL_01")
                .name("High Profile CABAC Full")
                .description("Tests CABAC with all High profile features")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .expected_frames(60)
                .categories(vec!["high", "cabac", "full"])
                .build(),

            // Monochrome (High profile allows it)
            TestStream::builder("HP_MONO_01")
                .name("High Profile Monochrome")
                .description("Tests monochrome 4:0:0 format")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .expected_frames(30)
                .categories(vec!["high", "monochrome", "chroma_format"])
                .build(),
        ]
    }

    /// Get all test streams
    pub fn streams(&self) -> &[TestStream] {
        &self.streams
    }

    /// Get streams by category
    pub fn by_category(&self, category: &str) -> Vec<&TestStream> {
        self.streams
            .iter()
            .filter(|s| s.categories.contains(&category.to_string()))
            .collect()
    }
}

/// Get all conformance tests for a given profile
pub fn get_tests_for_profile(profile: H264Profile) -> Vec<TestStream> {
    match profile {
        H264Profile::Baseline | H264Profile::ConstrainedBaseline => {
            BaselineProfileTests::new().streams.clone()
        }
        H264Profile::Main => MainProfileTests::new().streams.clone(),
        H264Profile::High
        | H264Profile::High10
        | H264Profile::High422
        | H264Profile::High444 => HighProfileTests::new().streams.clone(),
        H264Profile::Extended => {
            // Extended profile shares many tests with Main
            MainProfileTests::new().streams.clone()
        }
    }
}

/// Get all conformance tests based on configuration
pub fn get_tests_for_config(config: &ConformanceConfig) -> Vec<TestStream> {
    let mut streams = Vec::new();

    for profile in &config.profiles {
        streams.extend(get_tests_for_profile(*profile));
    }

    streams
}

/// Conformance test categories
pub mod categories {
    pub const BASIC: &str = "basic";
    pub const INTRA: &str = "intra";
    pub const INTER: &str = "inter";
    pub const B_SLICE: &str = "b_slice";
    pub const P_SLICE: &str = "p_slice";
    pub const CABAC: &str = "cabac";
    pub const CAVLC: &str = "cavlc";
    pub const ENTROPY: &str = "entropy";
    pub const DEBLOCK: &str = "deblock";
    pub const FILTER: &str = "filter";
    pub const TRANSFORM: &str = "transform";
    pub const SLICES: &str = "slices";
    pub const WEIGHTED_PRED: &str = "weighted_pred";
    pub const MBAFF: &str = "mbaff";
    pub const PAFF: &str = "paff";
    pub const INTERLACE: &str = "interlace";
    pub const HD: &str = "hd";
    pub const FMO: &str = "fmo";
    pub const ASO: &str = "aso";
    pub const SCALING: &str = "scaling";
    pub const QUANTIZATION: &str = "quantization";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_streams() {
        let tests = BaselineProfileTests::new();
        assert!(!tests.streams().is_empty());

        for stream in tests.streams() {
            assert_eq!(stream.profile, H264Profile::Baseline);
        }
    }

    #[test]
    fn test_main_streams() {
        let tests = MainProfileTests::new();
        assert!(!tests.streams().is_empty());

        for stream in tests.streams() {
            assert_eq!(stream.profile, H264Profile::Main);
        }
    }

    #[test]
    fn test_high_streams() {
        let tests = HighProfileTests::new();
        assert!(!tests.streams().is_empty());

        for stream in tests.streams() {
            assert_eq!(stream.profile, H264Profile::High);
        }
    }

    #[test]
    fn test_get_tests_for_profile() {
        let baseline = get_tests_for_profile(H264Profile::Baseline);
        assert!(!baseline.is_empty());

        let main = get_tests_for_profile(H264Profile::Main);
        assert!(!main.is_empty());

        let high = get_tests_for_profile(H264Profile::High);
        assert!(!high.is_empty());
    }

    #[test]
    fn test_category_filter() {
        let tests = MainProfileTests::new();
        let cabac = tests.by_category(categories::CABAC);
        assert!(!cabac.is_empty());

        for stream in cabac {
            assert!(stream.categories.contains(&categories::CABAC.to_string()));
        }
    }
}
