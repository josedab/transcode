//! Test stream utilities and predefined test cases
//!
//! Provides utilities for working with H.264 test streams and predefined
//! ITU-T conformance stream definitions.

use crate::{H264Level, H264Profile, TestStream};
use std::path::PathBuf;

/// ITU-T H.264/AVC conformance test stream collection
///
/// These are the official ITU-T test vectors for H.264 decoder conformance.
/// The actual streams need to be downloaded from ITU-T or obtained through
/// official channels.
pub struct ItuConformanceStreams;

impl ItuConformanceStreams {
    /// Get all ITU-T baseline profile test streams
    pub fn baseline() -> Vec<TestStream> {
        vec![
            // BA1_Sony_D - Basic I/P slice decoding
            TestStream::builder("BA1_Sony_D")
                .name("BA1_Sony_D - Basic I/P slice")
                .description("Tests basic I and P slice decoding with CAVLC entropy coding")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(176, 144)
                .expected_frames(17)
                .categories(vec!["itu-t", "baseline", "cavlc", "basic"])
                .build(),

            // BA2_Sony_F - Multiple slices per frame
            TestStream::builder("BA2_Sony_F")
                .name("BA2_Sony_F - Multiple slices")
                .description("Tests frames with multiple slices")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(176, 144)
                .expected_frames(300)
                .categories(vec!["itu-t", "baseline", "slices"])
                .build(),

            // BA3_SVA_C - Frame cropping
            TestStream::builder("BA3_SVA_C")
                .name("BA3_SVA_C - Frame cropping")
                .description("Tests frame cropping parameters in SPS")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .categories(vec!["itu-t", "baseline", "cropping"])
                .build(),

            // CVBS3_Sony_C - CAVLC coding
            TestStream::builder("CVBS3_Sony_C")
                .name("CVBS3_Sony_C - CAVLC coding")
                .description("Tests CAVLC entropy coding comprehensively")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .categories(vec!["itu-t", "baseline", "cavlc", "entropy"])
                .build(),

            // SVA_NL1_B - NAL unit parsing
            TestStream::builder("SVA_NL1_B")
                .name("SVA_NL1_B - NAL unit parsing")
                .description("Tests NAL unit parsing and startcode detection")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .categories(vec!["itu-t", "baseline", "nal"])
                .build(),

            // SVA_NL2_E - Emulation prevention
            TestStream::builder("SVA_NL2_E")
                .name("SVA_NL2_E - Emulation prevention")
                .description("Tests emulation prevention byte handling")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .categories(vec!["itu-t", "baseline", "nal", "emulation_prevention"])
                .build(),
        ]
    }

    /// Get all ITU-T main profile test streams
    pub fn main() -> Vec<TestStream> {
        vec![
            // CAPM3_Sony_D - Main profile CABAC
            TestStream::builder("CAPM3_Sony_D")
                .name("CAPM3_Sony_D - Main profile CABAC")
                .description("Tests CABAC entropy coding for Main profile")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .categories(vec!["itu-t", "main", "cabac", "entropy"])
                .build(),

            // CAMP_MOT_MBAFF_L30 - MBAFF coding
            TestStream::builder("CAMP_MOT_MBAFF_L30")
                .name("CAMP_MOT_MBAFF_L30 - MBAFF")
                .description("Tests Macroblock-Adaptive Frame-Field coding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 576)
                .categories(vec!["itu-t", "main", "mbaff", "interlace"])
                .build(),

            // CAMP_MOT_PICAFF_L30 - PAFF coding
            TestStream::builder("CAMP_MOT_PICAFF_L30")
                .name("CAMP_MOT_PICAFF_L30 - PAFF")
                .description("Tests Picture-Adaptive Frame-Field coding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 576)
                .categories(vec!["itu-t", "main", "paff", "interlace"])
                .build(),

            // CANL1_TOSHIBA_G - Long-term reference
            TestStream::builder("CANL1_TOSHIBA_G")
                .name("CANL1_TOSHIBA_G - Long-term reference")
                .description("Tests long-term reference picture management")
                .profile(H264Profile::Main)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .categories(vec!["itu-t", "main", "long_term_ref", "dpb"])
                .build(),

            // CAWP1_TOSHIBA_E - Weighted prediction
            TestStream::builder("CAWP1_TOSHIBA_E")
                .name("CAWP1_TOSHIBA_E - Weighted prediction")
                .description("Tests explicit weighted prediction")
                .profile(H264Profile::Main)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .categories(vec!["itu-t", "main", "weighted_pred"])
                .build(),

            // CABAST3_Sony_E - B-slice coding
            TestStream::builder("CABAST3_Sony_E")
                .name("CABAST3_Sony_E - B-slice coding")
                .description("Tests B-slice bidirectional prediction")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .categories(vec!["itu-t", "main", "b_slice", "inter"])
                .build(),
        ]
    }

    /// Get all ITU-T high profile test streams
    pub fn high() -> Vec<TestStream> {
        vec![
            // CAPH_HP_B - High profile basic
            TestStream::builder("CAPH_HP_B")
                .name("CAPH_HP_B - High Profile basic")
                .description("Tests basic High profile decoding")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .categories(vec!["itu-t", "high", "basic"])
                .build(),

            // CAH1_Sony_B - 8x8 transform
            TestStream::builder("CAH1_Sony_B")
                .name("CAH1_Sony_B - 8x8 transform")
                .description("Tests 8x8 integer transform for High profile")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1280, 720)
                .categories(vec!["itu-t", "high", "8x8", "transform"])
                .build(),

            // CVHP_Toshiba_B - High profile CAVLC
            TestStream::builder("CVHP_Toshiba_B")
                .name("CVHP_Toshiba_B - High profile CAVLC")
                .description("Tests CAVLC with High profile features")
                .profile(H264Profile::High)
                .level(H264Level::Level41)
                .resolution(1920, 1080)
                .categories(vec!["itu-t", "high", "cavlc"])
                .build(),

            // CAHP_TOSHIBA_E - Scaling matrices
            TestStream::builder("CAHP_TOSHIBA_E")
                .name("CAHP_TOSHIBA_E - Scaling matrices")
                .description("Tests custom quantization scaling matrices")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1280, 720)
                .categories(vec!["itu-t", "high", "scaling", "quantization"])
                .build(),

            // CVHPP_TOSHIBA_E - High profile progressive
            TestStream::builder("CVHPP_TOSHIBA_E")
                .name("CVHPP_TOSHIBA_E - Progressive High")
                .description("Tests progressive content in High profile")
                .profile(H264Profile::High)
                .level(H264Level::Level41)
                .resolution(1920, 1080)
                .categories(vec!["itu-t", "high", "progressive"])
                .build(),
        ]
    }

    /// Get all ITU-T test streams
    pub fn all() -> Vec<TestStream> {
        let mut streams = Vec::new();
        streams.extend(Self::baseline());
        streams.extend(Self::main());
        streams.extend(Self::high());
        streams
    }

    /// Get streams by profile
    pub fn by_profile(profile: H264Profile) -> Vec<TestStream> {
        match profile {
            H264Profile::Baseline | H264Profile::ConstrainedBaseline => Self::baseline(),
            H264Profile::Main => Self::main(),
            H264Profile::High | H264Profile::High10 | H264Profile::High422 | H264Profile::High444 => {
                Self::high()
            }
            H264Profile::Extended => Self::main(), // Extended shares many tests with Main
        }
    }
}

/// Local test stream loader
pub struct LocalStreamLoader {
    base_path: PathBuf,
}

impl LocalStreamLoader {
    /// Create a new loader with the given base path
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    /// Load streams from local directory
    pub fn load_streams(&self) -> Vec<TestStream> {
        let mut streams = Vec::new();

        if !self.base_path.exists() {
            return streams;
        }

        // Look for .264 and .h264 files
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "264" || ext == "h264" {
                        if let Some(stem) = path.file_stem() {
                            let id = stem.to_string_lossy().to_string();
                            let stream = TestStream::builder(&id)
                                .name(&id)
                                .description("Local test stream")
                                .profile(H264Profile::Baseline) // Default, would need detection
                                .local_path(path.clone())
                                .category("local")
                                .build();
                            streams.push(stream);
                        }
                    }
                }
            }
        }

        streams
    }

    /// Load a specific stream by ID
    pub fn load_stream(&self, id: &str) -> Option<TestStream> {
        let extensions = ["264", "h264"];

        for ext in extensions {
            let path = self.base_path.join(id).with_extension(ext);
            if path.exists() {
                return Some(
                    TestStream::builder(id)
                        .name(id)
                        .profile(H264Profile::Baseline)
                        .local_path(path)
                        .build(),
                );
            }
        }

        None
    }
}

/// Stream format detection
pub mod detect {
    /// Detect if data is Annex B format (has start codes)
    pub fn is_annex_b(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        // Look for start codes: 0x00000001 or 0x000001
        for i in 0..data.len().saturating_sub(3) {
            if data[i] == 0 && data[i + 1] == 0 {
                if data[i + 2] == 1 {
                    return true;
                }
                if i + 3 < data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                    return true;
                }
            }
        }

        false
    }

    /// Detect if data is AVCC format (length-prefixed NALUs)
    pub fn is_avcc(data: &[u8]) -> bool {
        // AVCC typically starts with configuration record or length-prefixed NALUs
        // This is a simplified check
        if data.len() < 5 {
            return false;
        }

        // Check for AVCC configuration record signature
        // configurationVersion = 1
        if data[0] == 1 {
            // Check for valid profile_idc in byte 1
            let profile_idc = data[1];
            let valid_profiles = [66, 77, 88, 100, 110, 122, 244];
            return valid_profiles.contains(&profile_idc);
        }

        false
    }

    /// Detect profile from SPS data
    pub fn detect_profile(sps_data: &[u8]) -> Option<super::H264Profile> {
        if sps_data.len() < 2 {
            return None;
        }

        // Skip NAL header if present
        let offset = if (sps_data[0] & 0x1f) == 7 { 1 } else { 0 };

        if sps_data.len() <= offset {
            return None;
        }

        let profile_idc = sps_data[offset];

        Some(match profile_idc {
            66 => super::H264Profile::Baseline,
            77 => super::H264Profile::Main,
            88 => super::H264Profile::Extended,
            100 => super::H264Profile::High,
            110 => super::H264Profile::High10,
            122 => super::H264Profile::High422,
            244 => super::H264Profile::High444,
            _ => super::H264Profile::Baseline, // Default fallback
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_itu_streams_baseline() {
        let streams = ItuConformanceStreams::baseline();
        assert!(!streams.is_empty());

        for stream in &streams {
            assert_eq!(stream.profile, H264Profile::Baseline);
            assert!(stream.categories.contains(&"baseline".to_string()));
        }
    }

    #[test]
    fn test_itu_streams_main() {
        let streams = ItuConformanceStreams::main();
        assert!(!streams.is_empty());

        for stream in &streams {
            assert_eq!(stream.profile, H264Profile::Main);
            assert!(stream.categories.contains(&"main".to_string()));
        }
    }

    #[test]
    fn test_itu_streams_high() {
        let streams = ItuConformanceStreams::high();
        assert!(!streams.is_empty());

        for stream in &streams {
            assert_eq!(stream.profile, H264Profile::High);
            assert!(stream.categories.contains(&"high".to_string()));
        }
    }

    #[test]
    fn test_itu_streams_all() {
        let all = ItuConformanceStreams::all();
        let baseline = ItuConformanceStreams::baseline();
        let main = ItuConformanceStreams::main();
        let high = ItuConformanceStreams::high();

        assert_eq!(all.len(), baseline.len() + main.len() + high.len());
    }

    #[test]
    fn test_detect_annex_b() {
        let annex_b = [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a];
        assert!(detect::is_annex_b(&annex_b));

        let not_annex_b = [0x01, 0x42, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00];
        assert!(!detect::is_annex_b(&not_annex_b));
    }

    #[test]
    fn test_detect_profile() {
        // SPS with Baseline profile (profile_idc = 66)
        let sps = [0x67, 66, 0x00, 0x0a];
        assert_eq!(detect::detect_profile(&sps), Some(H264Profile::Baseline));

        // SPS with High profile (profile_idc = 100)
        let sps = [0x67, 100, 0x00, 0x0a];
        assert_eq!(detect::detect_profile(&sps), Some(H264Profile::High));
    }
}
