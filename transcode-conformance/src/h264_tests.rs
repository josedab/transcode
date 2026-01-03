//! H.264 Conformance Test Suite
//!
//! This module contains conformance tests for H.264/AVC decoder and encoder
//! implementations. Tests are organized by profile and feature.
//!
//! ## Running Tests
//!
//! Tests requiring external files are marked with `#[ignore]`. To run them:
//!
//! ```bash
//! # Run all unit tests
//! cargo test -p transcode-conformance
//!
//! # Run ignored tests (requires external files)
//! cargo test -p transcode-conformance --ignored
//!
//! # Run with download support
//! cargo test -p transcode-conformance --features download --ignored
//! ```

use transcode_conformance::{
    cache::StreamCache,
    checksum::{self, ChecksumAlgorithm, ChecksumCalculator},
    download::ConformanceStreamIndex,
    profiles::{BaselineProfileTests, HighProfileTests, MainProfileTests},
    report::{ReportFormat, ReportGenerator, ReportSummary},
    runner::{ConformanceRunner, TestFilter},
    stream::{detect, ItuConformanceStreams, LocalStreamLoader},
    validation::{BitstreamValidator, ValidationErrorCode},
    ConformanceConfig, H264Level, H264Profile, TestResult, TestStatus, TestStream,
};
use std::path::PathBuf;

// =============================================================================
// Test Infrastructure Tests
// =============================================================================

mod infrastructure {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ConformanceConfig::default();
        assert!(!config.profiles.is_empty());
        assert!(!config.allow_download);
    }

    #[test]
    fn test_baseline_config() {
        let config = ConformanceConfig::baseline_only();
        assert_eq!(config.profiles.len(), 1);
        assert_eq!(config.profiles[0], H264Profile::Baseline);
    }

    #[test]
    fn test_main_config() {
        let config = ConformanceConfig::main_only();
        assert_eq!(config.profiles.len(), 1);
        assert_eq!(config.profiles[0], H264Profile::Main);
    }

    #[test]
    fn test_high_config() {
        let config = ConformanceConfig::high_only();
        assert_eq!(config.profiles.len(), 1);
        assert_eq!(config.profiles[0], H264Profile::High);
    }

    #[test]
    fn test_stream_builder() {
        let stream = TestStream::builder("test_stream_001")
            .name("Test Stream 001")
            .description("A test stream for unit testing")
            .profile(H264Profile::Baseline)
            .level(H264Level::Level21)
            .resolution(352, 288)
            .expected_frames(100)
            .frame_rate(30, 1)
            .categories(vec!["test", "basic"])
            .build();

        assert_eq!(stream.id, "test_stream_001");
        assert_eq!(stream.profile, H264Profile::Baseline);
        assert_eq!(stream.resolution, Some((352, 288)));
        assert_eq!(stream.expected_frame_count, Some(100));
    }

    #[test]
    fn test_cache_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache = StreamCache::new(temp_dir.path().to_path_buf());
        assert!(cache.is_ok());
    }

    #[test]
    fn test_runner_creation() {
        let config = ConformanceConfig::default();
        let runner = ConformanceRunner::new(config);
        assert!(runner.is_ok());
    }
}

// =============================================================================
// Bitstream Validation Tests
// =============================================================================

mod bitstream_validation {
    use super::*;

    #[test]
    fn test_empty_bitstream_validation() {
        let validator = BitstreamValidator::new();
        let result = validator.validate(&[]).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_no_start_code_validation() {
        let validator = BitstreamValidator::new();
        let data = vec![0x67, 0x42, 0x00, 0x0a, 0x00, 0x00];
        let result = validator.validate(&data).unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_valid_sps_detection() {
        let validator = BitstreamValidator::new();
        // Start code + SPS NAL (type 7) + minimal SPS data
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x67,                   // NAL header: SPS (type 7)
            0x42,                   // profile_idc = 66 (Baseline)
            0x00,                   // constraint_set_flags
            0x0a,                   // level_idc = 10 (Level 1)
            0x00, 0x00, 0x00, 0x00, // padding
        ];

        let result = validator.validate(&data).unwrap();
        assert_eq!(result.nal_stats.sps_count, 1);
        assert!(result.sps_info.is_some());

        let sps = result.sps_info.unwrap();
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 10);
    }

    #[test]
    fn test_valid_pps_detection() {
        let validator = BitstreamValidator::new();
        // Start code + PPS NAL (type 8) + minimal PPS data
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x68,                   // NAL header: PPS (type 8)
            0x00, 0x00,             // minimal PPS data
        ];

        let result = validator.validate(&data).unwrap();
        assert_eq!(result.nal_stats.pps_count, 1);
    }

    #[test]
    fn test_multiple_nal_units() {
        let validator = BitstreamValidator::new();
        // SPS + PPS + IDR slice
        let data = vec![
            // SPS
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0x00,
            // PPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0x00, 0x00,
            // IDR slice (type 5)
            0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00, 0x00,
        ];

        let result = validator.validate(&data).unwrap();
        assert_eq!(result.nal_stats.total_count, 3);
        assert_eq!(result.nal_stats.sps_count, 1);
        assert_eq!(result.nal_stats.pps_count, 1);
        assert_eq!(result.nal_stats.idr_count, 1);
    }

    #[test]
    fn test_forbidden_bit_detection() {
        let validator = BitstreamValidator::new();
        // NAL with forbidden bit set (bit 7 = 1)
        let data = vec![
            0x00, 0x00, 0x00, 0x01,
            0x87,  // forbidden_bit = 1 (invalid)
            0x42, 0x00, 0x0a,
        ];

        let result = validator.validate(&data).unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == ValidationErrorCode::BitstreamCorruption));
    }

    #[test]
    fn test_profile_constraint_validation() {
        let validator = BitstreamValidator::new()
            .expect_profile(H264Profile::Main);

        // SPS with Baseline profile when expecting Main
        let data = vec![
            0x00, 0x00, 0x00, 0x01,
            0x67, 0x42, 0x00, 0x1e,  // profile_idc = 66 (Baseline)
            0x00, 0x00, 0x00, 0x00,
        ];

        let result = validator.validate(&data).unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == ValidationErrorCode::ProfileConstraintViolation));
    }

    #[test]
    fn test_three_byte_start_code() {
        let validator = BitstreamValidator::new();
        // 3-byte start code (0x000001)
        let data = vec![
            0x00, 0x00, 0x01,  // 3-byte start code
            0x67, 0x42, 0x00, 0x0a, 0x00,
        ];

        let result = validator.validate(&data).unwrap();
        assert_eq!(result.nal_stats.sps_count, 1);
    }
}

// =============================================================================
// Checksum Tests
// =============================================================================

mod checksum_tests {
    use super::*;

    #[test]
    fn test_md5_consistency() {
        let data = b"Hello, H.264 conformance testing!";
        let hash1 = checksum::compute_md5(data);
        let hash2 = checksum::compute_md5(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_sha256_consistency() {
        let data = b"Hello, H.264 conformance testing!";
        let hash1 = checksum::compute_sha256(data);
        let hash2 = checksum::compute_sha256(data);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 = 64 hex chars
    }

    #[test]
    fn test_frame_checksum() {
        let width = 16u32;
        let height = 16u32;
        let y_size = (width * height) as usize;
        let uv_size = (width * height / 4) as usize;

        let y = vec![128u8; y_size];
        let u = vec![128u8; uv_size];
        let v = vec![128u8; uv_size];

        let checksum = checksum::compute_frame_md5(&y, &u, &v, width, height);
        assert!(!checksum.is_empty());
    }

    #[test]
    fn test_streaming_sha256() {
        let data = b"Hello, World!";

        let direct = checksum::compute_sha256(data);

        let mut calc = ChecksumCalculator::new(ChecksumAlgorithm::Sha256);
        calc.update(b"Hello, ");
        calc.update(b"World!");
        let streaming = calc.finalize();

        assert_eq!(direct, streaming);
    }

    #[test]
    fn test_different_data_different_hash() {
        let hash1 = checksum::compute_md5(b"data1");
        let hash2 = checksum::compute_md5(b"data2");
        assert_ne!(hash1, hash2);
    }
}

// =============================================================================
// Profile Test Suite Definitions
// =============================================================================

mod profile_suites {
    use super::*;

    #[test]
    fn test_baseline_profile_tests_defined() {
        let tests = BaselineProfileTests::new();
        let streams = tests.streams();

        assert!(!streams.is_empty());
        for stream in streams {
            assert_eq!(stream.profile, H264Profile::Baseline);
        }
    }

    #[test]
    fn test_main_profile_tests_defined() {
        let tests = MainProfileTests::new();
        let streams = tests.streams();

        assert!(!streams.is_empty());
        for stream in streams {
            assert_eq!(stream.profile, H264Profile::Main);
        }
    }

    #[test]
    fn test_high_profile_tests_defined() {
        let tests = HighProfileTests::new();
        let streams = tests.streams();

        assert!(!streams.is_empty());
        for stream in streams {
            assert_eq!(stream.profile, H264Profile::High);
        }
    }

    #[test]
    fn test_category_filtering() {
        let tests = BaselineProfileTests::new();
        let cavlc = tests.by_category("cavlc");
        assert!(!cavlc.is_empty());

        for stream in cavlc {
            assert!(stream.categories.contains(&"cavlc".to_string()));
        }
    }

    #[test]
    fn test_itu_stream_index() {
        let index = ConformanceStreamIndex::new();
        assert!(!index.all().is_empty());

        let baseline = index.by_profile(H264Profile::Baseline);
        assert!(!baseline.is_empty());
    }
}

// =============================================================================
// Stream Detection Tests
// =============================================================================

mod stream_detection {
    use super::*;

    #[test]
    fn test_annex_b_detection_4_byte() {
        let annex_b = [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a];
        assert!(detect::is_annex_b(&annex_b));
    }

    #[test]
    fn test_annex_b_detection_3_byte() {
        let annex_b = [0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a];
        assert!(detect::is_annex_b(&annex_b));
    }

    #[test]
    fn test_not_annex_b() {
        let not_annex_b = [0x01, 0x42, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00];
        assert!(!detect::is_annex_b(&not_annex_b));
    }

    #[test]
    fn test_profile_detection_baseline() {
        let sps = [0x67, 66, 0x00, 0x0a];
        assert_eq!(detect::detect_profile(&sps), Some(H264Profile::Baseline));
    }

    #[test]
    fn test_profile_detection_main() {
        let sps = [0x67, 77, 0x00, 0x1e];
        assert_eq!(detect::detect_profile(&sps), Some(H264Profile::Main));
    }

    #[test]
    fn test_profile_detection_high() {
        let sps = [0x67, 100, 0x00, 0x28];
        assert_eq!(detect::detect_profile(&sps), Some(H264Profile::High));
    }

    #[test]
    fn test_itu_streams_organization() {
        let baseline = ItuConformanceStreams::baseline();
        let main = ItuConformanceStreams::main();
        let high = ItuConformanceStreams::high();
        let all = ItuConformanceStreams::all();

        assert_eq!(all.len(), baseline.len() + main.len() + high.len());
    }
}

// =============================================================================
// Report Generation Tests
// =============================================================================

mod report_tests {
    use super::*;

    fn create_sample_results() -> Vec<TestResult> {
        vec![
            TestResult {
                stream_id: "test_pass".to_string(),
                test_name: "Passing Test".to_string(),
                status: TestStatus::Passed,
                duration_ms: 100,
                error_message: None,
                decoded_frames: Some(10),
                checksum_results: vec![],
                notes: vec![],
            },
            TestResult {
                stream_id: "test_fail".to_string(),
                test_name: "Failing Test".to_string(),
                status: TestStatus::Failed,
                duration_ms: 200,
                error_message: Some("Checksum mismatch".to_string()),
                decoded_frames: Some(5),
                checksum_results: vec![],
                notes: vec![],
            },
            TestResult {
                stream_id: "test_skip".to_string(),
                test_name: "Skipped Test".to_string(),
                status: TestStatus::Skipped,
                duration_ms: 0,
                error_message: Some("Stream not available".to_string()),
                decoded_frames: None,
                checksum_results: vec![],
                notes: vec![],
            },
        ]
    }

    #[test]
    fn test_summary_calculation() {
        let results = create_sample_results();
        let summary = ReportSummary::from_results(&results);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.skipped, 1);
        assert!((summary.pass_rate - 33.33).abs() < 1.0);
    }

    #[test]
    fn test_json_report_generation() {
        let config = ConformanceConfig::baseline_only();
        let streams: Vec<TestStream> = vec![];
        let results = create_sample_results();

        let generator = ReportGenerator::new(ReportFormat::Json);
        let report = generator.generate("Test Report", &config, &streams, &results);

        assert_eq!(report.title, "Test Report");
        assert_eq!(report.summary.total, 3);
    }

    #[test]
    fn test_report_formats() {
        let config = ConformanceConfig::default();
        let streams: Vec<TestStream> = vec![];
        let results = create_sample_results();

        // Test each format can be generated without error
        for format in [ReportFormat::Json, ReportFormat::Text, ReportFormat::Html, ReportFormat::Markdown] {
            let generator = ReportGenerator::new(format);
            let report = generator.generate("Test", &config, &streams, &results);
            assert!(!report.timestamp.is_empty());
        }
    }
}

// =============================================================================
// Test Filter Tests
// =============================================================================

mod filter_tests {
    use super::*;

    fn create_test_streams() -> Vec<TestStream> {
        vec![
            TestStream::builder("baseline_1")
                .profile(H264Profile::Baseline)
                .category("basic")
                .build(),
            TestStream::builder("baseline_2")
                .profile(H264Profile::Baseline)
                .category("cabac")
                .build(),
            TestStream::builder("main_1")
                .profile(H264Profile::Main)
                .category("basic")
                .build(),
            TestStream::builder("high_1")
                .profile(H264Profile::High)
                .category("advanced")
                .build(),
        ]
    }

    #[test]
    fn test_no_filter() {
        let streams = create_test_streams();
        let filter = TestFilter::new();
        let filtered = filter.apply(&streams);
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_filter_by_profile() {
        let streams = create_test_streams();
        let filter = TestFilter::new().include_profiles(vec![H264Profile::Baseline]);
        let filtered = filter.apply(&streams);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|s| s.profile == H264Profile::Baseline));
    }

    #[test]
    fn test_filter_by_category() {
        let streams = create_test_streams();
        let filter = TestFilter::new().include_categories(vec!["basic".to_string()]);
        let filtered = filter.apply(&streams);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|s| s.categories.contains(&"basic".to_string())));
    }

    #[test]
    fn test_filter_exclude_ids() {
        let streams = create_test_streams();
        let filter = TestFilter::new().exclude_ids(vec!["baseline_1".to_string(), "high_1".to_string()]);
        let filtered = filter.apply(&streams);

        assert_eq!(filtered.len(), 2);
        assert!(!filtered.iter().any(|s| s.id == "baseline_1" || s.id == "high_1"));
    }

    #[test]
    fn test_combined_filters() {
        let streams = create_test_streams();
        let filter = TestFilter::new()
            .include_profiles(vec![H264Profile::Baseline])
            .include_categories(vec!["basic".to_string()]);
        let filtered = filter.apply(&streams);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "baseline_1");
    }
}

// =============================================================================
// Baseline Profile Conformance Tests (Ignored - require external files)
// =============================================================================

mod baseline_conformance {
    use super::*;

    #[test]
    #[ignore = "Requires external test stream BA1_Sony_D"]
    fn test_ba1_sony_d_basic_ip_slice() {
        let config = ConformanceConfig::baseline_only()
            .with_cache_dir(std::env::temp_dir().join("transcode-conformance-test"));

        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("BA1_Sony_D")
            .name("BA1_Sony_D - Basic I/P slice")
            .profile(H264Profile::Baseline)
            .level(H264Level::Level21)
            .resolution(176, 144)
            .expected_frames(17)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream BA2_Sony_F"]
    fn test_ba2_sony_f_multiple_slices() {
        let config = ConformanceConfig::baseline_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("BA2_Sony_F")
            .name("BA2_Sony_F - Multiple slices")
            .profile(H264Profile::Baseline)
            .expected_frames(300)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream CVBS3_Sony_C"]
    fn test_cvbs3_sony_c_cavlc() {
        let config = ConformanceConfig::baseline_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CVBS3_Sony_C")
            .name("CVBS3_Sony_C - CAVLC coding")
            .profile(H264Profile::Baseline)
            .level(H264Level::Level3)
            .resolution(720, 480)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires all baseline profile test streams"]
    fn test_all_baseline_streams() {
        let config = ConformanceConfig::baseline_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let streams = ItuConformanceStreams::baseline();

        let results = runner.run_all(&streams);

        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let total = results.len();

        println!("Baseline profile: {}/{} tests passed", passed, total);
        assert_eq!(passed, total, "Not all baseline tests passed");
    }
}

// =============================================================================
// Main Profile Conformance Tests (Ignored - require external files)
// =============================================================================

mod main_conformance {
    use super::*;

    #[test]
    #[ignore = "Requires external test stream CAPM3_Sony_D"]
    fn test_capm3_sony_d_cabac() {
        let config = ConformanceConfig::main_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CAPM3_Sony_D")
            .name("CAPM3_Sony_D - Main profile CABAC")
            .profile(H264Profile::Main)
            .level(H264Level::Level3)
            .resolution(720, 480)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream CAMP_MOT_MBAFF_L30"]
    fn test_camp_mot_mbaff() {
        let config = ConformanceConfig::main_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CAMP_MOT_MBAFF_L30")
            .name("CAMP_MOT_MBAFF_L30 - MBAFF")
            .profile(H264Profile::Main)
            .level(H264Level::Level3)
            .resolution(720, 576)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream CAWP1_TOSHIBA_E"]
    fn test_cawp1_toshiba_weighted_prediction() {
        let config = ConformanceConfig::main_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CAWP1_TOSHIBA_E")
            .name("CAWP1_TOSHIBA_E - Weighted prediction")
            .profile(H264Profile::Main)
            .level(H264Level::Level21)
            .resolution(352, 288)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires all main profile test streams"]
    fn test_all_main_streams() {
        let config = ConformanceConfig::main_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let streams = ItuConformanceStreams::main();

        let results = runner.run_all(&streams);

        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let total = results.len();

        println!("Main profile: {}/{} tests passed", passed, total);
        assert_eq!(passed, total, "Not all main profile tests passed");
    }
}

// =============================================================================
// High Profile Conformance Tests (Ignored - require external files)
// =============================================================================

mod high_conformance {
    use super::*;

    #[test]
    #[ignore = "Requires external test stream CAPH_HP_B"]
    fn test_caph_hp_b_basic_high() {
        let config = ConformanceConfig::high_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CAPH_HP_B")
            .name("CAPH_HP_B - High Profile basic")
            .profile(H264Profile::High)
            .level(H264Level::Level4)
            .resolution(1920, 1080)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream CAH1_Sony_B"]
    fn test_cah1_sony_b_8x8_transform() {
        let config = ConformanceConfig::high_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CAH1_Sony_B")
            .name("CAH1_Sony_B - 8x8 transform")
            .profile(H264Profile::High)
            .level(H264Level::Level4)
            .resolution(1280, 720)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires external test stream CVHP_Toshiba_B"]
    fn test_cvhp_toshiba_b_high_cavlc() {
        let config = ConformanceConfig::high_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let stream = TestStream::builder("CVHP_Toshiba_B")
            .name("CVHP_Toshiba_B - High profile CAVLC")
            .profile(H264Profile::High)
            .level(H264Level::Level41)
            .resolution(1920, 1080)
            .build();

        let result = runner.run_test(&stream);
        assert_eq!(result.status, TestStatus::Passed, "Failed: {:?}", result.error_message);
    }

    #[test]
    #[ignore = "Requires all high profile test streams"]
    fn test_all_high_streams() {
        let config = ConformanceConfig::high_only();
        let runner = ConformanceRunner::new(config).unwrap();
        let streams = ItuConformanceStreams::high();

        let results = runner.run_all(&streams);

        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let total = results.len();

        println!("High profile: {}/{} tests passed", passed, total);
        assert_eq!(passed, total, "Not all high profile tests passed");
    }
}

// =============================================================================
// Full Conformance Suite (Ignored - requires all external files)
// =============================================================================

mod full_conformance {
    use super::*;

    #[test]
    #[ignore = "Requires all ITU-T conformance test streams"]
    fn test_full_conformance_suite() {
        let config = ConformanceConfig::default();
        let runner = ConformanceRunner::new(config).unwrap();
        let streams = ItuConformanceStreams::all();

        println!("Running full conformance suite with {} streams", streams.len());

        let results = runner.run_all(&streams);

        // Generate report
        let report = runner.generate_report(&streams, &results, ReportFormat::Text);

        // Print summary
        println!("\n{}", "=".repeat(60));
        println!("CONFORMANCE TEST RESULTS");
        println!("{}", "=".repeat(60));
        println!("Total:   {}", report.summary.total);
        println!("Passed:  {} ({:.1}%)", report.summary.passed, report.summary.pass_rate);
        println!("Failed:  {}", report.summary.failed);
        println!("Skipped: {}", report.summary.skipped);
        println!("Errors:  {}", report.summary.errors);
        println!("{}", "=".repeat(60));

        // Verify high pass rate
        assert!(report.summary.pass_rate >= 95.0,
            "Pass rate {:.1}% is below required 95%", report.summary.pass_rate);
    }

    #[test]
    #[ignore = "Requires conformance test streams and generates reports"]
    fn test_generate_conformance_report() {
        let config = ConformanceConfig::default()
            .with_cache_dir(std::env::temp_dir().join("transcode-conformance"));

        let runner = ConformanceRunner::new(config).unwrap();
        let streams = ItuConformanceStreams::all();
        let results = runner.run_all(&streams);

        // Generate reports in different formats
        let report_dir = std::env::temp_dir().join("transcode-reports");
        std::fs::create_dir_all(&report_dir).unwrap();

        for (format, ext) in [
            (ReportFormat::Json, "json"),
            (ReportFormat::Html, "html"),
            (ReportFormat::Markdown, "md"),
            (ReportFormat::Text, "txt"),
        ] {
            let generator = ReportGenerator::new(format);
            let report = generator.generate("H.264 Conformance Report", &ConformanceConfig::default(), &streams, &results);

            let path = report_dir.join(format!("conformance_report.{}", ext));
            generator.write_to_file(&report, &path).unwrap();

            println!("Generated report: {:?}", path);
        }
    }
}

// =============================================================================
// Decoded Frame Comparison Tests (Ignored - require external files)
// =============================================================================

mod frame_comparison {
    use super::*;

    #[test]
    #[ignore = "Requires reference decoded frames"]
    fn test_decoded_frame_md5_verification() {
        // This test would verify that decoded frames match expected MD5 checksums
        // It requires reference decoded frames from a known-good decoder

        let expected_checksums = vec![
            "d41d8cd98f00b204e9800998ecf8427e", // frame 0
            "098f6bcd4621d373cade4e832627b4f6", // frame 1
            // ... more frame checksums
        ];

        // Simulated decoded frames (would come from actual decoder)
        let decoded_frames: Vec<Vec<u8>> = vec![];

        for (i, frame) in decoded_frames.iter().enumerate() {
            if i < expected_checksums.len() {
                let actual = checksum::compute_md5(frame);
                assert_eq!(
                    actual, expected_checksums[i],
                    "Frame {} checksum mismatch", i
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires YUV reference frames"]
    fn test_yuv_frame_comparison() {
        // Test comparing YUV planes individually
        let width = 1920u32;
        let height = 1080u32;

        let y_size = (width * height) as usize;
        let uv_size = (width * height / 4) as usize;

        // Reference frame (would be loaded from file)
        let ref_y = vec![0u8; y_size];
        let ref_u = vec![0u8; uv_size];
        let ref_v = vec![0u8; uv_size];

        // Decoded frame (would come from decoder)
        let dec_y = vec![0u8; y_size];
        let dec_u = vec![0u8; uv_size];
        let dec_v = vec![0u8; uv_size];

        let ref_checksum = checksum::compute_frame_md5(&ref_y, &ref_u, &ref_v, width, height);
        let dec_checksum = checksum::compute_frame_md5(&dec_y, &dec_u, &dec_v, width, height);

        assert_eq!(ref_checksum, dec_checksum, "Decoded frame does not match reference");
    }
}

// =============================================================================
// Local Stream Tests (Can run with local test files)
// =============================================================================

mod local_streams {
    use super::*;

    #[test]
    #[ignore = "Requires local test streams directory"]
    fn test_load_local_streams() {
        let local_dir = PathBuf::from("/path/to/local/streams");
        let loader = LocalStreamLoader::new(local_dir);

        let streams = loader.load_streams();
        println!("Found {} local streams", streams.len());

        for stream in &streams {
            println!("  - {} ({:?})", stream.id, stream.local_path);
        }

        assert!(!streams.is_empty(), "No local streams found");
    }

    #[test]
    #[ignore = "Requires specific local test stream"]
    fn test_run_local_stream() {
        let local_path = PathBuf::from("/path/to/test.264");

        let stream = TestStream::builder("local_test")
            .name("Local Test Stream")
            .profile(H264Profile::Baseline)
            .local_path(local_path)
            .build();

        let config = ConformanceConfig::baseline_only();
        let runner = ConformanceRunner::new(config).unwrap();

        let result = runner.run_test(&stream);
        println!("Result: {:?}", result.status);

        if let Some(error) = &result.error_message {
            println!("Error: {}", error);
        }
    }
}
