//! Quality metrics integration tests.
//!
//! Tests for PSNR, SSIM, MS-SSIM, and VMAF quality assessment.

use transcode_quality::{
    psnr, ssim, ms_ssim, vmaf, Frame,
    Psnr, PsnrConfig, Ssim, SsimConfig,
    QualityAssessment, QualityConfig, QualityMetrics,
    BatchQualityAssessment,
};

/// Create a test frame with uniform color.
fn create_uniform_frame(width: u32, height: u32, value: u8) -> Frame {
    let data = vec![value; (width * height * 3) as usize];
    Frame::new(data, width, height, 3)
}

/// Create a test frame with a gradient pattern.
fn create_gradient_frame(width: u32, height: u32) -> Frame {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let val = ((x + y) % 256) as u8;
            data.push(val); // R
            data.push(val); // G
            data.push(val); // B
        }
    }
    Frame::new(data, width, height, 3)
}

/// Create a noisy version of a frame.
fn add_noise(frame: &Frame, noise_level: u8) -> Frame {
    let mut data = frame.data.clone();
    for (i, byte) in data.iter_mut().enumerate() {
        let noise = ((i * 7) % (noise_level as usize * 2)) as i16 - noise_level as i16;
        *byte = ((*byte as i16 + noise).clamp(0, 255)) as u8;
    }
    Frame::new(data, frame.width, frame.height, frame.channels)
}

// === PSNR Tests ===

/// Test PSNR with identical frames (infinite PSNR).
#[test]
fn test_psnr_identical_frames() {
    let frame = create_uniform_frame(64, 64, 128);
    let result = psnr(&frame, &frame).expect("PSNR should succeed");

    assert!(result.is_infinite(), "PSNR of identical frames should be infinite");
}

/// Test PSNR with different frames.
#[test]
fn test_psnr_different_frames() {
    let reference = create_uniform_frame(64, 64, 128);
    let distorted = create_uniform_frame(64, 64, 138);

    let result = psnr(&reference, &distorted).expect("PSNR should succeed");

    // PSNR should be a reasonable positive value
    assert!(result > 0.0);
    assert!(result < 100.0);
    assert!(result.is_finite());
}

/// Test PSNR calculator reuse.
#[test]
fn test_psnr_calculator() {
    let calc = Psnr::new(PsnrConfig { bit_depth: 8 });

    let ref_frame = create_uniform_frame(64, 64, 100);
    let dist_frame = create_uniform_frame(64, 64, 110);

    let result1 = calc.calculate(&ref_frame, &dist_frame).unwrap();
    let result2 = calc.calculate(&ref_frame, &dist_frame).unwrap();

    // Results should be consistent
    assert!((result1.psnr - result2.psnr).abs() < 0.001);
}

/// Test PSNR with varying noise levels.
#[test]
fn test_psnr_noise_levels() {
    let reference = create_gradient_frame(64, 64);

    let low_noise = add_noise(&reference, 5);
    let high_noise = add_noise(&reference, 30);

    let psnr_low = psnr(&reference, &low_noise).unwrap();
    let psnr_high = psnr(&reference, &high_noise).unwrap();

    // Lower noise should give higher PSNR
    assert!(psnr_low > psnr_high, "Lower noise should yield higher PSNR");
}

// === SSIM Tests ===

/// Test SSIM with identical frames (SSIM = 1.0).
#[test]
fn test_ssim_identical_frames() {
    let frame = create_gradient_frame(64, 64);
    let result = ssim(&frame, &frame).expect("SSIM should succeed");

    // SSIM of identical frames should be 1.0
    assert!((result - 1.0).abs() < 0.01, "SSIM of identical frames should be 1.0");
}

/// Test SSIM with different frames.
#[test]
fn test_ssim_different_frames() {
    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 20);

    let result = ssim(&reference, &distorted).expect("SSIM should succeed");

    // SSIM should be less than 1.0 for different frames
    assert!(result < 1.0);
    // But still positive for similar frames
    assert!(result > 0.0);
}

/// Test SSIM calculator reuse.
#[test]
fn test_ssim_calculator() {
    let calc = Ssim::new(SsimConfig::default());

    let ref_frame = create_gradient_frame(64, 64);
    let dist_frame = add_noise(&ref_frame, 10);

    let result1 = calc.calculate(&ref_frame, &dist_frame).unwrap();
    let result2 = calc.calculate(&ref_frame, &dist_frame).unwrap();

    // Results should be consistent
    assert!((result1.ssim - result2.ssim).abs() < 0.001);
}

/// Test SSIM with varying noise levels.
#[test]
fn test_ssim_noise_levels() {
    let reference = create_gradient_frame(64, 64);

    let low_noise = add_noise(&reference, 5);
    let high_noise = add_noise(&reference, 30);

    let ssim_low = ssim(&reference, &low_noise).unwrap();
    let ssim_high = ssim(&reference, &high_noise).unwrap();

    // Lower noise should give higher SSIM
    assert!(ssim_low > ssim_high, "Lower noise should yield higher SSIM");
}

// === MS-SSIM Tests ===

/// Test MS-SSIM with identical frames.
#[test]
fn test_ms_ssim_identical_frames() {
    // MS-SSIM requires larger frames for multi-scale analysis
    let frame = create_gradient_frame(256, 256);
    let result = ms_ssim(&frame, &frame).expect("MS-SSIM should succeed");

    // MS-SSIM of identical frames should be 1.0
    assert!((result - 1.0).abs() < 0.01, "MS-SSIM of identical frames should be 1.0");
}

/// Test MS-SSIM with different frames.
#[test]
fn test_ms_ssim_different_frames() {
    // MS-SSIM requires larger frames for multi-scale analysis
    let reference = create_gradient_frame(256, 256);
    let distorted = add_noise(&reference, 20);

    let result = ms_ssim(&reference, &distorted).expect("MS-SSIM should succeed");

    // MS-SSIM should be less than 1.0 for different frames
    assert!(result < 1.0);
    assert!(result > 0.0);
}

// === VMAF Tests ===

/// Test VMAF with identical frames.
#[test]
fn test_vmaf_identical_frames() {
    let frame = create_gradient_frame(64, 64);
    let result = vmaf(&frame, &frame).expect("VMAF should succeed");

    // VMAF of identical frames should be 100
    assert!(result >= 95.0, "VMAF of identical frames should be near 100");
}

/// Test VMAF with different frames.
#[test]
fn test_vmaf_different_frames() {
    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 20);

    let result = vmaf(&reference, &distorted).expect("VMAF should succeed");

    // VMAF should be less than 100 for different frames
    assert!(result < 100.0);
    assert!(result > 0.0);
}

// === Unified Assessment Tests ===

/// Test unified quality assessment.
#[test]
fn test_quality_assessment() {
    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 10);

    let qa = QualityAssessment::default();
    let report = qa.assess(&reference, &distorted).expect("Assessment should succeed");

    // Should have all metrics
    assert!(report.psnr.is_some());
    assert!(report.ssim.is_some());
    assert!(report.vmaf.is_some());

    // Overall score should be valid
    let score = report.overall_score();
    assert!(score >= 0.0);
    assert!(score <= 100.0);
}

/// Test fast quality assessment.
#[test]
fn test_fast_quality_assessment() {
    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 10);

    let qa = QualityAssessment::default();
    let report = qa.assess_fast(&reference, &distorted).expect("Assessment should succeed");

    // Should have only PSNR and SSIM
    assert!(report.psnr.is_some());
    assert!(report.ssim.is_some());
    assert!(report.vmaf.is_none());
    assert!(report.ms_ssim.is_none());
}

/// Test quality assessment with custom config.
#[test]
fn test_quality_assessment_config() {
    let config = QualityConfig {
        metrics: QualityMetrics::fast(),
        bit_depth: 8,
        vmaf_model: transcode_quality::VmafModel::Default,
    };

    let qa = QualityAssessment::new(config);
    let reference = create_gradient_frame(64, 64);

    let report = qa.assess(&reference, &reference).expect("Assessment should succeed");

    // With fast metrics, only PSNR and SSIM are computed
    assert!(report.psnr.is_some());
    assert!(report.ssim.is_some());
}

/// Test quality report rating.
#[test]
fn test_quality_report_rating() {
    let reference = create_gradient_frame(64, 64);

    let qa = QualityAssessment::default();

    // Identical frames should rate as "Excellent"
    let report = qa.assess(&reference, &reference).expect("Assessment should succeed");
    assert_eq!(report.rating(), "Excellent");
}

/// Test quality report display.
#[test]
fn test_quality_report_display() {
    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 10);

    let qa = QualityAssessment::default();
    let report = qa.assess(&reference, &distorted).expect("Assessment should succeed");

    let display = format!("{}", report);
    assert!(display.contains("PSNR"));
    assert!(display.contains("SSIM"));
    assert!(display.contains("VMAF"));
    assert!(display.contains("Overall"));
}

// === Batch Assessment Tests ===

/// Test batch quality assessment.
#[test]
fn test_batch_quality_assessment() {
    let mut batch = BatchQualityAssessment::new();
    let qa = QualityAssessment::default();

    let reference = create_gradient_frame(64, 64);

    // Simulate multiple frames with varying quality
    for noise_level in [5, 10, 15, 20] {
        let distorted = add_noise(&reference, noise_level);
        let report = qa.assess(&reference, &distorted).expect("Assessment should succeed");
        batch.add(report);
    }

    // Check averages
    let avg_psnr = batch.average_psnr();
    assert!(avg_psnr.is_some());
    assert!(avg_psnr.unwrap() > 0.0);

    let avg_ssim = batch.average_ssim();
    assert!(avg_ssim.is_some());
    assert!(avg_ssim.unwrap() > 0.0);
    assert!(avg_ssim.unwrap() <= 1.0);

    // Check minimums
    let (min_psnr, min_ssim, _) = batch.min_scores();
    assert!(min_psnr.is_some());
    assert!(min_ssim.is_some());

    // Min should be less than or equal to average
    assert!(min_psnr.unwrap() <= avg_psnr.unwrap());
    assert!(min_ssim.unwrap() <= avg_ssim.unwrap());
}

/// Test batch quality summary.
#[test]
fn test_batch_quality_summary() {
    let mut batch = BatchQualityAssessment::new();
    let qa = QualityAssessment::default();

    let reference = create_gradient_frame(64, 64);
    let distorted = add_noise(&reference, 10);

    let report = qa.assess(&reference, &distorted).expect("Assessment should succeed");
    batch.add(report);

    let summary = batch.summary();
    assert!(summary.contains("Batch Quality Summary"));
    assert!(summary.contains("Frames analyzed: 1"));
    assert!(summary.contains("Average PSNR"));
}

// === Frame Validation Tests ===

/// Test frame validation.
#[test]
fn test_frame_validation() {
    // Valid frame
    let valid = Frame::new(vec![0u8; 64 * 64 * 3], 64, 64, 3);
    assert!(valid.validate().is_ok());

    // Invalid frame (wrong size)
    let invalid = Frame::new(vec![0u8; 100], 64, 64, 3);
    assert!(invalid.validate().is_err());
}

/// Test quality metrics with different resolutions.
#[test]
fn test_quality_multiple_resolutions() {
    for (width, height) in [(32, 32), (64, 64), (128, 128), (256, 256)] {
        let reference = create_gradient_frame(width, height);
        let distorted = add_noise(&reference, 10);

        // All metrics should work at different resolutions
        let psnr_result = psnr(&reference, &distorted);
        assert!(psnr_result.is_ok(), "PSNR should work at {}x{}", width, height);

        let ssim_result = ssim(&reference, &distorted);
        assert!(ssim_result.is_ok(), "SSIM should work at {}x{}", width, height);
    }
}
