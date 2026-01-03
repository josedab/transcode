//! Quality metrics example - comparing video quality using PSNR, SSIM, and MS-SSIM.
//!
//! Run with: cargo run --example quality_metrics

use transcode_quality::{
    Frame, MsSsim, Psnr, PsnrConfig, QualityAssessment, QualityConfig, Ssim, SsimConfig,
};

fn main() -> transcode_quality::Result<()> {
    println!("=== Transcode Quality Metrics Demo ===\n");

    // Create test frames (in practice, these would come from decoded video)
    let width = 256;
    let height = 256;

    // Reference frame - clean image with gradient
    let reference = create_gradient_frame(width, height);

    // Distorted frame - add some noise
    let distorted = add_noise(&reference, 10);

    // Calculate PSNR
    println!("1. PSNR (Peak Signal-to-Noise Ratio)");
    println!("   Higher is better, typical range: 30-50 dB\n");

    let psnr = Psnr::new(PsnrConfig::default());
    let psnr_result = psnr.calculate(&reference, &distorted)?;

    println!("   Overall:     {:.2} dB", psnr_result.psnr);
    println!("   MSE:         {:.4}", psnr_result.mse);
    println!("   Channel R:   {:.2} dB", psnr_result.per_channel[0]);
    println!("   Channel G:   {:.2} dB", psnr_result.per_channel[1]);
    println!("   Channel B:   {:.2} dB\n", psnr_result.per_channel[2]);

    // Calculate SSIM
    println!("2. SSIM (Structural Similarity Index)");
    println!("   Range: -1.0 to 1.0, higher is better\n");

    let ssim = Ssim::new(SsimConfig::default());
    let ssim_result = ssim.calculate(&reference, &distorted)?;

    println!("   Score:       {:.4}", ssim_result.ssim);
    println!("   Luminance:   {:.4}", ssim_result.luminance);
    println!("   Contrast:    {:.4}", ssim_result.contrast);
    println!("   Structure:   {:.4}\n", ssim_result.structure);

    // Calculate MS-SSIM (multi-scale)
    println!("3. MS-SSIM (Multi-Scale SSIM)");
    println!("   Better correlation with human perception\n");

    let ms_ssim = MsSsim::new(5, SsimConfig::default()); // 5 scales
    let ms_ssim_result = ms_ssim.calculate(&reference, &distorted)?;

    println!("   Score:       {:.4}\n", ms_ssim_result);

    // Comprehensive quality assessment
    println!("4. Comprehensive Quality Assessment");
    println!("   Combines multiple metrics\n");

    let qa = QualityAssessment::new(QualityConfig::default());
    let report = qa.assess(&reference, &distorted)?;

    if let Some(ref psnr) = report.psnr {
        println!("   PSNR:        {:.2} dB", psnr.psnr);
    }
    if let Some(ref ssim) = report.ssim {
        println!("   SSIM:        {:.4}", ssim.ssim);
    }
    println!("   Overall:     {:.1}/100 ({})", report.overall_score(), report.rating());

    // Quality interpretation
    println!("\n5. Quality Interpretation");
    let quality = interpret_quality(psnr_result.psnr, ssim_result.ssim);
    println!("   Assessment:  {}", quality);

    Ok(())
}

/// Create a gradient test frame.
fn create_gradient_frame(width: u32, height: u32) -> Frame {
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Gradient from top-left to bottom-right
            let lum = ((x + y) * 255 / (width + height)) as u8;
            data[idx] = lum; // R
            data[idx + 1] = lum; // G
            data[idx + 2] = lum; // B
        }
    }

    Frame::new(data, width, height, 3)
}

/// Add random noise to a frame.
fn add_noise(frame: &Frame, amount: u8) -> Frame {
    let mut data = frame.data.clone();

    // Simple deterministic "noise" for reproducibility
    for (i, pixel) in data.iter_mut().enumerate() {
        let noise = ((i * 17 + i / 3) % (amount as usize * 2)) as i16 - amount as i16;
        *pixel = (*pixel as i16 + noise).clamp(0, 255) as u8;
    }

    Frame::new(data, frame.width, frame.height, frame.channels)
}

/// Interpret quality based on metrics.
fn interpret_quality(psnr: f64, ssim: f64) -> &'static str {
    if psnr > 40.0 && ssim > 0.95 {
        "Excellent - Visually indistinguishable from original"
    } else if psnr > 35.0 && ssim > 0.90 {
        "Good - Minor artifacts, high quality"
    } else if psnr > 30.0 && ssim > 0.80 {
        "Fair - Noticeable degradation but acceptable"
    } else if psnr > 25.0 && ssim > 0.70 {
        "Poor - Significant quality loss"
    } else {
        "Bad - Severe degradation"
    }
}
