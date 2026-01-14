//! Hardware detection example.
//!
//! This example demonstrates how to detect available hardware
//! acceleration capabilities on the current system.
//!
//! Run with:
//! ```sh
//! cargo run --example detect_hardware -p transcode-hwaccel
//! ```

use transcode_hwaccel::{HwAccelType, HwCapabilities, HwCodec, detect_accelerators};

fn main() -> transcode_hwaccel::Result<()> {
    println!("Hardware Acceleration Detection");
    println!("================================\n");

    // Detect available accelerators
    let accelerators = detect_accelerators();

    if accelerators.is_empty() {
        println!("No hardware accelerators detected.");
        println!("Software encoding/decoding will be used as fallback.\n");
    } else {
        println!("Detected {} hardware accelerator(s):\n", accelerators.len());
        for info in &accelerators {
            println!("  {} - {}", info.accel_type.name(), info.capabilities.device_name);
        }
        println!();
    }

    // Check individual accelerator availability
    println!("Accelerator availability:");
    println!("  VideoToolbox: {}", HwAccelType::VideoToolbox.is_available());
    println!("  VA-API:       {}", HwAccelType::Vaapi.is_available());
    println!("  NVENC:        {}", HwAccelType::Nvenc.is_available());
    println!("  QSV:          {}", HwAccelType::Qsv.is_available());
    println!("  Software:     {}", HwAccelType::Software.is_available());
    println!();

    // Print codec names
    println!("Supported codec types:");
    println!("  H.264/AVC:  {}", HwCodec::H264.name());
    println!("  H.265/HEVC: {}", HwCodec::Hevc.name());
    println!("  AV1:        {}", HwCodec::Av1.name());
    println!("  VP9:        {}", HwCodec::Vp9.name());
    println!();

    // Example capabilities (what a typical system might report)
    let example_caps = HwCapabilities {
        accel_type: HwAccelType::Software,
        encode_codecs: vec![HwCodec::H264, HwCodec::Hevc],
        decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
        max_width: 4096,
        max_height: 2160,
        supports_bframes: true,
        supports_10bit: true,
        supports_hdr: false,
        device_name: "Software Fallback".to_string(),
    };

    println!("Example capabilities (software fallback):");
    println!("  Device: {}", example_caps.device_name);
    println!("  Max resolution: {}x{}", example_caps.max_width, example_caps.max_height);
    println!("  B-frames: {}", example_caps.supports_bframes);
    println!("  10-bit: {}", example_caps.supports_10bit);
    println!("  HDR: {}", example_caps.supports_hdr);
    println!("  Encode codecs: {:?}", example_caps.encode_codecs.iter().map(|c| c.name()).collect::<Vec<_>>());
    println!("  Decode codecs: {:?}", example_caps.decode_codecs.iter().map(|c| c.name()).collect::<Vec<_>>());

    Ok(())
}
