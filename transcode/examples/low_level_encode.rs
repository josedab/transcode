//! Low-level encoding example.
//!
//! This example demonstrates how to use the codec APIs directly
//! for custom encoding workflows.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example low_level_encode
//! ```

use transcode_codecs::video::h264::{
    H264Encoder, H264EncoderConfig, H264Profile, H264Level,
    RateControlMode, EncoderPreset, ThreadingConfig,
};
use transcode_codecs::traits::VideoEncoder;
use transcode_core::{Frame, PixelFormat, TimeBase, Timestamp};

fn main() -> transcode_core::Result<()> {
    println!("Low-Level H.264 Encoding Example");
    println!("══════════════════════════════════════════════════════════");
    println!();

    // Configure the encoder
    let config = H264EncoderConfig {
        width: 1920,
        height: 1080,
        frame_rate: (30, 1),
        profile: H264Profile::High,
        level: H264Level::from_idc(40),
        rate_control: RateControlMode::Cbr(5_000_000),  // 5 Mbps
        preset: EncoderPreset::Fast,
        gop_size: 60,  // GOP size
        bframes: 0,
        ref_frames: 3,
        cabac: true,
        threading: ThreadingConfig::default(),
    };

    println!("Encoder Configuration:");
    println!("  Resolution:  {}x{}", config.width, config.height);
    println!("  Bitrate:     5000 kbps");
    println!("  Framerate:   {} fps", config.frame_rate.0);
    println!("  Profile:     {:?}", config.profile);
    println!("  GOP size:    {} frames", config.gop_size);
    println!();

    // Create the encoder
    let mut encoder = H264Encoder::new(config)?;

    println!("Encoding synthetic frames...");
    println!("────────────────────────────────────────────────────────────");

    let time_base = TimeBase::new(1, 30);  // 30 fps
    let mut total_bytes = 0u64;
    let num_frames = 90;  // 3 seconds at 30 fps

    for i in 0..num_frames {
        // Create a synthetic test frame
        let frame = create_test_frame(1920, 1080, i, time_base);

        // Encode the frame
        let packets = encoder.encode(&frame)?;

        for packet in &packets {
            total_bytes += packet.data().len() as u64;

            // Show keyframe information
            if packet.is_keyframe() {
                println!("  Frame {:3}: Keyframe, {} bytes", i, packet.data().len());
            }
        }
    }

    // Flush remaining frames
    let flush_packets = encoder.flush()?;
    for packet in &flush_packets {
        total_bytes += packet.data().len() as u64;
    }

    println!();
    println!("────────────────────────────────────────────────────────────");
    println!("Encoding complete!");
    println!("  Frames encoded: {}", num_frames);
    println!("  Total output:   {} bytes ({:.2} KB)", total_bytes, total_bytes as f64 / 1024.0);
    println!("  Avg per frame:  {:.0} bytes", total_bytes as f64 / num_frames as f64);
    println!();
    println!("══════════════════════════════════════════════════════════");

    Ok(())
}

/// Create a synthetic test frame with a gradient pattern.
fn create_test_frame(width: u32, height: u32, frame_num: u32, time_base: TimeBase) -> Frame {
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);

    // Set timestamp
    frame.pts = Timestamp::new(frame_num as i64, time_base);

    // Fill Y plane with a moving gradient
    if let Some(y_plane) = frame.plane_mut(0) {
        for row in 0..height as usize {
            for col in 0..width as usize {
                let val = ((col + row + frame_num as usize * 2) % 256) as u8;
                y_plane[row * width as usize + col] = val;
            }
        }
    }

    // Fill U and V planes (half resolution for YUV420)
    let uv_width = width as usize / 2;
    let uv_height = height as usize / 2;

    if let Some(u_plane) = frame.plane_mut(1) {
        for i in 0..(uv_width * uv_height) {
            u_plane[i] = 128;  // Neutral chroma
        }
    }

    if let Some(v_plane) = frame.plane_mut(2) {
        for i in 0..(uv_width * uv_height) {
            v_plane[i] = 128;  // Neutral chroma
        }
    }

    frame
}
