//! Basic neural upscaling example.
//!
//! This example demonstrates how to use the neural upscaler with mock inference
//! for testing and development purposes.
//!
//! Run with:
//! ```sh
//! cargo run --example basic_upscale -p transcode-neural
//! ```

use transcode_neural::{NeuralConfig, NeuralFrame, NeuralUpscaler, ModelType};

fn main() -> transcode_neural::Result<()> {
    // Create configuration for 2x upscaling
    let config = NeuralConfig {
        model: ModelType::RealEsrgan,
        scale: 2,
        use_gpu: false, // Use CPU for this example
        tile_size: 256,
        ..Default::default()
    };

    // Create upscaler
    let upscaler = NeuralUpscaler::new(config)?;

    // Create a test frame (64x64 RGB)
    let mut frame = NeuralFrame::new(64, 64);

    // Fill with gradient pattern for visualization
    for y in 0..64 {
        for x in 0..64 {
            let r = x as f32 / 64.0;
            let g = y as f32 / 64.0;
            let b = 0.5;
            frame.set(x, y, 0, r);
            frame.set(x, y, 1, g);
            frame.set(x, y, 2, b);
        }
    }

    println!("Input frame: {}x{}", frame.width, frame.height);

    // Upscale the frame
    let upscaled = upscaler.upscale(&frame)?;

    println!("Output frame: {}x{}", upscaled.width, upscaled.height);
    println!("Scale factor: {}x", upscaled.width / frame.width);

    // Verify dimensions
    assert_eq!(upscaled.width, frame.width * 2);
    assert_eq!(upscaled.height, frame.height * 2);

    println!("Upscaling completed successfully!");

    Ok(())
}
