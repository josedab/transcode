//! Content intelligence example - scene detection and content classification.
//!
//! Run with: cargo run --example content_intelligence

use transcode_intel::{
    ContentClassifier, DetectionMethod, Frame, SceneConfig, SceneDetector, VideoAnalyzer,
};

fn main() -> transcode_intel::Result<()> {
    println!("=== Transcode Content Intelligence Demo ===\n");

    // Create a sequence of test frames simulating a video
    let frames = create_test_video_sequence();

    // 1. Scene Detection
    println!("1. Scene Detection");
    println!("   Detecting scene changes in {} frames\n", frames.len());

    let config = SceneConfig::default()
        .with_method(DetectionMethod::Combined)
        .with_threshold(0.2)
        .with_min_scene_length(2);

    let mut detector = SceneDetector::new(config);

    for (i, frame) in frames.iter().enumerate() {
        if let Some(confidence) = detector.process_frame(frame)? {
            println!(
                "   Scene change at frame {}: confidence {:.2}",
                i, confidence
            );
        }
    }

    // 2. Content Classification
    println!("\n2. Content Classification");
    println!("   Classifying frame content\n");

    let mut classifier = ContentClassifier::new();

    // Classify a few representative frames
    for i in [0, 4, 8, 12, 16] {
        if i < frames.len() {
            let classification = classifier.classify(&frames[i])?;
            println!("   Frame {}: {:?}", i, classification.shot_type);
            println!("      Content: {:?}", classification.content_type);
            println!("      Motion:  {:?}", classification.motion_level);
            println!("      Complexity: {:.2}", classification.complexity);
            println!();
        }
    }

    // 3. Full Video Analysis
    println!("3. Full Video Analysis");
    println!("   Analyzing complete sequence\n");

    let mut analyzer = VideoAnalyzer::default();
    let analysis = analyzer.analyze_sequence(&frames)?;

    println!("   Total frames:     {}", analysis.frame_count);
    println!("   Scene count:      {}", analysis.scene_count());
    println!(
        "   Avg scene length: {:.1} frames",
        analysis.avg_scene_length()
    );
    println!("   Dominant content: {:?}", analysis.dominant_content);
    println!("   Average motion:   {:.3}", analysis.avg_motion);
    println!("   Avg complexity:   {:.3}", analysis.avg_complexity);

    // 4. Encoding Recommendations
    println!("\n4. Encoding Recommendations");
    let bitrate_factor = analysis.recommended_bitrate_factor();
    println!("   Bitrate factor:   {:.2}x", bitrate_factor);

    if bitrate_factor < 0.9 {
        println!("   Recommendation:   Static content, can use lower bitrate");
    } else if bitrate_factor > 1.1 {
        println!("   Recommendation:   High motion/complexity, use higher bitrate");
    } else {
        println!("   Recommendation:   Standard bitrate should be sufficient");
    }

    println!("\n5. Detected Scenes:");
    for scene in &analysis.scenes {
        println!(
            "   Scene {}: frames {}-{} ({} frames)",
            scene.index,
            scene.start_frame,
            scene.end_frame,
            scene.frame_count()
        );
    }

    Ok(())
}

/// Create a test video sequence simulating different scenes.
fn create_test_video_sequence() -> Vec<Frame> {
    let mut frames = Vec::new();
    let width = 64;
    let height = 64;

    // Scene 1: Dark scene (frames 0-4)
    for _ in 0..5 {
        frames.push(create_uniform_frame(width, height, 30));
    }

    // Scene 2: Bright scene (frames 5-9)
    for _ in 0..5 {
        frames.push(create_uniform_frame(width, height, 200));
    }

    // Scene 3: Gradient/motion scene (frames 10-14)
    for i in 0..5 {
        frames.push(create_gradient_frame(width, height, i * 20));
    }

    // Scene 4: High contrast scene (frames 15-19)
    for _ in 0..5 {
        frames.push(create_checkerboard_frame(width, height));
    }

    frames
}

/// Create a uniform colored frame.
fn create_uniform_frame(width: u32, height: u32, value: u8) -> Frame {
    let data = vec![value; (width * height * 3) as usize];
    Frame::new(data, width, height, 3)
}

/// Create a gradient frame.
fn create_gradient_frame(width: u32, height: u32, offset: u8) -> Frame {
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let val = ((x * 255 / width) as u8).wrapping_add(offset);
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    Frame::new(data, width, height, 3)
}

/// Create a checkerboard pattern frame.
fn create_checkerboard_frame(width: u32, height: u32) -> Frame {
    let mut data = vec![0u8; (width * height * 3) as usize];
    let block_size = 8;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let block_x = x / block_size;
            let block_y = y / block_size;
            let val = if (block_x + block_y) % 2 == 0 { 255 } else { 0 };
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    Frame::new(data, width, height, 3)
}
