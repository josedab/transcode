//! Integration tests for transcode-neural.
//!
//! These tests work without actual ONNX models by using the mock inference path.

use transcode_neural::{
    benchmark::{Benchmarker, InferenceBenchmark, MemoryUsage, Timer},
    inference::{InputTensor, MockInference, OutputTensor},
    models::{ModelManager, ModelRegistry},
    postprocessing::{Postprocessor, PostprocessConfig, nchw_to_hwc, remove_padding},
    preprocessing::{Preprocessor, PreprocessConfig, NormalizationMode, ChannelOrder},
    upscaler::{CpuUpscaler, OnnxUpscaler, OnnxUpscalerConfig, StreamingUpscaler},
    NeuralConfig, NeuralFrame, ModelType,
};
use std::time::Duration;

// ============================================================================
// Preprocessing Tests
// ============================================================================

#[test]
fn test_preprocess_pipeline() {
    let config = PreprocessConfig {
        normalization: NormalizationMode::ZeroOne,
        size_multiple: Some(8),
        ..Default::default()
    };
    let preprocessor = Preprocessor::new(config);

    // Create a non-multiple-of-8 frame
    let mut frame = NeuralFrame::new(100, 100);
    for i in 0..frame.data.len() {
        frame.data[i] = (i % 256) as f32 / 255.0;
    }

    let result = preprocessor.process(&frame).unwrap();

    // Should be padded to 104x104 (next multiple of 8)
    assert_eq!(result.width, 104);
    assert_eq!(result.height, 104);
    assert_eq!(result.original_width, 100);
    assert_eq!(result.original_height, 100);
    assert!(result.is_padded());
}

#[test]
fn test_preprocess_bgr_to_rgb() {
    let config = PreprocessConfig {
        input_order: ChannelOrder::Bgr,
        output_order: ChannelOrder::Rgb,
        size_multiple: None,
        ..Default::default()
    };
    let preprocessor = Preprocessor::new(config);

    // Create a 1x1 BGR frame
    let frame = NeuralFrame::from_rgb(vec![0.1, 0.5, 0.9], 1, 1).unwrap();
    let result = preprocessor.process(&frame).unwrap();

    // B and R should be swapped
    assert!((result.data[0] - 0.9).abs() < 1e-5); // R <- B
    assert!((result.data[1] - 0.5).abs() < 1e-5); // G stays
    assert!((result.data[2] - 0.1).abs() < 1e-5); // B <- R
}

#[test]
fn test_preprocess_normalization_modes() {
    let frame = NeuralFrame::from_rgb(vec![0.5; 12], 2, 2).unwrap();

    // Test NegOneOne normalization
    let config = PreprocessConfig {
        normalization: NormalizationMode::NegOneOne,
        size_multiple: None,
        ..Default::default()
    };
    let preprocessor = Preprocessor::new(config);
    let result = preprocessor.process(&frame).unwrap();

    // 0.5 normalized to NegOneOne: 0.5 * 2 - 1 = 0.0
    assert!((result.data[0] - 0.0).abs() < 1e-5);
}

// ============================================================================
// Postprocessing Tests
// ============================================================================

#[test]
fn test_postprocess_pipeline() {
    let config = PostprocessConfig::default();
    let postprocessor = Postprocessor::new(config);

    // Create NCHW output: 1x3x4x4
    let nchw_data = vec![0.5f32; 48];
    let frame = postprocessor.process_single(&nchw_data, 4, 4, None, 1).unwrap();

    assert_eq!(frame.width, 4);
    assert_eq!(frame.height, 4);
    assert_eq!(frame.data.len(), 48);
}

#[test]
fn test_nchw_to_hwc_conversion() {
    let nchw = vec![
        1.0, 2.0, 3.0, 4.0,  // C0
        5.0, 6.0, 7.0, 8.0,  // C1
        9.0, 10.0, 11.0, 12.0, // C2
    ];

    let hwc = nchw_to_hwc(&nchw, 3, 2, 2);

    // First pixel should be [1, 5, 9]
    assert_eq!(hwc[0], 1.0);
    assert_eq!(hwc[1], 5.0);
    assert_eq!(hwc[2], 9.0);
}

#[test]
fn test_remove_padding() {
    let padded = vec![
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let unpadded = remove_padding(&padded, 3, 3, 2, 2, 3).unwrap();

    assert_eq!(unpadded.len(), 12);
    assert_eq!(&unpadded[0..3], &[1.0, 1.0, 1.0]);
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_mock_inference() {
    let mock = MockInference::new(2);
    let input = InputTensor::new(1, 3, 64, 64);

    let output = mock.run(&input).unwrap();

    assert_eq!(output.shape, vec![1, 3, 128, 128]);
}

#[test]
fn test_input_tensor_from_hwc() {
    let hwc = vec![
        0.1, 0.2, 0.3,  // (0,0)
        0.4, 0.5, 0.6,  // (0,1)
        0.7, 0.8, 0.9,  // (1,0)
        1.0, 1.1, 1.2,  // (1,1)
    ];

    let tensor = InputTensor::from_hwc(&hwc, 2, 2, 3);

    assert_eq!(tensor.batch, 1);
    assert_eq!(tensor.channels, 3);
    assert_eq!(tensor.height, 2);
    assert_eq!(tensor.width, 2);
}

#[test]
fn test_output_tensor_to_hwc() {
    let output = OutputTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![1, 3, 2, 2],
    );

    let hwc = output.to_hwc().unwrap();

    assert_eq!(hwc.len(), 12);
}

#[test]
fn test_batch_input_tensor() {
    let frame1: Vec<f32> = vec![0.0; 12];
    let frame2: Vec<f32> = vec![1.0; 12];

    let tensor = InputTensor::from_batch_hwc(&[&frame1, &frame2], 2, 2, 3);

    assert_eq!(tensor.batch, 2);
    assert_eq!(tensor.shape(), [2, 3, 2, 2]);
}

// ============================================================================
// Upscaler Tests
// ============================================================================

#[test]
fn test_cpu_upscaler_all_algorithms() {
    let frame = NeuralFrame::new(16, 16);

    // Test all CPU algorithms
    let nearest = CpuUpscaler::nearest(&frame, 32, 32).unwrap();
    assert_eq!(nearest.width, 32);
    assert_eq!(nearest.height, 32);

    let bilinear = CpuUpscaler::bilinear(&frame, 32, 32).unwrap();
    assert_eq!(bilinear.width, 32);

    let bicubic = CpuUpscaler::bicubic(&frame, 32, 32).unwrap();
    assert_eq!(bicubic.width, 32);

    let lanczos = CpuUpscaler::lanczos(&frame, 32, 32, 3).unwrap();
    assert_eq!(lanczos.width, 32);
}

#[test]
fn test_onnx_upscaler_with_mock() {
    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 4,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(4);

    let frame = NeuralFrame::new(64, 64);
    let result = upscaler.upscale(&frame).unwrap();

    assert_eq!(result.width, 256);
    assert_eq!(result.height, 256);
}

#[test]
fn test_onnx_upscaler_batch() {
    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(2);

    let frame1 = NeuralFrame::new(32, 32);
    let frame2 = NeuralFrame::new(32, 32);

    let results = upscaler.upscale_batch(&[&frame1, &frame2]).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].width, 64);
    assert_eq!(results[1].width, 64);
}

#[test]
fn test_streaming_upscaler() {
    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut onnx_upscaler = OnnxUpscaler::new(config);
    onnx_upscaler.enable_mock(2);

    let mut streamer = StreamingUpscaler::new(onnx_upscaler, 3);

    // Push frames
    assert!(streamer.push(NeuralFrame::new(8, 8)).unwrap().is_none());
    assert_eq!(streamer.pending(), 1);

    assert!(streamer.push(NeuralFrame::new(8, 8)).unwrap().is_none());
    assert_eq!(streamer.pending(), 2);

    // Third frame triggers flush
    let result = streamer.push(NeuralFrame::new(8, 8)).unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 3);
    assert_eq!(streamer.pending(), 0);
}

#[test]
fn test_progress_callback() {
    use std::sync::{Arc, Mutex};

    let config = OnnxUpscalerConfig::default();
    let progress_messages = Arc::new(Mutex::new(Vec::new()));
    let messages_clone = progress_messages.clone();

    let mut upscaler = OnnxUpscaler::new(config)
        .with_progress(move |progress, msg| {
            messages_clone.lock().unwrap().push((progress, msg.to_string()));
        });
    upscaler.enable_mock(2);

    let frame = NeuralFrame::new(16, 16);
    upscaler.upscale(&frame).unwrap();

    let messages = progress_messages.lock().unwrap();
    assert!(!messages.is_empty());

    // Check first progress is 0 and last is 1.0
    assert!(messages.first().unwrap().0 < 0.5);
    assert!((messages.last().unwrap().0 - 1.0).abs() < 0.1);
}

// ============================================================================
// Model Management Tests
// ============================================================================

#[test]
fn test_model_registry() {
    let registry = ModelRegistry::new();

    // Should have builtin models
    let models = registry.list();
    assert!(!models.is_empty());

    // Find Real-ESRGAN
    let esrgan = registry.get("realesrgan-x4");
    assert!(esrgan.is_some());
    assert_eq!(esrgan.unwrap().scale, 4);
}

#[test]
fn test_model_registry_by_type() {
    let registry = ModelRegistry::new();

    let esrgan = registry.get_by_type(&ModelType::RealEsrgan);
    assert!(esrgan.is_some());

    let anime = registry.get_by_type(&ModelType::RealEsrganAnime);
    assert!(anime.is_some());
}

#[test]
fn test_model_registry_by_scale() {
    let registry = ModelRegistry::new();

    let x4_models = registry.list_by_scale(4);
    assert!(!x4_models.is_empty());

    for model in x4_models {
        assert_eq!(model.scale, 4);
    }
}

#[test]
fn test_model_manager() {
    let temp_dir = std::env::temp_dir().join("transcode_neural_test");
    let manager = ModelManager::new(temp_dir.clone());

    assert_eq!(manager.cache_dir(), &temp_dir);
    assert!(!manager.registry().list().is_empty());
}

// ============================================================================
// Benchmark Tests
// ============================================================================

#[test]
fn test_benchmarker() {
    let benchmarker = Benchmarker::new()
        .with_warmup(1)
        .with_iterations(3);

    let frame = NeuralFrame::new(16, 16);

    let result = benchmarker.run(&frame, 2, |f| {
        Ok(NeuralFrame::new(f.width * 2, f.height * 2))
    }).unwrap();

    assert_eq!(result.warmup_times.len(), 1);
    assert_eq!(result.inference_times.len(), 3);
}

#[test]
fn test_inference_benchmark_stats() {
    let mut benchmark = InferenceBenchmark::new(1920, 1080, 4);
    benchmark.inference_times = vec![
        Duration::from_millis(100),
        Duration::from_millis(110),
        Duration::from_millis(90),
        Duration::from_millis(105),
        Duration::from_millis(95),
    ];

    assert!(benchmark.average_inference_time().as_millis() >= 95);
    assert!(benchmark.average_inference_time().as_millis() <= 105);
    assert_eq!(benchmark.min_inference_time(), Duration::from_millis(90));
    assert_eq!(benchmark.max_inference_time(), Duration::from_millis(110));
    assert!(benchmark.fps() > 9.0 && benchmark.fps() < 11.0);
}

#[test]
fn test_benchmark_report() {
    let mut benchmark = InferenceBenchmark::new(1920, 1080, 4);
    benchmark.model_load_time = Some(Duration::from_millis(500));
    benchmark.inference_times = vec![Duration::from_millis(100)];

    let report = benchmark.report();

    assert!(report.contains("1920x1080"));
    assert!(report.contains("7680x4320"));
    assert!(report.contains("Model load time"));
}

#[test]
fn test_memory_estimation() {
    let frame = NeuralFrame::new(1920, 1080);
    let mem = MemoryUsage::estimate_frame_memory(&frame);

    // 1920 * 1080 * 3 * 4 bytes = ~24 MB
    let expected = 1920 * 1080 * 3 * 4;
    assert_eq!(mem, expected as u64);
}

#[test]
fn test_timer() {
    let timer = Timer::start("test_operation");
    std::thread::sleep(Duration::from_millis(10));
    let metrics = timer.stop();

    assert!(metrics.millis() >= 10.0);
    assert_eq!(metrics.name, "test_operation");
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_full_upscale_pipeline() {
    // Create input frame
    let mut frame = NeuralFrame::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            for c in 0..3 {
                let idx = ((y * 64 + x) * 3 + c) as usize;
                frame.data[idx] = ((x + y) % 256) as f32 / 255.0;
            }
        }
    }

    // Configure upscaler
    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 2,
            ..Default::default()
        },
        preprocess: PreprocessConfig {
            normalization: NormalizationMode::ZeroOne,
            size_multiple: Some(8),
            ..Default::default()
        },
        postprocess: PostprocessConfig::default(),
        ..Default::default()
    };

    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(2);

    // Run upscale
    let result = upscaler.upscale(&frame).unwrap();

    // Verify output
    assert_eq!(result.width, 128);
    assert_eq!(result.height, 128);
    assert_eq!(result.data.len(), 128 * 128 * 3);

    // Check values are in valid range
    for &v in &result.data {
        assert!(v >= 0.0 && v <= 1.0, "Value out of range: {}", v);
    }
}

#[test]
fn test_yuv_to_rgb_and_back() {
    let mut frame = NeuralFrame::new(8, 8);

    // Fill with known RGB values
    for y in 0..8 {
        for x in 0..8 {
            let idx = ((y * 8 + x) * 3) as usize;
            frame.data[idx] = 0.5;     // R
            frame.data[idx + 1] = 0.3; // G
            frame.data[idx + 2] = 0.8; // B
        }
    }

    let (y, u, v) = frame.to_yuv420();
    let restored = NeuralFrame::from_yuv420(&y, &u, &v, 8, 8);

    // YUV conversion is lossy, but should be close
    assert_eq!(restored.width, 8);
    assert_eq!(restored.height, 8);
}

#[test]
fn test_frame_to_bytes() {
    let mut frame = NeuralFrame::new(2, 2);
    frame.data = vec![
        0.0, 0.5, 1.0,  // Pixel 0
        0.25, 0.75, 0.0, // Pixel 1
        1.0, 0.0, 0.5,  // Pixel 2
        0.5, 0.5, 0.5,  // Pixel 3
    ];

    let bytes = frame.to_bytes();

    assert_eq!(bytes.len(), 12);
    assert_eq!(bytes[0], 0);   // 0.0 -> 0
    assert_eq!(bytes[1], 128); // 0.5 -> 128
    assert_eq!(bytes[2], 255); // 1.0 -> 255
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_single_pixel_frame() {
    let frame = NeuralFrame::new(1, 1);

    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 4,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(4);

    let result = upscaler.upscale(&frame).unwrap();

    assert_eq!(result.width, 4);
    assert_eq!(result.height, 4);
}

#[test]
fn test_non_square_frame() {
    let frame = NeuralFrame::new(100, 50);

    let config = OnnxUpscalerConfig {
        neural: NeuralConfig {
            scale: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(2);

    let result = upscaler.upscale(&frame).unwrap();

    assert_eq!(result.width, 200);
    assert_eq!(result.height, 100);
}

#[test]
fn test_empty_batch() {
    let config = OnnxUpscalerConfig::default();
    let mut upscaler = OnnxUpscaler::new(config);
    upscaler.enable_mock(2);

    let empty: &[&NeuralFrame] = &[];
    let results = upscaler.upscale_batch(empty).unwrap();

    assert!(results.is_empty());
}
