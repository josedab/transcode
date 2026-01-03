//! Neural upscaling implementations.
//!
//! This module provides:
//! - CPU-based upscaling algorithms (nearest, bilinear, bicubic, lanczos)
//! - ONNX-based neural upscaling when feature enabled
//! - Tile-based processing for large images
//! - Progress reporting
//! - Memory-efficient streaming

use crate::{NeuralConfig, NeuralError, NeuralFrame, Result};
use crate::inference::{InferenceConfig, InputTensor, MockInference};
use crate::preprocessing::{PreprocessConfig, Preprocessor};
use crate::postprocessing::{PostprocessConfig, Postprocessor};
use crate::tiled::{TileConfig, TiledProcessor};

#[cfg(feature = "onnx")]
use crate::inference::{ExecutionProvider, InferenceSession};
#[cfg(feature = "onnx")]
use std::sync::Arc;

use std::path::Path;

/// Upscaling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleAlgorithm {
    /// Nearest neighbor (fastest, lowest quality).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos resampling.
    Lanczos,
    /// Neural network upscaling.
    #[default]
    Neural,
}

/// Progress callback type.
pub type ProgressCallback = Box<dyn Fn(f32, &str) + Send + Sync>;

/// CPU-based upscaler for fallback.
pub struct CpuUpscaler;

impl CpuUpscaler {
    /// Upscale using nearest neighbor.
    pub fn nearest(frame: &NeuralFrame, new_width: u32, new_height: u32) -> Result<NeuralFrame> {
        let mut output = NeuralFrame::new(new_width, new_height);

        let x_ratio = frame.width as f32 / new_width as f32;
        let y_ratio = frame.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as u32;
                let src_y = (y as f32 * y_ratio) as u32;

                let src_x = src_x.min(frame.width - 1);
                let src_y = src_y.min(frame.height - 1);

                for c in 0..3 {
                    let src_idx = ((src_y * frame.width + src_x) * 3 + c) as usize;
                    let dst_idx = ((y * new_width + x) * 3 + c) as usize;
                    output.data[dst_idx] = frame.data[src_idx];
                }
            }
        }

        Ok(output)
    }

    /// Upscale using bilinear interpolation.
    pub fn bilinear(frame: &NeuralFrame, new_width: u32, new_height: u32) -> Result<NeuralFrame> {
        let mut output = NeuralFrame::new(new_width, new_height);

        let x_ratio = frame.width as f32 / new_width as f32;
        let y_ratio = frame.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
                let src_y = (y as f32 + 0.5) * y_ratio - 0.5;

                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                for c in 0..3 {
                    let p00 = Self::sample(frame, x0, y0, c);
                    let p01 = Self::sample(frame, x1, y0, c);
                    let p10 = Self::sample(frame, x0, y1, c);
                    let p11 = Self::sample(frame, x1, y1, c);

                    let value = p00 * (1.0 - dx) * (1.0 - dy)
                        + p01 * dx * (1.0 - dy)
                        + p10 * (1.0 - dx) * dy
                        + p11 * dx * dy;

                    let dst_idx = ((y * new_width + x) * 3 + c) as usize;
                    output.data[dst_idx] = value.clamp(0.0, 1.0);
                }
            }
        }

        Ok(output)
    }

    /// Upscale using bicubic interpolation.
    pub fn bicubic(frame: &NeuralFrame, new_width: u32, new_height: u32) -> Result<NeuralFrame> {
        let mut output = NeuralFrame::new(new_width, new_height);

        let x_ratio = frame.width as f32 / new_width as f32;
        let y_ratio = frame.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
                let src_y = (y as f32 + 0.5) * y_ratio - 0.5;

                for c in 0..3 {
                    let value = Self::bicubic_sample(frame, src_x, src_y, c);
                    let dst_idx = ((y * new_width + x) * 3 + c) as usize;
                    output.data[dst_idx] = value.clamp(0.0, 1.0);
                }
            }
        }

        Ok(output)
    }

    /// Upscale using Lanczos resampling.
    pub fn lanczos(
        frame: &NeuralFrame,
        new_width: u32,
        new_height: u32,
        a: i32,
    ) -> Result<NeuralFrame> {
        let mut output = NeuralFrame::new(new_width, new_height);

        let x_ratio = frame.width as f32 / new_width as f32;
        let y_ratio = frame.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
                let src_y = (y as f32 + 0.5) * y_ratio - 0.5;

                for c in 0..3 {
                    let value = Self::lanczos_sample(frame, src_x, src_y, c, a);
                    let dst_idx = ((y * new_width + x) * 3 + c) as usize;
                    output.data[dst_idx] = value.clamp(0.0, 1.0);
                }
            }
        }

        Ok(output)
    }

    fn sample(frame: &NeuralFrame, x: i32, y: i32, c: u32) -> f32 {
        let x = x.clamp(0, frame.width as i32 - 1) as u32;
        let y = y.clamp(0, frame.height as i32 - 1) as u32;
        let idx = ((y * frame.width + x) * 3 + c) as usize;
        frame.data[idx]
    }

    fn bicubic_sample(frame: &NeuralFrame, x: f32, y: f32, c: u32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let mut result = 0.0;

        for j in -1..=2 {
            for i in -1..=2 {
                let px = (x0 + i).clamp(0, frame.width as i32 - 1);
                let py = (y0 + j).clamp(0, frame.height as i32 - 1);

                let idx = ((py as u32 * frame.width + px as u32) * 3 + c) as usize;
                let value = frame.data.get(idx).copied().unwrap_or(0.0);

                let wx = Self::cubic_weight(i as f32 - dx);
                let wy = Self::cubic_weight(j as f32 - dy);

                result += value * wx * wy;
            }
        }

        result
    }

    fn cubic_weight(t: f32) -> f32 {
        let t = t.abs();
        if t <= 1.0 {
            (1.5 * t - 2.5) * t * t + 1.0
        } else if t < 2.0 {
            ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
        } else {
            0.0
        }
    }

    fn lanczos_sample(frame: &NeuralFrame, x: f32, y: f32, c: u32, a: i32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let mut result = 0.0;
        let mut weight_sum = 0.0;

        for j in (1 - a)..=a {
            for i in (1 - a)..=a {
                let px = (x0 + i).clamp(0, frame.width as i32 - 1);
                let py = (y0 + j).clamp(0, frame.height as i32 - 1);

                let idx = ((py as u32 * frame.width + px as u32) * 3 + c) as usize;
                let value = frame.data.get(idx).copied().unwrap_or(0.0);

                let wx = Self::lanczos_kernel(i as f32 - dx, a);
                let wy = Self::lanczos_kernel(j as f32 - dy, a);
                let weight = wx * wy;

                result += value * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            result / weight_sum
        } else {
            0.0
        }
    }

    fn lanczos_kernel(x: f32, a: i32) -> f32 {
        if x.abs() < 1e-6 {
            return 1.0;
        }
        if x.abs() >= a as f32 {
            return 0.0;
        }

        let pi_x = std::f32::consts::PI * x;
        let pi_x_a = pi_x / a as f32;

        (pi_x.sin() / pi_x) * (pi_x_a.sin() / pi_x_a)
    }
}

/// Configuration for the ONNX upscaler.
#[derive(Debug, Clone)]
pub struct OnnxUpscalerConfig {
    /// Neural processing configuration.
    pub neural: NeuralConfig,
    /// Preprocessing configuration.
    pub preprocess: PreprocessConfig,
    /// Postprocessing configuration.
    pub postprocess: PostprocessConfig,
    /// Inference configuration.
    pub inference: InferenceConfig,
    /// Use tiled processing for large images.
    pub use_tiling: bool,
    /// Maximum image dimension before tiling is required.
    pub max_dimension: u32,
}

impl Default for OnnxUpscalerConfig {
    fn default() -> Self {
        Self {
            neural: NeuralConfig::default(),
            preprocess: PreprocessConfig::default(),
            postprocess: PostprocessConfig::default(),
            inference: InferenceConfig::default(),
            use_tiling: true,
            max_dimension: 1920,
        }
    }
}

/// ONNX-based neural upscaler.
pub struct OnnxUpscaler {
    config: OnnxUpscalerConfig,
    #[cfg(feature = "onnx")]
    session: Option<Arc<InferenceSession>>,
    preprocessor: Preprocessor,
    postprocessor: Postprocessor,
    tiled_processor: TiledProcessor,
    mock_inference: Option<MockInference>,
    progress_callback: Option<ProgressCallback>,
}

impl OnnxUpscaler {
    /// Create a new ONNX upscaler.
    pub fn new(config: OnnxUpscalerConfig) -> Self {
        let preprocessor = Preprocessor::new(config.preprocess.clone());
        let postprocessor = Postprocessor::new(config.postprocess.clone());
        let tile_config = TileConfig::new(
            config.neural.tile_size,
            config.neural.tile_overlap,
            config.neural.scale,
        );
        let tiled_processor = TiledProcessor::new(tile_config);

        Self {
            config,
            #[cfg(feature = "onnx")]
            session: None,
            preprocessor,
            postprocessor,
            tiled_processor,
            mock_inference: None,
            progress_callback: None,
        }
    }

    /// Set progress callback.
    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(f32, &str) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Load model from file.
    #[cfg(feature = "onnx")]
    pub fn load_model(&mut self, path: &Path) -> Result<()> {
        tracing::info!("Loading ONNX model from {:?}", path);

        let inference_config = InferenceConfig {
            execution_provider: if self.config.neural.use_gpu {
                ExecutionProvider::Cuda
            } else {
                ExecutionProvider::Cpu
            },
            device_id: self.config.neural.device_id,
            ..self.config.inference.clone()
        };

        let session = InferenceSession::from_file(path, inference_config)?;
        self.session = Some(Arc::new(session));

        tracing::info!("Model loaded successfully");
        Ok(())
    }

    /// Load model (stub when ONNX disabled).
    #[cfg(not(feature = "onnx"))]
    pub fn load_model(&mut self, _path: &Path) -> Result<()> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Enable mock inference for testing.
    pub fn enable_mock(&mut self, scale: u32) {
        self.mock_inference = Some(MockInference::new(scale));
    }

    /// Check if a model is loaded.
    #[cfg(feature = "onnx")]
    pub fn is_model_loaded(&self) -> bool {
        self.session.is_some()
    }

    #[cfg(not(feature = "onnx"))]
    pub fn is_model_loaded(&self) -> bool {
        false
    }

    /// Upscale a single frame.
    pub fn upscale(&self, frame: &NeuralFrame) -> Result<NeuralFrame> {
        self.report_progress(0.0, "Starting upscale");

        // Check if we need tiling
        let needs_tiling = self.config.use_tiling
            && (frame.width > self.config.max_dimension
                || frame.height > self.config.max_dimension);

        if needs_tiling {
            self.upscale_tiled(frame)
        } else {
            self.upscale_single(frame)
        }
    }

    /// Upscale a single tile/frame without tiling.
    fn upscale_single(&self, frame: &NeuralFrame) -> Result<NeuralFrame> {
        self.report_progress(0.1, "Preprocessing");

        // Preprocess
        let preprocessed = self.preprocessor.process(frame)?;

        self.report_progress(0.2, "Running inference");

        // Run inference
        let output = self.run_inference(&preprocessed)?;

        self.report_progress(0.9, "Postprocessing");

        // Postprocess
        let scale = self.config.neural.scale;
        let out_height = preprocessed.height * scale;
        let out_width = preprocessed.width * scale;

        let result = self.postprocessor.process_single(
            &output,
            out_height as usize,
            out_width as usize,
            Some(&preprocessed),
            scale,
        )?;

        self.report_progress(1.0, "Complete");

        Ok(result)
    }

    /// Upscale using tile-based processing.
    fn upscale_tiled(&self, frame: &NeuralFrame) -> Result<NeuralFrame> {
        self.report_progress(0.0, "Splitting into tiles");

        let tiles = self.tiled_processor.split(frame);
        let total_tiles = tiles.len();
        let mut processed_tiles = Vec::with_capacity(total_tiles);

        for (i, tile) in tiles.iter().enumerate() {
            let progress = i as f32 / total_tiles as f32;
            self.report_progress(
                0.1 + progress * 0.8,
                &format!("Processing tile {}/{}", i + 1, total_tiles),
            );

            // Convert tile to NeuralFrame
            let tile_frame = NeuralFrame {
                data: tile.data.clone(),
                width: tile.width,
                height: tile.height,
                channels: 3,
            };

            // Process tile
            let upscaled = self.upscale_single(&tile_frame)?;

            // Convert back to Tile with scaled dimensions
            let scale = self.config.neural.scale;
            processed_tiles.push(crate::tiled::Tile {
                data: upscaled.data,
                width: tile.width * scale,
                height: tile.height * scale,
                src_x: tile.src_x,
                src_y: tile.src_y,
                tile_x: tile.tile_x,
                tile_y: tile.tile_y,
            });
        }

        self.report_progress(0.95, "Merging tiles");

        let result = self.tiled_processor.merge(&processed_tiles, frame.width, frame.height)?;

        self.report_progress(1.0, "Complete");

        Ok(result)
    }

    /// Run inference on preprocessed input.
    fn run_inference(&self, preprocessed: &crate::preprocessing::PreprocessedInput) -> Result<Vec<f32>> {
        let input_tensor = InputTensor::from_hwc(
            &preprocessed.data,
            preprocessed.height as usize,
            preprocessed.width as usize,
            3,
        );

        // Try mock inference first (for testing)
        if let Some(ref mock) = self.mock_inference {
            let output = mock.run(&input_tensor)?;
            return Ok(output.data);
        }

        // Try real ONNX inference
        #[cfg(feature = "onnx")]
        {
            if let Some(ref session) = self.session {
                let output = session.run(&input_tensor)?;
                return Ok(output.data);
            }
        }

        // Fallback to CPU bicubic
        tracing::warn!("No model loaded, falling back to bicubic upscale");
        let scale = self.config.neural.scale;
        let new_width = preprocessed.width * scale;
        let new_height = preprocessed.height * scale;

        let frame = NeuralFrame::from_rgb(
            preprocessed.data.clone(),
            preprocessed.width,
            preprocessed.height,
        )?;

        let upscaled = CpuUpscaler::bicubic(&frame, new_width, new_height)?;

        // Convert to NCHW format for consistency
        let nchw = crate::preprocessing::PreprocessedInput {
            data: upscaled.data,
            width: new_width,
            height: new_height,
            original_width: preprocessed.original_width * scale,
            original_height: preprocessed.original_height * scale,
            pad_right: preprocessed.pad_right * scale,
            pad_bottom: preprocessed.pad_bottom * scale,
        };

        Ok(nchw.to_nchw())
    }

    /// Upscale a batch of frames.
    pub fn upscale_batch(&self, frames: &[&NeuralFrame]) -> Result<Vec<NeuralFrame>> {
        let total = frames.len();
        let mut results = Vec::with_capacity(total);

        for (i, frame) in frames.iter().enumerate() {
            self.report_progress(
                i as f32 / total as f32,
                &format!("Processing frame {}/{}", i + 1, total),
            );
            results.push(self.upscale(frame)?);
        }

        self.report_progress(1.0, "Batch complete");
        Ok(results)
    }

    /// Report progress via callback.
    fn report_progress(&self, progress: f32, message: &str) {
        if let Some(ref callback) = self.progress_callback {
            callback(progress, message);
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &OnnxUpscalerConfig {
        &self.config
    }

    /// Get the scale factor.
    pub fn scale(&self) -> u32 {
        self.config.neural.scale
    }
}

/// Stream-based upscaler for memory efficiency.
pub struct StreamingUpscaler {
    upscaler: OnnxUpscaler,
    buffer: Vec<NeuralFrame>,
    buffer_size: usize,
}

impl StreamingUpscaler {
    /// Create a new streaming upscaler.
    pub fn new(upscaler: OnnxUpscaler, buffer_size: usize) -> Self {
        Self {
            upscaler,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Push a frame for processing.
    pub fn push(&mut self, frame: NeuralFrame) -> Result<Option<Vec<NeuralFrame>>> {
        self.buffer.push(frame);

        if self.buffer.len() >= self.buffer_size {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Flush the buffer and process all frames.
    pub fn flush(&mut self) -> Result<Option<Vec<NeuralFrame>>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let frames: Vec<&NeuralFrame> = self.buffer.iter().collect();
        let results = self.upscaler.upscale_batch(&frames)?;
        self.buffer.clear();

        Ok(Some(results))
    }

    /// Get remaining frames in buffer.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_upscaler_nearest() {
        let frame = NeuralFrame::new(4, 4);
        let result = CpuUpscaler::nearest(&frame, 8, 8).unwrap();

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_cpu_upscaler_bilinear() {
        let mut frame = NeuralFrame::new(4, 4);
        // Fill with gradient
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    let idx = ((y * 4 + x) * 3 + c) as usize;
                    frame.data[idx] = (x + y) as f32 / 8.0;
                }
            }
        }

        let result = CpuUpscaler::bilinear(&frame, 8, 8).unwrap();

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_cpu_upscaler_bicubic() {
        let frame = NeuralFrame::new(4, 4);
        let result = CpuUpscaler::bicubic(&frame, 8, 8).unwrap();

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_cpu_upscaler_lanczos() {
        let frame = NeuralFrame::new(4, 4);
        let result = CpuUpscaler::lanczos(&frame, 8, 8, 3).unwrap();

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_onnx_upscaler_mock() {
        let config = OnnxUpscalerConfig {
            neural: NeuralConfig {
                scale: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut upscaler = OnnxUpscaler::new(config);
        upscaler.enable_mock(2);

        let frame = NeuralFrame::new(16, 16);
        let result = upscaler.upscale(&frame).unwrap();

        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
    }

    #[test]
    fn test_onnx_upscaler_fallback() {
        let config = OnnxUpscalerConfig {
            neural: NeuralConfig {
                scale: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let upscaler = OnnxUpscaler::new(config);

        // Without model or mock, should fall back to bicubic
        let frame = NeuralFrame::new(8, 8);
        let result = upscaler.upscale(&frame).unwrap();

        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
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

        let mut streamer = StreamingUpscaler::new(onnx_upscaler, 2);

        // Push first frame
        let frame1 = NeuralFrame::new(8, 8);
        let result = streamer.push(frame1).unwrap();
        assert!(result.is_none()); // Buffer not full
        assert_eq!(streamer.pending(), 1);

        // Push second frame - should flush
        let frame2 = NeuralFrame::new(8, 8);
        let result = streamer.push(frame2).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
        assert_eq!(streamer.pending(), 0);
    }

    #[test]
    fn test_progress_callback() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = OnnxUpscalerConfig::default();
        let progress_count = Arc::new(AtomicU32::new(0));
        let progress_clone = progress_count.clone();

        let mut upscaler = OnnxUpscaler::new(config)
            .with_progress(move |_progress, _msg| {
                progress_clone.fetch_add(1, Ordering::SeqCst);
            });
        upscaler.enable_mock(2);

        let frame = NeuralFrame::new(8, 8);
        upscaler.upscale(&frame).unwrap();

        // Should have received multiple progress updates
        assert!(progress_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_upscale_algorithm_default() {
        let algo = UpscaleAlgorithm::default();
        assert_eq!(algo, UpscaleAlgorithm::Neural);
    }
}
