//! ONNX Runtime inference engine.
//!
//! This module provides the core ONNX inference functionality including:
//! - Runtime environment initialization
//! - Session creation with multiple execution providers
//! - Input tensor preparation (NCHW layout)
//! - Inference execution
//! - Output tensor extraction
//! - Batch processing support

use crate::{NeuralError, Result};

/// Execution provider for ONNX inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU execution (default).
    #[default]
    Cpu,
    /// CUDA GPU execution.
    Cuda,
    /// TensorRT optimized execution.
    TensorRT,
    /// DirectML (Windows).
    DirectML,
    /// CoreML (macOS/iOS).
    CoreML,
}

impl ExecutionProvider {
    /// Get the name of this execution provider.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::TensorRT => "TensorRT",
            Self::DirectML => "DirectML",
            Self::CoreML => "CoreML",
        }
    }
}

/// Configuration for ONNX inference.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Execution provider to use.
    pub execution_provider: ExecutionProvider,
    /// GPU device ID (for CUDA/TensorRT).
    pub device_id: u32,
    /// Number of intra-op threads.
    pub intra_op_threads: usize,
    /// Number of inter-op threads.
    pub inter_op_threads: usize,
    /// Enable memory pattern optimization.
    pub memory_pattern: bool,
    /// Enable model optimization.
    pub optimization_level: OptimizationLevel,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            device_id: 0,
            intra_op_threads: 0, // Use default (all cores)
            inter_op_threads: 0,
            memory_pattern: true,
            optimization_level: OptimizationLevel::All,
        }
    }
}

/// Graph optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization.
    None,
    /// Basic optimizations.
    Basic,
    /// Extended optimizations.
    Extended,
    /// All optimizations.
    #[default]
    All,
}

/// Input tensor for ONNX inference.
#[derive(Debug, Clone)]
pub struct InputTensor {
    /// Tensor data in NCHW layout.
    pub data: Vec<f32>,
    /// Batch size (N).
    pub batch: usize,
    /// Number of channels (C).
    pub channels: usize,
    /// Height (H).
    pub height: usize,
    /// Width (W).
    pub width: usize,
}

impl InputTensor {
    /// Create a new input tensor.
    pub fn new(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        let size = batch * channels * height * width;
        Self {
            data: vec![0.0; size],
            batch,
            channels,
            height,
            width,
        }
    }

    /// Create from HWC data, converting to NCHW.
    pub fn from_hwc(data: &[f32], height: usize, width: usize, channels: usize) -> Self {
        let batch = 1;
        let size = batch * channels * height * width;
        let mut nchw_data = vec![0.0; size];

        // Convert HWC to NCHW
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let hwc_idx = h * width * channels + w * channels + c;
                    let nchw_idx = c * height * width + h * width + w;
                    if hwc_idx < data.len() {
                        nchw_data[nchw_idx] = data[hwc_idx];
                    }
                }
            }
        }

        Self {
            data: nchw_data,
            batch,
            channels,
            height,
            width,
        }
    }

    /// Create batch from multiple HWC frames.
    pub fn from_batch_hwc(
        frames: &[&[f32]],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Self {
        let batch = frames.len();
        let size = batch * channels * height * width;
        let mut nchw_data = vec![0.0; size];

        for (b, frame) in frames.iter().enumerate() {
            let batch_offset = b * channels * height * width;
            for h in 0..height {
                for w in 0..width {
                    for c in 0..channels {
                        let hwc_idx = h * width * channels + w * channels + c;
                        let nchw_idx = batch_offset + c * height * width + h * width + w;
                        if hwc_idx < frame.len() {
                            nchw_data[nchw_idx] = frame[hwc_idx];
                        }
                    }
                }
            }
        }

        Self {
            data: nchw_data,
            batch,
            channels,
            height,
            width,
        }
    }

    /// Get the shape as [N, C, H, W].
    pub fn shape(&self) -> [usize; 4] {
        [self.batch, self.channels, self.height, self.width]
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Output tensor from ONNX inference.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    /// Tensor data in NCHW layout.
    pub data: Vec<f32>,
    /// Shape [N, C, H, W] or dynamic.
    pub shape: Vec<usize>,
}

impl OutputTensor {
    /// Create a new output tensor.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Convert to HWC format (for single batch).
    pub fn to_hwc(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(NeuralError::InvalidInput(
                "Expected 4D tensor (NCHW)".to_string(),
            ));
        }

        let [batch, channels, height, width] = [
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        ];

        if batch != 1 {
            return Err(NeuralError::InvalidInput(
                "to_hwc only supports batch size 1".to_string(),
            ));
        }

        let size = height * width * channels;
        let mut hwc_data = vec![0.0; size];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let nchw_idx = c * height * width + h * width + w;
                    let hwc_idx = h * width * channels + w * channels + c;
                    if nchw_idx < self.data.len() {
                        hwc_data[hwc_idx] = self.data[nchw_idx];
                    }
                }
            }
        }

        Ok(hwc_data)
    }

    /// Split batch into individual HWC tensors.
    pub fn to_batch_hwc(&self) -> Result<Vec<Vec<f32>>> {
        if self.shape.len() != 4 {
            return Err(NeuralError::InvalidInput(
                "Expected 4D tensor (NCHW)".to_string(),
            ));
        }

        let [batch, channels, height, width] = [
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        ];

        let frame_size = height * width * channels;
        let batch_stride = channels * height * width;
        let mut result = Vec::with_capacity(batch);

        for b in 0..batch {
            let mut hwc_data = vec![0.0; frame_size];
            let batch_offset = b * batch_stride;

            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let nchw_idx = batch_offset + c * height * width + h * width + w;
                        let hwc_idx = h * width * channels + w * channels + c;
                        if nchw_idx < self.data.len() {
                            hwc_data[hwc_idx] = self.data[nchw_idx];
                        }
                    }
                }
            }

            result.push(hwc_data);
        }

        Ok(result)
    }

    /// Get dimensions [height, width] for NCHW tensor.
    pub fn dimensions(&self) -> Option<(usize, usize)> {
        if self.shape.len() == 4 {
            Some((self.shape[2], self.shape[3]))
        } else {
            None
        }
    }
}

/// ONNX inference session.
#[cfg(feature = "onnx")]
pub struct InferenceSession {
    session: ort::session::Session,
    config: InferenceConfig,
}

#[cfg(feature = "onnx")]
impl InferenceSession {
    /// Create a new inference session from a model file.
    pub fn from_file(path: &std::path::Path, config: InferenceConfig) -> Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;

        tracing::info!("Loading ONNX model from {:?}", path);
        tracing::debug!("Execution provider: {:?}", config.execution_provider);

        let opt_level = match config.optimization_level {
            OptimizationLevel::None => GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OptimizationLevel::All => GraphOptimizationLevel::Level3,
        };

        let mut builder = Session::builder()
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(opt_level)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to set optimization level: {}", e)))?;

        // Configure threads
        if config.intra_op_threads > 0 {
            builder = builder
                .with_intra_threads(config.intra_op_threads)
                .map_err(|e| NeuralError::ModelLoad(format!("Failed to set intra threads: {}", e)))?;
        }

        if config.inter_op_threads > 0 {
            builder = builder
                .with_inter_threads(config.inter_op_threads)
                .map_err(|e| NeuralError::ModelLoad(format!("Failed to set inter threads: {}", e)))?;
        }

        // Memory pattern
        if config.memory_pattern {
            builder = builder
                .with_memory_pattern(true)
                .map_err(|e| NeuralError::ModelLoad(format!("Failed to enable memory pattern: {}", e)))?;
        }

        // Load the model
        let session = builder
            .commit_from_file(path)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to load model: {}", e)))?;

        tracing::info!("Model loaded successfully");
        Self::log_model_info(&session);

        Ok(Self { session, config })
    }

    /// Create from model bytes.
    pub fn from_bytes(data: &[u8], config: InferenceConfig) -> Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;

        let opt_level = match config.optimization_level {
            OptimizationLevel::None => GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OptimizationLevel::All => GraphOptimizationLevel::Level3,
        };

        let builder = Session::builder()
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(opt_level)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to set optimization level: {}", e)))?;

        let session = builder
            .commit_from_memory(data)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to load model from memory: {}", e)))?;

        Ok(Self { session, config })
    }

    /// Log model input/output information.
    fn log_model_info(session: &ort::session::Session) {
        tracing::debug!("Model inputs:");
        for input in session.inputs.iter() {
            tracing::debug!("  - {}: {:?}", input.name, input.input_type);
        }
        tracing::debug!("Model outputs:");
        for output in session.outputs.iter() {
            tracing::debug!("  - {}: {:?}", output.name, output.output_type);
        }
    }

    /// Get the input shape expected by the model.
    pub fn input_shape(&self) -> Option<Vec<Option<i64>>> {
        self.session.inputs.first().and_then(|input| {
            match &input.input_type {
                ort::session::input::Input::Tensor { dimensions, .. } => {
                    Some(dimensions.clone())
                }
            }
        })
    }

    /// Get the output shape of the model.
    pub fn output_shape(&self) -> Option<Vec<Option<i64>>> {
        self.session.outputs.first().and_then(|output| {
            match &output.output_type {
                ort::session::output::Output::Tensor { dimensions, .. } => {
                    Some(dimensions.clone())
                }
            }
        })
    }

    /// Run inference on a single input tensor.
    pub fn run(&self, input: &InputTensor) -> Result<OutputTensor> {
        use ndarray::Array4;
        use ort::value::TensorRef;

        let shape = input.shape();
        let array = Array4::from_shape_vec(
            (shape[0], shape[1], shape[2], shape[3]),
            input.data.clone(),
        )
        .map_err(|e| NeuralError::Inference(format!("Failed to create input array: {}", e)))?;

        let tensor = TensorRef::from_array_view(array.view())
            .map_err(|e| NeuralError::Inference(format!("Failed to create tensor: {}", e)))?;

        let input_name = self
            .session
            .inputs
            .first()
            .map(|i| i.name.as_str())
            .unwrap_or("input");

        let outputs = self
            .session
            .run(ort::inputs![input_name => tensor].map_err(|e| {
                NeuralError::Inference(format!("Failed to create inputs: {}", e))
            })?)
            .map_err(|e| NeuralError::Inference(format!("Inference failed: {}", e)))?;

        // Extract first output
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| NeuralError::Inference("No output from model".to_string()))?;

        let (_name, value) = output;
        let tensor = value
            .try_extract_tensor::<f32>()
            .map_err(|e| NeuralError::Inference(format!("Failed to extract output: {}", e)))?;

        let shape_slice = tensor.shape();
        let shape: Vec<usize> = shape_slice.iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = tensor.iter().copied().collect();

        Ok(OutputTensor::new(data, shape))
    }

    /// Run batch inference.
    pub fn run_batch(&self, inputs: &[InputTensor]) -> Result<Vec<OutputTensor>> {
        inputs.iter().map(|input| self.run(input)).collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Mock inference session for testing without ONNX.
#[cfg(not(feature = "onnx"))]
pub struct InferenceSession {
    config: InferenceConfig,
}

#[cfg(not(feature = "onnx"))]
impl InferenceSession {
    /// Create a mock session (always fails without onnx feature).
    pub fn from_file(_path: &std::path::Path, _config: InferenceConfig) -> Result<Self> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Create a mock session from bytes.
    pub fn from_bytes(_data: &[u8], _config: InferenceConfig) -> Result<Self> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Get mock input shape.
    pub fn input_shape(&self) -> Option<Vec<Option<i64>>> {
        Some(vec![Some(1), Some(3), None, None])
    }

    /// Get mock output shape.
    pub fn output_shape(&self) -> Option<Vec<Option<i64>>> {
        Some(vec![Some(1), Some(3), None, None])
    }

    /// Mock run (returns error).
    pub fn run(&self, _input: &InputTensor) -> Result<OutputTensor> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Mock batch run.
    pub fn run_batch(&self, _inputs: &[InputTensor]) -> Result<Vec<OutputTensor>> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Get the configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Mock inference for testing.
pub struct MockInference {
    /// Scale factor for output.
    pub scale: u32,
}

impl MockInference {
    /// Create a new mock inference with given scale.
    pub fn new(scale: u32) -> Self {
        Self { scale }
    }

    /// Run mock inference (bilinear upscale).
    pub fn run(&self, input: &InputTensor) -> Result<OutputTensor> {
        let out_height = input.height * self.scale as usize;
        let out_width = input.width * self.scale as usize;
        let out_size = input.batch * input.channels * out_height * out_width;
        let mut output_data = vec![0.0f32; out_size];

        // Simple bilinear upscale in NCHW format
        let scale_f = self.scale as f32;
        for b in 0..input.batch {
            for c in 0..input.channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let src_h = oh as f32 / scale_f;
                        let src_w = ow as f32 / scale_f;

                        let h0 = src_h.floor() as usize;
                        let w0 = src_w.floor() as usize;
                        let h1 = (h0 + 1).min(input.height - 1);
                        let w1 = (w0 + 1).min(input.width - 1);

                        let dh = src_h - h0 as f32;
                        let dw = src_w - w0 as f32;

                        let idx00 = b * input.channels * input.height * input.width
                            + c * input.height * input.width
                            + h0 * input.width
                            + w0;
                        let idx01 = b * input.channels * input.height * input.width
                            + c * input.height * input.width
                            + h0 * input.width
                            + w1;
                        let idx10 = b * input.channels * input.height * input.width
                            + c * input.height * input.width
                            + h1 * input.width
                            + w0;
                        let idx11 = b * input.channels * input.height * input.width
                            + c * input.height * input.width
                            + h1 * input.width
                            + w1;

                        let v00 = input.data.get(idx00).copied().unwrap_or(0.0);
                        let v01 = input.data.get(idx01).copied().unwrap_or(0.0);
                        let v10 = input.data.get(idx10).copied().unwrap_or(0.0);
                        let v11 = input.data.get(idx11).copied().unwrap_or(0.0);

                        let value = v00 * (1.0 - dh) * (1.0 - dw)
                            + v01 * (1.0 - dh) * dw
                            + v10 * dh * (1.0 - dw)
                            + v11 * dh * dw;

                        let out_idx = b * input.channels * out_height * out_width
                            + c * out_height * out_width
                            + oh * out_width
                            + ow;
                        output_data[out_idx] = value;
                    }
                }
            }
        }

        Ok(OutputTensor::new(
            output_data,
            vec![input.batch, input.channels, out_height, out_width],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_tensor_from_hwc() {
        // 2x2 RGB image in HWC format
        let hwc_data = vec![
            1.0, 2.0, 3.0, // (0,0) RGB
            4.0, 5.0, 6.0, // (0,1) RGB
            7.0, 8.0, 9.0, // (1,0) RGB
            10.0, 11.0, 12.0, // (1,1) RGB
        ];

        let tensor = InputTensor::from_hwc(&hwc_data, 2, 2, 3);

        assert_eq!(tensor.batch, 1);
        assert_eq!(tensor.channels, 3);
        assert_eq!(tensor.height, 2);
        assert_eq!(tensor.width, 2);

        // Check NCHW layout: C=0 should have [1, 4, 7, 10]
        assert_eq!(tensor.data[0], 1.0); // R(0,0)
        assert_eq!(tensor.data[1], 4.0); // R(0,1)
        assert_eq!(tensor.data[2], 7.0); // R(1,0)
        assert_eq!(tensor.data[3], 10.0); // R(1,1)
    }

    #[test]
    fn test_output_tensor_to_hwc() {
        // 1x3x2x2 NCHW tensor
        let nchw_data = vec![
            1.0, 4.0, 7.0, 10.0, // R channel
            2.0, 5.0, 8.0, 11.0, // G channel
            3.0, 6.0, 9.0, 12.0, // B channel
        ];

        let tensor = OutputTensor::new(nchw_data, vec![1, 3, 2, 2]);
        let hwc = tensor.to_hwc().unwrap();

        // Expected HWC: [(0,0) RGB, (0,1) RGB, (1,0) RGB, (1,1) RGB]
        assert_eq!(hwc[0], 1.0); // R(0,0)
        assert_eq!(hwc[1], 2.0); // G(0,0)
        assert_eq!(hwc[2], 3.0); // B(0,0)
        assert_eq!(hwc[3], 4.0); // R(0,1)
        assert_eq!(hwc[4], 5.0); // G(0,1)
        assert_eq!(hwc[5], 6.0); // B(0,1)
    }

    #[test]
    fn test_hwc_nchw_roundtrip() {
        let original = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];

        let tensor = InputTensor::from_hwc(&original, 2, 2, 3);
        let output = OutputTensor::new(tensor.data.clone(), vec![1, 3, 2, 2]);
        let recovered = output.to_hwc().unwrap();

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mock_inference() {
        let mock = MockInference::new(2);
        let input = InputTensor::new(1, 3, 4, 4);

        let output = mock.run(&input).unwrap();

        assert_eq!(output.shape, vec![1, 3, 8, 8]);
        assert_eq!(output.data.len(), 1 * 3 * 8 * 8);
    }

    #[test]
    fn test_batch_input_tensor() {
        let frame1 = vec![1.0; 12]; // 2x2x3
        let frame2 = vec![2.0; 12];

        let tensor = InputTensor::from_batch_hwc(&[&frame1, &frame2], 2, 2, 3);

        assert_eq!(tensor.batch, 2);
        assert_eq!(tensor.shape(), [2, 3, 2, 2]);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(config.device_id, 0);
    }
}
