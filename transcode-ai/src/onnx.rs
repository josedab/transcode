//! ONNX Runtime integration for AI model inference.
//!
//! This module provides real ONNX Runtime integration when the `onnx` feature is enabled.
//! It supports multiple execution providers for hardware acceleration:
//!
//! - CPU (default)
//! - CUDA (NVIDIA GPUs)
//! - TensorRT (optimized NVIDIA inference)
//! - CoreML (Apple Silicon)
//! - DirectML (Windows GPU)
//!
//! # Example
//!
//! ```ignore
//! use transcode_ai::onnx::{OnnxSession, OnnxConfig, ExecutionProvider};
//!
//! let config = OnnxConfig::new()
//!     .with_provider(ExecutionProvider::Cuda { device_id: 0 })
//!     .with_optimization_level(OptimizationLevel::All);
//!
//! let session = OnnxSession::load("model.onnx", config)?;
//! let output = session.run(&input_tensor)?;
//! ```

use crate::error::{AiError, Result};
use std::path::Path;

#[cfg(feature = "onnx")]
use tracing::{debug, info, warn};

/// Execution provider for ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU execution (always available).
    #[default]
    Cpu,
    /// CUDA execution (NVIDIA GPUs).
    Cuda {
        /// GPU device ID.
        device_id: i32,
    },
    /// TensorRT execution (optimized NVIDIA inference).
    TensorRT {
        /// GPU device ID.
        device_id: i32,
        /// Enable FP16 inference.
        fp16: bool,
        /// Maximum workspace size in bytes.
        max_workspace_size: usize,
    },
    /// CoreML execution (Apple Silicon/Neural Engine).
    CoreML {
        /// Use Neural Engine when available.
        use_neural_engine: bool,
        /// Only run on Neural Engine (fail if unavailable).
        only_neural_engine: bool,
    },
    /// DirectML execution (Windows GPU via DirectX 12).
    DirectML {
        /// GPU device ID.
        device_id: i32,
    },
    /// OpenVINO execution (Intel hardware).
    OpenVINO {
        /// Device type (CPU, GPU, MYRIAD, etc.).
        device_type: String,
    },
    /// ROCm execution (AMD GPUs).
    ROCm {
        /// GPU device ID.
        device_id: i32,
    },
}

impl ExecutionProvider {
    /// Check if this provider is available on the current system.
    #[cfg(feature = "onnx")]
    pub fn is_available(&self) -> bool {
        use ort::ExecutionProviderDispatch;

        match self {
            Self::Cpu => true,
            Self::Cuda { .. } => ort::CUDAExecutionProvider::default().is_available().unwrap_or(false),
            Self::TensorRT { .. } => ort::TensorRTExecutionProvider::default().is_available().unwrap_or(false),
            Self::CoreML { .. } => {
                #[cfg(target_os = "macos")]
                {
                    ort::CoreMLExecutionProvider::default().is_available().unwrap_or(false)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            }
            Self::DirectML { .. } => {
                #[cfg(target_os = "windows")]
                {
                    ort::DirectMLExecutionProvider::default().is_available().unwrap_or(false)
                }
                #[cfg(not(target_os = "windows"))]
                {
                    false
                }
            }
            Self::OpenVINO { .. } => ort::OpenVINOExecutionProvider::default().is_available().unwrap_or(false),
            Self::ROCm { .. } => ort::ROCmExecutionProvider::default().is_available().unwrap_or(false),
        }
    }

    #[cfg(not(feature = "onnx"))]
    pub fn is_available(&self) -> bool {
        false
    }

    /// Get the best available provider for the current system.
    pub fn best_available() -> Self {
        // Try hardware acceleration first
        #[cfg(target_os = "macos")]
        {
            let coreml = Self::CoreML {
                use_neural_engine: true,
                only_neural_engine: false,
            };
            if coreml.is_available() {
                return coreml;
            }
        }

        #[cfg(target_os = "windows")]
        {
            let dml = Self::DirectML { device_id: 0 };
            if dml.is_available() {
                return dml;
            }
        }

        let cuda = Self::Cuda { device_id: 0 };
        if cuda.is_available() {
            return cuda;
        }

        Self::Cpu
    }

    /// Get a human-readable name for this provider.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda { .. } => "CUDA",
            Self::TensorRT { .. } => "TensorRT",
            Self::CoreML { .. } => "CoreML",
            Self::DirectML { .. } => "DirectML",
            Self::OpenVINO { .. } => "OpenVINO",
            Self::ROCm { .. } => "ROCm",
        }
    }
}

/// Graph optimization level for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization.
    None,
    /// Basic optimizations.
    Basic,
    /// Extended optimizations.
    Extended,
    /// All optimizations (default).
    #[default]
    All,
}

/// Configuration for ONNX Runtime session.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Execution providers in priority order.
    pub providers: Vec<ExecutionProvider>,
    /// Graph optimization level.
    pub optimization_level: OptimizationLevel,
    /// Number of intra-op threads (0 = auto).
    pub intra_op_threads: usize,
    /// Number of inter-op threads (0 = auto).
    pub inter_op_threads: usize,
    /// Enable memory pattern optimization.
    pub enable_memory_pattern: bool,
    /// Enable memory arena.
    pub enable_memory_arena: bool,
    /// Path to save optimized model (for caching).
    pub optimized_model_path: Option<String>,
    /// Enable profiling.
    pub enable_profiling: bool,
    /// Profiling output path.
    pub profile_path: Option<String>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            providers: vec![ExecutionProvider::best_available()],
            optimization_level: OptimizationLevel::All,
            intra_op_threads: 0,
            inter_op_threads: 0,
            enable_memory_pattern: true,
            enable_memory_arena: true,
            optimized_model_path: None,
            enable_profiling: false,
            profile_path: None,
        }
    }
}

impl OnnxConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an execution provider.
    pub fn with_provider(mut self, provider: ExecutionProvider) -> Self {
        self.providers.insert(0, provider);
        self
    }

    /// Set execution providers.
    pub fn with_providers(mut self, providers: Vec<ExecutionProvider>) -> Self {
        self.providers = providers;
        self
    }

    /// Set optimization level.
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set number of intra-op threads.
    pub fn with_intra_op_threads(mut self, threads: usize) -> Self {
        self.intra_op_threads = threads;
        self
    }

    /// Set number of inter-op threads.
    pub fn with_inter_op_threads(mut self, threads: usize) -> Self {
        self.inter_op_threads = threads;
        self
    }

    /// Enable or disable memory pattern optimization.
    pub fn with_memory_pattern(mut self, enable: bool) -> Self {
        self.enable_memory_pattern = enable;
        self
    }

    /// Set path for saving optimized model.
    pub fn with_optimized_model_path(mut self, path: impl Into<String>) -> Self {
        self.optimized_model_path = Some(path.into());
        self
    }

    /// Enable profiling.
    pub fn with_profiling(mut self, enable: bool, path: Option<String>) -> Self {
        self.enable_profiling = enable;
        self.profile_path = path;
        self
    }

    /// Create configuration optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            providers: vec![ExecutionProvider::best_available()],
            optimization_level: OptimizationLevel::All,
            intra_op_threads: 1,
            inter_op_threads: 1,
            enable_memory_pattern: false,
            enable_memory_arena: true,
            optimized_model_path: None,
            enable_profiling: false,
            profile_path: None,
        }
    }

    /// Create configuration optimized for throughput.
    pub fn high_throughput() -> Self {
        Self {
            providers: vec![ExecutionProvider::best_available()],
            optimization_level: OptimizationLevel::All,
            intra_op_threads: 0, // Auto
            inter_op_threads: 0, // Auto
            enable_memory_pattern: true,
            enable_memory_arena: true,
            optimized_model_path: None,
            enable_profiling: false,
            profile_path: None,
        }
    }
}

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point.
    Float16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    Uint8,
    /// 8-bit signed integer.
    Int8,
    /// Boolean.
    Bool,
}

/// Input/output tensor information.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor shape (negative values indicate dynamic dimensions).
    pub shape: Vec<i64>,
    /// Tensor data type.
    pub dtype: TensorType,
}

/// Model metadata.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model producer name.
    pub producer: Option<String>,
    /// Model producer version.
    pub producer_version: Option<String>,
    /// Model domain.
    pub domain: Option<String>,
    /// Model description.
    pub description: Option<String>,
    /// Model version.
    pub version: i64,
    /// Custom metadata.
    pub custom: std::collections::HashMap<String, String>,
}

/// ONNX Runtime session for model inference.
#[cfg(feature = "onnx")]
pub struct OnnxSession {
    /// The underlying ort session.
    session: ort::Session,
    /// Input tensor information.
    inputs: Vec<TensorInfo>,
    /// Output tensor information.
    outputs: Vec<TensorInfo>,
    /// Model metadata.
    metadata: ModelMetadata,
    /// Active execution provider.
    active_provider: ExecutionProvider,
}

#[cfg(feature = "onnx")]
impl OnnxSession {
    /// Load a model from a file path.
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(AiError::ModelNotFound(path.display().to_string()));
        }

        info!("Loading ONNX model from: {:?}", path);

        // Build session options
        let mut builder = ort::Session::builder()
            .map_err(|e| AiError::ModelLoadError(format!("Failed to create session builder: {}", e)))?;

        // Set optimization level
        let opt_level = match config.optimization_level {
            OptimizationLevel::None => ort::GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => ort::GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => ort::GraphOptimizationLevel::Level2,
            OptimizationLevel::All => ort::GraphOptimizationLevel::Level3,
        };
        builder = builder.with_optimization_level(opt_level)
            .map_err(|e| AiError::ModelLoadError(format!("Failed to set optimization level: {}", e)))?;

        // Set thread counts
        if config.intra_op_threads > 0 {
            builder = builder.with_intra_threads(config.intra_op_threads)
                .map_err(|e| AiError::ModelLoadError(format!("Failed to set intra threads: {}", e)))?;
        }
        if config.inter_op_threads > 0 {
            builder = builder.with_inter_threads(config.inter_op_threads)
                .map_err(|e| AiError::ModelLoadError(format!("Failed to set inter threads: {}", e)))?;
        }

        // Set memory options
        if config.enable_memory_pattern {
            builder = builder.with_memory_pattern(true)
                .map_err(|e| AiError::ModelLoadError(format!("Failed to enable memory pattern: {}", e)))?;
        }

        // Register execution providers
        let mut active_provider = ExecutionProvider::Cpu;
        for provider in &config.providers {
            match Self::register_provider(&mut builder, provider) {
                Ok(()) => {
                    debug!("Registered execution provider: {}", provider.name());
                    if active_provider == ExecutionProvider::Cpu {
                        active_provider = provider.clone();
                    }
                }
                Err(e) => {
                    warn!("Failed to register {}: {}", provider.name(), e);
                }
            }
        }

        info!("Using execution provider: {}", active_provider.name());

        // Load the model
        let session = builder.commit_from_file(path)
            .map_err(|e| AiError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        // Extract input/output information
        let inputs = Self::extract_inputs(&session)?;
        let outputs = Self::extract_outputs(&session)?;
        let metadata = Self::extract_metadata(&session);

        debug!("Model inputs: {:?}", inputs);
        debug!("Model outputs: {:?}", outputs);

        Ok(Self {
            session,
            inputs,
            outputs,
            metadata,
            active_provider,
        })
    }

    /// Load a model from memory.
    pub fn load_from_memory(data: &[u8], config: OnnxConfig) -> Result<Self> {
        info!("Loading ONNX model from memory ({} bytes)", data.len());

        // Build session options
        let mut builder = ort::Session::builder()
            .map_err(|e| AiError::ModelLoadError(format!("Failed to create session builder: {}", e)))?;

        // Set optimization level
        let opt_level = match config.optimization_level {
            OptimizationLevel::None => ort::GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => ort::GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => ort::GraphOptimizationLevel::Level2,
            OptimizationLevel::All => ort::GraphOptimizationLevel::Level3,
        };
        builder = builder.with_optimization_level(opt_level)
            .map_err(|e| AiError::ModelLoadError(format!("Failed to set optimization level: {}", e)))?;

        // Register execution providers
        let mut active_provider = ExecutionProvider::Cpu;
        for provider in &config.providers {
            match Self::register_provider(&mut builder, provider) {
                Ok(()) => {
                    if active_provider == ExecutionProvider::Cpu {
                        active_provider = provider.clone();
                    }
                }
                Err(e) => {
                    warn!("Failed to register {}: {}", provider.name(), e);
                }
            }
        }

        // Load from memory
        let session = builder.commit_from_memory(data)
            .map_err(|e| AiError::ModelLoadError(format!("Failed to load model from memory: {}", e)))?;

        let inputs = Self::extract_inputs(&session)?;
        let outputs = Self::extract_outputs(&session)?;
        let metadata = Self::extract_metadata(&session);

        Ok(Self {
            session,
            inputs,
            outputs,
            metadata,
            active_provider,
        })
    }

    fn register_provider(builder: &mut ort::SessionBuilder, provider: &ExecutionProvider) -> Result<()> {
        use ort::ExecutionProviderDispatch;

        match provider {
            ExecutionProvider::Cpu => {
                // CPU is always available as fallback
                Ok(())
            }
            ExecutionProvider::Cuda { device_id } => {
                let cuda_ep = ort::CUDAExecutionProvider::default()
                    .with_device_id(*device_id);
                *builder = builder.clone().with_execution_providers([cuda_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("CUDA provider error: {}", e)))?;
                Ok(())
            }
            ExecutionProvider::TensorRT { device_id, fp16, max_workspace_size } => {
                let mut trt_ep = ort::TensorRTExecutionProvider::default()
                    .with_device_id(*device_id);
                if *fp16 {
                    trt_ep = trt_ep.with_fp16(true);
                }
                trt_ep = trt_ep.with_max_workspace_size(*max_workspace_size);
                *builder = builder.clone().with_execution_providers([trt_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("TensorRT provider error: {}", e)))?;
                Ok(())
            }
            #[cfg(target_os = "macos")]
            ExecutionProvider::CoreML { use_neural_engine, only_neural_engine } => {
                let mut coreml_ep = ort::CoreMLExecutionProvider::default();
                if *use_neural_engine {
                    coreml_ep = coreml_ep.with_ane_only(*only_neural_engine);
                }
                *builder = builder.clone().with_execution_providers([coreml_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("CoreML provider error: {}", e)))?;
                Ok(())
            }
            #[cfg(not(target_os = "macos"))]
            ExecutionProvider::CoreML { .. } => {
                Err(AiError::NotConfigured("CoreML is only available on macOS".into()))
            }
            #[cfg(target_os = "windows")]
            ExecutionProvider::DirectML { device_id } => {
                let dml_ep = ort::DirectMLExecutionProvider::default()
                    .with_device_id(*device_id);
                *builder = builder.clone().with_execution_providers([dml_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("DirectML provider error: {}", e)))?;
                Ok(())
            }
            #[cfg(not(target_os = "windows"))]
            ExecutionProvider::DirectML { .. } => {
                Err(AiError::NotConfigured("DirectML is only available on Windows".into()))
            }
            ExecutionProvider::OpenVINO { device_type } => {
                let openvino_ep = ort::OpenVINOExecutionProvider::default()
                    .with_device_type(device_type.clone());
                *builder = builder.clone().with_execution_providers([openvino_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("OpenVINO provider error: {}", e)))?;
                Ok(())
            }
            ExecutionProvider::ROCm { device_id } => {
                let rocm_ep = ort::ROCmExecutionProvider::default()
                    .with_device_id(*device_id);
                *builder = builder.clone().with_execution_providers([rocm_ep])
                    .map_err(|e| AiError::ModelLoadError(format!("ROCm provider error: {}", e)))?;
                Ok(())
            }
        }
    }

    fn extract_inputs(session: &ort::Session) -> Result<Vec<TensorInfo>> {
        let mut inputs = Vec::new();
        for input in session.inputs.iter() {
            let shape = input.input_type
                .tensor_dimensions()
                .map(|dims| dims.iter().map(|d| *d as i64).collect())
                .unwrap_or_default();

            let dtype = Self::convert_tensor_type(&input.input_type);

            inputs.push(TensorInfo {
                name: input.name.clone(),
                shape,
                dtype,
            });
        }
        Ok(inputs)
    }

    fn extract_outputs(session: &ort::Session) -> Result<Vec<TensorInfo>> {
        let mut outputs = Vec::new();
        for output in session.outputs.iter() {
            let shape = output.output_type
                .tensor_dimensions()
                .map(|dims| dims.iter().map(|d| *d as i64).collect())
                .unwrap_or_default();

            let dtype = Self::convert_tensor_type(&output.output_type);

            outputs.push(TensorInfo {
                name: output.name.clone(),
                shape,
                dtype,
            });
        }
        Ok(outputs)
    }

    fn convert_tensor_type(value_type: &ort::ValueType) -> TensorType {
        match value_type {
            ort::ValueType::Tensor { ty, .. } => {
                match ty {
                    ort::TensorElementType::Float32 => TensorType::Float32,
                    ort::TensorElementType::Float16 => TensorType::Float16,
                    ort::TensorElementType::Int32 => TensorType::Int32,
                    ort::TensorElementType::Int64 => TensorType::Int64,
                    ort::TensorElementType::Uint8 => TensorType::Uint8,
                    ort::TensorElementType::Int8 => TensorType::Int8,
                    ort::TensorElementType::Bool => TensorType::Bool,
                    _ => TensorType::Float32,
                }
            }
            _ => TensorType::Float32,
        }
    }

    fn extract_metadata(session: &ort::Session) -> ModelMetadata {
        let meta = session.metadata();
        let custom = match &meta {
            Ok(m) => m.custom().unwrap_or_default(),
            Err(_) => std::collections::HashMap::new(),
        };

        ModelMetadata {
            producer: meta.as_ref().ok().and_then(|m| m.producer().ok()),
            producer_version: meta.as_ref().ok().and_then(|m| m.producer_version().ok()),
            domain: meta.as_ref().ok().and_then(|m| m.domain().ok()),
            description: meta.as_ref().ok().and_then(|m| m.description().ok()),
            version: meta.as_ref().ok().and_then(|m| m.version().ok()).unwrap_or(0),
            custom,
        }
    }

    /// Get input tensor information.
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Get output tensor information.
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Get model metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get the active execution provider.
    pub fn active_provider(&self) -> &ExecutionProvider {
        &self.active_provider
    }

    /// Run inference with a single f32 input tensor.
    pub fn run_f32(&self, input: ndarray::ArrayViewD<f32>) -> Result<ndarray::ArrayD<f32>> {
        if self.inputs.is_empty() {
            return Err(AiError::InvalidConfig("Model has no inputs".into()));
        }

        let input_name = &self.inputs[0].name;
        let input_value = ort::Value::from_array(input)
            .map_err(|e| AiError::InferenceError(format!("Failed to create input tensor: {}", e)))?;

        let outputs = self.session.run(ort::inputs![input_name => input_value])
            .map_err(|e| AiError::InferenceError(format!("Inference failed: {}", e)))?;

        if outputs.is_empty() {
            return Err(AiError::InferenceError("Model produced no outputs".into()));
        }

        let output = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| AiError::InferenceError(format!("Failed to extract output: {}", e)))?;

        Ok(output.to_owned().into_dyn())
    }

    /// Run inference with a 4D f32 input tensor (batch, channels, height, width).
    pub fn run_image(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array4<f32>> {
        let input_view = input.view().into_dyn();
        let output = self.run_f32(input_view)?;

        let shape = output.shape();
        if shape.len() != 4 {
            return Err(AiError::DimensionMismatch {
                expected: "4D tensor [N,C,H,W]".into(),
                actual: format!("{}D tensor {:?}", shape.len(), shape),
            });
        }

        output.into_shape_with_order((shape[0], shape[1], shape[2], shape[3]))
            .map_err(|e| AiError::InferenceError(format!("Failed to reshape output: {}", e)))
    }

    /// Run inference with multiple named inputs.
    pub fn run_multi<'a>(
        &self,
        inputs: impl IntoIterator<Item = (&'a str, ndarray::ArrayViewD<'a, f32>)>,
    ) -> Result<Vec<ndarray::ArrayD<f32>>> {
        let mut ort_inputs = Vec::new();
        for (name, array) in inputs {
            let value = ort::Value::from_array(array)
                .map_err(|e| AiError::InferenceError(format!("Failed to create input '{}': {}", name, e)))?;
            ort_inputs.push((name.to_string().into(), value));
        }

        let outputs = self.session.run(ort_inputs)
            .map_err(|e| AiError::InferenceError(format!("Inference failed: {}", e)))?;

        let mut result = Vec::new();
        for output in outputs.iter() {
            let tensor = output.try_extract_tensor::<f32>()
                .map_err(|e| AiError::InferenceError(format!("Failed to extract output: {}", e)))?;
            result.push(tensor.to_owned().into_dyn());
        }

        Ok(result)
    }

    /// Run inference and return u8 image data (for visualization).
    pub fn run_image_u8(
        &self,
        input: &ndarray::Array4<f32>,
    ) -> Result<(Vec<u8>, usize, usize)> {
        let output = self.run_image(input)?;

        let (_, channels, height, width) = output.dim();
        if channels != 3 {
            return Err(AiError::InferenceError(format!(
                "Expected 3 channels for RGB output, got {}",
                channels
            )));
        }

        // Convert from NCHW float to HWC u8
        let mut data = Vec::with_capacity(height * width * 3);
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let value = output[[0, c, y, x]].clamp(0.0, 1.0) * 255.0;
                    data.push(value as u8);
                }
            }
        }

        Ok((data, width, height))
    }
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("active_provider", &self.active_provider)
            .finish()
    }
}

/// Placeholder session when ONNX feature is not enabled.
#[cfg(not(feature = "onnx"))]
pub struct OnnxSession {
    _private: (),
}

#[cfg(not(feature = "onnx"))]
impl OnnxSession {
    /// Load a model (returns error without ONNX feature).
    pub fn load(_path: impl AsRef<Path>, _config: OnnxConfig) -> Result<Self> {
        Err(AiError::NotConfigured(
            "ONNX Runtime not available. Enable the 'onnx' feature to use model inference.".into(),
        ))
    }

    /// Load from memory (returns error without ONNX feature).
    pub fn load_from_memory(_data: &[u8], _config: OnnxConfig) -> Result<Self> {
        Err(AiError::NotConfigured(
            "ONNX Runtime not available. Enable the 'onnx' feature to use model inference.".into(),
        ))
    }
}

/// Helper to preprocess image for model input.
pub fn preprocess_image(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> ndarray::Array4<f32> {
    // Convert HWC u8 to NCHW f32
    let mut arr = ndarray::Array4::zeros((1, channels, height, width));
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let idx = (y * width + x) * channels + c;
                arr[[0, c, y, x]] = data[idx] as f32 / 255.0;
            }
        }
    }
    arr
}

/// Helper to postprocess model output to image.
pub fn postprocess_image(output: &ndarray::Array4<f32>) -> (Vec<u8>, usize, usize) {
    let (_, channels, height, width) = output.dim();
    let mut data = Vec::with_capacity(height * width * channels);

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let value = output[[0, c, y, x]].clamp(0.0, 1.0) * 255.0;
                data.push(value as u8);
            }
        }
    }

    (data, width, height)
}

/// Batch multiple images for inference.
pub fn batch_images(images: &[ndarray::Array4<f32>]) -> Result<ndarray::Array4<f32>> {
    if images.is_empty() {
        return Err(AiError::InvalidFrame("No images to batch".into()));
    }

    let (_, c, h, w) = images[0].dim();
    let batch_size = images.len();

    // Verify all images have same dimensions
    for (i, img) in images.iter().enumerate() {
        let (_, ic, ih, iw) = img.dim();
        if ic != c || ih != h || iw != w {
            return Err(AiError::DimensionMismatch {
                expected: format!("[{}x{}x{}]", c, h, w),
                actual: format!("image {}: [{}x{}x{}]", i, ic, ih, iw),
            });
        }
    }

    let mut batch = ndarray::Array4::zeros((batch_size, c, h, w));
    for (i, img) in images.iter().enumerate() {
        batch.slice_mut(ndarray::s![i, .., .., ..]).assign(&img.slice(ndarray::s![0, .., .., ..]));
    }

    Ok(batch)
}

/// Split batched output into individual images.
pub fn unbatch_images(batch: &ndarray::Array4<f32>) -> Vec<ndarray::Array4<f32>> {
    let (n, c, h, w) = batch.dim();
    let mut images = Vec::with_capacity(n);

    for i in 0..n {
        let mut img = ndarray::Array4::zeros((1, c, h, w));
        img.slice_mut(ndarray::s![0, .., .., ..]).assign(&batch.slice(ndarray::s![i, .., .., ..]));
        images.push(img);
    }

    images
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_provider_default() {
        let provider = ExecutionProvider::default();
        assert_eq!(provider, ExecutionProvider::Cpu);
    }

    #[test]
    fn test_execution_provider_names() {
        assert_eq!(ExecutionProvider::Cpu.name(), "CPU");
        assert_eq!(ExecutionProvider::Cuda { device_id: 0 }.name(), "CUDA");
        assert_eq!(ExecutionProvider::TensorRT { device_id: 0, fp16: true, max_workspace_size: 1 << 30 }.name(), "TensorRT");
    }

    #[test]
    fn test_config_builder() {
        let config = OnnxConfig::new()
            .with_provider(ExecutionProvider::Cuda { device_id: 1 })
            .with_optimization_level(OptimizationLevel::Extended)
            .with_intra_op_threads(4);

        assert_eq!(config.intra_op_threads, 4);
        assert_eq!(config.optimization_level, OptimizationLevel::Extended);
    }

    #[test]
    fn test_preprocess_image() {
        let data = vec![255u8, 128, 64, 0, 255, 128]; // 2 pixels, 3 channels
        let arr = preprocess_image(&data, 2, 1, 3);

        assert_eq!(arr.dim(), (1, 3, 1, 2));
        assert!((arr[[0, 0, 0, 0]] - 1.0).abs() < 0.01); // R=255 -> 1.0
        assert!((arr[[0, 1, 0, 0]] - 0.5).abs() < 0.01); // G=128 -> 0.5
    }

    #[test]
    fn test_batch_unbatch() {
        let img1 = ndarray::Array4::ones((1, 3, 64, 64));
        let img2 = ndarray::Array4::zeros((1, 3, 64, 64));

        let batched = batch_images(&[img1.clone(), img2.clone()]).unwrap();
        assert_eq!(batched.dim(), (2, 3, 64, 64));

        let unbatched = unbatch_images(&batched);
        assert_eq!(unbatched.len(), 2);
        assert_eq!(unbatched[0].dim(), (1, 3, 64, 64));
    }
}
