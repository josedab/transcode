//! Inference engine for GenAI model loading, caching, and batch inference.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use crate::error::{GenAiError, Result};
use crate::models::{ModelBackend, ModelPrecision};

/// Manages a loaded model inference session.
#[derive(Debug)]
pub struct InferenceSession {
    /// Model identifier.
    pub model_id: String,
    /// Backend runtime.
    pub backend: ModelBackend,
    /// Model precision.
    pub precision: ModelPrecision,
    /// Whether the session has been warmed up.
    pub warm: bool,
    /// Number of inferences performed.
    pub inference_count: u64,
    /// Total inference time in milliseconds.
    pub total_inference_ms: u64,
}

impl InferenceSession {
    /// Create a new inference session.
    pub fn new(model_id: impl Into<String>, backend: ModelBackend, precision: ModelPrecision) -> Self {
        Self {
            model_id: model_id.into(),
            backend,
            precision,
            warm: false,
            inference_count: 0,
            total_inference_ms: 0,
        }
    }

    /// Warm up the session by running a dummy inference pass.
    pub fn warmup(&mut self) -> Result<()> {
        if self.warm {
            return Ok(());
        }
        // Simulate warmup with a small dummy inference
        let dummy = vec![0.0f32; 16];
        let start = Instant::now();
        let _output = simulate_inference(&dummy, self.precision);
        let elapsed = start.elapsed().as_millis() as u64;
        self.total_inference_ms += elapsed;
        self.warm = true;
        Ok(())
    }

    /// Run inference on the given input tensor.
    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Err(GenAiError::InvalidInput("Empty input tensor".into()));
        }
        let start = Instant::now();
        let output = simulate_inference(input, self.precision);
        let elapsed = start.elapsed().as_millis() as u64;
        self.inference_count += 1;
        self.total_inference_ms += elapsed;
        Ok(output)
    }

    /// Average latency per inference in milliseconds.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.inference_count == 0 {
            return 0.0;
        }
        self.total_inference_ms as f64 / self.inference_count as f64
    }
}

/// Simulate inference by applying a simple transform.
fn simulate_inference(input: &[f32], precision: ModelPrecision) -> Vec<f32> {
    let scale = match precision {
        ModelPrecision::Fp32 => 0.99,
        ModelPrecision::Fp16 => 0.985,
        ModelPrecision::Bf16 => 0.987,
        ModelPrecision::Int8 => 0.97,
        ModelPrecision::Int4 => 0.95,
    };
    input
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            // Deterministic small perturbation based on index
            let noise = ((i as f64 % 7.0) - 3.0) * 0.001;
            (v as f64 * scale + noise) as f32
        })
        .collect()
}

/// LRU-style model cache for managing loaded inference sessions.
#[derive(Debug)]
pub struct ModelCache {
    /// Maximum allowed memory in bytes.
    max_memory_bytes: u64,
    /// Loaded sessions keyed by model ID.
    sessions: HashMap<String, InferenceSession>,
    /// Access order for LRU eviction (most recent at end).
    access_order: Vec<String>,
    /// Current memory usage in bytes.
    current_memory: u64,
    /// Memory cost per model (tracked for eviction).
    model_memory: HashMap<String, u64>,
}

impl ModelCache {
    /// Create a new model cache with the given memory limit.
    pub fn new(max_memory_bytes: u64) -> Self {
        Self {
            max_memory_bytes,
            sessions: HashMap::new(),
            access_order: Vec::new(),
            current_memory: 0,
            model_memory: HashMap::new(),
        }
    }

    /// Load a model into the cache, evicting LRU entries if needed.
    pub fn load_model(
        &mut self,
        model_id: impl Into<String>,
        backend: ModelBackend,
        precision: ModelPrecision,
        model_memory: u64,
    ) -> Result<()> {
        let model_id = model_id.into();

        if model_memory > self.max_memory_bytes {
            return Err(GenAiError::ResourceExhausted(format!(
                "Model {} requires {} bytes but cache max is {} bytes",
                model_id, model_memory, self.max_memory_bytes
            )));
        }

        // Evict until there's room
        while self.current_memory + model_memory > self.max_memory_bytes {
            self.evict_lru()?;
        }

        let session = InferenceSession::new(&model_id, backend, precision);
        self.sessions.insert(model_id.clone(), session);
        self.model_memory.insert(model_id.clone(), model_memory);
        self.current_memory += model_memory;
        // Update access order
        self.access_order.retain(|id| id != &model_id);
        self.access_order.push(model_id);

        Ok(())
    }

    /// Get an immutable reference to a session, updating access order.
    pub fn get_session(&mut self, model_id: &str) -> Option<&InferenceSession> {
        if self.sessions.contains_key(model_id) {
            self.touch(model_id);
            self.sessions.get(model_id)
        } else {
            None
        }
    }

    /// Get a mutable reference to a session, updating access order.
    pub fn get_session_mut(&mut self, model_id: &str) -> Option<&mut InferenceSession> {
        if self.sessions.contains_key(model_id) {
            self.touch(model_id);
            self.sessions.get_mut(model_id)
        } else {
            None
        }
    }

    /// Evict the least recently used model from the cache.
    pub fn evict_lru(&mut self) -> Result<()> {
        if self.access_order.is_empty() {
            return Err(GenAiError::ResourceExhausted(
                "No models to evict".into(),
            ));
        }
        let evicted = self.access_order.remove(0);
        self.sessions.remove(&evicted);
        if let Some(mem) = self.model_memory.remove(&evicted) {
            self.current_memory = self.current_memory.saturating_sub(mem);
        }
        Ok(())
    }

    /// Return (used, max) memory in bytes.
    pub fn memory_usage(&self) -> (u64, u64) {
        (self.current_memory, self.max_memory_bytes)
    }

    /// List all loaded model IDs.
    pub fn loaded_models(&self) -> Vec<String> {
        self.access_order.clone()
    }

    fn touch(&mut self, model_id: &str) {
        self.access_order.retain(|id| id != model_id);
        self.access_order.push(model_id.to_string());
    }
}

/// Result of a single frame inference.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Index of the frame in the batch.
    pub frame_index: usize,
    /// Output tensor data.
    pub output: Vec<f32>,
    /// Inference latency in milliseconds.
    pub latency_ms: u64,
}

/// Batch inference processor for multiple frames.
#[derive(Debug)]
pub struct BatchInference {
    /// Session (model) ID to use.
    pub session_id: String,
    /// Maximum batch size.
    pub batch_size: usize,
    /// Collected results.
    pub results: Vec<InferenceResult>,
}

impl BatchInference {
    /// Create a new batch inference processor.
    pub fn new(session_id: impl Into<String>, batch_size: usize) -> Self {
        Self {
            session_id: session_id.into(),
            batch_size,
            results: Vec::new(),
        }
    }

    /// Process a batch of inputs through the cached model session.
    pub fn process_batch(
        &mut self,
        cache: &mut ModelCache,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<InferenceResult>> {
        let session = cache
            .get_session_mut(&self.session_id)
            .ok_or_else(|| GenAiError::ModelNotFound(self.session_id.clone()))?;

        let mut results = Vec::with_capacity(inputs.len());

        for (i, chunk) in inputs.chunks(self.batch_size).enumerate() {
            for (j, input) in chunk.iter().enumerate() {
                let frame_index = i * self.batch_size + j;
                let start = Instant::now();
                let output = session.infer(input)?;
                let latency_ms = start.elapsed().as_millis() as u64;
                results.push(InferenceResult {
                    frame_index,
                    output,
                    latency_ms,
                });
            }
        }

        self.results.extend(results.clone());
        Ok(results)
    }
}

/// Result of a quality validation check.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the output passed the quality threshold.
    pub passed: bool,
    /// Similarity score (0.0 to 1.0).
    pub score: f64,
    /// Peak Signal-to-Noise Ratio in dB.
    pub psnr: f64,
}

/// Validates inference output quality against a reference.
#[derive(Debug)]
pub struct QualityValidator {
    /// Reference tensor for comparison.
    reference: Vec<f32>,
    /// Minimum PSNR threshold in dB.
    threshold: f64,
}

impl QualityValidator {
    /// Create a new quality validator.
    pub fn new(reference: Vec<f32>, threshold: f64) -> Self {
        Self {
            reference,
            threshold,
        }
    }

    /// Validate an output tensor against the reference.
    pub fn validate(&self, output: &[f32]) -> ValidationResult {
        let mse = self.compute_mse(output);
        let psnr = if mse > 0.0 {
            let max_val: f64 = 1.0;
            10.0 * (max_val * max_val / mse).log10()
        } else {
            f64::INFINITY
        };

        // Score: normalized similarity (1.0 = identical)
        let score = if psnr.is_infinite() {
            1.0
        } else {
            (psnr / 100.0).min(1.0).max(0.0)
        };

        ValidationResult {
            passed: psnr >= self.threshold,
            score,
            psnr,
        }
    }

    fn compute_mse(&self, output: &[f32]) -> f64 {
        let len = self.reference.len().min(output.len());
        if len == 0 {
            return f64::MAX;
        }
        let sum_sq: f64 = self.reference[..len]
            .iter()
            .zip(output[..len].iter())
            .map(|(&r, &o)| {
                let diff = r as f64 - o as f64;
                diff * diff
            })
            .sum();
        sum_sq / len as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_session_new() {
        let session = InferenceSession::new("test-model", ModelBackend::Onnx, ModelPrecision::Fp32);
        assert_eq!(session.model_id, "test-model");
        assert!(!session.warm);
        assert_eq!(session.inference_count, 0);
        assert_eq!(session.avg_latency_ms(), 0.0);
    }

    #[test]
    fn test_inference_session_warmup_and_infer() {
        let mut session =
            InferenceSession::new("test-model", ModelBackend::Onnx, ModelPrecision::Fp32);

        session.warmup().unwrap();
        assert!(session.warm);

        // Warmup again should be a no-op
        session.warmup().unwrap();

        let input = vec![0.5, 0.6, 0.7, 0.8];
        let output = session.infer(&input).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(session.inference_count, 1);

        // Output should be close but not identical
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let diff = (inp - out).abs();
            assert!(diff < 0.1, "element {} diverged: {} vs {}", i, inp, out);
        }
    }

    #[test]
    fn test_inference_session_empty_input() {
        let mut session =
            InferenceSession::new("test-model", ModelBackend::Onnx, ModelPrecision::Fp32);
        let result = session.infer(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_cache_load_and_get() {
        let mut cache = ModelCache::new(1_000_000);
        cache
            .load_model("model-a", ModelBackend::Onnx, ModelPrecision::Fp32, 500_000)
            .unwrap();

        assert_eq!(cache.loaded_models(), vec!["model-a"]);
        assert_eq!(cache.memory_usage(), (500_000, 1_000_000));

        let session = cache.get_session("model-a");
        assert!(session.is_some());
        assert_eq!(session.unwrap().model_id, "model-a");
    }

    #[test]
    fn test_model_cache_lru_eviction() {
        let mut cache = ModelCache::new(1_000);

        cache
            .load_model("model-a", ModelBackend::Onnx, ModelPrecision::Fp32, 400)
            .unwrap();
        cache
            .load_model("model-b", ModelBackend::Onnx, ModelPrecision::Fp16, 400)
            .unwrap();

        // Access model-a so model-b becomes LRU
        let _ = cache.get_session("model-a");

        // Loading model-c should evict model-b (LRU)
        cache
            .load_model("model-c", ModelBackend::Onnx, ModelPrecision::Fp32, 400)
            .unwrap();

        let models = cache.loaded_models();
        assert!(models.contains(&"model-a".to_string()));
        assert!(models.contains(&"model-c".to_string()));
        assert!(!models.contains(&"model-b".to_string()));
    }

    #[test]
    fn test_model_cache_exceeds_max() {
        let mut cache = ModelCache::new(100);
        let result = cache.load_model("huge-model", ModelBackend::Onnx, ModelPrecision::Fp32, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_inference() {
        let mut cache = ModelCache::new(1_000_000);
        cache
            .load_model("batch-model", ModelBackend::Onnx, ModelPrecision::Fp32, 100_000)
            .unwrap();

        let mut batch = BatchInference::new("batch-model", 2);
        let inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let results = batch.process_batch(&mut cache, &inputs).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].frame_index, 0);
        assert_eq!(results[1].frame_index, 1);
        assert_eq!(results[2].frame_index, 2);
        assert_eq!(results[0].output.len(), 3);
        assert_eq!(batch.results.len(), 3);
    }

    #[test]
    fn test_quality_validator_pass() {
        let reference = vec![0.5, 0.6, 0.7, 0.8];
        let validator = QualityValidator::new(reference.clone(), 20.0);

        // Slightly perturbed output should pass
        let output: Vec<f32> = reference.iter().map(|&v| v * 0.99).collect();
        let result = validator.validate(&output);
        assert!(result.passed);
        assert!(result.psnr > 20.0);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_quality_validator_fail() {
        let reference = vec![0.5, 0.6, 0.7, 0.8];
        let validator = QualityValidator::new(reference, 60.0);

        // Very different output should fail high threshold
        let output = vec![0.0, 0.0, 0.0, 0.0];
        let result = validator.validate(&output);
        assert!(!result.passed);
        assert!(result.psnr < 60.0);
    }

    #[test]
    fn test_quality_validator_identical() {
        let reference = vec![0.5, 0.6, 0.7, 0.8];
        let validator = QualityValidator::new(reference.clone(), 30.0);

        let result = validator.validate(&reference);
        assert!(result.passed);
        assert!(result.psnr.is_infinite());
        assert_eq!(result.score, 1.0);
    }
}
