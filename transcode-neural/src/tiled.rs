//! Tile-based inference for memory-efficient neural processing.
//!
//! This module provides tile-based processing for large images that don't fit
//! in GPU memory. Images are split into overlapping tiles, processed individually,
//! and blended back together.
//!
//! # Memory Efficiency
//!
//! For a 4K image (3840x2160) at 4x scale with float32:
//! - Full image: ~100MB input, ~1.6GB output
//! - 512x512 tiles: ~3MB per tile, processed sequentially
//!
//! # Overlap Blending
//!
//! Tiles overlap to avoid visible seams. The overlap region uses linear
//! blending to smoothly transition between adjacent tiles.

use crate::{NeuralError, NeuralFrame, Result};
#[cfg(feature = "onnx")]
use std::sync::Arc;

/// Tile configuration.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Overlap in pixels.
    pub overlap: u32,
    /// Scale factor.
    pub scale: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_width: 512,
            tile_height: 512,
            overlap: 32,
            scale: 4,
        }
    }
}

impl TileConfig {
    /// Create a new tile configuration.
    pub fn new(tile_size: u32, overlap: u32, scale: u32) -> Self {
        Self {
            tile_width: tile_size,
            tile_height: tile_size,
            overlap,
            scale,
        }
    }

    /// Calculate the number of tiles needed for an image.
    pub fn tile_count(&self, width: u32, height: u32) -> (u32, u32) {
        let step_x = self.tile_width - self.overlap;
        let step_y = self.tile_height - self.overlap;

        let tiles_x = ((width as f32 - self.overlap as f32) / step_x as f32).ceil() as u32;
        let tiles_y = ((height as f32 - self.overlap as f32) / step_y as f32).ceil() as u32;

        (tiles_x.max(1), tiles_y.max(1))
    }
}

/// A single tile for processing.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile data (RGB float32).
    pub data: Vec<f32>,
    /// Tile width.
    pub width: u32,
    /// Tile height.
    pub height: u32,
    /// X position in source image.
    pub src_x: u32,
    /// Y position in source image.
    pub src_y: u32,
    /// Tile index X.
    pub tile_x: u32,
    /// Tile index Y.
    pub tile_y: u32,
}

impl Tile {
    /// Create a new empty tile.
    pub fn new(width: u32, height: u32, src_x: u32, src_y: u32, tile_x: u32, tile_y: u32) -> Self {
        Self {
            data: vec![0.0; (width * height * 3) as usize],
            width,
            height,
            src_x,
            src_y,
            tile_x,
            tile_y,
        }
    }

    /// Get pixel value at (x, y) for channel c.
    pub fn get(&self, x: u32, y: u32, c: u32) -> f32 {
        let idx = ((y * self.width + x) * 3 + c) as usize;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Set pixel value at (x, y) for channel c.
    pub fn set(&mut self, x: u32, y: u32, c: u32, value: f32) {
        let idx = ((y * self.width + x) * 3 + c) as usize;
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }
}

/// Tile-based neural processor.
pub struct TiledProcessor {
    config: TileConfig,
}

impl TiledProcessor {
    /// Create a new tiled processor.
    pub fn new(config: TileConfig) -> Self {
        Self { config }
    }

    /// Split a frame into tiles.
    pub fn split(&self, frame: &NeuralFrame) -> Vec<Tile> {
        let (tiles_x, tiles_y) = self.config.tile_count(frame.width, frame.height);
        let step_x = self.config.tile_width - self.config.overlap;
        let step_y = self.config.tile_height - self.config.overlap;

        let mut tiles = Vec::with_capacity((tiles_x * tiles_y) as usize);

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let src_x = (tx * step_x).min(frame.width.saturating_sub(self.config.tile_width));
                let src_y = (ty * step_y).min(frame.height.saturating_sub(self.config.tile_height));

                let tile_w = self.config.tile_width.min(frame.width - src_x);
                let tile_h = self.config.tile_height.min(frame.height - src_y);

                let mut tile = Tile::new(tile_w, tile_h, src_x, src_y, tx, ty);

                // Copy pixel data
                for y in 0..tile_h {
                    for x in 0..tile_w {
                        for c in 0..3 {
                            let src_idx = (((src_y + y) * frame.width + (src_x + x)) * 3 + c) as usize;
                            let value = frame.data.get(src_idx).copied().unwrap_or(0.0);
                            tile.set(x, y, c, value);
                        }
                    }
                }

                tiles.push(tile);
            }
        }

        tiles
    }

    /// Merge tiles back into a frame with overlap blending.
    pub fn merge(&self, tiles: &[Tile], width: u32, height: u32) -> Result<NeuralFrame> {
        let out_width = width * self.config.scale;
        let out_height = height * self.config.scale;
        let scaled_overlap = self.config.overlap * self.config.scale;

        let mut output = NeuralFrame::new(out_width, out_height);
        let mut weight_map = vec![0.0f32; (out_width * out_height) as usize];

        for tile in tiles {
            let out_x = tile.src_x * self.config.scale;
            let out_y = tile.src_y * self.config.scale;
            let tile_w = tile.width * self.config.scale;
            let tile_h = tile.height * self.config.scale;

            for y in 0..tile_h {
                for x in 0..tile_w {
                    let dst_x = out_x + x;
                    let dst_y = out_y + y;

                    if dst_x >= out_width || dst_y >= out_height {
                        continue;
                    }

                    // Calculate blend weight based on distance from edges
                    let weight = self.blend_weight(x, y, tile_w, tile_h, scaled_overlap);
                    let weight_idx = (dst_y * out_width + dst_x) as usize;

                    for c in 0..3 {
                        let tile_idx = ((y * tile_w + x) * 3 + c) as usize;
                        let dst_idx = ((dst_y * out_width + dst_x) * 3 + c) as usize;

                        if tile_idx < tile.data.len() && dst_idx < output.data.len() {
                            output.data[dst_idx] += tile.data[tile_idx] * weight;
                        }
                    }

                    weight_map[weight_idx] += weight;
                }
            }
        }

        // Normalize by weight
        for y in 0..out_height {
            for x in 0..out_width {
                let weight_idx = (y * out_width + x) as usize;
                let weight = weight_map[weight_idx];

                if weight > 0.0 {
                    for c in 0..3 {
                        let idx = ((y * out_width + x) * 3 + c) as usize;
                        output.data[idx] /= weight;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Calculate blend weight for smooth tile transitions.
    fn blend_weight(&self, x: u32, y: u32, width: u32, height: u32, overlap: u32) -> f32 {
        let mut weight = 1.0f32;

        // Left edge ramp
        if x < overlap {
            weight *= x as f32 / overlap as f32;
        }
        // Right edge ramp
        if x >= width - overlap {
            weight *= (width - 1 - x) as f32 / overlap as f32;
        }
        // Top edge ramp
        if y < overlap {
            weight *= y as f32 / overlap as f32;
        }
        // Bottom edge ramp
        if y >= height - overlap {
            weight *= (height - 1 - y) as f32 / overlap as f32;
        }

        weight.clamp(0.0, 1.0)
    }

    /// Process tiles with a custom function.
    pub fn process<F>(&self, frame: &NeuralFrame, process_fn: F) -> Result<NeuralFrame>
    where
        F: Fn(&Tile) -> Result<Tile>,
    {
        let tiles = self.split(frame);
        let mut processed_tiles = Vec::with_capacity(tiles.len());

        for tile in &tiles {
            let processed = process_fn(tile)?;
            processed_tiles.push(processed);
        }

        self.merge(&processed_tiles, frame.width, frame.height)
    }
}

/// Model cache for efficient model reuse.
pub struct ModelCache {
    /// Cached models by path.
    #[cfg(feature = "onnx")]
    models: std::sync::RwLock<std::collections::HashMap<String, Arc<ort::session::Session>>>,
    #[cfg(not(feature = "onnx"))]
    _phantom: std::marker::PhantomData<()>,
    /// Maximum cache size.
    #[cfg(feature = "onnx")]
    max_size: usize,
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new(5)
    }
}

impl ModelCache {
    /// Create a new model cache.
    #[cfg(feature = "onnx")]
    pub fn new(max_size: usize) -> Self {
        Self {
            models: std::sync::RwLock::new(std::collections::HashMap::new()),
            max_size,
        }
    }

    /// Create a new model cache.
    #[cfg(not(feature = "onnx"))]
    pub fn new(_max_size: usize) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get or load a model.
    #[cfg(feature = "onnx")]
    pub fn get_or_load(&self, path: &str) -> Result<Arc<ort::session::Session>> {
        use ort::session::{builder::GraphOptimizationLevel, Session};

        // Check cache first
        {
            let cache = self.models.read().map_err(|_| {
                NeuralError::ModelLoad("Cache lock poisoned".to_string())
            })?;

            if let Some(model) = cache.get(path) {
                tracing::debug!("Model cache hit: {}", path);
                return Ok(Arc::clone(model));
            }
        }

        // Load model
        tracing::info!("Loading model: {}", path);
        let session = Session::builder()
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?;

        let session = Arc::new(session);

        // Add to cache
        {
            let mut cache = self.models.write().map_err(|_| {
                NeuralError::ModelLoad("Cache lock poisoned".to_string())
            })?;

            // Evict if over limit
            while cache.len() >= self.max_size {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }

            cache.insert(path.to_string(), Arc::clone(&session));
        }

        Ok(session)
    }

    /// Get or load a model (stub when ONNX is disabled).
    #[cfg(not(feature = "onnx"))]
    pub fn get_or_load(&self, _path: &str) -> Result<()> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Clear the cache.
    #[cfg(feature = "onnx")]
    pub fn clear(&self) {
        if let Ok(mut cache) = self.models.write() {
            cache.clear();
        }
    }

    #[cfg(not(feature = "onnx"))]
    pub fn clear(&self) {}

    /// Get cache size.
    #[cfg(feature = "onnx")]
    pub fn len(&self) -> usize {
        self.models.read().map(|c| c.len()).unwrap_or(0)
    }

    #[cfg(not(feature = "onnx"))]
    pub fn len(&self) -> usize {
        0
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Global model cache instance.
static MODEL_CACHE: std::sync::OnceLock<ModelCache> = std::sync::OnceLock::new();

/// Get the global model cache.
pub fn global_model_cache() -> &'static ModelCache {
    MODEL_CACHE.get_or_init(ModelCache::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config() {
        let config = TileConfig::new(512, 32, 4);
        assert_eq!(config.tile_width, 512);
        assert_eq!(config.overlap, 32);

        let (tx, ty) = config.tile_count(1920, 1080);
        assert!(tx > 0);
        assert!(ty > 0);
    }

    #[test]
    fn test_tile_split_merge() {
        let config = TileConfig::new(128, 16, 1);
        let processor = TiledProcessor::new(config);

        let mut frame = NeuralFrame::new(256, 256);
        // Fill with gradient
        for y in 0..256 {
            for x in 0..256 {
                for c in 0..3 {
                    let idx = ((y * 256 + x) * 3 + c) as usize;
                    frame.data[idx] = (x + y) as f32 / 512.0;
                }
            }
        }

        let tiles = processor.split(&frame);
        assert!(!tiles.is_empty());

        // Identity processing (no scale)
        let merged = processor.merge(&tiles, 256, 256).unwrap();
        assert_eq!(merged.width, 256);
        assert_eq!(merged.height, 256);
    }

    #[test]
    fn test_blend_weight() {
        let config = TileConfig::new(100, 20, 1);
        let processor = TiledProcessor::new(config);

        // Center should have weight 1.0
        let center_weight = processor.blend_weight(50, 50, 100, 100, 20);
        assert!((center_weight - 1.0).abs() < 0.01);

        // Edge should have lower weight
        let edge_weight = processor.blend_weight(5, 50, 100, 100, 20);
        assert!(edge_weight < 1.0);
    }

    #[test]
    fn test_model_cache() {
        let cache = ModelCache::new(3);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
