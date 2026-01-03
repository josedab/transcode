//! Per-title encoding optimization for transcode
//!
//! This crate provides content-adaptive encoding with automatic bitrate ladder
//! generation based on video complexity analysis.

use transcode_core::Frame;

mod error;
mod analyzer;
mod ladder;
mod optimizer;

pub use error::*;
pub use analyzer::*;
pub use ladder::*;
pub use optimizer::*;

/// Result type for per-title operations
pub type Result<T> = std::result::Result<T, PerTitleError>;

/// Per-title encoding configuration
#[derive(Debug, Clone)]
pub struct PerTitleConfig {
    /// Target VMAF score for quality
    pub target_vmaf: f64,
    /// Minimum bitrate in kbps
    pub min_bitrate: u32,
    /// Maximum bitrate in kbps
    pub max_bitrate: u32,
    /// Number of renditions to generate
    pub num_renditions: usize,
    /// Resolution constraints
    pub resolutions: Vec<Resolution>,
    /// Analysis sample rate (frames per second to analyze)
    pub sample_rate: f32,
    /// Target VMAF configuration (threshold for quality)
    pub vmaf_threshold: Option<f64>,
}

impl Default for PerTitleConfig {
    fn default() -> Self {
        Self {
            target_vmaf: 93.0,
            min_bitrate: 200,
            max_bitrate: 15000,
            num_renditions: 6,
            resolutions: vec![
                Resolution::new(3840, 2160), // 4K
                Resolution::new(1920, 1080), // 1080p
                Resolution::new(1280, 720),  // 720p
                Resolution::new(854, 480),   // 480p
                Resolution::new(640, 360),   // 360p
                Resolution::new(426, 240),   // 240p
            ],
            sample_rate: 1.0,
            vmaf_threshold: None,
        }
    }
}

/// Video resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Calculate total pixels
    pub fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

/// A single rendition in the bitrate ladder
#[derive(Debug, Clone)]
pub struct Rendition {
    /// Resolution for this rendition
    pub resolution: Resolution,
    /// Target bitrate in kbps
    pub bitrate: u32,
    /// Estimated VMAF score
    pub estimated_vmaf: f64,
    /// Frame rate (may be reduced for lower renditions)
    pub frame_rate: f64,
    /// Codec preset to use
    pub preset: EncodingPreset,
}

/// Encoding preset for a rendition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodingPreset {
    /// Fastest encoding, lowest quality
    Ultrafast,
    /// Very fast encoding
    Veryfast,
    /// Fast encoding
    Fast,
    /// Medium encoding (balanced)
    #[default]
    Medium,
    /// Slow encoding, better quality
    Slow,
    /// Very slow encoding, high quality
    Veryslow,
}

/// Complete bitrate ladder for a video
#[derive(Debug, Clone)]
pub struct BitrateLadder {
    /// Source video info
    pub source_resolution: Resolution,
    pub source_frame_rate: f64,
    pub source_duration: f64,
    /// Complexity metrics
    pub complexity: ContentComplexity,
    /// Generated renditions (sorted by bitrate descending)
    pub renditions: Vec<Rendition>,
}

impl BitrateLadder {
    /// Get total estimated storage for all renditions (in bytes)
    pub fn estimated_storage(&self) -> u64 {
        self.renditions.iter()
            .map(|r| {
                let bits = r.bitrate as u64 * 1000 * self.source_duration as u64;
                bits / 8
            })
            .sum()
    }

    /// Get rendition closest to target resolution
    pub fn get_rendition(&self, target: Resolution) -> Option<&Rendition> {
        self.renditions.iter()
            .min_by_key(|r| {
                
                (r.resolution.pixels() as i64 - target.pixels() as i64).abs()
            })
    }
}

/// Content complexity analysis results
#[derive(Debug, Clone)]
pub struct ContentComplexity {
    /// Spatial complexity (0-100)
    pub spatial: f64,
    /// Temporal complexity (0-100)
    pub temporal: f64,
    /// Overall complexity score (0-100)
    pub overall: f64,
    /// Content type classification
    pub content_type: ContentType,
    /// Recommended base bitrate multiplier
    pub bitrate_multiplier: f64,
}

/// Content type classification for encoding hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Animation/cartoon content
    Animation,
    /// Film/cinematic content
    Film,
    /// Sports/high motion content
    Sports,
    /// Gaming/screen capture content
    Gaming,
    /// Talking head/presentation content
    TalkingHead,
    /// Nature/documentary content
    Nature,
    /// Generic/mixed content
    Generic,
}

/// Per-title encoder that generates optimal bitrate ladders
pub struct PerTitleEncoder {
    config: PerTitleConfig,
}

impl PerTitleEncoder {
    /// Create a new per-title encoder
    pub fn new(config: PerTitleConfig) -> Self {
        Self { config }
    }

    /// Analyze video and generate optimal bitrate ladder
    pub fn analyze(&self, frames: &[Frame]) -> Result<BitrateLadder> {
        if frames.is_empty() {
            return Err(PerTitleError::NoFrames);
        }

        let first_frame = &frames[0];
        let source_resolution = Resolution::new(first_frame.width(), first_frame.height());

        // Analyze content complexity
        let complexity = self.analyze_complexity(frames)?;

        // Generate bitrate ladder based on complexity
        let renditions = self.generate_ladder(&complexity, source_resolution)?;

        Ok(BitrateLadder {
            source_resolution,
            source_frame_rate: 30.0, // Would come from container
            source_duration: frames.len() as f64 / 30.0,
            complexity,
            renditions,
        })
    }

    /// Analyze content complexity from frames
    fn analyze_complexity(&self, frames: &[Frame]) -> Result<ContentComplexity> {
        let mut spatial_sum = 0.0;
        let mut temporal_sum = 0.0;
        let mut prev_frame: Option<&Frame> = None;

        for frame in frames {
            // Calculate spatial complexity using edge detection approximation
            spatial_sum += self.calculate_spatial_complexity(frame);

            // Calculate temporal complexity using frame difference
            if let Some(prev) = prev_frame {
                temporal_sum += self.calculate_temporal_complexity(prev, frame);
            }
            prev_frame = Some(frame);
        }

        let num_frames = frames.len() as f64;
        let spatial = (spatial_sum / num_frames).min(100.0);
        let temporal = if frames.len() > 1 {
            (temporal_sum / (num_frames - 1.0)).min(100.0)
        } else {
            0.0
        };

        let overall = (spatial * 0.6 + temporal * 0.4).min(100.0);
        let content_type = self.classify_content(spatial, temporal);
        let bitrate_multiplier = self.calculate_bitrate_multiplier(overall, content_type);

        Ok(ContentComplexity {
            spatial,
            temporal,
            overall,
            content_type,
            bitrate_multiplier,
        })
    }

    fn calculate_spatial_complexity(&self, frame: &Frame) -> f64 {
        // Sobel-like edge detection approximation
        let data = frame.plane(0).unwrap_or(&[]);
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let stride = frame.stride(0);

        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut edge_sum: u64 = 0;
        let mut count: u64 = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = |dy: i32, dx: i32| -> usize {
                    (y as i32 + dy) as usize * stride + (x as i32 + dx) as usize
                };

                // Simplified Sobel gradient
                let gx = data[idx(0, 1)] as i32 - data[idx(0, -1)] as i32;
                let gy = data[idx(1, 0)] as i32 - data[idx(-1, 0)] as i32;

                edge_sum += ((gx * gx + gy * gy) as f64).sqrt() as u64;
                count += 1;
            }
        }

        if count > 0 {
            (edge_sum as f64 / count as f64) * 0.5
        } else {
            0.0
        }
    }

    fn calculate_temporal_complexity(&self, prev: &Frame, curr: &Frame) -> f64 {
        let prev_data = prev.plane(0).unwrap_or(&[]);
        let curr_data = curr.plane(0).unwrap_or(&[]);

        let len = prev_data.len().min(curr_data.len());
        if len == 0 {
            return 0.0;
        }

        let diff_sum: u64 = prev_data.iter()
            .zip(curr_data.iter())
            .take(len)
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();

        (diff_sum as f64 / len as f64) * 0.5
    }

    fn classify_content(&self, spatial: f64, temporal: f64) -> ContentType {
        match (spatial, temporal) {
            (s, t) if s < 20.0 && t < 20.0 => ContentType::Animation,
            (s, t) if s < 40.0 && t < 15.0 => ContentType::TalkingHead,
            (_, t) if t > 60.0 => ContentType::Sports,
            (s, _) if s > 70.0 => ContentType::Nature,
            (s, t) if s > 50.0 && t > 40.0 => ContentType::Gaming,
            (s, _) if s > 40.0 => ContentType::Film,
            _ => ContentType::Generic,
        }
    }

    fn calculate_bitrate_multiplier(&self, complexity: f64, content_type: ContentType) -> f64 {
        let base = 0.5 + (complexity / 100.0);

        let type_factor = match content_type {
            ContentType::Animation => 0.7,
            ContentType::TalkingHead => 0.6,
            ContentType::Film => 1.0,
            ContentType::Sports => 1.3,
            ContentType::Gaming => 1.2,
            ContentType::Nature => 1.1,
            ContentType::Generic => 1.0,
        };

        base * type_factor
    }

    fn generate_ladder(
        &self,
        complexity: &ContentComplexity,
        source: Resolution
    ) -> Result<Vec<Rendition>> {
        let mut renditions = Vec::new();

        // Filter resolutions that don't exceed source
        let valid_resolutions: Vec<_> = self.config.resolutions.iter()
            .filter(|r| r.pixels() <= source.pixels())
            .cloned()
            .collect();

        if valid_resolutions.is_empty() {
            return Err(PerTitleError::NoValidResolutions);
        }

        // Calculate bitrate range based on complexity
        let bitrate_range = self.config.max_bitrate - self.config.min_bitrate;

        for (i, resolution) in valid_resolutions.iter().take(self.config.num_renditions).enumerate() {
            // Calculate bitrate based on resolution and complexity
            let resolution_factor = (resolution.pixels() as f64 / source.pixels() as f64).sqrt();
            let position_factor = 1.0 - (i as f64 / self.config.num_renditions as f64);

            let base_bitrate = self.config.min_bitrate as f64
                + bitrate_range as f64 * resolution_factor * position_factor;
            let adjusted_bitrate = (base_bitrate * complexity.bitrate_multiplier) as u32;
            let bitrate = adjusted_bitrate
                .max(self.config.min_bitrate)
                .min(self.config.max_bitrate);

            // Estimate VMAF based on bitrate and resolution
            let estimated_vmaf = self.estimate_vmaf(bitrate, *resolution, complexity);

            // Select preset based on rendition position
            let preset = if i == 0 {
                EncodingPreset::Slow
            } else if i < 3 {
                EncodingPreset::Medium
            } else {
                EncodingPreset::Fast
            };

            renditions.push(Rendition {
                resolution: *resolution,
                bitrate,
                estimated_vmaf,
                frame_rate: 30.0, // Could be reduced for lower renditions
                preset,
            });
        }

        // Sort by bitrate descending
        renditions.sort_by(|a, b| b.bitrate.cmp(&a.bitrate));

        Ok(renditions)
    }

    fn estimate_vmaf(&self, bitrate: u32, resolution: Resolution, complexity: &ContentComplexity) -> f64 {
        // Simplified VMAF estimation model
        let bpp = bitrate as f64 * 1000.0 / (resolution.pixels() as f64 * 30.0);
        let base_vmaf = 100.0 * (1.0 - (-bpp * 50.0).exp());

        // Adjust for complexity
        let complexity_penalty = complexity.overall * 0.1;

        (base_vmaf - complexity_penalty).clamp(0.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution() {
        let res = Resolution::new(1920, 1080);
        assert_eq!(res.pixels(), 2073600);
        assert!((res.aspect_ratio() - 1.778).abs() < 0.01);
    }

    #[test]
    fn test_default_config() {
        let config = PerTitleConfig::default();
        assert_eq!(config.target_vmaf, 93.0);
        assert_eq!(config.num_renditions, 6);
    }

    #[test]
    fn test_content_type_classification() {
        let encoder = PerTitleEncoder::new(PerTitleConfig::default());

        assert_eq!(encoder.classify_content(10.0, 10.0), ContentType::Animation);
        assert_eq!(encoder.classify_content(30.0, 10.0), ContentType::TalkingHead);
        assert_eq!(encoder.classify_content(50.0, 70.0), ContentType::Sports);
    }
}
