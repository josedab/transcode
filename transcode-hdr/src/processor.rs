//! HDR processing pipeline.
//!
//! This module provides a complete HDR processing pipeline for:
//! - HDR to SDR conversion
//! - SDR to HDR inverse tone mapping
//! - HDR format conversion (PQ to HLG)
//! - Frame-by-frame processing with metadata

use crate::{
    colorspace::ColorSpaceConverter,
    gamut::{GamutMapper, GamutMapConfig, GamutMappingAlgorithm},
    metadata::Hdr10PlusDynamicMetadata,
    tonemapping::{ToneMapAlgorithm, ToneMapConfig, ToneMapper},
    transfer::TransferConverter,
    ColorSpace, HdrError, HdrFormat, Result, TransferFunction,
};

/// HDR processing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessingMode {
    /// HDR to SDR conversion with tone mapping
    #[default]
    HdrToSdr,
    /// SDR to HDR conversion with inverse tone mapping
    SdrToHdr,
    /// HDR to HDR format conversion (e.g., PQ to HLG)
    HdrToHdr,
    /// Pass-through (no processing)
    Passthrough,
}

/// Configuration for HDR processing pipeline.
#[derive(Debug, Clone)]
pub struct HdrProcessorConfig {
    /// Processing mode
    pub mode: ProcessingMode,
    /// Source HDR format
    pub source_format: HdrFormat,
    /// Target HDR format
    pub target_format: HdrFormat,
    /// Source color space
    pub source_colorspace: ColorSpace,
    /// Target color space
    pub target_colorspace: ColorSpace,
    /// Source transfer function
    pub source_transfer: TransferFunction,
    /// Target transfer function
    pub target_transfer: TransferFunction,
    /// Source peak luminance (nits)
    pub source_peak_nits: f64,
    /// Target peak luminance (nits)
    pub target_peak_nits: f64,
    /// Source minimum luminance (nits)
    pub source_min_nits: f64,
    /// Target minimum luminance (nits)
    pub target_min_nits: f64,
    /// Tone mapping algorithm
    pub tone_map_algorithm: ToneMapAlgorithm,
    /// Gamut mapping algorithm
    pub gamut_map_algorithm: GamutMappingAlgorithm,
    /// Use HDR10+ dynamic metadata if available
    pub use_dynamic_metadata: bool,
    /// Preserve luminance during gamut mapping
    pub preserve_luminance: bool,
    /// Highlight desaturation amount (0.0 - 1.0)
    pub highlight_desaturation: f64,
}

impl Default for HdrProcessorConfig {
    fn default() -> Self {
        Self {
            mode: ProcessingMode::HdrToSdr,
            source_format: HdrFormat::Hdr10,
            target_format: HdrFormat::Sdr,
            source_colorspace: ColorSpace::Bt2020,
            target_colorspace: ColorSpace::Bt709,
            source_transfer: TransferFunction::Pq,
            target_transfer: TransferFunction::Gamma22,
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            source_min_nits: 0.0001,
            target_min_nits: 0.0,
            tone_map_algorithm: ToneMapAlgorithm::Bt2390,
            gamut_map_algorithm: GamutMappingAlgorithm::SoftClip,
            use_dynamic_metadata: true,
            preserve_luminance: true,
            highlight_desaturation: 0.3,
        }
    }
}

impl HdrProcessorConfig {
    /// Create configuration for HDR10 to SDR conversion.
    pub fn hdr10_to_sdr() -> Self {
        Self {
            mode: ProcessingMode::HdrToSdr,
            source_format: HdrFormat::Hdr10,
            target_format: HdrFormat::Sdr,
            source_colorspace: ColorSpace::Bt2020,
            target_colorspace: ColorSpace::Bt709,
            source_transfer: TransferFunction::Pq,
            target_transfer: TransferFunction::Gamma22,
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            ..Default::default()
        }
    }

    /// Create configuration for HLG to SDR conversion.
    pub fn hlg_to_sdr() -> Self {
        Self {
            mode: ProcessingMode::HdrToSdr,
            source_format: HdrFormat::Hlg,
            target_format: HdrFormat::Sdr,
            source_colorspace: ColorSpace::Bt2020,
            target_colorspace: ColorSpace::Bt709,
            source_transfer: TransferFunction::Hlg,
            target_transfer: TransferFunction::Gamma22,
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            ..Default::default()
        }
    }

    /// Create configuration for SDR to HDR10 conversion.
    pub fn sdr_to_hdr10() -> Self {
        Self {
            mode: ProcessingMode::SdrToHdr,
            source_format: HdrFormat::Sdr,
            target_format: HdrFormat::Hdr10,
            source_colorspace: ColorSpace::Bt709,
            target_colorspace: ColorSpace::Bt2020,
            source_transfer: TransferFunction::Gamma22,
            target_transfer: TransferFunction::Pq,
            source_peak_nits: 100.0,
            target_peak_nits: 1000.0,
            ..Default::default()
        }
    }

    /// Create configuration for PQ to HLG conversion.
    pub fn pq_to_hlg() -> Self {
        Self {
            mode: ProcessingMode::HdrToHdr,
            source_format: HdrFormat::Pq,
            target_format: HdrFormat::Hlg,
            source_colorspace: ColorSpace::Bt2020,
            target_colorspace: ColorSpace::Bt2020,
            source_transfer: TransferFunction::Pq,
            target_transfer: TransferFunction::Hlg,
            source_peak_nits: 1000.0,
            target_peak_nits: 1000.0,
            ..Default::default()
        }
    }

    /// Create configuration for HLG to PQ conversion.
    pub fn hlg_to_pq() -> Self {
        Self {
            mode: ProcessingMode::HdrToHdr,
            source_format: HdrFormat::Hlg,
            target_format: HdrFormat::Pq,
            source_colorspace: ColorSpace::Bt2020,
            target_colorspace: ColorSpace::Bt2020,
            source_transfer: TransferFunction::Hlg,
            target_transfer: TransferFunction::Pq,
            source_peak_nits: 1000.0,
            target_peak_nits: 1000.0,
            ..Default::default()
        }
    }
}

/// HDR processor for frame-by-frame processing.
pub struct HdrProcessor {
    /// Processing configuration
    config: HdrProcessorConfig,
    /// Color space converter
    colorspace_converter: Option<ColorSpaceConverter>,
    /// Transfer function converter
    transfer_converter: TransferConverter,
    /// Tone mapper
    tone_mapper: Option<ToneMapper>,
    /// Gamut mapper
    gamut_mapper: Option<GamutMapper>,
    /// Current frame metadata (for HDR10+)
    current_metadata: Option<Hdr10PlusDynamicMetadata>,
    /// Frame counter
    frame_count: u64,
}

impl HdrProcessor {
    /// Create a new HDR processor with the given configuration.
    pub fn new(config: HdrProcessorConfig) -> Self {
        // Set up color space converter if needed
        let colorspace_converter = if config.source_colorspace != config.target_colorspace {
            Some(ColorSpaceConverter::new(
                config.source_colorspace,
                config.target_colorspace,
            ))
        } else {
            None
        };

        // Set up transfer function converter
        let transfer_converter = TransferConverter::new(
            config.source_transfer,
            config.target_transfer,
        ).with_reference_luminance(config.target_peak_nits);

        // Set up tone mapper for HDR to SDR
        let tone_mapper = if config.mode == ProcessingMode::HdrToSdr {
            let tm_config = ToneMapConfig {
                source_peak: config.source_peak_nits,
                target_peak: config.target_peak_nits,
                source_min: config.source_min_nits,
                target_min: config.target_min_nits,
                highlight_desaturation: config.highlight_desaturation,
                use_bt2020_luminance: config.source_colorspace == ColorSpace::Bt2020,
                ..Default::default()
            };
            Some(ToneMapper::new(config.tone_map_algorithm, tm_config))
        } else {
            None
        };

        // Set up gamut mapper if needed
        let gamut_mapper = if config.source_colorspace != config.target_colorspace {
            let gm_config = GamutMapConfig {
                source: config.source_colorspace,
                target: config.target_colorspace,
                algorithm: config.gamut_map_algorithm,
                preserve_luminance: config.preserve_luminance,
                ..Default::default()
            };
            Some(GamutMapper::new(gm_config))
        } else {
            None
        };

        Self {
            config,
            colorspace_converter,
            transfer_converter,
            tone_mapper,
            gamut_mapper,
            current_metadata: None,
            frame_count: 0,
        }
    }

    /// Create an HDR10 to SDR processor.
    pub fn hdr10_to_sdr() -> Self {
        Self::new(HdrProcessorConfig::hdr10_to_sdr())
    }

    /// Create an HLG to SDR processor.
    pub fn hlg_to_sdr() -> Self {
        Self::new(HdrProcessorConfig::hlg_to_sdr())
    }

    /// Create an SDR to HDR10 processor.
    pub fn sdr_to_hdr10() -> Self {
        Self::new(HdrProcessorConfig::sdr_to_hdr10())
    }

    /// Create a PQ to HLG processor.
    pub fn pq_to_hlg() -> Self {
        Self::new(HdrProcessorConfig::pq_to_hlg())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HdrProcessorConfig {
        &self.config
    }

    /// Get the number of processed frames.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Set dynamic metadata for the current frame.
    pub fn set_frame_metadata(&mut self, metadata: Hdr10PlusDynamicMetadata) {
        self.current_metadata = Some(metadata);
    }

    /// Clear dynamic metadata.
    pub fn clear_frame_metadata(&mut self) {
        self.current_metadata = None;
    }

    /// Process a single RGB pixel.
    /// Input: RGB in source transfer function encoding
    /// Output: RGB in target transfer function encoding
    pub fn process_pixel(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        match self.config.mode {
            ProcessingMode::HdrToSdr => self.process_hdr_to_sdr(r, g, b),
            ProcessingMode::SdrToHdr => self.process_sdr_to_hdr(r, g, b),
            ProcessingMode::HdrToHdr => self.process_hdr_to_hdr(r, g, b),
            ProcessingMode::Passthrough => (r, g, b),
        }
    }

    /// Process HDR to SDR conversion.
    fn process_hdr_to_sdr(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Step 1: Convert to linear light
        let (lr, lg, lb) = self.to_linear(r, g, b);

        // Step 2: Apply tone mapping
        let (tr, tg, tb) = if let Some(ref tone_mapper) = self.tone_mapper {
            tone_mapper.map_rgb(lr, lg, lb)
        } else {
            (lr, lg, lb)
        };

        // Step 3: Convert color space
        let (cr, cg, cb) = if let Some(ref converter) = self.colorspace_converter {
            converter.convert(tr, tg, tb)
        } else {
            (tr, tg, tb)
        };

        // Step 4: Apply gamut mapping
        let (gr, gg, gb) = if let Some(ref mapper) = self.gamut_mapper {
            mapper.map(cr, cg, cb)
        } else {
            (cr, cg, cb)
        };

        // Step 5: Convert to target transfer function
        self.encode_transfer(gr, gg, gb)
    }

    /// Process SDR to HDR conversion (inverse tone mapping).
    fn process_sdr_to_hdr(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Step 1: Convert to linear light
        let (lr, lg, lb) = self.to_linear(r, g, b);

        // Step 2: Convert color space
        let (cr, cg, cb) = if let Some(ref converter) = self.colorspace_converter {
            converter.convert(lr, lg, lb)
        } else {
            (lr, lg, lb)
        };

        // Step 3: Apply inverse tone mapping (simple expansion)
        let expand_factor = self.config.target_peak_nits / self.config.source_peak_nits;
        let (er, eg, eb) = (
            cr * expand_factor,
            cg * expand_factor,
            cb * expand_factor,
        );

        // Step 4: Convert to target transfer function
        self.encode_transfer(er, eg, eb)
    }

    /// Process HDR to HDR format conversion.
    fn process_hdr_to_hdr(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        // Step 1: Convert to linear light
        let (lr, lg, lb) = self.to_linear(r, g, b);

        // Step 2: Convert color space if needed
        let (cr, cg, cb) = if let Some(ref converter) = self.colorspace_converter {
            converter.convert(lr, lg, lb)
        } else {
            (lr, lg, lb)
        };

        // Step 3: Apply luminance scaling if peaks differ
        let (sr, sg, sb) = if (self.config.source_peak_nits - self.config.target_peak_nits).abs() > 1.0 {
            let scale = self.config.target_peak_nits / self.config.source_peak_nits;
            (cr * scale, cg * scale, cb * scale)
        } else {
            (cr, cg, cb)
        };

        // Step 4: Convert to target transfer function
        self.encode_transfer(sr, sg, sb)
    }

    /// Convert to linear light using source transfer function.
    fn to_linear(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let lr = self.transfer_converter.to_linear(r);
        let lg = self.transfer_converter.to_linear(g);
        let lb = self.transfer_converter.to_linear(b);
        (lr, lg, lb)
    }

    /// Convert from linear light using target transfer function.
    fn encode_transfer(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let tr = self.transfer_converter.from_linear(r);
        let tg = self.transfer_converter.from_linear(g);
        let tb = self.transfer_converter.from_linear(b);
        (tr.clamp(0.0, 1.0), tg.clamp(0.0, 1.0), tb.clamp(0.0, 1.0))
    }

    /// Process a frame of RGB pixels.
    /// Pixels are stored as contiguous RGB triplets.
    pub fn process_frame(&mut self, pixels: &mut [f64]) -> Result<()> {
        if pixels.len() % 3 != 0 {
            return Err(HdrError::Processing(
                "Pixel buffer length must be multiple of 3".into(),
            ));
        }

        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.process_pixel(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Process a frame of f32 RGB pixels.
    pub fn process_frame_f32(&mut self, pixels: &mut [f32]) -> Result<()> {
        if pixels.len() % 3 != 0 {
            return Err(HdrError::Processing(
                "Pixel buffer length must be multiple of 3".into(),
            ));
        }

        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.process_pixel(
                chunk[0] as f64,
                chunk[1] as f64,
                chunk[2] as f64,
            );
            chunk[0] = r as f32;
            chunk[1] = g as f32;
            chunk[2] = b as f32;
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Process a frame with separate R, G, B planes.
    pub fn process_frame_planar(
        &mut self,
        r_plane: &mut [f64],
        g_plane: &mut [f64],
        b_plane: &mut [f64],
    ) -> Result<()> {
        if r_plane.len() != g_plane.len() || g_plane.len() != b_plane.len() {
            return Err(HdrError::Processing(
                "Plane sizes must match".into(),
            ));
        }

        for i in 0..r_plane.len() {
            let (r, g, b) = self.process_pixel(r_plane[i], g_plane[i], b_plane[i]);
            r_plane[i] = r;
            g_plane[i] = g;
            b_plane[i] = b;
        }

        self.frame_count += 1;
        Ok(())
    }
}

/// Batch processor for processing multiple frames.
pub struct BatchProcessor {
    /// HDR processor
    processor: HdrProcessor,
    /// Frame metadata queue
    metadata_queue: Vec<Option<Hdr10PlusDynamicMetadata>>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(config: HdrProcessorConfig) -> Self {
        Self {
            processor: HdrProcessor::new(config),
            metadata_queue: Vec::new(),
        }
    }

    /// Add metadata for a frame.
    pub fn add_frame_metadata(&mut self, frame_index: usize, metadata: Hdr10PlusDynamicMetadata) {
        // Ensure queue is large enough
        while self.metadata_queue.len() <= frame_index {
            self.metadata_queue.push(None);
        }
        self.metadata_queue[frame_index] = Some(metadata);
    }

    /// Process a batch of frames.
    pub fn process_batch(&mut self, frames: &mut [Vec<f64>]) -> Result<()> {
        for (i, frame) in frames.iter_mut().enumerate() {
            // Set metadata if available
            if let Some(Some(metadata)) = self.metadata_queue.get(i) {
                self.processor.set_frame_metadata(metadata.clone());
            } else {
                self.processor.clear_frame_metadata();
            }

            self.processor.process_frame(frame)?;
        }
        Ok(())
    }

    /// Get the underlying processor.
    pub fn processor(&self) -> &HdrProcessor {
        &self.processor
    }

    /// Get the underlying processor mutably.
    pub fn processor_mut(&mut self) -> &mut HdrProcessor {
        &mut self.processor
    }
}

/// Quick conversion functions for common use cases.
pub mod quick {
    use super::*;

    /// Convert HDR10 frame to SDR.
    pub fn hdr10_to_sdr(pixels: &mut [f64]) -> Result<()> {
        let mut processor = HdrProcessor::hdr10_to_sdr();
        processor.process_frame(pixels)
    }

    /// Convert HLG frame to SDR.
    pub fn hlg_to_sdr(pixels: &mut [f64]) -> Result<()> {
        let mut processor = HdrProcessor::hlg_to_sdr();
        processor.process_frame(pixels)
    }

    /// Convert SDR frame to HDR10.
    pub fn sdr_to_hdr10(pixels: &mut [f64]) -> Result<()> {
        let mut processor = HdrProcessor::sdr_to_hdr10();
        processor.process_frame(pixels)
    }

    /// Convert PQ to HLG.
    pub fn pq_to_hlg(pixels: &mut [f64]) -> Result<()> {
        let mut processor = HdrProcessor::pq_to_hlg();
        processor.process_frame(pixels)
    }

    /// Convert HLG to PQ.
    pub fn hlg_to_pq(pixels: &mut [f64]) -> Result<()> {
        let mut processor = HdrProcessor::new(HdrProcessorConfig::hlg_to_pq());
        processor.process_frame(pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-4;

    #[test]
    fn test_hdr_processor_creation() {
        let processor = HdrProcessor::hdr10_to_sdr();
        assert_eq!(processor.config().mode, ProcessingMode::HdrToSdr);
        assert_eq!(processor.config().source_format, HdrFormat::Hdr10);
        assert_eq!(processor.config().target_format, HdrFormat::Sdr);
    }

    #[test]
    fn test_hdr_to_sdr_black() {
        let processor = HdrProcessor::hdr10_to_sdr();
        let (r, g, b) = processor.process_pixel(0.0, 0.0, 0.0);
        assert!(r.abs() < EPSILON);
        assert!(g.abs() < EPSILON);
        assert!(b.abs() < EPSILON);
    }

    #[test]
    fn test_hdr_to_sdr_white() {
        let processor = HdrProcessor::hdr10_to_sdr();
        // PQ-encoded white (approximately 100 nits)
        let pq_100nits = 0.508;
        let (r, g, b) = processor.process_pixel(pq_100nits, pq_100nits, pq_100nits);

        // Should be close to neutral gray/white
        assert!(r > 0.0 && r <= 1.0);
        assert!(g > 0.0 && g <= 1.0);
        assert!(b > 0.0 && b <= 1.0);
        // RGB should be roughly equal for neutral input
        assert!((r - g).abs() < 0.1);
        assert!((g - b).abs() < 0.1);
    }

    #[test]
    fn test_process_frame() {
        let mut processor = HdrProcessor::hdr10_to_sdr();
        let mut pixels = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5];

        processor.process_frame(&mut pixels).unwrap();

        assert_eq!(processor.frame_count(), 1);
        assert!(pixels.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_process_frame_f32() {
        let mut processor = HdrProcessor::hdr10_to_sdr();
        let mut pixels: Vec<f32> = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5];

        processor.process_frame_f32(&mut pixels).unwrap();

        assert!(pixels.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_process_frame_planar() {
        let mut processor = HdrProcessor::hdr10_to_sdr();
        let mut r_plane = vec![0.5, 0.6];
        let mut g_plane = vec![0.5, 0.6];
        let mut b_plane = vec![0.5, 0.6];

        processor.process_frame_planar(&mut r_plane, &mut g_plane, &mut b_plane).unwrap();

        assert!(r_plane.iter().all(|&v| v.is_finite()));
        assert!(g_plane.iter().all(|&v| v.is_finite()));
        assert!(b_plane.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_sdr_to_hdr() {
        let processor = HdrProcessor::sdr_to_hdr10();
        let (r, g, b) = processor.process_pixel(0.5, 0.5, 0.5);

        // Output should be valid PQ values
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_pq_to_hlg() {
        let processor = HdrProcessor::pq_to_hlg();
        let (r, g, b) = processor.process_pixel(0.5, 0.5, 0.5);

        // Output should be valid HLG values
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_passthrough() {
        let config = HdrProcessorConfig {
            mode: ProcessingMode::Passthrough,
            ..Default::default()
        };
        let processor = HdrProcessor::new(config);
        let (r, g, b) = processor.process_pixel(0.5, 0.3, 0.7);

        assert!((r - 0.5).abs() < EPSILON);
        assert!((g - 0.3).abs() < EPSILON);
        assert!((b - 0.7).abs() < EPSILON);
    }

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::new(HdrProcessorConfig::hdr10_to_sdr());

        let mut frames = vec![
            vec![0.5, 0.5, 0.5],
            vec![0.3, 0.3, 0.3],
        ];

        processor.process_batch(&mut frames).unwrap();

        assert!(frames[0].iter().all(|&v| v.is_finite()));
        assert!(frames[1].iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_quick_conversions() {
        let pixels = vec![0.5, 0.5, 0.5];

        quick::hdr10_to_sdr(&mut pixels.clone()).unwrap();
        quick::hlg_to_sdr(&mut pixels.clone()).unwrap();
        quick::sdr_to_hdr10(&mut pixels.clone()).unwrap();
        quick::pq_to_hlg(&mut pixels.clone()).unwrap();
    }

    #[test]
    fn test_frame_metadata() {
        let mut processor = HdrProcessor::hdr10_to_sdr();

        let metadata = Hdr10PlusDynamicMetadata {
            targeted_system_display_max_luminance: 1000,
            ..Default::default()
        };

        processor.set_frame_metadata(metadata);

        let mut pixels = vec![0.5, 0.5, 0.5];
        processor.process_frame(&mut pixels).unwrap();

        processor.clear_frame_metadata();
    }

    #[test]
    fn test_invalid_buffer_length() {
        let mut processor = HdrProcessor::hdr10_to_sdr();
        let mut pixels = vec![0.5, 0.5]; // Invalid: not multiple of 3

        let result = processor.process_frame(&mut pixels);
        assert!(result.is_err());
    }
}
