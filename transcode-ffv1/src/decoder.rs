//! FFV1 decoder implementation.

use crate::error::{Ffv1Error, Result};
use crate::range_coder::RangeDecoder;
use crate::{crc32, Ffv1Config, PredictionContext, SliceInfo};

/// Decoded FFV1 frame.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Plane data (Y, Cb, Cr, optionally Alpha).
    pub planes: Vec<Vec<u16>>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Bits per sample.
    pub bits: u8,
    /// Is keyframe.
    pub keyframe: bool,
}

impl DecodedFrame {
    /// Convert to 8-bit planar format.
    pub fn to_u8_planes(&self) -> Vec<Vec<u8>> {
        let shift = self.bits.saturating_sub(8) as u32;
        self.planes
            .iter()
            .map(|plane| plane.iter().map(|&v| (v >> shift) as u8).collect())
            .collect()
    }

    /// Get Y plane.
    pub fn y_plane(&self) -> &[u16] {
        &self.planes[0]
    }

    /// Get Cb plane (if exists).
    pub fn cb_plane(&self) -> Option<&[u16]> {
        self.planes.get(1).map(|v| v.as_slice())
    }

    /// Get Cr plane (if exists).
    pub fn cr_plane(&self) -> Option<&[u16]> {
        self.planes.get(2).map(|v| v.as_slice())
    }

    /// Get alpha plane (if exists).
    pub fn alpha_plane(&self) -> Option<&[u16]> {
        self.planes.get(3).map(|v| v.as_slice())
    }
}

/// FFV1 decoder.
pub struct Ffv1Decoder {
    config: Ffv1Config,
    /// Context states for each quant table.
    context_states: Vec<Vec<[u8; 32]>>,
    /// Frame counter.
    frame_count: u64,
}

impl Ffv1Decoder {
    /// Create a new FFV1 decoder.
    pub fn new(config: Ffv1Config) -> Result<Self> {
        let context_states = config
            .context_counts
            .iter()
            .map(|&count| vec![[128u8; 32]; count])
            .collect();

        Ok(Self {
            config,
            context_states,
            frame_count: 0,
        })
    }

    /// Get decoder configuration.
    pub fn config(&self) -> &Ffv1Config {
        &self.config
    }

    /// Set frame dimensions.
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
    }

    /// Decode a frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedFrame> {
        if data.is_empty() {
            return Err(Ffv1Error::EndOfStream);
        }

        let keyframe = self.is_keyframe(data)?;

        // Reset context states on keyframe
        if keyframe {
            self.reset_states();
        }

        // Decode frame
        let frame = if self.config.version >= 3 {
            self.decode_frame_v3(data, keyframe)?
        } else {
            self.decode_frame_v0(data, keyframe)?
        };

        self.frame_count += 1;
        Ok(frame)
    }

    /// Check if frame is a keyframe.
    fn is_keyframe(&self, data: &[u8]) -> Result<bool> {
        if data.is_empty() {
            return Err(Ffv1Error::EndOfStream);
        }

        // For version 3+, check frame header
        if self.config.version >= 3 {
            // First bit indicates keyframe
            Ok(data[0] & 0x80 == 0)
        } else {
            // Version 0/1: use range coder
            let mut decoder = RangeDecoder::new(data)?;
            let mut state = 128u8;
            let keyframe = !decoder.get_bit(&mut state)?;
            Ok(keyframe)
        }
    }

    /// Decode frame (version 0/1).
    fn decode_frame_v0(&mut self, data: &[u8], keyframe: bool) -> Result<DecodedFrame> {
        let mut decoder = RangeDecoder::new(data)?;

        // Skip keyframe bit (already read)
        let mut state = 128u8;
        let _ = decoder.get_bit(&mut state)?;

        // Allocate output planes
        let planes = self.allocate_planes();

        // Decode each plane
        let mut decoded_planes = Vec::with_capacity(planes.len());
        for (plane_idx, plane_data) in planes.into_iter().enumerate() {
            let (w, h) = self.config.plane_dimensions(plane_idx);
            let decoded = self.decode_plane(&mut decoder, plane_data, w, h, 0)?;
            decoded_planes.push(decoded);
        }

        Ok(DecodedFrame {
            planes: decoded_planes,
            width: self.config.width,
            height: self.config.height,
            bits: self.config.bits_per_raw_sample,
            keyframe,
        })
    }

    /// Decode frame (version 3+).
    fn decode_frame_v3(&mut self, data: &[u8], keyframe: bool) -> Result<DecodedFrame> {
        // Parse slice structure
        let slices = self.parse_slices(data)?;

        // Allocate output planes
        let mut planes = self.allocate_planes();

        // Decode each slice
        for slice in slices {
            self.decode_slice(data, &slice, &mut planes)?;
        }

        Ok(DecodedFrame {
            planes,
            width: self.config.width,
            height: self.config.height,
            bits: self.config.bits_per_raw_sample,
            keyframe,
        })
    }

    /// Parse slice information from frame data.
    fn parse_slices(&self, data: &[u8]) -> Result<Vec<SliceInfo>> {
        let num_slices = (self.config.num_h_slices * self.config.num_v_slices) as usize;
        let mut slices = Vec::with_capacity(num_slices);

        let slice_width = self.config.width / self.config.num_h_slices;
        let slice_height = self.config.height / self.config.num_v_slices;

        for y in 0..self.config.num_v_slices {
            for x in 0..self.config.num_h_slices {
                let w = if x == self.config.num_h_slices - 1 {
                    self.config.width - x * slice_width
                } else {
                    slice_width
                };

                let h = if y == self.config.num_v_slices - 1 {
                    self.config.height - y * slice_height
                } else {
                    slice_height
                };

                slices.push(SliceInfo {
                    x,
                    y,
                    width: w,
                    height: h,
                    quant_table_index: 0,
                });
            }
        }

        // Verify CRC if enabled
        if self.config.crc_protection && data.len() >= 4 {
            let stored_crc = u32::from_le_bytes([
                data[data.len() - 4],
                data[data.len() - 3],
                data[data.len() - 2],
                data[data.len() - 1],
            ]);
            let computed_crc = crc32(&data[..data.len() - 4]);
            if stored_crc != computed_crc {
                return Err(Ffv1Error::CrcMismatch {
                    expected: stored_crc,
                    actual: computed_crc,
                });
            }
        }

        Ok(slices)
    }

    /// Decode a single slice.
    fn decode_slice(
        &mut self,
        data: &[u8],
        slice: &SliceInfo,
        planes: &mut [Vec<u16>],
    ) -> Result<()> {
        let crc_len = if self.config.crc_protection { 4 } else { 0 };
        let slice_data = &data[..data.len().saturating_sub(crc_len)];

        let mut decoder = RangeDecoder::new(slice_data)?;

        // Decode each plane in the slice
        for (plane_idx, plane) in planes.iter_mut().enumerate() {
            let (plane_w, _plane_h) = self.config.plane_dimensions(plane_idx);

            // Calculate slice region in this plane
            let h_shift = if plane_idx > 0 && plane_idx < 3 {
                self.config.chroma_subsampling.h_shift()
            } else {
                0
            };
            let v_shift = if plane_idx > 0 && plane_idx < 3 {
                self.config.chroma_subsampling.v_shift()
            } else {
                0
            };

            let slice_x = (slice.x * slice.width) >> h_shift;
            let slice_y = (slice.y * slice.height) >> v_shift;
            let slice_w = slice.width >> h_shift;
            let slice_h = slice.height >> v_shift;

            // Decode samples in slice region
            self.decode_slice_region(
                &mut decoder,
                plane,
                plane_w,
                slice_x,
                slice_y,
                slice_w,
                slice_h,
                slice.quant_table_index,
            )?;
        }

        Ok(())
    }

    /// Decode a region of a plane.
    fn decode_slice_region(
        &mut self,
        decoder: &mut RangeDecoder,
        plane: &mut [u16],
        plane_stride: u32,
        slice_x: u32,
        slice_y: u32,
        slice_w: u32,
        slice_h: u32,
        quant_idx: usize,
    ) -> Result<()> {
        let max_val = (1u32 << self.config.bits_per_raw_sample) - 1;
        let quant_table = &self.config.quant_tables[quant_idx];

        for y in 0..slice_h {
            let row_start = ((slice_y + y) * plane_stride + slice_x) as usize;

            for x in 0..slice_w {
                let idx = row_start + x as usize;

                // Get prediction context
                let ctx = self.get_prediction_context(plane, plane_stride, slice_x + x, slice_y + y);

                // Calculate prediction
                let pred = ctx.median_pred();

                // Get context index from gradients
                let context_idx = self.get_context_index(quant_table, &ctx);

                // Decode residual
                let residual =
                    decoder.get_symbol(&mut self.context_states[quant_idx][context_idx])?;

                // Apply residual to prediction
                let sample = (pred + residual).clamp(0, max_val as i32) as u16;
                plane[idx] = sample;
            }
        }

        Ok(())
    }

    /// Decode a full plane.
    fn decode_plane(
        &mut self,
        decoder: &mut RangeDecoder,
        mut plane: Vec<u16>,
        width: u32,
        height: u32,
        quant_idx: usize,
    ) -> Result<Vec<u16>> {
        let max_val = (1u32 << self.config.bits_per_raw_sample) - 1;
        let quant_table = &self.config.quant_tables[quant_idx];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Get prediction context
                let ctx = self.get_prediction_context(&plane, width, x, y);

                // Calculate prediction
                let pred = ctx.median_pred();

                // Get context index
                let context_idx = self.get_context_index(quant_table, &ctx);

                // Decode residual
                let residual =
                    decoder.get_symbol(&mut self.context_states[quant_idx][context_idx])?;

                // Apply residual
                let sample = (pred + residual).clamp(0, max_val as i32) as u16;
                plane[idx] = sample;
            }
        }

        Ok(plane)
    }

    /// Get prediction context for a sample.
    fn get_prediction_context(
        &self,
        plane: &[u16],
        stride: u32,
        x: u32,
        y: u32,
    ) -> PredictionContext {
        let idx = |x: u32, y: u32| (y * stride + x) as usize;

        let c = if x > 0 && y > 0 {
            plane[idx(x - 1, y - 1)] as i32
        } else {
            0
        };

        let b = if y > 0 {
            plane[idx(x, y - 1)] as i32
        } else {
            0
        };

        let a = if x > 0 {
            plane[idx(x - 1, y)] as i32
        } else if y > 0 {
            plane[idx(x, y - 1)] as i32
        } else {
            1 << (self.config.bits_per_raw_sample - 1)
        };

        let d = if x + 1 < stride && y > 0 {
            plane[idx(x + 1, y - 1)] as i32
        } else {
            0
        };

        PredictionContext { a, b, c, d }
    }

    /// Get context index from gradients.
    fn get_context_index(&self, quant_table: &[i8; 256], ctx: &PredictionContext) -> usize {
        // Calculate gradients
        let dh = ctx.a - ctx.c; // Horizontal gradient
        let dv = ctx.b - ctx.c; // Vertical gradient

        // Quantize gradients
        let qh = quant_table[((dh + 128).clamp(0, 255)) as usize] as i32;
        let qv = quant_table[((dv + 128).clamp(0, 255)) as usize] as i32;

        // Combine into context index
        let context = qh + qv * 16;
        (context.unsigned_abs() as usize) % self.config.context_counts[0]
    }

    /// Allocate planes for output.
    fn allocate_planes(&self) -> Vec<Vec<u16>> {
        (0..self.config.num_planes())
            .map(|i| {
                let (w, h) = self.config.plane_dimensions(i);
                vec![0u16; (w * h) as usize]
            })
            .collect()
    }

    /// Reset context states.
    fn reset_states(&mut self) {
        for states in &mut self.context_states {
            for state in states.iter_mut() {
                state.fill(128);
            }
        }
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.reset_states();
        self.frame_count = 0;
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Ffv1Config {
        let mut config = Ffv1Config::default();
        config.width = 64;
        config.height = 64;
        config.bits_per_raw_sample = 8;
        config
    }

    #[test]
    fn test_decoder_creation() {
        let config = create_test_config();
        let decoder = Ffv1Decoder::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_reset() {
        let config = create_test_config();
        let mut decoder = Ffv1Decoder::new(config).unwrap();
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_set_dimensions() {
        let config = create_test_config();
        let mut decoder = Ffv1Decoder::new(config).unwrap();
        decoder.set_dimensions(1920, 1080);
        assert_eq!(decoder.config().width, 1920);
        assert_eq!(decoder.config().height, 1080);
    }

    #[test]
    fn test_allocate_planes() {
        let config = create_test_config();
        let decoder = Ffv1Decoder::new(config).unwrap();
        let planes = decoder.allocate_planes();
        assert_eq!(planes.len(), 3); // Y, Cb, Cr
    }

    #[test]
    fn test_prediction_context() {
        let config = create_test_config();
        let decoder = Ffv1Decoder::new(config).unwrap();

        // Create a simple plane
        let plane = vec![100u16; 64 * 64];
        let ctx = decoder.get_prediction_context(&plane, 64, 32, 32);

        assert_eq!(ctx.a, 100);
        assert_eq!(ctx.b, 100);
        assert_eq!(ctx.c, 100);
    }

    #[test]
    fn test_decoded_frame_to_u8() {
        let frame = DecodedFrame {
            planes: vec![vec![256u16; 100], vec![128u16; 25], vec![128u16; 25]],
            width: 10,
            height: 10,
            bits: 8,
            keyframe: true,
        };

        let u8_planes = frame.to_u8_planes();
        assert_eq!(u8_planes.len(), 3);
    }

    #[test]
    fn test_decoded_frame_accessors() {
        let frame = DecodedFrame {
            planes: vec![
                vec![100u16; 100],
                vec![50u16; 25],
                vec![60u16; 25],
                vec![200u16; 100],
            ],
            width: 10,
            height: 10,
            bits: 8,
            keyframe: true,
        };

        assert_eq!(frame.y_plane().len(), 100);
        assert!(frame.cb_plane().is_some());
        assert!(frame.cr_plane().is_some());
        assert!(frame.alpha_plane().is_some());
    }

    #[test]
    fn test_empty_data_error() {
        let config = create_test_config();
        let mut decoder = Ffv1Decoder::new(config).unwrap();
        let result = decoder.decode(&[]);
        assert!(matches!(result, Err(Ffv1Error::EndOfStream)));
    }
}
