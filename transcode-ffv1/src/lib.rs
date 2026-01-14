//! FFV1 lossless video codec implementation.
//!
//! FFV1 is a lossless intra-frame video codec developed by FFmpeg.
//! It provides efficient lossless compression for archival purposes.
//!
//! ## Features
//!
//! - Lossless video compression
//! - Multiple bit depths (8, 9, 10, 12, 14, 16)
//! - Multiple color spaces (YCbCr, RGB, GBRP)
//! - Slice-based encoding for parallelism
//! - CRC32 integrity checking (version 3+)
//!
//! ## Example
//!
//! ```ignore
//! use transcode_ffv1::{Ffv1Decoder, Ffv1Config};
//!
//! let config = Ffv1Config::from_record(&config_record)?;
//! let mut decoder = Ffv1Decoder::new(config)?;
//! let frame = decoder.decode(&encoded_frame)?;
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod range_coder;
pub mod decoder;

#[cfg(feature = "encoder")]
pub mod encoder;

pub use error::{Ffv1Error, Result};
pub use decoder::{Ffv1Decoder, DecodedFrame};
pub use range_coder::{RangeDecoder, State};

#[cfg(feature = "encoder")]
pub use range_coder::RangeEncoder;

#[cfg(feature = "encoder")]
pub use encoder::{Ffv1Encoder, Ffv1EncoderConfig};

/// FFV1 version 0.
pub const FFV1_VERSION_0: u8 = 0;
/// FFV1 version 1.
pub const FFV1_VERSION_1: u8 = 1;
/// FFV1 version 3 (most common).
pub const FFV1_VERSION_3: u8 = 3;

/// Maximum slice count.
pub const MAX_SLICES: usize = 256;

/// Default quant table count.
pub const DEFAULT_QUANT_TABLES: usize = 1;

/// Context count for different quant table sizes.
pub const CONTEXT_COUNT_DEFAULT: usize = 32;

/// FFV1 color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorSpace {
    /// YCbCr (4:2:0, 4:2:2, 4:4:4).
    YCbCr = 0,
    /// RGB planar.
    Rgb = 1,
}

impl ColorSpace {
    /// Create from u8 value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::YCbCr),
            1 => Some(Self::Rgb),
            _ => None,
        }
    }
}

/// Chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 (no subsampling).
    Yuv444,
    /// 4:2:2 (horizontal subsampling).
    Yuv422,
    /// 4:2:0 (horizontal and vertical subsampling).
    Yuv420,
    /// 4:1:1.
    Yuv411,
    /// 4:1:0.
    Yuv410,
}

impl ChromaSubsampling {
    /// Get horizontal chroma shift.
    pub fn h_shift(&self) -> u32 {
        match self {
            Self::Yuv444 => 0,
            Self::Yuv422 | Self::Yuv420 => 1,
            Self::Yuv411 | Self::Yuv410 => 2,
        }
    }

    /// Get vertical chroma shift.
    pub fn v_shift(&self) -> u32 {
        match self {
            Self::Yuv444 | Self::Yuv422 | Self::Yuv411 => 0,
            Self::Yuv420 => 1,
            Self::Yuv410 => 2,
        }
    }
}

/// FFV1 configuration from configuration record.
#[derive(Debug, Clone)]
pub struct Ffv1Config {
    /// FFV1 version (0, 1, or 3).
    pub version: u8,
    /// Micro version (for version 3+).
    pub micro_version: u8,
    /// Color space.
    pub colorspace: ColorSpace,
    /// Chroma subsampling.
    pub chroma_subsampling: ChromaSubsampling,
    /// Has alpha plane.
    pub has_alpha: bool,
    /// Bits per raw sample.
    pub bits_per_raw_sample: u8,
    /// Number of horizontal slices.
    pub num_h_slices: u32,
    /// Number of vertical slices.
    pub num_v_slices: u32,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// CRC protection (version 3+).
    pub crc_protection: bool,
    /// Quantization table count.
    pub quant_table_count: usize,
    /// Quantization tables.
    pub quant_tables: Vec<[i8; 256]>,
    /// Context count per table.
    pub context_counts: Vec<usize>,
    /// Initial states.
    pub initial_states: Vec<Vec<[u8; 2]>>,
}

impl Default for Ffv1Config {
    fn default() -> Self {
        Self {
            version: FFV1_VERSION_3,
            micro_version: 4,
            colorspace: ColorSpace::YCbCr,
            chroma_subsampling: ChromaSubsampling::Yuv420,
            has_alpha: false,
            bits_per_raw_sample: 8,
            num_h_slices: 1,
            num_v_slices: 1,
            width: 0,
            height: 0,
            crc_protection: true,
            quant_table_count: 1,
            quant_tables: vec![Self::default_quant_table()],
            context_counts: vec![CONTEXT_COUNT_DEFAULT],
            initial_states: vec![vec![[128, 128]; CONTEXT_COUNT_DEFAULT]],
        }
    }
}

impl Ffv1Config {
    /// Create configuration for given parameters.
    pub fn new(width: u32, height: u32, bits: u8) -> Result<Self> {
        if bits != 8 && bits != 10 && bits != 12 && bits != 16 {
            return Err(Ffv1Error::InvalidConfigRecord);
        }

        Ok(Self {
            width,
            height,
            bits_per_raw_sample: bits,
            ..Default::default()
        })
    }

    /// Parse from configuration record (version 3+).
    pub fn from_record(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Ffv1Error::InvalidConfigRecord);
        }

        // Parse using range decoder
        let mut decoder = RangeDecoder::new(data)?;
        let mut state = [128u8; 128];

        // Read configuration parameters
        let version = Self::get_symbol(&mut decoder, &mut state, 0)?;
        if version > 3 {
            return Err(Ffv1Error::UnsupportedVersion(version));
        }

        let micro_version = if version >= 3 {
            Self::get_symbol(&mut decoder, &mut state, 0)?
        } else {
            0
        };

        let colorspace_val = Self::get_symbol(&mut decoder, &mut state, 0)?;
        let colorspace = ColorSpace::from_u8(colorspace_val)
            .ok_or(Ffv1Error::UnsupportedColorSpace(colorspace_val))?;

        let bits_per_raw_sample = if version >= 1 {
            Self::get_symbol(&mut decoder, &mut state, 0)?
        } else {
            8
        };

        let h_chroma_subsample = Self::get_symbol(&mut decoder, &mut state, 0)?;
        let v_chroma_subsample = Self::get_symbol(&mut decoder, &mut state, 0)?;

        let chroma_subsampling = match (h_chroma_subsample, v_chroma_subsample) {
            (0, 0) => ChromaSubsampling::Yuv444,
            (1, 0) => ChromaSubsampling::Yuv422,
            (1, 1) => ChromaSubsampling::Yuv420,
            (2, 0) => ChromaSubsampling::Yuv411,
            (2, 2) => ChromaSubsampling::Yuv410,
            _ => ChromaSubsampling::Yuv420,
        };

        let has_alpha = Self::get_symbol(&mut decoder, &mut state, 0)? != 0;

        let num_h_slices = if version >= 3 {
            Self::get_symbol(&mut decoder, &mut state, 0)? as u32 + 1
        } else {
            1
        };

        let num_v_slices = if version >= 3 {
            Self::get_symbol(&mut decoder, &mut state, 0)? as u32 + 1
        } else {
            1
        };

        let quant_table_count = if version >= 3 {
            Self::get_symbol(&mut decoder, &mut state, 0)? as usize + 1
        } else {
            1
        };

        let crc_protection = version >= 3;

        // Parse quantization tables
        let mut quant_tables = Vec::with_capacity(quant_table_count);
        let mut context_counts = Vec::with_capacity(quant_table_count);

        for _ in 0..quant_table_count {
            let (table, count) = Self::parse_quant_table(&mut decoder, &mut state)?;
            quant_tables.push(table);
            context_counts.push(count);
        }

        // Initialize states
        let initial_states: Vec<Vec<[u8; 2]>> = context_counts
            .iter()
            .map(|&count| vec![[128, 128]; count])
            .collect();

        Ok(Self {
            version,
            micro_version,
            colorspace,
            chroma_subsampling,
            has_alpha,
            bits_per_raw_sample,
            num_h_slices,
            num_v_slices,
            width: 0,
            height: 0,
            crc_protection,
            quant_table_count,
            quant_tables,
            context_counts,
            initial_states,
        })
    }

    /// Get symbol using range decoder.
    fn get_symbol(decoder: &mut RangeDecoder, state: &mut [u8], _context: usize) -> Result<u8> {
        let mut value = 0u8;
        for (i, s) in state.iter_mut().take(8).enumerate() {
            let bit = decoder.get_bit(s)?;
            value |= (bit as u8) << i;
        }
        Ok(value)
    }

    /// Parse quantization table.
    fn parse_quant_table(
        decoder: &mut RangeDecoder,
        state: &mut [u8],
    ) -> Result<([i8; 256], usize)> {
        let mut table = [0i8; 256];
        let mut context_count = 1usize;

        // Parse table using range decoder
        let mut i = 0usize;
        while i < 256 {
            let len = Self::get_symbol(decoder, state, 0)? as usize;
            if len == 0 || i + len > 256 {
                break;
            }

            let val = (context_count as i8) / 2;
            for _ in 0..len {
                table[i] = val;
                i += 1;
            }
            context_count += 1;
        }

        // Fill remaining with last value
        let last_val = if context_count > 1 {
            (context_count as i8) / 2
        } else {
            0
        };
        while i < 256 {
            table[i] = last_val;
            i += 1;
        }

        Ok((table, context_count.max(CONTEXT_COUNT_DEFAULT)))
    }

    /// Get default quantization table.
    pub fn default_quant_table() -> [i8; 256] {
        let mut table = [0i8; 256];

        // Standard FFV1 quant table
        for (i, entry) in table.iter_mut().enumerate() {
            let signed_i = i as i16 - 128;
            *entry = match signed_i.abs() {
                0 => 0,
                1..=2 => signed_i.signum() as i8,
                3..=6 => (signed_i.signum() * 2) as i8,
                7..=14 => (signed_i.signum() * 3) as i8,
                15..=30 => (signed_i.signum() * 4) as i8,
                31..=62 => (signed_i.signum() * 5) as i8,
                63..=126 => (signed_i.signum() * 6) as i8,
                _ => (signed_i.signum() * 7) as i8,
            };
        }

        table
    }

    /// Get number of planes.
    pub fn num_planes(&self) -> usize {
        match self.colorspace {
            ColorSpace::YCbCr => {
                if self.has_alpha {
                    4
                } else {
                    3
                }
            }
            ColorSpace::Rgb => {
                if self.has_alpha {
                    4
                } else {
                    3
                }
            }
        }
    }

    /// Get plane dimensions.
    pub fn plane_dimensions(&self, plane: usize) -> (u32, u32) {
        if plane == 0 || self.colorspace == ColorSpace::Rgb || plane == 3 {
            (self.width, self.height)
        } else {
            let h_shift = self.chroma_subsampling.h_shift();
            let v_shift = self.chroma_subsampling.v_shift();
            (
                (self.width + (1 << h_shift) - 1) >> h_shift,
                (self.height + (1 << v_shift) - 1) >> v_shift,
            )
        }
    }
}

/// Slice information.
#[derive(Debug, Clone)]
pub struct SliceInfo {
    /// Slice X position (in slice units).
    pub x: u32,
    /// Slice Y position (in slice units).
    pub y: u32,
    /// Slice width.
    pub width: u32,
    /// Slice height.
    pub height: u32,
    /// Quant table index.
    pub quant_table_index: usize,
}

/// Context for prediction.
#[derive(Debug, Clone, Copy)]
pub struct PredictionContext {
    /// Top-left sample.
    pub a: i32,
    /// Top sample.
    pub b: i32,
    /// Left sample.
    pub c: i32,
    /// Top-right sample.
    pub d: i32,
}

impl PredictionContext {
    /// Median prediction.
    pub fn median_pred(&self) -> i32 {
        let min = self.a.min(self.b);
        let max = self.a.max(self.b);

        if self.c >= max {
            min
        } else if self.c <= min {
            max
        } else {
            self.a + self.b - self.c
        }
    }
}

/// CRC-32 computation for FFV1.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Ffv1Config::default();
        assert_eq!(config.version, FFV1_VERSION_3);
        assert_eq!(config.bits_per_raw_sample, 8);
        assert_eq!(config.colorspace, ColorSpace::YCbCr);
    }

    #[test]
    fn test_config_new() {
        let config = Ffv1Config::new(1920, 1080, 8).unwrap();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.bits_per_raw_sample, 8);
    }

    #[test]
    fn test_config_invalid_bits() {
        let result = Ffv1Config::new(1920, 1080, 9);
        assert!(result.is_err());
    }

    #[test]
    fn test_colorspace_from_u8() {
        assert_eq!(ColorSpace::from_u8(0), Some(ColorSpace::YCbCr));
        assert_eq!(ColorSpace::from_u8(1), Some(ColorSpace::Rgb));
        assert_eq!(ColorSpace::from_u8(2), None);
    }

    #[test]
    fn test_chroma_subsampling() {
        assert_eq!(ChromaSubsampling::Yuv444.h_shift(), 0);
        assert_eq!(ChromaSubsampling::Yuv444.v_shift(), 0);
        assert_eq!(ChromaSubsampling::Yuv420.h_shift(), 1);
        assert_eq!(ChromaSubsampling::Yuv420.v_shift(), 1);
    }

    #[test]
    fn test_num_planes() {
        let mut config = Ffv1Config::default();
        assert_eq!(config.num_planes(), 3);

        config.has_alpha = true;
        assert_eq!(config.num_planes(), 4);
    }

    #[test]
    fn test_plane_dimensions() {
        let mut config = Ffv1Config::default();
        config.width = 1920;
        config.height = 1080;
        config.chroma_subsampling = ChromaSubsampling::Yuv420;

        assert_eq!(config.plane_dimensions(0), (1920, 1080));
        assert_eq!(config.plane_dimensions(1), (960, 540));
        assert_eq!(config.plane_dimensions(2), (960, 540));
    }

    #[test]
    fn test_median_pred() {
        let ctx = PredictionContext {
            a: 100,
            b: 120,
            c: 110,
            d: 0,
        };
        let pred = ctx.median_pred();
        assert!(pred >= 100 && pred <= 120);
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, World!";
        let crc = crc32(data);
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_default_quant_table() {
        let table = Ffv1Config::default_quant_table();
        assert_eq!(table[128], 0); // Center should be 0
    }
}
