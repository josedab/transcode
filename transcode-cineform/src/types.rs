//! CineForm type definitions

/// Bit depth for samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 8-bit samples
    Bit8,
    /// 10-bit samples
    Bit10,
    /// 12-bit samples
    Bit12,
    /// 16-bit samples
    Bit16,
}

impl BitDepth {
    /// Get the number of bits
    pub fn bits(&self) -> u8 {
        match self {
            BitDepth::Bit8 => 8,
            BitDepth::Bit10 => 10,
            BitDepth::Bit12 => 12,
            BitDepth::Bit16 => 16,
        }
    }

    /// Get the maximum sample value
    pub fn max_value(&self) -> u16 {
        match self {
            BitDepth::Bit8 => 255,
            BitDepth::Bit10 => 1023,
            BitDepth::Bit12 => 4095,
            BitDepth::Bit16 => 65535,
        }
    }

    /// Parse from bits
    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            8 => Some(BitDepth::Bit8),
            10 => Some(BitDepth::Bit10),
            12 => Some(BitDepth::Bit12),
            16 => Some(BitDepth::Bit16),
            _ => None,
        }
    }
}

/// Pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// YUV 4:2:2 format
    YUV422,
    /// YUV 4:4:4 format
    YUV444,
    /// RGBA format
    RGBA,
    /// BGRA format
    BGRA,
    /// Bayer pattern (raw sensor data)
    Bayer,
}

impl PixelFormat {
    /// Get number of components
    pub fn components(&self) -> u8 {
        match self {
            PixelFormat::YUV422 => 3,
            PixelFormat::YUV444 => 3,
            PixelFormat::RGBA => 4,
            PixelFormat::BGRA => 4,
            PixelFormat::Bayer => 1,
        }
    }

    /// Get horizontal chroma subsampling
    pub fn chroma_h_shift(&self) -> u32 {
        match self {
            PixelFormat::YUV422 => 1,
            _ => 0,
        }
    }

    /// Check if format has alpha
    pub fn has_alpha(&self) -> bool {
        matches!(self, PixelFormat::RGBA | PixelFormat::BGRA)
    }
}

/// Quality level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Low quality (proxy)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Film Scan 1 quality
    FilmScan1,
    /// Film Scan 2 quality (highest)
    FilmScan2,
}

impl Quality {
    /// Get quantization divisor for this quality level
    pub fn quant_divisor(&self) -> u32 {
        match self {
            Quality::Low => 32,
            Quality::Medium => 16,
            Quality::High => 8,
            Quality::FilmScan1 => 4,
            Quality::FilmScan2 => 2,
        }
    }

    /// Get approximate bits per pixel
    pub fn bits_per_pixel(&self) -> f32 {
        match self {
            Quality::Low => 2.0,
            Quality::Medium => 4.0,
            Quality::High => 6.0,
            Quality::FilmScan1 => 8.0,
            Quality::FilmScan2 => 10.0,
        }
    }

    /// Create from integer quality level (1-5)
    pub fn from_level(level: u8) -> Self {
        match level {
            1 => Quality::Low,
            2 => Quality::Medium,
            3 => Quality::High,
            4 => Quality::FilmScan1,
            _ => Quality::FilmScan2,
        }
    }

    /// Convert to integer level
    pub fn to_level(&self) -> u8 {
        match self {
            Quality::Low => 1,
            Quality::Medium => 2,
            Quality::High => 3,
            Quality::FilmScan1 => 4,
            Quality::FilmScan2 => 5,
        }
    }
}

/// CineForm codec tag values
pub mod tags {
    /// Sample type tag
    pub const SAMPLE_TYPE: u16 = 0x4001;
    /// Sample flags tag
    pub const SAMPLE_FLAGS: u16 = 0x4002;
    /// Image width tag
    pub const IMAGE_WIDTH: u16 = 0x4003;
    /// Image height tag
    pub const IMAGE_HEIGHT: u16 = 0x4004;
    /// Pixel format tag
    pub const PIXEL_FORMAT: u16 = 0x4005;
    /// Bits per component tag
    pub const BITS_PER_COMPONENT: u16 = 0x4006;
    /// Channel count tag
    pub const CHANNEL_COUNT: u16 = 0x4007;
    /// Subband count tag
    pub const SUBBAND_COUNT: u16 = 0x4008;
    /// Encoded format tag
    pub const ENCODED_FORMAT: u16 = 0x4009;
    /// Frame type tag
    pub const FRAME_TYPE: u16 = 0x400A;
    /// Frame index tag
    pub const FRAME_INDEX: u16 = 0x400B;
    /// Quality level tag
    pub const QUALITY_LEVEL: u16 = 0x400C;
    /// Wavelet type tag
    pub const WAVELET_TYPE: u16 = 0x400D;
    /// Transform levels tag
    pub const TRANSFORM_LEVELS: u16 = 0x400E;
    /// Quantization table tag
    pub const QUANT_TABLE: u16 = 0x400F;
    /// Band header tag
    pub const BAND_HEADER: u16 = 0x4010;
    /// Band data tag
    pub const BAND_DATA: u16 = 0x4011;
    /// Band end tag
    pub const BAND_END: u16 = 0x4012;
    /// Channel header tag
    pub const CHANNEL_HEADER: u16 = 0x4020;
    /// Channel end tag
    pub const CHANNEL_END: u16 = 0x4021;
    /// Frame header tag
    pub const FRAME_HEADER: u16 = 0x4030;
    /// Frame end tag
    pub const FRAME_END: u16 = 0x4031;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_depth() {
        assert_eq!(BitDepth::Bit8.bits(), 8);
        assert_eq!(BitDepth::Bit10.bits(), 10);
        assert_eq!(BitDepth::Bit12.bits(), 12);
        assert_eq!(BitDepth::Bit16.bits(), 16);

        assert_eq!(BitDepth::Bit8.max_value(), 255);
        assert_eq!(BitDepth::Bit10.max_value(), 1023);
    }

    #[test]
    fn test_bit_depth_from_bits() {
        assert_eq!(BitDepth::from_bits(8), Some(BitDepth::Bit8));
        assert_eq!(BitDepth::from_bits(10), Some(BitDepth::Bit10));
        assert_eq!(BitDepth::from_bits(14), None);
    }

    #[test]
    fn test_pixel_format() {
        assert_eq!(PixelFormat::YUV422.components(), 3);
        assert_eq!(PixelFormat::RGBA.components(), 4);
        assert!(PixelFormat::RGBA.has_alpha());
        assert!(!PixelFormat::YUV422.has_alpha());
    }

    #[test]
    fn test_quality() {
        assert_eq!(Quality::Low.quant_divisor(), 32);
        assert_eq!(Quality::FilmScan2.quant_divisor(), 2);
        assert_eq!(Quality::from_level(1), Quality::Low);
        assert_eq!(Quality::High.to_level(), 3);
    }
}
