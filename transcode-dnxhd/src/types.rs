//! DNxHD type definitions

/// Bit depth for samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 8-bit samples
    Bit8,
    /// 10-bit samples
    Bit10,
    /// 12-bit samples (DNxHR 444)
    Bit12,
}

impl BitDepth {
    /// Get the number of bits
    pub fn bits(&self) -> u8 {
        match self {
            BitDepth::Bit8 => 8,
            BitDepth::Bit10 => 10,
            BitDepth::Bit12 => 12,
        }
    }

    /// Get the maximum sample value
    pub fn max_value(&self) -> u16 {
        match self {
            BitDepth::Bit8 => 255,
            BitDepth::Bit10 => 1023,
            BitDepth::Bit12 => 4095,
        }
    }
}

/// Chroma subsampling format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:2:2 chroma subsampling
    YUV422,
    /// 4:4:4 chroma subsampling (no subsampling)
    YUV444,
}

impl ChromaFormat {
    /// Get the horizontal chroma shift
    pub fn chroma_h_shift(&self) -> u32 {
        match self {
            ChromaFormat::YUV422 => 1,
            ChromaFormat::YUV444 => 0,
        }
    }

    /// Get the vertical chroma shift (always 0 for DNxHD)
    pub fn chroma_v_shift(&self) -> u32 {
        0
    }
}

/// Colorimetry information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colorimetry {
    /// Unknown/unspecified
    Unknown,
    /// ITU-R BT.601
    BT601,
    /// ITU-R BT.709
    BT709,
    /// ITU-R BT.2020
    BT2020,
}

impl Colorimetry {
    /// Parse from header byte
    pub fn from_byte(byte: u8) -> Self {
        match byte & 0x07 {
            1 => Colorimetry::BT601,
            2 => Colorimetry::BT709,
            3 => Colorimetry::BT2020,
            _ => Colorimetry::Unknown,
        }
    }

    /// Convert to header byte
    pub fn to_byte(self) -> u8 {
        match self {
            Colorimetry::Unknown => 0,
            Colorimetry::BT601 => 1,
            Colorimetry::BT709 => 2,
            Colorimetry::BT2020 => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_depth() {
        assert_eq!(BitDepth::Bit8.bits(), 8);
        assert_eq!(BitDepth::Bit10.bits(), 10);
        assert_eq!(BitDepth::Bit12.bits(), 12);

        assert_eq!(BitDepth::Bit8.max_value(), 255);
        assert_eq!(BitDepth::Bit10.max_value(), 1023);
    }

    #[test]
    fn test_chroma_format() {
        assert_eq!(ChromaFormat::YUV422.chroma_h_shift(), 1);
        assert_eq!(ChromaFormat::YUV444.chroma_h_shift(), 0);
    }

    #[test]
    fn test_colorimetry() {
        assert_eq!(Colorimetry::from_byte(1), Colorimetry::BT601);
        assert_eq!(Colorimetry::from_byte(2), Colorimetry::BT709);
        assert_eq!(Colorimetry::BT709.to_byte(), 2);
    }
}
