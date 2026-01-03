//! ProRes type definitions

/// ProRes profile variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProResProfile {
    /// ProRes 422 Proxy - Lowest quality, highest compression
    Proxy,
    /// ProRes 422 LT - Low quality
    LT,
    /// ProRes 422 Standard - Standard quality
    Standard,
    /// ProRes 422 HQ - High quality
    HQ,
    /// ProRes 4444 - High quality with 4:4:4 chroma and optional alpha
    P4444,
    /// ProRes 4444 XQ - Highest quality with 4:4:4 chroma and optional alpha
    P4444XQ,
}

impl ProResProfile {
    /// Returns the FourCC code for this profile
    pub fn fourcc(&self) -> &'static [u8; 4] {
        match self {
            ProResProfile::Proxy => b"apco",
            ProResProfile::LT => b"apcs",
            ProResProfile::Standard => b"apcn",
            ProResProfile::HQ => b"apch",
            ProResProfile::P4444 => b"ap4h",
            ProResProfile::P4444XQ => b"ap4x",
        }
    }

    /// Parse profile from FourCC code
    pub fn from_fourcc(fourcc: &[u8]) -> Option<ProResProfile> {
        if fourcc.len() < 4 {
            return None;
        }
        match &fourcc[..4] {
            b"apco" => Some(ProResProfile::Proxy),
            b"apcs" => Some(ProResProfile::LT),
            b"apcn" => Some(ProResProfile::Standard),
            b"apch" => Some(ProResProfile::HQ),
            b"ap4h" => Some(ProResProfile::P4444),
            b"ap4x" => Some(ProResProfile::P4444XQ),
            _ => None,
        }
    }

    /// Returns true if this profile supports alpha channel
    pub fn supports_alpha(&self) -> bool {
        matches!(self, ProResProfile::P4444 | ProResProfile::P4444XQ)
    }

    /// Returns true if this is a 4:4:4 profile
    pub fn is_444(&self) -> bool {
        matches!(self, ProResProfile::P4444 | ProResProfile::P4444XQ)
    }

    /// Returns the default bit depth for this profile
    pub fn default_bit_depth(&self) -> u8 {
        match self {
            ProResProfile::P4444XQ => 12,
            _ => 10,
        }
    }

    /// Returns the quantization scale factor for this profile
    pub fn quant_scale(&self) -> u8 {
        match self {
            ProResProfile::Proxy => 4,
            ProResProfile::LT => 1,
            ProResProfile::Standard => 1,
            ProResProfile::HQ => 1,
            ProResProfile::P4444 => 1,
            ProResProfile::P4444XQ => 1,
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
    /// Returns the chroma width divisor
    pub fn chroma_h_shift(&self) -> u32 {
        match self {
            ChromaFormat::YUV422 => 1,
            ChromaFormat::YUV444 => 0,
        }
    }

    /// Returns the chroma height divisor (always 0 for ProRes)
    pub fn chroma_v_shift(&self) -> u32 {
        0
    }
}

/// Interlace mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlaceMode {
    /// Progressive scan
    Progressive,
    /// Interlaced, top field first
    InterlacedTFF,
    /// Interlaced, bottom field first
    InterlacedBFF,
}

impl InterlaceMode {
    /// Parse from frame flags
    pub fn from_flags(interlaced: bool, top_field_first: bool) -> Self {
        if !interlaced {
            InterlaceMode::Progressive
        } else if top_field_first {
            InterlaceMode::InterlacedTFF
        } else {
            InterlaceMode::InterlacedBFF
        }
    }
}

/// Color primaries (based on ITU-T H.273)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    /// Unknown/unspecified
    Unknown,
    /// ITU-R BT.709 (HD)
    BT709,
    /// ITU-R BT.601 (SD NTSC)
    BT601NTSC,
    /// ITU-R BT.601 (SD PAL)
    BT601PAL,
    /// ITU-R BT.2020 (UHD)
    BT2020,
    /// DCI-P3
    DCIP3,
}

impl ColorPrimaries {
    /// Parse from ProRes color primaries code
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => ColorPrimaries::Unknown,
            1 => ColorPrimaries::BT709,
            5 | 6 => ColorPrimaries::BT601NTSC,
            7 => ColorPrimaries::BT601PAL,
            9 => ColorPrimaries::BT2020,
            11 | 12 => ColorPrimaries::DCIP3,
            _ => ColorPrimaries::Unknown,
        }
    }
}

/// Transfer characteristics (gamma)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferCharacteristic {
    /// Unknown/unspecified
    Unknown,
    /// ITU-R BT.709
    BT709,
    /// ITU-R BT.601
    BT601,
    /// SMPTE ST 2084 (PQ/HDR10)
    PQ,
    /// ARIB STD-B67 (HLG)
    HLG,
}

impl TransferCharacteristic {
    /// Parse from ProRes transfer characteristic code
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => TransferCharacteristic::Unknown,
            1 => TransferCharacteristic::BT709,
            6 => TransferCharacteristic::BT601,
            16 => TransferCharacteristic::PQ,
            18 => TransferCharacteristic::HLG,
            _ => TransferCharacteristic::Unknown,
        }
    }
}

/// Matrix coefficients for YCbCr to RGB conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixCoefficients {
    /// Unknown/unspecified
    Unknown,
    /// ITU-R BT.709
    BT709,
    /// ITU-R BT.601
    BT601,
    /// ITU-R BT.2020 non-constant luminance
    BT2020NCL,
    /// ITU-R BT.2020 constant luminance
    BT2020CL,
}

impl MatrixCoefficients {
    /// Parse from ProRes matrix code
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => MatrixCoefficients::Unknown,
            1 => MatrixCoefficients::BT709,
            5 | 6 => MatrixCoefficients::BT601,
            9 => MatrixCoefficients::BT2020NCL,
            10 => MatrixCoefficients::BT2020CL,
            _ => MatrixCoefficients::Unknown,
        }
    }
}

/// Bit depth for samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 10-bit samples
    Bit10,
    /// 12-bit samples
    Bit12,
}

impl BitDepth {
    /// Parse from bit depth code
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 | 2 => Some(BitDepth::Bit10),
            1 | 3 => Some(BitDepth::Bit12),
            _ => None,
        }
    }

    /// Get the number of bits
    pub fn bits(&self) -> u8 {
        match self {
            BitDepth::Bit10 => 10,
            BitDepth::Bit12 => 12,
        }
    }

    /// Get the maximum sample value
    pub fn max_value(&self) -> u16 {
        match self {
            BitDepth::Bit10 => 1023,
            BitDepth::Bit12 => 4095,
        }
    }
}
