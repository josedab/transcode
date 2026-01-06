//! PCM format definitions.

/// PCM audio formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcmFormat {
    // Unsigned 8-bit
    /// Unsigned 8-bit PCM.
    U8,

    // Signed integers - Little Endian
    /// Signed 8-bit PCM.
    S8,
    /// Signed 16-bit PCM, little endian.
    S16Le,
    /// Signed 24-bit PCM, little endian.
    S24Le,
    /// Signed 32-bit PCM, little endian.
    S32Le,

    // Signed integers - Big Endian
    /// Signed 16-bit PCM, big endian.
    S16Be,
    /// Signed 24-bit PCM, big endian.
    S24Be,
    /// Signed 32-bit PCM, big endian.
    S32Be,

    // Floating point - Little Endian
    /// 32-bit floating point PCM, little endian.
    F32Le,
    /// 64-bit floating point PCM, little endian.
    F64Le,

    // Floating point - Big Endian
    /// 32-bit floating point PCM, big endian.
    F32Be,
    /// 64-bit floating point PCM, big endian.
    F64Be,

    // Packed formats
    /// 24-bit in 32-bit container, little endian (low 24 bits used).
    S24Le32,
    /// 24-bit in 32-bit container, big endian (low 24 bits used).
    S24Be32,
    /// 20-bit in 32-bit container, little endian (low 20 bits used).
    S20Le32,
    /// 20-bit in 32-bit container, big endian (low 20 bits used).
    S20Be32,

    // Compressed PCM
    /// A-law encoded PCM (8-bit).
    ALaw,
    /// mu-law encoded PCM (8-bit).
    MuLaw,

    // Planar formats (for multi-channel, each channel in separate buffer)
    /// Unsigned 8-bit PCM, planar.
    U8Planar,
    /// Signed 16-bit PCM, planar.
    S16Planar,
    /// Signed 32-bit PCM, planar.
    S32Planar,
    /// 32-bit float PCM, planar.
    F32Planar,
    /// 64-bit float PCM, planar.
    F64Planar,

    // Special formats
    /// DVD LPCM (20-bit packed in 24-bit, big endian).
    DvdLpcm,
    /// Blu-ray LPCM (16/24-bit, big endian).
    BlurayLpcm,
}

impl PcmFormat {
    /// Get all supported PCM formats.
    pub fn all() -> &'static [PcmFormat] {
        &[
            PcmFormat::U8,
            PcmFormat::S8,
            PcmFormat::S16Le,
            PcmFormat::S16Be,
            PcmFormat::S24Le,
            PcmFormat::S24Be,
            PcmFormat::S32Le,
            PcmFormat::S32Be,
            PcmFormat::F32Le,
            PcmFormat::F32Be,
            PcmFormat::F64Le,
            PcmFormat::F64Be,
            PcmFormat::S24Le32,
            PcmFormat::S24Be32,
            PcmFormat::S20Le32,
            PcmFormat::S20Be32,
            PcmFormat::ALaw,
            PcmFormat::MuLaw,
            PcmFormat::U8Planar,
            PcmFormat::S16Planar,
            PcmFormat::S32Planar,
            PcmFormat::F32Planar,
            PcmFormat::F64Planar,
            PcmFormat::DvdLpcm,
            PcmFormat::BlurayLpcm,
        ]
    }

    /// Get the short codec name (FFmpeg-compatible).
    pub const fn codec_name(&self) -> &'static str {
        match self {
            PcmFormat::U8 => "pcm_u8",
            PcmFormat::S8 => "pcm_s8",
            PcmFormat::S16Le => "pcm_s16le",
            PcmFormat::S16Be => "pcm_s16be",
            PcmFormat::S24Le => "pcm_s24le",
            PcmFormat::S24Be => "pcm_s24be",
            PcmFormat::S32Le => "pcm_s32le",
            PcmFormat::S32Be => "pcm_s32be",
            PcmFormat::F32Le => "pcm_f32le",
            PcmFormat::F32Be => "pcm_f32be",
            PcmFormat::F64Le => "pcm_f64le",
            PcmFormat::F64Be => "pcm_f64be",
            PcmFormat::S24Le32 => "pcm_s24le_32",
            PcmFormat::S24Be32 => "pcm_s24be_32",
            PcmFormat::S20Le32 => "pcm_s20le_32",
            PcmFormat::S20Be32 => "pcm_s20be_32",
            PcmFormat::ALaw => "pcm_alaw",
            PcmFormat::MuLaw => "pcm_mulaw",
            PcmFormat::U8Planar => "pcm_u8p",
            PcmFormat::S16Planar => "pcm_s16p",
            PcmFormat::S32Planar => "pcm_s32p",
            PcmFormat::F32Planar => "pcm_f32p",
            PcmFormat::F64Planar => "pcm_f64p",
            PcmFormat::DvdLpcm => "pcm_dvd",
            PcmFormat::BlurayLpcm => "pcm_bluray",
        }
    }

    /// Get the long codec name.
    pub const fn codec_long_name(&self) -> &'static str {
        match self {
            PcmFormat::U8 => "PCM unsigned 8-bit",
            PcmFormat::S8 => "PCM signed 8-bit",
            PcmFormat::S16Le => "PCM signed 16-bit little-endian",
            PcmFormat::S16Be => "PCM signed 16-bit big-endian",
            PcmFormat::S24Le => "PCM signed 24-bit little-endian",
            PcmFormat::S24Be => "PCM signed 24-bit big-endian",
            PcmFormat::S32Le => "PCM signed 32-bit little-endian",
            PcmFormat::S32Be => "PCM signed 32-bit big-endian",
            PcmFormat::F32Le => "PCM 32-bit floating point little-endian",
            PcmFormat::F32Be => "PCM 32-bit floating point big-endian",
            PcmFormat::F64Le => "PCM 64-bit floating point little-endian",
            PcmFormat::F64Be => "PCM 64-bit floating point big-endian",
            PcmFormat::S24Le32 => "PCM signed 24-bit in 32-bit container, little-endian",
            PcmFormat::S24Be32 => "PCM signed 24-bit in 32-bit container, big-endian",
            PcmFormat::S20Le32 => "PCM signed 20-bit in 32-bit container, little-endian",
            PcmFormat::S20Be32 => "PCM signed 20-bit in 32-bit container, big-endian",
            PcmFormat::ALaw => "PCM A-law",
            PcmFormat::MuLaw => "PCM mu-law",
            PcmFormat::U8Planar => "PCM unsigned 8-bit planar",
            PcmFormat::S16Planar => "PCM signed 16-bit planar",
            PcmFormat::S32Planar => "PCM signed 32-bit planar",
            PcmFormat::F32Planar => "PCM 32-bit floating point planar",
            PcmFormat::F64Planar => "PCM 64-bit floating point planar",
            PcmFormat::DvdLpcm => "PCM DVD LPCM",
            PcmFormat::BlurayLpcm => "PCM Blu-ray LPCM",
        }
    }

    /// Get bits per sample.
    pub const fn bits_per_sample(&self) -> u8 {
        match self {
            PcmFormat::U8 | PcmFormat::S8 | PcmFormat::ALaw | PcmFormat::MuLaw
            | PcmFormat::U8Planar => 8,
            PcmFormat::S16Le | PcmFormat::S16Be | PcmFormat::S16Planar => 16,
            PcmFormat::S20Le32 | PcmFormat::S20Be32 => 20,
            PcmFormat::S24Le | PcmFormat::S24Be => 24,
            PcmFormat::S24Le32 | PcmFormat::S24Be32 | PcmFormat::DvdLpcm
            | PcmFormat::BlurayLpcm => 24,
            PcmFormat::S32Le | PcmFormat::S32Be | PcmFormat::F32Le | PcmFormat::F32Be
            | PcmFormat::S32Planar | PcmFormat::F32Planar => 32,
            PcmFormat::F64Le | PcmFormat::F64Be | PcmFormat::F64Planar => 64,
        }
    }

    /// Get bytes per sample (may differ from bits_per_sample / 8 for packed formats).
    pub const fn bytes_per_sample(&self) -> u8 {
        match self {
            PcmFormat::U8 | PcmFormat::S8 | PcmFormat::ALaw | PcmFormat::MuLaw
            | PcmFormat::U8Planar => 1,
            PcmFormat::S16Le | PcmFormat::S16Be | PcmFormat::S16Planar => 2,
            PcmFormat::S24Le | PcmFormat::S24Be => 3,
            PcmFormat::S24Le32 | PcmFormat::S24Be32 | PcmFormat::S20Le32
            | PcmFormat::S20Be32 | PcmFormat::S32Le | PcmFormat::S32Be
            | PcmFormat::F32Le | PcmFormat::F32Be | PcmFormat::S32Planar
            | PcmFormat::F32Planar | PcmFormat::DvdLpcm => 4,
            PcmFormat::F64Le | PcmFormat::F64Be | PcmFormat::F64Planar
            | PcmFormat::BlurayLpcm => 8,
        }
    }

    /// Check if format uses floating point.
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            PcmFormat::F32Le | PcmFormat::F32Be | PcmFormat::F64Le
            | PcmFormat::F64Be | PcmFormat::F32Planar | PcmFormat::F64Planar
        )
    }

    /// Check if format is signed.
    pub const fn is_signed(&self) -> bool {
        !matches!(self, PcmFormat::U8 | PcmFormat::U8Planar)
    }

    /// Check if format is big endian.
    pub const fn is_big_endian(&self) -> bool {
        matches!(
            self,
            PcmFormat::S16Be | PcmFormat::S24Be | PcmFormat::S32Be
            | PcmFormat::F32Be | PcmFormat::F64Be | PcmFormat::S24Be32
            | PcmFormat::S20Be32 | PcmFormat::DvdLpcm | PcmFormat::BlurayLpcm
        )
    }

    /// Check if format is planar (each channel in separate buffer).
    pub const fn is_planar(&self) -> bool {
        matches!(
            self,
            PcmFormat::U8Planar | PcmFormat::S16Planar | PcmFormat::S32Planar
            | PcmFormat::F32Planar | PcmFormat::F64Planar
        )
    }

    /// Check if format uses compression (A-law or mu-law).
    pub const fn is_compressed(&self) -> bool {
        matches!(self, PcmFormat::ALaw | PcmFormat::MuLaw)
    }

    /// Parse format from string (FFmpeg-compatible names).
    pub fn from_str(s: &str) -> Option<PcmFormat> {
        match s.to_lowercase().as_str() {
            "pcm_u8" | "u8" => Some(PcmFormat::U8),
            "pcm_s8" | "s8" => Some(PcmFormat::S8),
            "pcm_s16le" | "s16le" | "s16" => Some(PcmFormat::S16Le),
            "pcm_s16be" | "s16be" => Some(PcmFormat::S16Be),
            "pcm_s24le" | "s24le" | "s24" => Some(PcmFormat::S24Le),
            "pcm_s24be" | "s24be" => Some(PcmFormat::S24Be),
            "pcm_s32le" | "s32le" | "s32" => Some(PcmFormat::S32Le),
            "pcm_s32be" | "s32be" => Some(PcmFormat::S32Be),
            "pcm_f32le" | "f32le" | "f32" | "flt" => Some(PcmFormat::F32Le),
            "pcm_f32be" | "f32be" => Some(PcmFormat::F32Be),
            "pcm_f64le" | "f64le" | "f64" | "dbl" => Some(PcmFormat::F64Le),
            "pcm_f64be" | "f64be" => Some(PcmFormat::F64Be),
            "pcm_s24le_32" => Some(PcmFormat::S24Le32),
            "pcm_s24be_32" => Some(PcmFormat::S24Be32),
            "pcm_s20le_32" => Some(PcmFormat::S20Le32),
            "pcm_s20be_32" => Some(PcmFormat::S20Be32),
            "pcm_alaw" | "alaw" => Some(PcmFormat::ALaw),
            "pcm_mulaw" | "mulaw" | "ulaw" => Some(PcmFormat::MuLaw),
            "pcm_u8p" | "u8p" => Some(PcmFormat::U8Planar),
            "pcm_s16p" | "s16p" => Some(PcmFormat::S16Planar),
            "pcm_s32p" | "s32p" => Some(PcmFormat::S32Planar),
            "pcm_f32p" | "f32p" | "fltp" => Some(PcmFormat::F32Planar),
            "pcm_f64p" | "f64p" | "dblp" => Some(PcmFormat::F64Planar),
            "pcm_dvd" | "dvd" => Some(PcmFormat::DvdLpcm),
            "pcm_bluray" | "bluray" => Some(PcmFormat::BlurayLpcm),
            _ => None,
        }
    }
}

impl std::fmt::Display for PcmFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.codec_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_properties() {
        assert_eq!(PcmFormat::S16Le.bits_per_sample(), 16);
        assert_eq!(PcmFormat::S16Le.bytes_per_sample(), 2);
        assert!(PcmFormat::S16Le.is_signed());
        assert!(!PcmFormat::S16Le.is_big_endian());
        assert!(!PcmFormat::S16Le.is_float());
        assert!(!PcmFormat::S16Le.is_planar());
    }

    #[test]
    fn test_float_formats() {
        assert!(PcmFormat::F32Le.is_float());
        assert!(PcmFormat::F64Be.is_float());
        assert!(!PcmFormat::S32Le.is_float());
    }

    #[test]
    fn test_endianness() {
        assert!(!PcmFormat::S16Le.is_big_endian());
        assert!(PcmFormat::S16Be.is_big_endian());
        assert!(PcmFormat::DvdLpcm.is_big_endian());
    }

    #[test]
    fn test_planar_formats() {
        assert!(PcmFormat::F32Planar.is_planar());
        assert!(PcmFormat::S16Planar.is_planar());
        assert!(!PcmFormat::S16Le.is_planar());
    }

    #[test]
    fn test_compressed_formats() {
        assert!(PcmFormat::ALaw.is_compressed());
        assert!(PcmFormat::MuLaw.is_compressed());
        assert!(!PcmFormat::S16Le.is_compressed());
    }

    #[test]
    fn test_from_str() {
        assert_eq!(PcmFormat::from_str("pcm_s16le"), Some(PcmFormat::S16Le));
        assert_eq!(PcmFormat::from_str("s16le"), Some(PcmFormat::S16Le));
        assert_eq!(PcmFormat::from_str("S16LE"), Some(PcmFormat::S16Le));
        assert_eq!(PcmFormat::from_str("fltp"), Some(PcmFormat::F32Planar));
        assert_eq!(PcmFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_24bit_packed() {
        assert_eq!(PcmFormat::S24Le.bytes_per_sample(), 3);
        assert_eq!(PcmFormat::S24Le32.bytes_per_sample(), 4);
        assert_eq!(PcmFormat::S24Le.bits_per_sample(), 24);
        assert_eq!(PcmFormat::S24Le32.bits_per_sample(), 24);
    }
}
