//! MXF type definitions

use std::fmt;

/// Rational number for frame rates and timing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub numerator: i32,
    pub denominator: i32,
}

impl Rational {
    /// Create new rational
    pub fn new(numerator: i32, denominator: i32) -> Self {
        Rational { numerator, denominator }
    }

    /// Convert to f64
    pub fn to_f64(&self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            self.numerator as f64 / self.denominator as f64
        }
    }

    /// Common frame rates
    pub fn fps_24() -> Self {
        Rational::new(24, 1)
    }

    pub fn fps_25() -> Self {
        Rational::new(25, 1)
    }

    pub fn fps_30() -> Self {
        Rational::new(30, 1)
    }

    pub fn fps_23_976() -> Self {
        Rational::new(24000, 1001)
    }

    pub fn fps_29_97() -> Self {
        Rational::new(30000, 1001)
    }

    pub fn fps_59_94() -> Self {
        Rational::new(60000, 1001)
    }
}

impl Default for Rational {
    fn default() -> Self {
        Rational::new(1, 1)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

/// Edit rate (typically same as frame rate for video)
pub type EditRate = Rational;

/// UMID (Unique Material Identifier) - 32 bytes
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Umid(pub [u8; 32]);

impl Umid {
    /// Create new UMID
    pub fn new(bytes: [u8; 32]) -> Self {
        Umid(bytes)
    }

    /// Create zero UMID
    pub fn zero() -> Self {
        Umid([0; 32])
    }

    /// Generate a new random UMID
    pub fn generate() -> Self {
        use uuid::Uuid;

        let mut bytes = [0u8; 32];

        // UMID structure:
        // Bytes 0-11: Universal Label (SMPTE 330M)
        bytes[0..12].copy_from_slice(&[
            0x06, 0x0A, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x05, 0x01, 0x01, 0x0D, 0x20,
        ]);

        // Byte 12: Length (13h = remaining bytes)
        bytes[12] = 0x13;

        // Byte 13: Instance type
        bytes[13] = 0x00;

        // Byte 14-15: Material number generation method
        bytes[14] = 0x00;
        bytes[15] = 0x00;

        // Bytes 16-31: Material number (UUID)
        let uuid = Uuid::new_v4();
        bytes[16..32].copy_from_slice(uuid.as_bytes());

        Umid(bytes)
    }

    /// Check if this is a zero UMID
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Get the UUID portion (bytes 16-31)
    pub fn uuid_portion(&self) -> &[u8] {
        &self.0[16..32]
    }
}

impl fmt::Debug for Umid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UMID(")?;
        for (i, byte) in self.0.iter().enumerate() {
            if i > 0 && i % 4 == 0 {
                write!(f, "-")?;
            }
            write!(f, "{:02x}", byte)?;
        }
        write!(f, ")")
    }
}

impl Default for Umid {
    fn default() -> Self {
        Umid::zero()
    }
}

/// MXF timestamp
#[derive(Debug, Clone, Copy, Default)]
pub struct MxfTimestamp {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub millisecond: u8,
}

impl MxfTimestamp {
    /// Create new timestamp
    pub fn new(
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u8,
    ) -> Self {
        MxfTimestamp {
            year,
            month,
            day,
            hour,
            minute,
            second,
            millisecond,
        }
    }

    /// Create timestamp for current time (approximation without chrono)
    pub fn now() -> Self {
        // Without chrono, return a default timestamp
        MxfTimestamp::new(2024, 1, 1, 0, 0, 0, 0)
    }

    /// Parse from 8-byte MXF format
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }

        Some(MxfTimestamp {
            year: u16::from_be_bytes([bytes[0], bytes[1]]),
            month: bytes[2],
            day: bytes[3],
            hour: bytes[4],
            minute: bytes[5],
            second: bytes[6],
            millisecond: bytes[7],
        })
    }

    /// Convert to 8-byte MXF format
    pub fn to_bytes(&self) -> [u8; 8] {
        let year_bytes = self.year.to_be_bytes();
        [
            year_bytes[0],
            year_bytes[1],
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond,
        ]
    }
}

impl fmt::Display for MxfTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond
        )
    }
}

/// Track kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Video track
    Video,
    /// Audio track
    Audio,
    /// Data track
    Data,
    /// Timecode track
    Timecode,
    /// Unknown track type
    Unknown,
}

impl Default for TrackKind {
    fn default() -> Self {
        TrackKind::Unknown
    }
}

/// Video frame size
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

impl FrameSize {
    pub fn new(width: u32, height: u32) -> Self {
        FrameSize { width, height }
    }

    /// Common video sizes
    pub fn hd() -> Self {
        FrameSize::new(1920, 1080)
    }

    pub fn sd_ntsc() -> Self {
        FrameSize::new(720, 486)
    }

    pub fn sd_pal() -> Self {
        FrameSize::new(720, 576)
    }

    pub fn uhd() -> Self {
        FrameSize::new(3840, 2160)
    }
}

/// Aspect ratio
#[derive(Debug, Clone, Copy)]
pub struct AspectRatio {
    pub numerator: u32,
    pub denominator: u32,
}

impl AspectRatio {
    pub fn new(numerator: u32, denominator: u32) -> Self {
        AspectRatio { numerator, denominator }
    }

    /// 16:9 aspect ratio
    pub fn widescreen() -> Self {
        AspectRatio::new(16, 9)
    }

    /// 4:3 aspect ratio
    pub fn standard() -> Self {
        AspectRatio::new(4, 3)
    }

    pub fn to_f64(&self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            self.numerator as f64 / self.denominator as f64
        }
    }
}

impl Default for AspectRatio {
    fn default() -> Self {
        AspectRatio::widescreen()
    }
}

/// Color primaries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    BT601,
    BT709,
    BT2020,
    Unknown,
}

impl Default for ColorPrimaries {
    fn default() -> Self {
        ColorPrimaries::BT709
    }
}

/// Transfer characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferCharacteristic {
    BT709,
    BT2020,
    PQ,   // Perceptual Quantizer (HDR)
    HLG,  // Hybrid Log-Gamma (HDR)
    Unknown,
}

impl Default for TransferCharacteristic {
    fn default() -> Self {
        TransferCharacteristic::BT709
    }
}

/// Essence coding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssenceCoding {
    /// Uncompressed video
    Uncompressed,
    /// MPEG-2 video
    Mpeg2,
    /// AVC/H.264
    Avc,
    /// HEVC/H.265
    Hevc,
    /// JPEG 2000
    Jpeg2000,
    /// ProRes
    ProRes,
    /// DNxHD/DNxHR
    DnxHd,
    /// PCM audio
    Pcm,
    /// Unknown
    Unknown,
}

impl Default for EssenceCoding {
    fn default() -> Self {
        EssenceCoding::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational() {
        let r = Rational::fps_29_97();
        let f = r.to_f64();
        assert!((f - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_umid() {
        let umid = Umid::generate();
        assert!(!umid.is_zero());

        let zero = Umid::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_timestamp() {
        let ts = MxfTimestamp::new(2024, 6, 15, 14, 30, 45, 50);
        let bytes = ts.to_bytes();
        let parsed = MxfTimestamp::from_bytes(&bytes).unwrap();

        assert_eq!(ts.year, parsed.year);
        assert_eq!(ts.month, parsed.month);
        assert_eq!(ts.day, parsed.day);
    }

    #[test]
    fn test_frame_size() {
        let hd = FrameSize::hd();
        assert_eq!(hd.width, 1920);
        assert_eq!(hd.height, 1080);
    }

    #[test]
    fn test_aspect_ratio() {
        let ar = AspectRatio::widescreen();
        let ratio = ar.to_f64();
        assert!((ratio - 1.778).abs() < 0.001);
    }
}
