//! AV1 encoder configuration.

use crate::error::Av1Error;
use crate::Result;

/// Encoding speed preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Av1Preset {
    /// Slowest encoding, best quality (speed 0-1).
    Placebo,
    /// Very slow encoding, excellent quality (speed 2).
    VerySlow,
    /// Slow encoding, great quality (speed 3).
    Slower,
    /// Below average speed, above average quality (speed 4).
    Slow,
    /// Balanced speed and quality (speed 5-6).
    #[default]
    Medium,
    /// Above average speed, below average quality (speed 7).
    Fast,
    /// Fast encoding, lower quality (speed 8).
    Faster,
    /// Very fast encoding, low quality (speed 9).
    VeryFast,
    /// Fastest encoding, lowest quality (speed 10).
    UltraFast,
}

impl Av1Preset {
    /// Convert to rav1e speed value (0-10).
    pub fn to_speed(self) -> u8 {
        match self {
            Self::Placebo => 0,
            Self::VerySlow => 2,
            Self::Slower => 3,
            Self::Slow => 4,
            Self::Medium => 6,
            Self::Fast => 7,
            Self::Faster => 8,
            Self::VeryFast => 9,
            Self::UltraFast => 10,
        }
    }

    /// Create from speed value.
    pub fn from_speed(speed: u8) -> Self {
        match speed {
            0..=1 => Self::Placebo,
            2 => Self::VerySlow,
            3 => Self::Slower,
            4 => Self::Slow,
            5..=6 => Self::Medium,
            7 => Self::Fast,
            8 => Self::Faster,
            9 => Self::VeryFast,
            _ => Self::UltraFast,
        }
    }
}

/// Rate control mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant quality (CRF-like).
    ConstantQuality {
        /// Quality value (0-63, lower is better).
        quantizer: u8,
    },
    /// Variable bitrate.
    Vbr {
        /// Target bitrate in bits per second.
        bitrate: u64,
    },
    /// Constant bitrate.
    Cbr {
        /// Target bitrate in bits per second.
        bitrate: u64,
    },
    /// Two-pass VBR (first pass).
    TwoPassFirst,
    /// Two-pass VBR (second pass).
    TwoPassSecond {
        /// First pass statistics.
        stats: Vec<u8>,
    },
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::ConstantQuality { quantizer: 28 }
    }
}

/// Color primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPrimaries {
    #[default]
    Bt709,
    Bt470m,
    Bt470bg,
    Bt601,
    Smpte240,
    GenericFilm,
    Bt2020,
    Xyz,
    Smpte431,
    Smpte432,
}

/// Transfer characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferCharacteristics {
    #[default]
    Bt709,
    Bt470m,
    Bt470bg,
    Bt601,
    Smpte240,
    Linear,
    Log100,
    Log100Sqrt10,
    Iec61966,
    Bt1361,
    Srgb,
    Bt2020_10bit,
    Bt2020_12bit,
    Smpte2084,
    Smpte428,
    Hlg,
}

/// Matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixCoefficients {
    Identity,
    #[default]
    Bt709,
    Bt470m,
    Bt470bg,
    Bt601,
    Smpte240,
    Ycgco,
    Bt2020Ncl,
    Bt2020Cl,
    Smpte2085,
    ChromaDerivedNcl,
    ChromaDerivedCl,
    Ictcp,
}

/// AV1 encoder configuration.
#[derive(Debug, Clone)]
pub struct Av1Config {
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Framerate numerator.
    pub framerate_num: u32,
    /// Framerate denominator.
    pub framerate_den: u32,
    /// Encoding preset.
    pub preset: Av1Preset,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Bit depth (8 or 10).
    pub bit_depth: u8,
    /// Number of tile columns (log2).
    pub tile_cols_log2: u8,
    /// Number of tile rows (log2).
    pub tile_rows_log2: u8,
    /// Keyframe interval (0 for auto).
    pub keyframe_interval: u32,
    /// Minimum keyframe interval.
    pub min_keyframe_interval: u32,
    /// Enable low latency mode.
    pub low_latency: bool,
    /// Number of threads (0 for auto).
    pub threads: usize,
    /// Color primaries.
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristics.
    pub transfer: TransferCharacteristics,
    /// Matrix coefficients.
    pub matrix: MatrixCoefficients,
    /// Full range color.
    pub full_range: bool,
    /// Chroma sample position.
    pub chroma_sample_position: u8,
    /// Content type hint.
    pub content_type: ContentType,
}

/// Content type hint for encoder optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentType {
    /// Unknown or mixed content.
    #[default]
    Unknown,
    /// Film/movie content.
    Film,
    /// Animation/cartoon content.
    Animation,
    /// Screen capture/presentation.
    Screen,
    /// Gaming content.
    Gaming,
}

impl Av1Config {
    /// Create a new configuration with required parameters.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            framerate_num: 30,
            framerate_den: 1,
            preset: Av1Preset::default(),
            rate_control: RateControlMode::default(),
            bit_depth: 8,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            keyframe_interval: 240,
            min_keyframe_interval: 12,
            low_latency: false,
            threads: 0,
            color_primaries: ColorPrimaries::default(),
            transfer: TransferCharacteristics::default(),
            matrix: MatrixCoefficients::default(),
            full_range: false,
            chroma_sample_position: 0,
            content_type: ContentType::default(),
        }
    }

    /// Set the encoding preset.
    pub fn with_preset(mut self, preset: Av1Preset) -> Self {
        self.preset = preset;
        self
    }

    /// Set the framerate.
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        self
    }

    /// Set the target bitrate (VBR mode).
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.rate_control = RateControlMode::Vbr { bitrate };
        self
    }

    /// Set constant quality mode.
    pub fn with_quality(mut self, quantizer: u8) -> Self {
        self.rate_control = RateControlMode::ConstantQuality {
            quantizer: quantizer.min(63),
        };
        self
    }

    /// Set the bit depth.
    pub fn with_bit_depth(mut self, depth: u8) -> Self {
        self.bit_depth = if depth >= 10 { 10 } else { 8 };
        self
    }

    /// Set tile configuration for parallelism.
    pub fn with_tiles(mut self, cols_log2: u8, rows_log2: u8) -> Self {
        self.tile_cols_log2 = cols_log2.min(6);
        self.tile_rows_log2 = rows_log2.min(6);
        self
    }

    /// Set keyframe interval.
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Enable low latency mode.
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Set number of threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set color properties for HDR content.
    pub fn with_hdr(mut self) -> Self {
        self.bit_depth = 10;
        self.color_primaries = ColorPrimaries::Bt2020;
        self.transfer = TransferCharacteristics::Smpte2084;
        self.matrix = MatrixCoefficients::Bt2020Ncl;
        self
    }

    /// Set content type hint.
    pub fn with_content_type(mut self, content_type: ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Av1Error::InvalidConfig(
                "Width and height must be non-zero".into(),
            ));
        }

        if self.width > 8192 || self.height > 4320 {
            return Err(Av1Error::InvalidConfig(
                "Resolution exceeds AV1 maximum (8192x4320)".into(),
            ));
        }

        if self.framerate_num == 0 || self.framerate_den == 0 {
            return Err(Av1Error::InvalidConfig("Invalid framerate".into()));
        }

        if self.bit_depth != 8 && self.bit_depth != 10 {
            return Err(Av1Error::InvalidConfig(
                "Bit depth must be 8 or 10".into(),
            ));
        }

        Ok(())
    }

    /// Calculate optimal tile configuration for given resolution and threads.
    pub fn auto_tiles(&mut self) {
        let threads = if self.threads == 0 {
            num_cpus()
        } else {
            self.threads
        };

        // Simple heuristic: more tiles for higher resolutions and more threads
        let pixels = (self.width * self.height) as usize;

        if pixels >= 3840 * 2160 && threads >= 8 {
            self.tile_cols_log2 = 2;
            self.tile_rows_log2 = 2;
        } else if pixels >= 1920 * 1080 && threads >= 4 {
            self.tile_cols_log2 = 1;
            self.tile_rows_log2 = 1;
        }
    }
}

/// Get number of CPUs (placeholder).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

impl Default for Av1Config {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_speed() {
        assert_eq!(Av1Preset::Placebo.to_speed(), 0);
        assert_eq!(Av1Preset::Medium.to_speed(), 6);
        assert_eq!(Av1Preset::UltraFast.to_speed(), 10);
    }

    #[test]
    fn test_config_validation() {
        let config = Av1Config::new(1920, 1080);
        assert!(config.validate().is_ok());

        let bad_config = Av1Config::new(0, 1080);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = Av1Config::new(3840, 2160)
            .with_preset(Av1Preset::Fast)
            .with_bitrate(10_000_000)
            .with_bit_depth(10)
            .with_tiles(2, 2);

        assert_eq!(config.preset, Av1Preset::Fast);
        assert_eq!(config.bit_depth, 10);
        assert_eq!(config.tile_cols_log2, 2);
    }
}
