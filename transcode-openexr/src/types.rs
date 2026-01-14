//! OpenEXR type definitions

use std::fmt;

/// 2D integer vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2i {
    pub x: i32,
    pub y: i32,
}

impl V2i {
    pub fn new(x: i32, y: i32) -> Self {
        V2i { x, y }
    }
}

/// 2D float vector
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct V2f {
    pub x: f32,
    pub y: f32,
}

impl V2f {
    pub fn new(x: f32, y: f32) -> Self {
        V2f { x, y }
    }
}

/// 3D integer vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl V3i {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        V3i { x, y, z }
    }
}

/// 3D float vector
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct V3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl V3f {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        V3f { x, y, z }
    }
}

/// 2D integer bounding box
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Box2i {
    pub min: V2i,
    pub max: V2i,
}

impl Box2i {
    pub fn new(min: V2i, max: V2i) -> Self {
        Box2i { min, max }
    }

    /// Create from dimensions
    pub fn from_dimensions(width: i32, height: i32) -> Self {
        Box2i {
            min: V2i::new(0, 0),
            max: V2i::new(width - 1, height - 1),
        }
    }

    /// Get width
    pub fn width(&self) -> i32 {
        self.max.x - self.min.x + 1
    }

    /// Get height
    pub fn height(&self) -> i32 {
        self.max.y - self.min.y + 1
    }

    /// Check if box is valid
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y
    }
}

impl fmt::Display for Box2i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[({}, {}) - ({}, {})]",
            self.min.x, self.min.y, self.max.x, self.max.y
        )
    }
}

/// 2D float bounding box
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Box2f {
    pub min: V2f,
    pub max: V2f,
}

impl Box2f {
    pub fn new(min: V2f, max: V2f) -> Self {
        Box2f { min, max }
    }
}

/// Display window (visible area)
pub type DisplayWindow = Box2i;

/// Data window (actual pixel data bounds)
pub type DataWindow = Box2i;

/// Line order for scanline images
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineOrder {
    /// Scanlines stored top to bottom
    #[default]
    IncreasingY,
    /// Scanlines stored bottom to top
    DecreasingY,
    /// Random order (typically for tiled images)
    RandomY,
}

impl LineOrder {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(LineOrder::IncreasingY),
            1 => Some(LineOrder::DecreasingY),
            2 => Some(LineOrder::RandomY),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            LineOrder::IncreasingY => 0,
            LineOrder::DecreasingY => 1,
            LineOrder::RandomY => 2,
        }
    }
}

/// Tile level mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LevelMode {
    /// Single resolution level
    #[default]
    OneLevel,
    /// Mipmap levels
    MipmapLevels,
    /// Ripmap levels
    RipmapLevels,
}

/// Tile rounding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    #[default]
    RoundDown,
    RoundUp,
}

/// Tile description
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDescription {
    pub x_size: u32,
    pub y_size: u32,
    pub level_mode: LevelMode,
    pub rounding_mode: RoundingMode,
}

impl Default for TileDescription {
    fn default() -> Self {
        TileDescription {
            x_size: 64,
            y_size: 64,
            level_mode: LevelMode::OneLevel,
            rounding_mode: RoundingMode::RoundDown,
        }
    }
}

/// Rational number
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub numerator: i32,
    pub denominator: u32,
}

impl Rational {
    pub fn new(numerator: i32, denominator: u32) -> Self {
        Rational {
            numerator,
            denominator,
        }
    }

    pub fn to_f64(&self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            self.numerator as f64 / self.denominator as f64
        }
    }
}

impl Default for Rational {
    fn default() -> Self {
        Rational::new(1, 1)
    }
}

/// 3x3 matrix (row-major)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M33f {
    pub data: [[f32; 3]; 3],
}

impl Default for M33f {
    fn default() -> Self {
        M33f {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

/// 4x4 matrix (row-major)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M44f {
    pub data: [[f32; 4]; 4],
}

impl Default for M44f {
    fn default() -> Self {
        M44f {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

/// Chromaticities for color space definition
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticities {
    pub red: V2f,
    pub green: V2f,
    pub blue: V2f,
    pub white: V2f,
}

impl Default for Chromaticities {
    /// Default to Rec. 709 (sRGB) primaries
    fn default() -> Self {
        Chromaticities {
            red: V2f::new(0.64, 0.33),
            green: V2f::new(0.30, 0.60),
            blue: V2f::new(0.15, 0.06),
            white: V2f::new(0.3127, 0.3290), // D65
        }
    }
}

/// Half-float (16-bit floating point)
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Half(pub u16);

impl Half {
    /// Create from u16 bits
    pub fn from_bits(bits: u16) -> Self {
        Half(bits)
    }

    /// Get raw bits
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32
    pub fn from_f32(value: f32) -> Self {
        Half(f32_to_half(value))
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        half_to_f32(self.0)
    }

    /// Zero
    pub fn zero() -> Self {
        Half(0)
    }

    /// One
    pub fn one() -> Self {
        Half(0x3C00)
    }

    /// Positive infinity
    pub fn infinity() -> Self {
        Half(0x7C00)
    }

    /// Negative infinity
    pub fn neg_infinity() -> Self {
        Half(0xFC00)
    }

    /// NaN
    pub fn nan() -> Self {
        Half(0x7E00)
    }

    /// Check if NaN
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0
    }

    /// Check if infinite
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Check if zero
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }
}

impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({})", self.to_f32())
    }
}

impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for Half {
    fn from(value: f32) -> Self {
        Half::from_f32(value)
    }
}

impl From<Half> for f32 {
    fn from(value: Half) -> Self {
        value.to_f32()
    }
}

/// Convert f32 to half-float bits
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x7FFFFF;

    if exp <= 0 {
        if exp < -10 {
            // Too small, round to zero
            sign as u16
        } else {
            // Denormalized number
            let mant = (mant | 0x800000) >> (1 - exp);
            (sign | (mant >> 13)) as u16
        }
    } else if exp >= 31 {
        if exp == 128 && mant != 0 {
            // NaN
            (sign | 0x7E00) as u16
        } else {
            // Overflow to infinity
            (sign | 0x7C00) as u16
        }
    } else {
        (sign | ((exp as u32) << 10) | (mant >> 13)) as u16
    }
}

/// Convert half-float bits to f32
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    let result = if exp == 0 {
        if mant == 0 {
            // Zero
            sign << 31
        } else {
            // Denormalized
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let mant = (m & 0x3FF) << 13;
            let exp = ((127 - 15 + e + 1) as u32) << 23;
            (sign << 31) | exp | mant
        }
    } else if exp == 31 {
        if mant == 0 {
            // Infinity
            (sign << 31) | 0x7F800000
        } else {
            // NaN
            (sign << 31) | 0x7FC00000
        }
    } else {
        // Normalized
        let exp = ((exp + 127 - 15) as u32) << 23;
        let mant = mant << 13;
        (sign << 31) | exp | mant
    };

    f32::from_bits(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2i() {
        let v = V2i::new(10, 20);
        assert_eq!(v.x, 10);
        assert_eq!(v.y, 20);
    }

    #[test]
    fn test_box2i() {
        let b = Box2i::from_dimensions(1920, 1080);
        assert_eq!(b.width(), 1920);
        assert_eq!(b.height(), 1080);
        assert!(b.is_valid());
    }

    #[test]
    fn test_half() {
        let h = Half::from_f32(1.0);
        assert!((h.to_f32() - 1.0).abs() < 0.001);

        let h = Half::from_f32(0.0);
        assert!(h.is_zero());

        let h = Half::infinity();
        assert!(h.is_infinite());

        let h = Half::nan();
        assert!(h.is_nan());
    }

    #[test]
    fn test_line_order() {
        assert_eq!(LineOrder::from_u8(0), Some(LineOrder::IncreasingY));
        assert_eq!(LineOrder::from_u8(1), Some(LineOrder::DecreasingY));
        assert_eq!(LineOrder::from_u8(2), Some(LineOrder::RandomY));
        assert_eq!(LineOrder::from_u8(3), None);
    }

    #[test]
    fn test_rational() {
        let r = Rational::new(30000, 1001);
        assert!((r.to_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_chromaticities() {
        let c = Chromaticities::default();
        assert!((c.red.x - 0.64).abs() < 0.01);
        assert!((c.white.x - 0.3127).abs() < 0.01);
    }
}
