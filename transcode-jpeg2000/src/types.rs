//! JPEG2000 type definitions.
//!
//! This module contains all data types for JPEG2000 codestream parsing
//! and image representation.

/// JPEG2000 profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Jpeg2000Profile {
    /// Cinema 2K profile (DCI).
    Cinema2k,
    /// Cinema 4K profile (DCI).
    Cinema4k,
    /// Broadcast profile.
    Broadcast,
    /// IMF (Interoperable Master Format) profile.
    Imf,
    /// Part-1 compliant.
    #[default]
    Part1,
    /// Part-2 extensions.
    Part2,
    /// Lossless profile.
    Lossless,
    /// Custom/unknown profile.
    Custom,
}

/// Wavelet transform type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveletTransform {
    /// 9/7 irreversible (CDF 9/7) - lossy compression.
    #[default]
    Irreversible9x7,
    /// 5/3 reversible (Le Gall 5/3) - lossless compression.
    Reversible5x3,
}

/// Progression order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressionOrder {
    /// Layer-Resolution-Component-Position.
    #[default]
    Lrcp,
    /// Resolution-Layer-Component-Position.
    Rlcp,
    /// Resolution-Position-Component-Layer.
    Rpcl,
    /// Position-Component-Resolution-Layer.
    Pcrl,
    /// Component-Position-Resolution-Layer.
    Cprl,
}

impl ProgressionOrder {
    /// Get the code for this progression order.
    pub fn code(&self) -> u8 {
        match self {
            ProgressionOrder::Lrcp => 0,
            ProgressionOrder::Rlcp => 1,
            ProgressionOrder::Rpcl => 2,
            ProgressionOrder::Pcrl => 3,
            ProgressionOrder::Cprl => 4,
        }
    }

    /// Create from code.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(ProgressionOrder::Lrcp),
            1 => Some(ProgressionOrder::Rlcp),
            2 => Some(ProgressionOrder::Rpcl),
            3 => Some(ProgressionOrder::Pcrl),
            4 => Some(ProgressionOrder::Cprl),
            _ => None,
        }
    }
}

/// Color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Unknown/unspecified.
    #[default]
    Unknown,
    /// Grayscale.
    Grayscale,
    /// sRGB.
    Srgb,
    /// YCbCr (ITU-R BT.601).
    YCbCr,
    /// YCbCr (ITU-R BT.709).
    YCbCr709,
    /// XYZ (CIE 1931).
    Xyz,
    /// CMYK.
    Cmyk,
    /// Lab.
    Lab,
}

/// Component information.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Horizontal sub-sampling factor.
    pub dx: u8,
    /// Vertical sub-sampling factor.
    pub dy: u8,
    /// Bit depth (1-38).
    pub bit_depth: u8,
    /// Signed samples.
    pub is_signed: bool,
    /// Component width (computed).
    pub width: u32,
    /// Component height (computed).
    pub height: u32,
}

impl Default for ComponentInfo {
    fn default() -> Self {
        Self {
            dx: 1,
            dy: 1,
            bit_depth: 8,
            is_signed: false,
            width: 0,
            height: 0,
        }
    }
}

/// Image size parameters (SIZ marker).
#[derive(Debug, Clone, Default)]
pub struct SizMarker {
    /// Reference grid width.
    pub ref_grid_width: u32,
    /// Reference grid height.
    pub ref_grid_height: u32,
    /// Image area horizontal offset.
    pub x_offset: u32,
    /// Image area vertical offset.
    pub y_offset: u32,
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Tile horizontal offset.
    pub tile_x_offset: u32,
    /// Tile vertical offset.
    pub tile_y_offset: u32,
    /// Number of components.
    pub num_components: u16,
    /// Component information.
    pub components: Vec<ComponentInfo>,
    /// Profile (Rsiz field).
    pub profile: u16,
}

impl SizMarker {
    /// Get the image width.
    pub fn image_width(&self) -> u32 {
        self.ref_grid_width - self.x_offset
    }

    /// Get the image height.
    pub fn image_height(&self) -> u32 {
        self.ref_grid_height - self.y_offset
    }

    /// Get the number of tiles horizontally.
    pub fn num_tiles_x(&self) -> u32 {
        let width = self.ref_grid_width - self.tile_x_offset;
        width.div_ceil(self.tile_width)
    }

    /// Get the number of tiles vertically.
    pub fn num_tiles_y(&self) -> u32 {
        let height = self.ref_grid_height - self.tile_y_offset;
        height.div_ceil(self.tile_height)
    }

    /// Get total number of tiles.
    pub fn num_tiles(&self) -> u32 {
        self.num_tiles_x() * self.num_tiles_y()
    }
}

/// Coding style default (COD marker).
#[derive(Debug, Clone)]
pub struct CodMarker {
    /// Coding style flags.
    pub coding_style: u8,
    /// Progression order.
    pub progression_order: ProgressionOrder,
    /// Number of quality layers.
    pub num_layers: u16,
    /// Multiple component transform (0=none, 1=ICT/RCT).
    pub mct: u8,
    /// Number of decomposition levels.
    pub num_decomposition_levels: u8,
    /// Code-block width exponent (2^(value+2)).
    pub code_block_width_exp: u8,
    /// Code-block height exponent (2^(value+2)).
    pub code_block_height_exp: u8,
    /// Code-block style flags.
    pub code_block_style: u8,
    /// Wavelet transform type.
    pub wavelet_transform: WaveletTransform,
    /// Precinct sizes (if defined).
    pub precinct_sizes: Vec<(u8, u8)>,
}

impl Default for CodMarker {
    fn default() -> Self {
        Self {
            coding_style: 0,
            progression_order: ProgressionOrder::Lrcp,
            num_layers: 1,
            mct: 1,
            num_decomposition_levels: 5,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            code_block_style: 0,
            wavelet_transform: WaveletTransform::Irreversible9x7,
            precinct_sizes: Vec::new(),
        }
    }
}

impl CodMarker {
    /// Get code-block width.
    pub fn code_block_width(&self) -> u32 {
        1 << (self.code_block_width_exp + 2)
    }

    /// Get code-block height.
    pub fn code_block_height(&self) -> u32 {
        1 << (self.code_block_height_exp + 2)
    }

    /// Check if precincts are used.
    pub fn uses_precincts(&self) -> bool {
        (self.coding_style & 0x01) != 0
    }

    /// Check if SOP markers are used.
    pub fn uses_sop(&self) -> bool {
        (self.coding_style & 0x02) != 0
    }

    /// Check if EPH markers are used.
    pub fn uses_eph(&self) -> bool {
        (self.coding_style & 0x04) != 0
    }
}

/// Quantization default (QCD marker).
#[derive(Debug, Clone)]
pub struct QcdMarker {
    /// Quantization style.
    pub quantization_style: u8,
    /// Number of guard bits.
    pub guard_bits: u8,
    /// Step sizes (for irreversible transform).
    pub step_sizes: Vec<QuantizationStepSize>,
}

impl Default for QcdMarker {
    fn default() -> Self {
        Self {
            quantization_style: 0,
            guard_bits: 2,
            step_sizes: Vec::new(),
        }
    }
}

impl QcdMarker {
    /// Check if no quantization is used (reversible).
    pub fn is_no_quantization(&self) -> bool {
        (self.quantization_style & 0x1F) == 0
    }

    /// Check if scalar derived quantization is used.
    pub fn is_scalar_derived(&self) -> bool {
        (self.quantization_style & 0x1F) == 1
    }

    /// Check if scalar expounded quantization is used.
    pub fn is_scalar_expounded(&self) -> bool {
        (self.quantization_style & 0x1F) == 2
    }
}

/// Quantization step size.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationStepSize {
    /// Exponent.
    pub exponent: u8,
    /// Mantissa.
    pub mantissa: u16,
}

impl QuantizationStepSize {
    /// Get the step size as a float.
    pub fn as_float(&self) -> f64 {
        let mantissa = self.mantissa as f64 / 2048.0;
        (1.0 + mantissa) * 2.0_f64.powi(-(self.exponent as i32))
    }
}

/// Tile-part header (SOT marker).
#[derive(Debug, Clone, Default)]
pub struct SotMarker {
    /// Tile index.
    pub tile_index: u16,
    /// Length of tile-part.
    pub tile_part_length: u32,
    /// Tile-part index.
    pub tile_part_index: u8,
    /// Number of tile-parts (0 = unknown).
    pub num_tile_parts: u8,
}

/// Comment marker (COM).
#[derive(Debug, Clone)]
pub struct ComMarker {
    /// Registration value (0 = binary, 1 = ISO-8859-15).
    pub registration: u16,
    /// Comment data.
    pub data: Vec<u8>,
}

impl ComMarker {
    /// Check if comment is text.
    pub fn is_text(&self) -> bool {
        self.registration == 1
    }

    /// Get comment as text (if applicable).
    pub fn as_text(&self) -> Option<String> {
        if self.is_text() {
            String::from_utf8(self.data.clone()).ok()
        } else {
            None
        }
    }
}

/// Parsed tile.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile index.
    pub index: u16,
    /// Tile-part data (may be split across multiple tile-parts).
    pub data: Vec<u8>,
    /// Number of tile-parts.
    pub num_parts: u8,
}

/// Codestream markers and data.
#[derive(Debug, Default)]
pub struct Codestream {
    /// SIZ marker (image and tile size).
    pub siz: Option<SizMarker>,
    /// COD marker (coding style default).
    pub cod: Option<CodMarker>,
    /// QCD marker (quantization default).
    pub qcd: Option<QcdMarker>,
    /// Comments.
    pub comments: Vec<ComMarker>,
    /// Tiles.
    pub tiles: Vec<Tile>,
}

impl Codestream {
    /// Check if codestream has all required markers.
    pub fn is_valid(&self) -> bool {
        self.siz.is_some() && self.cod.is_some() && self.qcd.is_some()
    }

    /// Get image width.
    pub fn image_width(&self) -> Option<u32> {
        self.siz.as_ref().map(|s| s.image_width())
    }

    /// Get image height.
    pub fn image_height(&self) -> Option<u32> {
        self.siz.as_ref().map(|s| s.image_height())
    }

    /// Get number of components.
    pub fn num_components(&self) -> Option<u16> {
        self.siz.as_ref().map(|s| s.num_components)
    }
}

/// JP2 file format box types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxType {
    /// JPEG2000 Signature box.
    Signature,
    /// File Type box.
    FileType,
    /// JP2 Header box.
    Jp2Header,
    /// Image Header box.
    ImageHeader,
    /// Colour Specification box.
    ColourSpec,
    /// Contiguous Codestream box.
    CodestreamBox,
    /// Intellectual Property box.
    Ip,
    /// XML box.
    Xml,
    /// UUID box.
    Uuid,
    /// UUID Info box.
    UuidInfo,
    /// Unknown box.
    Unknown(u32),
}

impl BoxType {
    /// Get the box type code.
    pub fn code(&self) -> u32 {
        match self {
            BoxType::Signature => 0x6A502020,    // 'jP  '
            BoxType::FileType => 0x66747970,     // 'ftyp'
            BoxType::Jp2Header => 0x6A703268,    // 'jp2h'
            BoxType::ImageHeader => 0x69686472,  // 'ihdr'
            BoxType::ColourSpec => 0x636F6C72,   // 'colr'
            BoxType::CodestreamBox => 0x6A703263, // 'jp2c'
            BoxType::Ip => 0x6A703269,           // 'jp2i'
            BoxType::Xml => 0x786D6C20,          // 'xml '
            BoxType::Uuid => 0x75756964,         // 'uuid'
            BoxType::UuidInfo => 0x75696E66,     // 'uinf'
            BoxType::Unknown(code) => *code,
        }
    }

    /// Create from box type code.
    pub fn from_code(code: u32) -> Self {
        match code {
            0x6A502020 => BoxType::Signature,
            0x66747970 => BoxType::FileType,
            0x6A703268 => BoxType::Jp2Header,
            0x69686472 => BoxType::ImageHeader,
            0x636F6C72 => BoxType::ColourSpec,
            0x6A703263 => BoxType::CodestreamBox,
            0x6A703269 => BoxType::Ip,
            0x786D6C20 => BoxType::Xml,
            0x75756964 => BoxType::Uuid,
            0x75696E66 => BoxType::UuidInfo,
            _ => BoxType::Unknown(code),
        }
    }
}

/// JP2 file header.
#[derive(Debug, Clone)]
pub struct Jp2Box {
    /// Box type.
    pub box_type: BoxType,
    /// Box data offset in file.
    pub offset: u64,
    /// Box data length.
    pub length: u64,
}

/// JP2 file structure.
#[derive(Debug, Default)]
pub struct Jp2File {
    /// File type box brand.
    pub brand: Option<u32>,
    /// Minor version.
    pub minor_version: u32,
    /// Compatibility list.
    pub compatibility: Vec<u32>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components.
    pub num_components: u16,
    /// Bits per component.
    pub bits_per_component: u8,
    /// Color space.
    pub color_space: ColorSpace,
    /// Codestream offset in file.
    pub codestream_offset: u64,
    /// Codestream length.
    pub codestream_length: u64,
}

impl Jp2File {
    /// Check if file is valid JP2.
    pub fn is_valid(&self) -> bool {
        self.brand.is_some() && self.codestream_length > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_transform() {
        assert_eq!(WaveletTransform::default(), WaveletTransform::Irreversible9x7);
    }

    #[test]
    fn test_progression_order() {
        assert_eq!(ProgressionOrder::Lrcp.code(), 0);
        assert_eq!(ProgressionOrder::Rlcp.code(), 1);
        assert_eq!(ProgressionOrder::from_code(0), Some(ProgressionOrder::Lrcp));
        assert_eq!(ProgressionOrder::from_code(5), None);
    }

    #[test]
    fn test_color_space() {
        assert_eq!(ColorSpace::default(), ColorSpace::Unknown);
    }

    #[test]
    fn test_component_info() {
        let comp = ComponentInfo::default();
        assert_eq!(comp.bit_depth, 8);
        assert!(!comp.is_signed);
    }

    #[test]
    fn test_siz_marker() {
        let siz = SizMarker {
            ref_grid_width: 1920,
            ref_grid_height: 1080,
            x_offset: 0,
            y_offset: 0,
            tile_width: 256,
            tile_height: 256,
            tile_x_offset: 0,
            tile_y_offset: 0,
            num_components: 3,
            components: vec![],
            profile: 0,
        };
        assert_eq!(siz.image_width(), 1920);
        assert_eq!(siz.image_height(), 1080);
        assert_eq!(siz.num_tiles_x(), 8);
        assert_eq!(siz.num_tiles_y(), 5);
        assert_eq!(siz.num_tiles(), 40);
    }

    #[test]
    fn test_cod_marker() {
        let cod = CodMarker::default();
        assert_eq!(cod.code_block_width(), 64);
        assert_eq!(cod.code_block_height(), 64);
        assert!(!cod.uses_precincts());
    }

    #[test]
    fn test_qcd_marker() {
        let qcd = QcdMarker::default();
        assert!(qcd.is_no_quantization());
    }

    #[test]
    fn test_quantization_step_size() {
        let step = QuantizationStepSize {
            exponent: 10,
            mantissa: 0,
        };
        assert!((step.as_float() - 0.0009765625).abs() < 1e-10);
    }

    #[test]
    fn test_com_marker() {
        let com = ComMarker {
            registration: 1,
            data: b"Test comment".to_vec(),
        };
        assert!(com.is_text());
        assert_eq!(com.as_text(), Some("Test comment".to_string()));
    }

    #[test]
    fn test_box_type() {
        assert_eq!(BoxType::Signature.code(), 0x6A502020);
        assert_eq!(BoxType::from_code(0x6A502020), BoxType::Signature);
        assert_eq!(BoxType::from_code(0x6A703263), BoxType::CodestreamBox);
    }

    #[test]
    fn test_jpeg2000_profile() {
        assert_eq!(Jpeg2000Profile::default(), Jpeg2000Profile::Part1);
    }

    #[test]
    fn test_codestream() {
        let cs = Codestream::default();
        assert!(!cs.is_valid());
        assert!(cs.image_width().is_none());
    }
}
