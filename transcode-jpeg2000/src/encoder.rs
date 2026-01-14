//! JPEG2000 encoder.
//!
//! Provides JPEG2000 image encoding with optional OpenJPEG FFI support.

use crate::types::*;
use crate::{Jpeg2000Error, MarkerType, Result};
use byteorder::{BigEndian, ByteOrder};

#[cfg(feature = "ffi-openjpeg")]
use crate::ffi::{Jpeg2000EncoderConfigFfi, Jpeg2000FfiEncoder};

use std::fmt;

/// JPEG2000 encoder configuration.
#[derive(Debug, Clone)]
pub struct Jpeg2000EncoderConfig {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components.
    pub num_components: u16,
    /// Bit depth per component.
    pub bit_depth: u8,
    /// Is signed data.
    pub is_signed: bool,
    /// Tile width (0 = single tile).
    pub tile_width: u32,
    /// Tile height (0 = single tile).
    pub tile_height: u32,
    /// Number of decomposition levels.
    pub num_decomposition_levels: u8,
    /// Number of quality layers.
    pub num_layers: u16,
    /// Progression order.
    pub progression_order: ProgressionOrder,
    /// Wavelet transform type.
    pub wavelet_transform: WaveletTransform,
    /// Enable multiple component transform.
    pub mct: bool,
    /// Target compression ratio (0 = lossless).
    pub compression_ratio: f32,
    /// Target quality (PSNR, 0 = lossless).
    pub target_psnr: f32,
    /// Profile.
    pub profile: Jpeg2000Profile,
    /// Code-block width exponent (4-6).
    pub code_block_width_exp: u8,
    /// Code-block height exponent (4-6).
    pub code_block_height_exp: u8,
    /// Output JP2 file format (vs raw codestream).
    pub output_jp2: bool,
}

impl Jpeg2000EncoderConfig {
    /// Create a new encoder configuration.
    pub fn new(width: u32, height: u32, num_components: u16) -> Self {
        Self {
            width,
            height,
            num_components,
            bit_depth: 8,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_decomposition_levels: 5,
            num_layers: 1,
            progression_order: ProgressionOrder::Lrcp,
            wavelet_transform: WaveletTransform::Irreversible9x7,
            mct: true,
            compression_ratio: 10.0,
            target_psnr: 0.0,
            profile: Jpeg2000Profile::Part1,
            code_block_width_exp: 6,
            code_block_height_exp: 6,
            output_jp2: false,
        }
    }

    /// Create lossless configuration.
    pub fn lossless(width: u32, height: u32, num_components: u16) -> Self {
        Self {
            width,
            height,
            num_components,
            bit_depth: 8,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_decomposition_levels: 5,
            num_layers: 1,
            progression_order: ProgressionOrder::Lrcp,
            wavelet_transform: WaveletTransform::Reversible5x3,
            mct: true,
            compression_ratio: 0.0,
            target_psnr: 0.0,
            profile: Jpeg2000Profile::Lossless,
            code_block_width_exp: 6,
            code_block_height_exp: 6,
            output_jp2: false,
        }
    }

    /// Create Cinema 2K configuration.
    pub fn cinema_2k(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            num_components: 3,
            bit_depth: 12,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_decomposition_levels: 5,
            num_layers: 1,
            progression_order: ProgressionOrder::Cprl,
            wavelet_transform: WaveletTransform::Irreversible9x7,
            mct: true,
            compression_ratio: 24.0,
            target_psnr: 0.0,
            profile: Jpeg2000Profile::Cinema2k,
            code_block_width_exp: 5,
            code_block_height_exp: 5,
            output_jp2: false,
        }
    }

    /// Create Cinema 4K configuration.
    pub fn cinema_4k(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            num_components: 3,
            bit_depth: 12,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_decomposition_levels: 6,
            num_layers: 1,
            progression_order: ProgressionOrder::Cprl,
            wavelet_transform: WaveletTransform::Irreversible9x7,
            mct: true,
            compression_ratio: 24.0,
            target_psnr: 0.0,
            profile: Jpeg2000Profile::Cinema4k,
            code_block_width_exp: 5,
            code_block_height_exp: 5,
            output_jp2: false,
        }
    }

    /// Set bit depth.
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set compression ratio.
    pub fn with_compression_ratio(mut self, ratio: f32) -> Self {
        self.compression_ratio = ratio;
        self
    }

    /// Set target PSNR.
    pub fn with_target_psnr(mut self, psnr: f32) -> Self {
        self.target_psnr = psnr;
        self
    }

    /// Set tile size.
    pub fn with_tiles(mut self, width: u32, height: u32) -> Self {
        self.tile_width = width;
        self.tile_height = height;
        self
    }

    /// Set number of decomposition levels.
    pub fn with_decomposition_levels(mut self, levels: u8) -> Self {
        self.num_decomposition_levels = levels;
        self
    }

    /// Set progression order.
    pub fn with_progression_order(mut self, order: ProgressionOrder) -> Self {
        self.progression_order = order;
        self
    }

    /// Enable lossless mode.
    pub fn with_lossless(mut self) -> Self {
        self.wavelet_transform = WaveletTransform::Reversible5x3;
        self.compression_ratio = 0.0;
        self.profile = Jpeg2000Profile::Lossless;
        self
    }

    /// Output JP2 file format.
    pub fn with_jp2_output(mut self) -> Self {
        self.output_jp2 = true;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Jpeg2000Error::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        if self.bit_depth == 0 || self.bit_depth > 38 {
            return Err(Jpeg2000Error::UnsupportedBitDepth(self.bit_depth));
        }

        if self.num_components == 0 || self.num_components > 16384 {
            return Err(Jpeg2000Error::UnsupportedComponentCount(self.num_components));
        }

        if self.code_block_width_exp < 2 || self.code_block_width_exp > 10 {
            return Err(Jpeg2000Error::EncodingError(
                "Code-block width exponent must be 2-10".into(),
            ));
        }

        if self.code_block_height_exp < 2 || self.code_block_height_exp > 10 {
            return Err(Jpeg2000Error::EncodingError(
                "Code-block height exponent must be 2-10".into(),
            ));
        }

        Ok(())
    }

    /// Get effective tile width.
    pub fn effective_tile_width(&self) -> u32 {
        if self.tile_width == 0 {
            self.width
        } else {
            self.tile_width
        }
    }

    /// Get effective tile height.
    pub fn effective_tile_height(&self) -> u32 {
        if self.tile_height == 0 {
            self.height
        } else {
            self.tile_height
        }
    }

    /// Check if lossless encoding.
    pub fn is_lossless(&self) -> bool {
        self.wavelet_transform == WaveletTransform::Reversible5x3
            && self.compression_ratio == 0.0
    }
}

impl Default for Jpeg2000EncoderConfig {
    fn default() -> Self {
        Self::new(256, 256, 3)
    }
}

/// JPEG2000 encoder.
///
/// Encodes images to JPEG2000 format.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_jpeg2000::{Jpeg2000Encoder, Jpeg2000EncoderConfig};
///
/// let config = Jpeg2000EncoderConfig::new(1920, 1080, 3);
/// let mut encoder = Jpeg2000Encoder::new(config)?;
/// let encoded = encoder.encode(&image_data)?;
/// ```
pub struct Jpeg2000Encoder {
    /// Encoder configuration.
    config: Jpeg2000EncoderConfig,
    /// Images encoded.
    images_encoded: u64,
    /// Bytes output.
    bytes_output: u64,
    /// FFI encoder (when available).
    #[cfg(feature = "ffi-openjpeg")]
    ffi_encoder: Option<Jpeg2000FfiEncoder>,
}

impl fmt::Debug for Jpeg2000Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Jpeg2000Encoder");
        s.field("config", &self.config);
        s.field("images_encoded", &self.images_encoded);
        s.field("bytes_output", &self.bytes_output);
        #[cfg(feature = "ffi-openjpeg")]
        s.field("ffi_encoder", &self.ffi_encoder.is_some());
        s.finish_non_exhaustive()
    }
}

/// Encoded JPEG2000 packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Is JP2 file format (vs raw codestream).
    pub is_jp2: bool,
}

impl Jpeg2000Encoder {
    /// Create a new encoder with FFI support.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn new(config: Jpeg2000EncoderConfig) -> Result<Self> {
        config.validate()?;

        // Create FFI encoder configuration
        let ffi_config = Jpeg2000EncoderConfigFfi {
            width: config.width,
            height: config.height,
            num_components: config.num_components,
            bit_depth: config.bit_depth,
            is_signed: config.is_signed,
            tile_width: config.tile_width,
            tile_height: config.tile_height,
            num_decomposition_levels: config.num_decomposition_levels,
            num_layers: config.num_layers,
            compression_ratio: config.compression_ratio,
            lossless: config.is_lossless(),
            output_jp2: config.output_jp2,
        };

        let ffi_encoder = Jpeg2000FfiEncoder::new(ffi_config)?;

        Ok(Self {
            config,
            images_encoded: 0,
            bytes_output: 0,
            ffi_encoder: Some(ffi_encoder),
        })
    }

    /// Create a new encoder (without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn new(config: Jpeg2000EncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            images_encoded: 0,
            bytes_output: 0,
        })
    }

    /// Check if FFI encoding is available.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn is_encoding_available(&self) -> bool {
        self.ffi_encoder.is_some()
    }

    /// Check if FFI encoding is available (without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn is_encoding_available(&self) -> bool {
        false
    }

    /// Encode an image using OpenJPEG.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn encode(&mut self, data: &[u8]) -> Result<EncodedPacket> {
        // Use FFI encoder if available
        if let Some(ref mut ffi) = self.ffi_encoder {
            let packet = ffi.encode(data)?;
            self.images_encoded += 1;
            self.bytes_output += packet.data.len() as u64;
            return Ok(packet);
        }

        // Fallback to minimal codestream
        self.encode_minimal(data)
    }

    /// Encode an image (stub without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn encode(&mut self, data: &[u8]) -> Result<EncodedPacket> {
        self.encode_minimal(data)
    }

    /// Encode a minimal valid codestream (without actual compression).
    fn encode_minimal(&mut self, _data: &[u8]) -> Result<EncodedPacket> {
        // Without FFI, we can only generate a minimal valid codestream
        let mut output = Vec::new();

        if self.config.output_jp2 {
            self.write_jp2_header(&mut output)?;
        }

        self.write_codestream(&mut output)?;

        self.images_encoded += 1;
        self.bytes_output += output.len() as u64;

        Ok(EncodedPacket {
            data: output,
            is_jp2: self.config.output_jp2,
        })
    }

    /// Write JP2 file header boxes.
    fn write_jp2_header(&self, data: &mut Vec<u8>) -> Result<()> {
        // Signature box
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0C]); // Length
        data.extend_from_slice(b"jP  "); // Type
        data.extend_from_slice(&[0x0D, 0x0A, 0x87, 0x0A]); // Signature

        // File Type box
        let ftyp_len = 20u32;
        let mut ftyp = vec![0u8; ftyp_len as usize];
        BigEndian::write_u32(&mut ftyp[0..4], ftyp_len);
        ftyp[4..8].copy_from_slice(b"ftyp");
        ftyp[8..12].copy_from_slice(b"jp2 "); // Brand
        BigEndian::write_u32(&mut ftyp[12..16], 0); // Minor version
        ftyp[16..20].copy_from_slice(b"jp2 "); // Compatibility
        data.extend_from_slice(&ftyp);

        // JP2 Header super box
        let ihdr_len = 22u32;
        let colr_len = 15u32;
        let jp2h_len = 8 + ihdr_len + colr_len;

        // jp2h box header
        let mut jp2h = vec![0u8; 8];
        BigEndian::write_u32(&mut jp2h[0..4], jp2h_len);
        jp2h[4..8].copy_from_slice(b"jp2h");
        data.extend_from_slice(&jp2h);

        // ihdr box
        let mut ihdr = vec![0u8; ihdr_len as usize];
        BigEndian::write_u32(&mut ihdr[0..4], ihdr_len);
        ihdr[4..8].copy_from_slice(b"ihdr");
        BigEndian::write_u32(&mut ihdr[8..12], self.config.height);
        BigEndian::write_u32(&mut ihdr[12..16], self.config.width);
        BigEndian::write_u16(&mut ihdr[16..18], self.config.num_components);
        ihdr[18] = self.config.bit_depth - 1; // BPC
        ihdr[19] = 7; // Compression type (JPEG2000)
        ihdr[20] = 0; // Colorspace unknown
        ihdr[21] = 0; // No IP
        data.extend_from_slice(&ihdr);

        // colr box (enumerated colorspace)
        let mut colr = vec![0u8; colr_len as usize];
        BigEndian::write_u32(&mut colr[0..4], colr_len);
        colr[4..8].copy_from_slice(b"colr");
        colr[8] = 1; // Method: enumerated
        colr[9] = 0; // Precedence
        colr[10] = 0; // Approximation
        let cs = if self.config.num_components == 1 { 17u32 } else { 16u32 }; // Gray or sRGB
        BigEndian::write_u32(&mut colr[11..15], cs);
        data.extend_from_slice(&colr);

        Ok(())
    }

    /// Write codestream.
    fn write_codestream(&self, data: &mut Vec<u8>) -> Result<()> {
        // Mark start for jp2c box if needed
        let cs_start = if self.config.output_jp2 {
            // jp2c box header (will update length later)
            let jp2c_start = data.len();
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Length placeholder
            data.extend_from_slice(b"jp2c");
            jp2c_start
        } else {
            0
        };

        let cs_data_start = data.len();

        // SOC (Start of Codestream)
        let soc_code = MarkerType::Soc.code();
        data.push((soc_code >> 8) as u8);
        data.push(soc_code as u8);

        // SIZ (Image and Tile Size)
        self.write_siz(data)?;

        // COD (Coding Style Default)
        self.write_cod(data)?;

        // QCD (Quantization Default)
        self.write_qcd(data)?;

        // SOT (Start of Tile)
        self.write_sot(data)?;

        // SOD (Start of Data)
        let sod_code = MarkerType::Sod.code();
        data.push((sod_code >> 8) as u8);
        data.push(sod_code as u8);

        // Placeholder tile data (minimal)
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        // EOC (End of Codestream)
        let eoc_code = MarkerType::Eoc.code();
        data.push((eoc_code >> 8) as u8);
        data.push(eoc_code as u8);

        // Update jp2c box length if needed
        if self.config.output_jp2 {
            let cs_len = (data.len() - cs_start) as u32;
            BigEndian::write_u32(&mut data[cs_start..cs_start + 4], cs_len);
        }

        let _ = cs_data_start; // Used for offset calculation

        Ok(())
    }

    /// Write SIZ marker.
    fn write_siz(&self, data: &mut Vec<u8>) -> Result<()> {
        let siz_code = MarkerType::Siz.code();
        data.push((siz_code >> 8) as u8);
        data.push(siz_code as u8);

        // Lsiz (length)
        let num_comps = self.config.num_components as usize;
        let lsiz = (38 + 3 * num_comps) as u16;
        data.push((lsiz >> 8) as u8);
        data.push(lsiz as u8);

        // Rsiz (profile)
        let rsiz = match self.config.profile {
            Jpeg2000Profile::Cinema2k => 3u16,
            Jpeg2000Profile::Cinema4k => 4u16,
            _ => 0u16,
        };
        data.push((rsiz >> 8) as u8);
        data.push(rsiz as u8);

        // Xsiz (width)
        let mut buf = [0u8; 4];
        BigEndian::write_u32(&mut buf, self.config.width);
        data.extend_from_slice(&buf);

        // Ysiz (height)
        BigEndian::write_u32(&mut buf, self.config.height);
        data.extend_from_slice(&buf);

        // XOsiz (horizontal offset)
        BigEndian::write_u32(&mut buf, 0);
        data.extend_from_slice(&buf);

        // YOsiz (vertical offset)
        BigEndian::write_u32(&mut buf, 0);
        data.extend_from_slice(&buf);

        // XTsiz (tile width)
        BigEndian::write_u32(&mut buf, self.config.effective_tile_width());
        data.extend_from_slice(&buf);

        // YTsiz (tile height)
        BigEndian::write_u32(&mut buf, self.config.effective_tile_height());
        data.extend_from_slice(&buf);

        // XTOsiz (tile horizontal offset)
        BigEndian::write_u32(&mut buf, 0);
        data.extend_from_slice(&buf);

        // YTOsiz (tile vertical offset)
        BigEndian::write_u32(&mut buf, 0);
        data.extend_from_slice(&buf);

        // Csiz (number of components)
        data.push((self.config.num_components >> 8) as u8);
        data.push(self.config.num_components as u8);

        // Component parameters
        for _ in 0..self.config.num_components {
            // Ssiz (bit depth)
            let ssiz = if self.config.is_signed {
                0x80 | (self.config.bit_depth - 1)
            } else {
                self.config.bit_depth - 1
            };
            data.push(ssiz);

            // XRsiz (horizontal sub-sampling)
            data.push(1);

            // YRsiz (vertical sub-sampling)
            data.push(1);
        }

        Ok(())
    }

    /// Write COD marker.
    fn write_cod(&self, data: &mut Vec<u8>) -> Result<()> {
        let cod_code = MarkerType::Cod.code();
        data.push((cod_code >> 8) as u8);
        data.push(cod_code as u8);

        // Lcod (length)
        let lcod = 12u16;
        data.push((lcod >> 8) as u8);
        data.push(lcod as u8);

        // Scod (coding style)
        data.push(0);

        // SGcod (progression order)
        data.push(self.config.progression_order.code());

        // Number of layers
        data.push((self.config.num_layers >> 8) as u8);
        data.push(self.config.num_layers as u8);

        // MCT (multiple component transform)
        data.push(if self.config.mct && self.config.num_components >= 3 { 1 } else { 0 });

        // SPcod
        // Number of decomposition levels
        data.push(self.config.num_decomposition_levels);

        // Code-block width
        data.push(self.config.code_block_width_exp - 2);

        // Code-block height
        data.push(self.config.code_block_height_exp - 2);

        // Code-block style
        data.push(0);

        // Wavelet transform
        data.push(if self.config.wavelet_transform == WaveletTransform::Reversible5x3 { 1 } else { 0 });

        Ok(())
    }

    /// Write QCD marker.
    fn write_qcd(&self, data: &mut Vec<u8>) -> Result<()> {
        let qcd_code = MarkerType::Qcd.code();
        data.push((qcd_code >> 8) as u8);
        data.push(qcd_code as u8);

        let num_bands = 1 + 3 * self.config.num_decomposition_levels as usize;

        if self.config.is_lossless() {
            // No quantization (reversible)
            let lqcd = (3 + num_bands) as u16;
            data.push((lqcd >> 8) as u8);
            data.push(lqcd as u8);

            // Sqcd (no quantization, 2 guard bits)
            data.push(0x40);

            // SPqcd (exponent for each band)
            for i in 0..num_bands {
                let exp = if i == 0 { 8 } else { 8 + (i - 1) / 3 };
                data.push((exp as u8) << 3);
            }
        } else {
            // Scalar derived quantization
            let lqcd = 5u16;
            data.push((lqcd >> 8) as u8);
            data.push(lqcd as u8);

            // Sqcd (scalar derived, 2 guard bits)
            data.push(0x41);

            // SPqcd (base step size)
            // Use reasonable default values
            data.push(0x48); // Exponent = 9, mantissa MSB
            data.push(0x00); // Mantissa LSB
        }

        Ok(())
    }

    /// Write SOT marker.
    fn write_sot(&self, data: &mut Vec<u8>) -> Result<()> {
        let sot_code = MarkerType::Sot.code();
        data.push((sot_code >> 8) as u8);
        data.push(sot_code as u8);

        // Lsot (length = 10)
        data.push(0x00);
        data.push(0x0A);

        // Isot (tile index = 0)
        data.push(0x00);
        data.push(0x00);

        // Psot (tile-part length = 0, meaning rest of codestream)
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);

        // TPsot (tile-part index = 0)
        data.push(0x00);

        // TNsot (number of tile-parts = 1)
        data.push(0x01);

        Ok(())
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &Jpeg2000EncoderConfig {
        &self.config
    }

    /// Get total images encoded.
    pub fn images_encoded(&self) -> u64 {
        self.images_encoded
    }

    /// Get total bytes output.
    pub fn bytes_output(&self) -> u64 {
        self.bytes_output
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.images_encoded = 0;
        self.bytes_output = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = Jpeg2000EncoderConfig::new(1920, 1080, 3);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.num_components, 3);
    }

    #[test]
    fn test_config_lossless() {
        let config = Jpeg2000EncoderConfig::lossless(1920, 1080, 3);
        assert!(config.is_lossless());
        assert_eq!(config.wavelet_transform, WaveletTransform::Reversible5x3);
    }

    #[test]
    fn test_config_cinema_2k() {
        let config = Jpeg2000EncoderConfig::cinema_2k(2048, 1080);
        assert_eq!(config.profile, Jpeg2000Profile::Cinema2k);
        assert_eq!(config.bit_depth, 12);
    }

    #[test]
    fn test_config_cinema_4k() {
        let config = Jpeg2000EncoderConfig::cinema_4k(4096, 2160);
        assert_eq!(config.profile, Jpeg2000Profile::Cinema4k);
        assert_eq!(config.num_decomposition_levels, 6);
    }

    #[test]
    fn test_config_validation() {
        let config = Jpeg2000EncoderConfig::new(1920, 1080, 3);
        assert!(config.validate().is_ok());

        // Invalid dimensions
        let config = Jpeg2000EncoderConfig::new(0, 1080, 3);
        assert!(config.validate().is_err());

        // Invalid bit depth
        let config = Jpeg2000EncoderConfig::new(1920, 1080, 3).with_bit_depth(0);
        assert!(config.validate().is_err());

        // Invalid component count
        let mut config = Jpeg2000EncoderConfig::new(1920, 1080, 3);
        config.num_components = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_effective_tile_size() {
        let config = Jpeg2000EncoderConfig::new(1920, 1080, 3);
        assert_eq!(config.effective_tile_width(), 1920);
        assert_eq!(config.effective_tile_height(), 1080);

        let config = config.with_tiles(256, 256);
        assert_eq!(config.effective_tile_width(), 256);
        assert_eq!(config.effective_tile_height(), 256);
    }

    #[test]
    fn test_encoder_new() {
        let config = Jpeg2000EncoderConfig::new(256, 256, 3);
        let encoder = Jpeg2000Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_is_encoding_available() {
        let config = Jpeg2000EncoderConfig::new(256, 256, 3);
        let encoder = Jpeg2000Encoder::new(config).unwrap();
        #[cfg(feature = "ffi-openjpeg")]
        assert!(encoder.is_encoding_available());
        #[cfg(not(feature = "ffi-openjpeg"))]
        assert!(!encoder.is_encoding_available());
    }

    #[test]
    fn test_encode_minimal() {
        let config = Jpeg2000EncoderConfig::new(8, 8, 1);
        let mut encoder = Jpeg2000Encoder::new(config).unwrap();

        let data = vec![128u8; 64];
        let result = encoder.encode(&data);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(!packet.data.is_empty());
        assert!(!packet.is_jp2);
    }

    #[test]
    fn test_encode_jp2() {
        let config = Jpeg2000EncoderConfig::new(8, 8, 3).with_jp2_output();
        let mut encoder = Jpeg2000Encoder::new(config).unwrap();

        let data = vec![128u8; 192];
        let result = encoder.encode(&data);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.is_jp2);
        // Check JP2 signature
        assert_eq!(&packet.data[4..8], b"jP  ");
    }

    #[test]
    fn test_encoder_stats() {
        let config = Jpeg2000EncoderConfig::new(8, 8, 1);
        let mut encoder = Jpeg2000Encoder::new(config).unwrap();

        let data = vec![128u8; 64];
        encoder.encode(&data).unwrap();

        assert_eq!(encoder.images_encoded(), 1);
        assert!(encoder.bytes_output() > 0);

        encoder.reset_stats();
        assert_eq!(encoder.images_encoded(), 0);
        assert_eq!(encoder.bytes_output(), 0);
    }

    #[test]
    fn test_config_builders() {
        let config = Jpeg2000EncoderConfig::new(256, 256, 3)
            .with_compression_ratio(20.0)
            .with_decomposition_levels(6)
            .with_progression_order(ProgressionOrder::Cprl);

        assert_eq!(config.compression_ratio, 20.0);
        assert_eq!(config.num_decomposition_levels, 6);
        assert_eq!(config.progression_order, ProgressionOrder::Cprl);
    }

    #[test]
    fn test_encoded_packet() {
        let packet = EncodedPacket {
            data: vec![0xFF, 0x4F],
            is_jp2: false,
        };
        assert_eq!(packet.data.len(), 2);
        assert!(!packet.is_jp2);
    }
}
