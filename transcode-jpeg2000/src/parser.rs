//! JPEG2000 codestream parser.
//!
//! Parses JPEG2000 codestream (J2K) and file format (JP2) headers.

use crate::types::*;
use crate::{Jpeg2000Error, MarkerType, Result};
use byteorder::{BigEndian, ByteOrder};

/// Codestream information extracted from headers.
#[derive(Debug, Clone)]
pub struct CodestreamInfo {
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
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Number of tiles.
    pub num_tiles: u32,
    /// Number of decomposition levels.
    pub num_decomposition_levels: u8,
    /// Number of quality layers.
    pub num_layers: u16,
    /// Progression order.
    pub progression_order: ProgressionOrder,
    /// Wavelet transform type.
    pub wavelet_transform: WaveletTransform,
    /// Multiple component transform.
    pub mct: bool,
    /// Profile.
    pub profile: Jpeg2000Profile,
}

impl Default for CodestreamInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            num_components: 0,
            bit_depth: 8,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_tiles: 0,
            num_decomposition_levels: 5,
            num_layers: 1,
            progression_order: ProgressionOrder::Lrcp,
            wavelet_transform: WaveletTransform::Irreversible9x7,
            mct: false,
            profile: Jpeg2000Profile::Part1,
        }
    }
}

/// JPEG2000 codestream parser.
#[derive(Debug, Default)]
pub struct Jpeg2000Parser {
    /// Parsed codestream.
    codestream: Codestream,
    /// Current position in data.
    position: usize,
}

impl Jpeg2000Parser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse codestream header and return info.
    pub fn parse_header(&mut self, data: &[u8]) -> Result<CodestreamInfo> {
        self.position = 0;
        self.codestream = Codestream::default();

        // Check for JP2 file format signature
        if data.len() >= 12 && &data[4..8] == b"jP  " {
            return self.parse_jp2_file(data);
        }

        // Check for raw codestream (SOC marker)
        if data.len() >= 2 {
            let marker = BigEndian::read_u16(&data[0..2]);
            if marker == MarkerType::Soc.code() {
                return self.parse_codestream(data);
            }
        }

        Err(Jpeg2000Error::InvalidCodestream(
            "Not a valid JPEG2000 file".into(),
        ))
    }

    /// Parse JP2 file format.
    fn parse_jp2_file(&mut self, data: &[u8]) -> Result<CodestreamInfo> {
        self.position = 0;

        // Parse boxes until we find codestream
        while self.position + 8 <= data.len() {
            let box_len = BigEndian::read_u32(&data[self.position..self.position + 4]) as usize;
            let box_type = BigEndian::read_u32(&data[self.position + 4..self.position + 8]);

            let box_type_enum = BoxType::from_code(box_type);

            // Handle extended length boxes
            let (actual_len, header_size) = if box_len == 1 {
                if self.position + 16 > data.len() {
                    break;
                }
                let ext_len = BigEndian::read_u64(&data[self.position + 8..self.position + 16]);
                (ext_len as usize, 16)
            } else if box_len == 0 {
                // Box extends to end of file
                (data.len() - self.position, 8)
            } else {
                (box_len, 8)
            };

            if box_type_enum == BoxType::CodestreamBox {
                // Found codestream, parse it
                let cs_start = self.position + header_size;
                if cs_start < data.len() {
                    return self.parse_codestream(&data[cs_start..]);
                }
            }

            self.position += actual_len;
        }

        Err(Jpeg2000Error::InvalidCodestream(
            "No codestream found in JP2 file".into(),
        ))
    }

    /// Parse raw codestream.
    fn parse_codestream(&mut self, data: &[u8]) -> Result<CodestreamInfo> {
        self.position = 0;

        // Check SOC marker
        if data.len() < 2 {
            return Err(Jpeg2000Error::BufferTooSmall {
                needed: 2,
                available: data.len(),
            });
        }

        let soc = BigEndian::read_u16(&data[0..2]);
        if soc != MarkerType::Soc.code() {
            return Err(Jpeg2000Error::MissingMarker {
                marker: MarkerType::Soc,
            });
        }
        self.position = 2;

        // Parse main header markers
        while self.position + 2 <= data.len() {
            let marker_code = BigEndian::read_u16(&data[self.position..self.position + 2]);
            let marker = MarkerType::from_code(marker_code);

            // SOD or EOC marks end of main header
            if matches!(marker, MarkerType::Sod | MarkerType::Eoc) {
                break;
            }

            // SOT marks start of tile data
            if marker == MarkerType::Sot {
                break;
            }

            self.position += 2;

            if marker.has_length() {
                if self.position + 2 > data.len() {
                    break;
                }
                let length = BigEndian::read_u16(&data[self.position..self.position + 2]) as usize;
                if length < 2 || self.position + length > data.len() {
                    break;
                }

                let segment_data = &data[self.position + 2..self.position + length];
                self.parse_marker_segment(marker, segment_data)?;
                self.position += length;
            }
        }

        self.build_codestream_info()
    }

    /// Parse a marker segment.
    fn parse_marker_segment(&mut self, marker: MarkerType, data: &[u8]) -> Result<()> {
        match marker {
            MarkerType::Siz => self.parse_siz(data),
            MarkerType::Cod => self.parse_cod(data),
            MarkerType::Qcd => self.parse_qcd(data),
            MarkerType::Com => self.parse_com(data),
            _ => Ok(()), // Ignore other markers for now
        }
    }

    /// Parse SIZ marker (image and tile size).
    fn parse_siz(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 36 {
            return Err(Jpeg2000Error::InvalidMarkerSegment {
                marker: MarkerType::Siz,
            });
        }

        let profile = BigEndian::read_u16(&data[0..2]);
        let ref_grid_width = BigEndian::read_u32(&data[2..6]);
        let ref_grid_height = BigEndian::read_u32(&data[6..10]);
        let x_offset = BigEndian::read_u32(&data[10..14]);
        let y_offset = BigEndian::read_u32(&data[14..18]);
        let tile_width = BigEndian::read_u32(&data[18..22]);
        let tile_height = BigEndian::read_u32(&data[22..26]);
        let tile_x_offset = BigEndian::read_u32(&data[26..30]);
        let tile_y_offset = BigEndian::read_u32(&data[30..34]);
        let num_components = BigEndian::read_u16(&data[34..36]);

        let mut components = Vec::with_capacity(num_components as usize);
        let mut pos = 36;

        for _ in 0..num_components {
            if pos + 3 > data.len() {
                break;
            }
            let ssiz = data[pos];
            let xrsiz = data[pos + 1];
            let yrsiz = data[pos + 2];

            let is_signed = (ssiz & 0x80) != 0;
            let bit_depth = (ssiz & 0x7F) + 1;

            let comp_width = (ref_grid_width - x_offset).div_ceil(xrsiz as u32);
            let comp_height = (ref_grid_height - y_offset).div_ceil(yrsiz as u32);

            components.push(ComponentInfo {
                dx: xrsiz,
                dy: yrsiz,
                bit_depth,
                is_signed,
                width: comp_width,
                height: comp_height,
            });

            pos += 3;
        }

        self.codestream.siz = Some(SizMarker {
            ref_grid_width,
            ref_grid_height,
            x_offset,
            y_offset,
            tile_width,
            tile_height,
            tile_x_offset,
            tile_y_offset,
            num_components,
            components,
            profile,
        });

        Ok(())
    }

    /// Parse COD marker (coding style default).
    fn parse_cod(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 10 {
            return Err(Jpeg2000Error::InvalidMarkerSegment {
                marker: MarkerType::Cod,
            });
        }

        let coding_style = data[0];
        let progression_order =
            ProgressionOrder::from_code(data[1]).unwrap_or(ProgressionOrder::Lrcp);
        let num_layers = BigEndian::read_u16(&data[2..4]);
        let mct = data[4];
        let num_decomposition_levels = data[5];
        let code_block_width_exp = data[6] & 0x0F;
        let code_block_height_exp = data[7] & 0x0F;
        let code_block_style = data[8];
        let wavelet_byte = data[9];

        let wavelet_transform = if wavelet_byte == 0 {
            WaveletTransform::Irreversible9x7
        } else {
            WaveletTransform::Reversible5x3
        };

        // Parse precinct sizes if present
        let mut precinct_sizes = Vec::new();
        if (coding_style & 0x01) != 0 && data.len() > 10 {
            for &byte in data.iter().skip(10) {
                let ppx = byte & 0x0F;
                let ppy = (byte >> 4) & 0x0F;
                precinct_sizes.push((ppx, ppy));
            }
        }

        self.codestream.cod = Some(CodMarker {
            coding_style,
            progression_order,
            num_layers,
            mct,
            num_decomposition_levels,
            code_block_width_exp,
            code_block_height_exp,
            code_block_style,
            wavelet_transform,
            precinct_sizes,
        });

        Ok(())
    }

    /// Parse QCD marker (quantization default).
    fn parse_qcd(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(Jpeg2000Error::InvalidMarkerSegment {
                marker: MarkerType::Qcd,
            });
        }

        let quantization_style = data[0];
        let guard_bits = (quantization_style >> 5) & 0x07;
        let qstyle = quantization_style & 0x1F;

        let mut step_sizes = Vec::new();
        let mut pos = 1;

        if qstyle == 0 {
            // No quantization (reversible)
            while pos < data.len() {
                let exponent = (data[pos] >> 3) & 0x1F;
                step_sizes.push(QuantizationStepSize {
                    exponent,
                    mantissa: 0,
                });
                pos += 1;
            }
        } else if qstyle == 1 || qstyle == 2 {
            // Scalar quantization
            while pos + 1 < data.len() {
                let byte0 = data[pos] as u16;
                let byte1 = data[pos + 1] as u16;
                let val = (byte0 << 8) | byte1;
                let exponent = ((val >> 11) & 0x1F) as u8;
                let mantissa = val & 0x07FF;
                step_sizes.push(QuantizationStepSize { exponent, mantissa });
                pos += 2;
            }
        }

        self.codestream.qcd = Some(QcdMarker {
            quantization_style,
            guard_bits,
            step_sizes,
        });

        Ok(())
    }

    /// Parse COM marker (comment).
    fn parse_com(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 2 {
            return Ok(());
        }

        let registration = BigEndian::read_u16(&data[0..2]);
        let comment_data = data[2..].to_vec();

        self.codestream.comments.push(ComMarker {
            registration,
            data: comment_data,
        });

        Ok(())
    }

    /// Build codestream info from parsed markers.
    fn build_codestream_info(&self) -> Result<CodestreamInfo> {
        let siz = self.codestream.siz.as_ref().ok_or(Jpeg2000Error::MissingMarker {
            marker: MarkerType::Siz,
        })?;

        let cod = self.codestream.cod.as_ref().ok_or(Jpeg2000Error::MissingMarker {
            marker: MarkerType::Cod,
        })?;

        let bit_depth = siz
            .components
            .first()
            .map(|c| c.bit_depth)
            .unwrap_or(8);

        let is_signed = siz
            .components
            .first()
            .map(|c| c.is_signed)
            .unwrap_or(false);

        // Determine profile from Rsiz
        let profile = match siz.profile {
            0 => Jpeg2000Profile::Part1,
            1 => Jpeg2000Profile::Part1,
            3 => Jpeg2000Profile::Cinema2k,
            4 => Jpeg2000Profile::Cinema4k,
            0x0100..=0x01FF => Jpeg2000Profile::Broadcast,
            0x0400..=0x04FF => Jpeg2000Profile::Imf,
            _ => Jpeg2000Profile::Custom,
        };

        Ok(CodestreamInfo {
            width: siz.image_width(),
            height: siz.image_height(),
            num_components: siz.num_components,
            bit_depth,
            is_signed,
            tile_width: siz.tile_width,
            tile_height: siz.tile_height,
            num_tiles: siz.num_tiles(),
            num_decomposition_levels: cod.num_decomposition_levels,
            num_layers: cod.num_layers,
            progression_order: cod.progression_order,
            wavelet_transform: cod.wavelet_transform,
            mct: cod.mct != 0,
            profile,
        })
    }

    /// Reset the parser.
    pub fn reset(&mut self) {
        self.codestream = Codestream::default();
        self.position = 0;
    }

    /// Get the parsed codestream.
    pub fn codestream(&self) -> &Codestream {
        &self.codestream
    }

    /// Check if data starts with JP2 signature.
    pub fn is_jp2_file(data: &[u8]) -> bool {
        data.len() >= 12 && &data[4..8] == b"jP  "
    }

    /// Check if data starts with codestream signature (SOC).
    pub fn is_codestream(data: &[u8]) -> bool {
        data.len() >= 2 && BigEndian::read_u16(&data[0..2]) == MarkerType::Soc.code()
    }

    /// Find the codestream in a JP2 file.
    pub fn find_codestream(data: &[u8]) -> Option<(usize, usize)> {
        if !Self::is_jp2_file(data) {
            if Self::is_codestream(data) {
                return Some((0, data.len()));
            }
            return None;
        }

        let mut pos = 0;
        while pos + 8 <= data.len() {
            let box_len = BigEndian::read_u32(&data[pos..pos + 4]) as usize;
            let box_type = BigEndian::read_u32(&data[pos + 4..pos + 8]);

            let (actual_len, header_size) = if box_len == 1 {
                if pos + 16 > data.len() {
                    break;
                }
                let ext_len = BigEndian::read_u64(&data[pos + 8..pos + 16]);
                (ext_len as usize, 16)
            } else if box_len == 0 {
                (data.len() - pos, 8)
            } else {
                (box_len, 8)
            };

            if BoxType::from_code(box_type) == BoxType::CodestreamBox {
                let cs_start = pos + header_size;
                let cs_len = actual_len - header_size;
                return Some((cs_start, cs_len));
            }

            pos += actual_len;
        }

        None
    }
}

/// JP2 file format parser.
#[derive(Debug, Default)]
pub struct Jp2Parser {
    /// Parsed file structure.
    file: Jp2File,
}

impl Jp2Parser {
    /// Create a new JP2 parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse JP2 file header.
    pub fn parse(&mut self, data: &[u8]) -> Result<&Jp2File> {
        self.file = Jp2File::default();

        if data.len() < 12 {
            return Err(Jpeg2000Error::BufferTooSmall {
                needed: 12,
                available: data.len(),
            });
        }

        // Check signature box
        let sig_len = BigEndian::read_u32(&data[0..4]);
        if sig_len != 12 || &data[4..8] != b"jP  " {
            return Err(Jpeg2000Error::InvalidCodestream(
                "Invalid JP2 signature".into(),
            ));
        }

        // Check signature value (0x0D0A870A)
        let sig_val = BigEndian::read_u32(&data[8..12]);
        if sig_val != 0x0D0A870A {
            return Err(Jpeg2000Error::InvalidCodestream(
                "Invalid JP2 signature value".into(),
            ));
        }

        // Parse remaining boxes
        let mut pos = 12;
        while pos + 8 <= data.len() {
            let box_len = BigEndian::read_u32(&data[pos..pos + 4]) as usize;
            let box_type = BigEndian::read_u32(&data[pos + 4..pos + 8]);

            let (actual_len, header_size) = if box_len == 1 {
                if pos + 16 > data.len() {
                    break;
                }
                let ext_len = BigEndian::read_u64(&data[pos + 8..pos + 16]);
                (ext_len as usize, 16)
            } else if box_len == 0 {
                (data.len() - pos, 8)
            } else {
                (box_len, 8)
            };

            let box_data = if pos + header_size + actual_len - header_size <= data.len() {
                &data[pos + header_size..pos + actual_len]
            } else {
                &[]
            };

            match BoxType::from_code(box_type) {
                BoxType::FileType => self.parse_ftyp(box_data)?,
                BoxType::Jp2Header => self.parse_jp2h(box_data)?,
                BoxType::CodestreamBox => {
                    self.file.codestream_offset = (pos + header_size) as u64;
                    self.file.codestream_length = (actual_len - header_size) as u64;
                }
                _ => {}
            }

            pos += actual_len;
        }

        Ok(&self.file)
    }

    /// Parse ftyp box.
    fn parse_ftyp(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 8 {
            return Ok(());
        }

        self.file.brand = Some(BigEndian::read_u32(&data[0..4]));
        self.file.minor_version = BigEndian::read_u32(&data[4..8]);

        let mut pos = 8;
        while pos + 4 <= data.len() {
            self.file
                .compatibility
                .push(BigEndian::read_u32(&data[pos..pos + 4]));
            pos += 4;
        }

        Ok(())
    }

    /// Parse jp2h box (JP2 header super box).
    fn parse_jp2h(&mut self, data: &[u8]) -> Result<()> {
        let mut pos = 0;
        while pos + 8 <= data.len() {
            let box_len = BigEndian::read_u32(&data[pos..pos + 4]) as usize;
            let box_type = BigEndian::read_u32(&data[pos + 4..pos + 8]);

            if box_len < 8 || pos + box_len > data.len() {
                break;
            }

            let box_data = &data[pos + 8..pos + box_len];

            match BoxType::from_code(box_type) {
                BoxType::ImageHeader => self.parse_ihdr(box_data)?,
                BoxType::ColourSpec => self.parse_colr(box_data)?,
                _ => {}
            }

            pos += box_len;
        }

        Ok(())
    }

    /// Parse ihdr box (image header).
    fn parse_ihdr(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 14 {
            return Ok(());
        }

        self.file.height = BigEndian::read_u32(&data[0..4]);
        self.file.width = BigEndian::read_u32(&data[4..8]);
        self.file.num_components = BigEndian::read_u16(&data[8..10]);
        self.file.bits_per_component = data[10] + 1;
        // data[11] = compression type (should be 7 for JP2)
        // data[12] = colorspace unknown
        // data[13] = intellectual property

        Ok(())
    }

    /// Parse colr box (colour specification).
    fn parse_colr(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let method = data[0];
        if method == 1 && data.len() >= 7 {
            // Enumerated colorspace
            let cs = BigEndian::read_u32(&data[3..7]);
            self.file.color_space = match cs {
                16 => ColorSpace::Srgb,
                17 => ColorSpace::Grayscale,
                18 => ColorSpace::YCbCr,
                _ => ColorSpace::Unknown,
            };
        }

        Ok(())
    }

    /// Get parsed file.
    pub fn file(&self) -> &Jp2File {
        &self.file
    }

    /// Reset the parser.
    pub fn reset(&mut self) {
        self.file = Jp2File::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_new() {
        let parser = Jpeg2000Parser::new();
        assert_eq!(parser.position, 0);
    }

    #[test]
    fn test_is_jp2_file() {
        let jp2_sig = [0x00, 0x00, 0x00, 0x0C, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A];
        assert!(Jpeg2000Parser::is_jp2_file(&jp2_sig));
        assert!(!Jpeg2000Parser::is_jp2_file(&[0xFF, 0x4F]));
    }

    #[test]
    fn test_is_codestream() {
        let soc = [0xFF, 0x4F];
        assert!(Jpeg2000Parser::is_codestream(&soc));
        assert!(!Jpeg2000Parser::is_codestream(&[0x00, 0x00]));
    }

    #[test]
    fn test_codestream_info_default() {
        let info = CodestreamInfo::default();
        assert_eq!(info.width, 0);
        assert_eq!(info.height, 0);
        assert_eq!(info.bit_depth, 8);
    }

    #[test]
    fn test_parse_siz() {
        let mut parser = Jpeg2000Parser::new();

        // Build minimal SIZ marker data
        let mut siz_data = vec![0u8; 39];
        // Rsiz (profile)
        BigEndian::write_u16(&mut siz_data[0..2], 0);
        // Xsiz (width)
        BigEndian::write_u32(&mut siz_data[2..6], 1920);
        // Ysiz (height)
        BigEndian::write_u32(&mut siz_data[6..10], 1080);
        // XOsiz
        BigEndian::write_u32(&mut siz_data[10..14], 0);
        // YOsiz
        BigEndian::write_u32(&mut siz_data[14..18], 0);
        // XTsiz (tile width)
        BigEndian::write_u32(&mut siz_data[18..22], 1920);
        // YTsiz (tile height)
        BigEndian::write_u32(&mut siz_data[22..26], 1080);
        // XTOsiz
        BigEndian::write_u32(&mut siz_data[26..30], 0);
        // YTOsiz
        BigEndian::write_u32(&mut siz_data[30..34], 0);
        // Csiz (num components)
        BigEndian::write_u16(&mut siz_data[34..36], 1);
        // Component: Ssiz, XRsiz, YRsiz
        siz_data[36] = 7; // 8 bits
        siz_data[37] = 1;
        siz_data[38] = 1;

        parser.parse_siz(&siz_data).unwrap();

        let siz = parser.codestream.siz.as_ref().unwrap();
        assert_eq!(siz.image_width(), 1920);
        assert_eq!(siz.image_height(), 1080);
        assert_eq!(siz.num_components, 1);
    }

    #[test]
    fn test_parse_cod() {
        let mut parser = Jpeg2000Parser::new();

        // Build minimal COD marker data
        let mut cod_data = vec![0u8; 12];
        cod_data[0] = 0; // Scod
        cod_data[1] = 0; // Progression order (LRCP)
        BigEndian::write_u16(&mut cod_data[2..4], 1); // Layers
        cod_data[4] = 1; // MCT
        cod_data[5] = 5; // Decomposition levels
        cod_data[6] = 4; // Code block width exp
        cod_data[7] = 4; // Code block height exp
        cod_data[8] = 0; // Code block style
        cod_data[9] = 0; // Wavelet transform (9/7)

        parser.parse_cod(&cod_data).unwrap();

        let cod = parser.codestream.cod.as_ref().unwrap();
        assert_eq!(cod.num_decomposition_levels, 5);
        assert_eq!(cod.num_layers, 1);
        assert_eq!(cod.progression_order, ProgressionOrder::Lrcp);
        assert_eq!(cod.wavelet_transform, WaveletTransform::Irreversible9x7);
    }

    #[test]
    fn test_parse_qcd() {
        let mut parser = Jpeg2000Parser::new();

        // Build minimal QCD marker data (no quantization)
        let qcd_data = vec![0x40, 0x48, 0x48, 0x48]; // guard=2, no quant, 3 step sizes

        parser.parse_qcd(&qcd_data).unwrap();

        let qcd = parser.codestream.qcd.as_ref().unwrap();
        assert!(qcd.is_no_quantization());
        assert_eq!(qcd.guard_bits, 2);
    }

    #[test]
    fn test_parse_com() {
        let mut parser = Jpeg2000Parser::new();

        // Build COM marker data
        let mut com_data = vec![0u8; 14];
        BigEndian::write_u16(&mut com_data[0..2], 1); // ISO-8859-15
        com_data[2..14].copy_from_slice(b"Test comment");

        parser.parse_com(&com_data).unwrap();

        assert_eq!(parser.codestream.comments.len(), 1);
        assert!(parser.codestream.comments[0].is_text());
        assert_eq!(
            parser.codestream.comments[0].as_text(),
            Some("Test comment".to_string())
        );
    }

    #[test]
    fn test_jp2_parser_new() {
        let parser = Jp2Parser::new();
        assert!(parser.file.brand.is_none());
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = Jpeg2000Parser::new();
        parser.position = 100;
        parser.reset();
        assert_eq!(parser.position, 0);
    }

    #[test]
    fn test_find_codestream_raw() {
        let soc = [0xFF, 0x4F, 0xFF, 0x51];
        let result = Jpeg2000Parser::find_codestream(&soc);
        assert_eq!(result, Some((0, 4)));
    }
}
