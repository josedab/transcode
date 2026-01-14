//! PNG decoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};
use super::{
    ChunkType, ColorType, InterlaceMethod, PNG_SIGNATURE, ADAM7_PASSES,
    crc32, read_u32_be,
};
use super::filter::{FilterType, unfilter_row};

/// PNG image information.
#[derive(Debug, Clone)]
pub struct PngInfo {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Bit depth.
    pub bit_depth: u8,
    /// Color type.
    pub color_type: ColorType,
    /// Interlace method.
    pub interlace: InterlaceMethod,
    /// Gamma value (if present).
    pub gamma: Option<f32>,
    /// Is sRGB.
    pub srgb: bool,
    /// Text metadata.
    pub text: Vec<(String, String)>,
}

/// PNG decoder.
pub struct PngDecoder {
    /// Image info.
    info: Option<PngInfo>,
    /// Palette.
    palette: Option<Vec<[u8; 3]>>,
    /// Transparency data.
    transparency: Option<Vec<u8>>,
    /// Compressed image data.
    compressed_data: Vec<u8>,
}

impl PngDecoder {
    /// Create a new PNG decoder.
    pub fn new() -> Self {
        Self {
            info: None,
            palette: None,
            transparency: None,
            compressed_data: Vec::new(),
        }
    }

    /// Get image info.
    pub fn info(&self) -> Option<&PngInfo> {
        self.info.as_ref()
    }

    /// Decode a PNG image.
    pub fn decode(&mut self, data: &[u8]) -> Result<Image> {
        // Verify signature
        if data.len() < 8 || data[0..8] != PNG_SIGNATURE {
            return Err(ImageError::DecoderError("Invalid PNG signature".into()));
        }

        self.info = None;
        self.palette = None;
        self.transparency = None;
        self.compressed_data.clear();

        let mut pos = 8;

        // Parse chunks
        while pos + 12 <= data.len() {
            let length = read_u32_be(&data[pos..]) as usize;
            let chunk_type = ChunkType::new([
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);

            if pos + 12 + length > data.len() {
                return Err(ImageError::DecoderError("Truncated chunk".into()));
            }

            let chunk_data = &data[pos + 8..pos + 8 + length];
            let stored_crc = read_u32_be(&data[pos + 8 + length..]);

            // Verify CRC
            let mut crc_data = Vec::with_capacity(4 + length);
            crc_data.extend_from_slice(&data[pos + 4..pos + 8]);
            crc_data.extend_from_slice(chunk_data);
            let calculated_crc = crc32(&crc_data);

            if stored_crc != calculated_crc {
                return Err(ImageError::DecoderError(format!(
                    "CRC mismatch for chunk {}",
                    chunk_type
                )));
            }

            self.process_chunk(chunk_type, chunk_data)?;

            if chunk_type == ChunkType::IEND {
                break;
            }

            pos += 12 + length;
        }

        // Decompress and decode image
        self.decode_image()
    }

    /// Process a chunk.
    fn process_chunk(&mut self, chunk_type: ChunkType, data: &[u8]) -> Result<()> {
        if chunk_type == ChunkType::IHDR {
            self.parse_ihdr(data)?;
        } else if chunk_type == ChunkType::PLTE {
            self.parse_plte(data)?;
        } else if chunk_type == ChunkType::IDAT {
            self.compressed_data.extend_from_slice(data);
        } else if chunk_type == ChunkType::TRNS {
            self.parse_trns(data)?;
        } else if chunk_type == ChunkType::GAMA {
            self.parse_gama(data)?;
        } else if chunk_type == ChunkType::SRGB {
            if let Some(ref mut info) = self.info {
                info.srgb = true;
            }
        } else if chunk_type == ChunkType::TEXT {
            self.parse_text(data)?;
        }
        // Ignore other chunks

        Ok(())
    }

    /// Parse IHDR chunk.
    fn parse_ihdr(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != 13 {
            return Err(ImageError::DecoderError("Invalid IHDR length".into()));
        }

        let width = read_u32_be(&data[0..]);
        let height = read_u32_be(&data[4..]);
        let bit_depth = data[8];
        let color_type = ColorType::from_u8(data[9])
            .ok_or_else(|| ImageError::DecoderError("Invalid color type".into()))?;
        let _compression = data[10];
        let _filter = data[11];
        let interlace = InterlaceMethod::from_u8(data[12])
            .ok_or_else(|| ImageError::DecoderError("Invalid interlace method".into()))?;

        // Validate bit depth for color type
        let valid = match color_type {
            ColorType::Grayscale => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
            ColorType::Rgb => matches!(bit_depth, 8 | 16),
            ColorType::Indexed => matches!(bit_depth, 1 | 2 | 4 | 8),
            ColorType::GrayscaleAlpha => matches!(bit_depth, 8 | 16),
            ColorType::Rgba => matches!(bit_depth, 8 | 16),
        };

        if !valid {
            return Err(ImageError::DecoderError(format!(
                "Invalid bit depth {} for color type {:?}",
                bit_depth, color_type
            )));
        }

        self.info = Some(PngInfo {
            width,
            height,
            bit_depth,
            color_type,
            interlace,
            gamma: None,
            srgb: false,
            text: Vec::new(),
        });

        Ok(())
    }

    /// Parse PLTE chunk.
    fn parse_plte(&mut self, data: &[u8]) -> Result<()> {
        if !data.len().is_multiple_of(3) || data.len() > 256 * 3 {
            return Err(ImageError::DecoderError("Invalid PLTE chunk".into()));
        }

        let mut palette = Vec::with_capacity(data.len() / 3);
        for i in (0..data.len()).step_by(3) {
            palette.push([data[i], data[i + 1], data[i + 2]]);
        }

        self.palette = Some(palette);
        Ok(())
    }

    /// Parse tRNS chunk.
    fn parse_trns(&mut self, data: &[u8]) -> Result<()> {
        self.transparency = Some(data.to_vec());
        Ok(())
    }

    /// Parse gAMA chunk.
    fn parse_gama(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != 4 {
            return Err(ImageError::DecoderError("Invalid gAMA chunk".into()));
        }

        let gamma_int = read_u32_be(data);
        if let Some(ref mut info) = self.info {
            info.gamma = Some(gamma_int as f32 / 100000.0);
        }

        Ok(())
    }

    /// Parse tEXt chunk.
    fn parse_text(&mut self, data: &[u8]) -> Result<()> {
        // Find null separator
        if let Some(null_pos) = data.iter().position(|&b| b == 0) {
            let keyword = String::from_utf8_lossy(&data[..null_pos]).to_string();
            let value = String::from_utf8_lossy(&data[null_pos + 1..]).to_string();

            if let Some(ref mut info) = self.info {
                info.text.push((keyword, value));
            }
        }

        Ok(())
    }

    /// Decode the image from compressed data.
    fn decode_image(&self) -> Result<Image> {
        let info = self.info.as_ref()
            .ok_or_else(|| ImageError::DecoderError("Missing IHDR".into()))?;

        // Decompress data using zlib
        let decompressed = self.decompress(&self.compressed_data)?;

        // Decode based on interlace method
        let raw_data = if info.interlace == InterlaceMethod::Adam7 {
            self.decode_interlaced(&decompressed, info)?
        } else {
            self.decode_non_interlaced(&decompressed, info)?
        };

        // Convert to image
        self.raw_to_image(&raw_data, info)
    }

    /// Decompress zlib data.
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple zlib decompression
        if data.len() < 2 {
            return Err(ImageError::DecoderError("Compressed data too short".into()));
        }

        // Check zlib header
        let cmf = data[0];
        let flg = data[1];

        let cm = cmf & 0x0F;
        if cm != 8 {
            return Err(ImageError::DecoderError("Unsupported compression method".into()));
        }

        let fdict = (flg & 0x20) != 0;
        if fdict {
            return Err(ImageError::DecoderError("Preset dictionary not supported".into()));
        }

        let start = if fdict { 6 } else { 2 };
        let end = data.len().saturating_sub(4); // Remove adler32 checksum

        // Decompress using DEFLATE
        self.inflate(&data[start..end])
    }

    /// DEFLATE decompression.
    fn inflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut reader = BitReader::new(data);

        loop {
            let bfinal = reader.read_bits(1)?;
            let btype = reader.read_bits(2)?;

            match btype {
                0 => {
                    // Uncompressed block
                    reader.align_to_byte();
                    let len = reader.read_u16_le()?;
                    let nlen = reader.read_u16_le()?;
                    if len != !nlen {
                        return Err(ImageError::DecoderError("Invalid uncompressed block".into()));
                    }
                    for _ in 0..len {
                        output.push(reader.read_byte()?);
                    }
                }
                1 => {
                    // Fixed Huffman codes
                    self.inflate_fixed_huffman(&mut reader, &mut output)?;
                }
                2 => {
                    // Dynamic Huffman codes
                    self.inflate_dynamic_huffman(&mut reader, &mut output)?;
                }
                _ => {
                    return Err(ImageError::DecoderError("Invalid block type".into()));
                }
            }

            if bfinal == 1 {
                break;
            }
        }

        Ok(output)
    }

    /// Inflate with fixed Huffman codes.
    fn inflate_fixed_huffman(&self, reader: &mut BitReader, output: &mut Vec<u8>) -> Result<()> {
        loop {
            let code = self.decode_fixed_literal(reader)?;

            if code < 256 {
                output.push(code as u8);
            } else if code == 256 {
                break;
            } else {
                let length = self.decode_length(code, reader)?;
                let distance = self.decode_fixed_distance(reader)?;

                if distance > output.len() {
                    return Err(ImageError::DecoderError("Invalid distance".into()));
                }

                for _ in 0..length {
                    let idx = output.len() - distance;
                    output.push(output[idx]);
                }
            }
        }

        Ok(())
    }

    /// Decode fixed literal/length code.
    fn decode_fixed_literal(&self, reader: &mut BitReader) -> Result<u16> {
        // Read 7 bits first
        let code = reader.read_bits_rev(7)? as u16;

        if code < 24 {
            // 256-279: 7 bits (0000000-0010111)
            Ok(256 + code)
        } else if code < 72 {
            // 0-143: 8 bits (00110000-10111111)
            let extra = reader.read_bits_rev(1)? as u16;
            Ok((code - 24) * 2 + extra)
        } else if code < 96 {
            // 280-287: 8 bits (11000000-11000111)
            let extra = reader.read_bits_rev(1)? as u16;
            Ok(280 + (code - 72) * 2 + extra)
        } else {
            // 144-255: 9 bits (110010000-111111111)
            let extra = reader.read_bits_rev(2)? as u16;
            Ok(144 + (code - 96) * 4 + extra)
        }
    }

    /// Decode length from code.
    fn decode_length(&self, code: u16, reader: &mut BitReader) -> Result<usize> {
        let base_lengths: [usize; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
            35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
        ];
        let extra_bits: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        let idx = (code - 257) as usize;
        if idx >= 29 {
            return Err(ImageError::DecoderError("Invalid length code".into()));
        }

        let base = base_lengths[idx];
        let extra = reader.read_bits(extra_bits[idx] as u32)? as usize;

        Ok(base + extra)
    }

    /// Decode fixed distance code.
    fn decode_fixed_distance(&self, reader: &mut BitReader) -> Result<usize> {
        let base_distances: [usize; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
            257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
            8193, 12289, 16385, 24577,
        ];
        let extra_bits: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        ];

        let code = reader.read_bits_rev(5)? as usize;
        if code >= 30 {
            return Err(ImageError::DecoderError("Invalid distance code".into()));
        }

        let base = base_distances[code];
        let extra = reader.read_bits(extra_bits[code] as u32)? as usize;

        Ok(base + extra)
    }

    /// Inflate with dynamic Huffman codes.
    fn inflate_dynamic_huffman(&self, reader: &mut BitReader, output: &mut Vec<u8>) -> Result<()> {
        let hlit = reader.read_bits(5)? as usize + 257;
        let hdist = reader.read_bits(5)? as usize + 1;
        let hclen = reader.read_bits(4)? as usize + 4;

        // Code length order
        const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

        // Read code length code lengths
        let mut cl_lengths = [0u8; 19];
        for i in 0..hclen {
            cl_lengths[CL_ORDER[i]] = reader.read_bits(3)? as u8;
        }

        // Build code length Huffman tree
        let cl_tree = self.build_huffman_tree(&cl_lengths)?;

        // Read literal/length and distance code lengths
        let mut lengths = vec![0u8; hlit + hdist];
        let mut i = 0;

        while i < lengths.len() {
            let code = self.decode_huffman(reader, &cl_tree)?;

            if code < 16 {
                lengths[i] = code as u8;
                i += 1;
            } else if code == 16 {
                let repeat = reader.read_bits(2)? as usize + 3;
                let value = if i > 0 { lengths[i - 1] } else { 0 };
                for _ in 0..repeat {
                    if i < lengths.len() {
                        lengths[i] = value;
                        i += 1;
                    }
                }
            } else if code == 17 {
                let repeat = reader.read_bits(3)? as usize + 3;
                for _ in 0..repeat {
                    if i < lengths.len() {
                        lengths[i] = 0;
                        i += 1;
                    }
                }
            } else {
                let repeat = reader.read_bits(7)? as usize + 11;
                for _ in 0..repeat {
                    if i < lengths.len() {
                        lengths[i] = 0;
                        i += 1;
                    }
                }
            }
        }

        // Build literal/length and distance trees
        let lit_tree = self.build_huffman_tree(&lengths[..hlit])?;
        let dist_tree = self.build_huffman_tree(&lengths[hlit..])?;

        // Decode compressed data
        loop {
            let code = self.decode_huffman(reader, &lit_tree)?;

            if code < 256 {
                output.push(code as u8);
            } else if code == 256 {
                break;
            } else {
                let length = self.decode_length(code, reader)?;
                let dist_code = self.decode_huffman(reader, &dist_tree)?;
                let distance = self.decode_distance(dist_code, reader)?;

                if distance > output.len() {
                    return Err(ImageError::DecoderError("Invalid distance".into()));
                }

                for _ in 0..length {
                    let idx = output.len() - distance;
                    output.push(output[idx]);
                }
            }
        }

        Ok(())
    }

    /// Build Huffman tree from code lengths.
    fn build_huffman_tree(&self, lengths: &[u8]) -> Result<HuffmanTree> {
        let max_bits = lengths.iter().copied().max().unwrap_or(0) as usize;
        if max_bits == 0 {
            return Ok(HuffmanTree { codes: Vec::new(), max_bits: 0 });
        }

        // Count codes of each length
        let mut bl_count = vec![0u32; max_bits + 1];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Calculate first code for each length
        let mut next_code = vec![0u32; max_bits + 1];
        let mut code = 0u32;
        for bits in 1..=max_bits {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes
        let mut codes = vec![(0u32, 0u8); lengths.len()];
        for (i, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[i] = (next_code[len as usize], len);
                next_code[len as usize] += 1;
            }
        }

        Ok(HuffmanTree { codes, max_bits })
    }

    /// Decode a Huffman code.
    fn decode_huffman(&self, reader: &mut BitReader, tree: &HuffmanTree) -> Result<u16> {
        if tree.max_bits == 0 {
            return Err(ImageError::DecoderError("Empty Huffman tree".into()));
        }

        let mut code = 0u32;
        for len in 1..=tree.max_bits {
            code = (code << 1) | reader.read_bits(1)?;

            for (i, &(c, l)) in tree.codes.iter().enumerate() {
                if l == len as u8 && c == code {
                    return Ok(i as u16);
                }
            }
        }

        Err(ImageError::DecoderError("Invalid Huffman code".into()))
    }

    /// Decode distance from code.
    fn decode_distance(&self, code: u16, reader: &mut BitReader) -> Result<usize> {
        let base_distances: [usize; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
            257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
            8193, 12289, 16385, 24577,
        ];
        let extra_bits: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        ];

        let idx = code as usize;
        if idx >= 30 {
            return Err(ImageError::DecoderError("Invalid distance code".into()));
        }

        let base = base_distances[idx];
        let extra = reader.read_bits(extra_bits[idx] as u32)? as usize;

        Ok(base + extra)
    }

    /// Decode non-interlaced image.
    fn decode_non_interlaced(&self, data: &[u8], info: &PngInfo) -> Result<Vec<u8>> {
        let bytes_per_pixel = self.bytes_per_pixel(info);
        let row_bytes = (info.width as usize * info.bit_depth as usize * info.color_type.channels() as usize).div_ceil(8);

        let mut output = vec![0u8; info.height as usize * row_bytes];
        let mut pos = 0;

        for y in 0..info.height as usize {
            if pos >= data.len() {
                return Err(ImageError::DecoderError("Truncated image data".into()));
            }

            let filter_type = FilterType::from_u8(data[pos])
                .ok_or_else(|| ImageError::DecoderError("Invalid filter type".into()))?;
            pos += 1;

            let row_start = y * row_bytes;
            let row_end = row_start + row_bytes;

            if pos + row_bytes > data.len() {
                return Err(ImageError::DecoderError("Truncated row data".into()));
            }

            output[row_start..row_end].copy_from_slice(&data[pos..pos + row_bytes]);
            pos += row_bytes;

            // Unfilter
            let (prev, curr) = output.split_at_mut(row_start);
            let previous = if y > 0 {
                Some(&prev[row_start - row_bytes..row_start])
            } else {
                None
            };
            unfilter_row(filter_type, &mut curr[..row_bytes], previous, bytes_per_pixel);
        }

        Ok(output)
    }

    /// Decode interlaced (Adam7) image.
    fn decode_interlaced(&self, data: &[u8], info: &PngInfo) -> Result<Vec<u8>> {
        let bytes_per_pixel = self.bytes_per_pixel(info);
        let row_bytes = (info.width as usize * info.bit_depth as usize * info.color_type.channels() as usize).div_ceil(8);

        let mut output = vec![0u8; info.height as usize * row_bytes];
        let mut pos = 0;

        for &(start_x, start_y, step_x, step_y) in &ADAM7_PASSES {

            let pass_width = (info.width as usize + step_x - 1 - start_x) / step_x;
            let pass_height = (info.height as usize + step_y - 1 - start_y) / step_y;

            if pass_width == 0 || pass_height == 0 {
                continue;
            }

            let pass_row_bytes = (pass_width * info.bit_depth as usize * info.color_type.channels() as usize).div_ceil(8);
            let mut pass_data = vec![0u8; pass_height * pass_row_bytes];
            let mut prev_row: Option<Vec<u8>> = None;

            for y in 0..pass_height {
                if pos >= data.len() {
                    return Err(ImageError::DecoderError("Truncated interlaced data".into()));
                }

                let filter_type = FilterType::from_u8(data[pos])
                    .ok_or_else(|| ImageError::DecoderError("Invalid filter type".into()))?;
                pos += 1;

                if pos + pass_row_bytes > data.len() {
                    return Err(ImageError::DecoderError("Truncated row data".into()));
                }

                let row_start = y * pass_row_bytes;
                pass_data[row_start..row_start + pass_row_bytes]
                    .copy_from_slice(&data[pos..pos + pass_row_bytes]);
                pos += pass_row_bytes;

                unfilter_row(
                    filter_type,
                    &mut pass_data[row_start..row_start + pass_row_bytes],
                    prev_row.as_deref(),
                    bytes_per_pixel,
                );

                prev_row = Some(pass_data[row_start..row_start + pass_row_bytes].to_vec());

                // Scatter pixels to output
                let out_y = start_y + y * step_y;
                for x in 0..pass_width {
                    let out_x = start_x + x * step_x;
                    let src_offset = x * bytes_per_pixel;
                    let dst_offset = out_y * row_bytes + out_x * bytes_per_pixel;

                    for b in 0..bytes_per_pixel {
                        if src_offset + b < pass_row_bytes && dst_offset + b < output.len() {
                            output[dst_offset + b] = pass_data[row_start + src_offset + b];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get bytes per pixel.
    fn bytes_per_pixel(&self, info: &PngInfo) -> usize {
        let bits = info.bit_depth as usize * info.color_type.channels() as usize;
        bits.div_ceil(8)
    }

    /// Convert raw data to Image.
    fn raw_to_image(&self, data: &[u8], info: &PngInfo) -> Result<Image> {
        let (format, output_data) = match info.color_type {
            ColorType::Grayscale => {
                if info.bit_depth == 8 {
                    (PixelFormat::Gray8, data.to_vec())
                } else if info.bit_depth < 8 {
                    // Expand to 8-bit
                    let expanded = self.expand_bits(data, info)?;
                    (PixelFormat::Gray8, expanded)
                } else {
                    // 16-bit grayscale - reduce to 8-bit
                    let reduced: Vec<u8> = data.chunks(2)
                        .map(|c| c[0])
                        .collect();
                    (PixelFormat::Gray8, reduced)
                }
            }
            ColorType::Rgb => {
                if info.bit_depth == 8 {
                    (PixelFormat::Rgb8, data.to_vec())
                } else {
                    // 16-bit RGB - reduce to 8-bit
                    let reduced: Vec<u8> = data.chunks(2)
                        .map(|c| c[0])
                        .collect();
                    (PixelFormat::Rgb8, reduced)
                }
            }
            ColorType::Indexed => {
                // Convert indexed to RGB
                let palette = self.palette.as_ref()
                    .ok_or_else(|| ImageError::DecoderError("Missing palette".into()))?;

                let expanded = if info.bit_depth < 8 {
                    self.expand_bits(data, info)?
                } else {
                    data.to_vec()
                };

                let has_alpha = self.transparency.as_ref()
                    .map(|t| !t.is_empty())
                    .unwrap_or(false);

                if has_alpha {
                    let trans = self.transparency.as_ref().unwrap();
                    let mut rgb = Vec::with_capacity(expanded.len() * 4);
                    for &idx in &expanded {
                        let i = idx as usize;
                        if i < palette.len() {
                            rgb.push(palette[i][0]);
                            rgb.push(palette[i][1]);
                            rgb.push(palette[i][2]);
                            rgb.push(if i < trans.len() { trans[i] } else { 255 });
                        } else {
                            rgb.extend_from_slice(&[0, 0, 0, 0]);
                        }
                    }
                    (PixelFormat::Rgba8, rgb)
                } else {
                    let mut rgb = Vec::with_capacity(expanded.len() * 3);
                    for &idx in &expanded {
                        let i = idx as usize;
                        if i < palette.len() {
                            rgb.push(palette[i][0]);
                            rgb.push(palette[i][1]);
                            rgb.push(palette[i][2]);
                        } else {
                            rgb.extend_from_slice(&[0, 0, 0]);
                        }
                    }
                    (PixelFormat::Rgb8, rgb)
                }
            }
            ColorType::GrayscaleAlpha => {
                if info.bit_depth == 8 {
                    (PixelFormat::GrayA8, data.to_vec())
                } else {
                    // 16-bit - reduce to 8-bit
                    let reduced: Vec<u8> = data.chunks(2)
                        .map(|c| c[0])
                        .collect();
                    (PixelFormat::GrayA8, reduced)
                }
            }
            ColorType::Rgba => {
                if info.bit_depth == 8 {
                    (PixelFormat::Rgba8, data.to_vec())
                } else {
                    // 16-bit RGBA - reduce to 8-bit
                    let reduced: Vec<u8> = data.chunks(2)
                        .map(|c| c[0])
                        .collect();
                    (PixelFormat::Rgba8, reduced)
                }
            }
        };

        Image::from_data(info.width, info.height, format, output_data)
    }

    /// Expand sub-byte samples to 8-bit.
    fn expand_bits(&self, data: &[u8], info: &PngInfo) -> Result<Vec<u8>> {
        let width = info.width as usize;
        let height = info.height as usize;
        let bit_depth = info.bit_depth as usize;
        let row_bytes = (width * bit_depth).div_ceil(8);

        let mut output = Vec::with_capacity(width * height);

        for y in 0..height {
            let row_start = y * row_bytes;
            let mut bit_pos = 0usize;

            for _ in 0..width {
                let byte_pos = row_start + bit_pos / 8;
                let bit_offset = 8 - bit_depth - (bit_pos % 8);

                let mask = (1 << bit_depth) - 1;
                let value = (data[byte_pos] >> bit_offset) & mask as u8;

                // Scale to 8-bit
                let scaled = match bit_depth {
                    1 => if value != 0 { 255 } else { 0 },
                    2 => value * 85,
                    4 => value * 17,
                    _ => value,
                };

                output.push(scaled);
                bit_pos += bit_depth;
            }
        }

        Ok(output)
    }
}

impl Default for PngDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Huffman tree structure.
struct HuffmanTree {
    codes: Vec<(u32, u8)>, // (code, length) for each symbol
    max_bits: usize,
}

/// Bit reader for DEFLATE.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, bit_pos: 0 }
    }

    fn read_bits(&mut self, n: u32) -> Result<u32> {
        let mut result = 0u32;

        for i in 0..n {
            if self.pos >= self.data.len() {
                return Err(ImageError::DecoderError("Unexpected end of data".into()));
            }

            let bit = (self.data[self.pos] >> self.bit_pos) & 1;
            result |= (bit as u32) << i;

            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.pos += 1;
            }
        }

        Ok(result)
    }

    fn read_bits_rev(&mut self, n: u32) -> Result<u32> {
        let mut result = 0u32;

        for _ in 0..n {
            if self.pos >= self.data.len() {
                return Err(ImageError::DecoderError("Unexpected end of data".into()));
            }

            let bit = (self.data[self.pos] >> self.bit_pos) & 1;
            result = (result << 1) | bit as u32;

            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.pos += 1;
            }
        }

        Ok(result)
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.pos += 1;
        }
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(ImageError::DecoderError("Unexpected end of data".into()));
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let low = self.read_byte()? as u16;
        let high = self.read_byte()? as u16;
        Ok((high << 8) | low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = PngDecoder::new();
        assert!(decoder.info().is_none());
    }

    #[test]
    fn test_invalid_signature() {
        let mut decoder = PngDecoder::new();
        let result = decoder.decode(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        // Read LSB first
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);
    }
}
