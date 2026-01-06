//! GIF decoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};
use super::{
    GifFrame, LogicalScreenDescriptor, DisposalMethod,
    GIF87A_SIGNATURE, GIF89A_SIGNATURE,
    EXTENSION_INTRODUCER, IMAGE_SEPARATOR, TRAILER,
    GRAPHIC_CONTROL_LABEL, APPLICATION_LABEL, COMMENT_LABEL,
    MIN_LZW_CODE_SIZE, MAX_LZW_CODE_SIZE,
};

/// GIF decoder.
pub struct GifDecoder {
    /// Logical screen descriptor.
    screen_desc: Option<LogicalScreenDescriptor>,
    /// Global color table.
    global_palette: Option<Vec<[u8; 3]>>,
    /// Decoded frames.
    frames: Vec<GifFrame>,
    /// Loop count (0 = infinite).
    loop_count: u16,
}

impl GifDecoder {
    /// Create a new GIF decoder.
    pub fn new() -> Self {
        Self {
            screen_desc: None,
            global_palette: None,
            frames: Vec::new(),
            loop_count: 0,
        }
    }

    /// Decode a GIF from bytes.
    pub fn decode(&mut self, data: &[u8]) -> Result<Image> {
        self.parse(data)?;

        if self.frames.is_empty() {
            return Err(ImageError::InvalidData("No frames found".into()));
        }

        let screen = self.screen_desc.as_ref().unwrap();
        let frame = &self.frames[0];

        // Convert indexed to RGBA
        let palette = frame.local_palette.as_ref()
            .or(self.global_palette.as_ref())
            .ok_or_else(|| ImageError::InvalidData("No color table".into()))?;

        let mut rgba = vec![0u8; screen.width as usize * screen.height as usize * 4];

        for y in 0..frame.height.min(screen.height as u32) {
            for x in 0..frame.width.min(screen.width as u32) {
                let src_idx = (y * frame.width + x) as usize;
                let dst_x = frame.x_offset as u32 + x;
                let dst_y = frame.y_offset as u32 + y;

                if dst_x >= screen.width as u32 || dst_y >= screen.height as u32 {
                    continue;
                }

                let dst_idx = (dst_y * screen.width as u32 + dst_x) as usize * 4;
                let color_idx = frame.data[src_idx] as usize;

                if frame.transparent_index == Some(color_idx as u8) {
                    // Transparent pixel
                    rgba[dst_idx..dst_idx + 4].copy_from_slice(&[0, 0, 0, 0]);
                } else if color_idx < palette.len() {
                    let color = palette[color_idx];
                    rgba[dst_idx] = color[0];
                    rgba[dst_idx + 1] = color[1];
                    rgba[dst_idx + 2] = color[2];
                    rgba[dst_idx + 3] = 255;
                }
            }
        }

        Image::from_data(screen.width as u32, screen.height as u32, PixelFormat::Rgba8, rgba)
    }

    /// Decode all frames (for animation).
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<GifFrame>> {
        self.parse(data)?;
        Ok(std::mem::take(&mut self.frames))
    }

    /// Get logical screen descriptor.
    pub fn screen_descriptor(&self) -> Option<&LogicalScreenDescriptor> {
        self.screen_desc.as_ref()
    }

    /// Get global palette.
    pub fn global_palette(&self) -> Option<&[[u8; 3]]> {
        self.global_palette.as_deref()
    }

    /// Get loop count.
    pub fn loop_count(&self) -> u16 {
        self.loop_count
    }

    /// Parse GIF data.
    fn parse(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 13 {
            return Err(ImageError::TruncatedData { expected: 13, actual: data.len() });
        }

        // Check signature
        let sig = &data[0..6];
        if sig != GIF87A_SIGNATURE && sig != GIF89A_SIGNATURE {
            return Err(ImageError::InvalidHeader(format!(
                "Invalid GIF signature: {:?}", sig
            )));
        }

        // Parse logical screen descriptor
        let screen = LogicalScreenDescriptor {
            width: u16::from_le_bytes([data[6], data[7]]),
            height: u16::from_le_bytes([data[8], data[9]]),
            has_global_color_table: (data[10] & 0x80) != 0,
            color_resolution: ((data[10] >> 4) & 0x07) + 1,
            sorted: (data[10] & 0x08) != 0,
            global_color_table_size: data[10] & 0x07,
            background_color_index: data[11],
            pixel_aspect_ratio: data[12],
        };

        let mut pos = 13;

        // Read global color table
        if screen.has_global_color_table {
            let table_size = 3 * (1 << (screen.global_color_table_size + 1));
            if pos + table_size > data.len() {
                return Err(ImageError::TruncatedData {
                    expected: pos + table_size,
                    actual: data.len(),
                });
            }

            let mut palette = Vec::with_capacity(table_size / 3);
            for i in (0..table_size).step_by(3) {
                palette.push([
                    data[pos + i],
                    data[pos + i + 1],
                    data[pos + i + 2],
                ]);
            }
            self.global_palette = Some(palette);
            pos += table_size;
        }

        self.screen_desc = Some(screen);

        // Parse blocks
        let mut current_gce: Option<GraphicControlExtension> = None;

        while pos < data.len() {
            match data[pos] {
                EXTENSION_INTRODUCER => {
                    pos += 1;
                    if pos >= data.len() {
                        break;
                    }

                    match data[pos] {
                        GRAPHIC_CONTROL_LABEL => {
                            pos += 1;
                            current_gce = Some(self.parse_graphic_control(&data[pos..])?);
                            pos += 6; // Block size (1) + data (4) + terminator (1)
                        }
                        APPLICATION_LABEL => {
                            pos += 1;
                            pos = self.parse_application_extension(data, pos)?;
                        }
                        COMMENT_LABEL => {
                            pos += 1;
                            pos = skip_sub_blocks(data, pos)?;
                        }
                        _ => {
                            pos += 1;
                            pos = skip_sub_blocks(data, pos)?;
                        }
                    }
                }
                IMAGE_SEPARATOR => {
                    pos += 1;
                    let (frame, new_pos) = self.parse_image(data, pos, current_gce.take())?;
                    self.frames.push(frame);
                    pos = new_pos;
                }
                TRAILER => {
                    break;
                }
                _ => {
                    // Unknown block, try to skip
                    pos += 1;
                }
            }
        }

        Ok(())
    }

    /// Parse graphic control extension.
    fn parse_graphic_control(&self, data: &[u8]) -> Result<GraphicControlExtension> {
        if data.len() < 5 {
            return Err(ImageError::TruncatedData { expected: 5, actual: data.len() });
        }

        let _block_size = data[0]; // Should be 4
        let flags = data[1];
        let delay = u16::from_le_bytes([data[2], data[3]]);
        let transparent_index = data[4];

        Ok(GraphicControlExtension {
            disposal: DisposalMethod::from_byte(flags),
            user_input: (flags & 0x02) != 0,
            has_transparent: (flags & 0x01) != 0,
            delay,
            transparent_index,
        })
    }

    /// Parse application extension.
    fn parse_application_extension(&mut self, data: &[u8], mut pos: usize) -> Result<usize> {
        if pos + 12 > data.len() {
            return skip_sub_blocks(data, pos);
        }

        let block_size = data[pos];
        if block_size != 11 {
            return skip_sub_blocks(data, pos);
        }

        pos += 1;
        let app_id = &data[pos..pos + 8];
        let auth_code = &data[pos + 8..pos + 11];
        pos += 11;

        // Check for NETSCAPE extension (loop count)
        if app_id == b"NETSCAPE" && auth_code == b"2.0" {
            if pos + 3 < data.len() && data[pos] == 3 {
                pos += 1; // Sub-block size
                let _index = data[pos];
                pos += 1;
                self.loop_count = u16::from_le_bytes([data[pos], data[pos + 1]]);
                pos += 2;
            }
        }

        skip_sub_blocks(data, pos)
    }

    /// Parse image descriptor and data.
    fn parse_image(
        &self,
        data: &[u8],
        mut pos: usize,
        gce: Option<GraphicControlExtension>,
    ) -> Result<(GifFrame, usize)> {
        if pos + 9 > data.len() {
            return Err(ImageError::TruncatedData {
                expected: pos + 9,
                actual: data.len(),
            });
        }

        // Image descriptor
        let x_offset = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let y_offset = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
        let width = u16::from_le_bytes([data[pos + 4], data[pos + 5]]);
        let height = u16::from_le_bytes([data[pos + 6], data[pos + 7]]);
        let flags = data[pos + 8];
        pos += 9;

        let has_local_table = (flags & 0x80) != 0;
        let interlaced = (flags & 0x40) != 0;
        let local_table_size = flags & 0x07;

        // Read local color table
        let local_palette = if has_local_table {
            let table_size = 3 * (1 << (local_table_size + 1));
            if pos + table_size > data.len() {
                return Err(ImageError::TruncatedData {
                    expected: pos + table_size,
                    actual: data.len(),
                });
            }

            let mut palette = Vec::with_capacity(table_size / 3);
            for i in (0..table_size).step_by(3) {
                palette.push([
                    data[pos + i],
                    data[pos + i + 1],
                    data[pos + i + 2],
                ]);
            }
            pos += table_size;
            Some(palette)
        } else {
            None
        };

        // LZW minimum code size
        if pos >= data.len() {
            return Err(ImageError::TruncatedData {
                expected: pos + 1,
                actual: data.len(),
            });
        }

        let min_code_size = data[pos];
        pos += 1;

        if min_code_size < MIN_LZW_CODE_SIZE || min_code_size > MAX_LZW_CODE_SIZE {
            return Err(ImageError::InvalidData(format!(
                "Invalid LZW minimum code size: {}",
                min_code_size
            )));
        }

        // Collect compressed data from sub-blocks
        let (compressed, new_pos) = collect_sub_blocks(data, pos)?;
        pos = new_pos;

        // Decompress LZW data
        let decompressed = lzw_decode(&compressed, min_code_size, width as usize * height as usize)?;

        // Handle interlacing
        let frame_data = if interlaced {
            deinterlace(&decompressed, width as usize, height as usize)
        } else {
            decompressed
        };

        let mut frame = GifFrame {
            data: frame_data,
            width: width as u32,
            height: height as u32,
            x_offset,
            y_offset,
            delay: gce.as_ref().map(|g| g.delay).unwrap_or(10),
            disposal: gce.as_ref().map(|g| g.disposal).unwrap_or_default(),
            transparent_index: gce.and_then(|g| {
                if g.has_transparent { Some(g.transparent_index) } else { None }
            }),
            local_palette,
            interlaced,
        };

        // Ensure minimum delay
        if frame.delay == 0 {
            frame.delay = 10;
        }

        Ok((frame, pos))
    }
}

impl Default for GifDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Graphic control extension data.
#[derive(Debug, Clone)]
struct GraphicControlExtension {
    disposal: DisposalMethod,
    user_input: bool,
    has_transparent: bool,
    delay: u16,
    transparent_index: u8,
}

/// Skip sub-blocks.
fn skip_sub_blocks(data: &[u8], mut pos: usize) -> Result<usize> {
    while pos < data.len() {
        let size = data[pos] as usize;
        pos += 1;
        if size == 0 {
            break;
        }
        pos += size;
    }
    Ok(pos)
}

/// Collect data from sub-blocks.
fn collect_sub_blocks(data: &[u8], mut pos: usize) -> Result<(Vec<u8>, usize)> {
    let mut collected = Vec::new();

    while pos < data.len() {
        let size = data[pos] as usize;
        pos += 1;
        if size == 0 {
            break;
        }
        if pos + size > data.len() {
            return Err(ImageError::TruncatedData {
                expected: pos + size,
                actual: data.len(),
            });
        }
        collected.extend_from_slice(&data[pos..pos + size]);
        pos += size;
    }

    Ok((collected, pos))
}

/// LZW decoder for GIF.
fn lzw_decode(data: &[u8], min_code_size: u8, expected_size: usize) -> Result<Vec<u8>> {
    let clear_code = 1u16 << min_code_size;
    let eoi_code = clear_code + 1;

    let mut output = Vec::with_capacity(expected_size);
    let mut table: Vec<Vec<u8>> = Vec::with_capacity(4096);
    let mut code_size = min_code_size + 1;
    let mut code_mask = (1u16 << code_size) - 1;

    // Initialize table
    fn init_table(table: &mut Vec<Vec<u8>>, clear_code: u16) {
        table.clear();
        for i in 0..=clear_code + 1 {
            if i < clear_code {
                table.push(vec![i as u8]);
            } else {
                table.push(Vec::new());
            }
        }
    }

    init_table(&mut table, clear_code);

    let mut bit_pos = 0usize;
    let mut prev_code: Option<u16> = None;

    loop {
        // Read next code
        if bit_pos + code_size as usize > data.len() * 8 {
            break;
        }

        let byte_pos = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let mut code = 0u16;
        let mut bits_read = 0u32;

        while bits_read < code_size as u32 {
            let byte_idx = byte_pos + ((bit_offset as u32 + bits_read) / 8) as usize;
            if byte_idx >= data.len() {
                break;
            }

            let bit_idx = (bit_offset as u32 + bits_read) % 8;
            let bits_from_byte = (8 - bit_idx).min(code_size as u32 - bits_read);
            let mask = ((1u16 << bits_from_byte) - 1) as u8;
            let value = (data[byte_idx] >> bit_idx) & mask;
            code |= (value as u16) << bits_read;
            bits_read += bits_from_byte;
        }

        code &= code_mask;
        bit_pos += code_size as usize;

        if code == clear_code {
            init_table(&mut table, clear_code);
            code_size = min_code_size + 1;
            code_mask = (1u16 << code_size) - 1;
            prev_code = None;
            continue;
        }

        if code == eoi_code {
            break;
        }

        let entry = if (code as usize) < table.len() {
            table[code as usize].clone()
        } else if code as usize == table.len() {
            // Special case: code not yet in table
            if let Some(prev) = prev_code {
                let mut entry = table[prev as usize].clone();
                entry.push(entry[0]);
                entry
            } else {
                return Err(ImageError::CorruptedData("Invalid LZW code".into()));
            }
        } else {
            return Err(ImageError::CorruptedData(format!(
                "LZW code {} out of range (table size: {})",
                code,
                table.len()
            )));
        };

        output.extend_from_slice(&entry);

        // Add new entry to table
        if let Some(prev) = prev_code {
            if table.len() < 4096 {
                let mut new_entry = table[prev as usize].clone();
                new_entry.push(entry[0]);
                table.push(new_entry);

                // Increase code size if needed
                if table.len() == (1 << code_size) as usize && code_size < 12 {
                    code_size += 1;
                    code_mask = (1u16 << code_size) - 1;
                }
            }
        }

        prev_code = Some(code);

        if output.len() >= expected_size {
            break;
        }
    }

    // Pad or truncate to expected size
    output.resize(expected_size, 0);
    Ok(output)
}

/// Deinterlace GIF image data.
fn deinterlace(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; width * height];
    let mut src_row = 0;

    // Pass 1: Every 8th row, starting at row 0
    for y in (0..height).step_by(8) {
        if src_row * width + width <= data.len() {
            output[y * width..(y + 1) * width]
                .copy_from_slice(&data[src_row * width..(src_row + 1) * width]);
        }
        src_row += 1;
    }

    // Pass 2: Every 8th row, starting at row 4
    for y in (4..height).step_by(8) {
        if src_row * width + width <= data.len() {
            output[y * width..(y + 1) * width]
                .copy_from_slice(&data[src_row * width..(src_row + 1) * width]);
        }
        src_row += 1;
    }

    // Pass 3: Every 4th row, starting at row 2
    for y in (2..height).step_by(4) {
        if src_row * width + width <= data.len() {
            output[y * width..(y + 1) * width]
                .copy_from_slice(&data[src_row * width..(src_row + 1) * width]);
        }
        src_row += 1;
    }

    // Pass 4: Every 2nd row, starting at row 1
    for y in (1..height).step_by(2) {
        if src_row * width + width <= data.len() {
            output[y * width..(y + 1) * width]
                .copy_from_slice(&data[src_row * width..(src_row + 1) * width]);
        }
        src_row += 1;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = GifDecoder::new();
        assert!(decoder.screen_descriptor().is_none());
        assert!(decoder.global_palette().is_none());
    }

    #[test]
    fn test_invalid_signature() {
        let mut decoder = GifDecoder::new();
        // Need at least 13 bytes (6 sig + 7 LSD) for the parser to check signature
        let result = decoder.decode(b"NOTGIF0000000");
        assert!(matches!(result, Err(ImageError::InvalidHeader(_))));
    }

    #[test]
    fn test_truncated_data() {
        let mut decoder = GifDecoder::new();
        let result = decoder.decode(b"GIF89a");
        assert!(matches!(result, Err(ImageError::TruncatedData { .. })));
    }

    #[test]
    fn test_lzw_decode_simple() {
        // Simple LZW-compressed data for a 4-pixel image
        let data = vec![0x04, 0x01, 0x00]; // Simplified
        let result = lzw_decode(&data, 2, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deinterlace() {
        // 4x8 image for testing interlace
        let width = 4;
        let height = 8;
        let mut interlaced = Vec::new();

        // Interlaced order: rows 0,4,2,6,1,3,5,7 -> mapped to 0-7
        // Pass 1 (rows 0): row 0
        interlaced.extend(vec![0u8; width]);
        // Pass 2 (rows 4): row 1
        interlaced.extend(vec![4u8; width]);
        // Pass 3 (rows 2,6): rows 2,3
        interlaced.extend(vec![2u8; width]);
        interlaced.extend(vec![6u8; width]);
        // Pass 4 (rows 1,3,5,7): rows 4,5,6,7
        interlaced.extend(vec![1u8; width]);
        interlaced.extend(vec![3u8; width]);
        interlaced.extend(vec![5u8; width]);
        interlaced.extend(vec![7u8; width]);

        let result = deinterlace(&interlaced, width, height);

        // Check that rows are in correct order
        for y in 0..height {
            assert_eq!(result[y * width], y as u8);
        }
    }

    #[test]
    fn test_graphic_control_extension() {
        let decoder = GifDecoder::new();
        let data = [4, 0x0C, 0x0A, 0x00, 0x05, 0x00]; // disposal=3, delay=10, transparent=5
        let gce = decoder.parse_graphic_control(&data).unwrap();

        assert_eq!(gce.disposal, DisposalMethod::RestorePrevious);
        assert_eq!(gce.delay, 10);
        assert_eq!(gce.transparent_index, 5);
    }
}
