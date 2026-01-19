//! WebP encoder implementation
//!
//! Supports both lossless (VP8L) and lossy (VP8) encoding modes.

use crate::error::{Result, WebPError};
use crate::riff::{ChunkType, Vp8xFlags};
use std::io::Write;

/// WebP encoding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodingMode {
    /// Lossless encoding (VP8L)
    #[default]
    Lossless,
    /// Lossy encoding (VP8)
    Lossy,
}

/// WebP encoder configuration
#[derive(Debug, Clone)]
pub struct WebPEncoderConfig {
    /// Encoding mode (lossless or lossy)
    pub mode: EncodingMode,
    /// Quality level (0-100, only for lossy mode)
    pub quality: u8,
    /// Compression level for lossless (0-9, higher = slower + smaller)
    pub compression_level: u8,
    /// Enable alpha channel
    pub alpha: bool,
    /// Use exact mode for lossless (preserve exact RGB values)
    pub exact: bool,
}

impl Default for WebPEncoderConfig {
    fn default() -> Self {
        Self {
            mode: EncodingMode::Lossless,
            quality: 80,
            compression_level: 6,
            alpha: true,
            exact: false,
        }
    }
}

/// Encoded WebP frame
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Encoded data
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Is lossless
    pub is_lossless: bool,
    /// Has alpha
    pub has_alpha: bool,
}

/// WebP encoder
pub struct WebPEncoder {
    config: WebPEncoderConfig,
}

impl WebPEncoder {
    /// Create a new WebP encoder with default configuration
    pub fn new() -> Self {
        Self {
            config: WebPEncoderConfig::default(),
        }
    }

    /// Create a new WebP encoder with the given configuration
    pub fn with_config(config: WebPEncoderConfig) -> Self {
        Self { config }
    }

    /// Set encoding mode
    pub fn mode(mut self, mode: EncodingMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Set quality (0-100, lossy only)
    pub fn quality(mut self, quality: u8) -> Self {
        self.config.quality = quality.min(100);
        self
    }

    /// Set compression level (0-9, lossless only)
    pub fn compression_level(mut self, level: u8) -> Self {
        self.config.compression_level = level.min(9);
        self
    }

    /// Enable/disable alpha channel
    pub fn alpha(mut self, enable: bool) -> Self {
        self.config.alpha = enable;
        self
    }

    /// Encode RGBA image data to WebP
    pub fn encode_rgba(&self, data: &[u8], width: u32, height: u32) -> Result<EncodedFrame> {
        let expected_len = (width * height * 4) as usize;
        if data.len() != expected_len {
            return Err(WebPError::BufferTooSmall {
                expected: expected_len,
                actual: data.len(),
            });
        }

        // Check if image has meaningful alpha
        let has_alpha = self.config.alpha && has_non_opaque_alpha(data);

        match self.config.mode {
            EncodingMode::Lossless => {
                let encoded = encode_vp8l(data, width, height, has_alpha, self.config.compression_level)?;
                Ok(EncodedFrame {
                    data: encoded,
                    width,
                    height,
                    is_lossless: true,
                    has_alpha,
                })
            }
            EncodingMode::Lossy => {
                let encoded = encode_vp8(data, width, height, has_alpha, self.config.quality)?;
                Ok(EncodedFrame {
                    data: encoded,
                    width,
                    height,
                    is_lossless: false,
                    has_alpha,
                })
            }
        }
    }

    /// Encode RGB image data to WebP (no alpha)
    pub fn encode_rgb(&self, data: &[u8], width: u32, height: u32) -> Result<EncodedFrame> {
        let expected_len = (width * height * 3) as usize;
        if data.len() != expected_len {
            return Err(WebPError::BufferTooSmall {
                expected: expected_len,
                actual: data.len(),
            });
        }

        // Convert RGB to RGBA
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for chunk in data.chunks(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }

        let mut config = self.config.clone();
        config.alpha = false;
        let encoder = WebPEncoder::with_config(config);
        encoder.encode_rgba(&rgba, width, height)
    }

    /// Write encoded frame to RIFF container
    pub fn write_to_riff<W: Write>(&self, frame: &EncodedFrame, writer: &mut W) -> Result<()> {
        // Determine if we need VP8X extended format
        let needs_vp8x = frame.has_alpha;

        if needs_vp8x {
            write_extended_webp(writer, frame)?;
        } else {
            write_simple_webp(writer, frame)?;
        }

        Ok(())
    }
}

impl Default for WebPEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if RGBA data has non-opaque alpha values
fn has_non_opaque_alpha(data: &[u8]) -> bool {
    data.chunks(4).any(|pixel| pixel.len() >= 4 && pixel[3] != 255)
}

/// Write a simple WebP file (no VP8X header)
fn write_simple_webp<W: Write>(writer: &mut W, frame: &EncodedFrame) -> Result<()> {
    let chunk_type = if frame.is_lossless {
        ChunkType::VP8L.to_fourcc()
    } else {
        ChunkType::VP8.to_fourcc()
    };

    let chunk_size = frame.data.len() as u32;
    let file_size = 4 + 8 + chunk_size + (chunk_size % 2); // WEBP + chunk header + data + padding

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WEBP")?;

    // Image chunk
    writer.write_all(&chunk_type)?;
    writer.write_all(&chunk_size.to_le_bytes())?;
    writer.write_all(&frame.data)?;

    // Padding if needed
    if chunk_size % 2 != 0 {
        writer.write_all(&[0])?;
    }

    Ok(())
}

/// Write an extended WebP file (with VP8X header)
fn write_extended_webp<W: Write>(writer: &mut W, frame: &EncodedFrame) -> Result<()> {
    let chunk_type = if frame.is_lossless {
        ChunkType::VP8L.to_fourcc()
    } else {
        ChunkType::VP8.to_fourcc()
    };

    // VP8X chunk (10 bytes)
    let vp8x_flags = Vp8xFlags {
        icc: false,
        alpha: frame.has_alpha,
        exif: false,
        xmp: false,
        animation: false,
    };
    let flags_byte = encode_vp8x_flags(&vp8x_flags);

    let mut vp8x_data = vec![flags_byte, 0, 0, 0]; // flags + 3 reserved bytes
    // Width - 1 (24 bits)
    let w = frame.width - 1;
    vp8x_data.push((w & 0xFF) as u8);
    vp8x_data.push(((w >> 8) & 0xFF) as u8);
    vp8x_data.push(((w >> 16) & 0xFF) as u8);
    // Height - 1 (24 bits)
    let h = frame.height - 1;
    vp8x_data.push((h & 0xFF) as u8);
    vp8x_data.push(((h >> 8) & 0xFF) as u8);
    vp8x_data.push(((h >> 16) & 0xFF) as u8);

    let image_chunk_size = frame.data.len() as u32;
    let image_chunk_padded = image_chunk_size + (image_chunk_size % 2);

    // Calculate file size
    let file_size = 4  // WEBP
        + 8 + 10  // VP8X chunk
        + 8 + image_chunk_padded; // Image chunk

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WEBP")?;

    // VP8X chunk
    writer.write_all(&ChunkType::VP8X.to_fourcc())?;
    writer.write_all(&10u32.to_le_bytes())?;
    writer.write_all(&vp8x_data)?;

    // Image chunk
    writer.write_all(&chunk_type)?;
    writer.write_all(&image_chunk_size.to_le_bytes())?;
    writer.write_all(&frame.data)?;

    // Padding if needed
    if image_chunk_size % 2 != 0 {
        writer.write_all(&[0])?;
    }

    Ok(())
}

/// Encode VP8X flags to a byte
fn encode_vp8x_flags(flags: &Vp8xFlags) -> u8 {
    let mut byte = 0u8;
    if flags.icc {
        byte |= 0x20;
    }
    if flags.alpha {
        byte |= 0x10;
    }
    if flags.exif {
        byte |= 0x08;
    }
    if flags.xmp {
        byte |= 0x04;
    }
    if flags.animation {
        byte |= 0x02;
    }
    byte
}

// =============================================================================
// VP8L (Lossless) Encoder
// =============================================================================

/// VP8L signature byte
const VP8L_SIGNATURE: u8 = 0x2f;

/// Encode image as VP8L (lossless)
fn encode_vp8l(
    rgba: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    compression_level: u8,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();

    // VP8L signature
    output.push(VP8L_SIGNATURE);

    // Image size header (14 bits width-1, 14 bits height-1, 1 bit alpha, 3 bits version)
    let w = (width - 1) & 0x3FFF;
    let h = (height - 1) & 0x3FFF;
    let alpha_bit = if has_alpha { 1u32 } else { 0u32 };

    // Pack: [13:0] = width-1, [27:14] = height-1, [28] = alpha, [31:29] = version (0)
    let header = w | (h << 14) | (alpha_bit << 28);
    output.push((header & 0xFF) as u8);
    output.push(((header >> 8) & 0xFF) as u8);
    output.push(((header >> 16) & 0xFF) as u8);
    output.push(((header >> 24) & 0xFF) as u8);

    // Create bit writer for the encoded data
    let mut writer = Vp8lBitWriter::new();

    // Transform bits - we'll apply some simple transforms based on compression level
    #[allow(clippy::precedence)]
    let pixels: Vec<u32> = rgba
        .chunks(4)
        .map(|c| {
            u32::from(c[3]) << 24 | u32::from(c[0]) << 16 | u32::from(c[1]) << 8 | u32::from(c[2])
        })
        .collect();

    // Encode the image data
    if compression_level >= 5 {
        // Apply subtract green transform for better compression
        encode_vp8l_with_transforms(&mut writer, &pixels, width, height)?;
    } else {
        // Simple encoding without transforms
        encode_vp8l_simple(&mut writer, &pixels, width, height)?;
    }

    output.extend(writer.finish());
    Ok(output)
}

/// VP8L bit writer
struct Vp8lBitWriter {
    buffer: Vec<u8>,
    bits: u64,
    bit_count: u32,
}

impl Vp8lBitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bits: 0,
            bit_count: 0,
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u32) {
        self.bits |= (value as u64) << self.bit_count;
        self.bit_count += num_bits;

        while self.bit_count >= 8 {
            self.buffer.push((self.bits & 0xFF) as u8);
            self.bits >>= 8;
            self.bit_count -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.buffer.push((self.bits & 0xFF) as u8);
        }
        self.buffer
    }
}

/// Encode VP8L with transforms (subtract green)
fn encode_vp8l_with_transforms(
    writer: &mut Vp8lBitWriter,
    pixels: &[u32],
    width: u32,
    height: u32,
) -> Result<()> {
    // Write transform present bit
    writer.write_bits(1, 1); // has transform

    // Write subtract green transform
    writer.write_bits(2, 2); // SUBTRACT_GREEN transform type

    // Apply subtract green transform to pixels
    let transformed: Vec<u32> = pixels
        .iter()
        .map(|&p| {
            let a = (p >> 24) & 0xFF;
            let r = (p >> 16) & 0xFF;
            let g = (p >> 8) & 0xFF;
            let b = p & 0xFF;

            // Subtract green from red and blue
            let new_r = r.wrapping_sub(g) & 0xFF;
            let new_b = b.wrapping_sub(g) & 0xFF;

            (a << 24) | (new_r << 16) | (g << 8) | new_b
        })
        .collect();

    // No more transforms
    writer.write_bits(0, 1);

    // Encode the transformed pixels
    encode_vp8l_image_data(writer, &transformed, width, height)
}

/// Simple VP8L encoding without transforms
fn encode_vp8l_simple(
    writer: &mut Vp8lBitWriter,
    pixels: &[u32],
    width: u32,
    height: u32,
) -> Result<()> {
    // No transforms
    writer.write_bits(0, 1);

    // Encode image data
    encode_vp8l_image_data(writer, pixels, width, height)
}

/// Encode VP8L image data using prefix codes
fn encode_vp8l_image_data(
    writer: &mut Vp8lBitWriter,
    pixels: &[u32],
    width: u32,
    height: u32,
) -> Result<()> {
    // Color cache bits (0 = no cache)
    writer.write_bits(0, 1); // no color cache

    // Meta huffman (0 = single huffman code set)
    writer.write_bits(0, 1);

    // Build histogram of pixel values
    let mut green_hist = [0u32; 256];
    let mut red_hist = [0u32; 256];
    let mut blue_hist = [0u32; 256];
    let mut alpha_hist = [0u32; 256];

    for &p in pixels {
        let a = ((p >> 24) & 0xFF) as usize;
        let r = ((p >> 16) & 0xFF) as usize;
        let g = ((p >> 8) & 0xFF) as usize;
        let b = (p & 0xFF) as usize;

        alpha_hist[a] += 1;
        red_hist[r] += 1;
        green_hist[g] += 1;
        blue_hist[b] += 1;
    }

    // Build huffman codes for each channel
    // For simplicity, use fixed-length codes (8 bits per channel)
    // This is a simplified implementation; production would use optimal huffman trees

    // Write huffman code lengths
    // Using simple code: code_length_code = 0 means use simple_code_length
    write_huffman_codes(writer, &green_hist)?; // Green + length/distance
    write_huffman_codes(writer, &red_hist)?;   // Red
    write_huffman_codes(writer, &blue_hist)?;  // Blue
    write_huffman_codes(writer, &alpha_hist)?; // Alpha
    write_huffman_codes(writer, &[1; 40])?;    // Distance codes (simplified)

    // Write pixel data using the huffman codes
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let p = pixels[idx];

            let a = (p >> 24) & 0xFF;
            let r = (p >> 16) & 0xFF;
            let g = (p >> 8) & 0xFF;
            let b = p & 0xFF;

            // Write each channel (simplified: using literal values)
            writer.write_bits(g, 8);
            writer.write_bits(r, 8);
            writer.write_bits(b, 8);
            writer.write_bits(a, 8);
        }
    }

    Ok(())
}

/// Write huffman codes (simplified implementation)
fn write_huffman_codes(writer: &mut Vp8lBitWriter, histogram: &[u32]) -> Result<()> {
    // Count non-zero symbols
    let non_zero: Vec<usize> = histogram
        .iter()
        .enumerate()
        .filter(|(_, &count)| count > 0)
        .map(|(i, _)| i)
        .collect();

    if non_zero.is_empty() || non_zero.len() == 1 {
        // Use simple code with single symbol
        writer.write_bits(1, 1); // simple_code
        writer.write_bits(0, 1); // num_symbols - 1 = 0 (1 symbol)

        let symbol = non_zero.first().copied().unwrap_or(0);
        let is_first_8bits = symbol < 256;
        writer.write_bits(if is_first_8bits { 0 } else { 1 }, 1);
        if is_first_8bits {
            writer.write_bits(symbol as u32, 8);
        } else {
            // This shouldn't happen for our use case
            writer.write_bits(symbol as u32, 8);
        }
    } else if non_zero.len() == 2 {
        // Two symbols
        writer.write_bits(1, 1); // simple_code
        writer.write_bits(1, 1); // num_symbols - 1 = 1 (2 symbols)
        writer.write_bits(1, 1); // is_first_8bits (yes)
        writer.write_bits(non_zero[0] as u32, 8);
        writer.write_bits(non_zero[1] as u32, 8);
    } else {
        // Use normal huffman code (simplified: fixed 8-bit codes)
        writer.write_bits(0, 1); // not simple_code

        // Write code length code
        // num_code_lengths = 4 (for code lengths 0, 8, 17, 18)
        writer.write_bits(4, 4);

        // Code length order
        let _code_length_order = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        // Write code lengths for code length alphabet (simplified)
        // Using flat 3-bit codes for lengths
        for _ in 0..4 {
            writer.write_bits(3, 3); // 3-bit codes
        }

        // Write actual code lengths (all 8)
        let max_symbol = histogram.len().min(256);
        if max_symbol > 0 {
            // Write run of 8s
            let count = max_symbol;
            // Use repeat code
            writer.write_bits(8, 3); // length 8
            if count > 1 {
                // Use repeat codes for efficiency
                writer.write_bits(16, 5); // repeat previous
                writer.write_bits((count - 2).min(3) as u32, 2);
            }
        }
    }

    Ok(())
}

// =============================================================================
// VP8 (Lossy) Encoder
// =============================================================================

/// VP8 frame tag for keyframe
const VP8_KEYFRAME_TAG: u32 = 0;

/// Encode image as VP8 (lossy)
fn encode_vp8(
    rgba: &[u8],
    width: u32,
    height: u32,
    _has_alpha: bool,
    quality: u8,
) -> Result<Vec<u8>> {
    // Convert RGBA to YUV420
    let (y_plane, u_plane, v_plane) = rgba_to_yuv420(rgba, width, height);

    let mut output = Vec::new();

    // VP8 frame tag (3 bytes)
    // Bits 0: keyframe (0 = key)
    // Bits 1-3: version
    // Bit 4: show_frame
    // Bits 5-23: first_part_size (placeholder)
    let frame_tag = VP8_KEYFRAME_TAG | (1 << 4); // keyframe + show_frame
    output.push((frame_tag & 0xFF) as u8);
    output.push(((frame_tag >> 8) & 0xFF) as u8);
    output.push(((frame_tag >> 16) & 0xFF) as u8);

    // Keyframe header
    // Start code: 0x9d 0x01 0x2a
    output.push(0x9d);
    output.push(0x01);
    output.push(0x2a);

    // Width and height (16 bits each, with scale in upper 2 bits)
    output.push((width & 0xFF) as u8);
    output.push(((width >> 8) & 0x3F) as u8); // scale = 0
    output.push((height & 0xFF) as u8);
    output.push(((height >> 8) & 0x3F) as u8); // scale = 0

    // Encode frame header and macroblock data
    let mut bool_encoder = BoolEncoder::new();

    // Frame header
    encode_vp8_frame_header(&mut bool_encoder, width, height, quality)?;

    // Encode macroblocks
    encode_vp8_macroblocks(&mut bool_encoder, &y_plane, &u_plane, &v_plane, width, height, quality)?;

    // Finish and get encoded data
    let encoded_data = bool_encoder.finish();
    output.extend(encoded_data);

    // Update first_part_size in frame tag
    let first_part_size = output.len() - 10; // Subtract header size
    output[0] = ((first_part_size << 5) & 0xFF) as u8 | (frame_tag & 0x1F) as u8;
    output[1] = ((first_part_size >> 3) & 0xFF) as u8;
    output[2] = ((first_part_size >> 11) & 0xFF) as u8;

    Ok(output)
}

/// Convert RGBA to YUV420
fn rgba_to_yuv420(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; (w / 2) * (h / 2)];
    let mut v_plane = vec![0u8; (w / 2) * (h / 2)];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let r = rgba[idx] as i32;
            let g = rgba[idx + 1] as i32;
            let b = rgba[idx + 2] as i32;

            // BT.601 conversion
            let y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[y * w + x] = y_val.clamp(0, 255) as u8;

            // Subsample U and V
            if x % 2 == 0 && y % 2 == 0 {
                let u_idx = (y / 2) * (w / 2) + (x / 2);
                let u_val = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                let v_val = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                u_plane[u_idx] = u_val.clamp(0, 255) as u8;
                v_plane[u_idx] = v_val.clamp(0, 255) as u8;
            }
        }
    }

    (y_plane, u_plane, v_plane)
}

/// VP8 boolean encoder (arithmetic coder)
struct BoolEncoder {
    output: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    fn new() -> Self {
        Self {
            output: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: -24,
        }
    }

    fn encode_bool(&mut self, value: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);

        if value {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }

        // Renormalize
        while self.range < 128 {
            self.range <<= 1;
            self.bottom <<= 1;

            self.bit_count += 1;
            if self.bit_count >= 0 {
                self.output.push((self.bottom >> 24) as u8);
                self.bottom &= 0xFFFFFF;
            }
        }
    }

    fn encode_value(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.encode_bool((value >> i) & 1 != 0, 128);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Flush remaining bits
        for _ in 0..32 {
            self.encode_bool(false, 128);
        }
        self.output
    }
}

/// Encode VP8 frame header
fn encode_vp8_frame_header(
    encoder: &mut BoolEncoder,
    _width: u32,
    _height: u32,
    _quality: u8,
) -> Result<()> {
    // Color space (0 = YUV)
    encoder.encode_bool(false, 128);

    // Clamping type
    encoder.encode_bool(false, 128);

    // Segmentation enabled
    encoder.encode_bool(false, 128);

    // Filter type (0 = normal)
    encoder.encode_bool(false, 128);

    // Filter level (6 bits)
    encoder.encode_value(32, 6);

    // Sharpness (3 bits)
    encoder.encode_value(0, 3);

    // Mode lf adjustments enabled
    encoder.encode_bool(false, 128);

    // Number of partitions (0 = 1 partition)
    encoder.encode_value(0, 2);

    // Quantizer base value (7 bits)
    encoder.encode_value(32, 7);

    // Quantizer deltas (all zero)
    for _ in 0..5 {
        encoder.encode_bool(false, 128);
    }

    // Refresh entropy probs
    encoder.encode_bool(false, 128);

    // Refresh golden frame
    encoder.encode_bool(false, 128);

    // Refresh alternate frame
    encoder.encode_bool(false, 128);

    // Refresh last
    encoder.encode_bool(true, 128);

    Ok(())
}

/// Encode VP8 macroblocks
fn encode_vp8_macroblocks(
    encoder: &mut BoolEncoder,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<()> {
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);

    // Quantization based on quality
    let quant = 128 - (quality as i32).clamp(0, 100);

    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            // Encode macroblock mode (all I4x4 for simplicity)
            encoder.encode_value(0, 1); // Y mode = DC_PRED

            // Encode Y blocks (16x16 split into 4x4)
            for block_y in 0..4 {
                for block_x in 0..4 {
                    let px = (mb_x * 16 + block_x * 4) as usize;
                    let py = (mb_y * 16 + block_y * 4) as usize;
                    encode_4x4_block(encoder, y_plane, px, py, width as usize, quant)?;
                }
            }

            // Encode U block (8x8)
            let ux = (mb_x * 8) as usize;
            let uy = (mb_y * 8) as usize;
            encode_chroma_block(encoder, u_plane, ux, uy, (width / 2) as usize, quant)?;

            // Encode V block (8x8)
            encode_chroma_block(encoder, v_plane, ux, uy, (width / 2) as usize, quant)?;
        }
    }

    Ok(())
}

/// Encode a 4x4 Y block
fn encode_4x4_block(
    encoder: &mut BoolEncoder,
    plane: &[u8],
    x: usize,
    y: usize,
    stride: usize,
    quant: i32,
) -> Result<()> {
    // Extract 4x4 block
    let mut block = [0i32; 16];
    for by in 0..4 {
        for bx in 0..4 {
            let px = (x + bx).min(stride - 1);
            let py = y + by;
            let idx = py * stride + px;
            if idx < plane.len() {
                block[by * 4 + bx] = plane[idx] as i32;
            }
        }
    }

    // Apply 4x4 DCT
    let coeffs = dct_4x4(&block);

    // Quantize
    let quantized: Vec<i32> = coeffs.iter().map(|&c| c / quant.max(1)).collect();

    // Encode DC coefficient
    let dc = quantized[0];
    encode_signed_value(encoder, dc, 11);

    // Encode AC coefficients (simplified: RLE zeros)
    let mut last_nonzero = 0;
    for (i, &c) in quantized.iter().enumerate().skip(1) {
        if c != 0 {
            last_nonzero = i;
        }
    }

    for (i, &c) in quantized.iter().enumerate().skip(1) {
        if i > last_nonzero {
            break;
        }
        encode_signed_value(encoder, c, 8);
    }

    // EOB
    encoder.encode_bool(true, 128);

    Ok(())
}

/// Encode a chroma block
fn encode_chroma_block(
    encoder: &mut BoolEncoder,
    plane: &[u8],
    x: usize,
    y: usize,
    stride: usize,
    quant: i32,
) -> Result<()> {
    // Encode 4 4x4 blocks for 8x8 chroma
    for block_y in 0..2 {
        for block_x in 0..2 {
            let bx = x + block_x * 4;
            let by = y + block_y * 4;
            encode_4x4_block(encoder, plane, bx, by, stride, quant)?;
        }
    }
    Ok(())
}

/// Encode a signed value
fn encode_signed_value(encoder: &mut BoolEncoder, value: i32, bits: u32) {
    if value == 0 {
        encoder.encode_bool(false, 128);
    } else {
        encoder.encode_bool(true, 128);
        encoder.encode_bool(value < 0, 128);
        encoder.encode_value(value.unsigned_abs(), bits);
    }
}

/// 4x4 DCT
fn dct_4x4(block: &[i32; 16]) -> [i32; 16] {
    let mut temp = [0i32; 16];
    let mut output = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let a = block[i * 4] + block[i * 4 + 3];
        let b = block[i * 4 + 1] + block[i * 4 + 2];
        let c = block[i * 4 + 1] - block[i * 4 + 2];
        let d = block[i * 4] - block[i * 4 + 3];

        temp[i * 4] = a + b;
        temp[i * 4 + 2] = a - b;
        temp[i * 4 + 1] = (d * 2217 + c * 5352 + 14500) >> 12;
        temp[i * 4 + 3] = (d * 5352 - c * 2217 + 7500) >> 12;
    }

    // Vertical pass
    for i in 0..4 {
        let a = temp[i] + temp[12 + i];
        let b = temp[4 + i] + temp[8 + i];
        let c = temp[4 + i] - temp[8 + i];
        let d = temp[i] - temp[12 + i];

        output[i] = (a + b + 7) >> 4;
        output[8 + i] = (a - b + 7) >> 4;
        output[4 + i] = ((d * 2217 + c * 5352 + 12000) >> 16) + if d != 0 { 1 } else { 0 };
        output[12 + i] = (d * 5352 - c * 2217 + 51000) >> 16;
    }

    output
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Encode RGBA data to WebP bytes (lossless)
pub fn encode_webp_lossless(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let encoder = WebPEncoder::new().mode(EncodingMode::Lossless);
    let frame = encoder.encode_rgba(data, width, height)?;
    let mut output = Vec::new();
    encoder.write_to_riff(&frame, &mut output)?;
    Ok(output)
}

/// Encode RGBA data to WebP bytes (lossy)
pub fn encode_webp_lossy(data: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    let encoder = WebPEncoder::new()
        .mode(EncodingMode::Lossy)
        .quality(quality);
    let frame = encoder.encode_rgba(data, width, height)?;
    let mut output = Vec::new();
    encoder.write_to_riff(&frame, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = WebPEncoder::new();
        assert_eq!(encoder.config.mode, EncodingMode::Lossless);
        assert_eq!(encoder.config.quality, 80);
    }

    #[test]
    fn test_encoder_config() {
        let encoder = WebPEncoder::new()
            .mode(EncodingMode::Lossy)
            .quality(90)
            .compression_level(9);

        assert_eq!(encoder.config.mode, EncodingMode::Lossy);
        assert_eq!(encoder.config.quality, 90);
        assert_eq!(encoder.config.compression_level, 9);
    }

    #[test]
    fn test_rgba_to_yuv420() {
        let rgba = vec![255u8; 4 * 16 * 16]; // White 16x16 image
        let (y, u, v) = rgba_to_yuv420(&rgba, 16, 16);

        assert_eq!(y.len(), 16 * 16);
        assert_eq!(u.len(), 8 * 8);
        assert_eq!(v.len(), 8 * 8);

        // White should have high Y value
        assert!(y[0] > 200);
    }

    #[test]
    fn test_dct_4x4() {
        let block = [
            128, 128, 128, 128,
            128, 128, 128, 128,
            128, 128, 128, 128,
            128, 128, 128, 128,
        ];
        let coeffs = dct_4x4(&block);

        // DC coefficient should be proportional to average
        assert!(coeffs[0] > 0);
        // AC coefficients should be near zero for uniform block
        for &c in coeffs.iter().skip(1) {
            assert!(c.abs() < 10);
        }
    }

    #[test]
    fn test_encode_small_image_lossless() {
        // 4x4 red image
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            rgba.extend_from_slice(&[255, 0, 0, 255]); // Red, opaque
        }

        let encoder = WebPEncoder::new().mode(EncodingMode::Lossless);
        let frame = encoder.encode_rgba(&rgba, 4, 4);
        assert!(frame.is_ok());

        let frame = frame.unwrap();
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert!(frame.is_lossless);
    }

    #[test]
    fn test_encode_small_image_lossy() {
        // 16x16 gradient image
        let mut rgba = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16 {
            for x in 0..16 {
                rgba.push((x * 16) as u8);
                rgba.push((y * 16) as u8);
                rgba.push(128);
                rgba.push(255);
            }
        }

        let encoder = WebPEncoder::new()
            .mode(EncodingMode::Lossy)
            .quality(80);
        let frame = encoder.encode_rgba(&rgba, 16, 16);
        assert!(frame.is_ok());

        let frame = frame.unwrap();
        assert_eq!(frame.width, 16);
        assert_eq!(frame.height, 16);
        assert!(!frame.is_lossless);
    }

    #[test]
    fn test_write_simple_webp() {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            rgba.extend_from_slice(&[0, 128, 255, 255]);
        }

        let encoder = WebPEncoder::new();
        let frame = encoder.encode_rgba(&rgba, 4, 4).unwrap();

        let mut output = Vec::new();
        encoder.write_to_riff(&frame, &mut output).unwrap();

        // Check RIFF header
        assert_eq!(&output[0..4], b"RIFF");
        assert_eq!(&output[8..12], b"WEBP");
    }

    #[test]
    fn test_has_non_opaque_alpha() {
        let opaque = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        assert!(!has_non_opaque_alpha(&opaque));

        let transparent = vec![255u8, 0, 0, 128, 0, 255, 0, 255];
        assert!(has_non_opaque_alpha(&transparent));
    }

    #[test]
    fn test_vp8x_flags_encoding() {
        let flags = Vp8xFlags {
            icc: true,
            alpha: true,
            exif: false,
            xmp: false,
            animation: false,
        };

        let byte = encode_vp8x_flags(&flags);
        assert_eq!(byte & 0x20, 0x20); // ICC
        assert_eq!(byte & 0x10, 0x10); // Alpha
        assert_eq!(byte & 0x08, 0x00); // No EXIF
    }

    #[test]
    fn test_bool_encoder() {
        let mut encoder = BoolEncoder::new();
        encoder.encode_bool(true, 128);
        encoder.encode_bool(false, 128);
        encoder.encode_value(42, 8);
        let output = encoder.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_encode_convenience_functions() {
        let rgba = vec![128u8; 8 * 8 * 4];

        let lossless = encode_webp_lossless(&rgba, 8, 8);
        assert!(lossless.is_ok());

        let lossy = encode_webp_lossy(&rgba, 8, 8, 80);
        assert!(lossy.is_ok());
    }
}
