//! GIF encoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};
use super::{
    GifFrame, DisposalMethod,
    GIF89A_SIGNATURE, EXTENSION_INTRODUCER, IMAGE_SEPARATOR, TRAILER,
    GRAPHIC_CONTROL_LABEL, APPLICATION_LABEL,
};

/// GIF encoder configuration.
#[derive(Debug, Clone)]
pub struct GifConfig {
    /// Maximum colors in palette (2-256).
    pub max_colors: u16,
    /// Loop count (0 = infinite, 1+ = specific count).
    pub loop_count: u16,
    /// Enable dithering for color reduction.
    pub dithering: bool,
    /// Default frame delay (centiseconds).
    pub default_delay: u16,
    /// Enable interlacing.
    pub interlace: bool,
}

impl Default for GifConfig {
    fn default() -> Self {
        Self {
            max_colors: 256,
            loop_count: 0,
            dithering: false,
            default_delay: 10,
            interlace: false,
        }
    }
}

/// GIF encoder.
pub struct GifEncoder {
    config: GifConfig,
}

impl GifEncoder {
    /// Create a new GIF encoder with default configuration.
    pub fn new() -> Self {
        Self {
            config: GifConfig::default(),
        }
    }

    /// Create a new GIF encoder with custom configuration.
    pub fn with_config(config: GifConfig) -> Self {
        Self { config }
    }

    /// Set maximum colors.
    pub fn max_colors(mut self, colors: u16) -> Self {
        self.config.max_colors = colors.clamp(2, 256);
        self
    }

    /// Set loop count.
    pub fn loop_count(mut self, count: u16) -> Self {
        self.config.loop_count = count;
        self
    }

    /// Enable/disable dithering.
    pub fn dithering(mut self, enable: bool) -> Self {
        self.config.dithering = enable;
        self
    }

    /// Set default frame delay.
    pub fn default_delay(mut self, delay: u16) -> Self {
        self.config.default_delay = delay.max(1);
        self
    }

    /// Encode a single image to GIF.
    pub fn encode(&self, image: &Image) -> Result<Vec<u8>> {
        // Convert to RGBA if needed
        let rgba = if image.format() != PixelFormat::Rgba8 {
            image.convert(PixelFormat::Rgba8)?
        } else {
            image.clone()
        };

        // Quantize to palette
        let (palette, indexed) = self.quantize_image(&rgba)?;

        // Check for transparency
        let transparent_index = if image.format().has_alpha() {
            find_transparent_index(&rgba, &indexed, &palette)
        } else {
            None
        };

        let frame = GifFrame {
            data: indexed,
            width: image.width(),
            height: image.height(),
            x_offset: 0,
            y_offset: 0,
            delay: self.config.default_delay,
            disposal: DisposalMethod::None,
            transparent_index,
            local_palette: None,
            interlaced: self.config.interlace,
        };

        self.encode_frames(image.width(), image.height(), &palette, &[frame])
    }

    /// Encode multiple frames as animated GIF.
    pub fn encode_animation(&self, frames: &[GifFrame], width: u32, height: u32) -> Result<Vec<u8>> {
        if frames.is_empty() {
            return Err(ImageError::InvalidData("No frames to encode".into()));
        }

        // Use first frame's palette as global, or generate one
        let global_palette = if let Some(ref palette) = frames[0].local_palette {
            palette.clone()
        } else {
            // Generate default grayscale palette
            (0..256).map(|i| [i as u8, i as u8, i as u8]).collect()
        };

        self.encode_frames(width, height, &global_palette, frames)
    }

    /// Encode frames with a global palette.
    fn encode_frames(
        &self,
        width: u32,
        height: u32,
        global_palette: &[[u8; 3]],
        frames: &[GifFrame],
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        // Signature
        output.extend_from_slice(GIF89A_SIGNATURE);

        // Logical screen descriptor
        let color_table_size = palette_size_flag(global_palette.len());
        let packed = 0x80 // Global color table flag
            | ((color_table_size & 0x07) << 4) // Color resolution
            | (color_table_size & 0x07); // Global color table size

        output.extend_from_slice(&(width as u16).to_le_bytes());
        output.extend_from_slice(&(height as u16).to_le_bytes());
        output.push(packed);
        output.push(0); // Background color index
        output.push(0); // Pixel aspect ratio

        // Global color table
        write_color_table(&mut output, global_palette, color_table_size);

        // NETSCAPE extension for animation (if multiple frames)
        if frames.len() > 1 {
            write_netscape_extension(&mut output, self.config.loop_count);
        }

        // Encode each frame
        for frame in frames {
            // Graphic control extension
            write_graphic_control(&mut output, frame);

            // Image descriptor
            write_image_descriptor(&mut output, frame, self.config.interlace);

            // Local color table (if different from global)
            if let Some(ref local_palette) = frame.local_palette {
                let local_size = palette_size_flag(local_palette.len());
                write_color_table(&mut output, local_palette, local_size);
            }

            // Image data
            let data = if self.config.interlace && !frame.interlaced {
                interlace(&frame.data, frame.width as usize, frame.height as usize)
            } else {
                frame.data.clone()
            };

            let min_code_size = calculate_min_code_size(global_palette.len());
            let compressed = lzw_encode(&data, min_code_size)?;

            output.push(min_code_size);
            write_sub_blocks(&mut output, &compressed);
        }

        // Trailer
        output.push(TRAILER);

        Ok(output)
    }

    /// Quantize image to indexed palette.
    fn quantize_image(&self, image: &Image) -> Result<(Vec<[u8; 3]>, Vec<u8>)> {
        let max_colors = self.config.max_colors.min(256) as usize;
        let width = image.width() as usize;
        let height = image.height() as usize;
        let data = image.data();

        // Build histogram of colors
        let mut color_counts: std::collections::HashMap<[u8; 3], u32> =
            std::collections::HashMap::new();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let rgb = [data[idx], data[idx + 1], data[idx + 2]];
                *color_counts.entry(rgb).or_insert(0) += 1;
            }
        }

        // If we have few enough colors, use them directly
        let palette: Vec<[u8; 3]> = if color_counts.len() <= max_colors {
            color_counts.keys().cloned().collect()
        } else {
            // Use median cut quantization
            self.median_cut_quantize(image, max_colors)?
        };

        // Map pixels to palette indices
        let mut indexed = Vec::with_capacity(width * height);
        let palette_len = palette.len();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let rgb = [data[idx], data[idx + 1], data[idx + 2]];

                // Find closest color in palette
                let palette_idx = find_closest_color(&rgb, &palette, palette_len);
                indexed.push(palette_idx);
            }
        }

        Ok((palette, indexed))
    }

    /// Median cut color quantization.
    fn median_cut_quantize(&self, image: &Image, max_colors: usize) -> Result<Vec<[u8; 3]>> {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let data = image.data();

        // Collect all unique colors
        let mut colors: Vec<[u8; 3]> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                colors.push([data[idx], data[idx + 1], data[idx + 2]]);
            }
        }

        // Initial box containing all colors
        let mut boxes: Vec<ColorBox> = vec![ColorBox::new(colors)];

        // Split until we have enough colors
        while boxes.len() < max_colors {
            // Find box with most colors and largest range
            let best_idx = boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| b.colors.len() > 1)
                .max_by_key(|(_, b)| b.colors.len() * (b.range[0].max(b.range[1]).max(b.range[2])) as usize)
                .map(|(i, _)| i);

            let Some(best_idx) = best_idx else {
                break;
            };

            if boxes[best_idx].colors.len() <= 1 {
                break;
            }

            // Remove the box, split it, and add both halves back
            let best_box = boxes.swap_remove(best_idx);
            let (box1, box2) = best_box.split();
            boxes.push(box1);
            boxes.push(box2);
        }

        // Get representative color from each box
        let palette: Vec<[u8; 3]> = boxes.iter().map(|b| b.average()).collect();

        Ok(palette)
    }
}

impl Default for GifEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Color box for median cut quantization.
struct ColorBox {
    colors: Vec<[u8; 3]>,
    min: [u8; 3],
    max: [u8; 3],
    range: [u8; 3],
}

impl ColorBox {
    fn new(colors: Vec<[u8; 3]>) -> Self {
        let mut min = [255u8; 3];
        let mut max = [0u8; 3];

        for color in &colors {
            for i in 0..3 {
                min[i] = min[i].min(color[i]);
                max[i] = max[i].max(color[i]);
            }
        }

        let range = [
            max[0].saturating_sub(min[0]),
            max[1].saturating_sub(min[1]),
            max[2].saturating_sub(min[2]),
        ];

        Self { colors, min, max, range }
    }

    fn split(mut self) -> (ColorBox, ColorBox) {
        // Find channel with largest range
        let channel = if self.range[0] >= self.range[1] && self.range[0] >= self.range[2] {
            0
        } else if self.range[1] >= self.range[2] {
            1
        } else {
            2
        };

        // Sort by that channel
        self.colors.sort_by_key(|c| c[channel]);

        // Split at median
        let mid = self.colors.len() / 2;
        let colors2 = self.colors.split_off(mid);

        (ColorBox::new(self.colors), ColorBox::new(colors2))
    }

    fn average(&self) -> [u8; 3] {
        if self.colors.is_empty() {
            return [0, 0, 0];
        }

        let mut sum = [0u32; 3];
        for color in &self.colors {
            sum[0] += color[0] as u32;
            sum[1] += color[1] as u32;
            sum[2] += color[2] as u32;
        }

        let len = self.colors.len() as u32;
        [
            (sum[0] / len) as u8,
            (sum[1] / len) as u8,
            (sum[2] / len) as u8,
        ]
    }
}

/// Find closest color in palette.
fn find_closest_color(color: &[u8; 3], palette: &[[u8; 3]], palette_len: usize) -> u8 {
    let mut best_idx = 0;
    let mut best_dist = u32::MAX;

    for (i, p) in palette.iter().enumerate().take(palette_len) {
        let dr = color[0] as i32 - p[0] as i32;
        let dg = color[1] as i32 - p[1] as i32;
        let db = color[2] as i32 - p[2] as i32;
        let dist = (dr * dr + dg * dg + db * db) as u32;

        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }

        if dist == 0 {
            break;
        }
    }

    best_idx as u8
}

/// Find transparent color index.
fn find_transparent_index(
    rgba: &Image,
    indexed: &[u8],
    _palette: &[[u8; 3]],
) -> Option<u8> {
    let data = rgba.data();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;

    // Check if any pixel has alpha < 128
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            if data[idx + 3] < 128 {
                // Find an unused or least-used color index for transparency
                // For simplicity, use index 0 or find first transparent pixel's mapped index
                return Some(indexed[y * width + x]);
            }
        }
    }

    None
}

/// Calculate palette size flag.
fn palette_size_flag(len: usize) -> u8 {
    // Flag is log2(len) - 1
    let mut size = 0u8;
    let mut n = 2;
    while n < len && size < 7 {
        size += 1;
        n <<= 1;
    }
    size
}

/// Write color table.
fn write_color_table(output: &mut Vec<u8>, palette: &[[u8; 3]], size_flag: u8) {
    let table_size = 1 << (size_flag + 1);

    for color in palette.iter().take(table_size) {
        output.extend_from_slice(color);
    }

    // Pad with zeros if needed
    for _ in palette.len()..table_size {
        output.extend_from_slice(&[0, 0, 0]);
    }
}

/// Write NETSCAPE extension for animation.
fn write_netscape_extension(output: &mut Vec<u8>, loop_count: u16) {
    output.push(EXTENSION_INTRODUCER);
    output.push(APPLICATION_LABEL);
    output.push(11); // Block size
    output.extend_from_slice(b"NETSCAPE2.0");
    output.push(3); // Sub-block size
    output.push(1); // Sub-block ID
    output.extend_from_slice(&loop_count.to_le_bytes());
    output.push(0); // Block terminator
}

/// Write graphic control extension.
fn write_graphic_control(output: &mut Vec<u8>, frame: &GifFrame) {
    output.push(EXTENSION_INTRODUCER);
    output.push(GRAPHIC_CONTROL_LABEL);
    output.push(4); // Block size

    let mut flags = frame.disposal.to_byte() << 2;
    if frame.transparent_index.is_some() {
        flags |= 0x01;
    }
    output.push(flags);

    output.extend_from_slice(&frame.delay.to_le_bytes());
    output.push(frame.transparent_index.unwrap_or(0));
    output.push(0); // Block terminator
}

/// Write image descriptor.
fn write_image_descriptor(output: &mut Vec<u8>, frame: &GifFrame, interlace: bool) {
    output.push(IMAGE_SEPARATOR);
    output.extend_from_slice(&frame.x_offset.to_le_bytes());
    output.extend_from_slice(&frame.y_offset.to_le_bytes());
    output.extend_from_slice(&(frame.width as u16).to_le_bytes());
    output.extend_from_slice(&(frame.height as u16).to_le_bytes());

    let mut flags = 0u8;
    if frame.local_palette.is_some() {
        flags |= 0x80;
        let size = palette_size_flag(frame.local_palette.as_ref().unwrap().len());
        flags |= size;
    }
    if interlace || frame.interlaced {
        flags |= 0x40;
    }
    output.push(flags);
}

/// Write data as sub-blocks.
fn write_sub_blocks(output: &mut Vec<u8>, data: &[u8]) {
    for chunk in data.chunks(255) {
        output.push(chunk.len() as u8);
        output.extend_from_slice(chunk);
    }
    output.push(0); // Block terminator
}

/// Calculate minimum LZW code size.
fn calculate_min_code_size(palette_size: usize) -> u8 {
    let mut bits = 2u8;
    while (1 << bits) < palette_size && bits < 8 {
        bits += 1;
    }
    bits.max(2)
}

/// LZW encoder for GIF.
fn lzw_encode(data: &[u8], min_code_size: u8) -> Result<Vec<u8>> {
    let clear_code = 1u16 << min_code_size;
    let eoi_code = clear_code + 1;

    let mut output = Vec::new();
    let mut table: std::collections::HashMap<Vec<u8>, u16> = std::collections::HashMap::new();
    let mut code_size = min_code_size + 1;
    let mut next_code = eoi_code + 1;

    let mut bit_buffer = 0u32;
    let mut bits_in_buffer = 0u32;

    // Helper to write code
    let write_code = |output: &mut Vec<u8>,
                      code: u16,
                      bit_buffer: &mut u32,
                      bits_in_buffer: &mut u32,
                      code_size: u8| {
        *bit_buffer |= (code as u32) << *bits_in_buffer;
        *bits_in_buffer += code_size as u32;

        while *bits_in_buffer >= 8 {
            output.push((*bit_buffer & 0xFF) as u8);
            *bit_buffer >>= 8;
            *bits_in_buffer -= 8;
        }
    };

    // Initialize table
    fn init_table(
        table: &mut std::collections::HashMap<Vec<u8>, u16>,
        clear_code: u16,
    ) -> u16 {
        table.clear();
        for i in 0..clear_code {
            table.insert(vec![i as u8], i);
        }
        clear_code + 2
    }

    next_code = init_table(&mut table, clear_code);

    // Write clear code
    write_code(
        &mut output,
        clear_code,
        &mut bit_buffer,
        &mut bits_in_buffer,
        code_size,
    );

    if data.is_empty() {
        write_code(
            &mut output,
            eoi_code,
            &mut bit_buffer,
            &mut bits_in_buffer,
            code_size,
        );
        if bits_in_buffer > 0 {
            output.push((bit_buffer & 0xFF) as u8);
        }
        return Ok(output);
    }

    let mut current = vec![data[0]];

    for &byte in data.iter().skip(1) {
        let mut next = current.clone();
        next.push(byte);

        if table.contains_key(&next) {
            current = next;
        } else {
            // Output code for current
            let code = table[&current];
            write_code(
                &mut output,
                code,
                &mut bit_buffer,
                &mut bits_in_buffer,
                code_size,
            );

            // Add new string to table
            if next_code < 4096 {
                table.insert(next, next_code);
                next_code += 1;

                // Increase code size if needed
                if next_code > (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
            } else {
                // Table full, emit clear code and reset
                write_code(
                    &mut output,
                    clear_code,
                    &mut bit_buffer,
                    &mut bits_in_buffer,
                    code_size,
                );
                next_code = init_table(&mut table, clear_code);
                code_size = min_code_size + 1;
            }

            current = vec![byte];
        }
    }

    // Output final code
    if !current.is_empty() {
        let code = table[&current];
        write_code(
            &mut output,
            code,
            &mut bit_buffer,
            &mut bits_in_buffer,
            code_size,
        );
    }

    // Write EOI code
    write_code(
        &mut output,
        eoi_code,
        &mut bit_buffer,
        &mut bits_in_buffer,
        code_size,
    );

    // Flush remaining bits
    if bits_in_buffer > 0 {
        output.push((bit_buffer & 0xFF) as u8);
    }

    Ok(output)
}

/// Interlace image data.
fn interlace(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len());

    // Pass 1: Every 8th row, starting at row 0
    for y in (0..height).step_by(8) {
        output.extend_from_slice(&data[y * width..(y + 1) * width]);
    }

    // Pass 2: Every 8th row, starting at row 4
    for y in (4..height).step_by(8) {
        output.extend_from_slice(&data[y * width..(y + 1) * width]);
    }

    // Pass 3: Every 4th row, starting at row 2
    for y in (2..height).step_by(4) {
        output.extend_from_slice(&data[y * width..(y + 1) * width]);
    }

    // Pass 4: Every 2nd row, starting at row 1
    for y in (1..height).step_by(2) {
        output.extend_from_slice(&data[y * width..(y + 1) * width]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = GifEncoder::new();
        assert_eq!(encoder.config.max_colors, 256);
        assert_eq!(encoder.config.loop_count, 0);
    }

    #[test]
    fn test_encoder_config() {
        let encoder = GifEncoder::new()
            .max_colors(128)
            .loop_count(5)
            .dithering(true)
            .default_delay(20);

        assert_eq!(encoder.config.max_colors, 128);
        assert_eq!(encoder.config.loop_count, 5);
        assert!(encoder.config.dithering);
        assert_eq!(encoder.config.default_delay, 20);
    }

    #[test]
    fn test_palette_size_flag() {
        assert_eq!(palette_size_flag(2), 0);
        assert_eq!(palette_size_flag(4), 1);
        assert_eq!(palette_size_flag(8), 2);
        assert_eq!(palette_size_flag(16), 3);
        assert_eq!(palette_size_flag(256), 7);
    }

    #[test]
    fn test_min_code_size() {
        assert_eq!(calculate_min_code_size(2), 2);
        assert_eq!(calculate_min_code_size(4), 2);
        assert_eq!(calculate_min_code_size(8), 3);
        assert_eq!(calculate_min_code_size(256), 8);
    }

    #[test]
    fn test_lzw_encode_simple() {
        let data = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let result = lzw_encode(&data, 2);
        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_interlace() {
        let width = 4;
        let height = 8;
        let mut data = Vec::new();
        for y in 0..height {
            data.extend(vec![y as u8; width]);
        }

        let interlaced = interlace(&data, width, height);

        // Check interlaced order
        // Pass 1: rows 0
        assert_eq!(interlaced[0..width], vec![0u8; width]);
        // Pass 2: rows 4
        assert_eq!(interlaced[width..2 * width], vec![4u8; width]);
        // Pass 3: rows 2, 6
        assert_eq!(interlaced[2 * width..3 * width], vec![2u8; width]);
        assert_eq!(interlaced[3 * width..4 * width], vec![6u8; width]);
        // Pass 4: rows 1, 3, 5, 7
        assert_eq!(interlaced[4 * width..5 * width], vec![1u8; width]);
    }

    #[test]
    fn test_find_closest_color() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];

        assert_eq!(find_closest_color(&[0, 0, 0], &palette, 4), 0);
        assert_eq!(find_closest_color(&[255, 0, 0], &palette, 4), 1);
        assert_eq!(find_closest_color(&[200, 10, 10], &palette, 4), 1);
    }

    #[test]
    fn test_color_box() {
        let colors = vec![
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ];

        let cbox = ColorBox::new(colors);
        assert_eq!(cbox.min, [0, 0, 0]);
        assert_eq!(cbox.max, [255, 255, 255]);
    }

    #[test]
    fn test_encode_small_image() {
        let mut image = Image::new(4, 4, PixelFormat::Rgba8).unwrap();
        // Fill with red pixels
        for y in 0..4 {
            for x in 0..4 {
                image.set_pixel(x, y, &[255, 0, 0, 255]);
            }
        }

        let encoder = GifEncoder::new();
        let result = encoder.encode(&image);
        assert!(result.is_ok());

        let gif_data = result.unwrap();
        // Check GIF signature
        assert_eq!(&gif_data[0..6], b"GIF89a");
    }

    #[test]
    fn test_write_netscape_extension() {
        let mut output = Vec::new();
        write_netscape_extension(&mut output, 0);

        assert_eq!(output[0], EXTENSION_INTRODUCER);
        assert_eq!(output[1], APPLICATION_LABEL);
        assert_eq!(&output[3..14], b"NETSCAPE2.0");
    }
}
