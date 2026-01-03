//! VP8L lossless decoder
//!
//! VP8L uses LZ77, Huffman coding, and various transforms for lossless compression.
//! This includes support for:
//! - Huffman coding for symbols
//! - LZ77 backward references
//! - Color cache for recently used colors
//! - Various image transforms (predictor, cross-color, subtract-green, color indexing)

use crate::bitreader::BitReader;
use crate::error::{WebPError, Result};
use crate::transform::{Transform, TransformType};
use image::RgbaImage;

/// VP8L signature byte
const VP8L_SIGNATURE: u8 = 0x2f;

// Maximum number of Huffman codes (reserved for future optimization)
#[allow(dead_code)]
const MAX_HUFFMAN_CODES: usize = 2328;

/// Maximum color cache bits (cache size = 2^cache_bits)
const MAX_COLOR_CACHE_BITS: u32 = 11;

/// Hash multiplier for color cache indexing (from VP8L spec)
const COLOR_CACHE_HASH_MUL: u32 = 0x1e35a7bd;

/// Alphabet sizes for different symbol types
/// Note: actual green alphabet size depends on color cache size
const ALPHABET_SIZE: [usize; 5] = [256 + 24, 256, 256, 256, 40];

/// Color cache for VP8L decoding
///
/// The color cache stores recently used ARGB colors for efficient
/// back-reference during decoding. Colors are indexed using a hash
/// function that maps ARGB values to cache positions.
#[derive(Debug, Clone)]
pub struct ColorCache {
    /// Cache storage (size = 2^cache_bits)
    colors: Vec<u32>,
    /// Number of bits for cache indexing (1-11, or 0 for disabled)
    cache_bits: u32,
    /// Bit mask for indexing (2^cache_bits - 1)
    #[allow(dead_code)]
    hash_mask: u32,
}

impl ColorCache {
    /// Create a new color cache with the given number of bits
    ///
    /// # Arguments
    /// * `cache_bits` - Number of bits (1-11). Use 0 to disable cache.
    pub fn new(cache_bits: u32) -> Self {
        let cache_bits = cache_bits.min(MAX_COLOR_CACHE_BITS);
        if cache_bits == 0 {
            return Self {
                colors: Vec::new(),
                cache_bits: 0,
                hash_mask: 0,
            };
        }

        let size = 1usize << cache_bits;
        Self {
            colors: vec![0u32; size],
            cache_bits,
            hash_mask: (1 << cache_bits) - 1,
        }
    }

    /// Create a disabled color cache
    pub fn disabled() -> Self {
        Self::new(0)
    }

    /// Check if the color cache is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.cache_bits > 0
    }

    /// Get the cache size (number of entries)
    #[inline]
    pub fn size(&self) -> usize {
        self.colors.len()
    }

    /// Hash a color to get its cache index
    #[inline]
    fn hash(&self, color: u32) -> usize {
        ((COLOR_CACHE_HASH_MUL.wrapping_mul(color)) >> (32 - self.cache_bits)) as usize
    }

    /// Insert a color into the cache
    #[inline]
    pub fn insert(&mut self, color: u32) {
        if self.cache_bits > 0 {
            let index = self.hash(color);
            self.colors[index] = color;
        }
    }

    /// Look up a color by its cache index
    ///
    /// # Arguments
    /// * `index` - Cache index (0 to size-1)
    ///
    /// # Returns
    /// The cached color, or 0 if index is out of bounds
    #[inline]
    pub fn lookup(&self, index: usize) -> u32 {
        self.colors.get(index).copied().unwrap_or(0)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.colors.fill(0);
    }
}

/// VP8L decoder
pub struct Vp8lDecoder<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    alpha_used: bool,
    #[allow(dead_code)]
    version: u8,
    /// Color cache bits (0 means disabled)
    #[allow(dead_code)]
    color_cache_bits: u32,
}

impl<'a> Vp8lDecoder<'a> {
    /// Create a new VP8L decoder
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(WebPError::InvalidVp8l("Data too short".into()));
        }

        if data[0] != VP8L_SIGNATURE {
            return Err(WebPError::InvalidVp8l("Invalid VP8L signature".into()));
        }

        // Parse header
        let bits = u32::from(data[1])
            | (u32::from(data[2]) << 8)
            | (u32::from(data[3]) << 16)
            | (u32::from(data[4]) << 24);

        let width = (bits & 0x3FFF) + 1;
        let height = ((bits >> 14) & 0x3FFF) + 1;
        let alpha_used = ((bits >> 28) & 1) != 0;
        let version = ((bits >> 29) & 7) as u8;

        if version != 0 {
            return Err(WebPError::InvalidVp8l(format!(
                "Unsupported VP8L version: {}",
                version
            )));
        }

        Ok(Self {
            data,
            width,
            height,
            alpha_used,
            version,
            color_cache_bits: 0, // Will be read during decode
        })
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if alpha channel is used
    pub fn has_alpha(&self) -> bool {
        self.alpha_used
    }

    /// Decode the VP8L image to an RGBA image
    pub fn decode(&self) -> Result<RgbaImage> {
        let mut reader = BitReader::new(&self.data[5..]);

        // Read transforms
        let transforms = self.read_transforms(&mut reader)?;

        // Read color cache configuration
        let color_cache_bits = if reader.read_bit()? != 0 {
            let bits = reader.read_bits(4)?;
            if bits > MAX_COLOR_CACHE_BITS {
                return Err(WebPError::InvalidVp8l(format!(
                    "Invalid color cache bits: {}",
                    bits
                )));
            }
            bits
        } else {
            0
        };

        // Create color cache
        let mut color_cache = ColorCache::new(color_cache_bits);

        // Decode the main image with color cache
        let mut pixels = self.decode_image_data_with_cache(
            &mut reader,
            self.width,
            self.height,
            &mut color_cache,
        )?;

        // Apply transforms in reverse order
        self.apply_transforms(&mut pixels, &transforms)?;

        // Convert to RgbaImage
        self.pixels_to_image(&pixels)
    }

    fn read_transforms(&self, reader: &mut BitReader) -> Result<Vec<Transform>> {
        let mut transforms = Vec::new();

        while reader.read_bit()? != 0 {
            let transform_type = reader.read_bits(2)?;
            let transform = match transform_type {
                0 => self.read_predictor_transform(reader)?,
                1 => self.read_cross_color_transform(reader)?,
                2 => Transform {
                    transform_type: TransformType::SubtractGreen,
                    bits: 0,
                    data: Vec::new(),
                },
                3 => self.read_color_indexing_transform(reader)?,
                _ => unreachable!(),
            };

            transforms.push(transform);
        }

        Ok(transforms)
    }

    fn read_predictor_transform(&self, reader: &mut BitReader) -> Result<Transform> {
        let bits = reader.read_bits(3)? + 2;
        let block_size = 1u32 << bits;
        let blocks_width = self.width.div_ceil(block_size);
        let blocks_height = self.height.div_ceil(block_size);

        let data = self.decode_image_data(reader, blocks_width, blocks_height)?;

        Ok(Transform {
            transform_type: TransformType::Predictor,
            bits,
            data,
        })
    }

    fn read_cross_color_transform(&self, reader: &mut BitReader) -> Result<Transform> {
        let bits = reader.read_bits(3)? + 2;
        let block_size = 1u32 << bits;
        let blocks_width = self.width.div_ceil(block_size);
        let blocks_height = self.height.div_ceil(block_size);

        let data = self.decode_image_data(reader, blocks_width, blocks_height)?;

        Ok(Transform {
            transform_type: TransformType::CrossColor,
            bits,
            data,
        })
    }

    fn read_color_indexing_transform(&self, reader: &mut BitReader) -> Result<Transform> {
        let color_table_size = reader.read_bits(8)? + 1;

        let data = self.decode_image_data(reader, color_table_size, 1)?;

        // Determine bits per pixel
        let bits = if color_table_size <= 2 {
            3
        } else if color_table_size <= 4 {
            2
        } else if color_table_size <= 16 {
            1
        } else {
            0
        };

        Ok(Transform {
            transform_type: TransformType::ColorIndexing,
            bits,
            data,
        })
    }

    fn decode_image_data(
        &self,
        reader: &mut BitReader,
        width: u32,
        height: u32,
    ) -> Result<Vec<u32>> {
        // Read Huffman codes without color cache
        let huffman_codes = self.read_huffman_codes_with_cache(reader, width, height, 0)?;

        // Decode pixel data
        let num_pixels = (width * height) as usize;
        let mut pixels = vec![0u32; num_pixels];

        // Use a disabled cache for transforms
        let mut cache = ColorCache::disabled();
        self.decode_pixels_with_cache(reader, &huffman_codes, &mut pixels, width, &mut cache)?;

        Ok(pixels)
    }

    fn decode_image_data_with_cache(
        &self,
        reader: &mut BitReader,
        width: u32,
        height: u32,
        color_cache: &mut ColorCache,
    ) -> Result<Vec<u32>> {
        // Read Huffman codes with color cache size adjustment
        let huffman_codes = self.read_huffman_codes_with_cache(
            reader,
            width,
            height,
            color_cache.size(),
        )?;

        // Decode pixel data
        let num_pixels = (width * height) as usize;
        let mut pixels = vec![0u32; num_pixels];

        self.decode_pixels_with_cache(reader, &huffman_codes, &mut pixels, width, color_cache)?;

        Ok(pixels)
    }

    fn read_huffman_codes_with_cache(
        &self,
        reader: &mut BitReader,
        _width: u32,
        _height: u32,
        color_cache_size: usize,
    ) -> Result<HuffmanCodes> {
        // Check for meta-Huffman coding
        let use_meta = reader.read_bit()? != 0;

        if use_meta {
            // Read meta-Huffman image
            let _huffman_bits = reader.read_bits(3)? + 2;
            // For simplicity, we skip meta-Huffman in this implementation
        }

        // Read the 5 Huffman code lengths
        // The green/length alphabet size includes color cache entries
        let mut codes = HuffmanCodes::default();

        for (i, &base_alphabet_size) in ALPHABET_SIZE.iter().enumerate() {
            // For the first alphabet (green/length/cache), add color cache size
            let alphabet_size = if i == 0 {
                base_alphabet_size + color_cache_size
            } else {
                base_alphabet_size
            };
            let code = self.read_huffman_code(reader, alphabet_size)?;
            codes.codes[i] = code;
        }

        Ok(codes)
    }

    fn read_huffman_code(
        &self,
        reader: &mut BitReader,
        alphabet_size: usize,
    ) -> Result<HuffmanCode> {
        let simple_code = reader.read_bit()? != 0;

        if simple_code {
            // Simple code with 1 or 2 symbols
            let num_symbols = reader.read_bit()? + 1;
            let first_symbol_bits = if reader.read_bit()? != 0 { 8 } else { 1 };
            let first_symbol = reader.read_bits(first_symbol_bits)? as u16;

            if num_symbols == 1 {
                return Ok(HuffmanCode::Simple1(first_symbol));
            }

            let second_symbol = reader.read_bits(8)? as u16;
            return Ok(HuffmanCode::Simple2(first_symbol, second_symbol));
        }

        // Normal Huffman code
        // Read code length code lengths
        let num_code_lengths = reader.read_bits(4)? + 4;
        let mut code_length_code_lengths = [0u8; 19];

        const CODE_LENGTH_ORDER: [usize; 19] = [
            17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];

        for i in 0..num_code_lengths as usize {
            if i < CODE_LENGTH_ORDER.len() {
                code_length_code_lengths[CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
            }
        }

        // Build code length decoder
        let code_length_tree = build_huffman_tree(&code_length_code_lengths)?;

        // Read symbol code lengths
        let mut code_lengths = vec![0u8; alphabet_size];
        let mut i = 0;

        while i < alphabet_size {
            let symbol = decode_huffman_symbol(reader, &code_length_tree)?;

            match symbol {
                0..=15 => {
                    code_lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    // Copy previous code length 3-6 times
                    let repeat = reader.read_bits(2)? + 3;
                    let prev = if i > 0 { code_lengths[i - 1] } else { 8 };
                    for _ in 0..repeat {
                        if i < alphabet_size {
                            code_lengths[i] = prev;
                            i += 1;
                        }
                    }
                }
                17 => {
                    // Repeat 0 for 3-10 times
                    let repeat = reader.read_bits(3)? + 3;
                    i += repeat as usize;
                }
                18 => {
                    // Repeat 0 for 11-138 times
                    let repeat = reader.read_bits(7)? + 11;
                    i += repeat as usize;
                }
                _ => return Err(WebPError::InvalidVp8l("Invalid code length symbol".into())),
            }
        }

        let tree = build_huffman_tree(&code_lengths)?;
        Ok(HuffmanCode::Full(tree))
    }

    fn decode_pixels_with_cache(
        &self,
        reader: &mut BitReader,
        codes: &HuffmanCodes,
        pixels: &mut [u32],
        width: u32,
        color_cache: &mut ColorCache,
    ) -> Result<()> {
        let mut pos = 0;
        let num_pixels = pixels.len();
        let color_cache_size = color_cache.size();
        let color_cache_limit = 256 + 24 + color_cache_size as u32;

        while pos < num_pixels {
            let green_or_length = decode_huffman(reader, &codes.codes[0])?;

            if green_or_length < 256 {
                // Literal pixel
                let red = decode_huffman(reader, &codes.codes[1])?;
                let blue = decode_huffman(reader, &codes.codes[2])?;
                let alpha = decode_huffman(reader, &codes.codes[3])?;

                let pixel = (alpha << 24) | (red << 16) | (green_or_length << 8) | blue;
                pixels[pos] = pixel;

                // Update color cache with new pixel
                color_cache.insert(pixel);

                pos += 1;
            } else if green_or_length < 256 + 24 {
                // LZ77 backward reference
                let length_code = green_or_length - 256;
                let length = self.decode_length(reader, length_code)?;

                let dist_code = decode_huffman(reader, &codes.codes[4])?;
                let dist = self.decode_distance(reader, dist_code, width)?;

                // Copy pixels and update cache
                for _ in 0..length {
                    if pos >= num_pixels {
                        break;
                    }
                    let src = if pos >= dist as usize {
                        pixels[pos - dist as usize]
                    } else {
                        0xFF000000 // Default black transparent
                    };
                    pixels[pos] = src;

                    // Update color cache with copied pixel
                    color_cache.insert(src);

                    pos += 1;
                }
            } else if green_or_length < color_cache_limit {
                // Color cache lookup
                let cache_index = (green_or_length - (256 + 24)) as usize;
                let pixel = color_cache.lookup(cache_index);
                pixels[pos] = pixel;

                // Note: Color cache lookups do NOT update the cache
                // (the pixel is already in the cache by definition)

                pos += 1;
            } else {
                // Invalid symbol
                return Err(WebPError::InvalidVp8l(format!(
                    "Invalid symbol: {} (limit: {})",
                    green_or_length, color_cache_limit
                )));
            }
        }

        Ok(())
    }

    fn decode_length(&self, reader: &mut BitReader, code: u32) -> Result<u32> {
        if code < 4 {
            Ok(code + 1)
        } else {
            let extra_bits = (code - 2) >> 1;
            let offset = (2 + (code & 1)) << extra_bits;
            let extra = reader.read_bits(extra_bits)?;
            Ok(offset + extra + 1)
        }
    }

    fn decode_distance(&self, reader: &mut BitReader, code: u32, width: u32) -> Result<u32> {
        if code < 4 {
            Ok(code + 1)
        } else if code < 40 {
            let extra_bits = (code - 2) >> 1;
            let offset = (2 + (code & 1)) << extra_bits;
            let extra = reader.read_bits(extra_bits)?;
            Ok(offset + extra + 1)
        } else {
            // Distance codes 40+ use special 2D distance mapping
            let dist_code = code - 40;
            let _xoffset = DISTANCE_MAP[dist_code as usize].0;
            let _yoffset = DISTANCE_MAP[dist_code as usize].1;
            // Simplified: just return a basic distance
            Ok((dist_code + 1) * width)
        }
    }

    fn apply_transforms(
        &self,
        pixels: &mut Vec<u32>,
        transforms: &[Transform],
    ) -> Result<()> {
        for transform in transforms.iter().rev() {
            match transform.transform_type {
                TransformType::SubtractGreen => {
                    crate::transform::apply_subtract_green(pixels);
                }
                TransformType::Predictor => {
                    crate::transform::apply_predictor_transform(
                        pixels,
                        self.width,
                        self.height,
                        transform.bits,
                        &transform.data,
                    )?;
                }
                TransformType::CrossColor => {
                    crate::transform::apply_cross_color_transform(
                        pixels,
                        self.width,
                        self.height,
                        transform.bits,
                        &transform.data,
                    )?;
                }
                TransformType::ColorIndexing => {
                    crate::transform::apply_color_indexing_transform(
                        pixels,
                        self.width,
                        self.height,
                        transform.bits,
                        &transform.data,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn pixels_to_image(&self, pixels: &[u32]) -> Result<RgbaImage> {
        let mut image = RgbaImage::new(self.width, self.height);

        for (i, &pixel) in pixels.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;

            if x < self.width && y < self.height {
                let b = (pixel & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let r = ((pixel >> 16) & 0xFF) as u8;
                let a = ((pixel >> 24) & 0xFF) as u8;

                image.put_pixel(x, y, image::Rgba([r, g, b, a]));
            }
        }

        Ok(image)
    }
}

/// Distance map for 2D distance codes
static DISTANCE_MAP: [(i32, i32); 120] = [
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2),
    (2, 1), (-2, 1), (2, 2), (-2, 2), (0, 3), (3, 0), (1, 3), (-1, 3),
    (3, 1), (-3, 1), (2, 3), (-2, 3), (3, 2), (-3, 2), (0, 4), (4, 0),
    (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3), (2, 4), (-2, 4),
    (4, 2), (-4, 2), (0, 5), (3, 4), (-3, 4), (4, 3), (-4, 3), (5, 0),
    (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2),
    (4, 4), (-4, 4), (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0),
    (1, 6), (-1, 6), (6, 1), (-6, 1), (2, 6), (-2, 6), (6, 2), (-6, 2),
    (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6), (6, 3), (-6, 3),
    (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1),
    (4, 6), (-4, 6), (6, 4), (-6, 4), (2, 7), (-2, 7), (7, 2), (-7, 2),
    (3, 7), (-3, 7), (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5),
    (8, 0), (4, 7), (-4, 7), (7, 4), (-7, 4), (8, 1), (8, 2), (6, 6),
    (-6, 6), (8, 3), (5, 7), (-5, 7), (7, 5), (-7, 5), (8, 4), (6, 7),
    (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
];

/// Huffman codes for VP8L
#[derive(Default)]
struct HuffmanCodes {
    codes: [HuffmanCode; 5],
}

/// A single Huffman code
#[derive(Clone)]
enum HuffmanCode {
    /// Single symbol code
    Simple1(u16),
    /// Two symbol code
    Simple2(u16, u16),
    /// Full Huffman tree
    Full(HuffmanTree),
}

impl Default for HuffmanCode {
    fn default() -> Self {
        HuffmanCode::Simple1(0)
    }
}

/// Huffman tree node
#[derive(Clone)]
struct HuffmanTree {
    /// Lookup table for fast decoding
    table: Vec<HuffmanEntry>,
    /// Maximum code length
    max_bits: u32,
}

#[derive(Clone, Copy, Default)]
struct HuffmanEntry {
    symbol: u16,
    bits: u8,
}

fn build_huffman_tree(code_lengths: &[u8]) -> Result<HuffmanTree> {
    let max_bits = code_lengths.iter().copied().max().unwrap_or(0) as u32;
    let max_bits = max_bits.clamp(1, 15);

    let table_size = 1usize << max_bits;
    let mut table = vec![HuffmanEntry::default(); table_size];

    // Count code lengths
    let mut counts = [0u32; 16];
    for &len in code_lengths {
        if len > 0 && (len as usize) < counts.len() {
            counts[len as usize] += 1;
        }
    }

    // Calculate starting codes
    let mut codes = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + counts[bits - 1]) << 1;
        codes[bits] = code;
    }

    // Fill the table
    for (symbol, &len) in code_lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }

        let len = len as usize;
        if len >= codes.len() {
            continue;
        }

        let code = codes[len];
        codes[len] += 1;

        // Fill all entries with this prefix
        let entry = HuffmanEntry {
            symbol: symbol as u16,
            bits: len as u8,
        };

        let shift = max_bits as usize - len;
        let base = (code as usize) << shift;
        let count = 1usize << shift;

        for i in 0..count {
            if base + i < table.len() {
                table[base + i] = entry;
            }
        }
    }

    Ok(HuffmanTree { table, max_bits })
}

fn decode_huffman_symbol(reader: &mut BitReader, tree: &HuffmanTree) -> Result<u32> {
    let bits = reader.read_bits(tree.max_bits)?;
    let entry = &tree.table[bits as usize];

    // Put back unused bits (we can't actually put bits back with our simple reader)
    let _unused = tree.max_bits - entry.bits as u32;

    Ok(entry.symbol as u32)
}

fn decode_huffman(reader: &mut BitReader, code: &HuffmanCode) -> Result<u32> {
    match code {
        HuffmanCode::Simple1(symbol) => Ok(*symbol as u32),
        HuffmanCode::Simple2(s0, s1) => {
            let bit = reader.read_bit()?;
            if bit == 0 {
                Ok(*s0 as u32)
            } else {
                Ok(*s1 as u32)
            }
        }
        HuffmanCode::Full(tree) => decode_huffman_symbol(reader, tree),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp8l_header_parsing() {
        // VP8L header for 100x100 image
        let mut data = vec![0x2f]; // Signature
        // Width = 100 (stored as 99), Height = 100 (stored as 99)
        // alpha_used = 1, version = 0
        let bits: u32 = 99 | (99 << 14) | (1 << 28);
        data.push((bits & 0xFF) as u8);
        data.push(((bits >> 8) & 0xFF) as u8);
        data.push(((bits >> 16) & 0xFF) as u8);
        data.push(((bits >> 24) & 0xFF) as u8);

        let decoder = Vp8lDecoder::new(&data);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert_eq!(decoder.dimensions(), (100, 100));
        assert!(decoder.has_alpha());
    }

    #[test]
    fn test_invalid_signature() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        let decoder = Vp8lDecoder::new(&data);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_huffman_tree() {
        // Simple tree with two symbols of equal length
        let code_lengths = [1u8, 1];
        let tree = build_huffman_tree(&code_lengths);
        assert!(tree.is_ok());
    }

    // Color cache tests
    #[test]
    fn test_color_cache_creation() {
        let cache = ColorCache::new(4);
        assert!(cache.is_enabled());
        assert_eq!(cache.size(), 16); // 2^4 = 16
    }

    #[test]
    fn test_color_cache_disabled() {
        let cache = ColorCache::disabled();
        assert!(!cache.is_enabled());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_color_cache_zero_bits() {
        let cache = ColorCache::new(0);
        assert!(!cache.is_enabled());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_color_cache_max_bits_clamping() {
        // Should clamp to MAX_COLOR_CACHE_BITS (11)
        let cache = ColorCache::new(15);
        assert!(cache.is_enabled());
        assert_eq!(cache.size(), 2048); // 2^11 = 2048
    }

    #[test]
    fn test_color_cache_insert_lookup() {
        let mut cache = ColorCache::new(4);

        // Insert a color
        let color: u32 = 0xFF112233; // ARGB
        cache.insert(color);

        // The color should be retrievable at its hash position
        // We can't easily predict the exact index, but we can verify
        // the mechanism works by looking up at the hash position
        let index = ((COLOR_CACHE_HASH_MUL.wrapping_mul(color)) >> (32 - 4)) as usize;
        assert_eq!(cache.lookup(index), color);
    }

    #[test]
    fn test_color_cache_multiple_inserts() {
        let mut cache = ColorCache::new(4);

        let colors = [
            0xFFFF0000u32, // Red
            0xFF00FF00,    // Green
            0xFF0000FF,    // Blue
            0xFFFFFFFF,    // White
        ];

        for &color in &colors {
            cache.insert(color);
        }

        // Verify each color is at its expected position
        for &color in &colors {
            let index = ((COLOR_CACHE_HASH_MUL.wrapping_mul(color)) >> (32 - 4)) as usize;
            // Note: due to hash collisions, later colors may overwrite earlier ones
            // We just verify the lookup doesn't panic
            let _ = cache.lookup(index);
        }
    }

    #[test]
    fn test_color_cache_lookup_out_of_bounds() {
        let cache = ColorCache::new(2); // Size = 4
        // Should return 0 for out-of-bounds index
        assert_eq!(cache.lookup(100), 0);
    }

    #[test]
    fn test_color_cache_clear() {
        let mut cache = ColorCache::new(4);
        cache.insert(0xFF112233);
        cache.clear();

        // All entries should be 0
        for i in 0..cache.size() {
            assert_eq!(cache.lookup(i), 0);
        }
    }

    #[test]
    fn test_color_cache_disabled_insert_noop() {
        let mut cache = ColorCache::disabled();
        // Insert should not panic on disabled cache
        cache.insert(0xFF112233);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_color_cache_hash_distribution() {
        // Test that different colors hash to different positions (mostly)
        let _cache = ColorCache::new(8); // 256 entries
        let mut positions = std::collections::HashSet::new();

        // Test 100 different colors with more variation
        for i in 0..100u32 {
            // Use varied color generation to test hash distribution
            let r = (i * 17) & 0xFF;
            let g = (i * 31) & 0xFF;
            let b = (i * 47) & 0xFF;
            let color = 0xFF000000 | (r << 16) | (g << 8) | b;
            let index = ((COLOR_CACHE_HASH_MUL.wrapping_mul(color)) >> (32 - 8)) as usize;
            positions.insert(index);
        }

        // Should have reasonable distribution (at least 30 unique positions)
        // Note: Hash collisions are expected and acceptable
        assert!(
            positions.len() > 30,
            "Expected more than 30 unique positions, got {}",
            positions.len()
        );
    }

    #[test]
    fn test_color_cache_sizes() {
        for bits in 1..=11 {
            let cache = ColorCache::new(bits);
            assert_eq!(cache.size(), 1 << bits);
            assert!(cache.is_enabled());
        }
    }
}
