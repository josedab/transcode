//! TIFF compression methods

use crate::error::{Result, TiffError};

/// Compression methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// No compression
    #[default]
    None,
    /// CCITT Group 3 fax encoding
    CcittGroup3,
    /// CCITT Group 4 fax encoding
    CcittGroup4,
    /// LZW compression
    Lzw,
    /// JPEG compression (old-style)
    OldJpeg,
    /// JPEG compression
    Jpeg,
    /// Deflate/ZIP compression (Adobe)
    AdobeDeflate,
    /// PackBits RLE compression
    PackBits,
    /// Deflate compression
    Deflate,
}

impl Compression {
    /// Create from TIFF compression tag value
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Compression::None),
            2 => Some(Compression::CcittGroup3),
            3 => Some(Compression::CcittGroup3),
            4 => Some(Compression::CcittGroup4),
            5 => Some(Compression::Lzw),
            6 => Some(Compression::OldJpeg),
            7 => Some(Compression::Jpeg),
            8 => Some(Compression::AdobeDeflate),
            32773 => Some(Compression::PackBits),
            32946 => Some(Compression::Deflate),
            _ => None,
        }
    }

    /// Convert to TIFF compression tag value
    pub fn to_u16(self) -> u16 {
        match self {
            Compression::None => 1,
            Compression::CcittGroup3 => 3,
            Compression::CcittGroup4 => 4,
            Compression::Lzw => 5,
            Compression::OldJpeg => 6,
            Compression::Jpeg => 7,
            Compression::AdobeDeflate => 8,
            Compression::PackBits => 32773,
            Compression::Deflate => 32946,
        }
    }

    /// Get compression name
    pub fn name(&self) -> &'static str {
        match self {
            Compression::None => "None",
            Compression::CcittGroup3 => "CCITT Group 3",
            Compression::CcittGroup4 => "CCITT Group 4",
            Compression::Lzw => "LZW",
            Compression::OldJpeg => "Old JPEG",
            Compression::Jpeg => "JPEG",
            Compression::AdobeDeflate => "Adobe Deflate",
            Compression::PackBits => "PackBits",
            Compression::Deflate => "Deflate",
        }
    }
}

/// Decompress data
pub fn decompress(compression: Compression, data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::PackBits => decompress_packbits(data, expected_size),
        Compression::Lzw => decompress_lzw(data, expected_size),
        _ => Err(TiffError::UnsupportedCompression(compression.to_u16())),
    }
}

/// Compress data
pub fn compress(compression: Compression, data: &[u8]) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::PackBits => Ok(compress_packbits(data)),
        Compression::Lzw => Ok(compress_lzw(data)),
        _ => Err(TiffError::UnsupportedCompression(compression.to_u16())),
    }
}

/// Decompress PackBits RLE data
fn decompress_packbits(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut i = 0;

    while i < data.len() && output.len() < expected_size {
        let header = data[i] as i8;
        i += 1;

        if header >= 0 {
            // Literal run: copy next (header + 1) bytes
            let count = (header as usize) + 1;
            if i + count > data.len() {
                return Err(TiffError::DecompressionError(
                    "PackBits: unexpected end of data".into(),
                ));
            }
            output.extend_from_slice(&data[i..i + count]);
            i += count;
        } else if header != -128 {
            // Repeat run: repeat next byte (-header + 1) times
            let count = (-header as usize) + 1;
            if i >= data.len() {
                return Err(TiffError::DecompressionError(
                    "PackBits: unexpected end of data".into(),
                ));
            }
            let value = data[i];
            i += 1;
            for _ in 0..count {
                output.push(value);
            }
        }
        // header == -128 is a no-op
    }

    Ok(output)
}

/// Compress data using PackBits RLE
fn compress_packbits(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Look for runs
        let mut run_length = 1;
        while i + run_length < data.len()
            && run_length < 128
            && data[i + run_length] == data[i]
        {
            run_length += 1;
        }

        if run_length > 1 {
            // Encode run
            output.push((-(run_length as i8 - 1)) as u8);
            output.push(data[i]);
            i += run_length;
        } else {
            // Look for literal sequence
            let start = i;
            let mut literal_len = 1;
            i += 1;

            while i < data.len() && literal_len < 128 {
                // Check if we're starting a run
                if i + 1 < data.len() && data[i] == data[i + 1] {
                    break;
                }
                literal_len += 1;
                i += 1;
            }

            // Encode literal
            output.push((literal_len - 1) as u8);
            output.extend_from_slice(&data[start..start + literal_len]);
        }
    }

    output
}

/// LZW decoder state
struct LzwDecoder {
    table: Vec<Vec<u8>>,
    code_size: u8,
    next_code: u16,
    clear_code: u16,
    eoi_code: u16,
}

impl LzwDecoder {
    fn new() -> Self {
        let mut decoder = LzwDecoder {
            table: Vec::new(),
            code_size: 9,
            next_code: 0,
            clear_code: 256,
            eoi_code: 257,
        };
        decoder.reset();
        decoder
    }

    fn reset(&mut self) {
        self.table.clear();
        // Initialize with single-byte entries
        for i in 0..256 {
            self.table.push(vec![i as u8]);
        }
        // Clear code and EOI code
        self.table.push(vec![]); // 256 = clear
        self.table.push(vec![]); // 257 = EOI
        self.code_size = 9;
        self.next_code = 258;
    }

    fn add_entry(&mut self, entry: Vec<u8>) {
        self.table.push(entry);
        self.next_code += 1;

        // Increase code size when needed
        if self.next_code >= (1 << self.code_size) && self.code_size < 12 {
            self.code_size += 1;
        }
    }
}

/// Decompress LZW data
fn decompress_lzw(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut decoder = LzwDecoder::new();

    // Bit reader for LZW codes
    let mut bit_pos: usize = 0;
    let total_bits = data.len() * 8;

    let read_code = |data: &[u8], bit_pos: &mut usize, code_size: u8| -> Option<u16> {
        if *bit_pos + code_size as usize > total_bits {
            return None;
        }

        let mut code: u16 = 0;
        for i in 0..code_size {
            let byte_idx = (*bit_pos + i as usize) / 8;
            let bit_idx = (*bit_pos + i as usize) % 8;
            if byte_idx < data.len() && (data[byte_idx] >> bit_idx) & 1 != 0 {
                code |= 1 << i;
            }
        }
        *bit_pos += code_size as usize;
        Some(code)
    };

    let mut prev_code: Option<u16> = None;

    while let Some(code) = read_code(data, &mut bit_pos, decoder.code_size) {
        if code == decoder.eoi_code {
            break;
        }

        if code == decoder.clear_code {
            decoder.reset();
            prev_code = None;
            continue;
        }

        let entry = if (code as usize) < decoder.table.len() {
            decoder.table[code as usize].clone()
        } else if code == decoder.next_code {
            // Special case: code not in table yet
            if let Some(prev) = prev_code {
                let mut entry = decoder.table[prev as usize].clone();
                entry.push(entry[0]);
                entry
            } else {
                return Err(TiffError::DecompressionError("LZW: invalid code".into()));
            }
        } else {
            return Err(TiffError::DecompressionError("LZW: code out of range".into()));
        };

        output.extend_from_slice(&entry);

        // Add new entry to table
        if let Some(prev) = prev_code {
            if decoder.next_code < 4096 {
                let mut new_entry = decoder.table[prev as usize].clone();
                new_entry.push(entry[0]);
                decoder.add_entry(new_entry);
            }
        }

        prev_code = Some(code);

        if output.len() >= expected_size {
            break;
        }
    }

    Ok(output)
}

/// LZW encoder state
struct LzwEncoder {
    table: std::collections::HashMap<Vec<u8>, u16>,
    code_size: u8,
    next_code: u16,
    clear_code: u16,
    eoi_code: u16,
}

impl LzwEncoder {
    fn new() -> Self {
        let mut encoder = LzwEncoder {
            table: std::collections::HashMap::new(),
            code_size: 9,
            next_code: 0,
            clear_code: 256,
            eoi_code: 257,
        };
        encoder.reset();
        encoder
    }

    fn reset(&mut self) {
        self.table.clear();
        // Initialize with single-byte entries
        for i in 0..256 {
            self.table.insert(vec![i as u8], i as u16);
        }
        self.code_size = 9;
        self.next_code = 258;
    }

    fn add_entry(&mut self, entry: Vec<u8>) -> bool {
        if self.next_code >= 4095 {
            return false;
        }
        self.table.insert(entry, self.next_code);
        self.next_code += 1;

        // Increase code size when needed
        if self.next_code >= (1 << self.code_size) && self.code_size < 12 {
            self.code_size += 1;
        }
        true
    }
}

/// Compress data using LZW
fn compress_lzw(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = LzwEncoder::new();
    let mut bit_buffer: u32 = 0;
    let mut bits_in_buffer: u8 = 0;

    let write_code = |output: &mut Vec<u8>,
                          bit_buffer: &mut u32,
                          bits_in_buffer: &mut u8,
                          code: u16,
                          code_size: u8| {
        *bit_buffer |= (code as u32) << *bits_in_buffer;
        *bits_in_buffer += code_size;

        while *bits_in_buffer >= 8 {
            output.push(*bit_buffer as u8);
            *bit_buffer >>= 8;
            *bits_in_buffer -= 8;
        }
    };

    // Write clear code
    write_code(
        &mut output,
        &mut bit_buffer,
        &mut bits_in_buffer,
        encoder.clear_code,
        encoder.code_size,
    );

    if data.is_empty() {
        write_code(
            &mut output,
            &mut bit_buffer,
            &mut bits_in_buffer,
            encoder.eoi_code,
            encoder.code_size,
        );
        if bits_in_buffer > 0 {
            output.push(bit_buffer as u8);
        }
        return output;
    }

    let mut current = vec![data[0]];

    for &byte in &data[1..] {
        let mut next = current.clone();
        next.push(byte);

        if encoder.table.contains_key(&next) {
            current = next;
        } else {
            // Output code for current
            let code = encoder.table[&current];
            write_code(
                &mut output,
                &mut bit_buffer,
                &mut bits_in_buffer,
                code,
                encoder.code_size,
            );

            // Add new entry
            if !encoder.add_entry(next) {
                // Table full, emit clear code and reset
                write_code(
                    &mut output,
                    &mut bit_buffer,
                    &mut bits_in_buffer,
                    encoder.clear_code,
                    encoder.code_size,
                );
                encoder.reset();
            }

            current = vec![byte];
        }
    }

    // Output final code
    let code = encoder.table[&current];
    write_code(
        &mut output,
        &mut bit_buffer,
        &mut bits_in_buffer,
        code,
        encoder.code_size,
    );

    // Write EOI
    write_code(
        &mut output,
        &mut bit_buffer,
        &mut bits_in_buffer,
        encoder.eoi_code,
        encoder.code_size,
    );

    // Flush remaining bits
    if bits_in_buffer > 0 {
        output.push(bit_buffer as u8);
    }

    output
}

/// Horizontal differencing predictor
pub fn apply_horizontal_predictor(data: &mut [u8], width: usize, samples_per_pixel: usize) {
    let row_bytes = width * samples_per_pixel;
    for row_start in (0..data.len()).step_by(row_bytes) {
        let row_end = (row_start + row_bytes).min(data.len());
        let row = &mut data[row_start..row_end];

        // Apply differencing from right to left
        for i in (samples_per_pixel..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - samples_per_pixel]);
        }
    }
}

/// Reverse horizontal differencing predictor
pub fn reverse_horizontal_predictor(data: &mut [u8], width: usize, samples_per_pixel: usize) {
    let row_bytes = width * samples_per_pixel;
    for row_start in (0..data.len()).step_by(row_bytes) {
        let row_end = (row_start + row_bytes).min(data.len());
        let row = &mut data[row_start..row_end];

        // Reverse differencing from left to right
        for i in samples_per_pixel..row.len() {
            row[i] = row[i].wrapping_add(row[i - samples_per_pixel]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_values() {
        assert_eq!(Compression::from_u16(1), Some(Compression::None));
        assert_eq!(Compression::from_u16(5), Some(Compression::Lzw));
        assert_eq!(Compression::from_u16(32773), Some(Compression::PackBits));
    }

    #[test]
    fn test_packbits_roundtrip() {
        let data = vec![1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5];
        let compressed = compress_packbits(&data);
        let decompressed = decompress_packbits(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_packbits_literal() {
        let data = vec![1, 2, 3, 4, 5];
        let compressed = compress_packbits(&data);
        let decompressed = decompress_packbits(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_packbits_run() {
        let data = vec![42; 100];
        let compressed = compress_packbits(&data);
        assert!(compressed.len() < data.len());
        let decompressed = decompress_packbits(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_lzw_roundtrip() {
        let data = b"TOBEORNOTTOBEORTOBEORNOT".to_vec();
        let compressed = compress_lzw(&data);
        let decompressed = decompress_lzw(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_lzw_repetitive() {
        let data = vec![65; 1000]; // Highly repetitive
        let compressed = compress_lzw(&data);
        assert!(compressed.len() < data.len());
        let decompressed = decompress_lzw(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_horizontal_predictor() {
        let original = vec![10, 20, 30, 40, 50, 60];
        let mut data = original.clone();

        apply_horizontal_predictor(&mut data, 3, 2);
        reverse_horizontal_predictor(&mut data, 3, 2);

        assert_eq!(data, original);
    }
}
