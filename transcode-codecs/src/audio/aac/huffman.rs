//! Huffman decoding for AAC spectral data.

use transcode_core::bitstream::BitReader;
use transcode_core::error::Result;

/// Huffman decoder for AAC.
pub struct HuffmanDecoder;

impl HuffmanDecoder {
    /// Decode spectral data using the specified codebook.
    pub fn decode_spectral(
        reader: &mut BitReader<'_>,
        codebook: u8,
        output: &mut [i16],
    ) -> Result<()> {
        match codebook {
            0 => {
                // Zero codebook - all zeros
                output.fill(0);
            }
            1..=2 => {
                // 4-dimensional, signed, max value 1
                Self::decode_quad_signed(reader, output, codebook)?;
            }
            3..=4 => {
                // 4-dimensional, unsigned, max value 2
                Self::decode_quad_unsigned(reader, output, codebook)?;
            }
            5..=6 => {
                // 2-dimensional, signed, max value 4
                Self::decode_pair_signed(reader, output, codebook)?;
            }
            7..=10 => {
                // 2-dimensional, unsigned
                Self::decode_pair_unsigned(reader, output, codebook)?;
            }
            11 => {
                // Escape codebook
                Self::decode_escape(reader, output)?;
            }
            _ => {
                // Reserved/special codebooks
                output.fill(0);
            }
        }

        Ok(())
    }

    /// Decode using 4-dimensional signed codebook.
    fn decode_quad_signed(reader: &mut BitReader<'_>, output: &mut [i16], cb: u8) -> Result<()> {
        for chunk in output.chunks_exact_mut(4) {
            let code = Self::read_vlc(reader, cb)?;

            // Decode 4 values from the code
            let mut values = Self::decode_quad_code(code, true);

            // Read sign bits for non-zero values
            for val in &mut values {
                if *val != 0 && reader.read_bit()? {
                    *val = -*val;
                }
            }

            chunk.copy_from_slice(&values);
        }

        Ok(())
    }

    /// Decode using 4-dimensional unsigned codebook.
    fn decode_quad_unsigned(reader: &mut BitReader<'_>, output: &mut [i16], cb: u8) -> Result<()> {
        for chunk in output.chunks_exact_mut(4) {
            let code = Self::read_vlc(reader, cb)?;
            let values = Self::decode_quad_code(code, false);
            chunk.copy_from_slice(&values);
        }

        Ok(())
    }

    /// Decode using 2-dimensional signed codebook.
    fn decode_pair_signed(reader: &mut BitReader<'_>, output: &mut [i16], cb: u8) -> Result<()> {
        for chunk in output.chunks_exact_mut(2) {
            let code = Self::read_vlc(reader, cb)?;
            let mut values = Self::decode_pair_code(code, cb);

            for val in &mut values {
                if *val != 0 && reader.read_bit()? {
                    *val = -*val;
                }
            }

            chunk.copy_from_slice(&values);
        }

        Ok(())
    }

    /// Decode using 2-dimensional unsigned codebook.
    fn decode_pair_unsigned(reader: &mut BitReader<'_>, output: &mut [i16], cb: u8) -> Result<()> {
        for chunk in output.chunks_exact_mut(2) {
            let code = Self::read_vlc(reader, cb)?;
            let values = Self::decode_pair_code(code, cb);
            chunk.copy_from_slice(&values);
        }

        Ok(())
    }

    /// Decode escape codebook (11).
    fn decode_escape(reader: &mut BitReader<'_>, output: &mut [i16]) -> Result<()> {
        for chunk in output.chunks_exact_mut(2) {
            let code = Self::read_vlc(reader, 11)?;
            let mut values = Self::decode_pair_code(code, 11);

            // Read sign bits
            for val in &mut values {
                if *val != 0 && reader.read_bit()? {
                    *val = -*val;
                }
            }

            // Handle escape sequences for values == 16
            for val in &mut values {
                if val.abs() == 16 {
                    let sign = if *val < 0 { -1 } else { 1 };
                    let esc_val = Self::read_escape_value(reader)?;
                    *val = sign * esc_val;
                }
            }

            chunk.copy_from_slice(&values);
        }

        Ok(())
    }

    /// Read escape value (exponential-Golomb-like).
    fn read_escape_value(reader: &mut BitReader<'_>) -> Result<i16> {
        // Count leading ones
        let mut n = 4;
        while reader.read_bit()? {
            n += 1;
            if n > 16 {
                break;
            }
        }

        // Read n bits
        let value = reader.read_bits(n)? as i16;
        Ok(value + (1 << n))
    }

    /// Read variable-length code from bitstream.
    fn read_vlc(reader: &mut BitReader<'_>, _codebook: u8) -> Result<u16> {
        // Simplified VLC decoding
        // Real implementation would use proper Huffman tables

        let mut code = 0u16;
        let mut length = 0;

        // Read up to 21 bits (max VLC length in AAC)
        while length < 21 {
            let bit = reader.read_bit()? as u16;
            code = (code << 1) | bit;
            length += 1;

            // Check against codebook tables (simplified)
            // Real implementation would do table lookup here
            if length >= 1 && code == 0 {
                return Ok(0);
            }

            if length >= 4 {
                // Return a simplified code
                return Ok(code);
            }
        }

        Ok(code)
    }

    /// Decode 4-dimensional code into values.
    fn decode_quad_code(code: u16, _signed: bool) -> [i16; 4] {
        // Simplified decoding - real implementation uses lookup tables
        let mut values = [0i16; 4];

        // For signed codebooks (1,2), each value is -1, 0, or 1
        // For unsigned codebooks (3,4), each value is 0, 1, or 2
        values[0] = ((code >> 6) & 3) as i16;
        values[1] = ((code >> 4) & 3) as i16;
        values[2] = ((code >> 2) & 3) as i16;
        values[3] = (code & 3) as i16;

        values
    }

    /// Decode 2-dimensional code into values.
    fn decode_pair_code(code: u16, codebook: u8) -> [i16; 2] {
        // Simplified decoding
        let max_val = match codebook {
            5 | 6 => 4,
            7 | 8 => 7,
            9 | 10 => 12,
            11 => 16,
            _ => 4,
        };

        let modulo = (max_val + 1) as u16;
        [
            ((code / modulo) % modulo) as i16,
            (code % modulo) as i16,
        ]
    }
}

/// Huffman encoder for AAC.
pub struct HuffmanEncoder;

impl HuffmanEncoder {
    /// Estimate bits needed to encode spectral data.
    pub fn estimate_bits(coeffs: &[i16], codebook: u8) -> u32 {
        if codebook == 0 {
            return 0;
        }

        let mut bits = 0u32;

        match codebook {
            1..=4 => {
                // 4-dimensional codebook
                for chunk in coeffs.chunks(4) {
                    bits += Self::estimate_quad_bits(chunk, codebook);
                }
            }
            5..=11 => {
                // 2-dimensional codebook
                for chunk in coeffs.chunks(2) {
                    bits += Self::estimate_pair_bits(chunk, codebook);
                }
            }
            _ => {}
        }

        bits
    }

    fn estimate_quad_bits(values: &[i16], _codebook: u8) -> u32 {
        // Simplified estimation
        let max_abs = values.iter().map(|v| v.abs()).max().unwrap_or(0);

        if max_abs == 0 {
            1
        } else if max_abs <= 1 {
            4 + values.iter().filter(|&&v| v != 0).count() as u32
        } else {
            8 + values.iter().filter(|&&v| v != 0).count() as u32
        }
    }

    fn estimate_pair_bits(values: &[i16], codebook: u8) -> u32 {
        let max_abs = values.iter().map(|v| v.abs()).max().unwrap_or(0);

        let base_bits = match codebook {
            5 | 6 if max_abs <= 4 => 6,
            7 | 8 if max_abs <= 7 => 8,
            9 | 10 if max_abs <= 12 => 10,
            11 => {
                let mut bits = 10u32;
                for &v in values {
                    if v.abs() >= 16 {
                        bits += 8; // Escape code estimation
                    }
                }
                bits
            }
            _ => 12,
        };

        base_bits + values.iter().filter(|&&v| v != 0).count() as u32
    }

    /// Select best codebook for a section.
    pub fn select_codebook(coeffs: &[i16]) -> u8 {
        if coeffs.iter().all(|&c| c == 0) {
            return 0;
        }

        let max_abs = coeffs.iter().map(|v| v.abs()).max().unwrap_or(0);

        if max_abs <= 1 {
            1 // Signed quad, max 1
        } else if max_abs <= 2 {
            3 // Unsigned quad, max 2
        } else if max_abs <= 4 {
            5 // Signed pair, max 4
        } else if max_abs <= 7 {
            7 // Unsigned pair, max 7
        } else if max_abs <= 12 {
            9 // Unsigned pair, max 12
        } else {
            11 // Escape codebook
        }
    }
}
