//! Huffman decoding for MP3 spectral data.

use transcode_core::bitstream::BitReader;
use transcode_core::error::Result;

/// Huffman table entry.
#[derive(Debug, Clone, Copy)]
pub struct HuffEntry {
    /// X value.
    pub x: i8,
    /// Y value.
    pub y: i8,
    /// Code length in bits.
    pub len: u8,
}

/// Huffman table for quad values (count1).
#[derive(Debug, Clone, Copy)]
pub struct QuadEntry {
    /// V value.
    pub v: i8,
    /// W value.
    pub w: i8,
    /// X value.
    pub x: i8,
    /// Y value.
    pub y: i8,
    /// Code length in bits.
    pub len: u8,
}

/// MP3 Huffman decoder.
pub struct Mp3Huffman;

impl Mp3Huffman {
    /// Maximum linbits value per table.
    const LINBITS: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9, 11, 13,
    ];

    /// Decode big values region.
    pub fn decode_big_values(
        reader: &mut BitReader<'_>,
        table_select: u8,
        count: usize,
        output: &mut [i32],
    ) -> Result<()> {
        if table_select == 0 {
            // Zero table
            output[..count * 2].fill(0);
            return Ok(());
        }

        let linbits = Self::LINBITS[table_select as usize];

        for i in 0..count {
            let (mut x, mut y) = Self::decode_pair(reader, table_select)?;

            // Handle linbits for large values
            if x == 15 && linbits > 0 {
                x += reader.read_bits(linbits)? as i32;
            }
            if x != 0 && reader.read_bit()? {
                x = -x;
            }

            if y == 15 && linbits > 0 {
                y += reader.read_bits(linbits)? as i32;
            }
            if y != 0 && reader.read_bit()? {
                y = -y;
            }

            output[i * 2] = x;
            output[i * 2 + 1] = y;
        }

        Ok(())
    }

    /// Decode count1 region (quadruples).
    pub fn decode_count1(
        reader: &mut BitReader<'_>,
        table_select: bool,
        count: usize,
        output: &mut [i32],
    ) -> Result<usize> {
        let mut decoded = 0;

        for i in 0..count {
            let (v, w, x, y) = Self::decode_quad(reader, table_select)?;

            let base = i * 4;
            if base + 3 >= output.len() {
                break;
            }

            output[base] = v;
            output[base + 1] = w;
            output[base + 2] = x;
            output[base + 3] = y;

            decoded += 4;
        }

        Ok(decoded)
    }

    /// Decode a pair of values from the specified table.
    fn decode_pair(reader: &mut BitReader<'_>, table: u8) -> Result<(i32, i32)> {
        // Simplified table decoding
        // Real implementation would use full Huffman tables

        match table {
            0 => Ok((0, 0)),
            1..=3 => {
                // Small values table (max 1)
                let code = reader.read_bits(4)? as u8;
                let x = ((code >> 2) & 1) as i32;
                let y = (code & 1) as i32;
                Ok((x, y))
            }
            4..=6 => {
                // Medium values table (max 3)
                let code = reader.read_bits(6)? as u8;
                let x = ((code >> 3) & 3) as i32;
                let y = (code & 3) as i32;
                Ok((x, y))
            }
            7..=9 => {
                // Larger values table (max 7)
                let code = reader.read_bits(8)? as u8;
                let x = ((code >> 4) & 7) as i32;
                let y = (code & 7) as i32;
                Ok((x, y))
            }
            10..=15 => {
                // Large values table (max 15)
                let code = reader.read_bits(8)? as u8;
                let x = ((code >> 4) & 15) as i32;
                let y = (code & 15) as i32;
                Ok((x, y))
            }
            16..=31 => {
                // ESC tables with linbits
                let code = reader.read_bits(8)? as u8;
                let x = ((code >> 4) & 15) as i32;
                let y = (code & 15) as i32;
                Ok((x, y))
            }
            _ => Ok((0, 0)),
        }
    }

    /// Decode a quadruple of values.
    fn decode_quad(reader: &mut BitReader<'_>, table_b: bool) -> Result<(i32, i32, i32, i32)> {
        if table_b {
            // Table B: 4 bits, one bit per value
            let code = reader.read_bits(4)? as u8;

            let mut v = ((code >> 3) & 1) as i32;
            let mut w = ((code >> 2) & 1) as i32;
            let mut x = ((code >> 1) & 1) as i32;
            let mut y = (code & 1) as i32;

            // Sign bits
            if v != 0 && reader.read_bit()? {
                v = -v;
            }
            if w != 0 && reader.read_bit()? {
                w = -w;
            }
            if x != 0 && reader.read_bit()? {
                x = -x;
            }
            if y != 0 && reader.read_bit()? {
                y = -y;
            }

            Ok((v, w, x, y))
        } else {
            // Table A: VLC coded
            let mut v = 0i32;
            let mut w = 0i32;
            let mut x = 0i32;
            let mut y = 0i32;

            // Read VLC code
            let code = reader.read_bits(6)? as u8;

            // Decode (simplified)
            match code >> 4 {
                0 => {}
                1 => v = 1,
                2 => w = 1,
                3 => {
                    v = 1;
                    w = 1
                }
                _ => {}
            }

            match (code >> 2) & 3 {
                1 => x = 1,
                2 => y = 1,
                3 => {
                    x = 1;
                    y = 1
                }
                _ => {}
            }

            // Sign bits
            if v != 0 && reader.read_bit()? {
                v = -v;
            }
            if w != 0 && reader.read_bit()? {
                w = -w;
            }
            if x != 0 && reader.read_bit()? {
                x = -x;
            }
            if y != 0 && reader.read_bit()? {
                y = -y;
            }

            Ok((v, w, x, y))
        }
    }
}

/// Calculate region boundaries for Huffman decoding.
pub fn calculate_regions(
    big_values: u16,
    region0_count: u8,
    region1_count: u8,
    block_type: u8,
    sfb_table: &[u16],
) -> (usize, usize, usize) {
    let big_values_end = (big_values * 2) as usize;

    if block_type == 2 {
        // Short blocks
        let region1_start = 36;
        let region2_start = big_values_end;
        (region1_start, region2_start, big_values_end)
    } else {
        // Long blocks
        let region1_start = sfb_table[(region0_count + 1) as usize] as usize;
        let region2_start = sfb_table[(region0_count + region1_count + 2) as usize] as usize;

        (
            region1_start.min(big_values_end),
            region2_start.min(big_values_end),
            big_values_end,
        )
    }
}
