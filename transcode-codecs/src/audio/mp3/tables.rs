//! MP3 constant tables and lookup data.

#![allow(clippy::excessive_precision)]

use std::f32::consts::PI;

/// Scalefactor band boundaries for long blocks at 44100 Hz.
pub const SFB_LONG_44100: [u16; 23] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576,
];

/// Scalefactor band boundaries for short blocks at 44100 Hz.
pub const SFB_SHORT_44100: [u16; 14] = [
    0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136, 192,
];

/// Scalefactor band boundaries for long blocks at 48000 Hz.
pub const SFB_LONG_48000: [u16; 23] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576,
];

/// Scalefactor band boundaries for short blocks at 48000 Hz.
pub const SFB_SHORT_48000: [u16; 14] = [
    0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126, 192,
];

/// Scalefactor band boundaries for long blocks at 32000 Hz.
pub const SFB_LONG_32000: [u16; 23] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576,
];

/// Scalefactor band boundaries for short blocks at 32000 Hz.
pub const SFB_SHORT_32000: [u16; 14] = [
    0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180, 192,
];

/// Pretab values for scalefactor bands.
pub const PRETAB: [u8; 22] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0,
];

/// IMDCT window coefficients for long blocks.
pub fn imdct_window_long() -> [f32; 36] {
    let mut window = [0.0f32; 36];
    for i in 0..36 {
        window[i] = ((i as f32 + 0.5) * PI / 36.0).sin();
    }
    window
}

/// IMDCT window coefficients for short blocks.
pub fn imdct_window_short() -> [f32; 12] {
    let mut window = [0.0f32; 12];
    for i in 0..12 {
        window[i] = ((i as f32 + 0.5) * PI / 12.0).sin();
    }
    window
}

/// IMDCT window coefficients for start blocks.
pub fn imdct_window_start() -> [f32; 36] {
    let mut window = [0.0f32; 36];
    // First half: normal sine window
    for i in 0..18 {
        window[i] = ((i as f32 + 0.5) * PI / 36.0).sin();
    }
    // Transition region
    for i in 18..24 {
        window[i] = 1.0;
    }
    for i in 24..30 {
        window[i] = (((i - 18) as f32 + 0.5) * PI / 12.0).sin();
    }
    // Zeros at the end
    for i in 30..36 {
        window[i] = 0.0;
    }
    window
}

/// IMDCT window coefficients for stop blocks.
pub fn imdct_window_stop() -> [f32; 36] {
    let mut window = [0.0f32; 36];
    // Zeros at the start
    for i in 0..6 {
        window[i] = 0.0;
    }
    // Transition region
    for i in 6..12 {
        window[i] = (((i - 6) as f32 + 0.5) * PI / 12.0).sin();
    }
    for i in 12..18 {
        window[i] = 1.0;
    }
    // Second half: normal sine window
    for i in 18..36 {
        window[i] = ((i as f32 + 0.5) * PI / 36.0).sin();
    }
    window
}

/// Cosine coefficients for synthesis filterbank.
pub fn synthesis_cos_table() -> [[f32; 32]; 64] {
    let mut table = [[0.0f32; 32]; 64];

    for i in 0..64 {
        for j in 0..32 {
            let angle = PI / 64.0 * (2 * i + 1 + 32) as f32 * (2 * j + 1) as f32;
            table[i][j] = angle.cos();
        }
    }

    table
}

/// Synthesis filterbank window.
pub fn synthesis_window() -> [f32; 512] {
    // D[i] coefficients (ITU table)
    // This is a simplified version - real implementation uses the full table
    let mut window = [0.0f32; 512];

    for i in 0..512 {
        let n = i as f32;
        // Approximation of the prototype filter
        window[i] = ((n + 0.5) * PI / 512.0).sin() * 0.5;
    }

    window
}

/// Requantization power table (2^(1/4)).
pub fn requant_pow_table() -> [f32; 8192] {
    let mut table = [0.0f32; 8192];

    for i in 0..8192 {
        table[i] = (i as f32).powf(4.0 / 3.0);
    }

    table
}

/// Global gain adjustment table.
pub fn gain_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];

    for i in 0..256 {
        table[i] = 2.0f32.powf((i as f32 - 210.0) * 0.25);
    }

    table
}

/// Scalefactor band lengths for MPEG1 long blocks.
pub const SLEN1_TABLE: [u8; 16] = [0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4];
pub const SLEN2_TABLE: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3];

/// Number of scalefactor bands for short blocks.
pub const NR_OF_SFB_SHORT: [usize; 4] = [12, 13, 12, 12];

/// Number of scalefactor bands for long blocks.
pub const NR_OF_SFB_LONG: [usize; 4] = [21, 22, 22, 22];

/// Intensity stereo ratio table.
pub fn intensity_ratio_table() -> [f32; 16] {
    let mut table = [0.0f32; 16];

    for i in 0..16 {
        let pos = i as f32;
        table[i] = (pos / 16.0 * PI / 2.0).tan();
    }

    table
}

/// Alias reduction coefficients.
pub const ALIAS_CS: [f32; 8] = [
    0.857492926, 0.881741997, 0.949628649, 0.983314592,
    0.995517816, 0.999160558, 0.999899195, 0.999993155,
];

pub const ALIAS_CA: [f32; 8] = [
    -0.514495755, -0.471731969, -0.313377454, -0.181913200,
    -0.094574193, -0.040965583, -0.014198569, -0.003699975,
];

/// Subband mapping for reordering short blocks.
pub fn short_block_reorder_table() -> [[usize; 3]; 192] {
    let mut table = [[0usize; 3]; 192];

    // Mapping from window order to subband order
    let sfb_table = &SFB_SHORT_44100;

    let mut idx = 0;
    for sfb in 0..13 {
        let width = (sfb_table[sfb + 1] - sfb_table[sfb]) as usize;

        for win in 0..3 {
            for i in 0..width {
                let src = sfb_table[sfb] as usize + win * width + i;
                if idx < 192 {
                    table[idx] = [src, win, i];
                    idx += 1;
                }
            }
        }
    }

    table
}
