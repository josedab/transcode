//! AAC lookup tables and constants.

/// Number of scalefactor bands for each sample rate.
pub const SFB_COUNT_LONG: [usize; 12] = [41, 41, 47, 49, 49, 51, 47, 47, 43, 43, 43, 40];
pub const SFB_COUNT_SHORT: [usize; 12] = [12, 12, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15];

/// Scalefactor band offsets for long windows at 44100 Hz.
pub const SFB_OFFSET_LONG_44100: [u16; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896,
    960, 1024, 0, 0, 0, 0, 0, 0,
];

/// Scalefactor band offsets for short windows at 44100 Hz.
pub const SFB_OFFSET_SHORT_44100: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128, 0,
];

/// Window shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WindowShape {
    Sine = 0,
    Kaiser = 1,
}

/// Window sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WindowSequence {
    #[default]
    OnlyLong = 0,
    LongStart = 1,
    EightShort = 2,
    LongStop = 3,
}

/// Scalefactor gain table (2^(sf/4)).
pub fn scalefactor_gain(sf: u8) -> f32 {
    // sf ranges from 0 to 255
    2.0f32.powf(0.25 * (sf as f32 - 100.0))
}

/// Inverse quantization: sign(x) * |x|^(4/3).
pub fn inverse_quantize(x: i16) -> f32 {
    let sign = if x < 0 { -1.0 } else { 1.0 };
    sign * (x.abs() as f32).powf(4.0 / 3.0)
}

/// Kaiser-Bessel derived window for 1024 samples.
pub fn kbd_window_1024() -> [f32; 1024] {
    let mut window = [0.0f32; 1024];
    let alpha = 4.0;

    // Compute Kaiser-Bessel window
    let mut sum = 0.0f32;
    let mut kb = [0.0f32; 513];

    for i in 0..=512 {
        let x = 4.0 * alpha * ((i as f32 / 512.0) * (1.0 - i as f32 / 512.0)).sqrt();
        kb[i] = bessel_i0(x);
        sum += kb[i];
    }

    // Normalize and compute KBD window
    let mut cumsum = 0.0f32;
    for i in 0..512 {
        cumsum += kb[i];
        window[i] = (cumsum / sum).sqrt();
    }
    for i in 512..1024 {
        window[i] = window[1023 - i];
    }

    window
}

/// Kaiser-Bessel derived window for 128 samples.
pub fn kbd_window_128() -> [f32; 128] {
    let mut window = [0.0f32; 128];
    let alpha = 6.0;

    let mut sum = 0.0f32;
    let mut kb = [0.0f32; 65];

    for i in 0..=64 {
        let x = 4.0 * alpha * ((i as f32 / 64.0) * (1.0 - i as f32 / 64.0)).sqrt();
        kb[i] = bessel_i0(x);
        sum += kb[i];
    }

    let mut cumsum = 0.0f32;
    for i in 0..64 {
        cumsum += kb[i];
        window[i] = (cumsum / sum).sqrt();
    }
    for i in 64..128 {
        window[i] = window[127 - i];
    }

    window
}

/// Sine window for 1024 samples.
pub fn sine_window_1024() -> [f32; 1024] {
    let mut window = [0.0f32; 1024];
    for i in 0..1024 {
        window[i] = ((i as f32 + 0.5) * std::f32::consts::PI / 1024.0).sin();
    }
    window
}

/// Sine window for 128 samples.
pub fn sine_window_128() -> [f32; 128] {
    let mut window = [0.0f32; 128];
    for i in 0..128 {
        window[i] = ((i as f32 + 0.5) * std::f32::consts::PI / 128.0).sin();
    }
    window
}

/// Modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x_half = x * 0.5;

    for k in 1..50 {
        term *= (x_half / k as f32).powi(2);
        sum += term;
        if term < 1e-10 {
            break;
        }
    }

    sum
}

/// Huffman codebook IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HuffmanCodebook {
    Zero = 0,
    Cb1 = 1,
    Cb2 = 2,
    Cb3 = 3,
    Cb4 = 4,
    Cb5 = 5,
    Cb6 = 6,
    Cb7 = 7,
    Cb8 = 8,
    Cb9 = 9,
    Cb10 = 10,
    /// Codebook 11 (also used for escape sequences)
    Cb11 = 11,
    Reserved = 12,
    NoiseHcb = 13,
    IntensityStereo = 14,
    IntensityOutOfPhase = 15,
}

impl HuffmanCodebook {
    /// Number of values per codeword.
    pub fn dimension(&self) -> u8 {
        match *self as u8 {
            1..=4 => 4,
            5..=10 => 2,
            11 => 2,
            _ => 0,
        }
    }

    /// Is signed codebook.
    pub fn is_signed(&self) -> bool {
        matches!(*self as u8, 1 | 2 | 5 | 6)
    }

    /// Maximum absolute value.
    pub fn max_value(&self) -> u8 {
        match *self as u8 {
            1 | 2 => 1,
            3 | 4 => 2,
            5 | 6 => 4,
            7 | 8 => 7,
            9 | 10 => 12,
            11 => 16,
            _ => 0,
        }
    }
}
