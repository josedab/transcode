//! VP8 transform functions (Walsh-Hadamard and DCT).

#![allow(dead_code)]

/// Apply inverse Walsh-Hadamard transform to 4x4 block.
pub fn iwht4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i16; 16];

    // Rows
    for i in 0..4 {
        let a = input[i * 4] + input[i * 4 + 3];
        let b = input[i * 4 + 1] + input[i * 4 + 2];
        let c = input[i * 4 + 1] - input[i * 4 + 2];
        let d = input[i * 4] - input[i * 4 + 3];

        tmp[i * 4] = a + b;
        tmp[i * 4 + 1] = c + d;
        tmp[i * 4 + 2] = a - b;
        tmp[i * 4 + 3] = d - c;
    }

    // Columns
    for i in 0..4 {
        let a = tmp[i] + tmp[12 + i];
        let b = tmp[4 + i] + tmp[8 + i];
        let c = tmp[4 + i] - tmp[8 + i];
        let d = tmp[i] - tmp[12 + i];

        output[i] = (a + b + 3) >> 3;
        output[4 + i] = (c + d + 3) >> 3;
        output[8 + i] = (a - b + 3) >> 3;
        output[12 + i] = (d - c + 3) >> 3;
    }
}

/// Apply forward Walsh-Hadamard transform to 4x4 block.
pub fn fwht4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i16; 16];

    // Rows
    for i in 0..4 {
        let a = input[i * 4] + input[i * 4 + 1];
        let b = input[i * 4 + 2] + input[i * 4 + 3];
        let c = input[i * 4] - input[i * 4 + 1];
        let d = input[i * 4 + 2] - input[i * 4 + 3];

        tmp[i * 4] = a + b;
        tmp[i * 4 + 1] = c + d;
        tmp[i * 4 + 2] = a - b;
        tmp[i * 4 + 3] = c - d;
    }

    // Columns
    for i in 0..4 {
        let a = tmp[i] + tmp[4 + i];
        let b = tmp[8 + i] + tmp[12 + i];
        let c = tmp[i] - tmp[4 + i];
        let d = tmp[8 + i] - tmp[12 + i];

        output[i] = (a + b) >> 1;
        output[4 + i] = (c + d) >> 1;
        output[8 + i] = (a - b) >> 1;
        output[12 + i] = (c - d) >> 1;
    }
}

/// Apply inverse DCT to 4x4 block.
pub fn idct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // DCT constants
    const C1: i32 = 20091; // cos(pi/8) * sqrt(2) * 16384
    const C2: i32 = 35468; // sin(pi/8) * sqrt(2) * 16384

    let mut tmp = [0i32; 16];

    // Rows (horizontal 1-D IDCT)
    for i in 0..4 {
        let a = input[i * 4] as i32 + input[i * 4 + 2] as i32;
        let b = input[i * 4] as i32 - input[i * 4 + 2] as i32;
        let c = ((input[i * 4 + 1] as i32 * C2) >> 16) - ((input[i * 4 + 3] as i32 * C1) >> 16);
        let d = ((input[i * 4 + 1] as i32 * C1) >> 16) + ((input[i * 4 + 3] as i32 * C2) >> 16);

        tmp[i * 4] = a + d;
        tmp[i * 4 + 1] = b + c;
        tmp[i * 4 + 2] = b - c;
        tmp[i * 4 + 3] = a - d;
    }

    // Columns (vertical 1-D IDCT)
    for i in 0..4 {
        let a = tmp[i] + tmp[8 + i];
        let b = tmp[i] - tmp[8 + i];
        let c = ((tmp[4 + i] * C2) >> 16) - ((tmp[12 + i] * C1) >> 16);
        let d = ((tmp[4 + i] * C1) >> 16) + ((tmp[12 + i] * C2) >> 16);

        // Round and shift
        output[i] = ((a + d + 4) >> 3) as i16;
        output[4 + i] = ((b + c + 4) >> 3) as i16;
        output[8 + i] = ((b - c + 4) >> 3) as i16;
        output[12 + i] = ((a - d + 4) >> 3) as i16;
    }
}

/// Apply forward DCT to 4x4 block.
pub fn fdct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    const C1: i32 = 20091;
    const C2: i32 = 35468;

    let mut tmp = [0i32; 16];

    // Rows
    for i in 0..4 {
        let a = input[i * 4] as i32 + input[i * 4 + 3] as i32;
        let b = input[i * 4 + 1] as i32 + input[i * 4 + 2] as i32;
        let c = input[i * 4 + 1] as i32 - input[i * 4 + 2] as i32;
        let d = input[i * 4] as i32 - input[i * 4 + 3] as i32;

        tmp[i * 4] = a + b;
        tmp[i * 4 + 2] = a - b;
        tmp[i * 4 + 1] = ((c * C2) >> 16) + ((d * C1) >> 16);
        tmp[i * 4 + 3] = ((d * C2) >> 16) - ((c * C1) >> 16);
    }

    // Columns
    for i in 0..4 {
        let a = tmp[i] + tmp[12 + i];
        let b = tmp[4 + i] + tmp[8 + i];
        let c = tmp[4 + i] - tmp[8 + i];
        let d = tmp[i] - tmp[12 + i];

        output[i] = ((a + b + 1) >> 1) as i16;
        output[8 + i] = ((a - b + 1) >> 1) as i16;
        output[4 + i] = (((c * C2) >> 16) + ((d * C1) >> 16)) as i16;
        output[12 + i] = (((d * C2) >> 16) - ((c * C1) >> 16)) as i16;
    }
}

/// Add prediction to residual and clamp to [0, 255].
pub fn add_residual(
    prediction: &[u8],
    residual: &[i16; 16],
    output: &mut [u8],
    stride: usize,
) {
    for y in 0..4 {
        for x in 0..4 {
            let pred = prediction[y * stride + x] as i16;
            let res = residual[y * 4 + x];
            let val = (pred + res).clamp(0, 255) as u8;
            output[y * stride + x] = val;
        }
    }
}

/// Subtract prediction from source.
pub fn sub_prediction(
    source: &[u8],
    prediction: &[u8],
    residual: &mut [i16; 16],
    stride: usize,
) {
    for y in 0..4 {
        for x in 0..4 {
            let src = source[y * stride + x] as i16;
            let pred = prediction[y * stride + x] as i16;
            residual[y * 4 + x] = src - pred;
        }
    }
}

/// Dequantize coefficients.
pub fn dequantize(
    coeffs: &mut [i16; 16],
    dc_quant: i16,
    ac_quant: i16,
) {
    coeffs[0] *= dc_quant;
    for coeff in coeffs.iter_mut().skip(1) {
        *coeff *= ac_quant;
    }
}

/// VP8 dequantization tables.
pub mod quant_tables {
    /// DC quantizer table.
    pub const DC_TABLE: [i16; 128] = [
        4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
        18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
        75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
        122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
    ];

    /// AC quantizer table.
    pub const AC_TABLE: [i16; 128] = [
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
        78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
        110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
        155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
        213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
    ];

    /// Get DC quantizer for given QP index.
    pub fn get_dc_quant(qp: u8) -> i16 {
        DC_TABLE[(qp as usize).min(127)]
    }

    /// Get AC quantizer for given QP index.
    pub fn get_ac_quant(qp: u8) -> i16 {
        AC_TABLE[(qp as usize).min(127)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct_zero_input() {
        let input = [0i16; 16];
        let mut output = [0i16; 16];
        idct4x4(&input, &mut output);
        assert!(output.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_idct_dc_only() {
        let mut input = [0i16; 16];
        input[0] = 100;
        let mut output = [0i16; 16];
        idct4x4(&input, &mut output);
        // DC should distribute evenly
        assert!(output.iter().all(|&x| x.abs() < 20));
    }

    #[test]
    fn test_wht_forward() {
        // Test that forward WHT produces expected DC coefficient
        let input: [i16; 16] = [100; 16]; // All same value
        let mut transformed = [0i16; 16];

        fwht4x4(&input, &mut transformed);

        // DC coefficient should be scaled sum
        // With all 100s, the forward transform should produce significant DC
        assert!(transformed[0].abs() > 0);
    }

    #[test]
    fn test_wht_inverse() {
        // Test that inverse WHT works on known input
        let mut input = [0i16; 16];
        input[0] = 800; // DC only
        let mut output = [0i16; 16];

        iwht4x4(&input, &mut output);

        // DC-only input should produce relatively uniform output
        // All values should be similar (DC distributed)
        let avg: i32 = output.iter().map(|&x| x as i32).sum::<i32>() / 16;
        assert!(avg > 0, "Average should be positive for positive DC input");
    }

    #[test]
    fn test_quant_tables() {
        assert_eq!(quant_tables::get_dc_quant(0), 4);
        assert_eq!(quant_tables::get_ac_quant(0), 4);
        assert!(quant_tables::get_dc_quant(127) > quant_tables::get_dc_quant(0));
    }
}
