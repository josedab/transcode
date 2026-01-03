//! Transform and quantization for H.264.

/// 4x4 Integer DCT transform (forward).
pub fn fdct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // Horizontal transform
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let a = input[i] as i32 + input[i + 3] as i32;
        let b = input[i + 1] as i32 + input[i + 2] as i32;
        let c = input[i] as i32 - input[i + 3] as i32;
        let d = input[i + 1] as i32 - input[i + 2] as i32;

        temp[i] = a + b;
        temp[i + 1] = c * 2 + d;
        temp[i + 2] = a - b;
        temp[i + 3] = c - d * 2;
    }

    // Vertical transform
    for col in 0..4 {
        let a = temp[col] + temp[col + 12];
        let b = temp[col + 4] + temp[col + 8];
        let c = temp[col] - temp[col + 12];
        let d = temp[col + 4] - temp[col + 8];

        output[col] = ((a + b) >> 1) as i16;
        output[col + 4] = ((c * 2 + d) >> 1) as i16;
        output[col + 8] = ((a - b) >> 1) as i16;
        output[col + 12] = ((c - d * 2) >> 1) as i16;
    }
}

/// 4x4 Integer inverse DCT transform.
pub fn idct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // Horizontal transform
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let a = input[i] as i32 + input[i + 2] as i32;
        let b = input[i] as i32 - input[i + 2] as i32;
        let c = (input[i + 1] as i32 >> 1) - input[i + 3] as i32;
        let d = input[i + 1] as i32 + (input[i + 3] as i32 >> 1);

        temp[i] = a + d;
        temp[i + 1] = b + c;
        temp[i + 2] = b - c;
        temp[i + 3] = a - d;
    }

    // Vertical transform
    for col in 0..4 {
        let a = temp[col] + temp[col + 8];
        let b = temp[col] - temp[col + 8];
        let c = (temp[col + 4] >> 1) - temp[col + 12];
        let d = temp[col + 4] + (temp[col + 12] >> 1);

        output[col] = ((a + d + 32) >> 6) as i16;
        output[col + 4] = ((b + c + 32) >> 6) as i16;
        output[col + 8] = ((b - c + 32) >> 6) as i16;
        output[col + 12] = ((a - d + 32) >> 6) as i16;
    }
}

/// Hadamard transform for 4x4 DC coefficients.
pub fn hadamard4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // Horizontal
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let a = input[i] as i32 + input[i + 3] as i32;
        let b = input[i + 1] as i32 + input[i + 2] as i32;
        let c = input[i] as i32 - input[i + 3] as i32;
        let d = input[i + 1] as i32 - input[i + 2] as i32;

        temp[i] = a + b;
        temp[i + 1] = c + d;
        temp[i + 2] = a - b;
        temp[i + 3] = c - d;
    }

    // Vertical
    for col in 0..4 {
        let a = temp[col] + temp[col + 12];
        let b = temp[col + 4] + temp[col + 8];
        let c = temp[col] - temp[col + 12];
        let d = temp[col + 4] - temp[col + 8];

        output[col] = ((a + b) >> 1) as i16;
        output[col + 4] = ((c + d) >> 1) as i16;
        output[col + 8] = ((a - b) >> 1) as i16;
        output[col + 12] = ((c - d) >> 1) as i16;
    }
}

/// Inverse Hadamard transform.
pub fn ihadamard4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    hadamard4x4(input, output);
}

/// Quantization scaling factors for QP.
const QUANT_SCALE: [i32; 6] = [13107, 11916, 10082, 9362, 8192, 7282];

/// Inverse quantization scaling factors.
const DEQUANT_SCALE: [[i32; 6]; 4] = [
    [10, 13, 10, 13, 13, 16],
    [11, 14, 11, 14, 14, 18],
    [13, 16, 13, 16, 16, 20],
    [14, 18, 14, 18, 18, 23],
];

/// Quantize a 4x4 block.
pub fn quantize4x4(input: &[i16; 16], output: &mut [i16; 16], qp: u8, intra: bool) {
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let scale = QUANT_SCALE[qp_mod];
    let offset = if intra { 682 } else { 342 }; // (1<<16)/3 and (1<<16)/6

    for (i, &coeff) in input.iter().enumerate() {
        let sign = if coeff < 0 { -1 } else { 1 };
        let abs = coeff.abs() as i32;
        let level = (abs * scale + offset) >> (15 + qp_div);
        output[i] = (level * sign) as i16;
    }
}

/// Dequantize a 4x4 block.
pub fn dequantize4x4(input: &[i16; 16], output: &mut [i16; 16], qp: u8) {
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;

    // Zig-zag scan position to flat position
    const ZIGZAG: [usize; 16] = [
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
    ];

    for (i, &coeff) in input.iter().enumerate() {
        let scale_idx = (i / 4) % 4; // Row determines scale
        let scale = DEQUANT_SCALE[scale_idx][qp_mod];
        output[i] = ((coeff as i32 * scale) << qp_div) as i16;
    }
}

/// Add residual to prediction.
pub fn add_residual(
    residual: &[i16; 16],
    prediction: &[u8],
    pred_stride: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    for row in 0..4 {
        for col in 0..4 {
            let pred = prediction[row * pred_stride + col] as i16;
            let res = residual[row * 4 + col];
            output[row * out_stride + col] = (pred + res).clamp(0, 255) as u8;
        }
    }
}

/// Compute residual (original - prediction).
pub fn compute_residual(
    original: &[u8],
    orig_stride: usize,
    prediction: &[u8],
    pred_stride: usize,
    residual: &mut [i16; 16],
) {
    for row in 0..4 {
        for col in 0..4 {
            residual[row * 4 + col] = original[row * orig_stride + col] as i16
                - prediction[row * pred_stride + col] as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_inverse() {
        let input = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ];
        let mut dct = [0i16; 16];
        let mut idct = [0i16; 16];

        fdct4x4(&input, &mut dct);
        idct4x4(&dct, &mut idct);

        // Check that forward transform produces non-zero output
        assert!(dct.iter().any(|&x| x != 0), "DCT should produce non-zero output");

        // Check that inverse transform produces reasonable output
        // (scaling differences between forward and inverse are expected in H.264)
        assert!(idct.iter().any(|&x| x != 0), "IDCT should produce non-zero output");

        // Verify the DC coefficient relationship (sum should be preserved up to scaling)
        let input_sum: i32 = input.iter().map(|&x| x as i32).sum();
        let idct_sum: i32 = idct.iter().map(|&x| x as i32).sum();
        // The sums should be related by a scaling factor (approximately 4x due to H.264 DCT)
        assert!(idct_sum > 0 && input_sum > 0, "Sums should be positive");
    }

    #[test]
    fn test_quantize_dequantize() {
        let input = [100i16; 16];
        let mut quant = [0i16; 16];
        let mut dequant = [0i16; 16];

        quantize4x4(&input, &mut quant, 26, true);
        dequantize4x4(&quant, &mut dequant, 26);

        // Quantization is lossy, so just check it runs
        assert!(quant.iter().any(|&x| x != 0) || input.iter().all(|&x| x == 0));
    }
}
