//! Discrete Cosine Transform for JPEG.

use std::f32::consts::PI;

/// 8x8 DCT coefficient matrix (precomputed).
fn dct_coefficient(k: usize, n: usize) -> f32 {
    let c = if k == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
    c * (PI * (2 * n + 1) as f32 * k as f32 / 16.0).cos()
}

/// Forward 2D DCT on an 8x8 block.
pub fn forward_dct(block: &[i16; 64]) -> [i16; 64] {
    let mut temp = [0.0f32; 64];
    let mut output = [0i16; 64];

    // Row transform
    for row in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            let c_u = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };

            for x in 0..8 {
                let angle = PI * (2 * x + 1) as f32 * u as f32 / 16.0;
                sum += block[row * 8 + x] as f32 * angle.cos();
            }

            temp[row * 8 + u] = c_u * sum * 0.5;
        }
    }

    // Column transform
    for col in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0f32;
            let c_v = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };

            for y in 0..8 {
                let angle = PI * (2 * y + 1) as f32 * v as f32 / 16.0;
                sum += temp[y * 8 + col] * angle.cos();
            }

            output[v * 8 + col] = (c_v * sum * 0.5).round() as i16;
        }
    }

    output
}

/// Inverse 2D DCT on an 8x8 block.
pub fn inverse_dct(block: &[i16; 64]) -> [i16; 64] {
    let mut temp = [0.0f32; 64];
    let mut output = [0i16; 64];

    // Row transform
    for row in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;

            for u in 0..8 {
                let c_u = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                let angle = PI * (2 * x + 1) as f32 * u as f32 / 16.0;
                sum += c_u * block[row * 8 + u] as f32 * angle.cos();
            }

            temp[row * 8 + x] = sum * 0.5;
        }
    }

    // Column transform
    for col in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0f32;

            for v in 0..8 {
                let c_v = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                let angle = PI * (2 * y + 1) as f32 * v as f32 / 16.0;
                sum += c_v * temp[v * 8 + col] * angle.cos();
            }

            let val = (sum * 0.5).round();
            output[y * 8 + col] = val.clamp(-2048.0, 2047.0) as i16;
        }
    }

    output
}

/// Fast forward DCT using AAN algorithm (separable).
pub fn fast_forward_dct(block: &mut [i16; 64]) {
    // Constants for AAN algorithm
    const C1: f32 = 0.980785280;
    const C2: f32 = 0.923879533;
    const C3: f32 = 0.831469612;
    const C4: f32 = 0.707106781;
    const C5: f32 = 0.555570233;
    const C6: f32 = 0.382683433;
    const C7: f32 = 0.195090322;

    let mut workspace = [0.0f32; 64];

    // Copy to workspace
    for i in 0..64 {
        workspace[i] = block[i] as f32;
    }

    // Row pass
    for row in 0..8 {
        let offset = row * 8;
        let d0 = workspace[offset];
        let d1 = workspace[offset + 1];
        let d2 = workspace[offset + 2];
        let d3 = workspace[offset + 3];
        let d4 = workspace[offset + 4];
        let d5 = workspace[offset + 5];
        let d6 = workspace[offset + 6];
        let d7 = workspace[offset + 7];

        let tmp0 = d0 + d7;
        let tmp7 = d0 - d7;
        let tmp1 = d1 + d6;
        let tmp6 = d1 - d6;
        let tmp2 = d2 + d5;
        let tmp5 = d2 - d5;
        let tmp3 = d3 + d4;
        let tmp4 = d3 - d4;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        workspace[offset] = tmp10 + tmp11;
        workspace[offset + 4] = tmp10 - tmp11;

        let z1 = (tmp12 + tmp13) * C4;
        workspace[offset + 2] = tmp13 + z1;
        workspace[offset + 6] = tmp13 - z1;

        let tmp10 = tmp4 + tmp5;
        let tmp11 = tmp5 + tmp6;
        let tmp12 = tmp6 + tmp7;

        let z5 = (tmp10 - tmp12) * C6;
        let z2 = tmp10 * C2 + z5;
        let z4 = tmp12 * C3 + z5;
        let z3 = tmp11 * C4;

        let z11 = tmp7 + z3;
        let z13 = tmp7 - z3;

        workspace[offset + 5] = z13 + z2;
        workspace[offset + 3] = z13 - z2;
        workspace[offset + 1] = z11 + z4;
        workspace[offset + 7] = z11 - z4;
    }

    // Column pass
    for col in 0..8 {
        let d0 = workspace[col];
        let d1 = workspace[col + 8];
        let d2 = workspace[col + 16];
        let d3 = workspace[col + 24];
        let d4 = workspace[col + 32];
        let d5 = workspace[col + 40];
        let d6 = workspace[col + 48];
        let d7 = workspace[col + 56];

        let tmp0 = d0 + d7;
        let tmp7 = d0 - d7;
        let tmp1 = d1 + d6;
        let tmp6 = d1 - d6;
        let tmp2 = d2 + d5;
        let tmp5 = d2 - d5;
        let tmp3 = d3 + d4;
        let tmp4 = d3 - d4;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        block[col] = ((tmp10 + tmp11) / 8.0).round() as i16;
        block[col + 32] = ((tmp10 - tmp11) / 8.0).round() as i16;

        let z1 = (tmp12 + tmp13) * C4;
        block[col + 16] = ((tmp13 + z1) / 8.0).round() as i16;
        block[col + 48] = ((tmp13 - z1) / 8.0).round() as i16;

        let tmp10 = tmp4 + tmp5;
        let tmp11 = tmp5 + tmp6;
        let tmp12 = tmp6 + tmp7;

        let z5 = (tmp10 - tmp12) * C6;
        let z2 = tmp10 * C2 + z5;
        let z4 = tmp12 * C3 + z5;
        let z3 = tmp11 * C4;

        let z11 = tmp7 + z3;
        let z13 = tmp7 - z3;

        block[col + 40] = ((z13 + z2) / 8.0).round() as i16;
        block[col + 24] = ((z13 - z2) / 8.0).round() as i16;
        block[col + 8] = ((z11 + z4) / 8.0).round() as i16;
        block[col + 56] = ((z11 - z4) / 8.0).round() as i16;
    }
}

/// Fast inverse DCT using AAN algorithm.
pub fn fast_inverse_dct(block: &mut [i16; 64]) {
    const C4: f32 = 0.707106781;
    const C2: f32 = 0.541196100;
    const C6: f32 = 1.306562965;

    let mut workspace = [0.0f32; 64];

    // Dequantize and copy to workspace
    for i in 0..64 {
        workspace[i] = block[i] as f32;
    }

    // Column pass
    for col in 0..8 {
        let tmp0 = workspace[col];
        let tmp1 = workspace[col + 16];
        let tmp2 = workspace[col + 32];
        let tmp3 = workspace[col + 48];
        let tmp4 = workspace[col + 8];
        let tmp5 = workspace[col + 24];
        let tmp6 = workspace[col + 40];
        let tmp7 = workspace[col + 56];

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;
        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * C4 * 2.0 - tmp13;

        let d0 = tmp10 + tmp13;
        let d3 = tmp10 - tmp13;
        let d1 = tmp11 + tmp12;
        let d2 = tmp11 - tmp12;

        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;

        let tmp7 = z11 + z13;
        let tmp11 = (z11 - z13) * C4 * 2.0;

        let z5 = (z10 + z12) * C6;
        let tmp10 = z5 - z12 * C2 * 2.0;
        let tmp12 = z5 - z10 * C6 * 2.0;

        let tmp6 = tmp12 - tmp7;
        let tmp5 = tmp11 - tmp6;
        let tmp4 = tmp10 - tmp5;

        workspace[col] = d0 + tmp7;
        workspace[col + 56] = d0 - tmp7;
        workspace[col + 8] = d1 + tmp6;
        workspace[col + 48] = d1 - tmp6;
        workspace[col + 16] = d2 + tmp5;
        workspace[col + 40] = d2 - tmp5;
        workspace[col + 24] = d3 + tmp4;
        workspace[col + 32] = d3 - tmp4;
    }

    // Row pass
    for row in 0..8 {
        let offset = row * 8;
        let tmp0 = workspace[offset];
        let tmp1 = workspace[offset + 2];
        let tmp2 = workspace[offset + 4];
        let tmp3 = workspace[offset + 6];
        let tmp4 = workspace[offset + 1];
        let tmp5 = workspace[offset + 3];
        let tmp6 = workspace[offset + 5];
        let tmp7 = workspace[offset + 7];

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;
        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * C4 * 2.0 - tmp13;

        let d0 = tmp10 + tmp13;
        let d3 = tmp10 - tmp13;
        let d1 = tmp11 + tmp12;
        let d2 = tmp11 - tmp12;

        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;

        let tmp7 = z11 + z13;
        let tmp11 = (z11 - z13) * C4 * 2.0;

        let z5 = (z10 + z12) * C6;
        let tmp10 = z5 - z12 * C2 * 2.0;
        let tmp12 = z5 - z10 * C6 * 2.0;

        let tmp6 = tmp12 - tmp7;
        let tmp5 = tmp11 - tmp6;
        let tmp4 = tmp10 - tmp5;

        block[offset] = ((d0 + tmp7) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 7] = ((d0 - tmp7) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 1] = ((d1 + tmp6) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 6] = ((d1 - tmp6) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 2] = ((d2 + tmp5) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 5] = ((d2 - tmp5) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 3] = ((d3 + tmp4) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
        block[offset + 4] = ((d3 - tmp4) / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_roundtrip() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 4) - 128;
        }

        let dct = forward_dct(&input);
        let output = inverse_dct(&dct);

        // Check approximate roundtrip
        for i in 0..64 {
            assert!(
                (input[i] - output[i]).abs() < 10,
                "Mismatch at {}: {} vs {}",
                i,
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_dc_coefficient() {
        let mut input = [128i16; 64];
        let dct = forward_dct(&input);

        // DC coefficient should be sum / 8
        assert!(dct[0].abs() > 0);

        // AC coefficients should be near zero for constant input
        for i in 1..64 {
            assert!(dct[i].abs() < 2, "AC[{}] = {}", i, dct[i]);
        }
    }
}
