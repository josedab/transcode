//! VP8L transform operations
//!
//! VP8L supports four types of transforms:
//! 1. Predictor transform
//! 2. Color transform
//! 3. Subtract green transform
//! 4. Color indexing transform

use crate::error::Result;

/// Transform types in VP8L
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    Predictor,
    CrossColor,
    SubtractGreen,
    ColorIndexing,
}

/// A transform to be applied to the image
#[derive(Debug, Clone)]
pub struct Transform {
    pub transform_type: TransformType,
    pub bits: u32,
    pub data: Vec<u32>,
}

/// Apply the subtract green transform
pub fn apply_subtract_green(pixels: &mut [u32]) {
    for pixel in pixels.iter_mut() {
        let g = ((*pixel >> 8) & 0xFF) as u8;
        let r = (((*pixel >> 16) & 0xFF) as u8).wrapping_add(g);
        let b = (((*pixel) & 0xFF) as u8).wrapping_add(g);
        *pixel = (*pixel & 0xFF00FF00) | (u32::from(r) << 16) | u32::from(b);
    }
}

/// Apply the predictor transform
pub fn apply_predictor_transform(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    bits: u32,
    transform_data: &[u32],
) -> Result<()> {
    let block_size = 1u32 << bits;
    let blocks_per_row = width.div_ceil(block_size);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            // Get predictor mode from transform data
            let block_x = x / block_size;
            let block_y = y / block_size;
            let block_idx = (block_y * blocks_per_row + block_x) as usize;
            let mode = if block_idx < transform_data.len() {
                ((transform_data[block_idx] >> 8) & 0xF) as u8
            } else {
                0
            };

            // Get neighboring pixels
            let left = if x > 0 { pixels[idx - 1] } else { 0xFF000000 };
            let top = if y > 0 { pixels[idx - width as usize] } else { 0xFF000000 };
            let top_left = if x > 0 && y > 0 {
                pixels[idx - width as usize - 1]
            } else if y > 0 {
                pixels[idx - width as usize]
            } else if x > 0 {
                pixels[idx - 1]
            } else {
                0xFF000000
            };
            let top_right = if x + 1 < width && y > 0 {
                pixels[idx - width as usize + 1]
            } else {
                top
            };

            let predicted = predict_pixel(mode, left, top, top_left, top_right);

            // Add prediction to residual
            pixels[idx] = add_pixels(pixels[idx], predicted);
        }
    }

    Ok(())
}

/// Predict a pixel based on neighbors and mode
fn predict_pixel(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        0 => 0xFF000000,                    // Black
        1 => left,                          // Left
        2 => top,                           // Top
        3 => top_right,                     // Top-right
        4 => top_left,                      // Top-left
        5 => average2(average2(left, top_right), top), // Average of (L+TR)/2 and T
        6 => average2(left, top_left),      // Average of L and TL
        7 => average2(left, top),           // Average of L and T
        8 => average2(top_left, top),       // Average of TL and T
        9 => average2(top, top_right),      // Average of T and TR
        10 => average2(average2(left, top_left), average2(top, top_right)),
        11 => select(left, top, top_left),  // Select predictor
        12 => clamp_add_subtract_full(left, top, top_left),
        13 => clamp_add_subtract_half(average2(left, top), top_left),
        _ => 0xFF000000,
    }
}

/// Add two pixels component-wise with wrapping
fn add_pixels(a: u32, b: u32) -> u32 {
    let a_bytes = a.to_le_bytes();
    let b_bytes = b.to_le_bytes();

    let result = [
        a_bytes[0].wrapping_add(b_bytes[0]),
        a_bytes[1].wrapping_add(b_bytes[1]),
        a_bytes[2].wrapping_add(b_bytes[2]),
        a_bytes[3].wrapping_add(b_bytes[3]),
    ];

    u32::from_le_bytes(result)
}

/// Average of two pixels
fn average2(a: u32, b: u32) -> u32 {
    let a_bytes = a.to_le_bytes();
    let b_bytes = b.to_le_bytes();

    let result = [
        ((u16::from(a_bytes[0]) + u16::from(b_bytes[0])) / 2) as u8,
        ((u16::from(a_bytes[1]) + u16::from(b_bytes[1])) / 2) as u8,
        ((u16::from(a_bytes[2]) + u16::from(b_bytes[2])) / 2) as u8,
        ((u16::from(a_bytes[3]) + u16::from(b_bytes[3])) / 2) as u8,
    ];

    u32::from_le_bytes(result)
}

/// Select predictor
fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let l = left.to_le_bytes();
    let t = top.to_le_bytes();
    let tl = top_left.to_le_bytes();

    let pa = (i32::from(t[0]) - i32::from(tl[0])).abs()
        + (i32::from(t[1]) - i32::from(tl[1])).abs()
        + (i32::from(t[2]) - i32::from(tl[2])).abs()
        + (i32::from(t[3]) - i32::from(tl[3])).abs();

    let pb = (i32::from(l[0]) - i32::from(tl[0])).abs()
        + (i32::from(l[1]) - i32::from(tl[1])).abs()
        + (i32::from(l[2]) - i32::from(tl[2])).abs()
        + (i32::from(l[3]) - i32::from(tl[3])).abs();

    if pa <= pb { left } else { top }
}

/// Clamp add subtract full
fn clamp_add_subtract_full(a: u32, b: u32, c: u32) -> u32 {
    let a_bytes = a.to_le_bytes();
    let b_bytes = b.to_le_bytes();
    let c_bytes = c.to_le_bytes();

    let result = [
        clamp_byte(i32::from(a_bytes[0]) + i32::from(b_bytes[0]) - i32::from(c_bytes[0])),
        clamp_byte(i32::from(a_bytes[1]) + i32::from(b_bytes[1]) - i32::from(c_bytes[1])),
        clamp_byte(i32::from(a_bytes[2]) + i32::from(b_bytes[2]) - i32::from(c_bytes[2])),
        clamp_byte(i32::from(a_bytes[3]) + i32::from(b_bytes[3]) - i32::from(c_bytes[3])),
    ];

    u32::from_le_bytes(result)
}

/// Clamp add subtract half
fn clamp_add_subtract_half(avg: u32, c: u32) -> u32 {
    let avg_bytes = avg.to_le_bytes();
    let c_bytes = c.to_le_bytes();

    let result = [
        clamp_byte(i32::from(avg_bytes[0]) + (i32::from(avg_bytes[0]) - i32::from(c_bytes[0])) / 2),
        clamp_byte(i32::from(avg_bytes[1]) + (i32::from(avg_bytes[1]) - i32::from(c_bytes[1])) / 2),
        clamp_byte(i32::from(avg_bytes[2]) + (i32::from(avg_bytes[2]) - i32::from(c_bytes[2])) / 2),
        clamp_byte(i32::from(avg_bytes[3]) + (i32::from(avg_bytes[3]) - i32::from(c_bytes[3])) / 2),
    ];

    u32::from_le_bytes(result)
}

/// Clamp value to byte range
fn clamp_byte(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// Apply the cross-color (color) transform
pub fn apply_cross_color_transform(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    bits: u32,
    transform_data: &[u32],
) -> Result<()> {
    let block_size = 1u32 << bits;
    let blocks_per_row = width.div_ceil(block_size);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            let block_x = x / block_size;
            let block_y = y / block_size;
            let block_idx = (block_y * blocks_per_row + block_x) as usize;

            if block_idx >= transform_data.len() {
                continue;
            }

            let transform = transform_data[block_idx];

            // Extract color transform coefficients
            let green_to_red = ((transform >> 8) & 0xFF) as i8;
            let green_to_blue = ((transform >> 16) & 0xFF) as i8;
            let red_to_blue = ((transform >> 24) & 0xFF) as i8;

            let pixel = pixels[idx];
            let b = (pixel & 0xFF) as u8;
            let g = ((pixel >> 8) & 0xFF) as u8;
            let r = ((pixel >> 16) & 0xFF) as u8;
            let a = ((pixel >> 24) & 0xFF) as u8;

            // Apply inverse color transform
            let new_r = r.wrapping_add(color_transform_delta(green_to_red, g));
            let new_b = b.wrapping_add(color_transform_delta(green_to_blue, g))
                .wrapping_add(color_transform_delta(red_to_blue, new_r));

            pixels[idx] = u32::from(new_b)
                | (u32::from(g) << 8)
                | (u32::from(new_r) << 16)
                | (u32::from(a) << 24);
        }
    }

    Ok(())
}

/// Calculate color transform delta
fn color_transform_delta(t: i8, c: u8) -> u8 {
    ((i32::from(t) * i32::from(c as i8)) >> 5) as u8
}

/// Apply color indexing transform (palette)
pub fn apply_color_indexing_transform(
    pixels: &mut Vec<u32>,
    width: u32,
    height: u32,
    bits: u32,
    palette: &[u32],
) -> Result<(u32, u32)> {
    let pixels_per_byte = 1u32 << bits;
    let new_width = width.div_ceil(pixels_per_byte);

    if bits == 0 {
        // Simple palette lookup
        for pixel in pixels.iter_mut() {
            let idx = (*pixel >> 8) & 0xFF;
            if (idx as usize) < palette.len() {
                *pixel = palette[idx as usize];
            }
        }
    } else {
        // Packed pixels
        let mut new_pixels = Vec::with_capacity((new_width * height) as usize);

        for y in 0..height {
            for x in 0..new_width {
                let src_idx = (y * new_width + x) as usize;
                let packed = if src_idx < pixels.len() {
                    pixels[src_idx]
                } else {
                    0
                };

                for i in 0..pixels_per_byte {
                    let dst_x = x * pixels_per_byte + i;
                    if dst_x < width {
                        let shift = i * (8 / pixels_per_byte);
                        let idx = ((packed >> (8 + shift)) & ((1 << (8 / pixels_per_byte)) - 1)) as usize;
                        let color = if idx < palette.len() {
                            palette[idx]
                        } else {
                            0xFF000000
                        };
                        new_pixels.push(color);
                    }
                }
            }
        }

        *pixels = new_pixels;
    }

    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_pixels() {
        let a = 0x10203040u32;
        let b = 0x01020304u32;
        let result = add_pixels(a, b);
        assert_eq!(result, 0x11223344);
    }

    #[test]
    fn test_average2() {
        let a = 0x00000000u32;
        let b = 0x20202020u32;
        let result = average2(a, b);
        assert_eq!(result, 0x10101010);
    }

    #[test]
    fn test_clamp_byte() {
        assert_eq!(clamp_byte(-10), 0);
        assert_eq!(clamp_byte(128), 128);
        assert_eq!(clamp_byte(300), 255);
    }

    #[test]
    fn test_subtract_green() {
        let mut pixels = vec![0xFF808080u32]; // Gray pixel
        apply_subtract_green(&mut pixels);
        // After subtract green: R += G, B += G
        // R = 0x80 + 0x80 = 0x00 (wrapping)
        // B = 0x80 + 0x80 = 0x00 (wrapping)
        assert_eq!(pixels[0] & 0xFF, 0x00);
        assert_eq!((pixels[0] >> 16) & 0xFF, 0x00);
    }
}
