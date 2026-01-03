//! Animation support for WebP (ANIM and ANMF chunks)
//!
//! Animated WebP files contain:
//! - ANIM chunk: Global animation parameters (background color, loop count)
//! - ANMF chunks: Individual frames with position, duration, and disposal info

use std::io::{Read, Seek};
use byteorder::{LittleEndian, ByteOrder};
use image::RgbaImage;

use crate::error::{WebPError, Result};
use crate::riff::{RiffContainer, ChunkType};
use crate::vp8::Vp8Decoder;
use crate::vp8l::Vp8lDecoder;
use crate::alpha::AlphaDecoder;

/// Animation decoder for animated WebP files
#[allow(dead_code)]
pub struct AnimationDecoder<'a> {
    data: &'a [u8],
    info: AnimationInfo,
}

/// Global animation information from ANIM chunk
#[derive(Debug, Clone, Copy, Default)]
pub struct AnimationInfo {
    /// Canvas width
    pub width: u32,
    /// Canvas height
    pub height: u32,
    /// Background color (BGRA format)
    pub background_color: [u8; 4],
    /// Number of times to loop (0 = infinite)
    pub loop_count: u16,
}

/// A single animation frame
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    /// X offset of this frame on the canvas
    pub x_offset: u32,
    /// Y offset of this frame on the canvas
    pub y_offset: u32,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Blending mode
    pub blending: BlendingMode,
    /// Disposal method
    pub disposal: DisposalMethod,
    /// Decoded frame image
    pub image: RgbaImage,
}

/// How to blend this frame with the previous canvas state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendingMode {
    /// Use alpha blending
    #[default]
    AlphaBlending,
    /// Do not blend, overwrite
    NoBlending,
}

impl BlendingMode {
    fn from_bit(bit: bool) -> Self {
        if bit {
            BlendingMode::NoBlending
        } else {
            BlendingMode::AlphaBlending
        }
    }
}

/// How to dispose of the frame after display
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisposalMethod {
    /// Do not dispose, leave as is
    #[default]
    None,
    /// Dispose to background color
    Background,
}

impl DisposalMethod {
    fn from_bit(bit: bool) -> Self {
        if bit {
            DisposalMethod::Background
        } else {
            DisposalMethod::None
        }
    }
}

/// ANMF chunk data
#[derive(Debug, Clone)]
pub struct AnmfChunk {
    /// X offset (in pixels)
    pub x_offset: u32,
    /// Y offset (in pixels)
    pub y_offset: u32,
    /// Frame width minus 1
    pub width: u32,
    /// Frame height minus 1
    pub height: u32,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Blending mode
    pub blending: BlendingMode,
    /// Disposal method
    pub disposal: DisposalMethod,
    /// Frame data (VP8/VP8L with optional ALPH)
    pub frame_data: Vec<u8>,
}

impl AnmfChunk {
    /// Parse an ANMF chunk
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(WebPError::InvalidAnimation("ANMF chunk too small".into()));
        }

        // X offset (24 bits)
        let x_offset = u32::from(data[0])
            | (u32::from(data[1]) << 8)
            | (u32::from(data[2]) << 16);
        // Stored as value * 2, so divide by 2
        let x_offset = x_offset * 2;

        // Y offset (24 bits)
        let y_offset = u32::from(data[3])
            | (u32::from(data[4]) << 8)
            | (u32::from(data[5]) << 16);
        let y_offset = y_offset * 2;

        // Width minus 1 (24 bits)
        let width = (u32::from(data[6])
            | (u32::from(data[7]) << 8)
            | (u32::from(data[8]) << 16))
            + 1;

        // Height minus 1 (24 bits)
        let height = (u32::from(data[9])
            | (u32::from(data[10]) << 8)
            | (u32::from(data[11]) << 16))
            + 1;

        // Duration (24 bits)
        let duration_ms = u32::from(data[12])
            | (u32::from(data[13]) << 8)
            | (u32::from(data[14]) << 16);

        // Flags (8 bits)
        let flags = data[15];
        let blending = BlendingMode::from_bit((flags & 0x02) != 0);
        let disposal = DisposalMethod::from_bit((flags & 0x01) != 0);

        // Frame data follows
        let frame_data = data[16..].to_vec();

        Ok(Self {
            x_offset,
            y_offset,
            width,
            height,
            duration_ms,
            blending,
            disposal,
            frame_data,
        })
    }

    /// Decode the frame image
    pub fn decode_frame(&self) -> Result<RgbaImage> {
        let data = &self.frame_data;

        // Find ALPH chunk if present
        let mut alpha_data: Option<&[u8]> = None;
        let mut image_data: Option<&[u8]> = None;
        let mut is_lossless = false;

        let mut pos = 0;
        while pos + 8 <= data.len() {
            let fourcc = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
            let size = LittleEndian::read_u32(&data[pos + 4..pos + 8]) as usize;
            let chunk_end = pos + 8 + size + (size % 2);

            match &fourcc {
                b"ALPH" => {
                    if pos + 8 + size <= data.len() {
                        alpha_data = Some(&data[pos + 8..pos + 8 + size]);
                    }
                }
                b"VP8 " => {
                    if pos + 8 + size <= data.len() {
                        image_data = Some(&data[pos + 8..pos + 8 + size]);
                        is_lossless = false;
                    }
                }
                b"VP8L" => {
                    if pos + 8 + size <= data.len() {
                        image_data = Some(&data[pos + 8..pos + 8 + size]);
                        is_lossless = true;
                    }
                }
                _ => {}
            }

            pos = chunk_end;
        }

        // If no embedded chunks found, try treating the whole data as image data
        if image_data.is_none() {
            if !data.is_empty() {
                // Check for VP8L signature
                if data[0] == 0x2f {
                    image_data = Some(data);
                    is_lossless = true;
                } else {
                    image_data = Some(data);
                    is_lossless = false;
                }
            } else {
                return Err(WebPError::InvalidAnimation("No image data in frame".into()));
            }
        }

        let image_data = image_data.unwrap();

        // Decode the image
        let mut image = if is_lossless {
            let decoder = Vp8lDecoder::new(image_data)?;
            decoder.decode()?
        } else {
            let decoder = Vp8Decoder::new(image_data)?;
            decoder.decode()?
        };

        // Apply alpha if present
        if let Some(alpha) = alpha_data {
            let alpha_decoder = AlphaDecoder::new(alpha)?;
            alpha_decoder.apply_alpha(&mut image)?;
        }

        Ok(image)
    }
}

/// Get animation info from a parsed RIFF container
pub fn get_animation_info(container: &RiffContainer) -> Option<AnimationInfo> {
    // Need VP8X with animation flag
    if !container.is_animated() {
        return None;
    }

    let (width, height) = container.dimensions().ok()?;

    // Parse ANIM chunk if present
    let (background_color, loop_count) = if let Some(anim) = container.find_chunk(ChunkType::ANIM) {
        parse_anim_chunk(&anim.data)
    } else {
        ([0, 0, 0, 0], 0)
    };

    Some(AnimationInfo {
        width,
        height,
        background_color,
        loop_count,
    })
}

/// Parse ANIM chunk data
fn parse_anim_chunk(data: &[u8]) -> ([u8; 4], u16) {
    if data.len() < 6 {
        return ([0, 0, 0, 0], 0);
    }

    let background_color = [data[0], data[1], data[2], data[3]];
    let loop_count = LittleEndian::read_u16(&data[4..6]);

    (background_color, loop_count)
}

/// Decode all animation frames from a RIFF container
pub fn decode_animation<R: Read + Seek>(
    container: RiffContainer,
    _reader: &mut R,
) -> Result<Vec<AnimationFrame>> {
    let _info = get_animation_info(&container)
        .ok_or_else(|| WebPError::InvalidAnimation("Not an animated WebP".into()))?;

    let anmf_chunks = container.find_chunks(ChunkType::ANMF);
    if anmf_chunks.is_empty() {
        return Err(WebPError::InvalidAnimation("No animation frames found".into()));
    }

    let mut frames = Vec::with_capacity(anmf_chunks.len());

    for chunk in anmf_chunks {
        let anmf = AnmfChunk::parse(&chunk.data)?;
        let image = anmf.decode_frame()?;

        frames.push(AnimationFrame {
            x_offset: anmf.x_offset,
            y_offset: anmf.y_offset,
            width: anmf.width,
            height: anmf.height,
            duration_ms: anmf.duration_ms,
            blending: anmf.blending,
            disposal: anmf.disposal,
            image,
        });
    }

    Ok(frames)
}

/// Render animation frames to a canvas
pub fn render_animation(
    info: &AnimationInfo,
    frames: &[AnimationFrame],
) -> Vec<RgbaImage> {
    let mut rendered = Vec::with_capacity(frames.len());
    let mut canvas = RgbaImage::from_pixel(
        info.width,
        info.height,
        image::Rgba(info.background_color),
    );

    for frame in frames {
        // Apply blending
        match frame.blending {
            BlendingMode::AlphaBlending => {
                blend_frame(&mut canvas, frame);
            }
            BlendingMode::NoBlending => {
                copy_frame(&mut canvas, frame);
            }
        }

        // Store the rendered frame
        rendered.push(canvas.clone());

        // Apply disposal
        match frame.disposal {
            DisposalMethod::None => {
                // Keep the canvas as is
            }
            DisposalMethod::Background => {
                // Clear the frame area to background
                clear_frame_area(&mut canvas, frame, info.background_color);
            }
        }
    }

    rendered
}

/// Blend a frame onto the canvas using alpha blending
fn blend_frame(canvas: &mut RgbaImage, frame: &AnimationFrame) {
    for (x, y, pixel) in frame.image.enumerate_pixels() {
        let canvas_x = frame.x_offset + x;
        let canvas_y = frame.y_offset + y;

        if canvas_x < canvas.width() && canvas_y < canvas.height() {
            let src = pixel.0;
            let dst = canvas.get_pixel(canvas_x, canvas_y).0;

            let src_a = src[3] as u32;
            let dst_a = dst[3] as u32;

            if src_a == 255 {
                canvas.put_pixel(canvas_x, canvas_y, *pixel);
            } else if src_a > 0 {
                // Alpha blend
                let out_a = src_a + dst_a * (255 - src_a) / 255;
                if out_a > 0 {
                    let blend = |s: u8, d: u8| -> u8 {
                        let s = s as u32;
                        let d = d as u32;
                        ((s * src_a + d * dst_a * (255 - src_a) / 255) / out_a) as u8
                    };

                    let blended = image::Rgba([
                        blend(src[0], dst[0]),
                        blend(src[1], dst[1]),
                        blend(src[2], dst[2]),
                        out_a as u8,
                    ]);
                    canvas.put_pixel(canvas_x, canvas_y, blended);
                }
            }
        }
    }
}

/// Copy a frame onto the canvas without blending
fn copy_frame(canvas: &mut RgbaImage, frame: &AnimationFrame) {
    for (x, y, pixel) in frame.image.enumerate_pixels() {
        let canvas_x = frame.x_offset + x;
        let canvas_y = frame.y_offset + y;

        if canvas_x < canvas.width() && canvas_y < canvas.height() {
            canvas.put_pixel(canvas_x, canvas_y, *pixel);
        }
    }
}

/// Clear the frame area to background color
fn clear_frame_area(canvas: &mut RgbaImage, frame: &AnimationFrame, bg: [u8; 4]) {
    let bg_pixel = image::Rgba(bg);

    for y in 0..frame.height {
        for x in 0..frame.width {
            let canvas_x = frame.x_offset + x;
            let canvas_y = frame.y_offset + y;

            if canvas_x < canvas.width() && canvas_y < canvas.height() {
                canvas.put_pixel(canvas_x, canvas_y, bg_pixel);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blending_mode() {
        assert_eq!(BlendingMode::from_bit(false), BlendingMode::AlphaBlending);
        assert_eq!(BlendingMode::from_bit(true), BlendingMode::NoBlending);
    }

    #[test]
    fn test_disposal_method() {
        assert_eq!(DisposalMethod::from_bit(false), DisposalMethod::None);
        assert_eq!(DisposalMethod::from_bit(true), DisposalMethod::Background);
    }

    #[test]
    fn test_parse_anim_chunk() {
        let data = [0xFF, 0x00, 0x00, 0xFF, 0x02, 0x00];
        let (bg, loops) = parse_anim_chunk(&data);
        assert_eq!(bg, [0xFF, 0x00, 0x00, 0xFF]);
        assert_eq!(loops, 2);
    }

    #[test]
    fn test_anmf_parse() {
        // Minimal ANMF chunk
        let mut data = vec![0u8; 16];
        // x_offset = 0, y_offset = 0
        // width = 100 (stored as 99)
        data[6] = 99;
        // height = 100 (stored as 99)
        data[9] = 99;
        // duration = 100ms
        data[12] = 100;
        // flags = 0

        let anmf = AnmfChunk::parse(&data);
        assert!(anmf.is_ok());

        let anmf = anmf.unwrap();
        assert_eq!(anmf.width, 100);
        assert_eq!(anmf.height, 100);
        assert_eq!(anmf.duration_ms, 100);
    }
}
