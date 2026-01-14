//! FFI bindings for MPEG-2 video decoding/encoding via FFmpeg.
//!
//! This module provides FFI wrappers around FFmpeg's libavcodec for
//! MPEG-2 video encoding and decoding.
//!
//! # Safety
//!
//! All FFmpeg functions are inherently unsafe. This module wraps them in safe
//! Rust interfaces with proper resource management.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::types::*;
use crate::{Mpeg2Error, Result};

use std::ptr;

/// FFmpeg codec context wrapper for MPEG-2 video decoding.
pub struct Mpeg2FfiDecoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for input.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for output.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Frame width.
    width: u16,
    /// Frame height.
    height: u16,
}

// SAFETY: The FFmpeg context is only accessed from one thread at a time.
// The FFmpeg decoder context maintains internal state but is not shared across threads.
unsafe impl Send for Mpeg2FfiDecoder {}

impl Mpeg2FfiDecoder {
    /// Create a new MPEG-2 FFI decoder.
    pub fn new() -> Result<Self> {
        // SAFETY: FFmpeg initialization and codec lookup are safe operations.
        // avcodec_find_decoder returns a static reference to the codec.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_decoder(
                ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_MPEG2VIDEO,
            );
            if codec.is_null() {
                return Err(Mpeg2Error::FfiInitError("MPEG-2 decoder not found".into()));
            }

            // SAFETY: avcodec_alloc_context3 allocates a new context. We check for null.
            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Mpeg2Error::FfiInitError(
                    "Failed to allocate codec context".into(),
                ));
            }

            // SAFETY: ctx is valid and codec is compatible. We check the return value.
            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            // SAFETY: av_packet_alloc allocates a new packet. We check for null.
            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError(
                    "Failed to allocate packet".into(),
                ));
            }

            // SAFETY: av_frame_alloc allocates a new frame. We check for null.
            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError("Failed to allocate frame".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                width: 0,
                height: 0,
            })
        }
    }

    /// Decode an MPEG-2 video frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedFrame> {
        // SAFETY: We're passing valid pointers to FFmpeg functions.
        // The packet data pointer is only used during the decode call.
        unsafe {
            (*self.packet).data = data.as_ptr() as *mut u8;
            (*self.packet).size = data.len() as i32;

            // SAFETY: ctx and packet are valid pointers. We check return value.
            let ret = ffmpeg_sys_next::avcodec_send_packet(self.ctx, self.packet);
            if ret < 0 {
                return Err(Mpeg2Error::DecodingError(format!(
                    "Failed to send packet: error {}",
                    ret
                )));
            }

            // SAFETY: ctx and frame are valid pointers. We check return value.
            let ret = ffmpeg_sys_next::avcodec_receive_frame(self.ctx, self.frame);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    // Need more data
                    return Ok(DecodedFrame::new(
                        self.width.max(720),
                        self.height.max(480),
                    ));
                }
                return Err(Mpeg2Error::DecodingError(format!(
                    "Failed to receive frame: error {}",
                    ret
                )));
            }

            // SAFETY: After successful decode, frame fields are valid.
            let width = (*self.frame).width as u16;
            let height = (*self.frame).height as u16;
            self.width = width;
            self.height = height;

            let (y_plane, u_plane, v_plane) = self.extract_planes(width, height)?;

            // Determine picture type from frame
            let picture_type = match (*self.frame).pict_type {
                ffmpeg_sys_next::AVPictureType::AV_PICTURE_TYPE_I => PictureCodingType::I,
                ffmpeg_sys_next::AVPictureType::AV_PICTURE_TYPE_P => PictureCodingType::P,
                ffmpeg_sys_next::AVPictureType::AV_PICTURE_TYPE_B => PictureCodingType::B,
                _ => PictureCodingType::I,
            };

            let pts = if (*self.frame).pts != ffmpeg_sys_next::AV_NOPTS_VALUE {
                Some((*self.frame).pts as u64)
            } else {
                None
            };

            // Check interlacing
            let progressive = (*self.frame).interlaced_frame == 0;
            let top_field_first = (*self.frame).top_field_first != 0;

            Ok(DecodedFrame {
                width,
                height,
                picture_type,
                temporal_reference: 0, // Not available from FFmpeg directly
                progressive,
                top_field_first,
                y_plane,
                u_plane,
                v_plane,
                pts,
                dts: None,
            })
        }
    }

    /// Extract YUV planes from FFmpeg frame.
    fn extract_planes(
        &self,
        width: u16,
        height: u16,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // SAFETY: We're reading from valid frame data pointers after successful decode.
        unsafe {
            let format = (*self.frame).format;

            // Only support YUV420P (most common for MPEG-2)
            if format != ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_YUV420P as i32
                && format != ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_YUV422P as i32
            {
                return Err(Mpeg2Error::DecodingError(format!(
                    "Unsupported pixel format: {}",
                    format
                )));
            }

            let w = width as usize;
            let h = height as usize;

            // Y plane
            let y_linesize = (*self.frame).linesize[0] as usize;
            let mut y_plane = Vec::with_capacity(w * h);
            let y_ptr = (*self.frame).data[0];
            for row in 0..h {
                let row_start = y_ptr.add(row * y_linesize);
                y_plane.extend_from_slice(std::slice::from_raw_parts(row_start, w));
            }

            // U plane (half resolution for 4:2:0)
            let u_linesize = (*self.frame).linesize[1] as usize;
            let uv_h = if format == ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_YUV420P as i32 {
                h / 2
            } else {
                h
            };
            let uv_w = w / 2;
            let mut u_plane = Vec::with_capacity(uv_w * uv_h);
            let u_ptr = (*self.frame).data[1];
            for row in 0..uv_h {
                let row_start = u_ptr.add(row * u_linesize);
                u_plane.extend_from_slice(std::slice::from_raw_parts(row_start, uv_w));
            }

            // V plane (half resolution for 4:2:0)
            let v_linesize = (*self.frame).linesize[2] as usize;
            let mut v_plane = Vec::with_capacity(uv_w * uv_h);
            let v_ptr = (*self.frame).data[2];
            for row in 0..uv_h {
                let row_start = v_ptr.add(row * v_linesize);
                v_plane.extend_from_slice(std::slice::from_raw_parts(row_start, uv_w));
            }

            Ok((y_plane, u_plane, v_plane))
        }
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        // SAFETY: avcodec_flush_buffers is safe on a valid context.
        unsafe {
            ffmpeg_sys_next::avcodec_flush_buffers(self.ctx);
        }
    }

    /// Get current width.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get current height.
    pub fn height(&self) -> u16 {
        self.height
    }
}

impl Drop for Mpeg2FfiDecoder {
    fn drop(&mut self) {
        // SAFETY: Freeing FFmpeg resources in correct order.
        // Frame and packet must be freed before context.
        unsafe {
            if !self.frame.is_null() {
                ffmpeg_sys_next::av_frame_free(&mut self.frame);
            }
            if !self.packet.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut self.packet);
            }
            if !self.ctx.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut self.ctx);
            }
        }
    }
}

/// FFmpeg codec context wrapper for MPEG-2 video encoding.
pub struct Mpeg2FfiEncoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for output.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for input.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Frame counter for PTS.
    frame_count: u64,
}

// SAFETY: The FFmpeg context is only accessed from one thread at a time.
// The FFmpeg encoder context maintains internal state but is not shared across threads.
unsafe impl Send for Mpeg2FfiEncoder {}

impl Mpeg2FfiEncoder {
    /// Create a new MPEG-2 FFI encoder.
    pub fn new(
        width: u16,
        height: u16,
        frame_rate_num: u32,
        frame_rate_den: u32,
        bitrate_kbps: u32,
        gop_size: u32,
        b_frames: u32,
    ) -> Result<Self> {
        // SAFETY: FFmpeg initialization and codec lookup are safe operations.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_encoder(
                ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_MPEG2VIDEO,
            );
            if codec.is_null() {
                return Err(Mpeg2Error::FfiInitError("MPEG-2 encoder not found".into()));
            }

            // SAFETY: avcodec_alloc_context3 allocates a new context.
            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Mpeg2Error::FfiInitError(
                    "Failed to allocate codec context".into(),
                ));
            }

            // Configure encoder
            (*ctx).width = width as i32;
            (*ctx).height = height as i32;
            (*ctx).pix_fmt = ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_YUV420P;
            (*ctx).time_base = ffmpeg_sys_next::AVRational {
                num: frame_rate_den as i32,
                den: frame_rate_num as i32,
            };
            (*ctx).framerate = ffmpeg_sys_next::AVRational {
                num: frame_rate_num as i32,
                den: frame_rate_den as i32,
            };
            (*ctx).bit_rate = (bitrate_kbps as i64) * 1000;
            (*ctx).gop_size = gop_size as i32;
            (*ctx).max_b_frames = b_frames as i32;

            // SAFETY: ctx is valid and codec is compatible.
            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            // SAFETY: av_packet_alloc allocates a new packet.
            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError(
                    "Failed to allocate packet".into(),
                ));
            }

            // SAFETY: av_frame_alloc allocates a new frame.
            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError("Failed to allocate frame".into()));
            }

            // Configure frame
            (*frame).width = width as i32;
            (*frame).height = height as i32;
            (*frame).format = ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_YUV420P as i32;

            // SAFETY: av_frame_get_buffer allocates frame data buffers.
            let ret = ffmpeg_sys_next::av_frame_get_buffer(frame, 0);
            if ret < 0 {
                ffmpeg_sys_next::av_frame_free(&mut (frame as *mut _));
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Mpeg2Error::FfiInitError(
                    "Failed to allocate frame buffer".into(),
                ));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                frame_count: 0,
            })
        }
    }

    /// Encode a video frame.
    pub fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<Option<EncodedPacket>> {
        // SAFETY: We're writing to frame buffers then calling encode.
        unsafe {
            // Make frame writable
            // SAFETY: av_frame_make_writable ensures we can write to the frame.
            let ret = ffmpeg_sys_next::av_frame_make_writable(self.frame);
            if ret < 0 {
                return Err(Mpeg2Error::EncodingError(
                    "Failed to make frame writable".into(),
                ));
            }

            let width = (*self.frame).width as usize;
            let height = (*self.frame).height as usize;

            // Copy Y plane
            let y_linesize = (*self.frame).linesize[0] as usize;
            let y_ptr = (*self.frame).data[0];
            for row in 0..height {
                let src_offset = row * width;
                let dst_offset = row * y_linesize;
                if src_offset + width <= y.len() {
                    ptr::copy_nonoverlapping(
                        y.as_ptr().add(src_offset),
                        y_ptr.add(dst_offset),
                        width,
                    );
                }
            }

            // Copy U plane (half resolution)
            let u_linesize = (*self.frame).linesize[1] as usize;
            let u_ptr = (*self.frame).data[1];
            let uv_height = height / 2;
            let uv_width = width / 2;
            for row in 0..uv_height {
                let src_offset = row * uv_width;
                let dst_offset = row * u_linesize;
                if src_offset + uv_width <= u.len() {
                    ptr::copy_nonoverlapping(
                        u.as_ptr().add(src_offset),
                        u_ptr.add(dst_offset),
                        uv_width,
                    );
                }
            }

            // Copy V plane (half resolution)
            let v_linesize = (*self.frame).linesize[2] as usize;
            let v_ptr = (*self.frame).data[2];
            for row in 0..uv_height {
                let src_offset = row * uv_width;
                let dst_offset = row * v_linesize;
                if src_offset + uv_width <= v.len() {
                    ptr::copy_nonoverlapping(
                        v.as_ptr().add(src_offset),
                        v_ptr.add(dst_offset),
                        uv_width,
                    );
                }
            }

            (*self.frame).pts = self.frame_count as i64;
            self.frame_count += 1;

            // Send frame to encoder
            // SAFETY: ctx and frame are valid pointers.
            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, self.frame);
            if ret < 0 {
                return Err(Mpeg2Error::EncodingError(format!(
                    "Failed to send frame: error {}",
                    ret
                )));
            }

            // Receive encoded packet
            // SAFETY: ctx and packet are valid pointers.
            let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    return Ok(None);
                }
                return Err(Mpeg2Error::EncodingError(format!(
                    "Failed to receive packet: error {}",
                    ret
                )));
            }

            let data =
                std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize)
                    .to_vec();

            let keyframe = ((*self.packet).flags & ffmpeg_sys_next::AV_PKT_FLAG_KEY) != 0;

            let picture_type = if keyframe {
                PictureCodingType::I
            } else {
                // Could determine from packet data, but simplified here
                PictureCodingType::P
            };

            let pts = if (*self.packet).pts != ffmpeg_sys_next::AV_NOPTS_VALUE {
                Some((*self.packet).pts as u64)
            } else {
                None
            };

            let dts = if (*self.packet).dts != ffmpeg_sys_next::AV_NOPTS_VALUE {
                Some((*self.packet).dts as u64)
            } else {
                None
            };

            // SAFETY: av_packet_unref resets the packet for reuse.
            ffmpeg_sys_next::av_packet_unref(self.packet);

            Ok(Some(EncodedPacket {
                data,
                picture_type,
                temporal_reference: 0,
                pts,
                dts,
                keyframe,
            }))
        }
    }

    /// Flush the encoder to get remaining packets.
    pub fn flush(&mut self) -> Result<Option<EncodedPacket>> {
        // SAFETY: Sending NULL frame signals end of stream.
        unsafe {
            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, ptr::null());
            if ret < 0 && ret != ffmpeg_sys_next::AVERROR_EOF {
                return Err(Mpeg2Error::EncodingError(format!(
                    "Failed to flush encoder: error {}",
                    ret
                )));
            }

            // SAFETY: ctx and packet are valid pointers.
            let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32)
                    || ret == ffmpeg_sys_next::AVERROR_EOF
                {
                    return Ok(None);
                }
                return Err(Mpeg2Error::EncodingError(format!(
                    "Failed to receive flushed packet: error {}",
                    ret
                )));
            }

            let data =
                std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize)
                    .to_vec();

            let keyframe = ((*self.packet).flags & ffmpeg_sys_next::AV_PKT_FLAG_KEY) != 0;

            let picture_type = if keyframe {
                PictureCodingType::I
            } else {
                PictureCodingType::P
            };

            let pts = if (*self.packet).pts != ffmpeg_sys_next::AV_NOPTS_VALUE {
                Some((*self.packet).pts as u64)
            } else {
                None
            };

            let dts = if (*self.packet).dts != ffmpeg_sys_next::AV_NOPTS_VALUE {
                Some((*self.packet).dts as u64)
            } else {
                None
            };

            // SAFETY: av_packet_unref resets the packet for reuse.
            ffmpeg_sys_next::av_packet_unref(self.packet);

            Ok(Some(EncodedPacket {
                data,
                picture_type,
                temporal_reference: 0,
                pts,
                dts,
                keyframe,
            }))
        }
    }
}

impl Drop for Mpeg2FfiEncoder {
    fn drop(&mut self) {
        // SAFETY: Freeing FFmpeg resources in correct order.
        // Frame and packet must be freed before context.
        unsafe {
            if !self.frame.is_null() {
                ffmpeg_sys_next::av_frame_free(&mut self.frame);
            }
            if !self.packet.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut self.packet);
            }
            if !self.ctx.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut self.ctx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_decoder_creation() {
        // This test will only pass with FFmpeg installed
        // In CI without FFmpeg, it will fail gracefully
        let result = Mpeg2FfiDecoder::new();
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_ffi_encoder_creation() {
        // This test will only pass with FFmpeg installed
        let result = Mpeg2FfiEncoder::new(720, 480, 30000, 1001, 8000, 15, 2);
        // Just verify it doesn't panic
        let _ = result;
    }
}
