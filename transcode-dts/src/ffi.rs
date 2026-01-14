//! FFI bindings for DTS/TrueHD decoding via FFmpeg.
//!
//! This module provides FFI wrappers around FFmpeg's libavcodec for
//! DTS (including DTS-HD) and Dolby TrueHD decoding.
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
use crate::{DtsError, Result};

use std::ptr;

/// FFmpeg codec context wrapper for DTS decoding.
pub struct DtsFfiDecoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for input.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for output.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Bit depth.
    bit_depth: u8,
}

// SAFETY: The FFmpeg context is only accessed from one thread at a time.
unsafe impl Send for DtsFfiDecoder {}

impl DtsFfiDecoder {
    /// Create a new DTS FFI decoder.
    pub fn new() -> Result<Self> {
        // SAFETY: FFmpeg initialization and codec lookup are safe operations.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_decoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_DTS);
            if codec.is_null() {
                return Err(DtsError::FfiInitError("DTS decoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(DtsError::FfiInitError("Failed to allocate codec context".into()));
            }

            // Enable DTS-HD decoding
            (*ctx).request_sample_fmt = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP;

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError("Failed to allocate frame".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate: 48000,
                channels: 6,
                bit_depth: 24,
            })
        }
    }

    /// Decode a DTS frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // SAFETY: We're passing valid pointers to FFmpeg functions.
        unsafe {
            (*self.packet).data = data.as_ptr() as *mut u8;
            (*self.packet).size = data.len() as i32;

            let ret = ffmpeg_sys_next::avcodec_send_packet(self.ctx, self.packet);
            if ret < 0 {
                return Err(DtsError::DecodingError(format!(
                    "Failed to send packet: error {}",
                    ret
                )));
            }

            let ret = ffmpeg_sys_next::avcodec_receive_frame(self.ctx, self.frame);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    return Ok(DecodedAudio {
                        sample_rate: self.sample_rate,
                        channels: self.channels,
                        layout: ChannelLayout::surround_5_1(),
                        samples: Vec::new(),
                        samples_per_channel: 0,
                        bit_depth: self.bit_depth,
                    });
                }
                return Err(DtsError::DecodingError(format!(
                    "Failed to receive frame: error {}",
                    ret
                )));
            }

            let sample_rate = (*self.ctx).sample_rate as u32;
            let ch_layout = (*self.ctx).ch_layout;
            let channels = ch_layout.nb_channels as u8;
            let nb_samples = (*self.frame).nb_samples as usize;

            self.sample_rate = sample_rate;
            self.channels = channels;

            let samples = self.extract_samples(nb_samples, channels as usize)?;
            let layout = channel_layout_from_ffmpeg(channels);

            Ok(DecodedAudio {
                sample_rate,
                channels,
                layout,
                samples,
                samples_per_channel: nb_samples,
                bit_depth: self.bit_depth,
            })
        }
    }

    /// Extract samples from FFmpeg frame.
    fn extract_samples(&self, nb_samples: usize, channels: usize) -> Result<Vec<f32>> {
        // SAFETY: We're reading from valid frame data pointers.
        unsafe {
            let format = (*self.frame).format;
            let mut samples = Vec::with_capacity(nb_samples * channels);

            match format {
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const f32;
                            samples.push(*ptr.add(i));
                        }
                    }
                }
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S32P as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const i32;
                            let sample = *ptr.add(i);
                            samples.push(sample as f32 / 2147483648.0);
                        }
                    }
                }
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S16P as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const i16;
                            let sample = *ptr.add(i);
                            samples.push(sample as f32 / 32768.0);
                        }
                    }
                }
                _ => {
                    return Err(DtsError::DecodingError(format!(
                        "Unsupported sample format: {}",
                        format
                    )));
                }
            }

            Ok(samples)
        }
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        // SAFETY: avcodec_flush_buffers is safe on a valid context.
        unsafe {
            ffmpeg_sys_next::avcodec_flush_buffers(self.ctx);
        }
    }
}

impl Drop for DtsFfiDecoder {
    fn drop(&mut self) {
        // SAFETY: Freeing FFmpeg resources in correct order.
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

/// FFmpeg codec context wrapper for TrueHD decoding.
pub struct TrueHdFfiDecoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for input.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for output.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Bit depth.
    bit_depth: u8,
}

// SAFETY: Same rationale as DtsFfiDecoder.
unsafe impl Send for TrueHdFfiDecoder {}

impl TrueHdFfiDecoder {
    /// Create a new TrueHD FFI decoder.
    pub fn new() -> Result<Self> {
        // SAFETY: FFmpeg initialization and codec lookup are safe operations.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_decoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_TRUEHD);
            if codec.is_null() {
                return Err(DtsError::FfiInitError("TrueHD decoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(DtsError::FfiInitError("Failed to allocate codec context".into()));
            }

            (*ctx).request_sample_fmt = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S32P;

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(DtsError::FfiInitError("Failed to allocate frame".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate: 48000,
                channels: 8,
                bit_depth: 24,
            })
        }
    }

    /// Decode a TrueHD frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // SAFETY: We're passing valid pointers to FFmpeg functions.
        unsafe {
            (*self.packet).data = data.as_ptr() as *mut u8;
            (*self.packet).size = data.len() as i32;

            let ret = ffmpeg_sys_next::avcodec_send_packet(self.ctx, self.packet);
            if ret < 0 {
                return Err(DtsError::DecodingError(format!(
                    "Failed to send packet: error {}",
                    ret
                )));
            }

            let ret = ffmpeg_sys_next::avcodec_receive_frame(self.ctx, self.frame);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    return Ok(DecodedAudio {
                        sample_rate: self.sample_rate,
                        channels: self.channels,
                        layout: ChannelLayout::surround_7_1(),
                        samples: Vec::new(),
                        samples_per_channel: 0,
                        bit_depth: self.bit_depth,
                    });
                }
                return Err(DtsError::DecodingError(format!(
                    "Failed to receive frame: error {}",
                    ret
                )));
            }

            let sample_rate = (*self.ctx).sample_rate as u32;
            let ch_layout = (*self.ctx).ch_layout;
            let channels = ch_layout.nb_channels as u8;
            let nb_samples = (*self.frame).nb_samples as usize;

            self.sample_rate = sample_rate;
            self.channels = channels;

            let samples = self.extract_samples(nb_samples, channels as usize)?;
            let layout = channel_layout_from_ffmpeg(channels);

            Ok(DecodedAudio {
                sample_rate,
                channels,
                layout,
                samples,
                samples_per_channel: nb_samples,
                bit_depth: self.bit_depth,
            })
        }
    }

    /// Extract samples from FFmpeg frame.
    fn extract_samples(&self, nb_samples: usize, channels: usize) -> Result<Vec<f32>> {
        // SAFETY: We're reading from valid frame data pointers.
        unsafe {
            let format = (*self.frame).format;
            let mut samples = Vec::with_capacity(nb_samples * channels);

            match format {
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S32P as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const i32;
                            let sample = *ptr.add(i);
                            samples.push(sample as f32 / 2147483648.0);
                        }
                    }
                }
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const f32;
                            samples.push(*ptr.add(i));
                        }
                    }
                }
                _ => {
                    return Err(DtsError::DecodingError(format!(
                        "Unsupported sample format: {}",
                        format
                    )));
                }
            }

            Ok(samples)
        }
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        // SAFETY: avcodec_flush_buffers is safe on a valid context.
        unsafe {
            ffmpeg_sys_next::avcodec_flush_buffers(self.ctx);
        }
    }
}

impl Drop for TrueHdFfiDecoder {
    fn drop(&mut self) {
        // SAFETY: Freeing FFmpeg resources in correct order.
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

/// Convert FFmpeg channel count to our ChannelLayout.
fn channel_layout_from_ffmpeg(channels: u8) -> ChannelLayout {
    match channels {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        3 => ChannelLayout::surround_3_0(),
        4 => ChannelLayout::quad(),
        5 => ChannelLayout::surround_5_0(),
        6 => ChannelLayout::surround_5_1(),
        8 => ChannelLayout::surround_7_1(),
        _ => ChannelLayout::stereo(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout_conversion() {
        let layout = channel_layout_from_ffmpeg(6);
        assert_eq!(layout.channels(), 6);
    }
}
