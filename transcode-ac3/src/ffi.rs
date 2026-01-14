//! FFI bindings for AC-3/E-AC-3 encoding and decoding via FFmpeg.
//!
//! This module provides FFI wrappers around FFmpeg's libavcodec for
//! AC-3 (Dolby Digital) and E-AC-3 (Dolby Digital Plus) encoding/decoding.
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
use crate::{Ac3Error, Result, AC3_SAMPLES_PER_FRAME};

use std::ptr;

/// FFmpeg codec context wrapper for AC-3 decoding.
pub struct Ac3FfiDecoder {
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
}

// SAFETY: The FFmpeg context is only accessed from one thread at a time.
// FFmpeg codec contexts are not inherently thread-safe, but we ensure
// single-threaded access through Rust's ownership model.
unsafe impl Send for Ac3FfiDecoder {}

impl Ac3FfiDecoder {
    /// Create a new AC-3 FFI decoder.
    pub fn new() -> Result<Self> {
        // SAFETY: FFmpeg initialization is safe if called before any other FFmpeg functions.
        // avcodec_find_decoder returns a static pointer that doesn't need freeing.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_decoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_AC3);
            if codec.is_null() {
                return Err(Ac3Error::FfiInitError("AC-3 decoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Ac3Error::FfiInitError("Failed to allocate codec context".into()));
            }

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate: 48000,
                channels: 2,
            })
        }
    }

    /// Decode an AC-3 frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // SAFETY: We're passing valid pointers and sizes to FFmpeg functions.
        // The data slice is valid for the duration of this function call.
        unsafe {
            // Set packet data
            (*self.packet).data = data.as_ptr() as *mut u8;
            (*self.packet).size = data.len() as i32;

            // Send packet to decoder
            let ret = ffmpeg_sys_next::avcodec_send_packet(self.ctx, self.packet);
            if ret < 0 {
                return Err(Ac3Error::DecodingError(format!(
                    "Failed to send packet: error {}",
                    ret
                )));
            }

            // Receive decoded frame
            let ret = ffmpeg_sys_next::avcodec_receive_frame(self.ctx, self.frame);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    // Need more data
                    return Ok(DecodedAudio {
                        sample_rate: self.sample_rate,
                        channels: self.channels,
                        layout: ChannelLayout::Stereo,
                        samples: Vec::new(),
                        samples_per_channel: 0,
                    });
                }
                return Err(Ac3Error::DecodingError(format!(
                    "Failed to receive frame: error {}",
                    ret
                )));
            }

            // Extract decoded samples
            let sample_rate = (*self.ctx).sample_rate as u32;
            let ch_layout = (*self.ctx).ch_layout;
            let channels = ch_layout.nb_channels as u8;
            let nb_samples = (*self.frame).nb_samples as usize;

            self.sample_rate = sample_rate;
            self.channels = channels;

            // Convert samples to f32
            let samples = self.extract_samples(nb_samples, channels as usize)?;

            // Determine channel layout
            let layout = channel_layout_from_ffmpeg(ch_layout.nb_channels as u8);

            Ok(DecodedAudio {
                sample_rate,
                channels,
                layout,
                samples,
                samples_per_channel: nb_samples,
            })
        }
    }

    /// Extract samples from FFmpeg frame.
    fn extract_samples(&self, nb_samples: usize, channels: usize) -> Result<Vec<f32>> {
        // SAFETY: We're reading from valid frame data pointers allocated by FFmpeg.
        unsafe {
            let format = (*self.frame).format;
            let mut samples = Vec::with_capacity(nb_samples * channels);

            match format {
                // AV_SAMPLE_FMT_FLTP (planar float)
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const f32;
                            samples.push(*ptr.add(i));
                        }
                    }
                }
                // AV_SAMPLE_FMT_FLT (interleaved float)
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLT as i32 => {
                    let ptr = (*self.frame).data[0] as *const f32;
                    for i in 0..(nb_samples * channels) {
                        samples.push(*ptr.add(i));
                    }
                }
                // AV_SAMPLE_FMT_S16P (planar 16-bit signed)
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S16P as i32 => {
                    for i in 0..nb_samples {
                        for ch in 0..channels {
                            let ptr = (*self.frame).data[ch] as *const i16;
                            let sample = *ptr.add(i);
                            samples.push(sample as f32 / 32768.0);
                        }
                    }
                }
                // AV_SAMPLE_FMT_S16 (interleaved 16-bit signed)
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S16 as i32 => {
                    let ptr = (*self.frame).data[0] as *const i16;
                    for i in 0..(nb_samples * channels) {
                        let sample = *ptr.add(i);
                        samples.push(sample as f32 / 32768.0);
                    }
                }
                _ => {
                    return Err(Ac3Error::DecodingError(format!(
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
        // SAFETY: avcodec_flush_buffers is safe to call on a valid context.
        unsafe {
            ffmpeg_sys_next::avcodec_flush_buffers(self.ctx);
        }
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }
}

impl Drop for Ac3FfiDecoder {
    fn drop(&mut self) {
        // SAFETY: We're freeing resources allocated by FFmpeg in the correct order.
        // Frame must be freed before packet, and both before the context.
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

/// FFmpeg codec context wrapper for E-AC-3 decoding.
pub struct Eac3FfiDecoder {
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
}

// SAFETY: Same rationale as Ac3FfiDecoder.
unsafe impl Send for Eac3FfiDecoder {}

impl Eac3FfiDecoder {
    /// Create a new E-AC-3 FFI decoder.
    pub fn new() -> Result<Self> {
        // SAFETY: Same safety considerations as Ac3FfiDecoder::new().
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_decoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_EAC3);
            if codec.is_null() {
                return Err(Ac3Error::FfiInitError("E-AC-3 decoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Ac3Error::FfiInitError("Failed to allocate codec context".into()));
            }

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError(format!(
                    "Failed to open codec: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate: 48000,
                channels: 2,
            })
        }
    }

    /// Decode an E-AC-3 frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // SAFETY: Same safety considerations as Ac3FfiDecoder::decode().
        unsafe {
            (*self.packet).data = data.as_ptr() as *mut u8;
            (*self.packet).size = data.len() as i32;

            let ret = ffmpeg_sys_next::avcodec_send_packet(self.ctx, self.packet);
            if ret < 0 {
                return Err(Ac3Error::DecodingError(format!(
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
                        layout: ChannelLayout::Stereo,
                        samples: Vec::new(),
                        samples_per_channel: 0,
                    });
                }
                return Err(Ac3Error::DecodingError(format!(
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
            })
        }
    }

    /// Extract samples from FFmpeg frame (same implementation as Ac3FfiDecoder).
    fn extract_samples(&self, nb_samples: usize, channels: usize) -> Result<Vec<f32>> {
        // SAFETY: Same safety considerations as Ac3FfiDecoder::extract_samples().
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
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLT as i32 => {
                    let ptr = (*self.frame).data[0] as *const f32;
                    for i in 0..(nb_samples * channels) {
                        samples.push(*ptr.add(i));
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
                f if f == ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_S16 as i32 => {
                    let ptr = (*self.frame).data[0] as *const i16;
                    for i in 0..(nb_samples * channels) {
                        let sample = *ptr.add(i);
                        samples.push(sample as f32 / 32768.0);
                    }
                }
                _ => {
                    return Err(Ac3Error::DecodingError(format!(
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
        // SAFETY: avcodec_flush_buffers is safe to call on a valid context.
        unsafe {
            ffmpeg_sys_next::avcodec_flush_buffers(self.ctx);
        }
    }
}

impl Drop for Eac3FfiDecoder {
    fn drop(&mut self) {
        // SAFETY: Same safety considerations as Ac3FfiDecoder::drop().
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

/// FFmpeg codec context wrapper for AC-3 encoding.
pub struct Ac3FfiEncoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for output.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for input.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Bitrate.
    bitrate: u32,
    /// Samples buffered.
    sample_buffer: Vec<f32>,
    /// Frames encoded.
    frames_encoded: u64,
}

// SAFETY: Same rationale as Ac3FfiDecoder.
unsafe impl Send for Ac3FfiEncoder {}

impl Ac3FfiEncoder {
    /// Create a new AC-3 FFI encoder.
    pub fn new(sample_rate: u32, channels: u8, bitrate: u32) -> Result<Self> {
        // SAFETY: FFmpeg initialization is safe if called before any other FFmpeg functions.
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_encoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_AC3);
            if codec.is_null() {
                return Err(Ac3Error::FfiInitError("AC-3 encoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Ac3Error::FfiInitError("Failed to allocate codec context".into()));
            }

            // Configure encoder
            (*ctx).sample_rate = sample_rate as i32;
            (*ctx).bit_rate = bitrate as i64;
            (*ctx).sample_fmt = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP;
            (*ctx).frame_size = AC3_SAMPLES_PER_FRAME as i32;

            // Set channel layout
            let ch_layout = channel_layout_to_ffmpeg(channels);
            (*ctx).ch_layout = ch_layout;

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError(format!(
                    "Failed to open encoder: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame".into()));
            }

            // Configure frame
            (*frame).nb_samples = AC3_SAMPLES_PER_FRAME as i32;
            (*frame).format = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32;
            (*frame).ch_layout = ch_layout;

            let ret = ffmpeg_sys_next::av_frame_get_buffer(frame, 0);
            if ret < 0 {
                ffmpeg_sys_next::av_frame_free(&mut (frame as *mut _));
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame buffer".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate,
                channels,
                bitrate,
                sample_buffer: Vec::new(),
                frames_encoded: 0,
            })
        }
    }

    /// Encode samples to AC-3.
    pub fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        // Buffer samples
        self.sample_buffer.extend_from_slice(samples);

        let samples_needed = AC3_SAMPLES_PER_FRAME * self.channels as usize;
        if self.sample_buffer.len() < samples_needed {
            return Ok(Vec::new());
        }

        // SAFETY: We're writing to valid frame data pointers allocated by FFmpeg.
        unsafe {
            // Make frame writable
            let ret = ffmpeg_sys_next::av_frame_make_writable(self.frame);
            if ret < 0 {
                return Err(Ac3Error::EncodingError("Failed to make frame writable".into()));
            }

            // Copy samples to frame (deinterleave to planar)
            for ch in 0..self.channels as usize {
                let ptr = (*self.frame).data[ch] as *mut f32;
                for i in 0..AC3_SAMPLES_PER_FRAME {
                    *ptr.add(i) = self.sample_buffer[i * self.channels as usize + ch];
                }
            }

            // Remove consumed samples
            self.sample_buffer.drain(..samples_needed);

            // Send frame to encoder
            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, self.frame);
            if ret < 0 {
                return Err(Ac3Error::EncodingError(format!(
                    "Failed to send frame: error {}",
                    ret
                )));
            }

            // Receive encoded packet
            let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    return Ok(Vec::new());
                }
                return Err(Ac3Error::EncodingError(format!(
                    "Failed to receive packet: error {}",
                    ret
                )));
            }

            self.frames_encoded += 1;

            // Copy packet data
            let data = std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize);
            let result = data.to_vec();

            ffmpeg_sys_next::av_packet_unref(self.packet);

            Ok(result)
        }
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();

        // SAFETY: We're calling FFmpeg functions with valid pointers.
        unsafe {
            // Send flush signal
            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, ptr::null());
            if ret < 0 && ret != ffmpeg_sys_next::AVERROR_EOF {
                return Err(Ac3Error::EncodingError("Failed to flush encoder".into()));
            }

            // Receive all remaining packets
            loop {
                let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
                if ret < 0 {
                    break;
                }

                let data = std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize);
                packets.push(data.to_vec());

                ffmpeg_sys_next::av_packet_unref(self.packet);
            }
        }

        Ok(packets)
    }

    /// Get frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }
}

impl Drop for Ac3FfiEncoder {
    fn drop(&mut self) {
        // SAFETY: We're freeing resources allocated by FFmpeg in the correct order.
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

/// FFmpeg codec context wrapper for E-AC-3 encoding.
pub struct Eac3FfiEncoder {
    /// FFmpeg codec context.
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
    /// FFmpeg packet for output.
    packet: *mut ffmpeg_sys_next::AVPacket,
    /// FFmpeg frame for input.
    frame: *mut ffmpeg_sys_next::AVFrame,
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Bitrate.
    bitrate: u32,
    /// Samples buffered.
    sample_buffer: Vec<f32>,
    /// Frames encoded.
    frames_encoded: u64,
}

// SAFETY: Same rationale as Ac3FfiEncoder.
unsafe impl Send for Eac3FfiEncoder {}

impl Eac3FfiEncoder {
    /// Create a new E-AC-3 FFI encoder.
    pub fn new(sample_rate: u32, channels: u8, bitrate: u32) -> Result<Self> {
        // SAFETY: Same safety considerations as Ac3FfiEncoder::new().
        unsafe {
            let codec = ffmpeg_sys_next::avcodec_find_encoder(ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_EAC3);
            if codec.is_null() {
                return Err(Ac3Error::FfiInitError("E-AC-3 encoder not found".into()));
            }

            let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(Ac3Error::FfiInitError("Failed to allocate codec context".into()));
            }

            (*ctx).sample_rate = sample_rate as i32;
            (*ctx).bit_rate = bitrate as i64;
            (*ctx).sample_fmt = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP;
            (*ctx).frame_size = AC3_SAMPLES_PER_FRAME as i32;

            let ch_layout = channel_layout_to_ffmpeg(channels);
            (*ctx).ch_layout = ch_layout;

            let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError(format!(
                    "Failed to open encoder: error {}",
                    ret
                )));
            }

            let packet = ffmpeg_sys_next::av_packet_alloc();
            if packet.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate packet".into()));
            }

            let frame = ffmpeg_sys_next::av_frame_alloc();
            if frame.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame".into()));
            }

            (*frame).nb_samples = AC3_SAMPLES_PER_FRAME as i32;
            (*frame).format = ffmpeg_sys_next::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32;
            (*frame).ch_layout = ch_layout;

            let ret = ffmpeg_sys_next::av_frame_get_buffer(frame, 0);
            if ret < 0 {
                ffmpeg_sys_next::av_frame_free(&mut (frame as *mut _));
                ffmpeg_sys_next::av_packet_free(&mut (packet as *mut _));
                ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
                return Err(Ac3Error::FfiInitError("Failed to allocate frame buffer".into()));
            }

            Ok(Self {
                ctx,
                packet,
                frame,
                sample_rate,
                channels,
                bitrate,
                sample_buffer: Vec::new(),
                frames_encoded: 0,
            })
        }
    }

    /// Encode samples to E-AC-3.
    pub fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        self.sample_buffer.extend_from_slice(samples);

        let samples_needed = AC3_SAMPLES_PER_FRAME * self.channels as usize;
        if self.sample_buffer.len() < samples_needed {
            return Ok(Vec::new());
        }

        // SAFETY: Same safety considerations as Ac3FfiEncoder::encode().
        unsafe {
            let ret = ffmpeg_sys_next::av_frame_make_writable(self.frame);
            if ret < 0 {
                return Err(Ac3Error::EncodingError("Failed to make frame writable".into()));
            }

            for ch in 0..self.channels as usize {
                let ptr = (*self.frame).data[ch] as *mut f32;
                for i in 0..AC3_SAMPLES_PER_FRAME {
                    *ptr.add(i) = self.sample_buffer[i * self.channels as usize + ch];
                }
            }

            self.sample_buffer.drain(..samples_needed);

            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, self.frame);
            if ret < 0 {
                return Err(Ac3Error::EncodingError(format!(
                    "Failed to send frame: error {}",
                    ret
                )));
            }

            let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
            if ret < 0 {
                if ret == ffmpeg_sys_next::AVERROR(ffmpeg_sys_next::EAGAIN as i32) {
                    return Ok(Vec::new());
                }
                return Err(Ac3Error::EncodingError(format!(
                    "Failed to receive packet: error {}",
                    ret
                )));
            }

            self.frames_encoded += 1;

            let data = std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize);
            let result = data.to_vec();

            ffmpeg_sys_next::av_packet_unref(self.packet);

            Ok(result)
        }
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();

        // SAFETY: Same safety considerations as Ac3FfiEncoder::flush().
        unsafe {
            let ret = ffmpeg_sys_next::avcodec_send_frame(self.ctx, ptr::null());
            if ret < 0 && ret != ffmpeg_sys_next::AVERROR_EOF {
                return Err(Ac3Error::EncodingError("Failed to flush encoder".into()));
            }

            loop {
                let ret = ffmpeg_sys_next::avcodec_receive_packet(self.ctx, self.packet);
                if ret < 0 {
                    break;
                }

                let data = std::slice::from_raw_parts((*self.packet).data, (*self.packet).size as usize);
                packets.push(data.to_vec());

                ffmpeg_sys_next::av_packet_unref(self.packet);
            }
        }

        Ok(packets)
    }

    /// Get frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }
}

impl Drop for Eac3FfiEncoder {
    fn drop(&mut self) {
        // SAFETY: Same safety considerations as Ac3FfiEncoder::drop().
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

/// Convert channel count to FFmpeg channel layout.
fn channel_layout_to_ffmpeg(channels: u8) -> ffmpeg_sys_next::AVChannelLayout {
    // SAFETY: We're creating a properly initialized channel layout struct.
    unsafe {
        let mut layout: ffmpeg_sys_next::AVChannelLayout = std::mem::zeroed();

        match channels {
            1 => {
                layout.order = ffmpeg_sys_next::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE;
                layout.nb_channels = 1;
                layout.u.mask = ffmpeg_sys_next::AV_CH_LAYOUT_MONO;
            }
            2 => {
                layout.order = ffmpeg_sys_next::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE;
                layout.nb_channels = 2;
                layout.u.mask = ffmpeg_sys_next::AV_CH_LAYOUT_STEREO;
            }
            6 => {
                layout.order = ffmpeg_sys_next::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE;
                layout.nb_channels = 6;
                layout.u.mask = ffmpeg_sys_next::AV_CH_LAYOUT_5POINT1;
            }
            8 => {
                layout.order = ffmpeg_sys_next::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE;
                layout.nb_channels = 8;
                layout.u.mask = ffmpeg_sys_next::AV_CH_LAYOUT_7POINT1;
            }
            _ => {
                layout.order = ffmpeg_sys_next::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE;
                layout.nb_channels = channels as i32;
                layout.u.mask = ffmpeg_sys_next::AV_CH_LAYOUT_STEREO;
            }
        }

        layout
    }
}

/// Convert FFmpeg channel count to our ChannelLayout.
fn channel_layout_from_ffmpeg(channels: u8) -> ChannelLayout {
    match channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        3 => ChannelLayout::Surround3_0,
        4 => ChannelLayout::Quad,
        5 => ChannelLayout::Surround5_0,
        6 => ChannelLayout::Surround5_1,
        8 => ChannelLayout::Surround7_1,
        _ => ChannelLayout::Stereo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout_conversion() {
        let layout = channel_layout_from_ffmpeg(6);
        assert_eq!(layout, ChannelLayout::Surround5_1);
    }

    #[test]
    fn test_channel_layout_to_ffmpeg() {
        let layout = channel_layout_to_ffmpeg(6);
        assert_eq!(layout.nb_channels, 6);
    }
}
