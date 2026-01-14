//! FFI bindings for JPEG2000 encoding/decoding via OpenJPEG.
//!
//! This module provides FFI wrappers around OpenJPEG for
//! JPEG2000 image encoding and decoding.
//!
//! # Safety
//!
//! All OpenJPEG functions are inherently unsafe. This module wraps them in safe
//! Rust interfaces with proper resource management.
//!
//! # Requirements
//!
//! To use this module, you need:
//! - OpenJPEG development libraries installed
//! - The `ffi-openjpeg` feature enabled
//! - An `openjpeg-sys` or similar binding crate

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::decoder::DecodedImage;
use crate::encoder::EncodedPacket;
use crate::types::*;
use crate::{Jpeg2000Error, Result};

use std::ptr;

// OpenJPEG type definitions
// These would typically come from openjpeg-sys crate
mod opj_sys {
    use std::os::raw::{c_int, c_uint, c_void};

    pub const OPJ_CODEC_J2K: c_int = 0;
    pub const OPJ_CODEC_JP2: c_int = 2;

    pub const OPJ_CLRSPC_UNKNOWN: c_int = -1;
    pub const OPJ_CLRSPC_UNSPECIFIED: c_int = 0;
    pub const OPJ_CLRSPC_SRGB: c_int = 1;
    pub const OPJ_CLRSPC_GRAY: c_int = 2;
    pub const OPJ_CLRSPC_SYCC: c_int = 3;

    #[repr(C)]
    pub struct opj_cparameters_t {
        // Simplified - actual struct has many more fields
        pub cp_cinema: c_int,
        pub cp_rsiz: c_int,
        pub irreversible: c_int,
        pub tcp_numlayers: c_int,
        pub tcp_rates: [f32; 100],
        pub numresolution: c_int,
        pub cblockw_init: c_int,
        pub cblockh_init: c_int,
        pub prog_order: c_int,
        pub tile_size_on: c_int,
        pub cp_tx0: c_int,
        pub cp_ty0: c_int,
        pub cp_tdx: c_int,
        pub cp_tdy: c_int,
        _padding: [u8; 4096], // Reserve space for additional fields
    }

    #[repr(C)]
    pub struct opj_dparameters_t {
        pub cp_reduce: c_uint,
        pub cp_layer: c_uint,
        _padding: [u8; 1024],
    }

    #[repr(C)]
    pub struct opj_image_comp_t {
        pub dx: c_uint,
        pub dy: c_uint,
        pub w: c_uint,
        pub h: c_uint,
        pub x0: c_uint,
        pub y0: c_uint,
        pub prec: c_uint,
        pub bpp: c_uint,
        pub sgnd: c_uint,
        pub resno_decoded: c_uint,
        pub factor: c_uint,
        pub data: *mut i32,
    }

    #[repr(C)]
    pub struct opj_image_t {
        pub x0: c_uint,
        pub y0: c_uint,
        pub x1: c_uint,
        pub y1: c_uint,
        pub numcomps: c_uint,
        pub color_space: c_int,
        pub comps: *mut opj_image_comp_t,
        pub icc_profile_buf: *mut u8,
        pub icc_profile_len: c_uint,
    }

    pub type opj_codec_t = *mut c_void;
    pub type opj_stream_t = *mut c_void;

    // These would be provided by openjpeg-sys
    // For now, we define them as extern "C" functions
    extern "C" {
        pub fn opj_create_decompress(format: c_int) -> opj_codec_t;
        pub fn opj_create_compress(format: c_int) -> opj_codec_t;
        pub fn opj_destroy_codec(codec: opj_codec_t);

        pub fn opj_set_default_decoder_parameters(params: *mut opj_dparameters_t);
        pub fn opj_set_default_encoder_parameters(params: *mut opj_cparameters_t);

        pub fn opj_setup_decoder(codec: opj_codec_t, params: *mut opj_dparameters_t) -> c_int;
        pub fn opj_setup_encoder(codec: opj_codec_t, params: *mut opj_cparameters_t, image: *mut opj_image_t) -> c_int;

        pub fn opj_stream_create_default_memory_stream(
            buffer: *const u8,
            buffer_len: usize,
            is_input: c_int,
        ) -> opj_stream_t;
        pub fn opj_stream_destroy(stream: opj_stream_t);

        pub fn opj_read_header(
            stream: opj_stream_t,
            codec: opj_codec_t,
            image: *mut *mut opj_image_t,
        ) -> c_int;

        pub fn opj_decode(codec: opj_codec_t, stream: opj_stream_t, image: *mut opj_image_t) -> c_int;
        pub fn opj_end_decompress(codec: opj_codec_t, stream: opj_stream_t) -> c_int;

        pub fn opj_start_compress(
            codec: opj_codec_t,
            image: *mut opj_image_t,
            stream: opj_stream_t,
        ) -> c_int;
        pub fn opj_encode(codec: opj_codec_t, stream: opj_stream_t) -> c_int;
        pub fn opj_end_compress(codec: opj_codec_t, stream: opj_stream_t) -> c_int;

        pub fn opj_image_create(
            numcmpts: c_uint,
            cmptparms: *const opj_image_comp_t,
            clrspc: c_int,
        ) -> *mut opj_image_t;
        pub fn opj_image_destroy(image: *mut opj_image_t);
    }
}

/// OpenJPEG decoder wrapper.
pub struct Jpeg2000FfiDecoder {
    /// Whether the decoder is initialized.
    initialized: bool,
}

// SAFETY: The OpenJPEG context is only accessed from one thread at a time.
unsafe impl Send for Jpeg2000FfiDecoder {}

impl Jpeg2000FfiDecoder {
    /// Create a new OpenJPEG decoder.
    pub fn new() -> Result<Self> {
        Ok(Self { initialized: true })
    }

    /// Decode a JPEG2000 image.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedImage> {
        if data.len() < 2 {
            return Err(Jpeg2000Error::BufferTooSmall {
                needed: 2,
                available: data.len(),
            });
        }

        // Detect format (J2K codestream or JP2 file)
        let is_jp2 = data.len() >= 12 && &data[4..8] == b"jP  ";
        let format = if is_jp2 {
            opj_sys::OPJ_CODEC_JP2
        } else {
            opj_sys::OPJ_CODEC_J2K
        };

        // SAFETY: We're calling OpenJPEG C functions with proper null checks.
        unsafe {
            // Create decoder
            let codec = opj_sys::opj_create_decompress(format);
            if codec.is_null() {
                return Err(Jpeg2000Error::DecodingError(
                    "Failed to create decoder".into(),
                ));
            }

            // Set default parameters
            // SAFETY: Zeroing parameters struct is safe for opj_dparameters_t.
            let mut params: opj_sys::opj_dparameters_t = std::mem::zeroed();
            opj_sys::opj_set_default_decoder_parameters(&mut params);

            // Setup decoder
            // SAFETY: codec and params are valid pointers.
            if opj_sys::opj_setup_decoder(codec, &mut params) == 0 {
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::DecodingError(
                    "Failed to setup decoder".into(),
                ));
            }

            // Create memory stream
            // SAFETY: data is a valid slice with known length.
            let stream =
                opj_sys::opj_stream_create_default_memory_stream(data.as_ptr(), data.len(), 1);
            if stream.is_null() {
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::DecodingError(
                    "Failed to create stream".into(),
                ));
            }

            // Read header
            let mut image: *mut opj_sys::opj_image_t = ptr::null_mut();
            // SAFETY: stream, codec are valid. image is output parameter.
            if opj_sys::opj_read_header(stream, codec, &mut image) == 0 {
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::DecodingError("Failed to read header".into()));
            }

            if image.is_null() {
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::DecodingError("Null image returned".into()));
            }

            // Decode image
            // SAFETY: All pointers are valid at this point.
            if opj_sys::opj_decode(codec, stream, image) == 0 {
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::DecodingError("Failed to decode".into()));
            }

            // End decompression
            opj_sys::opj_end_decompress(codec, stream);

            // Extract image data
            // SAFETY: image fields are valid after successful decode.
            let width = (*image).x1 - (*image).x0;
            let height = (*image).y1 - (*image).y0;
            let num_components = (*image).numcomps as u16;

            // Get bit depth from first component
            let bit_depth = if num_components > 0 && !(*image).comps.is_null() {
                (*(*image).comps).prec as u8
            } else {
                8
            };

            let is_signed = if num_components > 0 && !(*image).comps.is_null() {
                (*(*image).comps).sgnd != 0
            } else {
                false
            };

            let color_space = match (*image).color_space {
                opj_sys::OPJ_CLRSPC_GRAY => ColorSpace::Grayscale,
                opj_sys::OPJ_CLRSPC_SRGB => ColorSpace::Srgb,
                opj_sys::OPJ_CLRSPC_SYCC => ColorSpace::YCbCr,
                _ => ColorSpace::Unknown,
            };

            // Extract component data
            let mut components = Vec::with_capacity(num_components as usize);
            for i in 0..num_components as usize {
                // SAFETY: We're iterating within bounds of comps array.
                let comp = (*image).comps.add(i);
                let comp_w = (*comp).w as usize;
                let comp_h = (*comp).h as usize;
                let num_samples = comp_w * comp_h;

                if (*comp).data.is_null() {
                    components.push(vec![0i32; num_samples]);
                } else {
                    // SAFETY: data pointer is valid and has num_samples elements.
                    let data_slice = std::slice::from_raw_parts((*comp).data, num_samples);
                    components.push(data_slice.to_vec());
                }
            }

            // Cleanup
            opj_sys::opj_image_destroy(image);
            opj_sys::opj_stream_destroy(stream);
            opj_sys::opj_destroy_codec(codec);

            Ok(DecodedImage {
                width,
                height,
                num_components,
                bit_depth,
                is_signed,
                color_space,
                components,
                profile: Jpeg2000Profile::Part1,
            })
        }
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        // No persistent state to reset
    }
}

/// OpenJPEG encoder wrapper.
pub struct Jpeg2000FfiEncoder {
    /// Encoder configuration.
    config: Jpeg2000EncoderConfigFfi,
}

/// FFI encoder configuration.
#[derive(Debug, Clone)]
pub struct Jpeg2000EncoderConfigFfi {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components.
    pub num_components: u16,
    /// Bit depth.
    pub bit_depth: u8,
    /// Is signed.
    pub is_signed: bool,
    /// Tile width (0 = single tile).
    pub tile_width: u32,
    /// Tile height (0 = single tile).
    pub tile_height: u32,
    /// Number of decomposition levels.
    pub num_decomposition_levels: u8,
    /// Number of quality layers.
    pub num_layers: u16,
    /// Compression ratio (0 = lossless).
    pub compression_ratio: f32,
    /// Lossless mode.
    pub lossless: bool,
    /// Output JP2 format.
    pub output_jp2: bool,
}

impl Default for Jpeg2000EncoderConfigFfi {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            num_components: 3,
            bit_depth: 8,
            is_signed: false,
            tile_width: 0,
            tile_height: 0,
            num_decomposition_levels: 5,
            num_layers: 1,
            compression_ratio: 10.0,
            lossless: false,
            output_jp2: false,
        }
    }
}

// SAFETY: The OpenJPEG context is only accessed from one thread at a time.
unsafe impl Send for Jpeg2000FfiEncoder {}

impl Jpeg2000FfiEncoder {
    /// Create a new OpenJPEG encoder.
    pub fn new(config: Jpeg2000EncoderConfigFfi) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(Jpeg2000Error::InvalidDimensions {
                width: config.width,
                height: config.height,
            });
        }

        Ok(Self { config })
    }

    /// Encode image data.
    ///
    /// `data` should contain interleaved pixel data in row-major order.
    pub fn encode(&mut self, data: &[u8]) -> Result<EncodedPacket> {
        let expected_size = self.config.width as usize
            * self.config.height as usize
            * self.config.num_components as usize
            * ((self.config.bit_depth as usize + 7) / 8);

        if data.len() < expected_size {
            return Err(Jpeg2000Error::BufferTooSmall {
                needed: expected_size,
                available: data.len(),
            });
        }

        let format = if self.config.output_jp2 {
            opj_sys::OPJ_CODEC_JP2
        } else {
            opj_sys::OPJ_CODEC_J2K
        };

        // SAFETY: We're calling OpenJPEG C functions with proper null checks.
        unsafe {
            // Create encoder
            let codec = opj_sys::opj_create_compress(format);
            if codec.is_null() {
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to create encoder".into(),
                ));
            }

            // Create image
            let image = self.create_image(data)?;
            if image.is_null() {
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to create image".into(),
                ));
            }

            // Set encoding parameters
            // SAFETY: Zeroing parameters struct is safe.
            let mut params: opj_sys::opj_cparameters_t = std::mem::zeroed();
            opj_sys::opj_set_default_encoder_parameters(&mut params);

            // Configure parameters
            params.numresolution = self.config.num_decomposition_levels as i32 + 1;
            params.tcp_numlayers = self.config.num_layers as i32;
            params.irreversible = if self.config.lossless { 0 } else { 1 };

            if self.config.compression_ratio > 0.0 {
                params.tcp_rates[0] = self.config.compression_ratio;
            }

            if self.config.tile_width > 0 {
                params.tile_size_on = 1;
                params.cp_tdx = self.config.tile_width as i32;
                params.cp_tdy = self.config.tile_height as i32;
            }

            // Setup encoder
            // SAFETY: All pointers are valid.
            if opj_sys::opj_setup_encoder(codec, &mut params, image) == 0 {
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to setup encoder".into(),
                ));
            }

            // Create output buffer
            let max_size = data.len() * 2; // Rough estimate
            let mut output_buffer = vec![0u8; max_size];

            // Create memory stream for output
            let stream = opj_sys::opj_stream_create_default_memory_stream(
                output_buffer.as_mut_ptr(),
                output_buffer.len(),
                0, // is_input = false
            );
            if stream.is_null() {
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to create output stream".into(),
                ));
            }

            // Start compression
            // SAFETY: All pointers are valid.
            if opj_sys::opj_start_compress(codec, image, stream) == 0 {
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to start compression".into(),
                ));
            }

            // Encode
            if opj_sys::opj_encode(codec, stream) == 0 {
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError("Failed to encode".into()));
            }

            // End compression
            if opj_sys::opj_end_compress(codec, stream) == 0 {
                opj_sys::opj_stream_destroy(stream);
                opj_sys::opj_image_destroy(image);
                opj_sys::opj_destroy_codec(codec);
                return Err(Jpeg2000Error::EncodingError(
                    "Failed to end compression".into(),
                ));
            }

            // Cleanup
            opj_sys::opj_stream_destroy(stream);
            opj_sys::opj_image_destroy(image);
            opj_sys::opj_destroy_codec(codec);

            // Find actual output size (look for EOC marker)
            let actual_size = find_codestream_end(&output_buffer);

            Ok(EncodedPacket {
                data: output_buffer[..actual_size].to_vec(),
                is_jp2: self.config.output_jp2,
            })
        }
    }

    /// Create OpenJPEG image structure from raw data.
    unsafe fn create_image(&self, data: &[u8]) -> Result<*mut opj_sys::opj_image_t> {
        let num_comps = self.config.num_components as u32;
        let width = self.config.width;
        let height = self.config.height;

        // Create component parameters
        // SAFETY: We're creating a valid array of component parameters.
        let mut comp_params: Vec<opj_sys::opj_image_comp_t> = Vec::with_capacity(num_comps as usize);
        for _ in 0..num_comps {
            comp_params.push(opj_sys::opj_image_comp_t {
                dx: 1,
                dy: 1,
                w: width,
                h: height,
                x0: 0,
                y0: 0,
                prec: self.config.bit_depth as u32,
                bpp: self.config.bit_depth as u32,
                sgnd: if self.config.is_signed { 1 } else { 0 },
                resno_decoded: 0,
                factor: 0,
                data: ptr::null_mut(),
            });
        }

        // Determine color space
        let color_space = if num_comps == 1 {
            opj_sys::OPJ_CLRSPC_GRAY
        } else if num_comps >= 3 {
            opj_sys::OPJ_CLRSPC_SRGB
        } else {
            opj_sys::OPJ_CLRSPC_UNSPECIFIED
        };

        // Create image
        // SAFETY: comp_params is a valid array.
        let image = opj_sys::opj_image_create(num_comps, comp_params.as_ptr(), color_space);

        if image.is_null() {
            return Ok(ptr::null_mut());
        }

        // Set image dimensions
        (*image).x0 = 0;
        (*image).y0 = 0;
        (*image).x1 = width;
        (*image).y1 = height;

        // Copy pixel data to image components
        let bytes_per_sample = (self.config.bit_depth as usize + 7) / 8;
        let num_pixels = (width * height) as usize;

        for c in 0..num_comps as usize {
            // SAFETY: We're iterating within bounds.
            let comp = (*image).comps.add(c);

            // Allocate component data if needed
            if (*comp).data.is_null() {
                // Note: In real code, OpenJPEG manages this memory
                continue;
            }

            // Copy data based on bit depth
            for i in 0..num_pixels {
                let src_idx = (i * num_comps as usize + c) * bytes_per_sample;
                if src_idx + bytes_per_sample <= data.len() {
                    let value = match bytes_per_sample {
                        1 => data[src_idx] as i32,
                        2 => i16::from_ne_bytes([data[src_idx], data[src_idx + 1]]) as i32,
                        _ => data[src_idx] as i32,
                    };
                    *(*comp).data.add(i) = value;
                }
            }
        }

        Ok(image)
    }
}

/// Find the end of the codestream (EOC marker).
fn find_codestream_end(data: &[u8]) -> usize {
    // Look for EOC marker (0xFFD9)
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == 0xFF && data[i + 1] == 0xD9 {
            return i + 2;
        }
    }
    // If no EOC found, return full length
    data.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let result = Jpeg2000FfiDecoder::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_creation() {
        let config = Jpeg2000EncoderConfigFfi::default();
        let result = Jpeg2000FfiEncoder::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let mut config = Jpeg2000EncoderConfigFfi::default();
        config.width = 0;
        let result = Jpeg2000FfiEncoder::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_codestream_end() {
        let data = vec![0x00, 0x01, 0xFF, 0xD9, 0x00, 0x00];
        assert_eq!(find_codestream_end(&data), 4);

        let data_no_eoc = vec![0x00, 0x01, 0x02, 0x03];
        assert_eq!(find_codestream_end(&data_no_eoc), 4);
    }
}
