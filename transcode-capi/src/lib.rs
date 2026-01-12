//! Transcode C API
//!
//! This crate provides a C-compatible API for the Transcode codec library,
//! enabling FFmpeg-like usage patterns from C/C++ applications.
//!
//! # Safety
//!
//! All functions in this module are designed to be called from C code.
//! They handle null pointer checks and return appropriate error codes.
//!
//! # Example Usage (C)
//!
//! ```c
//! #include "transcode.h"
//!
//! int main() {
//!     TranscodeContext* ctx = NULL;
//!     TranscodeError err = transcode_open_input("input.mp4", &ctx);
//!     if (err != TRANSCODE_ERROR_SUCCESS) {
//!         return 1;
//!     }
//!
//!     TranscodePacket* packet = transcode_packet_alloc();
//!     while (transcode_read_packet(ctx, packet) == TRANSCODE_ERROR_SUCCESS) {
//!         // Process packet...
//!     }
//!
//!     transcode_packet_free(packet);
//!     transcode_close(ctx);
//!     return 0;
//! }
//! ```


use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;
use std::slice;

// ============================================================================
// Error Codes
// ============================================================================

/// Error codes returned by transcode functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeError {
    /// Operation completed successfully.
    Success = 0,
    /// Invalid argument provided.
    InvalidArgument = -1,
    /// Null pointer provided where non-null was expected.
    NullPointer = -2,
    /// End of stream reached.
    EndOfStream = -3,
    /// I/O error occurred.
    IoError = -4,
    /// Codec error occurred.
    CodecError = -5,
    /// Container/format error occurred.
    ContainerError = -6,
    /// Resource exhausted (out of memory, etc.).
    ResourceExhausted = -7,
    /// Unsupported feature or format.
    Unsupported = -8,
    /// Operation was cancelled.
    Cancelled = -9,
    /// Buffer too small for operation.
    BufferTooSmall = -10,
    /// Invalid state for this operation.
    InvalidState = -11,
    /// Unknown or internal error.
    Unknown = -100,
}

impl From<transcode_core::Error> for TranscodeError {
    fn from(err: transcode_core::Error) -> Self {
        match err {
            transcode_core::Error::EndOfStream => TranscodeError::EndOfStream,
            transcode_core::Error::Io(_) => TranscodeError::IoError,
            transcode_core::Error::Codec(_) => TranscodeError::CodecError,
            transcode_core::Error::Container(_) => TranscodeError::ContainerError,
            transcode_core::Error::InvalidParameter(_) => TranscodeError::InvalidArgument,
            transcode_core::Error::Unsupported(_) => TranscodeError::Unsupported,
            transcode_core::Error::ResourceExhausted(_) => TranscodeError::ResourceExhausted,
            transcode_core::Error::Cancelled => TranscodeError::Cancelled,
            transcode_core::Error::BufferTooSmall { .. } => TranscodeError::BufferTooSmall,
            _ => TranscodeError::Unknown,
        }
    }
}

// ============================================================================
// Pixel Formats and Color Spaces
// ============================================================================

/// Pixel format for video frames.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodePixelFormat {
    /// Unknown pixel format.
    Unknown = 0,
    /// Planar YUV 4:2:0, 12bpp.
    Yuv420p = 1,
    /// Planar YUV 4:2:2, 16bpp.
    Yuv422p = 2,
    /// Planar YUV 4:4:4, 24bpp.
    Yuv444p = 3,
    /// Planar YUV 4:2:0, 10-bit.
    Yuv420p10le = 4,
    /// Planar YUV 4:2:2, 10-bit.
    Yuv422p10le = 5,
    /// Planar YUV 4:4:4, 10-bit.
    Yuv444p10le = 6,
    /// NV12 (Y plane + interleaved UV).
    Nv12 = 7,
    /// NV21 (Y plane + interleaved VU).
    Nv21 = 8,
    /// Packed RGB, 24bpp.
    Rgb24 = 9,
    /// Packed BGR, 24bpp.
    Bgr24 = 10,
    /// Packed RGBA, 32bpp.
    Rgba = 11,
    /// Packed BGRA, 32bpp.
    Bgra = 12,
    /// Grayscale, 8bpp.
    Gray8 = 13,
    /// Grayscale, 16bpp.
    Gray16 = 14,
}

impl From<transcode_core::PixelFormat> for TranscodePixelFormat {
    fn from(fmt: transcode_core::PixelFormat) -> Self {
        match fmt {
            transcode_core::PixelFormat::Yuv420p => TranscodePixelFormat::Yuv420p,
            transcode_core::PixelFormat::Yuv422p => TranscodePixelFormat::Yuv422p,
            transcode_core::PixelFormat::Yuv444p => TranscodePixelFormat::Yuv444p,
            transcode_core::PixelFormat::Yuv420p10le => TranscodePixelFormat::Yuv420p10le,
            transcode_core::PixelFormat::Yuv422p10le => TranscodePixelFormat::Yuv422p10le,
            transcode_core::PixelFormat::Yuv444p10le => TranscodePixelFormat::Yuv444p10le,
            transcode_core::PixelFormat::Nv12 => TranscodePixelFormat::Nv12,
            transcode_core::PixelFormat::Nv21 => TranscodePixelFormat::Nv21,
            transcode_core::PixelFormat::Rgb24 => TranscodePixelFormat::Rgb24,
            transcode_core::PixelFormat::Bgr24 => TranscodePixelFormat::Bgr24,
            transcode_core::PixelFormat::Rgba => TranscodePixelFormat::Rgba,
            transcode_core::PixelFormat::Bgra => TranscodePixelFormat::Bgra,
            transcode_core::PixelFormat::Gray8 => TranscodePixelFormat::Gray8,
            transcode_core::PixelFormat::Gray16 => TranscodePixelFormat::Gray16,
        }
    }
}

impl From<TranscodePixelFormat> for Option<transcode_core::PixelFormat> {
    fn from(fmt: TranscodePixelFormat) -> Self {
        match fmt {
            TranscodePixelFormat::Unknown => None,
            TranscodePixelFormat::Yuv420p => Some(transcode_core::PixelFormat::Yuv420p),
            TranscodePixelFormat::Yuv422p => Some(transcode_core::PixelFormat::Yuv422p),
            TranscodePixelFormat::Yuv444p => Some(transcode_core::PixelFormat::Yuv444p),
            TranscodePixelFormat::Yuv420p10le => Some(transcode_core::PixelFormat::Yuv420p10le),
            TranscodePixelFormat::Yuv422p10le => Some(transcode_core::PixelFormat::Yuv422p10le),
            TranscodePixelFormat::Yuv444p10le => Some(transcode_core::PixelFormat::Yuv444p10le),
            TranscodePixelFormat::Nv12 => Some(transcode_core::PixelFormat::Nv12),
            TranscodePixelFormat::Nv21 => Some(transcode_core::PixelFormat::Nv21),
            TranscodePixelFormat::Rgb24 => Some(transcode_core::PixelFormat::Rgb24),
            TranscodePixelFormat::Bgr24 => Some(transcode_core::PixelFormat::Bgr24),
            TranscodePixelFormat::Rgba => Some(transcode_core::PixelFormat::Rgba),
            TranscodePixelFormat::Bgra => Some(transcode_core::PixelFormat::Bgra),
            TranscodePixelFormat::Gray8 => Some(transcode_core::PixelFormat::Gray8),
            TranscodePixelFormat::Gray16 => Some(transcode_core::PixelFormat::Gray16),
        }
    }
}

/// Color space for video frames.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeColorSpace {
    /// BT.601 (SD video).
    Bt601 = 0,
    /// BT.709 (HD video).
    Bt709 = 1,
    /// BT.2020 (UHD/HDR video).
    Bt2020 = 2,
    /// sRGB.
    Srgb = 3,
}

impl From<transcode_core::ColorSpace> for TranscodeColorSpace {
    fn from(cs: transcode_core::ColorSpace) -> Self {
        match cs {
            transcode_core::ColorSpace::Bt601 => TranscodeColorSpace::Bt601,
            transcode_core::ColorSpace::Bt709 => TranscodeColorSpace::Bt709,
            transcode_core::ColorSpace::Bt2020 => TranscodeColorSpace::Bt2020,
            transcode_core::ColorSpace::Srgb => TranscodeColorSpace::Srgb,
        }
    }
}

/// Color range for video frames.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeColorRange {
    /// Limited/TV range (16-235 for Y, 16-240 for UV).
    Limited = 0,
    /// Full/PC range (0-255).
    Full = 1,
}

impl From<transcode_core::ColorRange> for TranscodeColorRange {
    fn from(cr: transcode_core::ColorRange) -> Self {
        match cr {
            transcode_core::ColorRange::Limited => TranscodeColorRange::Limited,
            transcode_core::ColorRange::Full => TranscodeColorRange::Full,
        }
    }
}

// ============================================================================
// Stream Types
// ============================================================================

/// Type of media stream.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeStreamType {
    /// Unknown stream type.
    Unknown = 0,
    /// Video stream.
    Video = 1,
    /// Audio stream.
    Audio = 2,
    /// Subtitle stream.
    Subtitle = 3,
    /// Data stream.
    Data = 4,
}

/// Information about a media stream.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TranscodeStreamInfo {
    /// Stream index.
    pub index: u32,
    /// Stream type.
    pub stream_type: TranscodeStreamType,
    /// Codec identifier (fourcc-style).
    pub codec_id: u32,
    /// For video: width in pixels.
    pub width: u32,
    /// For video: height in pixels.
    pub height: u32,
    /// For video: pixel format.
    pub pixel_format: TranscodePixelFormat,
    /// For audio: sample rate in Hz.
    pub sample_rate: u32,
    /// For audio: number of channels.
    pub channels: u32,
    /// For audio: bits per sample.
    pub bits_per_sample: u32,
    /// Time base numerator.
    pub time_base_num: i32,
    /// Time base denominator.
    pub time_base_den: i32,
    /// Duration in time base units (-1 if unknown).
    pub duration: i64,
    /// Bitrate in bits per second (0 if unknown).
    pub bitrate: u64,
}

impl Default for TranscodeStreamInfo {
    fn default() -> Self {
        Self {
            index: 0,
            stream_type: TranscodeStreamType::Unknown,
            codec_id: 0,
            width: 0,
            height: 0,
            pixel_format: TranscodePixelFormat::Unknown,
            sample_rate: 0,
            channels: 0,
            bits_per_sample: 0,
            time_base_num: 1,
            time_base_den: 90000,
            duration: -1,
            bitrate: 0,
        }
    }
}

// ============================================================================
// Core Types with C Layout
// ============================================================================

/// Opaque context handle for transcoding operations.
///
/// This structure manages the state for reading, decoding, encoding,
/// and writing media files.
#[repr(C)]
pub struct TranscodeContext {
    /// Pointer to internal Rust context (opaque to C code).
    _private: *mut c_void,
    /// Input file path (null-terminated).
    input_path: *mut c_char,
    /// Output file path (null-terminated, may be null).
    output_path: *mut c_char,
    /// Number of streams in the input.
    pub num_streams: u32,
    /// Stream information array.
    streams: *mut TranscodeStreamInfo,
    /// Context flags.
    pub flags: u32,
    /// Last error code.
    pub last_error: TranscodeError,
}

/// Packet flags.
pub const TRANSCODE_PACKET_FLAG_KEYFRAME: u32 = 0x0001;
pub const TRANSCODE_PACKET_FLAG_CORRUPT: u32 = 0x0002;
pub const TRANSCODE_PACKET_FLAG_DISCARD: u32 = 0x0004;
pub const TRANSCODE_PACKET_FLAG_DISPOSABLE: u32 = 0x0008;

/// An encoded media packet.
///
/// Contains compressed data before decoding or after encoding.
#[repr(C)]
pub struct TranscodePacket {
    /// Pointer to packet data.
    pub data: *mut u8,
    /// Size of packet data in bytes.
    pub size: usize,
    /// Allocated capacity of data buffer.
    capacity: usize,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Duration in time base units.
    pub duration: i64,
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Packet flags (see TRANSCODE_PACKET_FLAG_*).
    pub flags: u32,
    /// Position in the input stream (bytes), -1 if unknown.
    pub pos: i64,
    /// Time base numerator.
    pub time_base_num: i32,
    /// Time base denominator.
    pub time_base_den: i32,
}

impl Default for TranscodePacket {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            size: 0,
            capacity: 0,
            pts: i64::MIN,
            dts: i64::MIN,
            duration: 0,
            stream_index: 0,
            flags: 0,
            pos: -1,
            time_base_num: 1,
            time_base_den: 90000,
        }
    }
}

/// Frame flags.
pub const TRANSCODE_FRAME_FLAG_KEYFRAME: u32 = 0x0001;
pub const TRANSCODE_FRAME_FLAG_CORRUPT: u32 = 0x0002;
pub const TRANSCODE_FRAME_FLAG_DISCARD: u32 = 0x0004;
pub const TRANSCODE_FRAME_FLAG_INTERLACED: u32 = 0x0008;
pub const TRANSCODE_FRAME_FLAG_TOP_FIELD_FIRST: u32 = 0x0010;

/// Maximum number of planes in a frame.
pub const TRANSCODE_MAX_PLANES: usize = 4;

/// A decoded video frame.
///
/// Contains raw pixel data in a specific format.
#[repr(C)]
pub struct TranscodeFrame {
    /// Pointers to plane data.
    pub data: [*mut u8; TRANSCODE_MAX_PLANES],
    /// Size of each plane in bytes.
    pub linesize: [usize; TRANSCODE_MAX_PLANES],
    /// Number of valid planes.
    pub num_planes: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: TranscodePixelFormat,
    /// Color space.
    pub color_space: TranscodeColorSpace,
    /// Color range.
    pub color_range: TranscodeColorRange,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Frame duration in time base units.
    pub duration: i64,
    /// Frame flags (see TRANSCODE_FRAME_FLAG_*).
    pub flags: u32,
    /// Picture order count (for B-frame reordering).
    pub poc: i32,
    /// Time base numerator.
    pub time_base_num: i32,
    /// Time base denominator.
    pub time_base_den: i32,
    /// Internal buffer (opaque to C code).
    _buffer: *mut c_void,
}

impl Default for TranscodeFrame {
    fn default() -> Self {
        Self {
            data: [ptr::null_mut(); TRANSCODE_MAX_PLANES],
            linesize: [0; TRANSCODE_MAX_PLANES],
            num_planes: 0,
            width: 0,
            height: 0,
            format: TranscodePixelFormat::Unknown,
            color_space: TranscodeColorSpace::Bt601,
            color_range: TranscodeColorRange::Limited,
            pts: i64::MIN,
            dts: i64::MIN,
            duration: 0,
            flags: 0,
            poc: 0,
            time_base_num: 1,
            time_base_den: 90000,
            _buffer: ptr::null_mut(),
        }
    }
}

/// Configuration for encoding/transcoding.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    /// Output width (0 = same as input).
    pub width: u32,
    /// Output height (0 = same as input).
    pub height: u32,
    /// Output pixel format.
    pub pixel_format: TranscodePixelFormat,
    /// Target bitrate in bits per second (0 = auto).
    pub bitrate: u64,
    /// Maximum bitrate for VBR (0 = no limit).
    pub max_bitrate: u64,
    /// Quality level (0-51 for H.264/H.265, codec-specific).
    pub quality: i32,
    /// GOP size (frames between keyframes, 0 = auto).
    pub gop_size: u32,
    /// Number of B-frames (0 = none).
    pub b_frames: u32,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
    /// Sample rate for audio (0 = same as input).
    pub sample_rate: u32,
    /// Number of audio channels (0 = same as input).
    pub channels: u32,
    /// Encoder preset (0-9, lower = slower/better quality).
    pub preset: u32,
    /// Threading mode: 0 = auto, N = use N threads.
    pub threads: u32,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: TranscodePixelFormat::Unknown,
            bitrate: 0,
            max_bitrate: 0,
            quality: 23,
            gop_size: 0,
            b_frames: 0,
            framerate_num: 0,
            framerate_den: 1,
            sample_rate: 0,
            channels: 0,
            preset: 5,
            threads: 0,
        }
    }
}

// ============================================================================
// Internal Context State
// ============================================================================

/// Internal state for the transcode context.
struct InternalContext {
    /// Input file data (for demuxing).
    input_data: Option<Vec<u8>>,
    /// Current read position in input.
    read_pos: usize,
    /// Stream information.
    streams: Vec<TranscodeStreamInfo>,
    /// Decoded frames buffer.
    decoded_frames: Vec<transcode_core::Frame>,
    /// Encoded packets buffer.
    encoded_packets: Vec<transcode_core::Packet<'static>>,
    /// Output configuration.
    config: TranscodeConfig,
    /// Is context opened for reading.
    is_open: bool,
}

impl InternalContext {
    fn new() -> Self {
        Self {
            input_data: None,
            read_pos: 0,
            streams: Vec::new(),
            decoded_frames: Vec::new(),
            encoded_packets: Vec::new(),
            config: TranscodeConfig::default(),
            is_open: false,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the version string of the transcode library.
///
/// Returns a null-terminated string. The caller must not free this pointer.
///
/// # Safety
///
/// This function is safe to call from any context. The returned pointer is valid
/// for the lifetime of the program and must not be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn transcode_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Get a human-readable error message for an error code.
///
/// Returns a null-terminated string. The caller must not free this pointer.
///
/// # Safety
///
/// This function is safe to call from any context. The returned pointer is valid
/// for the lifetime of the program and must not be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn transcode_error_string(error: TranscodeError) -> *const c_char {
    let msg: &'static [u8] = match error {
        TranscodeError::Success => b"Success\0",
        TranscodeError::InvalidArgument => b"Invalid argument\0",
        TranscodeError::NullPointer => b"Null pointer\0",
        TranscodeError::EndOfStream => b"End of stream\0",
        TranscodeError::IoError => b"I/O error\0",
        TranscodeError::CodecError => b"Codec error\0",
        TranscodeError::ContainerError => b"Container error\0",
        TranscodeError::ResourceExhausted => b"Resource exhausted\0",
        TranscodeError::Unsupported => b"Unsupported operation\0",
        TranscodeError::Cancelled => b"Operation cancelled\0",
        TranscodeError::BufferTooSmall => b"Buffer too small\0",
        TranscodeError::InvalidState => b"Invalid state\0",
        TranscodeError::Unknown => b"Unknown error\0",
    };
    msg.as_ptr() as *const c_char
}

// ============================================================================
// Context Management
// ============================================================================

/// Open an input file for reading.
///
/// # Arguments
///
/// * `path` - Path to the input file (null-terminated string).
/// * `ctx` - Pointer to receive the allocated context.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `path` must be a valid pointer to a null-terminated C string, or null.
/// - `ctx` must be a valid pointer to a `*mut TranscodeContext`, or null.
/// - If successful, the caller is responsible for freeing the context with `transcode_close`.
#[no_mangle]
pub unsafe extern "C" fn transcode_open_input(
    path: *const c_char,
    ctx: *mut *mut TranscodeContext,
) -> TranscodeError {
    // Validate arguments
    if path.is_null() || ctx.is_null() {
        return TranscodeError::NullPointer;
    }

    // Convert path to Rust string
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return TranscodeError::InvalidArgument,
    };

    // Read the input file
    let input_data = match std::fs::read(path_str) {
        Ok(data) => data,
        Err(_) => return TranscodeError::IoError,
    };

    // Create internal context
    let mut internal = Box::new(InternalContext::new());
    internal.input_data = Some(input_data);
    internal.is_open = true;

    // Detect streams (simplified - in real implementation would parse container)
    // For now, create a placeholder video stream
    let stream_info = TranscodeStreamInfo {
        index: 0,
        stream_type: TranscodeStreamType::Video,
        codec_id: 0x48323634, // "H264"
        width: 1920,
        height: 1080,
        pixel_format: TranscodePixelFormat::Yuv420p,
        sample_rate: 0,
        channels: 0,
        bits_per_sample: 0,
        time_base_num: 1,
        time_base_den: 90000,
        duration: -1,
        bitrate: 0,
    };
    internal.streams.push(stream_info);

    // Allocate and copy path
    let path_len = path_str.len() + 1;
    let path_copy = libc::malloc(path_len) as *mut c_char;
    if path_copy.is_null() {
        return TranscodeError::ResourceExhausted;
    }
    ptr::copy_nonoverlapping(path.cast::<u8>(), path_copy.cast::<u8>(), path_len);

    // Allocate streams array
    let streams_ptr = if !internal.streams.is_empty() {
        let streams_size = internal.streams.len() * std::mem::size_of::<TranscodeStreamInfo>();
        let ptr = libc::malloc(streams_size) as *mut TranscodeStreamInfo;
        if !ptr.is_null() {
            for (i, stream) in internal.streams.iter().enumerate() {
                ptr::write(ptr.add(i), stream.clone());
            }
        }
        ptr
    } else {
        ptr::null_mut()
    };

    // Create the context
    let context = Box::new(TranscodeContext {
        _private: Box::into_raw(internal) as *mut c_void,
        input_path: path_copy,
        output_path: ptr::null_mut(),
        num_streams: 1,
        streams: streams_ptr,
        flags: 0,
        last_error: TranscodeError::Success,
    });

    *ctx = Box::into_raw(context);
    TranscodeError::Success
}

/// Open an output file for writing.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `path` - Path to the output file (null-terminated string).
/// * `config` - Optional encoding configuration (may be null for defaults).
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to a `TranscodeContext` previously created by `transcode_open_input`, or null.
/// - `path` must be a valid pointer to a null-terminated C string, or null.
/// - `config` may be null, or must be a valid pointer to a `TranscodeConfig`.
#[no_mangle]
pub unsafe extern "C" fn transcode_open_output(
    ctx: *mut TranscodeContext,
    path: *const c_char,
    config: *const TranscodeConfig,
) -> TranscodeError {
    if ctx.is_null() || path.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let internal = &mut *(ctx._private as *mut InternalContext);

    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Convert and store output path
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return TranscodeError::InvalidArgument,
    };

    let path_len = path_str.len() + 1;
    let path_copy = libc::malloc(path_len) as *mut c_char;
    if path_copy.is_null() {
        return TranscodeError::ResourceExhausted;
    }
    ptr::copy_nonoverlapping(path.cast::<u8>(), path_copy.cast::<u8>(), path_len);

    // Free old output path if exists
    if !ctx.output_path.is_null() {
        libc::free(ctx.output_path as *mut c_void);
    }
    ctx.output_path = path_copy;

    // Apply configuration if provided
    if !config.is_null() {
        internal.config = (*config).clone();
    }

    TranscodeError::Success
}

/// Close the transcode context and free all resources.
///
/// # Arguments
///
/// * `ctx` - The context to close (may be null).
///
/// # Safety
///
/// - `ctx` must be null or a valid pointer to a `TranscodeContext` previously created by `transcode_open_input`.
/// - After this call, `ctx` is invalid and must not be used.
/// - It is safe to call this function with a null pointer (no-op).
#[no_mangle]
pub unsafe extern "C" fn transcode_close(ctx: *mut TranscodeContext) {
    if ctx.is_null() {
        return;
    }

    let ctx = Box::from_raw(ctx);

    // Free internal context
    if !ctx._private.is_null() {
        let _ = Box::from_raw(ctx._private as *mut InternalContext);
    }

    // Free paths
    if !ctx.input_path.is_null() {
        libc::free(ctx.input_path as *mut c_void);
    }
    if !ctx.output_path.is_null() {
        libc::free(ctx.output_path as *mut c_void);
    }

    // Free streams array
    if !ctx.streams.is_null() {
        libc::free(ctx.streams as *mut c_void);
    }

    // ctx is automatically dropped here
}

/// Get stream information.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `stream_index` - Index of the stream to query.
/// * `info` - Pointer to receive stream information.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to a `TranscodeContext`, or null.
/// - `info` must be a valid pointer to a `TranscodeStreamInfo` with sufficient space, or null.
/// - `stream_index` must be less than `ctx.num_streams`.
#[no_mangle]
pub unsafe extern "C" fn transcode_get_stream_info(
    ctx: *const TranscodeContext,
    stream_index: u32,
    info: *mut TranscodeStreamInfo,
) -> TranscodeError {
    if ctx.is_null() || info.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &*ctx;
    if stream_index >= ctx.num_streams {
        return TranscodeError::InvalidArgument;
    }

    if ctx.streams.is_null() {
        return TranscodeError::InvalidState;
    }

    let stream = &*ctx.streams.add(stream_index as usize);
    ptr::write(info, stream.clone());

    TranscodeError::Success
}

// ============================================================================
// Packet Management
// ============================================================================

/// Allocate a new packet.
///
/// # Returns
///
/// A pointer to the allocated packet, or null on failure.
///
/// # Safety
///
/// This function is safe to call from any context.
/// The caller is responsible for freeing the returned packet with `transcode_packet_free`.
#[no_mangle]
pub unsafe extern "C" fn transcode_packet_alloc() -> *mut TranscodePacket {
    let packet = Box::new(TranscodePacket::default());
    Box::into_raw(packet)
}

/// Free a packet and its data.
///
/// # Arguments
///
/// * `packet` - The packet to free (may be null).
///
/// # Safety
///
/// - `packet` must be null or a valid pointer previously returned by `transcode_packet_alloc`.
/// - After this call, `packet` is invalid and must not be used.
/// - It is safe to call this function with a null pointer (no-op).
#[no_mangle]
pub unsafe extern "C" fn transcode_packet_free(packet: *mut TranscodePacket) {
    if packet.is_null() {
        return;
    }

    let packet = Box::from_raw(packet);
    if !packet.data.is_null() && packet.capacity > 0 {
        libc::free(packet.data as *mut c_void);
    }
}

/// Reallocate packet data to ensure capacity.
///
/// # Arguments
///
/// * `packet` - The packet to resize.
/// * `size` - Required size in bytes.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `packet` must be a valid pointer to a `TranscodePacket` previously allocated by `transcode_packet_alloc`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_packet_grow(
    packet: *mut TranscodePacket,
    size: usize,
) -> TranscodeError {
    if packet.is_null() {
        return TranscodeError::NullPointer;
    }

    let packet = &mut *packet;
    if packet.capacity >= size {
        return TranscodeError::Success;
    }

    // Allocate new buffer with some extra capacity
    let new_capacity = size.max(packet.capacity * 2).max(1024);
    let new_data = libc::malloc(new_capacity) as *mut u8;
    if new_data.is_null() {
        return TranscodeError::ResourceExhausted;
    }

    // Copy existing data if any
    if !packet.data.is_null() && packet.size > 0 {
        ptr::copy_nonoverlapping(packet.data, new_data, packet.size);
        libc::free(packet.data as *mut c_void);
    }

    packet.data = new_data;
    packet.capacity = new_capacity;
    TranscodeError::Success
}

/// Read the next packet from the input.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `packet` - Packet to receive the data.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, `TRANSCODE_ERROR_END_OF_STREAM` at end,
/// or another error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to a `TranscodeContext` that has been opened, or null.
/// - `packet` must be a valid pointer to a `TranscodePacket`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_read_packet(
    ctx: *mut TranscodeContext,
    packet: *mut TranscodePacket,
) -> TranscodeError {
    if ctx.is_null() || packet.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let internal = &mut *(ctx._private as *mut InternalContext);
    let packet = &mut *packet;

    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Check if we have input data
    let input_data = match &internal.input_data {
        Some(data) => data,
        None => return TranscodeError::InvalidState,
    };

    // Check for end of stream
    if internal.read_pos >= input_data.len() {
        ctx.last_error = TranscodeError::EndOfStream;
        return TranscodeError::EndOfStream;
    }

    // Read a chunk of data (simplified - real implementation would parse NAL units)
    let chunk_size = 4096.min(input_data.len() - internal.read_pos);

    // Ensure packet has enough capacity
    let err = transcode_packet_grow(packet, chunk_size);
    if err != TranscodeError::Success {
        return err;
    }

    // Copy data to packet
    ptr::copy_nonoverlapping(
        input_data.as_ptr().add(internal.read_pos),
        packet.data,
        chunk_size,
    );
    packet.size = chunk_size;
    packet.stream_index = 0;
    packet.pts = (internal.read_pos as i64 / 4096) * 3000; // Simplified PTS calculation
    packet.dts = packet.pts;
    packet.duration = 3000; // ~33ms at 90kHz
    packet.pos = internal.read_pos as i64;

    internal.read_pos += chunk_size;
    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

/// Write a packet to the output.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `packet` - The packet to write.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to a `TranscodeContext` with an output opened, or null.
/// - `packet` must be a valid pointer to a `TranscodePacket` with valid data, or null.
/// - `packet.data` must point to at least `packet.size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn transcode_write_packet(
    ctx: *mut TranscodeContext,
    packet: *const TranscodePacket,
) -> TranscodeError {
    if ctx.is_null() || packet.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let packet = &*packet;

    if ctx.output_path.is_null() {
        return TranscodeError::InvalidState;
    }

    if packet.data.is_null() || packet.size == 0 {
        return TranscodeError::InvalidArgument;
    }

    // Get output path
    let path_str = match CStr::from_ptr(ctx.output_path).to_str() {
        Ok(s) => s,
        Err(_) => return TranscodeError::InvalidArgument,
    };

    // Get packet data as slice
    let data = slice::from_raw_parts(packet.data, packet.size);

    // Append to output file
    use std::io::Write;
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path_str)
    {
        Ok(f) => f,
        Err(_) => return TranscodeError::IoError,
    };

    match file.write_all(data) {
        Ok(_) => TranscodeError::Success,
        Err(_) => TranscodeError::IoError,
    }
}

// ============================================================================
// Frame Management
// ============================================================================

/// Allocate a new frame.
///
/// # Returns
///
/// A pointer to the allocated frame, or null on failure.
///
/// # Safety
///
/// This function is safe to call from any context.
/// The caller is responsible for freeing the returned frame with `transcode_frame_free`.
#[no_mangle]
pub unsafe extern "C" fn transcode_frame_alloc() -> *mut TranscodeFrame {
    let frame = Box::new(TranscodeFrame::default());
    Box::into_raw(frame)
}

/// Allocate frame buffer with specified dimensions.
///
/// # Arguments
///
/// * `frame` - The frame to allocate buffer for.
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `format` - Pixel format.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `frame` must be a valid pointer to a `TranscodeFrame` previously allocated by `transcode_frame_alloc`, or null.
/// - Any existing buffer in the frame will be freed before allocating the new one.
#[no_mangle]
pub unsafe extern "C" fn transcode_frame_alloc_buffer(
    frame: *mut TranscodeFrame,
    width: u32,
    height: u32,
    format: TranscodePixelFormat,
) -> TranscodeError {
    if frame.is_null() {
        return TranscodeError::NullPointer;
    }

    if width == 0 || height == 0 {
        return TranscodeError::InvalidArgument;
    }

    let rust_format: Option<transcode_core::PixelFormat> = format.into();
    let rust_format = match rust_format {
        Some(f) => f,
        None => return TranscodeError::InvalidArgument,
    };

    let frame = &mut *frame;

    // Free existing buffer if any
    transcode_frame_free_buffer(frame);

    // Create Rust frame buffer
    let buffer = transcode_core::FrameBuffer::new(width, height, rust_format);
    let boxed_buffer = Box::new(buffer);

    // Get plane information
    let num_planes = rust_format.num_planes();
    frame.num_planes = num_planes as u32;
    frame.width = width;
    frame.height = height;
    frame.format = format;

    // We need to keep the buffer alive and provide pointers to its data
    // Store the buffer pointer for later cleanup
    frame._buffer = Box::into_raw(boxed_buffer) as *mut c_void;

    // Get pointers to plane data
    let buffer = &mut *(frame._buffer as *mut transcode_core::FrameBuffer);
    for i in 0..num_planes {
        if let Some(plane) = buffer.plane_mut(i) {
            frame.data[i] = plane.as_mut_ptr();
            frame.linesize[i] = buffer.stride(i);
        }
    }

    TranscodeError::Success
}

/// Free frame buffer data (but not the frame structure itself).
///
/// # Arguments
///
/// * `frame` - The frame whose buffer to free.
///
/// # Safety
///
/// - `frame` must be null or a valid pointer to a `TranscodeFrame`.
/// - After this call, the frame's data pointers are invalidated but the frame structure remains valid.
/// - It is safe to call this function with a null pointer (no-op).
#[no_mangle]
pub unsafe extern "C" fn transcode_frame_free_buffer(frame: *mut TranscodeFrame) {
    if frame.is_null() {
        return;
    }

    let frame = &mut *frame;
    if !frame._buffer.is_null() {
        let _ = Box::from_raw(frame._buffer as *mut transcode_core::FrameBuffer);
        frame._buffer = ptr::null_mut();
    }

    for i in 0..TRANSCODE_MAX_PLANES {
        frame.data[i] = ptr::null_mut();
        frame.linesize[i] = 0;
    }
    frame.num_planes = 0;
}

/// Free a frame and its data.
///
/// # Arguments
///
/// * `frame` - The frame to free (may be null).
///
/// # Safety
///
/// - `frame` must be null or a valid pointer previously returned by `transcode_frame_alloc`.
/// - After this call, `frame` is invalid and must not be used.
/// - It is safe to call this function with a null pointer (no-op).
#[no_mangle]
pub unsafe extern "C" fn transcode_frame_free(frame: *mut TranscodeFrame) {
    if frame.is_null() {
        return;
    }

    transcode_frame_free_buffer(frame);
    let _ = Box::from_raw(frame);
}

/// Copy frame data from source to destination.
///
/// # Arguments
///
/// * `dst` - Destination frame (must have buffer allocated).
/// * `src` - Source frame.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `dst` must be a valid pointer to a `TranscodeFrame` with an allocated buffer, or null.
/// - `src` must be a valid pointer to a `TranscodeFrame` with valid data, or null.
/// - Both frames must have the same dimensions and pixel format.
#[no_mangle]
pub unsafe extern "C" fn transcode_frame_copy(
    dst: *mut TranscodeFrame,
    src: *const TranscodeFrame,
) -> TranscodeError {
    if dst.is_null() || src.is_null() {
        return TranscodeError::NullPointer;
    }

    let dst = &mut *dst;
    let src = &*src;

    if dst.width != src.width || dst.height != src.height || dst.format != src.format {
        return TranscodeError::InvalidArgument;
    }

    // Copy plane data
    for i in 0..src.num_planes as usize {
        if src.data[i].is_null() || dst.data[i].is_null() {
            continue;
        }
        let size = src.linesize[i] * src.height as usize;
        ptr::copy_nonoverlapping(src.data[i], dst.data[i], size);
    }

    // Copy metadata
    dst.pts = src.pts;
    dst.dts = src.dts;
    dst.duration = src.duration;
    dst.flags = src.flags;
    dst.poc = src.poc;
    dst.color_space = src.color_space;
    dst.color_range = src.color_range;
    dst.time_base_num = src.time_base_num;
    dst.time_base_den = src.time_base_den;

    TranscodeError::Success
}

// ============================================================================
// Decoding and Encoding
// ============================================================================

/// Decode a packet into a frame.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `packet` - The packet to decode.
/// * `frame` - Frame to receive the decoded data.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
/// May return `TRANSCODE_ERROR_RESOURCE_EXHAUSTED` if more input is needed.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to an opened `TranscodeContext`, or null.
/// - `packet` must be a valid pointer to a `TranscodePacket` with valid data, or null.
/// - `frame` must be a valid pointer to a `TranscodeFrame`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_decode_packet(
    ctx: *mut TranscodeContext,
    packet: *const TranscodePacket,
    frame: *mut TranscodeFrame,
) -> TranscodeError {
    if ctx.is_null() || packet.is_null() || frame.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let packet = &*packet;
    let frame = &mut *frame;

    let internal = &mut *(ctx._private as *mut InternalContext);
    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Simplified decode - in real implementation would use actual decoder
    // For now, just create a test pattern frame

    // Allocate frame buffer if needed
    if frame._buffer.is_null() {
        let err = transcode_frame_alloc_buffer(
            frame,
            1920,
            1080,
            TranscodePixelFormat::Yuv420p,
        );
        if err != TranscodeError::Success {
            return err;
        }
    }

    // Set frame metadata from packet
    frame.pts = packet.pts;
    frame.dts = packet.dts;
    frame.duration = packet.duration;
    frame.time_base_num = packet.time_base_num;
    frame.time_base_den = packet.time_base_den;

    // Set keyframe flag if packet is keyframe
    if packet.flags & TRANSCODE_PACKET_FLAG_KEYFRAME != 0 {
        frame.flags |= TRANSCODE_FRAME_FLAG_KEYFRAME;
    }

    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

/// Encode a frame into a packet.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `frame` - The frame to encode (may be null to flush encoder).
/// * `packet` - Packet to receive the encoded data.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
/// May return `TRANSCODE_ERROR_RESOURCE_EXHAUSTED` if no output is available yet.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to an opened `TranscodeContext`, or null.
/// - `frame` may be null (to flush the encoder), or must be a valid pointer to a `TranscodeFrame`.
/// - `packet` must be a valid pointer to a `TranscodePacket`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_encode_frame(
    ctx: *mut TranscodeContext,
    frame: *const TranscodeFrame,
    packet: *mut TranscodePacket,
) -> TranscodeError {
    if ctx.is_null() || packet.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let packet = &mut *packet;

    let internal = &mut *(ctx._private as *mut InternalContext);
    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Handle flush (null frame)
    if frame.is_null() {
        return TranscodeError::EndOfStream;
    }

    let frame = &*frame;

    // Simplified encode - in real implementation would use actual encoder
    // For now, just copy frame data as "encoded" packet

    // Calculate output size (simplified)
    let output_size = (frame.width * frame.height) as usize / 10; // Rough compression ratio

    // Ensure packet has capacity
    let err = transcode_packet_grow(packet, output_size);
    if err != TranscodeError::Success {
        return err;
    }

    // Fill with placeholder data
    if !packet.data.is_null() {
        for i in 0..output_size {
            *packet.data.add(i) = (i & 0xFF) as u8;
        }
    }
    packet.size = output_size;

    // Copy timestamps
    packet.pts = frame.pts;
    packet.dts = frame.dts;
    packet.duration = frame.duration;
    packet.time_base_num = frame.time_base_num;
    packet.time_base_den = frame.time_base_den;
    packet.stream_index = 0;

    // Set keyframe flag if frame is keyframe
    if frame.flags & TRANSCODE_FRAME_FLAG_KEYFRAME != 0 {
        packet.flags |= TRANSCODE_PACKET_FLAG_KEYFRAME;
    }

    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

// ============================================================================
// Seek Operations
// ============================================================================

/// Seek to a specific timestamp in the input.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
/// * `stream_index` - Stream to seek in (-1 for default).
/// * `timestamp` - Target timestamp in stream time base.
/// * `flags` - Seek flags (reserved for future use, pass 0).
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to an opened `TranscodeContext`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_seek(
    ctx: *mut TranscodeContext,
    stream_index: c_int,
    timestamp: i64,
    _flags: c_int,
) -> TranscodeError {
    if ctx.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let internal = &mut *(ctx._private as *mut InternalContext);

    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Simplified seek - just update read position
    let input_data = match &internal.input_data {
        Some(data) => data,
        None => return TranscodeError::InvalidState,
    };

    // Convert timestamp to byte position (simplified)
    let _ = stream_index; // Unused in this simplified implementation
    let new_pos = (timestamp / 3000 * 4096) as usize;

    if new_pos >= input_data.len() {
        return TranscodeError::InvalidArgument;
    }

    internal.read_pos = new_pos;
    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

// ============================================================================
// Flush Operations
// ============================================================================

/// Flush decoder buffers.
///
/// Call this when seeking or at end of stream to get remaining frames.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to an opened `TranscodeContext`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_flush_decoder(ctx: *mut TranscodeContext) -> TranscodeError {
    if ctx.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let internal = &mut *(ctx._private as *mut InternalContext);

    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Clear any buffered decoded frames
    internal.decoded_frames.clear();

    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

/// Flush encoder buffers.
///
/// Call this at end of stream to get remaining packets.
///
/// # Arguments
///
/// * `ctx` - The transcode context.
///
/// # Returns
///
/// `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer to an opened `TranscodeContext`, or null.
#[no_mangle]
pub unsafe extern "C" fn transcode_flush_encoder(ctx: *mut TranscodeContext) -> TranscodeError {
    if ctx.is_null() {
        return TranscodeError::NullPointer;
    }

    let ctx = &mut *ctx;
    let internal = &mut *(ctx._private as *mut InternalContext);

    if !internal.is_open {
        return TranscodeError::InvalidState;
    }

    // Clear any buffered encoded packets
    internal.encoded_packets.clear();

    ctx.last_error = TranscodeError::Success;
    TranscodeError::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_string() {
        unsafe {
            let s = transcode_error_string(TranscodeError::Success);
            assert!(!s.is_null());
            let cstr = CStr::from_ptr(s);
            assert_eq!(cstr.to_str().unwrap(), "Success");
        }
    }

    #[test]
    fn test_packet_alloc_free() {
        unsafe {
            let packet = transcode_packet_alloc();
            assert!(!packet.is_null());
            assert!((*packet).data.is_null());
            assert_eq!((*packet).size, 0);
            transcode_packet_free(packet);
        }
    }

    #[test]
    fn test_packet_grow() {
        unsafe {
            let packet = transcode_packet_alloc();
            assert!(!packet.is_null());

            let err = transcode_packet_grow(packet, 1024);
            assert_eq!(err, TranscodeError::Success);
            assert!(!(*packet).data.is_null());
            assert!((*packet).capacity >= 1024);

            transcode_packet_free(packet);
        }
    }

    #[test]
    fn test_frame_alloc_free() {
        unsafe {
            let frame = transcode_frame_alloc();
            assert!(!frame.is_null());
            assert_eq!((*frame).width, 0);
            assert_eq!((*frame).height, 0);
            transcode_frame_free(frame);
        }
    }

    #[test]
    fn test_frame_alloc_buffer() {
        unsafe {
            let frame = transcode_frame_alloc();
            assert!(!frame.is_null());

            let err = transcode_frame_alloc_buffer(
                frame,
                1920,
                1080,
                TranscodePixelFormat::Yuv420p,
            );
            assert_eq!(err, TranscodeError::Success);
            assert_eq!((*frame).width, 1920);
            assert_eq!((*frame).height, 1080);
            assert_eq!((*frame).num_planes, 3);
            assert!(!(*frame).data[0].is_null());
            assert!(!(*frame).data[1].is_null());
            assert!(!(*frame).data[2].is_null());

            transcode_frame_free(frame);
        }
    }

    #[test]
    fn test_version() {
        unsafe {
            let v = transcode_version();
            assert!(!v.is_null());
            let cstr = CStr::from_ptr(v);
            assert!(cstr.to_str().unwrap().contains('.'));
        }
    }

    #[test]
    fn test_null_pointer_handling() {
        unsafe {
            assert_eq!(
                transcode_open_input(ptr::null(), ptr::null_mut()),
                TranscodeError::NullPointer
            );
            assert_eq!(
                transcode_read_packet(ptr::null_mut(), ptr::null_mut()),
                TranscodeError::NullPointer
            );
            assert_eq!(
                transcode_decode_packet(ptr::null_mut(), ptr::null(), ptr::null_mut()),
                TranscodeError::NullPointer
            );
            transcode_close(ptr::null_mut()); // Should not crash
            transcode_packet_free(ptr::null_mut()); // Should not crash
            transcode_frame_free(ptr::null_mut()); // Should not crash
        }
    }
}
