//! VideoToolbox implementation for macOS.
//!
//! This module provides hardware-accelerated video encoding and decoding
//! using Apple's VideoToolbox framework. On Apple Silicon Macs, this provides
//! extremely efficient H.264/HEVC encoding with hardware acceleration.
//!
//! # Architecture
//!
//! VideoToolbox uses a callback-based async model:
//! 1. Create a compression/decompression session
//! 2. Submit frames for processing
//! 3. Receive encoded/decoded frames via callbacks
//!
//! We wrap this in a sync API using channels for the callback data.
//!
//! # FFI Design
//!
//! This module provides real FFI bindings to VideoToolbox when the `videotoolbox`
//! feature is enabled. The types mirror Apple's C API:
//! - `VTCompressionSessionRef` for encoding sessions
//! - `VTDecompressionSessionRef` for decoding sessions
//! - `CVPixelBufferRef` for video frame buffers
//! - `CMSampleBufferRef` for compressed samples
//!
//! When the feature is disabled, a mock implementation is used for testing.

// =============================================================================
// Real FFI Bindings Module (enabled with videotoolbox feature)
// =============================================================================

/// Real FFI bindings to Apple's VideoToolbox, CoreVideo, and CoreMedia frameworks.
/// These bindings link directly to the system frameworks on macOS.
#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
pub mod ffi {
    use std::ffi::c_void;

    // -------------------------------------------------------------------------
    // Core Foundation Types
    // -------------------------------------------------------------------------

    /// Opaque Core Foundation object reference.
    pub type CFTypeRef = *const c_void;
    /// Core Foundation allocator reference.
    pub type CFAllocatorRef = *const c_void;
    /// Core Foundation dictionary reference.
    pub type CFDictionaryRef = *const c_void;
    /// Core Foundation mutable dictionary reference.
    pub type CFMutableDictionaryRef = *mut c_void;
    /// Core Foundation string reference.
    pub type CFStringRef = *const c_void;
    /// Core Foundation number reference.
    pub type CFNumberRef = *const c_void;
    /// Core Foundation boolean reference.
    pub type CFBooleanRef = *const c_void;
    /// Core Foundation data reference.
    pub type CFDataRef = *const c_void;
    /// Core Foundation array reference.
    pub type CFArrayRef = *const c_void;
    /// Core Foundation index type.
    pub type CFIndex = isize;

    /// Core Foundation number types.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CFNumberType {
        SInt8 = 1,
        SInt16 = 2,
        SInt32 = 3,
        SInt64 = 4,
        Float32 = 5,
        Float64 = 6,
        Char = 7,
        Short = 8,
        Int = 9,
        Long = 10,
        LongLong = 11,
        Float = 12,
        Double = 13,
        CFIndex = 14,
        NSInteger = 15,
        CGFloat = 16,
    }

    // -------------------------------------------------------------------------
    // CoreVideo Types
    // -------------------------------------------------------------------------

    /// CVPixelBuffer reference.
    pub type CVPixelBufferRef = *mut c_void;
    /// CVImageBuffer reference (CVPixelBuffer inherits from this).
    pub type CVImageBufferRef = *mut c_void;
    /// CVPixelBufferPool reference.
    pub type CVPixelBufferPoolRef = *mut c_void;
    /// CVReturn status code.
    pub type CVReturn = i32;

    /// CVReturn success value.
    pub const K_CV_RETURN_SUCCESS: CVReturn = 0;
    /// CVReturn invalid argument.
    pub const K_CV_RETURN_INVALID_ARGUMENT: CVReturn = -6661;
    /// CVReturn allocation failed.
    pub const K_CV_RETURN_ALLOCATION_FAILED: CVReturn = -6662;

    /// Pixel format types (OSType/FourCharCode).
    pub const K_CV_PIXEL_FORMAT_TYPE_420_YP_CB_CR_8_BI_PLANAR_VIDEO_RANGE: u32 = 0x34323076; // '420v' NV12
    pub const K_CV_PIXEL_FORMAT_TYPE_420_YP_CB_CR_8_BI_PLANAR_FULL_RANGE: u32 = 0x34323066; // '420f'
    pub const K_CV_PIXEL_FORMAT_TYPE_420_YP_CB_CR_10_BI_PLANAR_VIDEO_RANGE: u32 = 0x78343230; // 'x420' P010
    pub const K_CV_PIXEL_FORMAT_TYPE_420_YP_CB_CR_10_BI_PLANAR_FULL_RANGE: u32 = 0x78663230; // 'xf20'
    pub const K_CV_PIXEL_FORMAT_TYPE_32_ARGB: u32 = 0x00000020; // 32
    pub const K_CV_PIXEL_FORMAT_TYPE_32_BGRA: u32 = 0x42475241; // 'BGRA'
    pub const K_CV_PIXEL_FORMAT_TYPE_422_YP_CB_CR_8: u32 = 0x32767579; // '2vuy' UYVY
    pub const K_CV_PIXEL_FORMAT_TYPE_422_YP_CB_CR_8_YUVS: u32 = 0x79757673; // 'yuvs' YUYV

    // -------------------------------------------------------------------------
    // CoreMedia Types
    // -------------------------------------------------------------------------

    /// CMSampleBuffer reference.
    pub type CMSampleBufferRef = *mut c_void;
    /// CMFormatDescription reference.
    pub type CMFormatDescriptionRef = *const c_void;
    /// CMVideoFormatDescription reference.
    pub type CMVideoFormatDescriptionRef = *const c_void;
    /// CMBlockBuffer reference.
    pub type CMBlockBufferRef = *mut c_void;
    /// OSStatus return type.
    pub type OSStatus = i32;

    /// CMTime structure - represents time as a rational number.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CMTime {
        pub value: i64,
        pub timescale: i32,
        pub flags: u32,
        pub epoch: i64,
    }

    /// CMTime flags.
    pub const K_CM_TIME_FLAGS_VALID: u32 = 1 << 0;
    pub const K_CM_TIME_FLAGS_HAS_BEEN_ROUNDED: u32 = 1 << 1;
    pub const K_CM_TIME_FLAGS_POSITIVE_INFINITY: u32 = 1 << 2;
    pub const K_CM_TIME_FLAGS_NEGATIVE_INFINITY: u32 = 1 << 3;
    pub const K_CM_TIME_FLAGS_INDEFINITE: u32 = 1 << 4;

    impl CMTime {
        /// Create a valid CMTime.
        pub fn make(value: i64, timescale: i32) -> Self {
            Self {
                value,
                timescale,
                flags: K_CM_TIME_FLAGS_VALID,
                epoch: 0,
            }
        }

        /// Invalid CMTime.
        pub const INVALID: Self = Self {
            value: 0,
            timescale: 0,
            flags: 0,
            epoch: 0,
        };

        /// Check if valid.
        pub fn is_valid(&self) -> bool {
            (self.flags & K_CM_TIME_FLAGS_VALID) != 0
        }

        /// Convert to seconds.
        pub fn get_seconds(&self) -> f64 {
            if self.timescale == 0 {
                0.0
            } else {
                self.value as f64 / self.timescale as f64
            }
        }
    }

    /// CMTimeRange structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CMTimeRange {
        pub start: CMTime,
        pub duration: CMTime,
    }

    // -------------------------------------------------------------------------
    // VideoToolbox Types
    // -------------------------------------------------------------------------

    /// VTCompressionSession reference (encoding).
    pub type VTCompressionSessionRef = *mut c_void;
    /// VTDecompressionSession reference (decoding).
    pub type VTDecompressionSessionRef = *mut c_void;
    /// VTSession (base type).
    pub type VTSessionRef = *mut c_void;

    /// Encode info flags passed to compression callback.
    pub type VTEncodeInfoFlags = u32;
    /// Decode info flags passed to decompression callback.
    pub type VTDecodeInfoFlags = u32;

    /// Frame was dropped.
    pub const K_VT_ENCODE_INFO_FRAME_DROPPED: VTEncodeInfoFlags = 1 << 0;
    /// Decoding is async.
    pub const K_VT_DECODE_INFO_ASYNCHRONOUS: VTDecodeInfoFlags = 1 << 0;
    /// Frame was dropped.
    pub const K_VT_DECODE_INFO_FRAME_DROPPED: VTDecodeInfoFlags = 1 << 1;

    /// Compression output callback signature.
    pub type VTCompressionOutputCallback = extern "C" fn(
        output_callback_ref_con: *mut c_void,
        source_frame_ref_con: *mut c_void,
        status: OSStatus,
        info_flags: VTEncodeInfoFlags,
        sample_buffer: CMSampleBufferRef,
    );

    /// Decompression output callback record.
    #[repr(C)]
    pub struct VTDecompressionOutputCallbackRecord {
        pub decompression_output_callback: VTDecompressionOutputCallback,
        pub decompression_output_ref_con: *mut c_void,
    }

    /// Decompression output callback signature.
    pub type VTDecompressionOutputCallback = extern "C" fn(
        decompression_output_ref_con: *mut c_void,
        source_frame_ref_con: *mut c_void,
        status: OSStatus,
        info_flags: VTDecodeInfoFlags,
        image_buffer: CVImageBufferRef,
        presentation_time_stamp: CMTime,
        presentation_duration: CMTime,
    );

    /// Decode frame flags.
    pub type VTDecodeFrameFlags = u32;
    /// Enable async decoding.
    pub const K_VT_DECODE_FRAME_ENABLE_ASYNC_DECODE: VTDecodeFrameFlags = 1 << 0;
    /// Do not output frame.
    pub const K_VT_DECODE_FRAME_DO_NOT_OUTPUT_FRAME: VTDecodeFrameFlags = 1 << 1;

    // -------------------------------------------------------------------------
    // FFI Function Declarations
    // -------------------------------------------------------------------------

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        /// Default allocator.
        pub static kCFAllocatorDefault: CFAllocatorRef;
        /// Boolean true.
        pub static kCFBooleanTrue: CFBooleanRef;
        /// Boolean false.
        pub static kCFBooleanFalse: CFBooleanRef;

        /// Retain a CF object.
        pub fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
        /// Release a CF object.
        pub fn CFRelease(cf: CFTypeRef);
        /// Get retain count.
        pub fn CFGetRetainCount(cf: CFTypeRef) -> CFIndex;

        /// Create a mutable dictionary.
        pub fn CFDictionaryCreateMutable(
            allocator: CFAllocatorRef,
            capacity: CFIndex,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> CFMutableDictionaryRef;

        /// Set a dictionary value.
        pub fn CFDictionarySetValue(
            dict: CFMutableDictionaryRef,
            key: *const c_void,
            value: *const c_void,
        );

        /// Get a dictionary value.
        pub fn CFDictionaryGetValue(dict: CFDictionaryRef, key: *const c_void) -> *const c_void;

        /// Get dictionary count.
        pub fn CFDictionaryGetCount(dict: CFDictionaryRef) -> CFIndex;

        /// Create a number.
        pub fn CFNumberCreate(
            allocator: CFAllocatorRef,
            number_type: CFNumberType,
            value_ptr: *const c_void,
        ) -> CFNumberRef;

        /// Get number value.
        pub fn CFNumberGetValue(
            number: CFNumberRef,
            number_type: CFNumberType,
            value_ptr: *mut c_void,
        ) -> bool;

        /// Create a string from C string.
        pub fn CFStringCreateWithCString(
            allocator: CFAllocatorRef,
            c_str: *const i8,
            encoding: u32,
        ) -> CFStringRef;

        /// UTF-8 encoding constant.
        pub static kCFStringEncodingUTF8: u32;

        /// Get data length.
        pub fn CFDataGetLength(data: CFDataRef) -> CFIndex;
        /// Get data bytes.
        pub fn CFDataGetBytePtr(data: CFDataRef) -> *const u8;

        /// Get array count.
        pub fn CFArrayGetCount(array: CFArrayRef) -> CFIndex;
        /// Get array value at index.
        pub fn CFArrayGetValueAtIndex(array: CFArrayRef, idx: CFIndex) -> *const c_void;
    }

    #[link(name = "CoreVideo", kind = "framework")]
    extern "C" {
        /// Pixel format type key.
        pub static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
        /// Width key.
        pub static kCVPixelBufferWidthKey: CFStringRef;
        /// Height key.
        pub static kCVPixelBufferHeightKey: CFStringRef;
        /// IOSurface properties key.
        pub static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
        /// Metal compatibility key (macOS 10.11+).
        pub static kCVPixelBufferMetalCompatibilityKey: CFStringRef;
        /// OpenGL compatibility key.
        pub static kCVPixelBufferOpenGLCompatibilityKey: CFStringRef;
        /// Bytes per row alignment key.
        pub static kCVPixelBufferBytesPerRowAlignmentKey: CFStringRef;

        /// Create a pixel buffer.
        pub fn CVPixelBufferCreate(
            allocator: CFAllocatorRef,
            width: usize,
            height: usize,
            pixel_format_type: u32,
            pixel_buffer_attributes: CFDictionaryRef,
            pixel_buffer_out: *mut CVPixelBufferRef,
        ) -> CVReturn;

        /// Create a pixel buffer pool.
        pub fn CVPixelBufferPoolCreate(
            allocator: CFAllocatorRef,
            pool_attributes: CFDictionaryRef,
            pixel_buffer_attributes: CFDictionaryRef,
            pool_out: *mut CVPixelBufferPoolRef,
        ) -> CVReturn;

        /// Create a pixel buffer from a pool.
        pub fn CVPixelBufferPoolCreatePixelBuffer(
            allocator: CFAllocatorRef,
            pool: CVPixelBufferPoolRef,
            pixel_buffer_out: *mut CVPixelBufferRef,
        ) -> CVReturn;

        /// Flush pool (preallocate mode).
        pub fn CVPixelBufferPoolFlush(pool: CVPixelBufferPoolRef, options: u64);

        /// Release pool.
        pub fn CVPixelBufferPoolRelease(pool: CVPixelBufferPoolRef);
        /// Retain pool.
        pub fn CVPixelBufferPoolRetain(pool: CVPixelBufferPoolRef) -> CVPixelBufferPoolRef;

        /// Get pixel buffer width.
        pub fn CVPixelBufferGetWidth(pixel_buffer: CVPixelBufferRef) -> usize;
        /// Get pixel buffer height.
        pub fn CVPixelBufferGetHeight(pixel_buffer: CVPixelBufferRef) -> usize;
        /// Get pixel format type.
        pub fn CVPixelBufferGetPixelFormatType(pixel_buffer: CVPixelBufferRef) -> u32;
        /// Get bytes per row.
        pub fn CVPixelBufferGetBytesPerRow(pixel_buffer: CVPixelBufferRef) -> usize;
        /// Get base address (requires lock).
        pub fn CVPixelBufferGetBaseAddress(pixel_buffer: CVPixelBufferRef) -> *mut c_void;
        /// Get plane count.
        pub fn CVPixelBufferGetPlaneCount(pixel_buffer: CVPixelBufferRef) -> usize;
        /// Get base address of plane.
        pub fn CVPixelBufferGetBaseAddressOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane_index: usize,
        ) -> *mut c_void;
        /// Get bytes per row of plane.
        pub fn CVPixelBufferGetBytesPerRowOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane_index: usize,
        ) -> usize;
        /// Get width of plane.
        pub fn CVPixelBufferGetWidthOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane_index: usize,
        ) -> usize;
        /// Get height of plane.
        pub fn CVPixelBufferGetHeightOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane_index: usize,
        ) -> usize;
        /// Get data size.
        pub fn CVPixelBufferGetDataSize(pixel_buffer: CVPixelBufferRef) -> usize;

        /// Lock base address for CPU access.
        pub fn CVPixelBufferLockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> CVReturn;
        /// Unlock base address.
        pub fn CVPixelBufferUnlockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> CVReturn;

        /// Lock flags - read only.
        pub static kCVPixelBufferLock_ReadOnly: u64;

        /// Retain pixel buffer.
        pub fn CVPixelBufferRetain(pixel_buffer: CVPixelBufferRef) -> CVPixelBufferRef;
        /// Release pixel buffer.
        pub fn CVPixelBufferRelease(pixel_buffer: CVPixelBufferRef);

        /// Check if pixel buffer is planar.
        pub fn CVPixelBufferIsPlanar(pixel_buffer: CVPixelBufferRef) -> bool;
    }

    #[link(name = "CoreMedia", kind = "framework")]
    extern "C" {
        /// Create CMTime from seconds.
        pub fn CMTimeMakeWithSeconds(seconds: f64, preferred_timescale: i32) -> CMTime;
        /// Get seconds from CMTime.
        pub fn CMTimeGetSeconds(time: CMTime) -> f64;
        /// Compare two CMTimes.
        pub fn CMTimeCompare(time1: CMTime, time2: CMTime) -> i32;

        /// Get format description from sample buffer.
        pub fn CMSampleBufferGetFormatDescription(sbuf: CMSampleBufferRef)
            -> CMFormatDescriptionRef;
        /// Get sample timing info.
        pub fn CMSampleBufferGetSampleTimingInfo(
            sbuf: CMSampleBufferRef,
            sample_index: CFIndex,
            timing_info_out: *mut CMSampleTimingInfo,
        ) -> OSStatus;
        /// Get data buffer.
        pub fn CMSampleBufferGetDataBuffer(sbuf: CMSampleBufferRef) -> CMBlockBufferRef;
        /// Get image buffer (for uncompressed video).
        pub fn CMSampleBufferGetImageBuffer(sbuf: CMSampleBufferRef) -> CVImageBufferRef;
        /// Get total sample size.
        pub fn CMSampleBufferGetTotalSampleSize(sbuf: CMSampleBufferRef) -> usize;
        /// Check if sample buffer is valid.
        pub fn CMSampleBufferIsValid(sbuf: CMSampleBufferRef) -> bool;
        /// Check if sample buffer has data.
        pub fn CMSampleBufferDataIsReady(sbuf: CMSampleBufferRef) -> bool;

        /// Get block buffer data length.
        pub fn CMBlockBufferGetDataLength(buffer: CMBlockBufferRef) -> usize;
        /// Get block buffer data pointer.
        pub fn CMBlockBufferGetDataPointer(
            buffer: CMBlockBufferRef,
            offset: usize,
            length_at_offset_out: *mut usize,
            total_length_out: *mut usize,
            data_pointer_out: *mut *mut i8,
        ) -> OSStatus;

        /// Get video format description dimensions.
        pub fn CMVideoFormatDescriptionGetDimensions(
            video_desc: CMVideoFormatDescriptionRef,
        ) -> CMVideoDimensions;
        /// Get codec type from video format description.
        pub fn CMFormatDescriptionGetMediaSubType(format_desc: CMFormatDescriptionRef) -> u32;
    }

    /// CMSampleTimingInfo structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CMSampleTimingInfo {
        pub duration: CMTime,
        pub presentation_time_stamp: CMTime,
        pub decode_time_stamp: CMTime,
    }

    /// CMVideoDimensions structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CMVideoDimensions {
        pub width: i32,
        pub height: i32,
    }

    #[link(name = "VideoToolbox", kind = "framework")]
    extern "C" {
        // --- Property keys ---

        /// Average bit rate key.
        pub static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
        /// Data rate limits key.
        pub static kVTCompressionPropertyKey_DataRateLimits: CFStringRef;
        /// Quality key.
        pub static kVTCompressionPropertyKey_Quality: CFStringRef;
        /// Max key frame interval key.
        pub static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
        /// Max key frame interval duration key.
        pub static kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration: CFStringRef;
        /// Allow frame reordering key.
        pub static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
        /// Profile level key.
        pub static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
        /// H.264 entropy mode key.
        pub static kVTCompressionPropertyKey_H264EntropyMode: CFStringRef;
        /// Expected frame rate key.
        pub static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
        /// Real time key.
        pub static kVTCompressionPropertyKey_RealTime: CFStringRef;
        /// Allow temporal compression key.
        pub static kVTCompressionPropertyKey_AllowTemporalCompression: CFStringRef;
        /// Prioritize encoding speed key.
        pub static kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality: CFStringRef;
        /// Constant bit rate key.
        pub static kVTCompressionPropertyKey_ConstantBitRate: CFStringRef;
        /// Max frame delay count key.
        pub static kVTCompressionPropertyKey_MaxFrameDelayCount: CFStringRef;
        /// Enable hardware accelerated encoder key.
        pub static kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: CFStringRef;
        /// Require hardware accelerated encoder key.
        pub static kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: CFStringRef;
        /// Prefer low power encoder key.
        pub static kVTVideoEncoderSpecification_PreferEncodingSpeedOverQuality: CFStringRef;
        /// Color primaries key.
        pub static kVTCompressionPropertyKey_ColorPrimaries: CFStringRef;
        /// Transfer function key.
        pub static kVTCompressionPropertyKey_TransferFunction: CFStringRef;
        /// YCbCr matrix key.
        pub static kVTCompressionPropertyKey_YCbCrMatrix: CFStringRef;
        /// HDR metadata key.
        pub static kVTCompressionPropertyKey_MasteringDisplayColorVolume: CFStringRef;
        /// Content light level key.
        pub static kVTCompressionPropertyKey_ContentLightLevelInfo: CFStringRef;

        /// H.264 entropy mode CABAC.
        pub static kVTH264EntropyMode_CABAC: CFStringRef;
        /// H.264 entropy mode CAVLC.
        pub static kVTH264EntropyMode_CAVLC: CFStringRef;

        /// H.264 profile levels.
        pub static kVTProfileLevel_H264_Baseline_AutoLevel: CFStringRef;
        pub static kVTProfileLevel_H264_Main_AutoLevel: CFStringRef;
        pub static kVTProfileLevel_H264_High_AutoLevel: CFStringRef;
        pub static kVTProfileLevel_H264_High_5_0: CFStringRef;
        pub static kVTProfileLevel_H264_High_5_1: CFStringRef;
        pub static kVTProfileLevel_H264_High_5_2: CFStringRef;

        /// HEVC profile levels.
        pub static kVTProfileLevel_HEVC_Main_AutoLevel: CFStringRef;
        pub static kVTProfileLevel_HEVC_Main10_AutoLevel: CFStringRef;

        // --- Compression session functions ---

        /// Create a compression session.
        pub fn VTCompressionSessionCreate(
            allocator: CFAllocatorRef,
            width: i32,
            height: i32,
            codec_type: u32,
            encoder_specification: CFDictionaryRef,
            source_image_buffer_attributes: CFDictionaryRef,
            compressed_data_allocator: CFAllocatorRef,
            output_callback: VTCompressionOutputCallback,
            output_callback_ref_con: *mut c_void,
            compression_session_out: *mut VTCompressionSessionRef,
        ) -> OSStatus;

        /// Prepare to encode frames.
        pub fn VTCompressionSessionPrepareToEncodeFrames(
            session: VTCompressionSessionRef,
        ) -> OSStatus;

        /// Encode a frame.
        pub fn VTCompressionSessionEncodeFrame(
            session: VTCompressionSessionRef,
            image_buffer: CVImageBufferRef,
            presentation_time_stamp: CMTime,
            duration: CMTime,
            frame_properties: CFDictionaryRef,
            source_frame_ref_con: *mut c_void,
            info_flags_out: *mut VTEncodeInfoFlags,
        ) -> OSStatus;

        /// Complete encoding (flush).
        pub fn VTCompressionSessionCompleteFrames(
            session: VTCompressionSessionRef,
            complete_until_presentation_time_stamp: CMTime,
        ) -> OSStatus;

        /// Set session property.
        pub fn VTSessionSetProperty(
            session: VTSessionRef,
            property_key: CFStringRef,
            property_value: CFTypeRef,
        ) -> OSStatus;

        /// Get session property.
        pub fn VTSessionCopyProperty(
            session: VTSessionRef,
            property_key: CFStringRef,
            allocator: CFAllocatorRef,
            property_value_out: *mut CFTypeRef,
        ) -> OSStatus;

        /// Invalidate compression session.
        pub fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);

        // --- Decompression session functions ---

        /// Create a decompression session.
        pub fn VTDecompressionSessionCreate(
            allocator: CFAllocatorRef,
            video_format_description: CMVideoFormatDescriptionRef,
            video_decoder_specification: CFDictionaryRef,
            destination_image_buffer_attributes: CFDictionaryRef,
            output_callback: *const VTDecompressionOutputCallbackRecord,
            decompression_session_out: *mut VTDecompressionSessionRef,
        ) -> OSStatus;

        /// Decode a frame.
        pub fn VTDecompressionSessionDecodeFrame(
            session: VTDecompressionSessionRef,
            sample_buffer: CMSampleBufferRef,
            decode_flags: VTDecodeFrameFlags,
            source_frame_ref_con: *mut c_void,
            info_flags_out: *mut VTDecodeInfoFlags,
        ) -> OSStatus;

        /// Wait for async frames.
        pub fn VTDecompressionSessionWaitForAsynchronousFrames(
            session: VTDecompressionSessionRef,
        ) -> OSStatus;

        /// Finish delayed frames.
        pub fn VTDecompressionSessionFinishDelayedFrames(
            session: VTDecompressionSessionRef,
        ) -> OSStatus;

        /// Can accept format description.
        pub fn VTDecompressionSessionCanAcceptFormatDescription(
            session: VTDecompressionSessionRef,
            format_description: CMFormatDescriptionRef,
        ) -> bool;

        /// Invalidate decompression session.
        pub fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);

        // --- Format description ---

        /// Create video format description from H.264 parameter sets.
        pub fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            format_description_out: *mut CMVideoFormatDescriptionRef,
        ) -> OSStatus;

        /// Create video format description from HEVC parameter sets.
        pub fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            extensions: CFDictionaryRef,
            format_description_out: *mut CMVideoFormatDescriptionRef,
        ) -> OSStatus;

        /// Get H.264 parameter set.
        pub fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
            video_desc: CMVideoFormatDescriptionRef,
            parameter_set_index: usize,
            parameter_set_pointer_out: *mut *const u8,
            parameter_set_size_out: *mut usize,
            parameter_set_count_out: *mut usize,
            nal_unit_header_length_out: *mut i32,
        ) -> OSStatus;

        /// Get HEVC parameter set.
        pub fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
            video_desc: CMVideoFormatDescriptionRef,
            parameter_set_index: usize,
            parameter_set_pointer_out: *mut *const u8,
            parameter_set_size_out: *mut usize,
            parameter_set_count_out: *mut usize,
            nal_unit_header_length_out: *mut i32,
        ) -> OSStatus;

        // --- Sample buffer creation ---

        /// Create sample buffer from block buffer.
        pub fn CMSampleBufferCreate(
            allocator: CFAllocatorRef,
            data_buffer: CMBlockBufferRef,
            data_ready: bool,
            make_data_ready_callback: *const c_void,
            make_data_ready_ref_con: *mut c_void,
            format_description: CMFormatDescriptionRef,
            num_samples: CFIndex,
            num_sample_timing_entries: CFIndex,
            sample_timing_array: *const CMSampleTimingInfo,
            num_sample_size_entries: CFIndex,
            sample_size_array: *const usize,
            sample_buffer_out: *mut CMSampleBufferRef,
        ) -> OSStatus;

        /// Create block buffer with memory block.
        pub fn CMBlockBufferCreateWithMemoryBlock(
            structure_allocator: CFAllocatorRef,
            memory_block: *mut c_void,
            block_length: usize,
            block_allocator: CFAllocatorRef,
            custom_block_source: *const c_void,
            offset_to_data: usize,
            data_length: usize,
            flags: u32,
            block_buffer_out: *mut CMBlockBufferRef,
        ) -> OSStatus;

        /// Check if codec is supported for hardware encoding.
        pub fn VTIsHardwareDecodeSupported(codec_type: u32) -> bool;

        /// Copy supported property dictionary.
        pub fn VTSessionCopySupportedPropertyDictionary(
            session: VTSessionRef,
            supported_property_dictionary_out: *mut CFDictionaryRef,
        ) -> OSStatus;
    }

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------

    /// Create a CFNumber from an i32.
    pub unsafe fn cf_number_create_i32(value: i32) -> CFNumberRef {
        CFNumberCreate(
            kCFAllocatorDefault,
            CFNumberType::SInt32,
            &value as *const i32 as *const c_void,
        )
    }

    /// Create a CFNumber from an i64.
    pub unsafe fn cf_number_create_i64(value: i64) -> CFNumberRef {
        CFNumberCreate(
            kCFAllocatorDefault,
            CFNumberType::SInt64,
            &value as *const i64 as *const c_void,
        )
    }

    /// Create a CFNumber from a f64.
    pub unsafe fn cf_number_create_f64(value: f64) -> CFNumberRef {
        CFNumberCreate(
            kCFAllocatorDefault,
            CFNumberType::Float64,
            &value as *const f64 as *const c_void,
        )
    }

    /// Create a CFString from a Rust string.
    ///
    /// # Panics
    /// Panics if the input string contains a null byte.
    pub unsafe fn cf_string_create(s: &str) -> CFStringRef {
        let c_str = std::ffi::CString::new(s)
            .expect("String for CFString must not contain null bytes");
        CFStringCreateWithCString(kCFAllocatorDefault, c_str.as_ptr(), 0x08000100) // kCFStringEncodingUTF8
    }

    /// Create an empty mutable dictionary.
    pub unsafe fn cf_dictionary_create_mutable() -> CFMutableDictionaryRef {
        CFDictionaryCreateMutable(
            kCFAllocatorDefault,
            0,
            std::ptr::null(), // kCFTypeDictionaryKeyCallBacks
            std::ptr::null(), // kCFTypeDictionaryValueCallBacks
        )
    }

    /// RAII guard for CFTypeRef.
    pub struct CFGuard {
        ptr: CFTypeRef,
    }

    impl CFGuard {
        /// Create a new guard (takes ownership, no retain).
        pub unsafe fn new(ptr: CFTypeRef) -> Option<Self> {
            if ptr.is_null() {
                None
            } else {
                Some(Self { ptr })
            }
        }

        /// Create a guard with retain.
        pub unsafe fn retained(ptr: CFTypeRef) -> Option<Self> {
            if ptr.is_null() {
                None
            } else {
                CFRetain(ptr);
                Some(Self { ptr })
            }
        }

        /// Get the raw pointer.
        pub fn as_ptr(&self) -> CFTypeRef {
            self.ptr
        }
    }

    impl Drop for CFGuard {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { CFRelease(self.ptr) };
            }
        }
    }

    /// RAII guard for CVPixelBufferRef.
    pub struct CVPixelBufferGuard {
        ptr: CVPixelBufferRef,
    }

    impl CVPixelBufferGuard {
        /// Create a new guard.
        pub unsafe fn new(ptr: CVPixelBufferRef) -> Option<Self> {
            if ptr.is_null() {
                None
            } else {
                Some(Self { ptr })
            }
        }

        /// Get the raw pointer.
        pub fn as_ptr(&self) -> CVPixelBufferRef {
            self.ptr
        }

        /// Lock for CPU access.
        pub unsafe fn lock(&self, read_only: bool) -> CVReturn {
            let flags = if read_only { 0x00000001 } else { 0 }; // kCVPixelBufferLock_ReadOnly
            CVPixelBufferLockBaseAddress(self.ptr, flags)
        }

        /// Unlock.
        pub unsafe fn unlock(&self, read_only: bool) -> CVReturn {
            let flags = if read_only { 0x00000001 } else { 0 };
            CVPixelBufferUnlockBaseAddress(self.ptr, flags)
        }
    }

    impl Drop for CVPixelBufferGuard {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { CVPixelBufferRelease(self.ptr) };
            }
        }
    }
}

// =============================================================================
// Feature flag to conditionally use real FFI
// =============================================================================

/// Indicates whether real FFI bindings are available.
#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
pub const FFI_AVAILABLE: bool = true;

#[cfg(not(all(target_os = "macos", feature = "videotoolbox")))]
pub const FFI_AVAILABLE: bool = false;

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::{HwCapabilities, HwCodec};
use std::collections::VecDeque;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};

// =============================================================================
// FFI Type Definitions (conditional on videotoolbox feature)
// =============================================================================

/// OSStatus type - Apple's standard error code type.
/// 0 indicates success, negative values indicate errors.
#[cfg(feature = "videotoolbox")]
pub type OSStatus = i32;

#[cfg(not(feature = "videotoolbox"))]
pub type OSStatus = i32;

/// Success status code.
pub const NO_ERR: OSStatus = 0;

/// Generic VideoToolbox error codes.
pub mod vt_error {
    use super::OSStatus;

    /// Invalid session.
    pub const INVALID_SESSION: OSStatus = -12903;
    /// Allocation failed.
    pub const ALLOCATION_FAILED: OSStatus = -12904;
    /// Property not supported.
    pub const PROPERTY_NOT_SUPPORTED: OSStatus = -12900;
    /// Property read only.
    pub const PROPERTY_READ_ONLY: OSStatus = -12901;
    /// Parameter error.
    pub const PARAMETER: OSStatus = -12902;
    /// Format not supported.
    pub const FORMAT_NOT_SUPPORTED: OSStatus = -12910;
    /// Codec not found.
    pub const CODEC_NOT_FOUND: OSStatus = -12911;
    /// Frame was dropped.
    pub const FRAME_DROPPED: OSStatus = -12912;
}

// -----------------------------------------------------------------------------
// Core Foundation Type Placeholders
// -----------------------------------------------------------------------------

/// Opaque Core Foundation type reference.
/// In real FFI, this would be `*const c_void` from `core_foundation`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CFTypeRef(pub usize);

impl CFTypeRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(0)
    }

    /// Check if this is a null reference.
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl Default for CFTypeRef {
    fn default() -> Self {
        Self::null()
    }
}

/// Core Foundation dictionary reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CFDictionaryRef(pub CFTypeRef);

impl CFDictionaryRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Check if this is a null reference.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// Core Foundation string reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CFStringRef(pub CFTypeRef);

impl CFStringRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }
}

/// Core Foundation allocator reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CFAllocatorRef(pub CFTypeRef);

impl CFAllocatorRef {
    /// Default allocator (kCFAllocatorDefault).
    pub const fn default_allocator() -> Self {
        Self(CFTypeRef::null())
    }
}

/// Core Foundation number reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CFNumberRef(pub CFTypeRef);

/// Core Foundation boolean reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CFBooleanRef(pub CFTypeRef);

// -----------------------------------------------------------------------------
// VideoToolbox Session Types
// -----------------------------------------------------------------------------

/// VideoToolbox compression session reference.
/// Manages encoding state and hardware resources.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VTCompressionSessionRef(pub CFTypeRef);

impl VTCompressionSessionRef {
    /// Create a null session reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Check if this is a null/invalid session.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// VideoToolbox decompression session reference.
/// Manages decoding state and hardware resources.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VTDecompressionSessionRef(pub CFTypeRef);

impl VTDecompressionSessionRef {
    /// Create a null session reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Check if this is a null/invalid session.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

// -----------------------------------------------------------------------------
// CoreVideo Types
// -----------------------------------------------------------------------------

/// CoreVideo pixel buffer reference.
/// Represents a frame of video data in memory.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CVPixelBufferRef(pub CFTypeRef);

impl CVPixelBufferRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Check if this is a null reference.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// CoreVideo image buffer reference (alias for CVPixelBufferRef in most cases).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CVImageBufferRef(pub CFTypeRef);

impl CVImageBufferRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Convert to pixel buffer reference.
    pub fn as_pixel_buffer(&self) -> CVPixelBufferRef {
        CVPixelBufferRef(self.0)
    }
}

impl From<CVPixelBufferRef> for CVImageBufferRef {
    fn from(pb: CVPixelBufferRef) -> Self {
        Self(pb.0)
    }
}

/// CoreVideo pixel buffer pool reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CVPixelBufferPoolRef(pub CFTypeRef);

impl CVPixelBufferPoolRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }
}

// -----------------------------------------------------------------------------
// CoreMedia Types
// -----------------------------------------------------------------------------

/// CoreMedia sample buffer reference.
/// Contains compressed or uncompressed media samples with timing info.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMSampleBufferRef(pub CFTypeRef);

impl CMSampleBufferRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Check if this is a null reference.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// CoreMedia format description reference.
/// Describes the format of media data.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMFormatDescriptionRef(pub CFTypeRef);

impl CMFormatDescriptionRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }
}

/// CoreMedia video format description reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMVideoFormatDescriptionRef(pub CFTypeRef);

impl CMVideoFormatDescriptionRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }

    /// Convert to generic format description.
    pub fn as_format_description(&self) -> CMFormatDescriptionRef {
        CMFormatDescriptionRef(self.0)
    }
}

/// CoreMedia block buffer reference.
/// Contains the actual compressed data bytes.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMBlockBufferRef(pub CFTypeRef);

impl CMBlockBufferRef {
    /// Create a null reference.
    pub const fn null() -> Self {
        Self(CFTypeRef::null())
    }
}

/// CoreMedia time structure.
/// Represents time as a rational number (value/timescale).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMTime {
    /// Time value in units of timescale.
    pub value: i64,
    /// Number of units per second.
    pub timescale: i32,
    /// Flags indicating validity, rounding, etc.
    pub flags: u32,
    /// Reserved/epoch field.
    pub epoch: i64,
}

impl CMTime {
    /// Create an invalid time.
    pub const INVALID: CMTime = CMTime {
        value: 0,
        timescale: 0,
        flags: 0,
        epoch: 0,
    };

    /// Create a zero time.
    pub const ZERO: CMTime = CMTime {
        value: 0,
        timescale: 1,
        flags: 1, // kCMTimeFlags_Valid
        epoch: 0,
    };

    /// Create a positive infinity time.
    pub const POSITIVE_INFINITY: CMTime = CMTime {
        value: i64::MAX,
        timescale: 1,
        flags: 5, // kCMTimeFlags_Valid | kCMTimeFlags_PositiveInfinity
        epoch: 0,
    };

    /// kCMTimeFlags_Valid
    pub const FLAG_VALID: u32 = 1;
    /// kCMTimeFlags_HasBeenRounded
    pub const FLAG_HAS_BEEN_ROUNDED: u32 = 2;
    /// kCMTimeFlags_PositiveInfinity
    pub const FLAG_POSITIVE_INFINITY: u32 = 4;
    /// kCMTimeFlags_NegativeInfinity
    pub const FLAG_NEGATIVE_INFINITY: u32 = 8;
    /// kCMTimeFlags_Indefinite
    pub const FLAG_INDEFINITE: u32 = 16;

    /// Create a new CMTime.
    pub const fn new(value: i64, timescale: i32) -> Self {
        Self {
            value,
            timescale,
            flags: Self::FLAG_VALID,
            epoch: 0,
        }
    }

    /// Create CMTime from seconds.
    pub fn from_seconds(seconds: f64, timescale: i32) -> Self {
        Self {
            value: (seconds * timescale as f64) as i64,
            timescale,
            flags: Self::FLAG_VALID,
            epoch: 0,
        }
    }

    /// Convert to seconds.
    pub fn to_seconds(&self) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        self.value as f64 / self.timescale as f64
    }

    /// Check if this time is valid.
    pub fn is_valid(&self) -> bool {
        (self.flags & Self::FLAG_VALID) != 0
    }

    /// Check if this is positive infinity.
    pub fn is_positive_infinity(&self) -> bool {
        (self.flags & Self::FLAG_POSITIVE_INFINITY) != 0
    }
}

/// CoreMedia time range structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CMTimeRange {
    /// Start time.
    pub start: CMTime,
    /// Duration.
    pub duration: CMTime,
}

impl CMTimeRange {
    /// Create a new time range.
    pub const fn new(start: CMTime, duration: CMTime) -> Self {
        Self { start, duration }
    }
}

// =============================================================================
// Callback Infrastructure
// =============================================================================

/// Compression output callback function type.
/// Called by VideoToolbox when an encoded frame is ready.
///
/// # Safety
/// This type signature matches Apple's VTCompressionOutputCallback.
pub type VTCompressionOutputCallback = extern "C" fn(
    output_callback_ref_con: *mut std::ffi::c_void,
    source_frame_ref_con: *mut std::ffi::c_void,
    status: OSStatus,
    info_flags: u32,
    sample_buffer: CMSampleBufferRef,
);

/// Decompression output callback function type.
/// Called by VideoToolbox when a decoded frame is ready.
///
/// # Safety
/// This type signature matches Apple's VTDecompressionOutputCallback.
pub type VTDecompressionOutputCallback = extern "C" fn(
    decompress_output_ref_con: *mut std::ffi::c_void,
    source_frame_ref_con: *mut std::ffi::c_void,
    status: OSStatus,
    info_flags: u32,
    image_buffer: CVImageBufferRef,
    presentation_time_stamp: CMTime,
    presentation_duration: CMTime,
);

/// Callback info flags.
pub mod vt_callback_flags {
    /// Frame was dropped.
    pub const FRAME_DROPPED: u32 = 1 << 0;
    /// Decoding was done asynchronously.
    pub const ASYNC: u32 = 1 << 1;
}

/// Thread-safe callback context for compression output.
/// Uses Arc for shared ownership and Send+Sync for thread safety.
pub struct VTCompressionOutputContext {
    /// Queue for encoded frames.
    output_queue: Arc<Mutex<VecDeque<VTEncodedFrame>>>,
    /// Error status from callbacks.
    error: Arc<Mutex<Option<HwAccelError>>>,
    /// Frame counter for ordering (used by real FFI callbacks).
    #[allow(dead_code)]
    frame_counter: Arc<Mutex<u64>>,
}

impl VTCompressionOutputContext {
    /// Create a new compression output context.
    pub fn new() -> Self {
        Self {
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
            error: Arc::new(Mutex::new(None)),
            frame_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a weak reference to the output queue.
    pub fn output_queue_weak(&self) -> Weak<Mutex<VecDeque<VTEncodedFrame>>> {
        Arc::downgrade(&self.output_queue)
    }

    /// Push an encoded frame to the queue.
    pub fn push_frame(&self, frame: VTEncodedFrame) {
        let mut queue = self.output_queue.lock();
        queue.push_back(frame);
    }

    /// Pop an encoded frame from the queue.
    pub fn pop_frame(&self) -> Option<VTEncodedFrame> {
        let mut queue = self.output_queue.lock();
        queue.pop_front()
    }

    /// Set an error.
    pub fn set_error(&self, err: HwAccelError) {
        let mut error = self.error.lock();
        *error = Some(err);
    }

    /// Take the error if present.
    pub fn take_error(&self) -> Option<HwAccelError> {
        let mut error = self.error.lock();
        error.take()
    }

    /// Get pending frame count.
    pub fn pending_count(&self) -> usize {
        self.output_queue.lock().len()
    }
}

impl Default for VTCompressionOutputContext {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: The context uses Arc<Mutex<T>> internally which is Send+Sync
unsafe impl Send for VTCompressionOutputContext {}
unsafe impl Sync for VTCompressionOutputContext {}

/// Thread-safe callback context for decompression output.
pub struct VTDecompressionOutputContext {
    /// Queue for decoded frames.
    output_queue: Arc<Mutex<VecDeque<VTDecodedFrame>>>,
    /// Error status from callbacks.
    error: Arc<Mutex<Option<HwAccelError>>>,
}

impl VTDecompressionOutputContext {
    /// Create a new decompression output context.
    pub fn new() -> Self {
        Self {
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Push a decoded frame to the queue.
    pub fn push_frame(&self, frame: VTDecodedFrame) {
        let mut queue = self.output_queue.lock();
        queue.push_back(frame);
    }

    /// Pop a decoded frame from the queue.
    pub fn pop_frame(&self) -> Option<VTDecodedFrame> {
        let mut queue = self.output_queue.lock();
        queue.pop_front()
    }

    /// Set an error.
    pub fn set_error(&self, err: HwAccelError) {
        let mut error = self.error.lock();
        *error = Some(err);
    }

    /// Take the error if present.
    pub fn take_error(&self) -> Option<HwAccelError> {
        let mut error = self.error.lock();
        error.take()
    }
}

impl Default for VTDecompressionOutputContext {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: The context uses Arc<Mutex<T>> internally which is Send+Sync
unsafe impl Send for VTDecompressionOutputContext {}
unsafe impl Sync for VTDecompressionOutputContext {}

/// Decoded frame from decompression callback.
#[derive(Debug)]
pub struct VTDecodedFrame {
    /// Pixel data (CPU copy).
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: CVPixelFormat,
    /// Presentation timestamp.
    pub pts: CMTime,
    /// Duration.
    pub duration: CMTime,
}

// =============================================================================
// Property Keys
// =============================================================================

/// VideoToolbox compression property keys.
pub mod compression_property_keys {
    /// Target bitrate in bits per second.
    pub const AVERAGE_BIT_RATE: &str = "AverageBitRate";
    /// Data rate limits (array of bytes per second and duration pairs).
    pub const DATA_RATE_LIMITS: &str = "DataRateLimits";
    /// Expected frame rate.
    pub const EXPECTED_FRAME_RATE: &str = "ExpectedFrameRate";
    /// Maximum keyframe interval in frames.
    pub const MAX_KEY_FRAME_INTERVAL: &str = "MaxKeyFrameInterval";
    /// Maximum keyframe interval duration in seconds.
    pub const MAX_KEY_FRAME_INTERVAL_DURATION: &str = "MaxKeyFrameIntervalDuration";
    /// Allow frame reordering (B-frames).
    pub const ALLOW_FRAME_REORDERING: &str = "AllowFrameReordering";
    /// Profile level.
    pub const PROFILE_LEVEL: &str = "ProfileLevel";
    /// H.264 entropy mode (CAVLC or CABAC).
    pub const H264_ENTROPY_MODE: &str = "H264EntropyMode";
    /// Quality setting (0.0 to 1.0).
    pub const QUALITY: &str = "Quality";
    /// Enable real-time encoding.
    pub const REAL_TIME: &str = "RealTime";
    /// Maximum frame delay count.
    pub const MAX_FRAME_DELAY_COUNT: &str = "MaxFrameDelayCount";
    /// Allow temporal compression.
    pub const ALLOW_TEMPORAL_COMPRESSION: &str = "AllowTemporalCompression";
    /// Enable hardware acceleration.
    pub const ENABLE_HARDWARE_ACCELERATED_VIDEO_ENCODER: &str =
        "EnableHardwareAcceleratedVideoEncoder";
    /// Require hardware acceleration.
    pub const REQUIRE_HARDWARE_ACCELERATED_VIDEO_ENCODER: &str =
        "RequireHardwareAcceleratedVideoEncoder";
    /// Color primaries.
    pub const COLOR_PRIMARIES: &str = "ColorPrimaries";
    /// Transfer function.
    pub const TRANSFER_FUNCTION: &str = "TransferFunction";
    /// YCbCr matrix.
    pub const YCBCR_MATRIX: &str = "YCbCrMatrix";
    /// Clean aperture.
    pub const CLEAN_APERTURE: &str = "CleanAperture";
    /// Pixel aspect ratio.
    pub const PIXEL_ASPECT_RATIO: &str = "PixelAspectRatio";
    /// Field count.
    pub const FIELD_COUNT: &str = "FieldCount";
    /// More frames before start.
    pub const MORE_FRAMES_BEFORE_START: &str = "MoreFramesBeforeStart";
    /// More frames after end.
    pub const MORE_FRAMES_AFTER_END: &str = "MoreFramesAfterEnd";
    /// Prioritize encoding speed over quality.
    pub const PRIORITIZE_ENCODING_SPEED_OVER_QUALITY: &str = "PrioritizeEncodingSpeedOverQuality";
    /// Constant bit rate.
    pub const CONSTANT_BIT_RATE: &str = "ConstantBitRate";
    /// Target quality for VBR.
    pub const TARGET_QUALITY_FOR_ALPHA: &str = "TargetQualityForAlpha";
    /// Maximum allowed frame QP.
    pub const MAX_ALLOWED_FRAME_QP: &str = "MaxAllowedFrameQP";
    /// Minimum allowed frame QP.
    pub const MIN_ALLOWED_FRAME_QP: &str = "MinAllowedFrameQP";
    /// Enable low latency rate control.
    pub const ENABLE_LOW_LATENCY_RATE_CONTROL: &str = "EnableLowLatencyRateControl";
    /// Base layer bit rate fraction.
    pub const BASE_LAYER_BIT_RATE_FRACTION: &str = "BaseLayerBitRateFraction";
    /// Maximum slice bytes.
    pub const MAX_SLICE_BYTES: &str = "MaxSliceBytes";
    /// Preserve dynamic HDR metadata.
    pub const PRESERVE_DYNAMIC_HDR_METADATA: &str = "PreserveDynamicHDRMetadata";
    /// Output bit depth.
    pub const OUTPUT_BIT_DEPTH: &str = "OutputBitDepth";
}

/// VideoToolbox decompression property keys.
pub mod decompression_property_keys {
    /// Output pixel format.
    pub const PIXEL_BUFFER_POOL: &str = "CVPixelBufferPool";
    /// Real-time decoding mode.
    pub const REAL_TIME: &str = "RealTime";
    /// Reduce resolution.
    pub const REDUCE_RESOLUTION: &str = "ReduceResolution";
    /// Only decode keyframes.
    pub const ONLY_THE_SPECIFIED_FRAME: &str = "OnlyTheSpecifiedFrame";
    /// Thread count.
    pub const NUMBER_OF_THREADS: &str = "NumberOfThreads";
    /// Field mode.
    pub const FIELD_MODE: &str = "FieldMode";
    /// Deinterlace mode.
    pub const DEINTERLACE_MODE: &str = "DeinterlaceMode";
    /// Pixel transfer properties.
    pub const PIXEL_TRANSFER_PROPERTIES: &str = "PixelTransferProperties";
    /// Requested maximum frame count.
    pub const REQUESTED_MAX_FRAME_COUNT: &str = "RequestedMaxFrameCount";
}

/// CoreVideo pixel buffer keys.
pub mod cv_pixel_buffer_keys {
    /// Pixel format type key.
    pub const PIXEL_FORMAT_TYPE_KEY: &str = "PixelFormatType";
    /// Width key.
    pub const WIDTH_KEY: &str = "Width";
    /// Height key.
    pub const HEIGHT_KEY: &str = "Height";
    /// Extended pixels left.
    pub const EXTENDED_PIXELS_LEFT_KEY: &str = "ExtendedPixelsLeft";
    /// Extended pixels top.
    pub const EXTENDED_PIXELS_TOP_KEY: &str = "ExtendedPixelsTop";
    /// Extended pixels right.
    pub const EXTENDED_PIXELS_RIGHT_KEY: &str = "ExtendedPixelsRight";
    /// Extended pixels bottom.
    pub const EXTENDED_PIXELS_BOTTOM_KEY: &str = "ExtendedPixelsBottom";
    /// Bytes per row alignment.
    pub const BYTES_PER_ROW_ALIGNMENT_KEY: &str = "BytesPerRowAlignment";
    /// Memory allocator key.
    pub const MEMORY_ALLOCATOR_KEY: &str = "MemoryAllocator";
    /// IOSurface properties key.
    pub const IO_SURFACE_PROPERTIES_KEY: &str = "IOSurfaceProperties";
    /// Metal compatibility key.
    pub const METAL_COMPATIBILITY_KEY: &str = "MetalCompatibility";
    /// OpenGL compatibility key.
    pub const OPENGL_COMPATIBILITY_KEY: &str = "OpenGLCompatibility";
}

/// H.264 profile level constants.
pub mod h264_profile_level {
    /// Baseline Auto Level.
    pub const BASELINE_AUTO_LEVEL: &str = "H264_Baseline_AutoLevel";
    /// Baseline 3.0.
    pub const BASELINE_3_0: &str = "H264_Baseline_3_0";
    /// Baseline 3.1.
    pub const BASELINE_3_1: &str = "H264_Baseline_3_1";
    /// Baseline 4.0.
    pub const BASELINE_4_0: &str = "H264_Baseline_4_0";
    /// Baseline 4.1.
    pub const BASELINE_4_1: &str = "H264_Baseline_4_1";
    /// Main Auto Level.
    pub const MAIN_AUTO_LEVEL: &str = "H264_Main_AutoLevel";
    /// Main 3.0.
    pub const MAIN_3_0: &str = "H264_Main_3_0";
    /// Main 3.1.
    pub const MAIN_3_1: &str = "H264_Main_3_1";
    /// Main 4.0.
    pub const MAIN_4_0: &str = "H264_Main_4_0";
    /// Main 4.1.
    pub const MAIN_4_1: &str = "H264_Main_4_1";
    /// Main 4.2.
    pub const MAIN_4_2: &str = "H264_Main_4_2";
    /// Main 5.0.
    pub const MAIN_5_0: &str = "H264_Main_5_0";
    /// Main 5.1.
    pub const MAIN_5_1: &str = "H264_Main_5_1";
    /// Main 5.2.
    pub const MAIN_5_2: &str = "H264_Main_5_2";
    /// High Auto Level.
    pub const HIGH_AUTO_LEVEL: &str = "H264_High_AutoLevel";
    /// High 3.0.
    pub const HIGH_3_0: &str = "H264_High_3_0";
    /// High 3.1.
    pub const HIGH_3_1: &str = "H264_High_3_1";
    /// High 4.0.
    pub const HIGH_4_0: &str = "H264_High_4_0";
    /// High 4.1.
    pub const HIGH_4_1: &str = "H264_High_4_1";
    /// High 4.2.
    pub const HIGH_4_2: &str = "H264_High_4_2";
    /// High 5.0.
    pub const HIGH_5_0: &str = "H264_High_5_0";
    /// High 5.1.
    pub const HIGH_5_1: &str = "H264_High_5_1";
    /// High 5.2.
    pub const HIGH_5_2: &str = "H264_High_5_2";
}

/// HEVC profile level constants.
pub mod hevc_profile_level {
    /// Main Auto Level.
    pub const MAIN_AUTO_LEVEL: &str = "HEVC_Main_AutoLevel";
    /// Main10 Auto Level.
    pub const MAIN10_AUTO_LEVEL: &str = "HEVC_Main10_AutoLevel";
    /// Main 4.1.
    pub const MAIN_4_1: &str = "HEVC_Main_4_1";
    /// Main 5.0.
    pub const MAIN_5_0: &str = "HEVC_Main_5_0";
    /// Main 5.1.
    pub const MAIN_5_1: &str = "HEVC_Main_5_1";
    /// Main 5.2.
    pub const MAIN_5_2: &str = "HEVC_Main_5_2";
    /// Main 6.0.
    pub const MAIN_6_0: &str = "HEVC_Main_6_0";
    /// Main 6.1.
    pub const MAIN_6_1: &str = "HEVC_Main_6_1";
    /// Main 6.2.
    pub const MAIN_6_2: &str = "HEVC_Main_6_2";
    /// Main10 4.1.
    pub const MAIN10_4_1: &str = "HEVC_Main10_4_1";
    /// Main10 5.0.
    pub const MAIN10_5_0: &str = "HEVC_Main10_5_0";
    /// Main10 5.1.
    pub const MAIN10_5_1: &str = "HEVC_Main10_5_1";
    /// Main10 5.2.
    pub const MAIN10_5_2: &str = "HEVC_Main10_5_2";
    /// Main10 6.0.
    pub const MAIN10_6_0: &str = "HEVC_Main10_6_0";
    /// Main10 6.1.
    pub const MAIN10_6_1: &str = "HEVC_Main10_6_1";
    /// Main10 6.2.
    pub const MAIN10_6_2: &str = "HEVC_Main10_6_2";
}

/// AV1 profile level constants.
/// AV1 hardware encoding requires Apple Silicon M3 or later.
/// AV1 hardware decoding is available on Apple Silicon M1 and later.
pub mod av1_profile_level {
    /// Main Auto Level.
    pub const MAIN_AUTO_LEVEL: &str = "AV1_Main_AutoLevel";
    /// Main 4.0.
    pub const MAIN_4_0: &str = "AV1_Main_4_0";
    /// Main 4.1.
    pub const MAIN_4_1: &str = "AV1_Main_4_1";
    /// Main 5.0.
    pub const MAIN_5_0: &str = "AV1_Main_5_0";
    /// Main 5.1.
    pub const MAIN_5_1: &str = "AV1_Main_5_1";
    /// Main 5.2.
    pub const MAIN_5_2: &str = "AV1_Main_5_2";
    /// Main 6.0.
    pub const MAIN_6_0: &str = "AV1_Main_6_0";
    /// Main 6.1.
    pub const MAIN_6_1: &str = "AV1_Main_6_1";
    /// High Auto Level (10-bit).
    pub const HIGH_AUTO_LEVEL: &str = "AV1_High_AutoLevel";
    /// High 4.0.
    pub const HIGH_4_0: &str = "AV1_High_4_0";
    /// High 4.1.
    pub const HIGH_4_1: &str = "AV1_High_4_1";
    /// High 5.0.
    pub const HIGH_5_0: &str = "AV1_High_5_0";
    /// High 5.1.
    pub const HIGH_5_1: &str = "AV1_High_5_1";
    /// High 5.2.
    pub const HIGH_5_2: &str = "AV1_High_5_2";
}

/// H.264 entropy mode constants.
pub mod h264_entropy_mode {
    /// CAVLC (Context-Adaptive Variable-Length Coding).
    pub const CAVLC: &str = "CAVLC";
    /// CABAC (Context-Adaptive Binary Arithmetic Coding).
    pub const CABAC: &str = "CABAC";
}

/// Video codec types (FourCC).
pub mod codec_type {
    /// H.264/AVC.
    pub const H264: u32 = 0x6176_6331; // 'avc1'
    /// H.265/HEVC.
    pub const HEVC: u32 = 0x6865_7631; // 'hev1'
    /// HEVC with parameter sets in band.
    pub const HEVC_WITH_ALPHA: u32 = 0x6865_766D; // 'hevm'
    /// ProRes 4444.
    pub const PRORES_4444: u32 = 0x6170_3468; // 'ap4h'
    /// ProRes 4444 XQ.
    pub const PRORES_4444_XQ: u32 = 0x6170_3478; // 'ap4x'
    /// ProRes 422 HQ.
    pub const PRORES_422_HQ: u32 = 0x6170_6368; // 'apch'
    /// ProRes 422.
    pub const PRORES_422: u32 = 0x6170_636E; // 'apcn'
    /// ProRes 422 LT.
    pub const PRORES_422_LT: u32 = 0x6170_6373; // 'apcs'
    /// ProRes 422 Proxy.
    pub const PRORES_422_PROXY: u32 = 0x6170_636F; // 'apco'
    /// ProRes RAW.
    pub const PRORES_RAW: u32 = 0x6170_7277; // 'aprw'
    /// ProRes RAW HQ.
    pub const PRORES_RAW_HQ: u32 = 0x6170_7268; // 'aprh'
    /// JPEG.
    pub const JPEG: u32 = 0x6A70_6567; // 'jpeg'
    /// AV1.
    pub const AV1: u32 = 0x6176_3031; // 'av01'
}

// =============================================================================
// ProRes Support
// =============================================================================

/// ProRes profile for VideoToolbox encoding.
/// VideoToolbox is the preferred way to encode ProRes on macOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProResProfile {
    /// ProRes 422 Proxy - Lowest data rate, for offline editing.
    /// ~45 Mbps @ 1080p 30fps
    Proxy,
    /// ProRes 422 LT - Low data rate, good for space-constrained workflows.
    /// ~102 Mbps @ 1080p 30fps
    Lt,
    /// ProRes 422 - Standard production quality.
    /// ~147 Mbps @ 1080p 30fps
    Standard,
    /// ProRes 422 HQ - High quality, visually lossless.
    /// ~220 Mbps @ 1080p 30fps
    Hq,
    /// ProRes 4444 - With alpha channel support and higher quality.
    /// ~330 Mbps @ 1080p 30fps
    P4444,
    /// ProRes 4444 XQ - Maximum quality 4:4:4 encoding.
    /// ~500 Mbps @ 1080p 30fps
    P4444Xq,
}

impl ProResProfile {
    /// Get the codec type (FourCC) for this profile.
    pub fn codec_type(&self) -> u32 {
        match self {
            ProResProfile::Proxy => codec_type::PRORES_422_PROXY,
            ProResProfile::Lt => codec_type::PRORES_422_LT,
            ProResProfile::Standard => codec_type::PRORES_422,
            ProResProfile::Hq => codec_type::PRORES_422_HQ,
            ProResProfile::P4444 => codec_type::PRORES_4444,
            ProResProfile::P4444Xq => codec_type::PRORES_4444_XQ,
        }
    }

    /// Get the FourCC string for this profile.
    pub fn fourcc(&self) -> &'static str {
        match self {
            ProResProfile::Proxy => "apco",
            ProResProfile::Lt => "apcs",
            ProResProfile::Standard => "apcn",
            ProResProfile::Hq => "apch",
            ProResProfile::P4444 => "ap4h",
            ProResProfile::P4444Xq => "ap4x",
        }
    }

    /// Get approximate bitrate for 1080p @ 30fps in Mbps.
    pub fn approx_bitrate_mbps(&self) -> u32 {
        match self {
            ProResProfile::Proxy => 45,
            ProResProfile::Lt => 102,
            ProResProfile::Standard => 147,
            ProResProfile::Hq => 220,
            ProResProfile::P4444 => 330,
            ProResProfile::P4444Xq => 500,
        }
    }

    /// Check if this profile supports alpha channel.
    pub fn supports_alpha(&self) -> bool {
        matches!(self, ProResProfile::P4444 | ProResProfile::P4444Xq)
    }

    /// Get the chroma subsampling for this profile.
    pub fn chroma_subsampling(&self) -> &'static str {
        match self {
            ProResProfile::P4444 | ProResProfile::P4444Xq => "4:4:4",
            _ => "4:2:2",
        }
    }

    /// Get bit depth (all ProRes profiles support up to 12-bit).
    pub fn bit_depth(&self) -> u8 {
        match self {
            ProResProfile::P4444 | ProResProfile::P4444Xq => 12,
            _ => 10,
        }
    }
}

/// ProRes encoder configuration.
#[derive(Debug, Clone)]
pub struct ProResEncoderConfig {
    /// ProRes profile.
    pub profile: ProResProfile,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Frame rate (numerator, denominator).
    pub frame_rate: (u32, u32),
    /// Pixel format for input (should be 422 or 4444 based on profile).
    pub input_format: CVPixelFormat,
    /// Include alpha channel (requires P4444 or P4444Xq).
    pub include_alpha: bool,
    /// Enable hardware acceleration.
    pub hardware_accelerated: bool,
}

impl ProResEncoderConfig {
    /// Create a new ProRes encoder config.
    pub fn new(profile: ProResProfile, width: u32, height: u32) -> Self {
        let input_format = if profile.supports_alpha() {
            CVPixelFormat::Argb32
        } else {
            CVPixelFormat::Yuyv422
        };

        Self {
            profile,
            width,
            height,
            frame_rate: (24, 1),
            input_format,
            include_alpha: profile.supports_alpha(),
            hardware_accelerated: true,
        }
    }

    /// Set frame rate.
    pub fn with_frame_rate(mut self, num: u32, den: u32) -> Self {
        self.frame_rate = (num, den);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.include_alpha && !self.profile.supports_alpha() {
            return Err(HwAccelError::Config(
                "Alpha channel requires ProRes 4444 or 4444 XQ profile".to_string(),
            ));
        }

        // ProRes requires specific dimensions (multiples of 2 for all profiles)
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            return Err(HwAccelError::Config(
                "ProRes requires even dimensions".to_string(),
            ));
        }

        Ok(())
    }
}

/// ProRes decoder configuration.
#[derive(Debug, Clone)]
pub struct ProResDecoderConfig {
    /// Output pixel format.
    pub output_format: CVPixelFormat,
    /// Enable hardware acceleration.
    pub hardware_accelerated: bool,
}

impl Default for ProResDecoderConfig {
    fn default() -> Self {
        Self {
            output_format: CVPixelFormat::Yuyv422,
            hardware_accelerated: true,
        }
    }
}

// =============================================================================
// Memory Management Patterns
// =============================================================================

/// Simulated CFRetain - increments reference count.
/// In real FFI, this would call `CFRetain` from CoreFoundation.
#[inline]
pub fn cf_retain<T: Copy>(cf_ref: T) -> T {
    // No-op in mock implementation.
    // Real implementation: unsafe { CFRetain(cf_ref as CFTypeRef); }
    cf_ref
}

/// Simulated CFRelease - decrements reference count.
/// In real FFI, this would call `CFRelease` from CoreFoundation.
#[inline]
pub fn cf_release<T>(_cf_ref: T) {
    // No-op in mock implementation.
    // Real implementation: unsafe { CFRelease(cf_ref as CFTypeRef); }
}

/// RAII wrapper for CF types that automatically calls CFRelease on drop.
#[derive(Debug)]
pub struct CFGuard<T: Copy> {
    inner: T,
    /// Whether we own this reference (should release on drop).
    owned: bool,
}

impl<T: Copy> CFGuard<T> {
    /// Create a new guard that owns the reference.
    pub fn new(inner: T) -> Self {
        Self { inner, owned: true }
    }

    /// Create a guard that doesn't own the reference (borrowed).
    pub fn borrowed(inner: T) -> Self {
        Self { inner, owned: false }
    }

    /// Get the inner reference.
    pub fn get(&self) -> T {
        self.inner
    }

    /// Release ownership and return the inner reference.
    pub fn release(mut self) -> T {
        self.owned = false;
        self.inner
    }
}

impl<T: Copy> Drop for CFGuard<T> {
    fn drop(&mut self) {
        if self.owned {
            cf_release(self.inner);
        }
    }
}

impl<T: Copy> std::ops::Deref for CFGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Autorelease pool simulation.
/// In real FFI, this would wrap `@autoreleasepool` or NSAutoreleasePool.
pub struct AutoreleasePool {
    /// Objects to release when pool is drained.
    _marker: std::marker::PhantomData<()>,
}

impl AutoreleasePool {
    /// Create a new autorelease pool.
    pub fn new() -> Self {
        // Real implementation: objc_autoreleasePoolPush()
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    /// Execute a closure within an autorelease pool.
    pub fn execute<T, F: FnOnce() -> T>(f: F) -> T {
        let _pool = AutoreleasePool::new();
        f()
    }
}

impl Default for AutoreleasePool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        // Real implementation: objc_autoreleasePoolPop(pool)
    }
}

/// Buffer recycling pool for CVPixelBuffer-like objects.
pub struct BufferRecycler<T> {
    /// Available buffers for reuse.
    available: Mutex<VecDeque<T>>,
    /// Maximum number of buffers to keep.
    max_size: usize,
    /// Total buffers created.
    created: Mutex<u64>,
    /// Total buffers recycled.
    recycled: Mutex<u64>,
}

impl<T> BufferRecycler<T> {
    /// Create a new buffer recycler.
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Mutex::new(VecDeque::with_capacity(max_size)),
            max_size,
            created: Mutex::new(0),
            recycled: Mutex::new(0),
        }
    }

    /// Try to get a buffer from the pool.
    pub fn try_get(&self) -> Option<T> {
        let mut available = self.available.lock();
        available.pop_front()
    }

    /// Return a buffer to the pool for reuse.
    pub fn recycle(&self, buffer: T) {
        let mut available = self.available.lock();
        if available.len() < self.max_size {
            available.push_back(buffer);
            *self.recycled.lock() += 1;
        }
        // If pool is full, buffer is dropped
    }

    /// Record that a new buffer was created.
    pub fn record_creation(&self) {
        *self.created.lock() += 1;
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferRecyclerStats {
        BufferRecyclerStats {
            available: self.available.lock().len(),
            max_size: self.max_size,
            created: *self.created.lock(),
            recycled: *self.recycled.lock(),
        }
    }
}

/// Buffer recycler statistics.
#[derive(Debug, Clone)]
pub struct BufferRecyclerStats {
    /// Currently available buffers.
    pub available: usize,
    /// Maximum pool size.
    pub max_size: usize,
    /// Total buffers created.
    pub created: u64,
    /// Total buffers recycled.
    pub recycled: u64,
}

// =============================================================================
// Session Management
// =============================================================================

/// Session state for encoder/decoder lifecycle management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is uninitialized.
    Uninitialized,
    /// Session is being created.
    Creating,
    /// Session is ready for use.
    Ready,
    /// Session is encoding/decoding.
    Active,
    /// Session is being flushed.
    Flushing,
    /// Session has been invalidated.
    Invalidated,
    /// Session encountered an error.
    Error,
}

/// Flush mode for session completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushMode {
    /// Synchronous flush - block until all frames are processed.
    Sync,
    /// Asynchronous flush - return immediately, frames delivered via callback.
    Async,
    /// Emit all pending frames but don't wait for completion.
    EmitPending,
}

/// Hardware encoder selection hints.
#[derive(Debug, Clone)]
pub struct HardwareEncoderHints {
    /// Prefer hardware acceleration.
    pub prefer_hardware: bool,
    /// Require hardware acceleration (fail if not available).
    pub require_hardware: bool,
    /// Prefer low power encoder if available.
    pub prefer_low_power: bool,
    /// Specific encoder ID to use (if known).
    pub encoder_id: Option<String>,
}

impl Default for HardwareEncoderHints {
    fn default() -> Self {
        Self {
            prefer_hardware: true,
            require_hardware: false,
            prefer_low_power: false,
            encoder_id: None,
        }
    }
}

impl HardwareEncoderHints {
    /// Require hardware acceleration.
    pub fn require_hardware() -> Self {
        Self {
            prefer_hardware: true,
            require_hardware: true,
            prefer_low_power: false,
            encoder_id: None,
        }
    }

    /// Prefer low power mode (for battery life).
    pub fn low_power() -> Self {
        Self {
            prefer_hardware: true,
            require_hardware: false,
            prefer_low_power: true,
            encoder_id: None,
        }
    }
}

/// Session lifecycle manager with proper create/configure/encode/invalidate flow.
pub struct SessionManager {
    /// Current session state.
    state: SessionState,
    /// Compression session (if encoding).
    compression_session: VTCompressionSessionRef,
    /// Decompression session (if decoding).
    decompression_session: VTDecompressionSessionRef,
    /// Format description for the session.
    format_description: CMVideoFormatDescriptionRef,
    /// Hardware hints (used when creating sessions with real FFI).
    #[allow(dead_code)]
    hardware_hints: HardwareEncoderHints,
    /// Frames submitted but not yet returned.
    pending_frame_count: u32,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        Self {
            state: SessionState::Uninitialized,
            compression_session: VTCompressionSessionRef::null(),
            decompression_session: VTDecompressionSessionRef::null(),
            format_description: CMVideoFormatDescriptionRef::null(),
            hardware_hints: HardwareEncoderHints::default(),
            pending_frame_count: 0,
        }
    }

    /// Create a new session manager with hardware hints.
    pub fn with_hints(hints: HardwareEncoderHints) -> Self {
        Self {
            hardware_hints: hints,
            ..Self::new()
        }
    }

    /// Get current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Create a compression session.
    pub fn create_compression_session(
        &mut self,
        width: u32,
        height: u32,
        codec_type: u32,
        _encoder_spec: Option<CFDictionaryRef>,
        _source_image_buffer_attributes: Option<CFDictionaryRef>,
    ) -> Result<()> {
        if self.state != SessionState::Uninitialized {
            return Err(HwAccelError::DeviceInit(
                "Session already initialized".to_string(),
            ));
        }

        self.state = SessionState::Creating;

        // In real FFI:
        // let status = VTCompressionSessionCreate(
        //     kCFAllocatorDefault,
        //     width as i32,
        //     height as i32,
        //     codec_type,
        //     encoder_spec,
        //     source_image_buffer_attributes,
        //     kCFAllocatorDefault,
        //     callback,
        //     callback_ref_con,
        //     &mut self.compression_session
        // );

        let _ = (width, height, codec_type);

        // Simulate successful creation
        self.compression_session = VTCompressionSessionRef(CFTypeRef(0x1234));
        self.state = SessionState::Ready;

        tracing::debug!(
            "Created compression session for {}x{} codec {:08x}",
            width,
            height,
            codec_type
        );

        Ok(())
    }

    /// Create a decompression session.
    pub fn create_decompression_session(
        &mut self,
        format_description: CMVideoFormatDescriptionRef,
        _decoder_spec: Option<CFDictionaryRef>,
        _destination_image_buffer_attributes: Option<CFDictionaryRef>,
    ) -> Result<()> {
        if self.state != SessionState::Uninitialized {
            return Err(HwAccelError::DeviceInit(
                "Session already initialized".to_string(),
            ));
        }

        self.state = SessionState::Creating;

        // In real FFI:
        // let status = VTDecompressionSessionCreate(
        //     kCFAllocatorDefault,
        //     format_description,
        //     decoder_spec,
        //     destination_image_buffer_attributes,
        //     &callback_record,
        //     &mut self.decompression_session
        // );

        self.format_description = format_description;

        // Simulate successful creation
        self.decompression_session = VTDecompressionSessionRef(CFTypeRef(0x5678));
        self.state = SessionState::Ready;

        tracing::debug!("Created decompression session");

        Ok(())
    }

    /// Set a session property.
    pub fn set_property(
        &mut self,
        _key: &str,
        _value: CFTypeRef,
    ) -> Result<()> {
        if self.state == SessionState::Uninitialized || self.state == SessionState::Invalidated {
            return Err(HwAccelError::DeviceInit("Session not ready".to_string()));
        }

        // In real FFI:
        // let status = VTSessionSetProperty(
        //     self.compression_session.0,
        //     key_cfstring,
        //     value
        // );

        Ok(())
    }

    /// Prepare the session to encode frames.
    pub fn prepare_to_encode_frames(&mut self) -> Result<()> {
        if self.state != SessionState::Ready {
            return Err(HwAccelError::DeviceInit(
                "Session not in ready state".to_string(),
            ));
        }

        // In real FFI:
        // VTCompressionSessionPrepareToEncodeFrames(self.compression_session);

        self.state = SessionState::Active;
        Ok(())
    }

    /// Encode a frame.
    pub fn encode_frame(
        &mut self,
        _image_buffer: CVImageBufferRef,
        presentation_timestamp: CMTime,
        _duration: CMTime,
        _frame_properties: Option<CFDictionaryRef>,
    ) -> Result<()> {
        if self.state != SessionState::Active && self.state != SessionState::Ready {
            return Err(HwAccelError::Encode(format!(
                "Cannot encode in state {:?}",
                self.state
            )));
        }

        if self.state == SessionState::Ready {
            self.state = SessionState::Active;
        }

        // In real FFI:
        // let status = VTCompressionSessionEncodeFrame(
        //     self.compression_session,
        //     image_buffer,
        //     presentation_timestamp,
        //     duration,
        //     frame_properties,
        //     source_frame_ref_con,
        //     &info_flags
        // );

        self.pending_frame_count += 1;

        tracing::trace!(
            "Encoded frame at pts={}, pending={}",
            presentation_timestamp.to_seconds(),
            self.pending_frame_count
        );

        Ok(())
    }

    /// Decode a frame.
    pub fn decode_frame(
        &mut self,
        _sample_buffer: CMSampleBufferRef,
        _decode_flags: u32,
    ) -> Result<()> {
        if self.state != SessionState::Active && self.state != SessionState::Ready {
            return Err(HwAccelError::Decode(format!(
                "Cannot decode in state {:?}",
                self.state
            )));
        }

        if self.state == SessionState::Ready {
            self.state = SessionState::Active;
        }

        // In real FFI:
        // let status = VTDecompressionSessionDecodeFrame(
        //     self.decompression_session,
        //     sample_buffer,
        //     decode_flags,
        //     source_frame_ref_con,
        //     &info_flags
        // );

        self.pending_frame_count += 1;

        Ok(())
    }

    /// Flush the session.
    pub fn flush(&mut self, mode: FlushMode) -> Result<()> {
        if self.state == SessionState::Invalidated {
            return Ok(());
        }

        self.state = SessionState::Flushing;

        match mode {
            FlushMode::Sync => {
                // In real FFI:
                // VTCompressionSessionCompleteFrames(session, kCMTimePositiveInfinity);
                // or VTDecompressionSessionWaitForAsynchronousFrames(session);
                tracing::debug!("Synchronous flush, waiting for {} frames", self.pending_frame_count);
            }
            FlushMode::Async => {
                // In real FFI:
                // VTCompressionSessionCompleteFrames(session, kCMTimeInvalid);
                tracing::debug!("Asynchronous flush initiated");
            }
            FlushMode::EmitPending => {
                // Emit frames but don't block
                tracing::debug!("Emitting pending frames");
            }
        }

        self.pending_frame_count = 0;
        self.state = SessionState::Ready;

        Ok(())
    }

    /// Invalidate the session (release resources).
    pub fn invalidate(&mut self) {
        if self.state == SessionState::Invalidated {
            return;
        }

        // In real FFI:
        // if !self.compression_session.is_null() {
        //     VTCompressionSessionInvalidate(self.compression_session);
        //     CFRelease(self.compression_session);
        // }
        // if !self.decompression_session.is_null() {
        //     VTDecompressionSessionInvalidate(self.decompression_session);
        //     CFRelease(self.decompression_session);
        // }

        self.compression_session = VTCompressionSessionRef::null();
        self.decompression_session = VTDecompressionSessionRef::null();
        self.state = SessionState::Invalidated;

        tracing::debug!("Session invalidated");
    }

    /// Reset the session for format changes.
    pub fn reset_for_format_change(&mut self) -> Result<()> {
        if self.state == SessionState::Active || self.state == SessionState::Flushing {
            self.flush(FlushMode::Sync)?;
        }

        self.invalidate();
        self.state = SessionState::Uninitialized;

        Ok(())
    }

    /// Get pending frame count.
    pub fn pending_frames(&self) -> u32 {
        self.pending_frame_count
    }

    /// Notify that a frame was output (from callback).
    pub fn on_frame_output(&mut self) {
        if self.pending_frame_count > 0 {
            self.pending_frame_count -= 1;
        }
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SessionManager {
    fn drop(&mut self) {
        self.invalidate();
    }
}

// =============================================================================
// Original Types (preserved for compatibility)
// =============================================================================

/// VideoToolbox encoder preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VTPreset {
    /// Real-time encoding (lowest latency).
    Realtime,
    /// Balanced quality and speed.
    Balanced,
    /// Maximum quality (higher latency).
    MaxQuality,
}

impl VTPreset {
    /// Get the CFString key for this preset.
    pub fn key(&self) -> &'static str {
        match self {
            VTPreset::Realtime => compression_property_keys::ENABLE_LOW_LATENCY_RATE_CONTROL,
            VTPreset::Balanced => compression_property_keys::REAL_TIME,
            VTPreset::MaxQuality => compression_property_keys::MAX_KEY_FRAME_INTERVAL_DURATION,
        }
    }
}

/// VideoToolbox profile level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VTProfile {
    /// H.264 Baseline Profile.
    H264Baseline,
    /// H.264 Main Profile.
    H264Main,
    /// H.264 High Profile.
    H264High,
    /// HEVC Main Profile.
    HevcMain,
    /// HEVC Main 10 Profile (10-bit).
    HevcMain10,
    /// AV1 Main Profile (8-bit and 10-bit 4:2:0).
    /// Requires Apple Silicon M3 or later for hardware encoding.
    Av1Main,
    /// AV1 High Profile (10-bit, adds 4:4:4 support).
    /// Requires Apple Silicon M3 or later for hardware encoding.
    Av1High,
}

impl VTProfile {
    /// Get codec for this profile.
    pub fn codec(&self) -> HwCodec {
        match self {
            VTProfile::H264Baseline | VTProfile::H264Main | VTProfile::H264High => HwCodec::H264,
            VTProfile::HevcMain | VTProfile::HevcMain10 => HwCodec::Hevc,
            VTProfile::Av1Main | VTProfile::Av1High => HwCodec::Av1,
        }
    }

    /// Check if profile supports 10-bit.
    pub fn is_10bit(&self) -> bool {
        matches!(self, VTProfile::HevcMain10 | VTProfile::Av1High)
    }

    /// Check if this is an AV1 profile.
    pub fn is_av1(&self) -> bool {
        matches!(self, VTProfile::Av1Main | VTProfile::Av1High)
    }

    /// Get the profile level string for VideoToolbox.
    pub fn profile_level(&self) -> &'static str {
        match self {
            VTProfile::H264Baseline => h264_profile_level::BASELINE_AUTO_LEVEL,
            VTProfile::H264Main => h264_profile_level::MAIN_AUTO_LEVEL,
            VTProfile::H264High => h264_profile_level::HIGH_AUTO_LEVEL,
            VTProfile::HevcMain => hevc_profile_level::MAIN_AUTO_LEVEL,
            VTProfile::HevcMain10 => hevc_profile_level::MAIN10_AUTO_LEVEL,
            VTProfile::Av1Main => av1_profile_level::MAIN_AUTO_LEVEL,
            VTProfile::Av1High => av1_profile_level::HIGH_AUTO_LEVEL,
        }
    }

    /// Get the codec type (FourCC).
    pub fn codec_type(&self) -> u32 {
        match self {
            VTProfile::H264Baseline | VTProfile::H264Main | VTProfile::H264High => {
                codec_type::H264
            }
            VTProfile::HevcMain | VTProfile::HevcMain10 => codec_type::HEVC,
            VTProfile::Av1Main | VTProfile::Av1High => codec_type::AV1,
        }
    }
}

/// VideoToolbox rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VTRateControl {
    /// Constant bit rate.
    Cbr,
    /// Average bit rate.
    Abr,
    /// Variable bit rate with quality target.
    Vbr,
    /// Constant quality (CRF-like).
    Cq { quality: u8 },
}

/// CVPixelBuffer format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CVPixelFormat {
    /// NV12 (4:2:0 bi-planar).
    Nv12,
    /// BGRA 32-bit.
    Bgra32,
    /// ARGB 32-bit.
    Argb32,
    /// YUV 4:2:2 packed.
    Yuyv422,
    /// 10-bit P010 format.
    P010,
}

impl CVPixelFormat {
    /// Get the OSType value for this format.
    pub fn os_type(&self) -> u32 {
        match self {
            CVPixelFormat::Nv12 => 0x3432_564E,   // '420v'
            CVPixelFormat::Bgra32 => 0x4247_5241, // 'BGRA'
            CVPixelFormat::Argb32 => 0x4152_4742, // 'ARGB'
            CVPixelFormat::Yuyv422 => 0x7975_7673, // 'yuvs'
            CVPixelFormat::P010 => 0x7030_3130,  // 'p010'
        }
    }

    /// Get bytes per pixel.
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            CVPixelFormat::Nv12 => 1.5,
            CVPixelFormat::Bgra32 | CVPixelFormat::Argb32 => 4.0,
            CVPixelFormat::Yuyv422 => 2.0,
            CVPixelFormat::P010 => 3.0, // 10-bit NV12
        }
    }
}

/// CVPixelBuffer pool for efficient buffer reuse.
pub struct CVPixelBufferPool {
    /// Pool width.
    width: u32,
    /// Pool height.
    height: u32,
    /// Pixel format.
    format: CVPixelFormat,
    /// Available buffers.
    available: Mutex<VecDeque<CVPixelBuffer>>,
    /// Maximum pool size.
    max_size: usize,
    /// Buffer recycler for tracking.
    recycler: BufferRecycler<()>,
}

impl CVPixelBufferPool {
    /// Create a new pixel buffer pool.
    pub fn new(width: u32, height: u32, format: CVPixelFormat, max_size: usize) -> Self {
        Self {
            width,
            height,
            format,
            available: Mutex::new(VecDeque::with_capacity(max_size)),
            max_size,
            recycler: BufferRecycler::new(max_size),
        }
    }

    /// Get a buffer from the pool (or create a new one).
    pub fn get_buffer(&self) -> CVPixelBuffer {
        let mut available = self.available.lock();
        if let Some(buffer) = available.pop_front() {
            buffer
        } else {
            self.recycler.record_creation();
            CVPixelBuffer::new(self.width, self.height, self.format)
        }
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&self, buffer: CVPixelBuffer) {
        let mut available = self.available.lock();
        if available.len() < self.max_size {
            available.push_back(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Get pool dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Flush all buffers from the pool.
    pub fn flush(&self) {
        let mut available = self.available.lock();
        available.clear();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> BufferRecyclerStats {
        self.recycler.stats()
    }
}

/// CVPixelBuffer wrapper.
pub struct CVPixelBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: CVPixelFormat,
    /// Plane data (for planar formats).
    planes: Vec<Vec<u8>>,
    /// Base address (for packed formats).
    data: Vec<u8>,
    /// Lock state for thread safety.
    locked: Mutex<bool>,
}

impl CVPixelBuffer {
    /// Create a new pixel buffer.
    pub fn new(width: u32, height: u32, format: CVPixelFormat) -> Self {
        let total_bytes = (width as f32 * height as f32 * format.bytes_per_pixel()) as usize;

        let (planes, data) = match format {
            CVPixelFormat::Nv12 => {
                // Y plane + UV plane
                let y_size = (width * height) as usize;
                let uv_size = (width * height / 2) as usize;
                (vec![vec![0u8; y_size], vec![0u8; uv_size]], Vec::new())
            }
            CVPixelFormat::P010 => {
                // 10-bit Y plane + UV plane (16-bit per component)
                let y_size = (width * height * 2) as usize;
                let uv_size = (width * height) as usize;
                (vec![vec![0u8; y_size], vec![0u8; uv_size]], Vec::new())
            }
            _ => (Vec::new(), vec![0u8; total_bytes]),
        };

        Self {
            width,
            height,
            format,
            planes,
            data,
            locked: Mutex::new(false),
        }
    }

    /// Get plane data (for planar formats).
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        self.planes.get(index).map(|v| v.as_slice())
    }

    /// Get mutable plane data.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        self.planes.get_mut(index).map(|v| v.as_mut_slice())
    }

    /// Get packed data (for packed formats).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable packed data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get stride for a plane.
    pub fn stride(&self, plane: usize) -> usize {
        let bytes_per_component = if self.format == CVPixelFormat::P010 {
            2
        } else {
            1
        };
        match self.format {
            CVPixelFormat::Nv12 | CVPixelFormat::P010 => {
                // Y plane and UV plane have same stride in standard NV12/P010
                let _ = plane; // Both planes use same width-based stride
                self.width as usize * bytes_per_component
            }
            CVPixelFormat::Bgra32 | CVPixelFormat::Argb32 => self.width as usize * 4,
            CVPixelFormat::Yuyv422 => self.width as usize * 2,
        }
    }

    /// Lock the buffer for CPU access.
    /// In real FFI, this would call CVPixelBufferLockBaseAddress.
    pub fn lock(&self) -> Result<()> {
        let mut locked = self.locked.lock();
        if *locked {
            return Err(HwAccelError::BufferAlloc(
                "Buffer already locked".to_string(),
            ));
        }
        *locked = true;
        Ok(())
    }

    /// Unlock the buffer.
    /// In real FFI, this would call CVPixelBufferUnlockBaseAddress.
    pub fn unlock(&self) -> Result<()> {
        let mut locked = self.locked.lock();
        if !*locked {
            return Err(HwAccelError::BufferAlloc("Buffer not locked".to_string()));
        }
        *locked = false;
        Ok(())
    }

    /// Check if buffer is locked.
    pub fn is_locked(&self) -> bool {
        *self.locked.lock()
    }
}

/// VideoToolbox context.
#[derive(Debug)]
pub struct VideoToolboxContext {
    /// Whether the context is initialized.
    _initialized: bool,
    /// Supported encode profiles.
    encode_profiles: Vec<VTProfile>,
    /// Supports hardware encoder.
    hardware_encoder: bool,
    /// Available ProRes profiles.
    prores_profiles: Vec<ProResProfile>,
    /// AV1 hardware encoding available (M3 or later).
    av1_encode_available: bool,
    /// AV1 hardware decoding available (M1 or later).
    av1_decode_available: bool,
}

impl VideoToolboxContext {
    /// Create a new VideoToolbox context.
    pub fn new() -> Result<Self> {
        // In a real implementation, this would:
        // 1. Check for VideoToolbox framework availability
        // 2. Query supported codecs and profiles
        // 3. Detect if hardware encoder is available
        // 4. Check for AV1 support (requires M1+ for decode, M3+ for encode)

        // Detect AV1 capabilities based on chip generation
        // In real implementation, this would query the actual hardware
        let av1_encode_available = Self::detect_av1_encode_support();
        let av1_decode_available = Self::detect_av1_decode_support();

        let mut encode_profiles = vec![
            VTProfile::H264Baseline,
            VTProfile::H264Main,
            VTProfile::H264High,
            VTProfile::HevcMain,
            VTProfile::HevcMain10,
        ];

        // Add AV1 profiles if hardware encoding is available
        if av1_encode_available {
            encode_profiles.push(VTProfile::Av1Main);
            encode_profiles.push(VTProfile::Av1High);
        }

        Ok(Self {
            _initialized: true,
            encode_profiles,
            hardware_encoder: true, // Apple Silicon always has HW encoder
            prores_profiles: vec![
                ProResProfile::Proxy,
                ProResProfile::Lt,
                ProResProfile::Standard,
                ProResProfile::Hq,
                ProResProfile::P4444,
                ProResProfile::P4444Xq,
            ],
            av1_encode_available,
            av1_decode_available,
        })
    }

    /// Detect if AV1 hardware encoding is available.
    /// AV1 encoding requires Apple Silicon M3 or later.
    fn detect_av1_encode_support() -> bool {
        // In a real implementation, this would:
        // 1. Check system version (macOS 14.0+)
        // 2. Query VTCopySupportedPropertyDictionaryForEncoder for AV1
        // 3. Check chip generation (M3 or later)
        //
        // For now, we return false to be conservative.
        // Real detection would use IOKit to query chip info.
        false
    }

    /// Detect if AV1 hardware decoding is available.
    /// AV1 decoding is available on Apple Silicon M1 and later.
    fn detect_av1_decode_support() -> bool {
        // In a real implementation, this would:
        // 1. Check system version (macOS 13.0+)
        // 2. Query VTIsHardwareDecodeSupported for AV1
        //
        // For now, we assume M1+ availability on Apple Silicon.
        #[cfg(target_arch = "aarch64")]
        {
            true // All Apple Silicon Macs support AV1 decode
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false // Intel Macs don't have AV1 hardware decode
        }
    }

    /// Get capabilities.
    pub fn capabilities(&self) -> HwCapabilities {
        let mut encode_codecs = vec![HwCodec::H264, HwCodec::Hevc];
        let mut decode_codecs = vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9];

        if self.av1_encode_available {
            encode_codecs.push(HwCodec::Av1);
        }
        if self.av1_decode_available {
            decode_codecs.push(HwCodec::Av1);
        }

        HwCapabilities {
            accel_type: crate::HwAccelType::VideoToolbox,
            encode_codecs,
            decode_codecs,
            max_width: 8192,
            max_height: 4320,
            supports_bframes: true,
            supports_10bit: true,
            supports_hdr: true,
            device_name: "Apple VideoToolbox".to_string(),
        }
    }

    /// Check if codec is supported for encoding.
    pub fn supports_encode(&self, codec: HwCodec) -> bool {
        match codec {
            HwCodec::H264 | HwCodec::Hevc => true,
            HwCodec::Av1 => self.av1_encode_available,
            _ => false,
        }
    }

    /// Check if codec is supported for decoding.
    pub fn supports_decode(&self, codec: HwCodec) -> bool {
        match codec {
            HwCodec::H264 | HwCodec::Hevc | HwCodec::Vp9 => true,
            HwCodec::Av1 => self.av1_decode_available,
            _ => false,
        }
    }

    /// Check if AV1 hardware encoding is available.
    pub fn has_av1_encode(&self) -> bool {
        self.av1_encode_available
    }

    /// Check if AV1 hardware decoding is available.
    pub fn has_av1_decode(&self) -> bool {
        self.av1_decode_available
    }

    /// Check if hardware encoder is available.
    pub fn has_hardware_encoder(&self) -> bool {
        self.hardware_encoder
    }

    /// Get supported profiles.
    pub fn supported_profiles(&self) -> &[VTProfile] {
        &self.encode_profiles
    }

    /// Check if ProRes encoding is supported.
    pub fn supports_prores(&self) -> bool {
        !self.prores_profiles.is_empty()
    }

    /// Get supported ProRes profiles.
    pub fn supported_prores_profiles(&self) -> &[ProResProfile] {
        &self.prores_profiles
    }
}

impl Default for VideoToolboxContext {
    fn default() -> Self {
        // Return an "unavailable" context instead of panicking
        // Callers should use new() and handle the Result for proper initialization
        Self::new().unwrap_or_else(|_| Self {
            _initialized: false,
            encode_profiles: Vec::new(),
            hardware_encoder: false,
            prores_profiles: Vec::new(),
            av1_encode_available: false,
            av1_decode_available: false,
        })
    }
}

/// VideoToolbox encoder configuration.
#[derive(Debug, Clone)]
pub struct VTEncoderConfig {
    /// Base configuration.
    pub base: crate::encoder::HwEncoderConfig,
    /// Encoder profile.
    pub profile: VTProfile,
    /// Rate control mode.
    pub rate_control: VTRateControl,
    /// Preset for quality/speed tradeoff.
    pub preset: VTPreset,
    /// Enable B-frames.
    pub b_frames: bool,
    /// Maximum B-frame count.
    pub max_b_frames: u32,
    /// Reference frame count.
    pub ref_frames: u32,
    /// Enable entropy mode (CABAC for H.264 High profile).
    pub cabac: bool,
    /// Allow frame reordering (required for B-frames).
    pub allow_frame_reordering: bool,
    /// Maximum keyframe interval in seconds.
    pub max_keyframe_interval: f64,
    /// Expected source frame rate.
    pub expected_fps: f64,
    /// Enable hardware acceleration explicitly.
    pub require_hardware: bool,
    /// Hardware encoder hints.
    pub hardware_hints: HardwareEncoderHints,
}

impl Default for VTEncoderConfig {
    fn default() -> Self {
        Self {
            base: crate::encoder::HwEncoderConfig::default(),
            profile: VTProfile::H264High,
            rate_control: VTRateControl::Vbr,
            preset: VTPreset::Balanced,
            b_frames: true,
            max_b_frames: 2,
            ref_frames: 4,
            cabac: true,
            allow_frame_reordering: true,
            max_keyframe_interval: 2.0,
            expected_fps: 30.0,
            require_hardware: true,
            hardware_hints: HardwareEncoderHints::default(),
        }
    }
}

/// Encoded frame output from async callback.
#[derive(Debug)]
pub struct VTEncodedFrame {
    /// NAL units.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Is keyframe (IDR).
    pub is_keyframe: bool,
    /// Frame type.
    pub frame_type: crate::encoder::FrameType,
    /// Encoded frame size in bytes.
    pub size: usize,
}

/// Async callback output queue.
struct EncoderOutputQueue {
    /// Pending encoded frames.
    frames: VecDeque<VTEncodedFrame>,
    /// Encoding error if any.
    error: Option<HwAccelError>,
}

impl EncoderOutputQueue {
    fn new() -> Self {
        Self {
            frames: VecDeque::new(),
            error: None,
        }
    }
}

/// VideoToolbox encoder session.
pub struct VTEncoder {
    _context: VideoToolboxContext,
    config: VTEncoderConfig,
    frame_count: u64,
    /// Pixel buffer pool for input frames.
    buffer_pool: CVPixelBufferPool,
    /// Output queue (filled by async callbacks).
    output_queue: Arc<Mutex<EncoderOutputQueue>>,
    /// Pending frames awaiting output.
    pending_frames: u32,
    /// Sequence parameter set (cached).
    sps: Option<Vec<u8>>,
    /// Picture parameter set (cached).
    pps: Option<Vec<u8>>,
    /// Video parameter set (HEVC only).
    vps: Option<Vec<u8>>,
    /// Session manager for lifecycle control.
    session_manager: SessionManager,
    /// Compression output context for callbacks.
    output_context: VTCompressionOutputContext,
}

impl VTEncoder {
    /// Create a new VideoToolbox encoder.
    pub fn new(config: VTEncoderConfig) -> Result<Self> {
        let context = VideoToolboxContext::new()?;

        if !context.supports_encode(config.profile.codec()) {
            return Err(HwAccelError::CodecNotSupported(
                config.profile.codec().name().to_string(),
            ));
        }

        let pixel_format = if config.profile.is_10bit() {
            CVPixelFormat::P010
        } else {
            CVPixelFormat::Nv12
        };

        let buffer_pool = CVPixelBufferPool::new(
            config.base.width,
            config.base.height,
            pixel_format,
            8, // Pool size for lookahead
        );

        let mut session_manager = SessionManager::with_hints(config.hardware_hints.clone());

        // Create compression session
        session_manager.create_compression_session(
            config.base.width,
            config.base.height,
            config.profile.codec_type(),
            None,
            None,
        )?;

        let bitrate_kbps = match &config.base.rate_control {
            crate::types::HwRateControl::Vbr { target, .. } => target / 1000,
            crate::types::HwRateControl::Cbr(bitrate) => bitrate / 1000,
            crate::types::HwRateControl::Cqp(_) | crate::types::HwRateControl::Cq(_) => 0,
        };

        tracing::info!(
            "Created VideoToolbox encoder: {:?} {}x{} @ {} kbps",
            config.profile,
            config.base.width,
            config.base.height,
            bitrate_kbps
        );

        Ok(Self {
            _context: context,
            config,
            frame_count: 0,
            buffer_pool,
            output_queue: Arc::new(Mutex::new(EncoderOutputQueue::new())),
            pending_frames: 0,
            sps: None,
            pps: None,
            vps: None,
            session_manager,
            output_context: VTCompressionOutputContext::new(),
        })
    }

    /// Create encoder with base config.
    pub fn with_base_config(base: crate::encoder::HwEncoderConfig) -> Result<Self> {
        let profile = match base.codec {
            HwCodec::H264 => VTProfile::H264High,
            HwCodec::Hevc => VTProfile::HevcMain,
            HwCodec::Av1 => VTProfile::Av1Main,
            _ => {
                return Err(HwAccelError::CodecNotSupported(
                    base.codec.name().to_string(),
                ))
            }
        };

        let config = VTEncoderConfig {
            base,
            profile,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Get a pixel buffer from the pool.
    pub fn get_input_buffer(&self) -> CVPixelBuffer {
        self.buffer_pool.get_buffer()
    }

    /// Return a pixel buffer to the pool.
    pub fn return_input_buffer(&self, buffer: CVPixelBuffer) {
        self.buffer_pool.return_buffer(buffer);
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &HwFrame) -> Result<Option<crate::encoder::HwPacket>> {
        // Ensure session is active
        if self.session_manager.state() == SessionState::Ready {
            self.session_manager.prepare_to_encode_frames()?;
        }

        let is_keyframe = self.frame_count.is_multiple_of(self.config.base.gop_size as u64);
        self.frame_count += 1;
        self.pending_frames += 1;

        // Encode via session manager
        let pts = CMTime::new(frame.pts, 90000); // 90kHz timebase
        let duration = CMTime::new(
            90000 / self.config.base.frame_rate.0 as i64,
            90000,
        );

        self.session_manager.encode_frame(
            CVImageBufferRef::null(),
            pts,
            duration,
            None,
        )?;

        // Simulate async callback result
        let frame_type = if is_keyframe {
            crate::encoder::FrameType::I
        } else if self.config.b_frames && self.frame_count % 3 == 2 {
            crate::encoder::FrameType::B
        } else {
            crate::encoder::FrameType::P
        };

        // Simulate callback completing
        self.session_manager.on_frame_output();
        self.pending_frames -= 1;

        Ok(Some(crate::encoder::HwPacket {
            data: vec![0u8; 1000],
            pts: frame.pts,
            dts: frame.pts,
            is_keyframe,
            frame_type,
        }))
    }

    /// Encode with explicit pixel buffer.
    pub fn encode_buffer(
        &mut self,
        buffer: CVPixelBuffer,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<()> {
        let _ = (buffer, pts, force_keyframe);
        self.pending_frames += 1;
        Ok(())
    }

    /// Get next encoded frame from output queue.
    pub fn get_encoded_frame(&mut self) -> Result<Option<VTEncodedFrame>> {
        // Check callback context first
        if let Some(err) = self.output_context.take_error() {
            return Err(err);
        }

        if let Some(frame) = self.output_context.pop_frame() {
            return Ok(Some(frame));
        }

        // Fall back to internal queue
        let mut queue = self.output_queue.lock();

        if let Some(error) = queue.error.take() {
            return Err(error);
        }

        Ok(queue.frames.pop_front())
    }

    /// Flush the encoder synchronously.
    pub fn flush(&mut self) -> Result<Vec<crate::encoder::HwPacket>> {
        self.session_manager.flush(FlushMode::Sync)?;

        tracing::debug!(
            "Flushing VideoToolbox encoder, {} pending frames",
            self.pending_frames
        );

        self.pending_frames = 0;
        Ok(Vec::new())
    }

    /// Flush the encoder asynchronously.
    pub fn flush_async(&mut self) -> Result<()> {
        self.session_manager.flush(FlushMode::Async)
    }

    /// Reset the encoder for format changes.
    pub fn reset(&mut self) -> Result<()> {
        self.session_manager.reset_for_format_change()?;

        // Recreate session
        self.session_manager.create_compression_session(
            self.config.base.width,
            self.config.base.height,
            self.config.profile.codec_type(),
            None,
            None,
        )?;

        self.frame_count = 0;
        self.pending_frames = 0;
        self.sps = None;
        self.pps = None;
        self.vps = None;

        Ok(())
    }

    /// Get sequence parameter set.
    pub fn get_sps(&self) -> Option<&[u8]> {
        self.sps.as_deref()
    }

    /// Get picture parameter set.
    pub fn get_pps(&self) -> Option<&[u8]> {
        self.pps.as_deref()
    }

    /// Get video parameter set (HEVC only).
    pub fn get_vps(&self) -> Option<&[u8]> {
        self.vps.as_deref()
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> VTEncoderStats {
        VTEncoderStats {
            frames_encoded: self.frame_count,
            pending_frames: self.pending_frames,
            average_bitrate: 0,
            buffer_stats: self.buffer_pool.stats(),
        }
    }

    /// Get session state.
    pub fn session_state(&self) -> SessionState {
        self.session_manager.state()
    }
}

/// VideoToolbox encoder statistics.
#[derive(Debug, Clone)]
pub struct VTEncoderStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Frames pending in encoder.
    pub pending_frames: u32,
    /// Average bitrate achieved.
    pub average_bitrate: u64,
    /// Buffer pool statistics.
    pub buffer_stats: BufferRecyclerStats,
}

/// VideoToolbox decoder session.
pub struct VTDecoder {
    _context: VideoToolboxContext,
    config: crate::decoder::HwDecoderConfig,
    width: u32,
    height: u32,
    /// Pixel buffer pool for output frames.
    buffer_pool: Option<CVPixelBufferPool>,
    /// Frames decoded.
    frames_decoded: u64,
    /// Session manager for lifecycle control.
    session_manager: SessionManager,
    /// Decompression output context for callbacks.
    output_context: VTDecompressionOutputContext,
}

impl VTDecoder {
    /// Create a new VideoToolbox decoder.
    pub fn new(config: crate::decoder::HwDecoderConfig) -> Result<Self> {
        let context = VideoToolboxContext::new()?;

        if !context.supports_decode(config.codec) {
            return Err(HwAccelError::CodecNotSupported(
                config.codec.name().to_string(),
            ));
        }

        let mut session_manager = SessionManager::new();

        // Create decompression session
        session_manager.create_decompression_session(
            CMVideoFormatDescriptionRef::null(),
            None,
            None,
        )?;

        tracing::info!("Created VideoToolbox decoder for {:?}", config.codec);

        Ok(Self {
            _context: context,
            config,
            width: 0,
            height: 0,
            buffer_pool: None,
            frames_decoded: 0,
            session_manager,
            output_context: VTDecompressionOutputContext::new(),
        })
    }

    /// Decode a packet.
    pub fn decode(&mut self, packet: &crate::decoder::DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        // Decode via session manager
        self.session_manager
            .decode_frame(CMSampleBufferRef::null(), 0)?;

        self.width = 1920;
        self.height = 1080;
        self.frames_decoded += 1;

        // Create buffer pool on first frame
        if self.buffer_pool.is_none() {
            self.buffer_pool = Some(CVPixelBufferPool::new(
                self.width,
                self.height,
                CVPixelFormat::Nv12,
                4,
            ));
        }

        // Simulate callback completing
        self.session_manager.on_frame_output();

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::Cpu(vec![0u8; (self.width * self.height * 3 / 2) as usize]),
        }))
    }

    /// Get next decoded frame from output queue.
    pub fn get_decoded_frame(&mut self) -> Result<Option<VTDecodedFrame>> {
        if let Some(err) = self.output_context.take_error() {
            return Err(err);
        }

        Ok(self.output_context.pop_frame())
    }

    /// Flush the decoder.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>> {
        self.session_manager.flush(FlushMode::Sync)?;
        Ok(Vec::new())
    }

    /// Reset the decoder for format changes.
    pub fn reset(&mut self) -> Result<()> {
        self.session_manager.reset_for_format_change()?;

        self.session_manager.create_decompression_session(
            CMVideoFormatDescriptionRef::null(),
            None,
            None,
        )?;

        self.width = 0;
        self.height = 0;
        self.frames_decoded = 0;
        self.buffer_pool = None;

        Ok(())
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get frames decoded count.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Get session state.
    pub fn session_state(&self) -> SessionState {
        self.session_manager.state()
    }
}

// =============================================================================
// ProRes Encoder/Decoder
// =============================================================================

/// ProRes encoder using VideoToolbox.
pub struct ProResEncoder {
    _context: VideoToolboxContext,
    config: ProResEncoderConfig,
    frame_count: u64,
    session_manager: SessionManager,
    /// Compression output context (used by real FFI callbacks).
    #[allow(dead_code)]
    output_context: VTCompressionOutputContext,
}

impl ProResEncoder {
    /// Create a new ProRes encoder.
    pub fn new(config: ProResEncoderConfig) -> Result<Self> {
        config.validate()?;

        let context = VideoToolboxContext::new()?;

        if !context.supports_prores() {
            return Err(HwAccelError::CodecNotSupported(
                "ProRes not supported".to_string(),
            ));
        }

        let mut session_manager = SessionManager::new();

        session_manager.create_compression_session(
            config.width,
            config.height,
            config.profile.codec_type(),
            None,
            None,
        )?;

        tracing::info!(
            "Created ProRes encoder: {} {}x{} @ {} Mbps (approx)",
            config.profile.fourcc(),
            config.width,
            config.height,
            config.profile.approx_bitrate_mbps()
        );

        Ok(Self {
            _context: context,
            config,
            frame_count: 0,
            session_manager,
            output_context: VTCompressionOutputContext::new(),
        })
    }

    /// Encode a frame.
    pub fn encode(&mut self, _frame: &CVPixelBuffer) -> Result<Option<Vec<u8>>> {
        let pts = CMTime::new(self.frame_count as i64, self.config.frame_rate.0 as i32);
        let duration = CMTime::new(1, self.config.frame_rate.0 as i32);

        self.session_manager.encode_frame(
            CVImageBufferRef::null(),
            pts,
            duration,
            None,
        )?;

        self.frame_count += 1;
        self.session_manager.on_frame_output();

        // ProRes frames are all intra-coded (I-frames only)
        Ok(Some(vec![0u8; 50000])) // Simulated ProRes frame
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        self.session_manager.flush(FlushMode::Sync)?;
        Ok(Vec::new())
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> ProResEncoderStats {
        ProResEncoderStats {
            frames_encoded: self.frame_count,
            profile: self.config.profile,
        }
    }
}

/// ProRes encoder statistics.
#[derive(Debug, Clone)]
pub struct ProResEncoderStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Profile being used.
    pub profile: ProResProfile,
}

/// ProRes decoder using VideoToolbox.
pub struct ProResDecoder {
    _context: VideoToolboxContext,
    config: ProResDecoderConfig,
    frames_decoded: u64,
    width: u32,
    height: u32,
    session_manager: SessionManager,
    /// Decompression output context (used by real FFI callbacks).
    #[allow(dead_code)]
    output_context: VTDecompressionOutputContext,
}

impl ProResDecoder {
    /// Create a new ProRes decoder.
    pub fn new(config: ProResDecoderConfig) -> Result<Self> {
        let context = VideoToolboxContext::new()?;

        let mut session_manager = SessionManager::new();

        session_manager.create_decompression_session(
            CMVideoFormatDescriptionRef::null(),
            None,
            None,
        )?;

        tracing::info!("Created ProRes decoder");

        Ok(Self {
            _context: context,
            config,
            frames_decoded: 0,
            width: 0,
            height: 0,
            session_manager,
            output_context: VTDecompressionOutputContext::new(),
        })
    }

    /// Decode a ProRes frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<CVPixelBuffer>> {
        if data.is_empty() {
            return Ok(None);
        }

        self.session_manager
            .decode_frame(CMSampleBufferRef::null(), 0)?;

        self.width = 1920;
        self.height = 1080;
        self.frames_decoded += 1;

        self.session_manager.on_frame_output();

        let buffer = CVPixelBuffer::new(self.width, self.height, self.config.output_format);
        Ok(Some(buffer))
    }

    /// Flush the decoder.
    pub fn flush(&mut self) -> Result<Vec<CVPixelBuffer>> {
        self.session_manager.flush(FlushMode::Sync)?;
        Ok(Vec::new())
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get frames decoded count.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_videotoolbox_context() {
        let ctx = VideoToolboxContext::new().unwrap();
        assert!(ctx.supports_encode(HwCodec::H264));
        assert!(ctx.supports_decode(HwCodec::Hevc));
        assert!(ctx.has_hardware_encoder());
    }

    #[test]
    fn test_videotoolbox_capabilities() {
        let ctx = VideoToolboxContext::new().unwrap();
        let caps = ctx.capabilities();
        assert!(caps.supports_hdr);
        assert_eq!(caps.max_width, 8192);
    }

    #[test]
    fn test_cv_pixel_buffer_pool() {
        let pool = CVPixelBufferPool::new(1920, 1080, CVPixelFormat::Nv12, 4);
        let buffer = pool.get_buffer();
        assert_eq!(buffer.width, 1920);
        assert_eq!(buffer.height, 1080);

        pool.return_buffer(buffer);
    }

    #[test]
    fn test_cv_pixel_buffer_planes() {
        let buffer = CVPixelBuffer::new(1920, 1080, CVPixelFormat::Nv12);

        // Y plane
        assert!(buffer.plane(0).is_some());
        assert_eq!(buffer.plane(0).unwrap().len(), 1920 * 1080);

        // UV plane
        assert!(buffer.plane(1).is_some());
        assert_eq!(buffer.plane(1).unwrap().len(), 1920 * 1080 / 2);
    }

    #[test]
    fn test_vt_encoder_config() {
        let config = VTEncoderConfig::default();
        assert_eq!(config.profile, VTProfile::H264High);
        assert!(config.b_frames);
        assert!(config.cabac);
    }

    #[test]
    fn test_vt_profile() {
        assert_eq!(VTProfile::H264High.codec(), HwCodec::H264);
        assert_eq!(VTProfile::HevcMain10.codec(), HwCodec::Hevc);
        assert!(VTProfile::HevcMain10.is_10bit());
        assert!(!VTProfile::H264High.is_10bit());
    }

    #[test]
    fn test_cmtime() {
        let time = CMTime::new(90000, 90000);
        assert_eq!(time.to_seconds(), 1.0);
        assert!(time.is_valid());

        let time2 = CMTime::from_seconds(2.5, 90000);
        assert!((time2.to_seconds() - 2.5).abs() < 0.0001);
    }

    #[test]
    fn test_cmtime_constants() {
        assert!(!CMTime::INVALID.is_valid());
        assert!(CMTime::ZERO.is_valid());
        assert!(CMTime::POSITIVE_INFINITY.is_positive_infinity());
    }

    #[test]
    fn test_session_state() {
        let mut manager = SessionManager::new();
        assert_eq!(manager.state(), SessionState::Uninitialized);

        manager
            .create_compression_session(1920, 1080, codec_type::H264, None, None)
            .unwrap();
        assert_eq!(manager.state(), SessionState::Ready);
    }

    #[test]
    fn test_session_flush() {
        let mut manager = SessionManager::new();
        manager
            .create_compression_session(1920, 1080, codec_type::H264, None, None)
            .unwrap();

        manager.prepare_to_encode_frames().unwrap();
        assert_eq!(manager.state(), SessionState::Active);

        manager.flush(FlushMode::Sync).unwrap();
        assert_eq!(manager.state(), SessionState::Ready);
    }

    #[test]
    fn test_prores_profile() {
        assert_eq!(ProResProfile::Hq.fourcc(), "apch");
        assert!(ProResProfile::P4444.supports_alpha());
        assert!(!ProResProfile::Standard.supports_alpha());
        assert_eq!(ProResProfile::Standard.chroma_subsampling(), "4:2:2");
        assert_eq!(ProResProfile::P4444.chroma_subsampling(), "4:4:4");
    }

    #[test]
    fn test_prores_encoder_config() {
        let config = ProResEncoderConfig::new(ProResProfile::Hq, 1920, 1080);
        assert!(config.validate().is_ok());

        let bad_config = ProResEncoderConfig {
            profile: ProResProfile::Standard,
            include_alpha: true,
            ..config.clone()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_cf_guard() {
        let ref_val = CFTypeRef(0x1234);
        let guard = CFGuard::new(ref_val);
        assert_eq!(guard.get().0, 0x1234);
    }

    #[test]
    fn test_autorelease_pool() {
        let result = AutoreleasePool::execute(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_buffer_recycler() {
        let recycler: BufferRecycler<Vec<u8>> = BufferRecycler::new(4);

        // Get buffer (creates new)
        assert!(recycler.try_get().is_none());

        // Recycle a buffer
        recycler.recycle(vec![0u8; 100]);
        assert!(recycler.try_get().is_some());
    }

    #[test]
    fn test_compression_output_context() {
        let ctx = VTCompressionOutputContext::new();

        let frame = VTEncodedFrame {
            data: vec![0u8; 100],
            pts: 0,
            dts: 0,
            is_keyframe: true,
            frame_type: crate::encoder::FrameType::I,
            size: 100,
        };

        ctx.push_frame(frame);
        assert_eq!(ctx.pending_count(), 1);

        let popped = ctx.pop_frame();
        assert!(popped.is_some());
        assert_eq!(ctx.pending_count(), 0);
    }

    #[test]
    fn test_pixel_buffer_lock() {
        let buffer = CVPixelBuffer::new(1920, 1080, CVPixelFormat::Nv12);
        assert!(!buffer.is_locked());

        buffer.lock();
        assert!(buffer.is_locked());

        // Double lock should fail
        assert!(buffer.lock().is_err());

        buffer.unlock().unwrap();
        assert!(!buffer.is_locked());

        // Double unlock should fail
        assert!(buffer.unlock().is_err());
    }

    #[test]
    fn test_hardware_encoder_hints() {
        let hints = HardwareEncoderHints::require_hardware();
        assert!(hints.require_hardware);
        assert!(hints.prefer_hardware);

        let low_power = HardwareEncoderHints::low_power();
        assert!(low_power.prefer_low_power);
    }

    #[test]
    fn test_vt_profile_level_strings() {
        assert_eq!(
            VTProfile::H264High.profile_level(),
            h264_profile_level::HIGH_AUTO_LEVEL
        );
        assert_eq!(
            VTProfile::HevcMain10.profile_level(),
            hevc_profile_level::MAIN10_AUTO_LEVEL
        );
    }

    #[test]
    fn test_codec_types() {
        assert_eq!(VTProfile::H264High.codec_type(), codec_type::H264);
        assert_eq!(VTProfile::HevcMain.codec_type(), codec_type::HEVC);
        assert_eq!(ProResProfile::Hq.codec_type(), codec_type::PRORES_422_HQ);
    }
}
