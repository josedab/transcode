//! VA-API (Video Acceleration API) implementation for Linux.
//!
//! VA-API provides hardware-accelerated video encoding and decoding on Linux
//! systems, supporting Intel, AMD, and NVIDIA GPUs (via nouveau).
//!
//! This module provides real FFI bindings to libva when the `vaapi` feature
//! is enabled. The types mirror the libva C API for complete compatibility.

// =============================================================================
// Real FFI Bindings Module (enabled with vaapi feature on Linux)
// =============================================================================

/// Real FFI bindings to VA-API (libva).
/// These bindings link directly to libva on Linux systems.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
pub mod ffi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_uint, c_ulong};

    // -------------------------------------------------------------------------
    // Core VA-API Types
    // -------------------------------------------------------------------------

    /// VA Display handle - connection to VA-API driver.
    pub type VADisplay = *mut c_void;

    /// VA Config ID - encoding/decoding configuration.
    pub type VAConfigID = c_uint;

    /// VA Context ID - encoding/decoding context.
    pub type VAContextID = c_uint;

    /// VA Surface ID - video surface.
    pub type VASurfaceID = c_uint;

    /// VA Buffer ID - data buffer.
    pub type VABufferID = c_uint;

    /// VA Image ID - image for CPU access.
    pub type VAImageID = c_uint;

    /// VA Status return code.
    pub type VAStatus = c_int;

    /// VA Profile.
    pub type VAProfile = c_int;

    /// VA Entrypoint.
    pub type VAEntrypoint = c_int;

    /// VA Buffer type.
    pub type VABufferType = c_int;

    /// VA Generic value.
    #[repr(C)]
    pub union VAGenericValue {
        pub i: c_int,
        pub f: f32,
        pub p: *mut c_void,
        pub func: Option<extern "C" fn()>,
    }

    impl Default for VAGenericValue {
        fn default() -> Self {
            VAGenericValue { i: 0 }
        }
    }

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    /// Invalid surface ID.
    pub const VA_INVALID_SURFACE: VASurfaceID = 0xFFFFFFFF;
    /// Invalid ID.
    pub const VA_INVALID_ID: c_uint = 0xFFFFFFFF;

    /// Status codes.
    pub const VA_STATUS_SUCCESS: VAStatus = 0x00000000;
    pub const VA_STATUS_ERROR_OPERATION_FAILED: VAStatus = 0x00000001;
    pub const VA_STATUS_ERROR_ALLOCATION_FAILED: VAStatus = 0x00000002;
    pub const VA_STATUS_ERROR_INVALID_DISPLAY: VAStatus = 0x00000003;
    pub const VA_STATUS_ERROR_INVALID_CONFIG: VAStatus = 0x00000004;
    pub const VA_STATUS_ERROR_INVALID_CONTEXT: VAStatus = 0x00000005;
    pub const VA_STATUS_ERROR_INVALID_SURFACE: VAStatus = 0x00000006;
    pub const VA_STATUS_ERROR_INVALID_BUFFER: VAStatus = 0x00000007;
    pub const VA_STATUS_ERROR_INVALID_IMAGE: VAStatus = 0x00000008;
    pub const VA_STATUS_ERROR_INVALID_SUBPICTURE: VAStatus = 0x00000009;
    pub const VA_STATUS_ERROR_ATTR_NOT_SUPPORTED: VAStatus = 0x0000000A;
    pub const VA_STATUS_ERROR_MAX_NUM_EXCEEDED: VAStatus = 0x0000000B;
    pub const VA_STATUS_ERROR_UNSUPPORTED_PROFILE: VAStatus = 0x0000000C;
    pub const VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT: VAStatus = 0x0000000D;
    pub const VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT: VAStatus = 0x0000000E;
    pub const VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE: VAStatus = 0x0000000F;
    pub const VA_STATUS_ERROR_SURFACE_BUSY: VAStatus = 0x00000010;
    pub const VA_STATUS_ERROR_INVALID_PARAMETER: VAStatus = 0x00000012;
    pub const VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED: VAStatus = 0x00000013;
    pub const VA_STATUS_ERROR_DECODING_ERROR: VAStatus = 0x00000017;
    pub const VA_STATUS_ERROR_ENCODING_ERROR: VAStatus = 0x00000018;
    pub const VA_STATUS_ERROR_HW_BUSY: VAStatus = 0x00000022;
    pub const VA_STATUS_ERROR_TIMEDOUT: VAStatus = 0x00000026;

    /// Profile constants.
    pub const VA_PROFILE_NONE: VAProfile = -1;
    pub const VA_PROFILE_MPEG2_SIMPLE: VAProfile = 0;
    pub const VA_PROFILE_MPEG2_MAIN: VAProfile = 1;
    pub const VA_PROFILE_H264_BASELINE: VAProfile = 5;
    pub const VA_PROFILE_H264_MAIN: VAProfile = 6;
    pub const VA_PROFILE_H264_HIGH: VAProfile = 7;
    pub const VA_PROFILE_H264_HIGH10: VAProfile = 14;
    pub const VA_PROFILE_HEVC_MAIN: VAProfile = 19;
    pub const VA_PROFILE_HEVC_MAIN10: VAProfile = 20;
    pub const VA_PROFILE_VP9_PROFILE0: VAProfile = 22;
    pub const VA_PROFILE_VP9_PROFILE2: VAProfile = 24;
    pub const VA_PROFILE_AV1_PROFILE0: VAProfile = 35;
    pub const VA_PROFILE_AV1_PROFILE1: VAProfile = 36;

    /// Entrypoint constants.
    pub const VA_ENTRYPOINT_VLD: VAEntrypoint = 1;
    pub const VA_ENTRYPOINT_ENCSLICE: VAEntrypoint = 6;
    pub const VA_ENTRYPOINT_ENCPICTURE: VAEntrypoint = 7;
    pub const VA_ENTRYPOINT_VIDEOPROC: VAEntrypoint = 10;
    pub const VA_ENTRYPOINT_ENCSLICE_LP: VAEntrypoint = 11;

    /// RT Format constants.
    pub const VA_RT_FORMAT_YUV420: c_uint = 0x00000001;
    pub const VA_RT_FORMAT_YUV422: c_uint = 0x00000004;
    pub const VA_RT_FORMAT_YUV444: c_uint = 0x00000008;
    pub const VA_RT_FORMAT_YUV420_10: c_uint = 0x00000100;
    pub const VA_RT_FORMAT_RGB32: c_uint = 0x00020000;

    /// Rate control modes.
    pub const VA_RC_NONE: c_uint = 0x00000001;
    pub const VA_RC_CBR: c_uint = 0x00000002;
    pub const VA_RC_VBR: c_uint = 0x00000004;
    pub const VA_RC_CQP: c_uint = 0x00000010;
    pub const VA_RC_ICQ: c_uint = 0x00000040;
    pub const VA_RC_QVBR: c_uint = 0x00000400;

    /// Surface attribute types.
    pub const VA_SURFACE_ATTRIB_NONE: c_uint = 0;
    pub const VA_SURFACE_ATTRIB_PIXEL_FORMAT: c_uint = 1;
    pub const VA_SURFACE_ATTRIB_MIN_WIDTH: c_uint = 2;
    pub const VA_SURFACE_ATTRIB_MAX_WIDTH: c_uint = 3;
    pub const VA_SURFACE_ATTRIB_MIN_HEIGHT: c_uint = 4;
    pub const VA_SURFACE_ATTRIB_MAX_HEIGHT: c_uint = 5;
    pub const VA_SURFACE_ATTRIB_MEM_TYPE: c_uint = 6;
    pub const VA_SURFACE_ATTRIB_EXTERNAL_BUFFER_DESCRIPTOR: c_uint = 7;
    pub const VA_SURFACE_ATTRIB_DRM_FORMAT_MODIFIERS: c_uint = 8;

    // -------------------------------------------------------------------------
    // Structures
    // -------------------------------------------------------------------------

    /// VA Config Attribute.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAConfigAttrib {
        pub type_: c_int,
        pub value: c_uint,
    }

    /// VA Surface Attribute.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct VASurfaceAttrib {
        pub type_: c_uint,
        pub flags: c_uint,
        pub value: VAGenericValue,
    }

    impl Default for VASurfaceAttrib {
        fn default() -> Self {
            Self {
                type_: 0,
                flags: 0,
                value: VAGenericValue::default(),
            }
        }
    }

    /// VA Image structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct VAImage {
        pub image_id: VAImageID,
        pub format: VAImageFormat,
        pub buf: VABufferID,
        pub width: u16,
        pub height: u16,
        pub data_size: c_uint,
        pub num_planes: c_uint,
        pub pitches: [c_uint; 3],
        pub offsets: [c_uint; 3],
        pub num_palette_entries: c_int,
        pub entry_bytes: c_int,
        pub component_order: [i8; 4],
    }

    impl Default for VAImage {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// VA Image Format.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAImageFormat {
        pub fourcc: c_uint,
        pub byte_order: c_uint,
        pub bits_per_pixel: c_uint,
        pub depth: c_uint,
        pub red_mask: c_uint,
        pub green_mask: c_uint,
        pub blue_mask: c_uint,
        pub alpha_mask: c_uint,
    }

    /// Coded buffer segment.
    #[repr(C)]
    #[derive(Debug)]
    pub struct VACodedBufferSegment {
        pub size: c_uint,
        pub bit_offset: c_uint,
        pub status: c_uint,
        pub reserved: c_uint,
        pub buf: *mut c_void,
        pub next: *mut VACodedBufferSegment,
    }

    /// Encode Misc Parameter Buffer header.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAEncMiscParameterBuffer {
        pub type_: c_uint,
        pub data: [c_uint; 0], // Flexible array member
    }

    /// Encode Misc Parameter - Rate Control.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAEncMiscParameterRateControl {
        pub bits_per_second: c_uint,
        pub target_percentage: c_uint,
        pub window_size: c_uint,
        pub initial_qp: c_uint,
        pub min_qp: c_uint,
        pub basic_unit_size: c_uint,
        pub rc_flags: c_uint,
        pub icq_quality_factor: c_uint,
        pub max_qp: c_uint,
        pub quality_factor: c_uint,
        pub target_frame_size: c_uint,
    }

    /// Encode Misc Parameter - Frame Rate.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAEncMiscParameterFrameRate {
        pub framerate: c_uint,
        pub framerate_flags: c_uint,
    }

    /// Encode Misc Parameter - HRD (Hypothetical Reference Decoder).
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAEncMiscParameterHRD {
        pub initial_buffer_fullness: c_uint,
        pub buffer_size: c_uint,
    }

    /// H.264 Picture structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct VAPictureH264 {
        pub picture_id: VASurfaceID,
        pub frame_idx: c_uint,
        pub flags: c_uint,
        pub top_field_order_cnt: c_int,
        pub bottom_field_order_cnt: c_int,
    }

    /// H.264 picture flags.
    pub const VA_PICTURE_H264_INVALID: c_uint = 0x00000001;
    pub const VA_PICTURE_H264_TOP_FIELD: c_uint = 0x00000002;
    pub const VA_PICTURE_H264_BOTTOM_FIELD: c_uint = 0x00000004;
    pub const VA_PICTURE_H264_SHORT_TERM_REFERENCE: c_uint = 0x00000008;
    pub const VA_PICTURE_H264_LONG_TERM_REFERENCE: c_uint = 0x00000010;

    // -------------------------------------------------------------------------
    // FFI Function Declarations
    // -------------------------------------------------------------------------

    #[link(name = "va")]
    extern "C" {
        // --- Display management ---

        /// Get display for DRM file descriptor.
        pub fn vaGetDisplayDRM(fd: c_int) -> VADisplay;

        /// Initialize VA-API display.
        pub fn vaInitialize(
            dpy: VADisplay,
            major_version: *mut c_int,
            minor_version: *mut c_int,
        ) -> VAStatus;

        /// Terminate VA-API display.
        pub fn vaTerminate(dpy: VADisplay) -> VAStatus;

        /// Get error string for status code.
        pub fn vaErrorStr(error_status: VAStatus) -> *const c_char;

        /// Get vendor string.
        pub fn vaQueryVendorString(dpy: VADisplay) -> *const c_char;

        /// Get maximum number of profiles.
        pub fn vaMaxNumProfiles(dpy: VADisplay) -> c_int;

        /// Get maximum number of entrypoints.
        pub fn vaMaxNumEntrypoints(dpy: VADisplay) -> c_int;

        /// Get maximum number of config attributes.
        pub fn vaMaxNumConfigAttributes(dpy: VADisplay) -> c_int;

        /// Get maximum number of image formats.
        pub fn vaMaxNumImageFormats(dpy: VADisplay) -> c_int;

        // --- Profile and entrypoint queries ---

        /// Query supported profiles.
        pub fn vaQueryConfigProfiles(
            dpy: VADisplay,
            profile_list: *mut VAProfile,
            num_profiles: *mut c_int,
        ) -> VAStatus;

        /// Query entrypoints for a profile.
        pub fn vaQueryConfigEntrypoints(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint_list: *mut VAEntrypoint,
            num_entrypoints: *mut c_int,
        ) -> VAStatus;

        /// Get config attributes.
        pub fn vaGetConfigAttributes(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint: VAEntrypoint,
            attrib_list: *mut VAConfigAttrib,
            num_attribs: c_int,
        ) -> VAStatus;

        // --- Configuration ---

        /// Create a configuration.
        pub fn vaCreateConfig(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint: VAEntrypoint,
            attrib_list: *mut VAConfigAttrib,
            num_attribs: c_int,
            config_id: *mut VAConfigID,
        ) -> VAStatus;

        /// Destroy a configuration.
        pub fn vaDestroyConfig(dpy: VADisplay, config_id: VAConfigID) -> VAStatus;

        /// Query config attributes.
        pub fn vaQueryConfigAttributes(
            dpy: VADisplay,
            config_id: VAConfigID,
            profile: *mut VAProfile,
            entrypoint: *mut VAEntrypoint,
            attrib_list: *mut VAConfigAttrib,
            num_attribs: *mut c_int,
        ) -> VAStatus;

        // --- Surfaces ---

        /// Create surfaces.
        pub fn vaCreateSurfaces(
            dpy: VADisplay,
            format: c_uint,
            width: c_uint,
            height: c_uint,
            surfaces: *mut VASurfaceID,
            num_surfaces: c_uint,
            attrib_list: *mut VASurfaceAttrib,
            num_attribs: c_uint,
        ) -> VAStatus;

        /// Destroy surfaces.
        pub fn vaDestroySurfaces(
            dpy: VADisplay,
            surfaces: *mut VASurfaceID,
            num_surfaces: c_int,
        ) -> VAStatus;

        /// Query surface status.
        pub fn vaQuerySurfaceStatus(
            dpy: VADisplay,
            render_target: VASurfaceID,
            status: *mut c_uint,
        ) -> VAStatus;

        /// Sync surface (wait for completion).
        pub fn vaSyncSurface(dpy: VADisplay, render_target: VASurfaceID) -> VAStatus;

        // --- Context ---

        /// Create a context.
        pub fn vaCreateContext(
            dpy: VADisplay,
            config_id: VAConfigID,
            picture_width: c_int,
            picture_height: c_int,
            flag: c_int,
            render_targets: *mut VASurfaceID,
            num_render_targets: c_int,
            context: *mut VAContextID,
        ) -> VAStatus;

        /// Destroy a context.
        pub fn vaDestroyContext(dpy: VADisplay, context: VAContextID) -> VAStatus;

        // --- Buffers ---

        /// Create a buffer.
        pub fn vaCreateBuffer(
            dpy: VADisplay,
            context: VAContextID,
            type_: VABufferType,
            size: c_uint,
            num_elements: c_uint,
            data: *mut c_void,
            buf_id: *mut VABufferID,
        ) -> VAStatus;

        /// Destroy a buffer.
        pub fn vaDestroyBuffer(dpy: VADisplay, buffer_id: VABufferID) -> VAStatus;

        /// Map buffer for CPU access.
        pub fn vaMapBuffer(
            dpy: VADisplay,
            buf_id: VABufferID,
            pbuf: *mut *mut c_void,
        ) -> VAStatus;

        /// Unmap buffer.
        pub fn vaUnmapBuffer(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;

        // --- Rendering ---

        /// Begin picture encoding/decoding.
        pub fn vaBeginPicture(
            dpy: VADisplay,
            context: VAContextID,
            render_target: VASurfaceID,
        ) -> VAStatus;

        /// Render picture (submit buffers).
        pub fn vaRenderPicture(
            dpy: VADisplay,
            context: VAContextID,
            buffers: *mut VABufferID,
            num_buffers: c_int,
        ) -> VAStatus;

        /// End picture encoding/decoding.
        pub fn vaEndPicture(dpy: VADisplay, context: VAContextID) -> VAStatus;

        // --- Images ---

        /// Query image formats.
        pub fn vaQueryImageFormats(
            dpy: VADisplay,
            format_list: *mut VAImageFormat,
            num_formats: *mut c_int,
        ) -> VAStatus;

        /// Create image.
        pub fn vaCreateImage(
            dpy: VADisplay,
            format: *mut VAImageFormat,
            width: c_int,
            height: c_int,
            image: *mut VAImage,
        ) -> VAStatus;

        /// Destroy image.
        pub fn vaDestroyImage(dpy: VADisplay, image: VAImageID) -> VAStatus;

        /// Get image from surface.
        pub fn vaGetImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            x: c_int,
            y: c_int,
            width: c_uint,
            height: c_uint,
            image: VAImageID,
        ) -> VAStatus;

        /// Put image to surface.
        pub fn vaPutImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            image: VAImageID,
            src_x: c_int,
            src_y: c_int,
            src_width: c_uint,
            src_height: c_uint,
            dest_x: c_int,
            dest_y: c_int,
            dest_width: c_uint,
            dest_height: c_uint,
        ) -> VAStatus;

        /// Derive image from surface (zero-copy).
        pub fn vaDeriveImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            image: *mut VAImage,
        ) -> VAStatus;

        // --- Surface attributes ---

        /// Query surface attributes.
        pub fn vaQuerySurfaceAttributes(
            dpy: VADisplay,
            config: VAConfigID,
            attrib_list: *mut VASurfaceAttrib,
            num_attribs: *mut c_uint,
        ) -> VAStatus;

        // --- Coded buffer ---

        /// Acquire coded buffer segment.
        pub fn vaAcquireBufferHandle(
            dpy: VADisplay,
            buf_id: VABufferID,
            buf_info: *mut c_void, // VABufferInfo
        ) -> VAStatus;

        /// Release coded buffer segment.
        pub fn vaReleaseBufferHandle(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;
    }

    // DRM display link
    #[link(name = "va-drm")]
    extern "C" {
        /// Get display for DRM file descriptor (from va-drm).
        #[link_name = "vaGetDisplayDRM"]
        pub fn vaGetDisplayDRM_drm(fd: c_int) -> VADisplay;
    }

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------

    /// Check if status indicates success.
    pub fn va_status_success(status: VAStatus) -> bool {
        status == VA_STATUS_SUCCESS
    }

    /// Get error description string.
    pub unsafe fn va_error_string(status: VAStatus) -> String {
        let ptr = vaErrorStr(status);
        if ptr.is_null() {
            format!("Unknown error: {}", status)
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// RAII guard for VA Display.
    pub struct VADisplayGuard {
        display: VADisplay,
        initialized: bool,
    }

    impl VADisplayGuard {
        /// Create from raw display handle (after vaInitialize).
        pub unsafe fn new(display: VADisplay) -> Option<Self> {
            if display.is_null() {
                None
            } else {
                Some(Self {
                    display,
                    initialized: true,
                })
            }
        }

        /// Get raw display handle.
        pub fn as_ptr(&self) -> VADisplay {
            self.display
        }
    }

    impl Drop for VADisplayGuard {
        fn drop(&mut self) {
            if self.initialized && !self.display.is_null() {
                unsafe { vaTerminate(self.display) };
            }
        }
    }

    /// RAII guard for VA Surface.
    pub struct VASurfaceGuard {
        display: VADisplay,
        surface: VASurfaceID,
    }

    impl VASurfaceGuard {
        /// Create guard for a surface.
        pub unsafe fn new(display: VADisplay, surface: VASurfaceID) -> Self {
            Self { display, surface }
        }

        /// Get surface ID.
        pub fn id(&self) -> VASurfaceID {
            self.surface
        }

        /// Sync (wait for completion).
        pub unsafe fn sync(&self) -> VAStatus {
            vaSyncSurface(self.display, self.surface)
        }
    }

    impl Drop for VASurfaceGuard {
        fn drop(&mut self) {
            if self.surface != VA_INVALID_SURFACE {
                unsafe {
                    let mut surfaces = [self.surface];
                    vaDestroySurfaces(self.display, surfaces.as_mut_ptr(), 1);
                }
            }
        }
    }

    /// RAII guard for VA Buffer.
    pub struct VABufferGuard {
        display: VADisplay,
        buffer: VABufferID,
    }

    impl VABufferGuard {
        /// Create guard for a buffer.
        pub unsafe fn new(display: VADisplay, buffer: VABufferID) -> Self {
            Self { display, buffer }
        }

        /// Get buffer ID.
        pub fn id(&self) -> VABufferID {
            self.buffer
        }

        /// Map buffer for CPU access.
        pub unsafe fn map(&self) -> Result<*mut c_void, VAStatus> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let status = vaMapBuffer(self.display, self.buffer, &mut ptr);
            if va_status_success(status) {
                Ok(ptr)
            } else {
                Err(status)
            }
        }

        /// Unmap buffer.
        pub unsafe fn unmap(&self) -> VAStatus {
            vaUnmapBuffer(self.display, self.buffer)
        }
    }

    impl Drop for VABufferGuard {
        fn drop(&mut self) {
            if self.buffer != VA_INVALID_ID {
                unsafe { vaDestroyBuffer(self.display, self.buffer) };
            }
        }
    }
}

// =============================================================================
// Feature flag for FFI availability
// =============================================================================

/// Indicates whether real VA-API FFI bindings are available.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
pub const FFI_AVAILABLE: bool = true;

#[cfg(not(all(target_os = "linux", feature = "vaapi")))]
pub const FFI_AVAILABLE: bool = false;

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::{HwCapabilities, HwCodec};
use std::collections::VecDeque;
use std::sync::Mutex;

// ============================================================================
// FFI Type Definitions (for future libva-sys binding)
// ============================================================================

/// VA Display handle - represents a connection to the VA-API.
/// In libva, this is a void* (opaque pointer).
pub type VADisplay = *mut std::ffi::c_void;

/// VA Config ID - identifies an encoding/decoding configuration.
pub type VAConfigID = u32;

/// VA Context ID - identifies an encoding/decoding context.
pub type VAContextID = u32;

/// VA Surface ID - identifies a video surface.
pub type VASurfaceID = u32;

/// VA Buffer ID - identifies a data buffer.
pub type VABufferID = u32;

/// VA Image ID - identifies an image for CPU access.
pub type VAImageID = u32;

/// VA Generic Value for attribute values.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub union VAGenericValue {
    /// Integer value.
    pub i: i32,
    /// Float value.
    pub f: f32,
    /// Pointer value.
    pub p: *mut std::ffi::c_void,
}

impl Default for VAGenericValue {
    fn default() -> Self {
        VAGenericValue { i: 0 }
    }
}

/// Invalid surface ID constant.
pub const VA_INVALID_SURFACE: VASurfaceID = 0xFFFFFFFF;

/// Invalid buffer ID constant.
pub const VA_INVALID_ID: u32 = 0xFFFFFFFF;

// ============================================================================
// VA Status and Error Codes
// ============================================================================

/// VA-API status code.
pub type VAStatus = i32;

/// VA-API status codes matching libva definitions.
pub mod va_status {
    use super::VAStatus;

    /// Operation successful.
    pub const VA_STATUS_SUCCESS: VAStatus = 0x00000000;
    /// Operation failed (generic error).
    pub const VA_STATUS_ERROR_OPERATION_FAILED: VAStatus = 0x00000001;
    /// Memory allocation failed.
    pub const VA_STATUS_ERROR_ALLOCATION_FAILED: VAStatus = 0x00000002;
    /// Invalid display.
    pub const VA_STATUS_ERROR_INVALID_DISPLAY: VAStatus = 0x00000003;
    /// Invalid configuration.
    pub const VA_STATUS_ERROR_INVALID_CONFIG: VAStatus = 0x00000004;
    /// Invalid context.
    pub const VA_STATUS_ERROR_INVALID_CONTEXT: VAStatus = 0x00000005;
    /// Invalid surface.
    pub const VA_STATUS_ERROR_INVALID_SURFACE: VAStatus = 0x00000006;
    /// Invalid buffer.
    pub const VA_STATUS_ERROR_INVALID_BUFFER: VAStatus = 0x00000007;
    /// Invalid image.
    pub const VA_STATUS_ERROR_INVALID_IMAGE: VAStatus = 0x00000008;
    /// Invalid subpicture.
    pub const VA_STATUS_ERROR_INVALID_SUBPICTURE: VAStatus = 0x00000009;
    /// Attribute not supported.
    pub const VA_STATUS_ERROR_ATTR_NOT_SUPPORTED: VAStatus = 0x0000000A;
    /// Maximum number of profiles reached.
    pub const VA_STATUS_ERROR_MAX_NUM_EXCEEDED: VAStatus = 0x0000000B;
    /// Profile not supported.
    pub const VA_STATUS_ERROR_UNSUPPORTED_PROFILE: VAStatus = 0x0000000C;
    /// Entrypoint not supported.
    pub const VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT: VAStatus = 0x0000000D;
    /// Memory type not supported.
    pub const VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT: VAStatus = 0x0000000E;
    /// Buffer type not supported.
    pub const VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE: VAStatus = 0x0000000F;
    /// Surface is busy.
    pub const VA_STATUS_ERROR_SURFACE_BUSY: VAStatus = 0x00000010;
    /// Flag not supported.
    pub const VA_STATUS_ERROR_FLAG_NOT_SUPPORTED: VAStatus = 0x00000011;
    /// Invalid parameter.
    pub const VA_STATUS_ERROR_INVALID_PARAMETER: VAStatus = 0x00000012;
    /// Resolution not supported.
    pub const VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED: VAStatus = 0x00000013;
    /// Unimplemented.
    pub const VA_STATUS_ERROR_UNIMPLEMENTED: VAStatus = 0x00000014;
    /// Surface in displaying.
    pub const VA_STATUS_ERROR_SURFACE_IN_DISPLAYING: VAStatus = 0x00000015;
    /// Invalid image format.
    pub const VA_STATUS_ERROR_INVALID_IMAGE_FORMAT: VAStatus = 0x00000016;
    /// Decoding error.
    pub const VA_STATUS_ERROR_DECODING_ERROR: VAStatus = 0x00000017;
    /// Encoding error.
    pub const VA_STATUS_ERROR_ENCODING_ERROR: VAStatus = 0x00000018;
    /// Invalid value.
    pub const VA_STATUS_ERROR_INVALID_VALUE: VAStatus = 0x00000019;
    /// Unsupported filter.
    pub const VA_STATUS_ERROR_UNSUPPORTED_FILTER: VAStatus = 0x00000020;
    /// Invalid filter chain.
    pub const VA_STATUS_ERROR_INVALID_FILTER_CHAIN: VAStatus = 0x00000021;
    /// Hardware busy.
    pub const VA_STATUS_ERROR_HW_BUSY: VAStatus = 0x00000022;
    /// Not enough buffer.
    pub const VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE: VAStatus = 0x00000024;
    /// Not enough data.
    pub const VA_STATUS_ERROR_NOT_ENOUGH_BUFFER: VAStatus = 0x00000025;
    /// Timeout.
    pub const VA_STATUS_ERROR_TIMEDOUT: VAStatus = 0x00000026;
    /// Unknown error.
    pub const VA_STATUS_ERROR_UNKNOWN: VAStatus = -1;
}

/// Convert VA status to a descriptive string.
pub fn va_status_string(status: VAStatus) -> &'static str {
    use va_status::*;
    match status {
        VA_STATUS_SUCCESS => "success",
        VA_STATUS_ERROR_OPERATION_FAILED => "operation failed",
        VA_STATUS_ERROR_ALLOCATION_FAILED => "allocation failed",
        VA_STATUS_ERROR_INVALID_DISPLAY => "invalid display",
        VA_STATUS_ERROR_INVALID_CONFIG => "invalid config",
        VA_STATUS_ERROR_INVALID_CONTEXT => "invalid context",
        VA_STATUS_ERROR_INVALID_SURFACE => "invalid surface",
        VA_STATUS_ERROR_INVALID_BUFFER => "invalid buffer",
        VA_STATUS_ERROR_INVALID_IMAGE => "invalid image",
        VA_STATUS_ERROR_ATTR_NOT_SUPPORTED => "attribute not supported",
        VA_STATUS_ERROR_MAX_NUM_EXCEEDED => "max number exceeded",
        VA_STATUS_ERROR_UNSUPPORTED_PROFILE => "unsupported profile",
        VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT => "unsupported entrypoint",
        VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT => "unsupported RT format",
        VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE => "unsupported buffer type",
        VA_STATUS_ERROR_SURFACE_BUSY => "surface busy",
        VA_STATUS_ERROR_INVALID_PARAMETER => "invalid parameter",
        VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED => "resolution not supported",
        VA_STATUS_ERROR_DECODING_ERROR => "decoding error",
        VA_STATUS_ERROR_ENCODING_ERROR => "encoding error",
        VA_STATUS_ERROR_HW_BUSY => "hardware busy",
        VA_STATUS_ERROR_TIMEDOUT => "timeout",
        _ => "unknown error",
    }
}

// ============================================================================
// VA Profile Enum (matching libva)
// ============================================================================

/// VA-API profile values matching libva definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VAProfile {
    /// No profile (used for video processing).
    None = -1,
    /// MPEG-2 Simple Profile.
    MPEG2Simple = 0,
    /// MPEG-2 Main Profile.
    MPEG2Main = 1,
    /// MPEG-4 Simple Profile.
    MPEG4Simple = 2,
    /// MPEG-4 Advanced Simple Profile.
    MPEG4AdvancedSimple = 3,
    /// MPEG-4 Main Profile.
    MPEG4Main = 4,
    /// H.264 Constrained Baseline.
    H264ConstrainedBaseline = 5,
    /// H.264 Baseline (deprecated, use ConstrainedBaseline).
    H264Baseline = 5,
    /// H.264 Main Profile.
    H264Main = 6,
    /// H.264 High Profile.
    H264High = 7,
    /// VC-1 Simple Profile.
    VC1Simple = 8,
    /// VC-1 Main Profile.
    VC1Main = 9,
    /// VC-1 Advanced Profile.
    VC1Advanced = 10,
    /// H.263 Baseline.
    H263Baseline = 11,
    /// JPEG Baseline.
    JPEGBaseline = 12,
    /// H.264 Extended Profile.
    H264Extended = 13,
    /// H.264 High 10 Profile (10-bit).
    H264High10 = 14,
    /// H.264 High 4:2:2 Profile.
    H264High422 = 15,
    /// H.264 High 4:4:4 Profile.
    H264High444 = 16,
    /// H.264 Stereo High Profile.
    H264StereoHigh = 17,
    /// H.264 Multiview High Profile.
    H264MultiviewHigh = 18,
    /// HEVC Main Profile.
    HEVCMain = 19,
    /// HEVC Main 10 Profile.
    HEVCMain10 = 20,
    /// VP8 Version 0-3.
    VP8Version0_3 = 21,
    /// VP9 Profile 0 (8-bit 4:2:0).
    VP9Profile0 = 22,
    /// VP9 Profile 1 (8-bit 4:2:2/4:4:4).
    VP9Profile1 = 23,
    /// VP9 Profile 2 (10/12-bit 4:2:0).
    VP9Profile2 = 24,
    /// VP9 Profile 3 (10/12-bit 4:2:2/4:4:4).
    VP9Profile3 = 25,
    /// HEVC Main 12 Profile.
    HEVCMain12 = 26,
    /// HEVC Main 4:2:2 10 Profile.
    HEVCMain422_10 = 27,
    /// HEVC Main 4:2:2 12 Profile.
    HEVCMain422_12 = 28,
    /// HEVC Main 4:4:4 Profile.
    HEVCMain444 = 29,
    /// HEVC Main 4:4:4 10 Profile.
    HEVCMain444_10 = 30,
    /// HEVC Main 4:4:4 12 Profile.
    HEVCMain444_12 = 31,
    /// HEVC SCC (Screen Content Coding) Main.
    HEVCSccMain = 32,
    /// HEVC SCC Main 10.
    HEVCSccMain10 = 33,
    /// HEVC SCC Main 4:4:4.
    HEVCSccMain444 = 34,
    /// AV1 Profile 0 (8-bit and 10-bit 4:2:0).
    AV1Profile0 = 35,
    /// AV1 Profile 1 (8-bit and 10-bit 4:4:4).
    AV1Profile1 = 36,
    /// HEVC SCC Main 4:4:4 10.
    HEVCSccMain444_10 = 37,
    /// Protected content.
    Protected = 38,
    /// H.264 High 10 Intra.
    H264High10Intra = 39,
    /// H.264 High 4:2:2 Intra.
    H264High422Intra = 40,
    /// H.264 High 4:4:4 Intra.
    H264High444Intra = 41,
    /// HEVC Main Intra.
    HEVCMainIntra = 42,
    /// VP9 Profile 0 (10-bit).
    VP9Profile0_10 = 43,
}

impl VAProfile {
    /// Get the profile IDC value for H.264 profiles.
    pub fn h264_idc(&self) -> Option<u8> {
        match self {
            VAProfile::H264ConstrainedBaseline | VAProfile::H264Baseline => Some(66),
            VAProfile::H264Main => Some(77),
            VAProfile::H264Extended => Some(88),
            VAProfile::H264High => Some(100),
            VAProfile::H264High10 | VAProfile::H264High10Intra => Some(110),
            VAProfile::H264High422 | VAProfile::H264High422Intra => Some(122),
            VAProfile::H264High444 | VAProfile::H264High444Intra => Some(244),
            _ => None,
        }
    }

    /// Check if this is an H.264 profile.
    pub fn is_h264(&self) -> bool {
        matches!(
            self,
            VAProfile::H264ConstrainedBaseline
                | VAProfile::H264Baseline
                | VAProfile::H264Main
                | VAProfile::H264Extended
                | VAProfile::H264High
                | VAProfile::H264High10
                | VAProfile::H264High422
                | VAProfile::H264High444
                | VAProfile::H264StereoHigh
                | VAProfile::H264MultiviewHigh
                | VAProfile::H264High10Intra
                | VAProfile::H264High422Intra
                | VAProfile::H264High444Intra
        )
    }

    /// Check if this is an HEVC profile.
    pub fn is_hevc(&self) -> bool {
        matches!(
            self,
            VAProfile::HEVCMain
                | VAProfile::HEVCMain10
                | VAProfile::HEVCMain12
                | VAProfile::HEVCMain422_10
                | VAProfile::HEVCMain422_12
                | VAProfile::HEVCMain444
                | VAProfile::HEVCMain444_10
                | VAProfile::HEVCMain444_12
                | VAProfile::HEVCSccMain
                | VAProfile::HEVCSccMain10
                | VAProfile::HEVCSccMain444
                | VAProfile::HEVCSccMain444_10
                | VAProfile::HEVCMainIntra
        )
    }

    /// Check if this is an AV1 profile.
    pub fn is_av1(&self) -> bool {
        matches!(self, VAProfile::AV1Profile0 | VAProfile::AV1Profile1)
    }

    /// Check if this is a VP9 profile.
    pub fn is_vp9(&self) -> bool {
        matches!(
            self,
            VAProfile::VP9Profile0
                | VAProfile::VP9Profile1
                | VAProfile::VP9Profile2
                | VAProfile::VP9Profile3
                | VAProfile::VP9Profile0_10
        )
    }
}

// ============================================================================
// VA Entrypoint Enum
// ============================================================================

/// VA-API entrypoint values matching libva definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VAEntrypoint {
    /// VLD (Variable Length Decode) - decoding.
    VLD = 1,
    /// IZZ (Inverse Zigzag) - deprecated.
    IZZ = 2,
    /// IDCT - deprecated.
    IDCT = 3,
    /// MoComp (Motion Compensation) - deprecated.
    MoComp = 4,
    /// Deblocking - deprecated.
    Deblocking = 5,
    /// Encoding with slice-level control.
    EncSlice = 6,
    /// Encoding with picture-level control.
    EncPicture = 7,
    /// Video processing (scaling, color conversion, etc.).
    VideoProc = 10,
    /// Low-power encoding with slice-level control (Intel-specific).
    EncSliceLP = 11,
    /// Protected content.
    ProtectedTEEComm = 12,
    /// Protected content playback.
    ProtectedContent = 13,
    /// Statistics gathering.
    Stats = 14,
}

impl VAEntrypoint {
    /// Check if this is an encoding entrypoint.
    pub fn is_encode(&self) -> bool {
        matches!(
            self,
            VAEntrypoint::EncSlice | VAEntrypoint::EncPicture | VAEntrypoint::EncSliceLP
        )
    }

    /// Check if this is a decoding entrypoint.
    pub fn is_decode(&self) -> bool {
        matches!(self, VAEntrypoint::VLD)
    }

    /// Check if this is a low-power encoding entrypoint.
    pub fn is_low_power(&self) -> bool {
        matches!(self, VAEntrypoint::EncSliceLP)
    }

    /// Check if this is a video processing entrypoint.
    pub fn is_video_proc(&self) -> bool {
        matches!(self, VAEntrypoint::VideoProc)
    }
}

// ============================================================================
// VA Config Attributes
// ============================================================================

/// VA Configuration attribute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VAConfigAttribType {
    /// RT format (YUV format for surfaces).
    RTFormat = 0,
    /// Spatial resolution.
    SpatialResidual = 1,
    /// Spatial statistic.
    SpatialStatistic = 2,
    /// Intra residual.
    IntraResidual = 3,
    /// Encryption.
    Encryption = 4,
    /// Rate control modes.
    RateControl = 5,
    /// Decoding slice mode.
    DecSliceMode = 6,
    /// Processing slice mode.
    DecProcessing = 7,
    /// Encode packed headers.
    EncPackedHeaders = 10,
    /// Encode interlaced mode.
    EncInterlaced = 11,
    /// Encode maximum reference frames.
    EncMaxRefFrames = 13,
    /// Encode maximum slice size.
    EncMaxSlices = 14,
    /// Encode slice structure.
    EncSliceStructure = 15,
    /// Encode macroblock info.
    EncMacroblockInfo = 16,
    /// JPEG encoding info.
    EncJPEG = 20,
    /// Encode quality range.
    EncQualityRange = 21,
    /// Encode quantization.
    EncQuantization = 22,
    /// Encode skip frame.
    EncSkipFrame = 24,
    /// ROI (Region of Interest) encoding.
    EncROI = 25,
    /// Encode rate control extension.
    EncRateControlExt = 26,
    /// Process supported filters.
    ProcessingRate = 27,
    /// Encode dirty rect.
    EncDirtyRect = 28,
    /// Encode parallel rate control.
    EncParallelRateControl = 29,
    /// Encode dynamic scaling.
    EncDynamicScaling = 30,
    /// Frame size tolerance.
    FrameSizeToleranceSupport = 31,
    /// FEI function type.
    FEIFunctionType = 32,
    /// FEI motion vector predictor.
    FEIMVPredictors = 33,
    /// Statistics.
    Stats = 34,
    /// Encode tile support.
    EncTileSupport = 35,
    /// Custom round control.
    CustomRoundingControl = 36,
    /// QP block size.
    QPBlockSize = 37,
    /// Max frame size.
    MaxFrameSize = 38,
    /// Prediction direction.
    PredictionDirection = 39,
    /// Multi-pass encoding.
    MultiPassEncoding = 40,
    /// Context priority.
    ContextPriority = 41,
    /// Encode per-block control.
    EncPerBlockControl = 42,
    /// Encode slice level control for HEVC.
    EncHEVCSliceLevelControl = 43,
    /// Encode AV1 features.
    EncAV1 = 44,
    /// Encode AV1 extended features.
    EncAV1Ext1 = 45,
    /// Encode AV1 extended features 2.
    EncAV1Ext2 = 46,
}

/// VA Configuration attribute structure.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAConfigAttrib {
    /// Attribute type.
    pub type_: VAConfigAttribType,
    /// Attribute value (query result or desired value).
    pub value: u32,
}

impl Default for VAConfigAttribType {
    fn default() -> Self {
        VAConfigAttribType::RTFormat
    }
}

// ============================================================================
// RT Format Constants
// ============================================================================

/// RT (Render Target) format constants for surface creation.
pub mod va_rt_format {
    /// YUV 4:2:0 format.
    pub const VA_RT_FORMAT_YUV420: u32 = 0x00000001;
    /// YUV 4:1:1 format.
    pub const VA_RT_FORMAT_YUV411: u32 = 0x00000002;
    /// YUV 4:2:2 format.
    pub const VA_RT_FORMAT_YUV422: u32 = 0x00000004;
    /// YUV 4:4:4 format.
    pub const VA_RT_FORMAT_YUV444: u32 = 0x00000008;
    /// Y400 (grayscale) format.
    pub const VA_RT_FORMAT_Y800: u32 = 0x00000010;
    /// YUV 4:2:0 10-bit format.
    pub const VA_RT_FORMAT_YUV420_10: u32 = 0x00000100;
    /// YUV 4:2:2 10-bit format.
    pub const VA_RT_FORMAT_YUV422_10: u32 = 0x00000200;
    /// YUV 4:4:4 10-bit format.
    pub const VA_RT_FORMAT_YUV444_10: u32 = 0x00000400;
    /// Y410 format.
    pub const VA_RT_FORMAT_Y410: u32 = 0x00000800;
    /// YUV 4:2:0 12-bit format.
    pub const VA_RT_FORMAT_YUV420_12: u32 = 0x00001000;
    /// YUV 4:2:2 12-bit format.
    pub const VA_RT_FORMAT_YUV422_12: u32 = 0x00002000;
    /// YUV 4:4:4 12-bit format.
    pub const VA_RT_FORMAT_YUV444_12: u32 = 0x00004000;
    /// RGB 16 format.
    pub const VA_RT_FORMAT_RGB16: u32 = 0x00010000;
    /// RGB 32 format.
    pub const VA_RT_FORMAT_RGB32: u32 = 0x00020000;
    /// RGBP format.
    pub const VA_RT_FORMAT_RGBP: u32 = 0x00100000;
    /// RGB 32 10-bit format.
    pub const VA_RT_FORMAT_RGB32_10: u32 = 0x00200000;
    /// Protected content format.
    pub const VA_RT_FORMAT_PROTECTED: u32 = 0x80000000;

    /// Convert RT format to a descriptive string.
    pub fn format_string(format: u32) -> &'static str {
        match format {
            VA_RT_FORMAT_YUV420 => "YUV420",
            VA_RT_FORMAT_YUV411 => "YUV411",
            VA_RT_FORMAT_YUV422 => "YUV422",
            VA_RT_FORMAT_YUV444 => "YUV444",
            VA_RT_FORMAT_Y800 => "Y800",
            VA_RT_FORMAT_YUV420_10 => "YUV420_10",
            VA_RT_FORMAT_YUV422_10 => "YUV422_10",
            VA_RT_FORMAT_YUV444_10 => "YUV444_10",
            VA_RT_FORMAT_YUV420_12 => "YUV420_12",
            VA_RT_FORMAT_YUV422_12 => "YUV422_12",
            VA_RT_FORMAT_YUV444_12 => "YUV444_12",
            VA_RT_FORMAT_RGB16 => "RGB16",
            VA_RT_FORMAT_RGB32 => "RGB32",
            VA_RT_FORMAT_RGBP => "RGBP",
            VA_RT_FORMAT_RGB32_10 => "RGB32_10",
            _ => "Unknown",
        }
    }
}

// ============================================================================
// Rate Control Mode Constants
// ============================================================================

/// Rate control mode flags.
pub mod va_rc_mode {
    /// No rate control (constant QP).
    pub const VA_RC_NONE: u32 = 0x00000001;
    /// Constant Bitrate.
    pub const VA_RC_CBR: u32 = 0x00000002;
    /// Variable Bitrate.
    pub const VA_RC_VBR: u32 = 0x00000004;
    /// Variable Bitrate Constrained.
    pub const VA_RC_VCM: u32 = 0x00000008;
    /// Constant QP.
    pub const VA_RC_CQP: u32 = 0x00000010;
    /// Variable Bitrate with Peak.
    pub const VA_RC_VBR_CONSTRAINED: u32 = 0x00000020;
    /// Intelligent Constant Quality (Intel-specific).
    pub const VA_RC_ICQ: u32 = 0x00000040;
    /// Macroblock-level bitrate control.
    pub const VA_RC_MB: u32 = 0x00000080;
    /// Constant Frame Size.
    pub const VA_RC_CFS: u32 = 0x00000100;
    /// Parallel BRC.
    pub const VA_RC_PARALLEL: u32 = 0x00000200;
    /// Quality-defined VBR (QVBR) - Intel-specific.
    pub const VA_RC_QVBR: u32 = 0x00000400;
    /// Average VBR (AVBR) - Intel-specific.
    pub const VA_RC_AVBR: u32 = 0x00000800;
    /// Target Constrained Quality (TCQ).
    pub const VA_RC_TCQ: u32 = 0x00001000;
}

// ============================================================================
// Buffer Types
// ============================================================================

/// VA Buffer types matching libva definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VABufferType {
    /// Picture parameter buffer.
    PictureParameterBufferType = 0,
    /// IQ matrix buffer.
    IQMatrixBufferType = 1,
    /// Bitplane buffer.
    BitPlaneBufferType = 2,
    /// Slice group map buffer.
    SliceGroupMapBufferType = 3,
    /// Slice parameter buffer.
    SliceParameterBufferType = 4,
    /// Slice data buffer.
    SliceDataBufferType = 5,
    /// Macroblock parameter buffer.
    MacroblockParameterBufferType = 6,
    /// Residual data buffer.
    ResidualDataBufferType = 7,
    /// Deblocking parameter buffer.
    DeblockingParameterBufferType = 8,
    /// Image data buffer.
    ImageBufferType = 9,
    /// Protected slice data buffer.
    ProtectedSliceDataBufferType = 10,
    /// QP matrix buffer.
    QMatrixBufferType = 11,
    /// Huffman table buffer.
    HuffmanTableBufferType = 12,
    /// Probability data buffer.
    ProbabilityDataBufferType = 13,

    // Encoding buffer types
    /// Encode coded buffer.
    EncCodedBufferType = 21,
    /// Encode sequence parameter buffer.
    EncSequenceParameterBufferType = 22,
    /// Encode picture parameter buffer.
    EncPictureParameterBufferType = 23,
    /// Encode slice parameter buffer.
    EncSliceParameterBufferType = 24,
    /// Encode packed header parameter buffer.
    EncPackedHeaderParameterBufferType = 25,
    /// Encode packed header data buffer.
    EncPackedHeaderDataBufferType = 26,
    /// Encode misc parameter buffer.
    EncMiscParameterBufferType = 27,
    /// Encode macroblock parameter buffer.
    EncMacroblockParameterBufferType = 28,
    /// Encode macroblock map buffer.
    EncMacroblockMapBufferType = 29,
    /// Encode QP buffer.
    EncQPBufferType = 30,

    // Video processing buffer types
    /// VPP pipeline parameter buffer.
    ProcPipelineParameterBufferType = 41,
    /// VPP filter parameter buffer.
    ProcFilterParameterBufferType = 42,

    // FEI buffer types
    /// FEI frame control buffer.
    EncFEIFrameControlBufferType = 51,
    /// FEI MV predictor buffer.
    EncFEIMVPredictorBufferType = 52,
    /// FEI MB code buffer.
    EncFEIMBCodeBufferType = 53,
    /// FEI distortion buffer.
    EncFEIDistortionBufferType = 54,
    /// FEI MB control buffer.
    EncFEIMBControlBufferType = 55,
    /// FEI MV data buffer.
    EncFEIMVBufferType = 56,

    // Statistics buffer types
    /// Statistics parameter buffer.
    StatsStatisticsParameterBufferType = 61,
    /// Statistics buffer.
    StatsStatisticsBufferType = 62,
    /// Statistics bottom field buffer.
    StatsStatisticsBottomFieldBufferType = 63,
    /// Statistics MV buffer.
    StatsMVBufferType = 64,
    /// Statistics MV predictor buffer.
    StatsMVPredictorBufferType = 65,

    // Decode stream out buffer
    /// Decode stream out buffer.
    DecodeStreamoutBufferType = 71,

    // Protected content buffer
    /// Subsample mapping buffer.
    SubsampleMappingBufferType = 72,
    /// Encryption parameters buffer.
    EncryptionParameterBufferType = 73,
}

// ============================================================================
// Packed Header Types
// ============================================================================

/// Packed header type flags.
pub mod va_enc_packed_header {
    /// No packed header.
    pub const VA_ENC_PACKED_HEADER_NONE: u32 = 0x00000000;
    /// Sequence header (SPS).
    pub const VA_ENC_PACKED_HEADER_SEQUENCE: u32 = 0x00000001;
    /// Picture header (PPS).
    pub const VA_ENC_PACKED_HEADER_PICTURE: u32 = 0x00000002;
    /// Slice header.
    pub const VA_ENC_PACKED_HEADER_SLICE: u32 = 0x00000004;
    /// Misc data (SEI, etc.).
    pub const VA_ENC_PACKED_HEADER_MISC: u32 = 0x00000008;
    /// Raw data (bypass encoder).
    pub const VA_ENC_PACKED_HEADER_RAW_DATA: u32 = 0x00000010;
}

/// Encode packed header parameter buffer structure.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncPackedHeaderParameterBuffer {
    /// Type of packed header (SPS, PPS, slice, etc.).
    pub type_: u32,
    /// Bit length of the header (not byte length!).
    pub bit_length: u32,
    /// Whether to insert emulation prevention bytes.
    pub has_emulation_bytes: u8,
}

// ============================================================================
// H.264 Encoding Parameter Structures
// ============================================================================

/// H.264 sequence parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncSequenceParameterBufferH264 {
    /// SPS ID.
    pub seq_parameter_set_id: u8,
    /// Level IDC.
    pub level_idc: u8,
    /// Intra period (GOP size).
    pub intra_period: u32,
    /// Intra IDR period.
    pub intra_idr_period: u32,
    /// IP period (I/P frame interval).
    pub ip_period: u32,
    /// Target bitrate in bits per second.
    pub bits_per_second: u32,
    /// Maximum number of reference frames.
    pub max_num_ref_frames: u32,
    /// Picture width in macroblocks (width / 16).
    pub picture_width_in_mbs: u16,
    /// Picture height in map units (height / 16 for progressive).
    pub picture_height_in_map_units: u16,

    // Sequence fields packed into bitfield
    /// chroma_format_idc (0=mono, 1=4:2:0, 2=4:2:2, 3=4:4:4).
    pub seq_fields_chroma_format_idc: u8,
    /// frame_mbs_only_flag.
    pub seq_fields_frame_mbs_only_flag: u8,
    /// mb_adaptive_frame_field_flag.
    pub seq_fields_mb_adaptive_frame_field_flag: u8,
    /// seq_scaling_matrix_present_flag.
    pub seq_fields_seq_scaling_matrix_present_flag: u8,
    /// direct_8x8_inference_flag.
    pub seq_fields_direct_8x8_inference_flag: u8,
    /// log2_max_frame_num_minus4.
    pub seq_fields_log2_max_frame_num_minus4: u8,
    /// pic_order_cnt_type.
    pub seq_fields_pic_order_cnt_type: u8,
    /// log2_max_pic_order_cnt_lsb_minus4.
    pub seq_fields_log2_max_pic_order_cnt_lsb_minus4: u8,
    /// delta_pic_order_always_zero_flag.
    pub seq_fields_delta_pic_order_always_zero_flag: u8,

    /// Bit depth for luma (usually 8).
    pub bit_depth_luma_minus8: u8,
    /// Bit depth for chroma (usually 8).
    pub bit_depth_chroma_minus8: u8,

    /// Number of units in tick.
    pub num_units_in_tick: u32,
    /// Time scale.
    pub time_scale: u32,

    /// Offset for non-reference pictures in POC.
    pub offset_for_non_ref_pic: i32,
    /// Offset for top to bottom field in POC.
    pub offset_for_top_to_bottom_field: i32,

    /// Frame crop offsets (in pixels).
    pub frame_cropping_flag: u8,
    /// Left crop offset.
    pub frame_crop_left_offset: u32,
    /// Right crop offset.
    pub frame_crop_right_offset: u32,
    /// Top crop offset.
    pub frame_crop_top_offset: u32,
    /// Bottom crop offset.
    pub frame_crop_bottom_offset: u32,

    /// VUI parameters present flag.
    pub vui_parameters_present_flag: u8,

    /// Aspect ratio info present flag.
    pub aspect_ratio_info_present_flag: u8,
    /// Aspect ratio IDC.
    pub aspect_ratio_idc: u8,
    /// Sample aspect ratio width.
    pub sar_width: u16,
    /// Sample aspect ratio height.
    pub sar_height: u16,

    /// Timing info present flag.
    pub timing_info_present_flag: u8,
    /// Fixed frame rate flag.
    pub fixed_frame_rate_flag: u8,

    /// Low delay HRD flag.
    pub low_delay_hrd_flag: u8,
    /// Bitstream restriction flag.
    pub bitstream_restriction_flag: u8,

    /// Motion vectors over picture boundaries flag.
    pub motion_vectors_over_pic_boundaries_flag: u8,
    /// Max bytes per picture denominator.
    pub max_bytes_per_pic_denom: u8,
    /// Max bits per macroblock denominator.
    pub max_bits_per_mb_denom: u8,
    /// Log2 max MV length horizontal.
    pub log2_max_mv_length_horizontal: u8,
    /// Log2 max MV length vertical.
    pub log2_max_mv_length_vertical: u8,
    /// Number of reorder frames.
    pub num_reorder_frames: u8,
    /// Max decoder frame buffering.
    pub max_dec_frame_buffering: u8,
}

/// H.264 picture parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncPictureParameterBufferH264 {
    /// Reconstructed picture surface.
    pub curr_pic: VAPictureH264,
    /// Coded buffer ID.
    pub coded_buf: VABufferID,
    /// PPS ID.
    pub pic_parameter_set_id: u8,
    /// SPS ID.
    pub seq_parameter_set_id: u8,

    /// Last reference frame index.
    pub last_picture: u8,
    /// Frame number (for reference).
    pub frame_num: u16,
    /// Picture coding type (I=1, P=2, B=3).
    pub pic_init_qp: u8,
    /// Number of active references for L0.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Number of active references for L1.
    pub num_ref_idx_l1_active_minus1: u8,
    /// Chroma QP index offset.
    pub chroma_qp_index_offset: i8,
    /// Second chroma QP index offset.
    pub second_chroma_qp_index_offset: i8,

    // Picture fields packed
    /// IDR picture flag.
    pub pic_fields_idr_pic_flag: u8,
    /// Reference picture flag.
    pub pic_fields_reference_pic_flag: u8,
    /// Entropy coding mode (0=CAVLC, 1=CABAC).
    pub pic_fields_entropy_coding_mode_flag: u8,
    /// Weighted prediction flag.
    pub pic_fields_weighted_pred_flag: u8,
    /// Weighted biprediction IDC.
    pub pic_fields_weighted_bipred_idc: u8,
    /// Constrained intra prediction flag.
    pub pic_fields_constrained_intra_pred_flag: u8,
    /// Transform 8x8 mode flag.
    pub pic_fields_transform_8x8_mode_flag: u8,
    /// Deblocking filter control present flag.
    pub pic_fields_deblocking_filter_control_present_flag: u8,
    /// Redundant picture count present flag.
    pub pic_fields_redundant_pic_cnt_present_flag: u8,
    /// Picture scaling matrix present flag.
    pub pic_fields_pic_scaling_matrix_present_flag: u8,
    /// Pic order present flag.
    pub pic_fields_pic_order_present_flag: u8,

    /// Reference pictures for L0.
    pub reference_frames: [VAPictureH264; 16],
}

/// H.264 slice parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncSliceParameterBufferH264 {
    /// Macroblock address of first macroblock in slice.
    pub macroblock_address: u32,
    /// Number of macroblocks in slice.
    pub num_macroblocks: u32,
    /// Macroblock info buffer (optional).
    pub macroblock_info: VABufferID,
    /// Slice type (I=2, P=0, B=1).
    pub slice_type: u8,
    /// PPS ID.
    pub pic_parameter_set_id: u8,
    /// IDR picture ID.
    pub idr_pic_id: u16,
    /// Picture order count LSB.
    pub pic_order_cnt_lsb: u16,
    /// Delta picture order count bottom.
    pub delta_pic_order_cnt_bottom: i32,
    /// Delta picture order count [0].
    pub delta_pic_order_cnt_0: i32,
    /// Delta picture order count [1].
    pub delta_pic_order_cnt_1: i32,
    /// Direct spatial MV prediction flag.
    pub direct_spatial_mv_pred_flag: u8,
    /// Number of reference pictures in L0.
    pub num_ref_idx_active_override_flag: u8,
    /// Number of references for L0.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Number of references for L1.
    pub num_ref_idx_l1_active_minus1: u8,
    /// Reference picture list L0.
    pub ref_pic_list_0: [VAPictureH264; 32],
    /// Reference picture list L1.
    pub ref_pic_list_1: [VAPictureH264; 32],
    /// Luma log2 weight denominator.
    pub luma_log2_weight_denom: u8,
    /// Chroma log2 weight denominator.
    pub chroma_log2_weight_denom: u8,
    /// Luma weights for L0.
    pub luma_weight_l0: [i16; 32],
    /// Luma offsets for L0.
    pub luma_offset_l0: [i16; 32],
    /// Chroma weights for L0.
    pub chroma_weight_l0: [[i16; 2]; 32],
    /// Chroma offsets for L0.
    pub chroma_offset_l0: [[i16; 2]; 32],
    /// Luma weights for L1.
    pub luma_weight_l1: [i16; 32],
    /// Luma offsets for L1.
    pub luma_offset_l1: [i16; 32],
    /// Chroma weights for L1.
    pub chroma_weight_l1: [[i16; 2]; 32],
    /// Chroma offsets for L1.
    pub chroma_offset_l1: [[i16; 2]; 32],
    /// CABAC init IDC.
    pub cabac_init_idc: u8,
    /// Slice QP delta.
    pub slice_qp_delta: i8,
    /// Disable deblocking filter IDC.
    pub disable_deblocking_filter_idc: u8,
    /// Slice alpha C0 offset div 2.
    pub slice_alpha_c0_offset_div2: i8,
    /// Slice beta offset div 2.
    pub slice_beta_offset_div2: i8,
}

/// H.264 picture reference structure.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAPictureH264 {
    /// Surface ID.
    pub picture_id: VASurfaceID,
    /// Frame index.
    pub frame_idx: u32,
    /// Flags (top/bottom field, short/long term reference).
    pub flags: u32,
    /// Top field order count.
    pub top_field_order_cnt: i32,
    /// Bottom field order count.
    pub bottom_field_order_cnt: i32,
}

/// H.264 picture flags.
pub mod va_picture_h264_flags {
    /// Invalid picture.
    pub const VA_PICTURE_H264_INVALID: u32 = 0x00000001;
    /// Top field.
    pub const VA_PICTURE_H264_TOP_FIELD: u32 = 0x00000002;
    /// Bottom field.
    pub const VA_PICTURE_H264_BOTTOM_FIELD: u32 = 0x00000004;
    /// Short-term reference.
    pub const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x00000008;
    /// Long-term reference.
    pub const VA_PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x00000010;
}

// ============================================================================
// HEVC Encoding Parameter Structures
// ============================================================================

/// HEVC sequence parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncSequenceParameterBufferHEVC {
    /// General profile IDC.
    pub general_profile_idc: u8,
    /// General level IDC.
    pub general_level_idc: u8,
    /// General tier flag.
    pub general_tier_flag: u8,
    /// Intra period (GOP size).
    pub intra_period: u32,
    /// Intra IDR period.
    pub intra_idr_period: u32,
    /// IP period.
    pub ip_period: u32,
    /// Target bitrate.
    pub bits_per_second: u32,
    /// Picture width in luma samples.
    pub pic_width_in_luma_samples: u16,
    /// Picture height in luma samples.
    pub pic_height_in_luma_samples: u16,

    // Sequence fields
    /// Chroma format IDC.
    pub seq_fields_chroma_format_idc: u8,
    /// Separate colour plane flag.
    pub seq_fields_separate_colour_plane_flag: u8,
    /// Bit depth for luma minus 8.
    pub seq_fields_bit_depth_luma_minus8: u8,
    /// Bit depth for chroma minus 8.
    pub seq_fields_bit_depth_chroma_minus8: u8,
    /// Scaling list enabled flag.
    pub seq_fields_scaling_list_enabled_flag: u8,
    /// Strong intra smoothing enabled flag.
    pub seq_fields_strong_intra_smoothing_enabled_flag: u8,
    /// AMP (asymmetric motion partition) enabled flag.
    pub seq_fields_amp_enabled_flag: u8,
    /// Sample adaptive offset enabled flag.
    pub seq_fields_sample_adaptive_offset_enabled_flag: u8,
    /// PCM enabled flag.
    pub seq_fields_pcm_enabled_flag: u8,
    /// PCM loop filter disabled flag.
    pub seq_fields_pcm_loop_filter_disabled_flag: u8,
    /// SPS temporal MVP enabled flag.
    pub seq_fields_sps_temporal_mvp_enabled_flag: u8,

    /// Log2 min luma coding block size minus 3.
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// Log2 diff max min luma coding block size.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// Log2 min transform block size minus 2.
    pub log2_min_transform_block_size_minus2: u8,
    /// Log2 diff max min transform block size.
    pub log2_diff_max_min_transform_block_size: u8,
    /// Max transform hierarchy depth inter.
    pub max_transform_hierarchy_depth_inter: u8,
    /// Max transform hierarchy depth intra.
    pub max_transform_hierarchy_depth_intra: u8,

    /// PCM sample bit depth luma minus 1.
    pub pcm_sample_bit_depth_luma_minus1: u8,
    /// PCM sample bit depth chroma minus 1.
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    /// Log2 min PCM luma coding block size minus 3.
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    /// Log2 max PCM luma coding block size minus 3.
    pub log2_max_pcm_luma_coding_block_size_minus3: u8,

    /// VUI parameters present flag.
    pub vui_parameters_present_flag: u8,
    /// VUI fields (similar to H.264).
    pub vui_num_units_in_tick: u32,
    /// VUI time scale.
    pub vui_time_scale: u32,
}

/// HEVC picture parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncPictureParameterBufferHEVC {
    /// Decoded picture buffer.
    pub decoded_curr_pic: VAPictureHEVC,
    /// Reference frame list.
    pub reference_frames: [VAPictureHEVC; 15],
    /// Last picture (IDR/I/P/B flag).
    pub last_picture: u8,
    /// PPS ID.
    pub pic_parameter_set_id: u8,
    /// Coded buffer ID.
    pub coded_buf: VABufferID,

    /// Collocated reference picture index.
    pub collocated_ref_pic_index: u8,

    // Picture coding fields
    /// IDR picture flag.
    pub pic_fields_idr_pic_flag: u8,
    /// Picture coding type (I=1, P=2, B=3).
    pub pic_fields_coding_type: u8,
    /// Reference picture flag.
    pub pic_fields_reference_pic_flag: u8,
    /// Dependent slice segments enabled flag.
    pub pic_fields_dependent_slice_segments_enabled_flag: u8,
    /// Sign data hiding enabled flag.
    pub pic_fields_sign_data_hiding_enabled_flag: u8,
    /// Constrained intra prediction flag.
    pub pic_fields_constrained_intra_pred_flag: u8,
    /// Transform skip enabled flag.
    pub pic_fields_transform_skip_enabled_flag: u8,
    /// CU QP delta enabled flag.
    pub pic_fields_cu_qp_delta_enabled_flag: u8,
    /// Weighted prediction flag.
    pub pic_fields_weighted_pred_flag: u8,
    /// Weighted biprediction flag.
    pub pic_fields_weighted_bipred_flag: u8,
    /// Transquant bypass enabled flag.
    pub pic_fields_transquant_bypass_enabled_flag: u8,
    /// Tiles enabled flag.
    pub pic_fields_tiles_enabled_flag: u8,
    /// Entropy coding sync enabled flag.
    pub pic_fields_entropy_coding_sync_enabled_flag: u8,
    /// Loop filter across tiles enabled flag.
    pub pic_fields_loop_filter_across_tiles_enabled_flag: u8,
    /// PPS loop filter across slices enabled flag.
    pub pic_fields_pps_loop_filter_across_slices_enabled_flag: u8,
    /// Scaling list data present flag.
    pub pic_fields_scaling_list_data_present_flag: u8,
    /// Screen content flag.
    pub pic_fields_screen_content_flag: u8,
    /// Enable GPU weighted prediction.
    pub pic_fields_enable_gpu_weighted_prediction: u8,
    /// No output of prior pictures flag.
    pub pic_fields_no_output_of_prior_pics_flag: u8,

    /// Diff CU QP delta depth.
    pub diff_cu_qp_delta_depth: u8,
    /// PPS Cb QP offset.
    pub pps_cb_qp_offset: i8,
    /// PPS Cr QP offset.
    pub pps_cr_qp_offset: i8,
    /// Number of tile columns minus 1.
    pub num_tile_columns_minus1: u8,
    /// Number of tile rows minus 1.
    pub num_tile_rows_minus1: u8,
    /// Tile column widths.
    pub column_width_minus1: [u16; 19],
    /// Tile row heights.
    pub row_height_minus1: [u16; 21],

    /// Log2 parallel merge level minus 2.
    pub log2_parallel_merge_level_minus2: u8,
    /// CQP value for I-frames.
    pub ctu_max_bitsize_allowed: u8,
    /// Number of reference pictures for L0.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// Number of reference pictures for L1.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// Slice pic parameter set ID.
    pub slice_pic_parameter_set_id: u8,
    /// NAL unit type.
    pub nal_unit_type: u8,

    /// Picture init QP.
    pub pic_init_qp: u8,

    /// Hierarchical level for current picture.
    pub hierarchical_level_plus1: u8,
}

/// HEVC picture reference structure.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAPictureHEVC {
    /// Surface ID.
    pub picture_id: VASurfaceID,
    /// Picture order count.
    pub pic_order_cnt: i32,
    /// Flags.
    pub flags: u32,
}

/// HEVC picture flags.
pub mod va_picture_hevc_flags {
    /// Invalid picture.
    pub const VA_PICTURE_HEVC_INVALID: u32 = 0x00000001;
    /// IDR picture.
    pub const VA_PICTURE_HEVC_IDR_PICTURE: u32 = 0x00000002;
    /// Long-term reference.
    pub const VA_PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x00000004;
    /// Field picture.
    pub const VA_PICTURE_HEVC_FIELD_PIC: u32 = 0x00000008;
    /// Bottom field.
    pub const VA_PICTURE_HEVC_BOTTOM_FIELD: u32 = 0x00000010;
    /// Reference picture set current.
    pub const VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x00000020;
    /// Reference picture set current after.
    pub const VA_PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x00000040;
    /// Reference picture set long term current.
    pub const VA_PICTURE_HEVC_RPS_LT_CURR: u32 = 0x00000080;
}

/// HEVC slice parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncSliceParameterBufferHEVC {
    /// Slice segment address.
    pub slice_segment_address: u32,
    /// Number of CTUs in slice.
    pub num_ctu_in_slice: u32,
    /// Slice type (I=2, P=0, B=1).
    pub slice_type: u8,
    /// Slice pic parameter set ID.
    pub slice_pic_parameter_set_id: u8,
    /// Number of reference pictures in L0.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Number of reference pictures in L1.
    pub num_ref_idx_l1_active_minus1: u8,
    /// Reference picture list L0.
    pub ref_pic_list0: [VAPictureHEVC; 15],
    /// Reference picture list L1.
    pub ref_pic_list1: [VAPictureHEVC; 15],

    /// Luma log2 weight denominator.
    pub luma_log2_weight_denom: u8,
    /// Delta chroma log2 weight denominator.
    pub delta_chroma_log2_weight_denom: i8,
    /// Delta luma weights for L0.
    pub delta_luma_weight_l0: [i8; 15],
    /// Luma offsets for L0.
    pub luma_offset_l0: [i8; 15],
    /// Delta chroma weights for L0.
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    /// Chroma offsets for L0.
    pub chroma_offset_l0: [[i16; 2]; 15],
    /// Delta luma weights for L1.
    pub delta_luma_weight_l1: [i8; 15],
    /// Luma offsets for L1.
    pub luma_offset_l1: [i8; 15],
    /// Delta chroma weights for L1.
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    /// Chroma offsets for L1.
    pub chroma_offset_l1: [[i16; 2]; 15],

    /// Max number of merge candidates minus 1.
    pub max_num_merge_cand: u8,
    /// Slice QP delta.
    pub slice_qp_delta: i8,
    /// Slice Cb QP offset.
    pub slice_cb_qp_offset: i8,
    /// Slice Cr QP offset.
    pub slice_cr_qp_offset: i8,
    /// Slice beta offset div 2.
    pub slice_beta_offset_div2: i8,
    /// Slice tc offset div 2.
    pub slice_tc_offset_div2: i8,

    // Slice fields
    /// Last slice of picture flag.
    pub slice_fields_last_slice_of_pic_flag: u8,
    /// Dependent slice segment flag.
    pub slice_fields_dependent_slice_segment_flag: u8,
    /// Colour plane ID.
    pub slice_fields_colour_plane_id: u8,
    /// Slice temporal MVP enabled flag.
    pub slice_fields_slice_temporal_mvp_enabled_flag: u8,
    /// Slice SAO luma flag.
    pub slice_fields_slice_sao_luma_flag: u8,
    /// Slice SAO chroma flag.
    pub slice_fields_slice_sao_chroma_flag: u8,
    /// Number of L0 references active override flag.
    pub slice_fields_num_ref_idx_active_override_flag: u8,
    /// MV from L0 flag.
    pub slice_fields_mvd_l1_zero_flag: u8,
    /// CABAC init flag.
    pub slice_fields_cabac_init_flag: u8,
    /// Slice deblocking filter disabled flag.
    pub slice_fields_slice_deblocking_filter_disabled_flag: u8,
    /// Slice loop filter across slices enabled flag.
    pub slice_fields_slice_loop_filter_across_slices_enabled_flag: u8,
    /// Collocated from L0 flag.
    pub slice_fields_collocated_from_l0_flag: u8,
}

// ============================================================================
// AV1 Encoding Parameter Structures
// ============================================================================

/// AV1 sequence parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncSequenceParameterBufferAV1 {
    /// Sequence profile.
    pub seq_profile: u8,
    /// Sequence level IDX.
    pub seq_level_idx: u8,
    /// Sequence tier.
    pub seq_tier: u8,
    /// Hierarchy level plus 1.
    pub hierarchical_flag: u8,
    /// Intra period.
    pub intra_period: u32,
    /// IP period.
    pub ip_period: u32,
    /// Target bitrate.
    pub bits_per_second: u32,

    // Sequence info fields
    /// Still picture flag.
    pub seq_fields_still_picture: u8,
    /// Use 128x128 superblock flag.
    pub seq_fields_use_128x128_superblock: u8,
    /// Enable filter intra flag.
    pub seq_fields_enable_filter_intra: u8,
    /// Enable intra edge filter flag.
    pub seq_fields_enable_intra_edge_filter: u8,
    /// Enable interintra compound flag.
    pub seq_fields_enable_interintra_compound: u8,
    /// Enable masked compound flag.
    pub seq_fields_enable_masked_compound: u8,
    /// Enable warped motion flag.
    pub seq_fields_enable_warped_motion: u8,
    /// Enable dual filter flag.
    pub seq_fields_enable_dual_filter: u8,
    /// Enable order hint flag.
    pub seq_fields_enable_order_hint: u8,
    /// Enable JNT compound flag.
    pub seq_fields_enable_jnt_comp: u8,
    /// Enable reference frame MVP flag.
    pub seq_fields_enable_ref_frame_mvs: u8,
    /// Enable superres flag.
    pub seq_fields_enable_superres: u8,
    /// Enable CDEF flag.
    pub seq_fields_enable_cdef: u8,
    /// Enable restoration flag.
    pub seq_fields_enable_restoration: u8,
    /// Bit depth minus 8.
    pub seq_fields_bit_depth_minus8: u8,
    /// Subsampling X (0=4:4:4, 1=4:2:0/4:2:2).
    pub seq_fields_subsampling_x: u8,
    /// Subsampling Y (0=4:4:4/4:2:2, 1=4:2:0).
    pub seq_fields_subsampling_y: u8,

    /// Order hint bits minus 1.
    pub order_hint_bits_minus_1: u8,
}

/// AV1 picture parameter buffer for encoding.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncPictureParameterBufferAV1 {
    /// Frame width in pixels.
    pub frame_width_minus_1: u16,
    /// Frame height in pixels.
    pub frame_height_minus_1: u16,
    /// Reconstructed frame surface.
    pub reconstructed_frame: VASurfaceID,
    /// Coded buffer ID.
    pub coded_buf: VABufferID,
    /// Reference frames.
    pub reference_frames: [VASurfaceID; 8],
    /// Reference frame index.
    pub ref_frame_idx: [u8; 7],
    /// Primary reference frame.
    pub primary_ref_frame: u8,
    /// Order hint.
    pub order_hint: u8,
    /// Refresh frame flags.
    pub refresh_frame_flags: u8,

    // Picture info fields
    /// Frame type (0=KEY, 1=INTER, 2=INTRA_ONLY, 3=SWITCH).
    pub pic_fields_frame_type: u8,
    /// Error resilient mode flag.
    pub pic_fields_error_resilient_mode: u8,
    /// Disable CDF update flag.
    pub pic_fields_disable_cdf_update: u8,
    /// Use superres flag.
    pub pic_fields_use_superres: u8,
    /// Allow high precision MV flag.
    pub pic_fields_allow_high_precision_mv: u8,
    /// Use reference frame MV flag.
    pub pic_fields_use_ref_frame_mvs: u8,
    /// Disable frame end update CDF flag.
    pub pic_fields_disable_frame_end_update_cdf: u8,
    /// Reduced TX set flag.
    pub pic_fields_reduced_tx_set: u8,
    /// Enable frame OBU flag.
    pub pic_fields_enable_frame_obu: u8,
    /// Long term reference flag.
    pub pic_fields_long_term_reference: u8,
    /// Disable frame recon flag.
    pub pic_fields_disable_frame_recon: u8,
    /// Allow intrabc flag.
    pub pic_fields_allow_intrabc: u8,
    /// Palette mode enabled flag.
    pub pic_fields_palette_mode_enable: u8,

    /// Superres scale denominator.
    pub superres_scale_denominator: u8,
    /// Interpolation filter.
    pub interpolation_filter: u8,
    /// Filter level [0] (Y vertical).
    pub filter_level_0: u8,
    /// Filter level [1] (Y horizontal).
    pub filter_level_1: u8,
    /// Filter level U.
    pub filter_level_u: u8,
    /// Filter level V.
    pub filter_level_v: u8,

    // Loop filter fields
    /// Sharpness level.
    pub lf_sharpness_level: u8,
    /// Loop filter delta enabled flag.
    pub lf_mode_ref_delta_enabled: u8,
    /// Loop filter delta update flag.
    pub lf_mode_ref_delta_update: u8,
    /// Reference deltas.
    pub lf_ref_deltas: [i8; 8],
    /// Mode deltas.
    pub lf_mode_deltas: [i8; 2],

    /// Base QP index.
    pub base_qindex: u8,
    /// Y DC delta Q.
    pub y_dc_delta_q: i8,
    /// U DC delta Q.
    pub u_dc_delta_q: i8,
    /// U AC delta Q.
    pub u_ac_delta_q: i8,
    /// V DC delta Q.
    pub v_dc_delta_q: i8,
    /// V AC delta Q.
    pub v_ac_delta_q: i8,

    /// Min base QP index.
    pub min_base_qindex: u8,
    /// Max base QP index.
    pub max_base_qindex: u8,

    // Quantization fields
    /// Using QMatrix flag.
    pub qmatrix_using_qmatrix: u8,
    /// QM Y.
    pub qmatrix_qm_y: u8,
    /// QM U.
    pub qmatrix_qm_u: u8,
    /// QM V.
    pub qmatrix_qm_v: u8,

    // Mode control fields
    /// Delta Q present flag.
    pub mode_control_delta_q_present: u8,
    /// Delta Q resolution.
    pub mode_control_delta_q_res: u8,
    /// Delta LF present flag.
    pub mode_control_delta_lf_present: u8,
    /// Delta LF resolution.
    pub mode_control_delta_lf_res: u8,
    /// Delta LF multi flag.
    pub mode_control_delta_lf_multi: u8,
    /// TX mode (0=ONLY_4X4, 1=LARGEST, 2=SELECT).
    pub mode_control_tx_mode: u8,
    /// Reference mode (0=SINGLE, 1=COMPOUND, 2=SELECT).
    pub mode_control_reference_mode: u8,
    /// Skip mode present flag.
    pub mode_control_skip_mode_present: u8,

    // Segmentation fields
    /// Segmentation enabled flag.
    pub seg_enabled: u8,
    /// Segmentation update map flag.
    pub seg_update_map: u8,
    /// Segmentation temporal update flag.
    pub seg_temporal_update: u8,
    /// Segmentation feature data.
    pub seg_feature_data: [[i16; 8]; 8],

    /// Tile group OBU header info.
    pub tile_group_obu_hdr_info: u8,
    /// Number of tile groups minus 1.
    pub number_tile_groups_minus1: u8,

    // Tile info
    /// Uniform tile spacing flag.
    pub tile_uniform_spacing: u8,
    /// Tile columns.
    pub tile_cols: u8,
    /// Tile rows.
    pub tile_rows: u8,
    /// Width in SB minus 1 for each tile column.
    pub width_in_sbs_minus_1: [u16; 63],
    /// Height in SB minus 1 for each tile row.
    pub height_in_sbs_minus_1: [u16; 63],
    /// Context update tile ID.
    pub context_update_tile_id: u16,

    // CDEF parameters
    /// CDEF damping minus 3.
    pub cdef_damping_minus_3: u8,
    /// CDEF bits.
    pub cdef_bits: u8,
    /// CDEF Y strengths.
    pub cdef_y_strengths: [u8; 8],
    /// CDEF UV strengths.
    pub cdef_uv_strengths: [u8; 8],

    // Loop restoration parameters
    /// Loop restoration type for Y.
    pub lr_type_y: u8,
    /// Loop restoration type for U.
    pub lr_type_u: u8,
    /// Loop restoration type for V.
    pub lr_type_v: u8,
    /// Loop restoration unit shift.
    pub lr_unit_shift: u8,
    /// Loop restoration UV shift.
    pub lr_uv_shift: u8,

    // Warp parameters
    /// Global motion type per reference.
    pub wm_type: [u8; 7],
    /// Warp motion parameters.
    pub wm: [[i32; 6]; 7],
}

// ============================================================================
// Surface Handling Structures
// ============================================================================

/// VA Surface attribute types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VASurfaceAttribType {
    /// No attribute.
    None = 0,
    /// Pixel format (fourcc).
    PixelFormat = 1,
    /// Minimum width.
    MinWidth = 2,
    /// Maximum width.
    MaxWidth = 3,
    /// Minimum height.
    MinHeight = 4,
    /// Maximum height.
    MaxHeight = 5,
    /// Memory type.
    MemoryType = 6,
    /// External buffer descriptor.
    ExternalBufferDescriptor = 7,
    /// Usage hint.
    UsageHint = 8,
    /// DRM format modifiers.
    DRMFormatModifiers = 9,
}

/// VA Surface attribute structure.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VASurfaceAttrib {
    /// Attribute type.
    pub type_: VASurfaceAttribType,
    /// Flags (read, write, etc.).
    pub flags: u32,
    /// Attribute value.
    pub value: VAGenericValue,
}

impl Default for VASurfaceAttrib {
    fn default() -> Self {
        Self {
            type_: VASurfaceAttribType::None,
            flags: 0,
            value: VAGenericValue::default(),
        }
    }
}

/// Surface attribute flags.
pub mod va_surface_attrib_flags {
    /// Attribute is gettable.
    pub const VA_SURFACE_ATTRIB_GETTABLE: u32 = 0x00000001;
    /// Attribute is settable.
    pub const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x00000002;
    /// Attribute is not set (use default).
    pub const VA_SURFACE_ATTRIB_NOT_SUPPORTED: u32 = 0x00000000;
}

/// Memory type flags for surface attributes.
pub mod va_surface_mem_type {
    /// VA surface memory (default).
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_VA: u32 = 0x00000001;
    /// V4L2 buffer.
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_V4L2: u32 = 0x00000002;
    /// User pointer.
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_USER_PTR: u32 = 0x00000004;
    /// Kernel DRM buffer.
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_KERNEL_DRM: u32 = 0x10000000;
    /// DRM PRIME handle.
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME: u32 = 0x20000000;
    /// DRM PRIME 2 (dma-buf with modifiers).
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x40000000;
    /// DRM PRIME 3 (with DRM PRIME file descriptors).
    pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_3: u32 = 0x80000000;
}

/// Usage hint flags for surface attributes.
pub mod va_surface_usage_hint {
    /// Hint for decoding.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_DECODER: u32 = 0x00000001;
    /// Hint for encoding.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_ENCODER: u32 = 0x00000002;
    /// Hint for video processing.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_VPP_READ: u32 = 0x00000004;
    /// Hint for video processing write.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_VPP_WRITE: u32 = 0x00000008;
    /// Hint for display.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_DISPLAY: u32 = 0x00000010;
    /// Hint for export.
    pub const VA_SURFACE_ATTRIB_USAGE_HINT_EXPORT: u32 = 0x00000020;
}

// ============================================================================
// DRM PRIME Surface Export Structures
// ============================================================================

/// DRM PRIME surface descriptor for dma-buf export.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VADRMPRIMESurfaceDescriptor {
    /// Fourcc code for pixel format.
    pub fourcc: u32,
    /// Surface width.
    pub width: u32,
    /// Surface height.
    pub height: u32,
    /// Number of DRM objects (dma-buf FDs).
    pub num_objects: u32,
    /// DRM objects (dma-buf FDs and sizes).
    pub objects: [VADRMPRIMESurfaceDescriptorObject; 4],
    /// Number of layers (usually 1 for NV12, 2 for YUV420P).
    pub num_layers: u32,
    /// Layer information.
    pub layers: [VADRMPRIMESurfaceDescriptorLayer; 4],
}

/// DRM PRIME object descriptor (dma-buf FD and size).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VADRMPRIMESurfaceDescriptorObject {
    /// DMA-BUF file descriptor.
    pub fd: i32,
    /// Total size of the dma-buf.
    pub size: u32,
    /// DRM format modifier.
    pub drm_format_modifier: u64,
}

/// DRM PRIME layer descriptor.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VADRMPRIMESurfaceDescriptorLayer {
    /// DRM format for this layer.
    pub drm_format: u32,
    /// Number of planes in this layer.
    pub num_planes: u32,
    /// Object index for each plane.
    pub object_index: [u32; 4],
    /// Offset in object for each plane.
    pub offset: [u32; 4],
    /// Pitch (stride) for each plane.
    pub pitch: [u32; 4],
}

/// Surface export flags.
pub mod va_surface_export_flags {
    /// Export as read-only.
    pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
    /// Export as write-only.
    pub const VA_EXPORT_SURFACE_WRITE_ONLY: u32 = 0x0002;
    /// Export as read-write.
    pub const VA_EXPORT_SURFACE_READ_WRITE: u32 = 0x0003;
    /// Separate layers.
    pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x0004;
    /// Composed layers.
    pub const VA_EXPORT_SURFACE_COMPOSED_LAYERS: u32 = 0x0008;
}

// ============================================================================
// Miscellaneous Parameter Structures
// ============================================================================

/// Miscellaneous parameter type for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VAEncMiscParameterType {
    /// Frame rate parameter.
    FrameRate = 0,
    /// Rate control parameter.
    RateControl = 1,
    /// Max slice size parameter.
    MaxSliceSize = 2,
    /// AI lookahead parameter.
    AILookahead = 3,
    /// Quantization range.
    Quantization = 4,
    /// Skip frame parameter.
    SkipFrame = 5,
    /// HRD parameter.
    HRD = 6,
    /// Quality level parameter.
    QualityLevel = 7,
    /// Max frame size parameter.
    MaxFrameSize = 8,
    /// ROI parameter.
    ROI = 9,
    /// Temporal layer structure.
    TemporalLayerStructure = 10,
    /// Dirty rectangle.
    DirtyRect = 11,
    /// Parallel BRC.
    ParallelBRC = 12,
    /// Multi-pass frame size.
    MultiPassFrameSize = 13,
    /// FEI frame control.
    FEIFrameControl = 14,
    /// Encode QP.
    EncQP = 15,
    /// Custom round control.
    CustomRoundingControl = 16,
}

/// Miscellaneous parameter buffer header.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterBuffer {
    /// Parameter type.
    pub type_: VAEncMiscParameterType,
    // Data follows immediately after this header
}

impl Default for VAEncMiscParameterType {
    fn default() -> Self {
        VAEncMiscParameterType::FrameRate
    }
}

/// Frame rate parameter.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterFrameRate {
    /// Frame rate (16.16 fixed point or simple fraction).
    pub framerate: u32,
    /// Flags.
    pub framerate_flags: u32,
}

/// Rate control parameter.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterRateControl {
    /// Bits per second.
    pub bits_per_second: u32,
    /// Target percentage.
    pub target_percentage: u32,
    /// Window size.
    pub window_size: u32,
    /// Initial QP.
    pub initial_qp: u32,
    /// Min QP.
    pub min_qp: u32,
    /// Basic unit size.
    pub basic_unit_size: u32,

    // Rate control flags
    /// Reset flag.
    pub rc_flags_reset: u8,
    /// Disable bit stuffing.
    pub rc_flags_disable_bit_stuffing: u8,
    /// MB rate control mode.
    pub rc_flags_mb_rate_control: u8,
    /// Enable parallel BRC.
    pub rc_flags_enable_parallel_brc: u8,
    /// Frame tolerance mode.
    pub rc_flags_frame_tolerance_mode: u8,
    /// Target frame size.
    pub target_frame_size: u32,

    /// ICQ quality factor (0-51, lower = better).
    pub icq_quality_factor: u8,
    /// Max QP.
    pub max_qp: u8,
    /// Quality factor.
    pub quality_factor: u8,
    /// Max frame size types.
    pub max_frame_size_types: u8,
    /// Max frame size for I/P/B.
    pub max_frame_size: [u32; 3],
}

/// HRD (Hypothetical Reference Decoder) parameter.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterHRD {
    /// Initial buffer fullness.
    pub initial_buffer_fullness: u32,
    /// Buffer size.
    pub buffer_size: u32,
}

/// Quality level parameter (Intel-specific).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterBufferQualityLevel {
    /// Quality level (1-7, lower = better quality/slower).
    pub quality_level: u32,
}

/// Max frame size parameter.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncMiscParameterBufferMaxFrameSize {
    /// Max frame size type.
    pub max_frame_size_type: u32,
    /// Max frame size in bytes.
    pub max_frame_size: u32,
}

/// ROI (Region of Interest) parameter.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct VAEncMiscParameterBufferROI {
    /// Number of ROI regions.
    pub num_roi: u32,
    /// Max delta QP.
    pub max_delta_qp: i8,
    /// Min delta QP.
    pub min_delta_qp: i8,
    /// ROI flags.
    pub roi_flags_differential_qp: u8,
    /// ROI regions.
    pub roi: [VAEncROI; 8],
}

/// ROI region definition.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VAEncROI {
    /// Left coordinate.
    pub roi_left: i16,
    /// Top coordinate.
    pub roi_top: i16,
    /// Right coordinate.
    pub roi_right: i16,
    /// Bottom coordinate.
    pub roi_bottom: i16,
    /// ROI value (QP delta or priority).
    pub roi_value: i8,
}

// ============================================================================
// Low-Power Encoding Support
// ============================================================================

/// Low-power encoding configuration.
#[derive(Debug, Clone, Default)]
pub struct VaLowPowerConfig {
    /// Use low-power encoding entrypoint (VAEntrypointEncSliceLP).
    pub enabled: bool,
    /// Quality level for low-power mode (Intel-specific, 1-7).
    /// 1 = best quality (slowest), 7 = worst quality (fastest).
    pub quality_level: u32,
    /// Target usage (Intel MSDK compatible).
    /// 1 = best quality, 4 = balanced, 7 = best speed.
    pub target_usage: u32,
    /// Enable look-ahead for better rate control.
    pub lookahead_depth: u32,
    /// Enable adaptive I-frame insertion.
    pub adaptive_i: bool,
    /// Enable adaptive B-frame decision.
    pub adaptive_b: bool,
    /// Enable MBBRC (macroblock-level rate control).
    pub mbbrc: bool,
    /// Low delay BRC for streaming.
    pub low_delay_brc: bool,
}

impl VaLowPowerConfig {
    /// Create a preset for maximum speed (streaming/real-time).
    pub fn max_speed() -> Self {
        Self {
            enabled: true,
            quality_level: 7,
            target_usage: 7,
            lookahead_depth: 0,
            adaptive_i: false,
            adaptive_b: false,
            mbbrc: false,
            low_delay_brc: true,
        }
    }

    /// Create a preset for balanced speed/quality.
    pub fn balanced() -> Self {
        Self {
            enabled: true,
            quality_level: 4,
            target_usage: 4,
            lookahead_depth: 8,
            adaptive_i: true,
            adaptive_b: true,
            mbbrc: true,
            low_delay_brc: false,
        }
    }

    /// Create a preset for quality (file encoding).
    pub fn max_quality() -> Self {
        Self {
            enabled: true,
            quality_level: 1,
            target_usage: 1,
            lookahead_depth: 40,
            adaptive_i: true,
            adaptive_b: true,
            mbbrc: true,
            low_delay_brc: false,
        }
    }
}

/// Quality vs speed tradeoff modes (Intel-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaQualitySpeedMode {
    /// Best quality, slowest encoding.
    Quality,
    /// Balanced quality and speed.
    Balanced,
    /// Fastest encoding, lower quality.
    Speed,
    /// Target specific quality level (1-7).
    Custom(u32),
}

impl VaQualitySpeedMode {
    /// Get the quality level value (1-7).
    pub fn quality_level(&self) -> u32 {
        match self {
            VaQualitySpeedMode::Quality => 1,
            VaQualitySpeedMode::Balanced => 4,
            VaQualitySpeedMode::Speed => 7,
            VaQualitySpeedMode::Custom(level) => (*level).clamp(1, 7),
        }
    }

    /// Get description of this mode.
    pub fn description(&self) -> &'static str {
        match self {
            VaQualitySpeedMode::Quality => "Best quality, slowest encoding",
            VaQualitySpeedMode::Balanced => "Balanced quality and speed",
            VaQualitySpeedMode::Speed => "Fastest encoding, lower quality",
            VaQualitySpeedMode::Custom(_) => "Custom quality level",
        }
    }
}

// ============================================================================
// Multi-GPU Support
// ============================================================================

/// DRM device information.
#[derive(Debug, Clone)]
pub struct VaDrmDevice {
    /// Device path (e.g., /dev/dri/renderD128).
    pub path: String,
    /// Device index.
    pub index: u32,
    /// PCI bus ID (if available).
    pub pci_bus_id: Option<String>,
    /// Vendor ID.
    pub vendor_id: u32,
    /// Device ID.
    pub device_id: u32,
    /// Vendor name.
    pub vendor_name: String,
    /// Device name/description.
    pub device_name: String,
    /// Is this the primary GPU?
    pub is_primary: bool,
    /// Supported VA-API profiles.
    pub supported_profiles: Vec<VAProfile>,
    /// Supported entrypoints per profile.
    pub supported_entrypoints: Vec<(VAProfile, Vec<VAEntrypoint>)>,
    /// Supported RT formats.
    pub supported_rt_formats: u32,
    /// Maximum resolution (width, height).
    pub max_resolution: (u32, u32),
    /// Supports low-power encoding.
    pub supports_low_power: bool,
}

impl Default for VaDrmDevice {
    fn default() -> Self {
        Self {
            path: String::new(),
            index: 0,
            pci_bus_id: None,
            vendor_id: 0,
            device_id: 0,
            vendor_name: String::new(),
            device_name: String::new(),
            is_primary: false,
            supported_profiles: Vec::new(),
            supported_entrypoints: Vec::new(),
            supported_rt_formats: 0,
            max_resolution: (0, 0),
            supports_low_power: false,
        }
    }
}

/// Multi-GPU device enumerator.
pub struct VaDeviceEnumerator {
    /// Discovered devices.
    devices: Vec<VaDrmDevice>,
    /// Selected device index.
    selected_index: Option<u32>,
}

impl VaDeviceEnumerator {
    /// Create a new device enumerator and scan for devices.
    pub fn new() -> Self {
        let mut enumerator = Self {
            devices: Vec::new(),
            selected_index: None,
        };
        enumerator.enumerate_devices();
        enumerator
    }

    /// Enumerate available DRM devices.
    fn enumerate_devices(&mut self) {
        // Standard DRM render node paths
        let render_node_paths = [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/dri/renderD130",
            "/dev/dri/renderD131",
        ];

        let mut index = 0;
        for path in &render_node_paths {
            // In real implementation, would:
            // 1. open() the device
            // 2. vaGetDisplayDRM() to get VADisplay
            // 3. vaInitialize() and query capabilities
            // 4. Parse vendor/device info from sysfs

            // Simulate device discovery
            if std::path::Path::new(path).exists() || index == 0 {
                let device = self.create_simulated_device(path, index);
                self.devices.push(device);
                index += 1;
            }
        }
    }

    /// Create a simulated device for testing.
    fn create_simulated_device(&self, path: &str, index: u32) -> VaDrmDevice {
        // Simulate Intel or AMD device based on index
        let (vendor_id, vendor_name, device_name, supports_low_power) = if index == 0 {
            (
                0x8086,
                "Intel Corporation".to_string(),
                "Intel HD Graphics".to_string(),
                true,
            )
        } else {
            (
                0x1002,
                "AMD".to_string(),
                "AMD Radeon Graphics".to_string(),
                false,
            )
        };

        VaDrmDevice {
            path: path.to_string(),
            index,
            pci_bus_id: Some(format!("0000:00:0{}.0", index + 2)),
            vendor_id,
            device_id: 0x9A49 + index,
            vendor_name,
            device_name,
            is_primary: index == 0,
            supported_profiles: vec![
                VAProfile::H264ConstrainedBaseline,
                VAProfile::H264Main,
                VAProfile::H264High,
                VAProfile::HEVCMain,
                VAProfile::HEVCMain10,
                VAProfile::AV1Profile0,
            ],
            supported_entrypoints: vec![
                (VAProfile::H264High, vec![VAEntrypoint::VLD, VAEntrypoint::EncSlice, VAEntrypoint::EncSliceLP]),
                (VAProfile::HEVCMain, vec![VAEntrypoint::VLD, VAEntrypoint::EncSlice, VAEntrypoint::EncSliceLP]),
                (VAProfile::AV1Profile0, vec![VAEntrypoint::VLD, VAEntrypoint::EncSlice]),
            ],
            supported_rt_formats: va_rt_format::VA_RT_FORMAT_YUV420
                | va_rt_format::VA_RT_FORMAT_YUV420_10,
            max_resolution: (4096, 2160),
            supports_low_power,
        }
    }

    /// Get list of discovered devices.
    pub fn devices(&self) -> &[VaDrmDevice] {
        &self.devices
    }

    /// Get device count.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Select a device by index.
    pub fn select_device(&mut self, index: u32) -> Result<&VaDrmDevice> {
        if (index as usize) < self.devices.len() {
            self.selected_index = Some(index);
            Ok(&self.devices[index as usize])
        } else {
            Err(HwAccelError::DeviceInit(format!(
                "Device index {} out of range (0-{})",
                index,
                self.devices.len() - 1
            )))
        }
    }

    /// Select a device by path.
    pub fn select_device_by_path(&mut self, path: &str) -> Result<&VaDrmDevice> {
        for (index, device) in self.devices.iter().enumerate() {
            if device.path == path {
                self.selected_index = Some(index as u32);
                return Ok(device);
            }
        }
        Err(HwAccelError::DeviceInit(format!(
            "Device not found: {}",
            path
        )))
    }

    /// Get the selected device.
    pub fn selected_device(&self) -> Option<&VaDrmDevice> {
        self.selected_index
            .and_then(|i| self.devices.get(i as usize))
    }

    /// Find devices that support a specific codec for encoding.
    pub fn find_encode_capable(&self, codec: HwCodec) -> Vec<&VaDrmDevice> {
        let target_profile = match codec {
            HwCodec::H264 => VAProfile::H264High,
            HwCodec::Hevc => VAProfile::HEVCMain,
            HwCodec::Av1 => VAProfile::AV1Profile0,
            HwCodec::Vp9 => VAProfile::VP9Profile0,
            _ => return Vec::new(),
        };

        self.devices
            .iter()
            .filter(|dev| {
                dev.supported_entrypoints.iter().any(|(profile, eps)| {
                    *profile == target_profile && eps.iter().any(|ep| ep.is_encode())
                })
            })
            .collect()
    }

    /// Find devices that support low-power encoding.
    pub fn find_low_power_capable(&self) -> Vec<&VaDrmDevice> {
        self.devices
            .iter()
            .filter(|dev| dev.supports_low_power)
            .collect()
    }
}

impl Default for VaDeviceEnumerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Encoding Workflow Functions (Stubbed)
// ============================================================================

/// Wrapper for VA-API encoding workflow.
/// This struct provides stubbed implementations of the core VA-API functions
/// that can be replaced with actual FFI calls when libva-sys bindings are available.
pub struct VaEncodingWorkflow {
    /// VA Display handle.
    display: VADisplay,
    /// Configuration ID.
    config_id: VAConfigID,
    /// Context ID.
    context_id: VAContextID,
    /// Created surfaces.
    surfaces: Vec<VASurfaceID>,
    /// Created buffers.
    buffers: Vec<VABufferID>,
    /// Whether initialized.
    initialized: bool,
    /// Current picture started.
    picture_started: bool,
    /// Device information.
    device_info: Option<VaDrmDevice>,
}

impl VaEncodingWorkflow {
    /// Create a new encoding workflow (stubbed).
    pub fn new() -> Self {
        Self {
            display: std::ptr::null_mut(),
            config_id: VA_INVALID_ID,
            context_id: VA_INVALID_ID,
            surfaces: Vec::new(),
            buffers: Vec::new(),
            initialized: false,
            picture_started: false,
            device_info: None,
        }
    }

    /// Initialize VA-API display from DRM device.
    ///
    /// In real implementation, would call:
    /// - open(device_path)
    /// - vaGetDisplayDRM(drm_fd)
    /// - vaInitialize(display, &major, &minor)
    pub fn va_initialize(&mut self, device_path: &str) -> Result<(i32, i32)> {
        // Stubbed: simulate successful initialization
        if device_path.is_empty() {
            return Err(HwAccelError::DeviceInit("Empty device path".to_string()));
        }

        self.display = 1 as VADisplay; // Simulated non-null display
        self.initialized = true;

        // Return simulated VA-API version 1.20
        Ok((1, 20))
    }

    /// Create a configuration for encoding.
    ///
    /// In real implementation, would call:
    /// - vaCreateConfig(display, profile, entrypoint, attribs, num_attribs, &config_id)
    pub fn va_create_config(
        &mut self,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attribs: &[VAConfigAttrib],
    ) -> Result<VAConfigID> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        // Validate profile/entrypoint combination
        if !entrypoint.is_encode() {
            return Err(HwAccelError::Config(
                "Entrypoint is not for encoding".to_string(),
            ));
        }

        // Stubbed: return simulated config ID
        self.config_id = 1;

        // Log the configuration
        tracing::debug!(
            "Created config: profile={:?}, entrypoint={:?}, attribs={}",
            profile,
            entrypoint,
            attribs.len()
        );

        Ok(self.config_id)
    }

    /// Create encoding context.
    ///
    /// In real implementation, would call:
    /// - vaCreateContext(display, config_id, width, height, flag, surfaces, num_surfaces, &context_id)
    pub fn va_create_context(
        &mut self,
        width: u32,
        height: u32,
        surfaces: &[VASurfaceID],
    ) -> Result<VAContextID> {
        if self.config_id == VA_INVALID_ID {
            return Err(HwAccelError::Config("Config not created".to_string()));
        }

        if width == 0 || height == 0 {
            return Err(HwAccelError::Config("Invalid dimensions".to_string()));
        }

        // Stubbed: return simulated context ID
        self.context_id = 1;
        self.surfaces = surfaces.to_vec();

        tracing::debug!(
            "Created context: {}x{}, {} surfaces",
            width,
            height,
            surfaces.len()
        );

        Ok(self.context_id)
    }

    /// Create surfaces for encoding/decoding.
    ///
    /// In real implementation, would call:
    /// - vaCreateSurfaces(display, rt_format, width, height, surfaces, num_surfaces, attribs, num_attribs)
    pub fn va_create_surfaces(
        &mut self,
        rt_format: u32,
        width: u32,
        height: u32,
        num_surfaces: u32,
        attribs: Option<&[VASurfaceAttrib]>,
    ) -> Result<Vec<VASurfaceID>> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        if width == 0 || height == 0 {
            return Err(HwAccelError::Config("Invalid dimensions".to_string()));
        }

        // Stubbed: create simulated surface IDs
        let mut surfaces = Vec::with_capacity(num_surfaces as usize);
        let base_id = self.surfaces.len() as u32 + 1;

        for i in 0..num_surfaces {
            surfaces.push(base_id + i);
        }

        self.surfaces.extend(&surfaces);

        tracing::debug!(
            "Created {} surfaces: {}x{}, format=0x{:08x}, attribs={}",
            num_surfaces,
            width,
            height,
            rt_format,
            attribs.map_or(0, |a| a.len())
        );

        Ok(surfaces)
    }

    /// Create a data buffer.
    ///
    /// In real implementation, would call:
    /// - vaCreateBuffer(display, context, type, size, num_elements, data, &buffer_id)
    pub fn va_create_buffer(
        &mut self,
        buffer_type: VABufferType,
        size: u32,
        num_elements: u32,
        data: Option<&[u8]>,
    ) -> Result<VABufferID> {
        if self.context_id == VA_INVALID_ID {
            return Err(HwAccelError::Config("Context not created".to_string()));
        }

        // Stubbed: create simulated buffer ID
        let buffer_id = self.buffers.len() as u32 + 1;
        self.buffers.push(buffer_id);

        tracing::debug!(
            "Created buffer: type={:?}, size={}, elements={}, has_data={}",
            buffer_type,
            size,
            num_elements,
            data.is_some()
        );

        Ok(buffer_id)
    }

    /// Begin encoding a picture.
    ///
    /// In real implementation, would call:
    /// - vaBeginPicture(display, context, surface)
    pub fn va_begin_picture(&mut self, surface: VASurfaceID) -> Result<()> {
        if self.context_id == VA_INVALID_ID {
            return Err(HwAccelError::Config("Context not created".to_string()));
        }

        if self.picture_started {
            return Err(HwAccelError::Encode(
                "Picture already started".to_string(),
            ));
        }

        self.picture_started = true;

        tracing::debug!("Begin picture: surface={}", surface);

        Ok(())
    }

    /// Render buffers for encoding.
    ///
    /// In real implementation, would call:
    /// - vaRenderPicture(display, context, buffers, num_buffers)
    pub fn va_render_picture(&mut self, buffers: &[VABufferID]) -> Result<()> {
        if !self.picture_started {
            return Err(HwAccelError::Encode(
                "Picture not started".to_string(),
            ));
        }

        tracing::debug!("Render picture: {} buffers", buffers.len());

        Ok(())
    }

    /// End encoding a picture.
    ///
    /// In real implementation, would call:
    /// - vaEndPicture(display, context)
    pub fn va_end_picture(&mut self) -> Result<()> {
        if !self.picture_started {
            return Err(HwAccelError::Encode(
                "Picture not started".to_string(),
            ));
        }

        self.picture_started = false;

        tracing::debug!("End picture");

        Ok(())
    }

    /// Synchronize a surface (wait for encoding to complete).
    ///
    /// In real implementation, would call:
    /// - vaSyncSurface(display, surface)
    pub fn va_sync_surface(&self, surface: VASurfaceID) -> Result<()> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        tracing::debug!("Sync surface: {}", surface);

        Ok(())
    }

    /// Map a buffer for CPU access.
    ///
    /// In real implementation, would call:
    /// - vaMapBuffer(display, buffer, &data_ptr)
    pub fn va_map_buffer(&self, buffer: VABufferID) -> Result<*mut u8> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        // Stubbed: return simulated pointer
        // In real implementation, this would return actual mapped memory

        tracing::debug!("Map buffer: {}", buffer);

        // Return a non-null pointer for simulation
        Ok(buffer as *mut u8)
    }

    /// Unmap a previously mapped buffer.
    ///
    /// In real implementation, would call:
    /// - vaUnmapBuffer(display, buffer)
    pub fn va_unmap_buffer(&self, buffer: VABufferID) -> Result<()> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        tracing::debug!("Unmap buffer: {}", buffer);

        Ok(())
    }

    /// Destroy buffers.
    ///
    /// In real implementation, would call:
    /// - vaDestroyBuffer(display, buffer) for each buffer
    pub fn va_destroy_buffers(&mut self, buffers: &[VABufferID]) -> Result<()> {
        for buffer in buffers {
            if let Some(pos) = self.buffers.iter().position(|&b| b == *buffer) {
                self.buffers.remove(pos);
            }
        }

        tracing::debug!("Destroyed {} buffers", buffers.len());

        Ok(())
    }

    /// Destroy surfaces.
    ///
    /// In real implementation, would call:
    /// - vaDestroySurfaces(display, surfaces, num_surfaces)
    pub fn va_destroy_surfaces(&mut self, surfaces: &[VASurfaceID]) -> Result<()> {
        for surface in surfaces {
            if let Some(pos) = self.surfaces.iter().position(|&s| s == *surface) {
                self.surfaces.remove(pos);
            }
        }

        tracing::debug!("Destroyed {} surfaces", surfaces.len());

        Ok(())
    }

    /// Destroy context.
    ///
    /// In real implementation, would call:
    /// - vaDestroyContext(display, context)
    pub fn va_destroy_context(&mut self) -> Result<()> {
        if self.context_id != VA_INVALID_ID {
            self.context_id = VA_INVALID_ID;
            tracing::debug!("Destroyed context");
        }

        Ok(())
    }

    /// Destroy configuration.
    ///
    /// In real implementation, would call:
    /// - vaDestroyConfig(display, config)
    pub fn va_destroy_config(&mut self) -> Result<()> {
        if self.config_id != VA_INVALID_ID {
            self.config_id = VA_INVALID_ID;
            tracing::debug!("Destroyed config");
        }

        Ok(())
    }

    /// Terminate VA-API.
    ///
    /// In real implementation, would call:
    /// - vaTerminate(display)
    pub fn va_terminate(&mut self) -> Result<()> {
        self.va_destroy_context()?;
        self.va_destroy_config()?;
        self.surfaces.clear();
        self.buffers.clear();
        self.display = std::ptr::null_mut();
        self.initialized = false;

        tracing::debug!("Terminated VA-API");

        Ok(())
    }

    /// Export surface as DRM PRIME (dma-buf).
    ///
    /// In real implementation, would call:
    /// - vaExportSurfaceHandle(display, surface, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
    ///                         flags, &descriptor)
    pub fn va_export_surface_drm_prime(
        &self,
        surface: VASurfaceID,
        flags: u32,
    ) -> Result<VADRMPRIMESurfaceDescriptor> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        // Stubbed: return simulated descriptor
        let descriptor = VADRMPRIMESurfaceDescriptor {
            fourcc: u32::from_le_bytes(*b"NV12"),
            width: 1920,
            height: 1080,
            num_objects: 1,
            objects: [
                VADRMPRIMESurfaceDescriptorObject {
                    fd: surface as i32, // Simulated FD
                    size: 1920 * 1080 * 3 / 2,
                    drm_format_modifier: 0,
                },
                VADRMPRIMESurfaceDescriptorObject::default(),
                VADRMPRIMESurfaceDescriptorObject::default(),
                VADRMPRIMESurfaceDescriptorObject::default(),
            ],
            num_layers: 2,
            layers: [
                VADRMPRIMESurfaceDescriptorLayer {
                    drm_format: u32::from_le_bytes(*b"R8  "),
                    num_planes: 1,
                    object_index: [0, 0, 0, 0],
                    offset: [0, 0, 0, 0],
                    pitch: [1920, 0, 0, 0],
                },
                VADRMPRIMESurfaceDescriptorLayer {
                    drm_format: u32::from_le_bytes(*b"GR88"),
                    num_planes: 1,
                    object_index: [0, 0, 0, 0],
                    offset: [1920 * 1080, 0, 0, 0],
                    pitch: [1920, 0, 0, 0],
                },
                VADRMPRIMESurfaceDescriptorLayer::default(),
                VADRMPRIMESurfaceDescriptorLayer::default(),
            ],
        };

        tracing::debug!(
            "Export surface {} as DRM PRIME, flags=0x{:08x}",
            surface,
            flags
        );

        Ok(descriptor)
    }

    /// Query supported config attributes.
    ///
    /// In real implementation, would call:
    /// - vaGetConfigAttributes(display, profile, entrypoint, attribs, num_attribs)
    pub fn va_get_config_attributes(
        &self,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attribs: &mut [VAConfigAttrib],
    ) -> Result<()> {
        if !self.initialized {
            return Err(HwAccelError::DeviceInit("Not initialized".to_string()));
        }

        // Stubbed: fill in simulated attribute values
        for attrib in attribs.iter_mut() {
            attrib.value = match attrib.type_ {
                VAConfigAttribType::RTFormat => {
                    va_rt_format::VA_RT_FORMAT_YUV420 | va_rt_format::VA_RT_FORMAT_YUV420_10
                }
                VAConfigAttribType::RateControl => {
                    va_rc_mode::VA_RC_CQP
                        | va_rc_mode::VA_RC_CBR
                        | va_rc_mode::VA_RC_VBR
                        | va_rc_mode::VA_RC_ICQ
                        | va_rc_mode::VA_RC_QVBR
                }
                VAConfigAttribType::EncPackedHeaders => {
                    va_enc_packed_header::VA_ENC_PACKED_HEADER_SEQUENCE
                        | va_enc_packed_header::VA_ENC_PACKED_HEADER_PICTURE
                        | va_enc_packed_header::VA_ENC_PACKED_HEADER_SLICE
                        | va_enc_packed_header::VA_ENC_PACKED_HEADER_MISC
                }
                VAConfigAttribType::EncMaxRefFrames => {
                    // Low byte = L0 refs, high byte = L1 refs (4 each)
                    0x04 | (0x04 << 16)
                }
                VAConfigAttribType::EncMaxSlices => 32,
                VAConfigAttribType::EncQualityRange => 7, // 1-7 quality levels
                VAConfigAttribType::EncROI => 8,          // 8 ROI regions
                _ => 0,
            };
        }

        tracing::debug!(
            "Query config attributes: profile={:?}, entrypoint={:?}, {} attribs",
            profile,
            entrypoint,
            attribs.len()
        );

        Ok(())
    }

    /// Get device info if available.
    pub fn device_info(&self) -> Option<&VaDrmDevice> {
        self.device_info.as_ref()
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the display handle.
    pub fn display(&self) -> VADisplay {
        self.display
    }

    /// Get config ID.
    pub fn config_id(&self) -> VAConfigID {
        self.config_id
    }

    /// Get context ID.
    pub fn context_id(&self) -> VAContextID {
        self.context_id
    }
}

impl Default for VaEncodingWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VaEncodingWorkflow {
    fn drop(&mut self) {
        if self.initialized {
            let _ = self.va_terminate();
        }
    }
}

// ============================================================================
// Original Types (preserved for compatibility)
// ============================================================================

/// VA-API profile for H.264 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaH264Profile {
    /// Constrained Baseline profile.
    ConstrainedBaseline,
    /// Main profile.
    Main,
    /// High profile.
    High,
    /// High 10 profile (10-bit).
    High10,
    /// High 4:2:2 profile.
    High422,
}

impl VaH264Profile {
    /// Get profile IDC value.
    pub fn idc(&self) -> u8 {
        match self {
            VaH264Profile::ConstrainedBaseline => 66,
            VaH264Profile::Main => 77,
            VaH264Profile::High => 100,
            VaH264Profile::High10 => 110,
            VaH264Profile::High422 => 122,
        }
    }

    /// Convert to VAProfile.
    pub fn to_va_profile(&self) -> VAProfile {
        match self {
            VaH264Profile::ConstrainedBaseline => VAProfile::H264ConstrainedBaseline,
            VaH264Profile::Main => VAProfile::H264Main,
            VaH264Profile::High => VAProfile::H264High,
            VaH264Profile::High10 => VAProfile::H264High10,
            VaH264Profile::High422 => VAProfile::H264High422,
        }
    }
}

/// VA-API profile for HEVC encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaHevcProfile {
    /// Main profile (8-bit).
    Main,
    /// Main 10 profile (10-bit).
    Main10,
    /// Main 4:4:4 profile.
    Main444,
    /// Main 4:4:4 10 profile.
    Main44410,
}

impl VaHevcProfile {
    /// Convert to VAProfile.
    pub fn to_va_profile(&self) -> VAProfile {
        match self {
            VaHevcProfile::Main => VAProfile::HEVCMain,
            VaHevcProfile::Main10 => VAProfile::HEVCMain10,
            VaHevcProfile::Main444 => VAProfile::HEVCMain444,
            VaHevcProfile::Main44410 => VAProfile::HEVCMain444_10,
        }
    }
}

/// VA-API rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaRateControl {
    /// Constant QP mode.
    Cqp,
    /// Constant Bitrate.
    Cbr,
    /// Variable Bitrate.
    Vbr,
    /// Variable Bitrate constrained.
    VbrConstrained,
    /// Quality CQP mode (Intel-specific).
    Qvbr,
    /// Intelligent Constant Quality (Intel-specific).
    Icq,
}

impl VaRateControl {
    /// Convert to VA_RC_* constant.
    pub fn to_va_rc(&self) -> u32 {
        match self {
            VaRateControl::Cqp => va_rc_mode::VA_RC_CQP,
            VaRateControl::Cbr => va_rc_mode::VA_RC_CBR,
            VaRateControl::Vbr => va_rc_mode::VA_RC_VBR,
            VaRateControl::VbrConstrained => va_rc_mode::VA_RC_VBR_CONSTRAINED,
            VaRateControl::Qvbr => va_rc_mode::VA_RC_QVBR,
            VaRateControl::Icq => va_rc_mode::VA_RC_ICQ,
        }
    }
}

/// VA surface format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaSurfaceFormat {
    /// NV12 (YUV 4:2:0, 8-bit).
    Nv12,
    /// P010 (YUV 4:2:0, 10-bit).
    P010,
    /// YUY2 (YUV 4:2:2, packed).
    Yuy2,
    /// RGBA (8-bit per component).
    Rgba,
    /// BGRA (8-bit per component).
    Bgra,
}

impl VaSurfaceFormat {
    /// Get fourcc value.
    pub fn fourcc(&self) -> u32 {
        match self {
            VaSurfaceFormat::Nv12 => u32::from_le_bytes(*b"NV12"),
            VaSurfaceFormat::P010 => u32::from_le_bytes(*b"P010"),
            VaSurfaceFormat::Yuy2 => u32::from_le_bytes(*b"YUY2"),
            VaSurfaceFormat::Rgba => u32::from_le_bytes(*b"RGBA"),
            VaSurfaceFormat::Bgra => u32::from_le_bytes(*b"BGRA"),
        }
    }

    /// Get bits per pixel.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            VaSurfaceFormat::Nv12 => 12,
            VaSurfaceFormat::P010 => 24,
            VaSurfaceFormat::Yuy2 => 16,
            VaSurfaceFormat::Rgba | VaSurfaceFormat::Bgra => 32,
        }
    }

    /// Get corresponding RT format.
    pub fn to_rt_format(&self) -> u32 {
        match self {
            VaSurfaceFormat::Nv12 => va_rt_format::VA_RT_FORMAT_YUV420,
            VaSurfaceFormat::P010 => va_rt_format::VA_RT_FORMAT_YUV420_10,
            VaSurfaceFormat::Yuy2 => va_rt_format::VA_RT_FORMAT_YUV422,
            VaSurfaceFormat::Rgba | VaSurfaceFormat::Bgra => va_rt_format::VA_RT_FORMAT_RGB32,
        }
    }
}

/// VA surface handle.
#[derive(Debug)]
pub struct VaSurface {
    /// Surface ID (would be VASurfaceID in real implementation).
    id: u32,
    /// Surface width.
    pub width: u32,
    /// Surface height.
    pub height: u32,
    /// Surface format.
    pub format: VaSurfaceFormat,
    /// Whether surface is in use.
    in_use: bool,
}

impl VaSurface {
    /// Create a new VA surface.
    fn new(id: u32, width: u32, height: u32, format: VaSurfaceFormat) -> Self {
        Self {
            id,
            width,
            height,
            format,
            in_use: false,
        }
    }

    /// Get the surface ID.
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// VA surface pool for efficient surface reuse.
pub struct VaSurfacePool {
    /// Width of surfaces in pool.
    width: u32,
    /// Height of surfaces in pool.
    height: u32,
    /// Surface format.
    format: VaSurfaceFormat,
    /// Available surfaces.
    available: Mutex<VecDeque<VaSurface>>,
    /// Maximum pool size.
    max_size: usize,
    /// Next surface ID.
    next_id: Mutex<u32>,
}

impl VaSurfacePool {
    /// Create a new surface pool.
    pub fn new(width: u32, height: u32, format: VaSurfaceFormat, max_size: usize) -> Self {
        Self {
            width,
            height,
            format,
            available: Mutex::new(VecDeque::new()),
            max_size,
            next_id: Mutex::new(0),
        }
    }

    /// Acquire a surface from the pool.
    pub fn acquire(&self) -> Result<VaSurface> {
        let mut available = self.available.lock().map_err(|_| {
            HwAccelError::DeviceError("Surface pool lock poisoned".to_string())
        })?;

        if let Some(mut surface) = available.pop_front() {
            surface.in_use = true;
            return Ok(surface);
        }

        // Allocate new surface
        let mut next_id = self.next_id.lock().map_err(|_| {
            HwAccelError::DeviceError("Surface ID lock poisoned".to_string())
        })?;

        if (*next_id as usize) >= self.max_size {
            return Err(HwAccelError::ResourceExhausted(
                "Surface pool exhausted".to_string(),
            ));
        }

        let surface = VaSurface::new(*next_id, self.width, self.height, self.format);
        *next_id += 1;
        Ok(surface)
    }

    /// Release a surface back to the pool.
    pub fn release(&self, mut surface: VaSurface) {
        surface.in_use = false;
        if let Ok(mut available) = self.available.lock() {
            available.push_back(surface);
        }
    }

    /// Get number of available surfaces.
    pub fn available_count(&self) -> usize {
        self.available.lock().map(|a| a.len()).unwrap_or(0)
    }
}

/// VA-API encoder configuration.
#[derive(Debug, Clone)]
pub struct VaEncoderConfig {
    /// Base hardware encoder config.
    pub base: crate::encoder::HwEncoderConfig,
    /// H.264 profile (if encoding H.264).
    pub h264_profile: VaH264Profile,
    /// HEVC profile (if encoding HEVC).
    pub hevc_profile: VaHevcProfile,
    /// Rate control mode.
    pub rate_control: VaRateControl,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Maximum bitrate (for VBR modes).
    pub max_bitrate: u32,
    /// Initial QP (for CQP mode).
    pub initial_qp: u32,
    /// Minimum QP.
    pub min_qp: u32,
    /// Maximum QP.
    pub max_qp: u32,
    /// GOP size.
    pub gop_size: u32,
    /// Number of B-frames.
    pub b_frames: u32,
    /// Number of reference frames.
    pub ref_frames: u32,
    /// Enable CABAC entropy coding.
    pub cabac: bool,
    /// Enable low power mode.
    pub low_power: bool,
    /// Quality preset (1-7, lower = better quality).
    pub quality_preset: u32,
    /// ICQ quality factor (for ICQ rate control).
    pub icq_quality: u32,
    /// Low-power encoding configuration.
    pub low_power_config: Option<VaLowPowerConfig>,
}

impl Default for VaEncoderConfig {
    fn default() -> Self {
        Self {
            base: crate::encoder::HwEncoderConfig::default(),
            h264_profile: VaH264Profile::High,
            hevc_profile: VaHevcProfile::Main,
            rate_control: VaRateControl::Vbr,
            bitrate: 5_000_000,
            max_bitrate: 10_000_000,
            initial_qp: 26,
            min_qp: 1,
            max_qp: 51,
            gop_size: 60,
            b_frames: 2,
            ref_frames: 4,
            cabac: true,
            low_power: false,
            quality_preset: 4,
            icq_quality: 24,
            low_power_config: None,
        }
    }
}

/// Encoded frame from VA-API encoder.
#[derive(Debug)]
pub struct VaEncodedFrame {
    /// Encoded bitstream data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Whether this is a key frame.
    pub is_keyframe: bool,
    /// Frame type.
    pub frame_type: VaFrameType,
}

/// VA frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaFrameType {
    /// I-frame (intra).
    I,
    /// P-frame (predicted).
    P,
    /// B-frame (bi-predicted).
    B,
}

/// VA-API context.
#[derive(Debug)]
pub struct VaapiContext {
    /// Device path.
    pub device_path: String,
    /// Display handle (would be VADisplay in real implementation).
    display: Option<usize>,
    /// VA version major.
    pub version_major: i32,
    /// VA version minor.
    pub version_minor: i32,
    /// Vendor string.
    pub vendor: String,
    /// DRM file descriptor.
    drm_fd: Option<i32>,
    /// Encoding workflow.
    workflow: Option<VaEncodingWorkflow>,
}

impl VaapiContext {
    /// Create a new VA-API context.
    pub fn new(device_path: &str) -> Result<Self> {
        // In a real implementation, this would:
        // 1. Open the DRM device
        // 2. Call vaGetDisplayDRM()
        // 3. Call vaInitialize()
        // 4. Query capabilities via vaQueryConfigEntrypoints()

        Ok(Self {
            device_path: device_path.to_string(),
            display: Some(1), // Simulated display handle
            version_major: 1,
            version_minor: 20,
            vendor: "Intel iHD driver - 24.1.5".to_string(),
            drm_fd: Some(3), // Simulated file descriptor
            workflow: None,
        })
    }

    /// Open default device.
    pub fn open_default() -> Result<Self> {
        // Try common device paths
        let device_paths = [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/dri/card0",
        ];

        for path in &device_paths {
            if std::path::Path::new(path).exists() {
                return Self::new(path);
            }
        }

        // Fallback for non-Linux or systems without VA-API
        Self::new("/dev/dri/renderD128")
    }

    /// Get capabilities.
    pub fn capabilities(&self) -> HwCapabilities {
        HwCapabilities {
            accel_type: crate::HwAccelType::Vaapi,
            encode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Av1],
            decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
            max_width: 4096,
            max_height: 2160,
            supports_bframes: true,
            supports_10bit: true,
            supports_hdr: true,
            device_name: self.vendor.clone(),
        }
    }

    /// Check if codec is supported for encoding.
    pub fn supports_encode(&self, codec: HwCodec) -> bool {
        matches!(codec, HwCodec::H264 | HwCodec::Hevc | HwCodec::Av1)
    }

    /// Check if codec is supported for decoding.
    pub fn supports_decode(&self, codec: HwCodec) -> bool {
        matches!(
            codec,
            HwCodec::H264 | HwCodec::Hevc | HwCodec::Vp9 | HwCodec::Av1
        )
    }

    /// Query supported entrypoints for a profile.
    pub fn query_entrypoints(&self, _profile: u32) -> Vec<u32> {
        // In real implementation, would call vaQueryConfigEntrypoints
        vec![1, 2, 5] // VAEntrypointVLD, VAEntrypointEncSlice, VAEntrypointEncSliceLP
    }

    /// Query supported rate control modes.
    pub fn query_rate_control_modes(&self) -> Vec<VaRateControl> {
        // In real implementation, would query VA_RC_* attributes
        vec![
            VaRateControl::Cqp,
            VaRateControl::Cbr,
            VaRateControl::Vbr,
            VaRateControl::VbrConstrained,
            VaRateControl::Icq,
        ]
    }

    /// Get driver info string.
    pub fn driver_info(&self) -> String {
        format!(
            "VA-API {}.{} ({})",
            self.version_major, self.version_minor, self.vendor
        )
    }

    /// Get encoding workflow (creates if not exists).
    pub fn workflow(&mut self) -> &mut VaEncodingWorkflow {
        if self.workflow.is_none() {
            let mut wf = VaEncodingWorkflow::new();
            let _ = wf.va_initialize(&self.device_path);
            self.workflow = Some(wf);
        }
        self.workflow.as_mut().unwrap()
    }
}

impl Drop for VaapiContext {
    fn drop(&mut self) {
        // In a real implementation, this would:
        // 1. Call vaTerminate()
        // 2. Close DRM file descriptor
        self.display = None;
        self.drm_fd = None;
        self.workflow = None;
    }
}

/// VA-API encoder.
pub struct VaapiEncoder {
    /// VA-API context.
    context: VaapiContext,
    /// Encoder configuration.
    config: VaEncoderConfig,
    /// Frame counter.
    frame_count: u64,
    /// Surface pool for input frames.
    surface_pool: VaSurfacePool,
    /// Reference frame surfaces.
    reference_surfaces: Vec<VaSurface>,
    /// Output buffer queue.
    output_queue: VecDeque<VaEncodedFrame>,
    /// Sequence parameter set (H.264/HEVC).
    sps: Option<Vec<u8>>,
    /// Picture parameter set (H.264/HEVC).
    pps: Option<Vec<u8>>,
    /// Video parameter set (HEVC only).
    vps: Option<Vec<u8>>,
    /// Last keyframe frame number.
    last_keyframe: u64,
}

impl VaapiEncoder {
    /// Create a new VA-API encoder.
    pub fn new(config: VaEncoderConfig) -> Result<Self> {
        let context = VaapiContext::open_default()?;

        if !context.supports_encode(config.base.codec) {
            return Err(HwAccelError::CodecNotSupported(
                config.base.codec.name().to_string(),
            ));
        }

        let surface_pool = VaSurfacePool::new(
            config.base.width,
            config.base.height,
            VaSurfaceFormat::Nv12,
            config.ref_frames as usize + config.b_frames as usize + 4,
        );

        Ok(Self {
            context,
            config,
            frame_count: 0,
            surface_pool,
            reference_surfaces: Vec::new(),
            output_queue: VecDeque::new(),
            sps: None,
            pps: None,
            vps: None,
            last_keyframe: 0,
        })
    }

    /// Initialize the encoder and generate parameter sets.
    pub fn initialize(&mut self) -> Result<()> {
        // Generate SPS
        self.sps = Some(self.generate_sps()?);

        // Generate PPS
        self.pps = Some(self.generate_pps()?);

        // Generate VPS for HEVC
        if self.config.base.codec == HwCodec::Hevc {
            self.vps = Some(self.generate_vps()?);
        }

        Ok(())
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame_data: &[u8], pts: i64) -> Result<Option<VaEncodedFrame>> {
        // Acquire input surface
        let input_surface = self.surface_pool.acquire()?;

        // Determine frame type
        let frame_type = self.determine_frame_type();
        let is_keyframe = frame_type == VaFrameType::I;

        if is_keyframe {
            self.last_keyframe = self.frame_count;
        }

        // In a real implementation, this would:
        // 1. Upload frame_data to the VA surface
        // 2. Set up VAEncPictureParameterBuffer
        // 3. Set up VAEncSliceParameterBuffer
        // 4. Call vaBeginPicture, vaRenderPicture, vaEndPicture
        // 5. Call vaSyncSurface and vaMapBuffer to retrieve encoded data

        let encoded_size = self.estimate_encoded_size(frame_data.len());
        let encoded_data = vec![0u8; encoded_size];

        let encoded_frame = VaEncodedFrame {
            data: encoded_data,
            pts,
            dts: pts - (self.config.b_frames as i64 * 1001 / 30), // Approximate DTS
            is_keyframe,
            frame_type,
        };

        // Release input surface
        self.surface_pool.release(input_surface);

        self.frame_count += 1;

        Ok(Some(encoded_frame))
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<VaEncodedFrame>> {
        let mut frames = Vec::new();
        while let Some(frame) = self.output_queue.pop_front() {
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Get the SPS.
    pub fn sps(&self) -> Option<&[u8]> {
        self.sps.as_deref()
    }

    /// Get the PPS.
    pub fn pps(&self) -> Option<&[u8]> {
        self.pps.as_deref()
    }

    /// Get the VPS (HEVC only).
    pub fn vps(&self) -> Option<&[u8]> {
        self.vps.as_deref()
    }

    /// Determine frame type based on GOP structure.
    fn determine_frame_type(&self) -> VaFrameType {
        let frame_in_gop = self.frame_count % self.config.gop_size as u64;

        if frame_in_gop == 0 {
            VaFrameType::I
        } else if self.config.b_frames > 0 {
            let b_period = self.config.b_frames as u64 + 1;
            if frame_in_gop % b_period == 0 {
                VaFrameType::P
            } else {
                VaFrameType::B
            }
        } else {
            VaFrameType::P
        }
    }

    /// Estimate encoded size.
    fn estimate_encoded_size(&self, raw_size: usize) -> usize {
        // Rough estimate: encoded is typically 1/10 to 1/50 of raw
        (raw_size / 20).max(4096)
    }

    /// Generate SPS for H.264/HEVC.
    fn generate_sps(&self) -> Result<Vec<u8>> {
        // In a real implementation, this would use vaCreateBuffer with
        // VAEncSequenceParameterBufferType and extract the SPS
        Ok(vec![0x00, 0x00, 0x00, 0x01, 0x67]) // H.264 SPS NAL header
    }

    /// Generate PPS for H.264/HEVC.
    fn generate_pps(&self) -> Result<Vec<u8>> {
        // In a real implementation, this would use vaCreateBuffer with
        // VAEncPictureParameterBufferType and extract the PPS
        Ok(vec![0x00, 0x00, 0x00, 0x01, 0x68]) // H.264 PPS NAL header
    }

    /// Generate VPS for HEVC.
    fn generate_vps(&self) -> Result<Vec<u8>> {
        Ok(vec![0x00, 0x00, 0x00, 0x01, 0x40]) // HEVC VPS NAL header
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> VaEncoderStats {
        VaEncoderStats {
            frames_encoded: self.frame_count,
            last_keyframe: self.last_keyframe,
            available_surfaces: self.surface_pool.available_count(),
        }
    }
}

/// VA-API encoder statistics.
#[derive(Debug, Clone)]
pub struct VaEncoderStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Frame number of last keyframe.
    pub last_keyframe: u64,
    /// Number of available surfaces in pool.
    pub available_surfaces: usize,
}

/// VA-API decoder.
pub struct VaapiDecoder {
    /// VA-API context.
    context: VaapiContext,
    /// Decoder configuration.
    config: crate::decoder::HwDecoderConfig,
    /// Frame counter.
    frame_count: u64,
    /// Surface pool for decoded frames.
    surface_pool: VaSurfacePool,
    /// Reference frame surfaces.
    reference_surfaces: Vec<VaSurface>,
    /// Decoded frame queue.
    output_queue: VecDeque<VaDecodedFrame>,
}

/// Decoded frame from VA-API decoder.
#[derive(Debug)]
pub struct VaDecodedFrame {
    /// Decoded frame data (NV12 or P010).
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Presentation timestamp.
    pub pts: i64,
    /// Surface format.
    pub format: VaSurfaceFormat,
}

impl VaapiDecoder {
    /// Create a new VA-API decoder.
    pub fn new(config: crate::decoder::HwDecoderConfig) -> Result<Self> {
        let context = VaapiContext::open_default()?;

        if !context.supports_decode(config.codec) {
            return Err(HwAccelError::CodecNotSupported(
                config.codec.name().to_string(),
            ));
        }

        let surface_pool = VaSurfacePool::new(
            config.width,
            config.height,
            VaSurfaceFormat::Nv12,
            16, // Typical decoder surface pool size
        );

        Ok(Self {
            context,
            config,
            frame_count: 0,
            surface_pool,
            reference_surfaces: Vec::new(),
            output_queue: VecDeque::new(),
        })
    }

    /// Decode a packet.
    pub fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Option<VaDecodedFrame>> {
        // Acquire output surface
        let output_surface = self.surface_pool.acquire()?;

        // In a real implementation, this would:
        // 1. Parse the bitstream to extract slice/picture parameters
        // 2. Create VASliceParameterBuffer and VASliceDataBuffer
        // 3. Call vaBeginPicture, vaRenderPicture, vaEndPicture
        // 4. Call vaSyncSurface
        // 5. Map surface and copy decoded data

        let frame_size = (self.config.width * self.config.height * 3 / 2) as usize;
        let decoded_data = vec![0u8; frame_size];

        let decoded_frame = VaDecodedFrame {
            data: decoded_data,
            width: self.config.width,
            height: self.config.height,
            pts,
            format: VaSurfaceFormat::Nv12,
        };

        // Release surface back to pool
        self.surface_pool.release(output_surface);

        self.frame_count += 1;

        // Check if packet is not empty to determine if we should output
        if !packet.is_empty() {
            Ok(Some(decoded_frame))
        } else {
            Ok(None)
        }
    }

    /// Flush remaining decoded frames.
    pub fn flush(&mut self) -> Result<Vec<VaDecodedFrame>> {
        let mut frames = Vec::new();
        while let Some(frame) = self.output_queue.pop_front() {
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Get decoder statistics.
    pub fn stats(&self) -> VaDecoderStats {
        VaDecoderStats {
            frames_decoded: self.frame_count,
            available_surfaces: self.surface_pool.available_count(),
        }
    }
}

/// VA-API decoder statistics.
#[derive(Debug, Clone)]
pub struct VaDecoderStats {
    /// Total frames decoded.
    pub frames_decoded: u64,
    /// Number of available surfaces in pool.
    pub available_surfaces: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vaapi_context() {
        let ctx = VaapiContext::new("/dev/dri/renderD128").unwrap();
        assert_eq!(ctx.device_path, "/dev/dri/renderD128");
        assert!(ctx.version_major >= 1);
    }

    #[test]
    fn test_vaapi_capabilities() {
        let ctx = VaapiContext::new("/dev/dri/renderD128").unwrap();
        let caps = ctx.capabilities();
        assert!(caps.supports_bframes);
        assert!(caps.max_width >= 1920);
    }

    #[test]
    fn test_va_surface_format() {
        assert_eq!(VaSurfaceFormat::Nv12.fourcc(), u32::from_le_bytes(*b"NV12"));
        assert_eq!(VaSurfaceFormat::Nv12.bits_per_pixel(), 12);
        assert_eq!(VaSurfaceFormat::P010.bits_per_pixel(), 24);
    }

    #[test]
    fn test_va_h264_profile() {
        assert_eq!(VaH264Profile::High.idc(), 100);
        assert_eq!(VaH264Profile::Main.idc(), 77);
    }

    #[test]
    fn test_va_surface_pool() {
        let pool = VaSurfacePool::new(1920, 1080, VaSurfaceFormat::Nv12, 4);

        let surface1 = pool.acquire().unwrap();
        assert_eq!(surface1.width, 1920);
        assert_eq!(surface1.height, 1080);

        let surface2 = pool.acquire().unwrap();
        assert_ne!(surface1.id(), surface2.id());

        pool.release(surface1);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_va_encoder_config() {
        let config = VaEncoderConfig::default();
        assert_eq!(config.h264_profile, VaH264Profile::High);
        assert!(config.cabac);
        assert_eq!(config.b_frames, 2);
    }

    #[test]
    fn test_va_encoder_creation() {
        let config = VaEncoderConfig {
            base: crate::encoder::HwEncoderConfig {
                codec: HwCodec::H264,
                width: 1920,
                height: 1080,
                ..Default::default()
            },
            ..Default::default()
        };

        let encoder = VaapiEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_va_decoder_creation() {
        let config = crate::decoder::HwDecoderConfig {
            codec: HwCodec::H264,
            width: 1920,
            height: 1080,
            ..Default::default()
        };

        let decoder = VaapiDecoder::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_rate_control_modes() {
        let ctx = VaapiContext::new("/dev/dri/renderD128").unwrap();
        let modes = ctx.query_rate_control_modes();
        assert!(!modes.is_empty());
        assert!(modes.contains(&VaRateControl::Vbr));
    }

    // New tests for FFI types

    #[test]
    fn test_va_status_codes() {
        use va_status::*;
        assert_eq!(VA_STATUS_SUCCESS, 0);
        assert_eq!(va_status_string(VA_STATUS_SUCCESS), "success");
        assert_eq!(va_status_string(VA_STATUS_ERROR_ENCODING_ERROR), "encoding error");
    }

    #[test]
    fn test_va_profile_enum() {
        assert!(VAProfile::H264High.is_h264());
        assert!(VAProfile::HEVCMain.is_hevc());
        assert!(VAProfile::AV1Profile0.is_av1());
        assert_eq!(VAProfile::H264High.h264_idc(), Some(100));
    }

    #[test]
    fn test_va_entrypoint_enum() {
        assert!(VAEntrypoint::VLD.is_decode());
        assert!(VAEntrypoint::EncSlice.is_encode());
        assert!(VAEntrypoint::EncSliceLP.is_low_power());
        assert!(VAEntrypoint::VideoProc.is_video_proc());
    }

    #[test]
    fn test_va_rt_formats() {
        use va_rt_format::*;
        assert_eq!(format_string(VA_RT_FORMAT_YUV420), "YUV420");
        assert_eq!(format_string(VA_RT_FORMAT_YUV420_10), "YUV420_10");
    }

    #[test]
    fn test_va_rc_modes() {
        assert_eq!(VaRateControl::Cqp.to_va_rc(), va_rc_mode::VA_RC_CQP);
        assert_eq!(VaRateControl::Vbr.to_va_rc(), va_rc_mode::VA_RC_VBR);
        assert_eq!(VaRateControl::Icq.to_va_rc(), va_rc_mode::VA_RC_ICQ);
    }

    #[test]
    fn test_low_power_config() {
        let max_speed = VaLowPowerConfig::max_speed();
        assert!(max_speed.enabled);
        assert_eq!(max_speed.quality_level, 7);

        let balanced = VaLowPowerConfig::balanced();
        assert_eq!(balanced.quality_level, 4);

        let max_quality = VaLowPowerConfig::max_quality();
        assert_eq!(max_quality.quality_level, 1);
    }

    #[test]
    fn test_quality_speed_mode() {
        assert_eq!(VaQualitySpeedMode::Quality.quality_level(), 1);
        assert_eq!(VaQualitySpeedMode::Balanced.quality_level(), 4);
        assert_eq!(VaQualitySpeedMode::Speed.quality_level(), 7);
        assert_eq!(VaQualitySpeedMode::Custom(3).quality_level(), 3);
        assert_eq!(VaQualitySpeedMode::Custom(10).quality_level(), 7); // Clamped
    }

    #[test]
    fn test_device_enumerator() {
        let enumerator = VaDeviceEnumerator::new();
        assert!(enumerator.device_count() > 0);

        let devices = enumerator.devices();
        assert!(!devices.is_empty());

        // First device should be Intel (simulated)
        let first = &devices[0];
        assert!(first.vendor_name.contains("Intel") || first.vendor_name.contains("AMD"));
    }

    #[test]
    fn test_encoding_workflow() {
        let mut workflow = VaEncodingWorkflow::new();
        assert!(!workflow.is_initialized());

        let result = workflow.va_initialize("/dev/dri/renderD128");
        assert!(result.is_ok());
        assert!(workflow.is_initialized());

        let config_result = workflow.va_create_config(
            VAProfile::H264High,
            VAEntrypoint::EncSlice,
            &[],
        );
        assert!(config_result.is_ok());

        let surfaces = workflow.va_create_surfaces(
            va_rt_format::VA_RT_FORMAT_YUV420,
            1920,
            1080,
            4,
            None,
        );
        assert!(surfaces.is_ok());
        let surfaces = surfaces.unwrap();
        assert_eq!(surfaces.len(), 4);

        let context = workflow.va_create_context(1920, 1080, &surfaces);
        assert!(context.is_ok());

        let buffer = workflow.va_create_buffer(
            VABufferType::EncSequenceParameterBufferType,
            1024,
            1,
            None,
        );
        assert!(buffer.is_ok());

        // Test picture encoding workflow
        assert!(workflow.va_begin_picture(surfaces[0]).is_ok());
        assert!(workflow.va_render_picture(&[buffer.unwrap()]).is_ok());
        assert!(workflow.va_end_picture().is_ok());

        assert!(workflow.va_sync_surface(surfaces[0]).is_ok());

        // Test surface export
        let descriptor = workflow.va_export_surface_drm_prime(
            surfaces[0],
            va_surface_export_flags::VA_EXPORT_SURFACE_READ_ONLY,
        );
        assert!(descriptor.is_ok());

        // Cleanup
        assert!(workflow.va_terminate().is_ok());
        assert!(!workflow.is_initialized());
    }

    #[test]
    fn test_drm_prime_descriptor() {
        let descriptor = VADRMPRIMESurfaceDescriptor::default();
        assert_eq!(descriptor.num_objects, 0);
        assert_eq!(descriptor.num_layers, 0);
    }

    #[test]
    fn test_h264_parameter_buffers() {
        let seq = VAEncSequenceParameterBufferH264::default();
        assert_eq!(seq.seq_parameter_set_id, 0);

        let pic = VAEncPictureParameterBufferH264::default();
        assert_eq!(pic.pic_parameter_set_id, 0);

        let slice = VAEncSliceParameterBufferH264::default();
        assert_eq!(slice.macroblock_address, 0);
    }

    #[test]
    fn test_hevc_parameter_buffers() {
        let seq = VAEncSequenceParameterBufferHEVC::default();
        assert_eq!(seq.general_profile_idc, 0);

        let pic = VAEncPictureParameterBufferHEVC::default();
        assert_eq!(pic.pic_parameter_set_id, 0);
    }

    #[test]
    fn test_av1_parameter_buffers() {
        let seq = VAEncSequenceParameterBufferAV1::default();
        assert_eq!(seq.seq_profile, 0);

        let pic = VAEncPictureParameterBufferAV1::default();
        assert_eq!(pic.frame_width_minus_1, 0);
    }

    #[test]
    fn test_surface_attributes() {
        let attrib = VASurfaceAttrib::default();
        assert_eq!(attrib.type_, VASurfaceAttribType::None);
        assert_eq!(attrib.flags, 0);
    }

    #[test]
    fn test_config_attributes() {
        let mut attribs = vec![
            VAConfigAttrib {
                type_: VAConfigAttribType::RTFormat,
                value: 0,
            },
            VAConfigAttrib {
                type_: VAConfigAttribType::RateControl,
                value: 0,
            },
        ];

        let mut workflow = VaEncodingWorkflow::new();
        let _ = workflow.va_initialize("/dev/dri/renderD128");

        let result = workflow.va_get_config_attributes(
            VAProfile::H264High,
            VAEntrypoint::EncSlice,
            &mut attribs,
        );
        assert!(result.is_ok());
        assert!(attribs[0].value & va_rt_format::VA_RT_FORMAT_YUV420 != 0);
    }

    #[test]
    fn test_profile_conversions() {
        assert_eq!(
            VaH264Profile::High.to_va_profile(),
            VAProfile::H264High
        );
        assert_eq!(
            VaHevcProfile::Main.to_va_profile(),
            VAProfile::HEVCMain
        );
    }

    #[test]
    fn test_surface_format_rt_format() {
        assert_eq!(
            VaSurfaceFormat::Nv12.to_rt_format(),
            va_rt_format::VA_RT_FORMAT_YUV420
        );
        assert_eq!(
            VaSurfaceFormat::P010.to_rt_format(),
            va_rt_format::VA_RT_FORMAT_YUV420_10
        );
    }
}
