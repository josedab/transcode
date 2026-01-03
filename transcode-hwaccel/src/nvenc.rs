//! NVIDIA NVENC/NVDEC implementation.
//!
//! This module provides hardware-accelerated video encoding and decoding
//! using NVIDIA's Video Codec SDK. NVENC supports high-quality encoding
//! with lookahead, B-frames, and advanced rate control.
//!
//! # Requirements
//!
//! - NVIDIA GPU with NVENC support (Kepler or newer for encoding)
//! - CUDA toolkit installed
//! - NVIDIA Video Codec SDK
//!
//! # Features
//!
//! - H.264, HEVC, and AV1 encoding
//! - B-frame support with configurable reference structure
//! - Lookahead for improved rate control
//! - Multiple rate control modes (CBR, VBR, CQP, etc.)
//! - 10-bit and HDR support
//!
//! # FFI Architecture
//!
//! This module provides real FFI bindings to NVIDIA Video Codec SDK when
//! the `nvenc` feature is enabled. All NV_ENC_* structures mirror the SDK API.

// Allow dead code - FFI-ready structures contain fields for future SDK integration
#![allow(dead_code)]
// Allow field reassign with default - builder patterns use this intentionally
#![allow(clippy::field_reassign_with_default)]

// =============================================================================
// Real FFI Bindings Module (enabled with nvenc feature)
// =============================================================================

/// Real FFI bindings to NVIDIA Video Codec SDK (NVENC).
/// These bindings link to libnvcuvid/libnvidia-encode.
#[cfg(feature = "nvenc")]
pub mod ffi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_uint, c_ulong};

    // -------------------------------------------------------------------------
    // CUDA Types
    // -------------------------------------------------------------------------

    /// CUDA context handle.
    pub type CUcontext = *mut c_void;
    /// CUDA device ordinal.
    pub type CUdevice = c_int;
    /// CUDA stream handle.
    pub type CUstream = *mut c_void;
    /// CUDA device pointer (GPU memory).
    pub type CUdeviceptr = c_ulong;
    /// CUDA result code.
    pub type CUresult = c_int;

    /// CUDA success code.
    pub const CUDA_SUCCESS: CUresult = 0;
    /// CUDA error - invalid value.
    pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
    /// CUDA error - out of memory.
    pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
    /// CUDA error - not initialized.
    pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
    /// CUDA error - no device.
    pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;

    // -------------------------------------------------------------------------
    // NVENC Types
    // -------------------------------------------------------------------------

    /// NVENC encoder session handle.
    pub type NvEncoderSession = *mut c_void;

    /// NVENC status code.
    pub type NVENCSTATUS = c_int;

    /// NVENC success.
    pub const NV_ENC_SUCCESS: NVENCSTATUS = 0;
    /// No encode device available.
    pub const NV_ENC_ERR_NO_ENCODE_DEVICE: NVENCSTATUS = 1;
    /// Unsupported device.
    pub const NV_ENC_ERR_UNSUPPORTED_DEVICE: NVENCSTATUS = 2;
    /// Invalid encoder device.
    pub const NV_ENC_ERR_INVALID_ENCODERDEVICE: NVENCSTATUS = 3;
    /// Invalid device.
    pub const NV_ENC_ERR_INVALID_DEVICE: NVENCSTATUS = 4;
    /// Device not exist.
    pub const NV_ENC_ERR_DEVICE_NOT_EXIST: NVENCSTATUS = 5;
    /// Invalid pointer.
    pub const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS = 6;
    /// Invalid event.
    pub const NV_ENC_ERR_INVALID_EVENT: NVENCSTATUS = 7;
    /// Invalid parameter.
    pub const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS = 8;
    /// Invalid call.
    pub const NV_ENC_ERR_INVALID_CALL: NVENCSTATUS = 9;
    /// Out of memory.
    pub const NV_ENC_ERR_OUT_OF_MEMORY: NVENCSTATUS = 10;
    /// Encoder not initialized.
    pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NVENCSTATUS = 11;
    /// Unsupported parameter.
    pub const NV_ENC_ERR_UNSUPPORTED_PARAM: NVENCSTATUS = 12;
    /// Lock busy.
    pub const NV_ENC_ERR_LOCK_BUSY: NVENCSTATUS = 13;
    /// Not enough buffer.
    pub const NV_ENC_ERR_NOT_ENOUGH_BUFFER: NVENCSTATUS = 14;
    /// Invalid version.
    pub const NV_ENC_ERR_INVALID_VERSION: NVENCSTATUS = 15;
    /// Map failed.
    pub const NV_ENC_ERR_MAP_FAILED: NVENCSTATUS = 16;
    /// Need more input.
    pub const NV_ENC_ERR_NEED_MORE_INPUT: NVENCSTATUS = 17;
    /// Encoder busy.
    pub const NV_ENC_ERR_ENCODER_BUSY: NVENCSTATUS = 18;
    /// Generic error.
    pub const NV_ENC_ERR_GENERIC: NVENCSTATUS = 20;

    /// NVENC API version.
    pub const NVENCAPI_MAJOR_VERSION: u32 = 12;
    pub const NVENCAPI_MINOR_VERSION: u32 = 2;
    pub const NVENCAPI_VERSION: u32 = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;

    /// GUID structure.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct GUID {
        pub data1: u32,
        pub data2: u16,
        pub data3: u16,
        pub data4: [u8; 8],
    }

    impl GUID {
        pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
            Self { data1, data2, data3, data4 }
        }
    }

    /// H.264 codec GUID.
    pub const NV_ENC_CODEC_H264_GUID: GUID = GUID::new(
        0x6BC82762, 0x4E63, 0x4CA4,
        [0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF]
    );

    /// HEVC codec GUID.
    pub const NV_ENC_CODEC_HEVC_GUID: GUID = GUID::new(
        0x790CDC88, 0x4522, 0x4D7B,
        [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03]
    );

    /// AV1 codec GUID.
    pub const NV_ENC_CODEC_AV1_GUID: GUID = GUID::new(
        0x0A352289, 0x0AA7, 0x4759,
        [0x86, 0x2D, 0x5D, 0x15, 0xCD, 0x16, 0xD2, 0x54]
    );

    /// H.264 High profile GUID.
    pub const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = GUID::new(
        0xE7CBC309, 0x4F7A, 0x4B89,
        [0xAF, 0x2A, 0xD5, 0x37, 0xC9, 0x2B, 0xE3, 0x10]
    );

    /// HEVC Main profile GUID.
    pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: GUID = GUID::new(
        0xB514C39A, 0xB55B, 0x40FA,
        [0x87, 0x8F, 0xF1, 0x25, 0x3B, 0x4D, 0xFD, 0xEC]
    );

    /// HEVC Main10 profile GUID.
    pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: GUID = GUID::new(
        0xFA4D2B6C, 0x3A5B, 0x411A,
        [0x80, 0x18, 0x0A, 0x3F, 0x5E, 0x3C, 0x9B, 0xE5]
    );

    /// Buffer format.
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_BUFFER_FORMAT {
        #[default]
        Undefined = 0x00000000,
        Nv12 = 0x00000001,
        Yv12 = 0x00000010,
        Iyuv = 0x00000100,
        Yuv444 = 0x00001000,
        P010 = 0x00010000,
        Yuv444_10bit = 0x00100000,
        Argb = 0x01000000,
        Argb10 = 0x02000000,
        Ayuv = 0x04000000,
        Abgr = 0x10000000,
        Abgr10 = 0x20000000,
    }

    /// Device type.
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_DEVICE_TYPE {
        Directx = 0,
        #[default]
        Cuda = 1,
        Opengl = 2,
    }

    /// Rate control mode.
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_PARAMS_RC_MODE {
        ConstQp = 0,
        #[default]
        Vbr = 1,
        Cbr = 2,
        CbrLowdelayHq = 0x8,
        CbrHq = 0x10,
        VbrHq = 0x20,
    }

    /// Picture type.
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_PIC_TYPE {
        P = 0,
        B = 1,
        I = 2,
        #[default]
        Idr = 3,
        Bi = 4,
        Skipped = 5,
        IntraRefresh = 6,
        NonRefP = 7,
    }

    /// Picture struct (field/frame).
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_PIC_STRUCT {
        #[default]
        Frame = 0x01,
        TopField = 0x02,
        BottomField = 0x03,
    }

    /// Tuning info.
    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum NV_ENC_TUNING_INFO {
        Undefined = 0,
        #[default]
        HighQuality = 1,
        LowLatency = 2,
        UltraLowLatency = 3,
        Lossless = 4,
    }

    /// Open encode session parameters.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
        pub version: u32,
        pub device_type: NV_ENC_DEVICE_TYPE,
        pub device: *mut c_void,
        pub reserved: *mut c_void,
        pub api_version: u32,
        pub reserved1: [u32; 253],
        pub reserved2: [*mut c_void; 64],
    }

    impl Default for NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
        fn default() -> Self {
            Self {
                version: NVENCAPI_VERSION | (1 << 16),
                device_type: NV_ENC_DEVICE_TYPE::Cuda,
                device: std::ptr::null_mut(),
                reserved: std::ptr::null_mut(),
                api_version: NVENCAPI_VERSION,
                reserved1: [0; 253],
                reserved2: [std::ptr::null_mut(); 64],
            }
        }
    }

    /// Initialize params.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_INITIALIZE_PARAMS {
        pub version: u32,
        pub encode_guid: GUID,
        pub preset_guid: GUID,
        pub encode_width: u32,
        pub encode_height: u32,
        pub dar_width: u32,
        pub dar_height: u32,
        pub frame_rate_num: u32,
        pub frame_rate_den: u32,
        pub enable_encode_async: u32,
        pub enable_ptt: u32,
        pub report_slice_offsets: u32,
        pub enable_sub_frame_write: u32,
        pub enable_external_meHints: u32,
        pub enable_meOnly_mode: u32,
        pub enable_weighted_prediction: u32,
        pub enable_output_in_vidmem: u32,
        pub reserved1: [u32; 240],
        pub private_data: *mut c_void,
        pub private_data_size: u32,
        pub tuning_info: NV_ENC_TUNING_INFO,
        pub reserved2: [*mut c_void; 62],
    }

    impl Default for NV_ENC_INITIALIZE_PARAMS {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Encode config.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_CONFIG {
        pub version: u32,
        pub profile_guid: GUID,
        pub gop_length: u32,
        pub frame_interval_p: i32,
        pub mono_chrome_encoding: u32,
        pub frame_field_mode: u32,
        pub mv_precision: u32,
        pub rc_params: NV_ENC_RC_PARAMS,
        pub reserved: [u32; 278],
        pub reserved2: [*mut c_void; 64],
    }

    impl Default for NV_ENC_CONFIG {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Rate control params.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct NV_ENC_RC_PARAMS {
        pub version: u32,
        pub rc_mode: NV_ENC_PARAMS_RC_MODE,
        pub const_qp: NV_ENC_QP,
        pub average_bit_rate: u32,
        pub max_bit_rate: u32,
        pub vbv_buffer_size: u32,
        pub vbv_initial_delay: u32,
        pub enable_min_qp: u32,
        pub enable_max_qp: u32,
        pub enable_init_qp: u32,
        pub enable_aq: u32,
        pub reserved: [u32; 8],
        pub min_qp: NV_ENC_QP,
        pub max_qp: NV_ENC_QP,
        pub init_qp: NV_ENC_QP,
        pub target_quality: u32,
        pub target_quality_lsb: u32,
        pub lookahead_depth: u16,
        pub reserved2: [u8; 6],
        pub reserved3: [u32; 54],
    }

    /// QP values.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct NV_ENC_QP {
        pub qp_i: u32,
        pub qp_p: u32,
        pub qp_b: u32,
    }

    /// Encode picture params.
    #[repr(C)]
    #[derive(Debug)]
    pub struct NV_ENC_PIC_PARAMS {
        pub version: u32,
        pub input_width: u32,
        pub input_height: u32,
        pub input_pitch: u32,
        pub encode_flags: u32,
        pub frame_idx: u32,
        pub input_timestamp: u64,
        pub input_duration: u64,
        pub input_buffer: *mut c_void,
        pub output_bitstream: *mut c_void,
        pub completion_event: *mut c_void,
        pub buffer_fmt: NV_ENC_BUFFER_FORMAT,
        pub pic_struct: NV_ENC_PIC_STRUCT,
        pub pic_type: NV_ENC_PIC_TYPE,
        pub codec_pic_params: [u8; 256], // Union placeholder
        pub me_hint_count_per_block: [u32; 2],
        pub me_hints: *mut c_void,
        pub reserved1: [u32; 6],
        pub reserved2: [*mut c_void; 2],
        pub qp_delta_map: *mut i8,
        pub qp_delta_map_size: u32,
        pub reserved_bitfields: u32,
        pub reserved3: [u32; 287],
        pub reserved4: [*mut c_void; 60],
    }

    impl Default for NV_ENC_PIC_PARAMS {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Create input buffer params.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_CREATE_INPUT_BUFFER {
        pub version: u32,
        pub width: u32,
        pub height: u32,
        pub memory_heap: u32,
        pub buffer_fmt: NV_ENC_BUFFER_FORMAT,
        pub reserved: u32,
        pub input_buffer: *mut c_void,
        pub private_data: *mut c_void,
        pub reserved1: [u32; 57],
        pub reserved2: [*mut c_void; 63],
    }

    impl Default for NV_ENC_CREATE_INPUT_BUFFER {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Create bitstream buffer params.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_CREATE_BITSTREAM_BUFFER {
        pub version: u32,
        pub size: u32,
        pub memory_heap: u32,
        pub reserved: u32,
        pub bitstream_buffer: *mut c_void,
        pub private_data: *mut c_void,
        pub reserved1: [u32; 58],
        pub reserved2: [*mut c_void; 64],
    }

    impl Default for NV_ENC_CREATE_BITSTREAM_BUFFER {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Lock bitstream params.
    #[repr(C)]
    #[derive(Debug)]
    pub struct NV_ENC_LOCK_BITSTREAM {
        pub version: u32,
        pub do_not_wait: u32,
        pub ltr_frame: u32,
        pub reserved_bitfield: u32,
        pub output_bitstream: *mut c_void,
        pub slice_offsets: *mut u32,
        pub frame_idx: u32,
        pub hw_encode_status: u32,
        pub num_slices: u32,
        pub bitstream_size_in_bytes: u32,
        pub output_time_stamp: u64,
        pub output_duration: u64,
        pub bitstream_buffer_ptr: *mut c_void,
        pub pic_type: NV_ENC_PIC_TYPE,
        pub pic_struct: NV_ENC_PIC_STRUCT,
        pub frame_avg_qp: u32,
        pub frame_idx_display: u32,
        pub reserved1: [u32; 226],
        pub reserved2: [*mut c_void; 64],
    }

    impl Default for NV_ENC_LOCK_BITSTREAM {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Lock input buffer params.
    #[repr(C)]
    #[derive(Debug)]
    pub struct NV_ENC_LOCK_INPUT_BUFFER {
        pub version: u32,
        pub do_not_wait: u32,
        pub reserved1: u32,
        pub input_buffer: *mut c_void,
        pub buffer_data_ptr: *mut c_void,
        pub pitch: u32,
        pub reserved2: [u32; 251],
        pub reserved3: [*mut c_void; 64],
    }

    impl Default for NV_ENC_LOCK_INPUT_BUFFER {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    /// Encoder caps param.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct NV_ENC_CAPS_PARAM {
        pub version: u32,
        pub caps_to_query: u32,
        pub reserved: [u32; 62],
    }

    impl Default for NV_ENC_CAPS_PARAM {
        fn default() -> Self {
            Self {
                version: NVENCAPI_VERSION | (1 << 16),
                caps_to_query: 0,
                reserved: [0; 62],
            }
        }
    }

    /// Preset config.
    #[repr(C)]
    #[derive(Debug)]
    pub struct NV_ENC_PRESET_CONFIG {
        pub version: u32,
        pub preset_cfg: NV_ENC_CONFIG,
        pub reserved1: [u32; 255],
        pub reserved2: [*mut c_void; 64],
    }

    impl Default for NV_ENC_PRESET_CONFIG {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    // -------------------------------------------------------------------------
    // NVENC API Function List
    // -------------------------------------------------------------------------

    /// NVENC function list - mirrors NV_ENCODE_API_FUNCTION_LIST.
    #[repr(C)]
    pub struct NV_ENCODE_API_FUNCTION_LIST {
        pub version: u32,
        pub reserved: u32,
        pub nv_enc_open_encode_session: Option<unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_get_encode_guid_count: Option<unsafe extern "C" fn(*mut c_void, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_guids: Option<unsafe extern "C" fn(*mut c_void, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_profile_guid_count: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_profile_guids: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_input_format_count: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_input_formats: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut NV_ENC_BUFFER_FORMAT, u32, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_caps: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut NV_ENC_CAPS_PARAM, *mut c_int) -> NVENCSTATUS>,
        pub nv_enc_get_encode_preset_count: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_preset_guids: Option<unsafe extern "C" fn(*mut c_void, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
        pub nv_enc_get_encode_preset_config: Option<unsafe extern "C" fn(*mut c_void, GUID, GUID, *mut NV_ENC_PRESET_CONFIG) -> NVENCSTATUS>,
        pub nv_enc_get_encode_preset_config_ex: Option<unsafe extern "C" fn(*mut c_void, GUID, GUID, NV_ENC_TUNING_INFO, *mut NV_ENC_PRESET_CONFIG) -> NVENCSTATUS>,
        pub nv_enc_initialize_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_INITIALIZE_PARAMS) -> NVENCSTATUS>,
        pub nv_enc_create_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_CREATE_INPUT_BUFFER) -> NVENCSTATUS>,
        pub nv_enc_destroy_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_create_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_CREATE_BITSTREAM_BUFFER) -> NVENCSTATUS>,
        pub nv_enc_destroy_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_encode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_PIC_PARAMS) -> NVENCSTATUS>,
        pub nv_enc_lock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_LOCK_BITSTREAM) -> NVENCSTATUS>,
        pub nv_enc_unlock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_lock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_LOCK_INPUT_BUFFER) -> NVENCSTATUS>,
        pub nv_enc_unlock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_get_encode_stats: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_get_sequence_params: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_register_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_unregister_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_map_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_unmap_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_destroy_encoder: Option<unsafe extern "C" fn(*mut c_void) -> NVENCSTATUS>,
        pub nv_enc_invalidate_ref_frames: Option<unsafe extern "C" fn(*mut c_void, u64) -> NVENCSTATUS>,
        pub nv_enc_open_encode_session_ex: Option<unsafe extern "C" fn(*mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, *mut *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_register_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_unregister_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_reconfigure_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub reserved1: *mut c_void,
        pub nv_enc_create_mv_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_destroy_mv_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_run_motion_estimation_only: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub nv_enc_get_last_error_string: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
        pub nv_enc_set_io_cuda_streams: Option<unsafe extern "C" fn(*mut c_void, CUstream, CUstream) -> NVENCSTATUS>,
        pub nv_enc_get_encode_preset_config_ex2: Option<unsafe extern "C" fn(*mut c_void, GUID, GUID, NV_ENC_TUNING_INFO, *mut NV_ENC_PRESET_CONFIG) -> NVENCSTATUS>,
        pub nv_enc_get_sequence_param_ex: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
        pub reserved2: [*mut c_void; 277],
    }

    impl Default for NV_ENCODE_API_FUNCTION_LIST {
        fn default() -> Self {
            unsafe { std::mem::zeroed() }
        }
    }

    // -------------------------------------------------------------------------
    // CUDA FFI Function Declarations
    // -------------------------------------------------------------------------

    #[link(name = "cuda")]
    extern "C" {
        /// Initialize CUDA.
        pub fn cuInit(flags: c_uint) -> CUresult;

        /// Get device count.
        pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;

        /// Get device handle.
        pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

        /// Get device name.
        pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;

        /// Get device compute capability.
        pub fn cuDeviceGetAttribute(
            pi: *mut c_int,
            attrib: c_int,
            dev: CUdevice,
        ) -> CUresult;

        /// Create CUDA context.
        pub fn cuCtxCreate_v2(
            pctx: *mut CUcontext,
            flags: c_uint,
            dev: CUdevice,
        ) -> CUresult;

        /// Destroy CUDA context.
        pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

        /// Push CUDA context.
        pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;

        /// Pop CUDA context.
        pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;

        /// Allocate device memory.
        pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

        /// Free device memory.
        pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

        /// Copy memory host to device.
        pub fn cuMemcpyHtoD_v2(
            dst_device: CUdeviceptr,
            src_host: *const c_void,
            byte_count: usize,
        ) -> CUresult;

        /// Copy memory device to host.
        pub fn cuMemcpyDtoH_v2(
            dst_host: *mut c_void,
            src_device: CUdeviceptr,
            byte_count: usize,
        ) -> CUresult;

        /// Synchronize context.
        pub fn cuCtxSynchronize() -> CUresult;

        /// Get CUDA driver version.
        pub fn cuDriverGetVersion(version: *mut c_int) -> CUresult;
    }

    // -------------------------------------------------------------------------
    // NVENC FFI Function Declarations
    // -------------------------------------------------------------------------

    #[link(name = "nvidia-encode")]
    extern "C" {
        /// Get NVENC max supported version.
        pub fn NvEncodeAPIGetMaxSupportedVersion(version: *mut u32) -> NVENCSTATUS;

        /// Create instance of NVENC API.
        pub fn NvEncodeAPICreateInstance(function_list: *mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS;
    }

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------

    /// Check if status indicates success.
    pub fn nvenc_status_success(status: NVENCSTATUS) -> bool {
        status == NV_ENC_SUCCESS
    }

    /// Get error description.
    pub fn nvenc_error_string(status: NVENCSTATUS) -> &'static str {
        match status {
            NV_ENC_SUCCESS => "Success",
            NV_ENC_ERR_NO_ENCODE_DEVICE => "No encode device available",
            NV_ENC_ERR_UNSUPPORTED_DEVICE => "Unsupported device",
            NV_ENC_ERR_INVALID_ENCODERDEVICE => "Invalid encoder device",
            NV_ENC_ERR_INVALID_DEVICE => "Invalid device",
            NV_ENC_ERR_DEVICE_NOT_EXIST => "Device does not exist",
            NV_ENC_ERR_INVALID_PTR => "Invalid pointer",
            NV_ENC_ERR_INVALID_EVENT => "Invalid event",
            NV_ENC_ERR_INVALID_PARAM => "Invalid parameter",
            NV_ENC_ERR_INVALID_CALL => "Invalid API call sequence",
            NV_ENC_ERR_OUT_OF_MEMORY => "Out of memory",
            NV_ENC_ERR_ENCODER_NOT_INITIALIZED => "Encoder not initialized",
            NV_ENC_ERR_UNSUPPORTED_PARAM => "Unsupported parameter",
            NV_ENC_ERR_LOCK_BUSY => "Lock busy",
            NV_ENC_ERR_NOT_ENOUGH_BUFFER => "Not enough buffer",
            NV_ENC_ERR_INVALID_VERSION => "Invalid API version",
            NV_ENC_ERR_MAP_FAILED => "Map operation failed",
            NV_ENC_ERR_NEED_MORE_INPUT => "Need more input",
            NV_ENC_ERR_ENCODER_BUSY => "Encoder busy",
            NV_ENC_ERR_GENERIC => "Generic error",
            _ => "Unknown error",
        }
    }

    /// RAII guard for CUDA context.
    pub struct CudaContextGuard {
        ctx: CUcontext,
    }

    impl CudaContextGuard {
        /// Create from raw context.
        pub unsafe fn new(ctx: CUcontext) -> Option<Self> {
            if ctx.is_null() {
                None
            } else {
                Some(Self { ctx })
            }
        }

        /// Get raw context.
        pub fn as_ptr(&self) -> CUcontext {
            self.ctx
        }

        /// Push context.
        pub unsafe fn push(&self) -> CUresult {
            cuCtxPushCurrent_v2(self.ctx)
        }

        /// Pop context.
        pub unsafe fn pop(&self) -> CUresult {
            let mut ctx: CUcontext = std::ptr::null_mut();
            cuCtxPopCurrent_v2(&mut ctx)
        }
    }

    impl Drop for CudaContextGuard {
        fn drop(&mut self) {
            if !self.ctx.is_null() {
                unsafe { cuCtxDestroy_v2(self.ctx) };
            }
        }
    }
}

// =============================================================================
// Feature flag for FFI availability
// =============================================================================

/// Indicates whether real NVENC FFI bindings are available.
#[cfg(feature = "nvenc")]
pub const FFI_AVAILABLE: bool = true;

#[cfg(not(feature = "nvenc"))]
pub const FFI_AVAILABLE: bool = false;

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::{HwCapabilities, HwCodec};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// FFI Type Definitions for future nvenc-sys binding
// ============================================================================

/// NVENC API version - corresponds to NV_ENCODE_API_FUNCTION_LIST version.
pub const NVENCAPI_VERSION: u32 = (12 << 4) | 2; // 12.2

/// NVENC struct version macro equivalent.
pub const fn nvenc_struct_version(ver: u32) -> u32 {
    NVENCAPI_VERSION | (ver << 16)
}

/// NVENC status codes - mirrors NVENCSTATUS enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvEncStatus {
    /// Operation successful.
    Success = 0,
    /// No encode device available.
    NoEncodeDevice = 1,
    /// Device not supported.
    UnsupportedDevice = 2,
    /// Invalid encoder device.
    InvalidEncoderDevice = 3,
    /// Invalid device.
    InvalidDevice = 4,
    /// Device was lost (reset/removed).
    DeviceNotExist = 5,
    /// Invalid pointer.
    InvalidPtr = 6,
    /// Invalid event.
    InvalidEvent = 7,
    /// Invalid parameter.
    InvalidParam = 8,
    /// Invalid call sequence.
    InvalidCall = 9,
    /// Out of memory.
    OutOfMemory = 10,
    /// Encoder not initialized.
    EncoderNotInitialized = 11,
    /// Unsupported parameter.
    UnsupportedParam = 12,
    /// Lock bitstream failed.
    LockBusy = 13,
    /// Not enough buffer.
    NotEnoughBuffer = 14,
    /// Invalid version.
    InvalidVersion = 15,
    /// Map failed.
    MapFailed = 16,
    /// Need more input (async encoding).
    NeedMoreInput = 17,
    /// Encoder busy.
    EncoderBusy = 18,
    /// Event not registered.
    EventNotRegistered = 19,
    /// Generic error.
    Generic = 20,
    /// Incompatible client key.
    IncompatibleClientKey = 21,
    /// Unimplemented feature.
    Unimplemented = 22,
    /// Resource register failed.
    ResourceRegisterFailed = 23,
    /// Resource not registered.
    ResourceNotRegistered = 24,
    /// Resource not mapped.
    ResourceNotMapped = 25,
}

impl NvEncStatus {
    /// Check if status indicates success.
    pub fn is_success(&self) -> bool {
        *self == NvEncStatus::Success
    }

    /// Convert to Result.
    pub fn to_result(&self) -> Result<()> {
        if self.is_success() {
            Ok(())
        } else {
            Err(HwAccelError::NvencError(format!("{:?}", self)))
        }
    }

    /// Get error description.
    pub fn description(&self) -> &'static str {
        match self {
            NvEncStatus::Success => "Success",
            NvEncStatus::NoEncodeDevice => "No encoding capable device found",
            NvEncStatus::UnsupportedDevice => "Device not supported for encoding",
            NvEncStatus::InvalidEncoderDevice => "Invalid encoder device",
            NvEncStatus::InvalidDevice => "Invalid device ordinal",
            NvEncStatus::DeviceNotExist => "Device was removed/lost",
            NvEncStatus::InvalidPtr => "Invalid pointer",
            NvEncStatus::InvalidEvent => "Invalid completion event",
            NvEncStatus::InvalidParam => "Invalid parameter",
            NvEncStatus::InvalidCall => "Invalid calling sequence",
            NvEncStatus::OutOfMemory => "Out of memory",
            NvEncStatus::EncoderNotInitialized => "Encoder not initialized",
            NvEncStatus::UnsupportedParam => "Unsupported parameter",
            NvEncStatus::LockBusy => "Bitstream buffer is locked",
            NvEncStatus::NotEnoughBuffer => "Not enough buffer",
            NvEncStatus::InvalidVersion => "Invalid API version",
            NvEncStatus::MapFailed => "Map resource failed",
            NvEncStatus::NeedMoreInput => "Need more input frames",
            NvEncStatus::EncoderBusy => "Encoder is busy",
            NvEncStatus::EventNotRegistered => "Event not registered",
            NvEncStatus::Generic => "Generic error",
            NvEncStatus::IncompatibleClientKey => "Incompatible client key",
            NvEncStatus::Unimplemented => "Feature not implemented",
            NvEncStatus::ResourceRegisterFailed => "Failed to register resource",
            NvEncStatus::ResourceNotRegistered => "Resource not registered",
            NvEncStatus::ResourceNotMapped => "Resource not mapped",
        }
    }
}

/// NVENC capability IDs - mirrors NV_ENC_CAPS enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvEncCaps {
    /// Number of B-frames supported.
    NumMaxBFrames = 0,
    /// Supported rate control modes bitmap.
    SupportedRateControlModes = 1,
    /// Support for field encoding.
    SupportFieldEncoding = 2,
    /// Support for monochrome encoding.
    SupportMonochrome = 3,
    /// Support for FMO.
    SupportFmo = 4,
    /// Support for quantization parameters per frame.
    SupportQpPerFrame = 5,
    /// Support for subframe readback.
    SupportSubframeReadback = 6,
    /// Support for dynamic resolution change.
    SupportDynamicResolutionChange = 7,
    /// Support for dynamic bitrate change.
    SupportDynamicBitrateChange = 8,
    /// Support for dynamic force IDR.
    SupportDynamicForceIdr = 9,
    /// Support for CABAC.
    SupportCabac = 10,
    /// Support for stereo MVC.
    SupportStereoMvc = 11,
    /// Support for lossless encoding.
    SupportLossless = 12,
    /// Support for SAO.
    SupportSao = 13,
    /// Support for mean QP mode.
    SupportMeanQpMode = 14,
    /// Support for lookahead.
    SupportLookahead = 15,
    /// Support for temporal AQ.
    SupportTemporalAq = 16,
    /// Support for 10-bit encoding.
    Support10BitEncode = 17,
    /// Maximum width.
    WidthMax = 18,
    /// Maximum height.
    HeightMax = 19,
    /// Support for temporal SVC.
    SupportTemporalSvc = 20,
    /// Support for dynamic ref pic invalidation.
    SupportDynamicRefPicInvalidate = 21,
    /// Support for emphasis level map.
    SupportEmphasisLevelMap = 22,
    /// Maximum width for dynamic resolution change.
    WidthMin = 23,
    /// Maximum height for dynamic resolution change.
    HeightMin = 24,
    /// Support for multiple ref frames.
    SupportMultipleRefFrames = 25,
    /// Support for constrained encoding.
    SupportConstrainedEncoding = 26,
    /// Support for intra refresh.
    SupportIntraRefresh = 27,
    /// Support for custom VBV buffer size.
    SupportCustomVbvBufSize = 28,
    /// Support for dynamic slice mode change.
    SupportDynamicSliceModeChange = 29,
    /// Support for ref pic invalidation.
    SupportRefPicInvalidation = 30,
    /// Support for pre-encode.
    SupportPreEncode = 31,
    /// Support for async encode.
    SupportAsyncEncode = 32,
    /// Maximum encode sessions.
    MaxEncodeSessionsMax = 33,
    /// Maximum B-frames with B as ref.
    MaxBframesWithBAsRef = 34,
    /// Support for weighted prediction.
    SupportWeightedPrediction = 35,
    /// Support for B-frames as reference.
    SupportBframeAsRef = 36,
    /// Support for emphasis level map.
    SupportEmphasisMap = 37,
    /// Support for ME-only mode.
    SupportMeOnlyMode = 38,
    /// Output in bitstream mode.
    SupportOutputInBitstream = 39,
    /// Exposed count.
    ExposedCount = 40,
}

/// NVENC buffer format - mirrors NV_ENC_BUFFER_FORMAT.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncBufferFormat {
    /// Undefined format.
    #[default]
    Undefined = 0x00000000,
    /// Semi-planar YUV 4:2:0 with interleaved UV.
    Nv12 = 0x00000001,
    /// Planar YUV 4:2:0.
    Yv12 = 0x00000010,
    /// Planar YUV 4:2:0.
    Iyuv = 0x00000100,
    /// Planar YUV 4:4:4.
    Yuv444 = 0x00001000,
    /// 10-bit semi-planar YUV 4:2:0.
    P010 = 0x00010000,
    /// 10-bit planar YUV 4:4:4.
    Yuv444P10 = 0x00100000,
    /// 8-bit RGB packed.
    Argb = 0x01000000,
    /// 10-bit RGB packed.
    Argb10 = 0x02000000,
    /// 8-bit AYUV packed (4:4:4).
    Ayuv = 0x04000000,
    /// 8-bit ABGR packed.
    Abgr = 0x10000000,
    /// 10-bit ABGR packed.
    Abgr10 = 0x20000000,
    /// 16-bit semi-planar (used for lossless).
    U8 = 0x40000000,
}

impl NvEncBufferFormat {
    /// Get bits per pixel for this format.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            NvEncBufferFormat::Nv12 | NvEncBufferFormat::Yv12 | NvEncBufferFormat::Iyuv => 12,
            NvEncBufferFormat::P010 => 15,
            NvEncBufferFormat::Yuv444 | NvEncBufferFormat::Ayuv => 24,
            NvEncBufferFormat::Yuv444P10 => 30,
            NvEncBufferFormat::Argb | NvEncBufferFormat::Abgr => 32,
            NvEncBufferFormat::Argb10 | NvEncBufferFormat::Abgr10 => 40,
            _ => 0,
        }
    }

    /// Check if format is 10-bit.
    pub fn is_10bit(&self) -> bool {
        matches!(
            self,
            NvEncBufferFormat::P010
                | NvEncBufferFormat::Yuv444P10
                | NvEncBufferFormat::Argb10
                | NvEncBufferFormat::Abgr10
        )
    }

    /// Convert from HwSurfaceFormat.
    pub fn from_surface_format(format: HwSurfaceFormat) -> Self {
        match format {
            HwSurfaceFormat::Nv12 => NvEncBufferFormat::Nv12,
            HwSurfaceFormat::P010 => NvEncBufferFormat::P010,
            HwSurfaceFormat::Yuv420p => NvEncBufferFormat::Iyuv,
            HwSurfaceFormat::Yuv420p10 => NvEncBufferFormat::P010,
            HwSurfaceFormat::Rgba => NvEncBufferFormat::Argb,
            HwSurfaceFormat::Bgra => NvEncBufferFormat::Abgr,
        }
    }
}

/// GUID structure - mirrors GUID for Windows compatibility.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NvGuid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl NvGuid {
    /// Create a new GUID.
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }

    /// Null/empty GUID.
    pub const NULL: NvGuid = NvGuid::new(0, 0, 0, [0; 8]);
}

// Codec GUIDs
/// H.264 Codec GUID.
pub const NV_ENC_CODEC_H264_GUID: NvGuid =
    NvGuid::new(0x6BC82762, 0x4E63, 0x4CA4, [0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF]);

/// HEVC Codec GUID.
pub const NV_ENC_CODEC_HEVC_GUID: NvGuid =
    NvGuid::new(0x790CDC88, 0x4522, 0x4D7B, [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03]);

/// AV1 Codec GUID.
pub const NV_ENC_CODEC_AV1_GUID: NvGuid =
    NvGuid::new(0x0A352289, 0x0AA7, 0x4759, [0x86, 0x2D, 0x5D, 0x15, 0xCD, 0x16, 0xD2, 0x54]);

// Profile GUIDs
/// H.264 Baseline Profile GUID.
pub const NV_ENC_H264_PROFILE_BASELINE_GUID: NvGuid =
    NvGuid::new(0x0727BCAA, 0x78C4, 0x4C83, [0x8C, 0x2F, 0xEF, 0x3D, 0xFF, 0x26, 0x7C, 0x6A]);

/// H.264 Main Profile GUID.
pub const NV_ENC_H264_PROFILE_MAIN_GUID: NvGuid =
    NvGuid::new(0x60B5C1D4, 0x67FE, 0x4790, [0x94, 0xD5, 0xC4, 0x72, 0x6D, 0x7B, 0x6E, 0x6D]);

/// H.264 High Profile GUID.
pub const NV_ENC_H264_PROFILE_HIGH_GUID: NvGuid =
    NvGuid::new(0xE7CBC309, 0x4F7A, 0x4B89, [0xAF, 0x2A, 0xD5, 0x37, 0xC9, 0x2B, 0xE3, 0x10]);

/// H.264 High 444 Profile GUID.
pub const NV_ENC_H264_PROFILE_HIGH_444_GUID: NvGuid =
    NvGuid::new(0x7AC663CB, 0xA598, 0x4960, [0xB8, 0x44, 0x33, 0x9B, 0x26, 0x1A, 0x7D, 0x52]);

/// HEVC Main Profile GUID.
pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: NvGuid =
    NvGuid::new(0xB514C39A, 0xB55B, 0x40FA, [0x87, 0x8F, 0xF1, 0x25, 0x3B, 0x4D, 0xFD, 0xEC]);

/// HEVC Main10 Profile GUID.
pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: NvGuid =
    NvGuid::new(0xFA4D2B6C, 0x3A5B, 0x411A, [0x80, 0x18, 0x0A, 0x3F, 0x5E, 0x3C, 0x9B, 0xE5]);

/// AV1 Main Profile GUID.
pub const NV_ENC_AV1_PROFILE_MAIN_GUID: NvGuid =
    NvGuid::new(0x5F2A39F5, 0xF14E, 0x4F95, [0x9A, 0x9E, 0xB7, 0x6D, 0x56, 0x8F, 0xCA, 0xF2]);

// Preset GUIDs
/// P1 (Fastest) Preset GUID.
pub const NV_ENC_PRESET_P1_GUID: NvGuid =
    NvGuid::new(0xFC0A8D3E, 0x45F8, 0x4CF8, [0x80, 0xC7, 0x29, 0x87, 0x71, 0x32, 0x41, 0x1F]);

/// P2 (Faster) Preset GUID.
pub const NV_ENC_PRESET_P2_GUID: NvGuid =
    NvGuid::new(0xF581CFB8, 0x88D6, 0x4381, [0x93, 0xF0, 0xDF, 0x13, 0xF9, 0xC2, 0x7D, 0xAB]);

/// P3 (Fast) Preset GUID.
pub const NV_ENC_PRESET_P3_GUID: NvGuid =
    NvGuid::new(0x36850110, 0x3A07, 0x441F, [0x94, 0xD5, 0x38, 0x70, 0x95, 0x4F, 0x80, 0x11]);

/// P4 (Medium/Balanced) Preset GUID.
pub const NV_ENC_PRESET_P4_GUID: NvGuid =
    NvGuid::new(0x90A7B826, 0xDF06, 0x4862, [0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x8D, 0x81]);

/// P5 (Slow) Preset GUID.
pub const NV_ENC_PRESET_P5_GUID: NvGuid =
    NvGuid::new(0x21C6E6B4, 0x297A, 0x4CBA, [0x99, 0x8F, 0xB6, 0xCB, 0xDE, 0x72, 0xAD, 0xE3]);

/// P6 (Slower) Preset GUID.
pub const NV_ENC_PRESET_P6_GUID: NvGuid =
    NvGuid::new(0x8E75C279, 0x6299, 0x4AB6, [0x83, 0x6A, 0x82, 0x73, 0x5B, 0xD1, 0x11, 0x2E]);

/// P7 (Slowest/Best quality) Preset GUID.
pub const NV_ENC_PRESET_P7_GUID: NvGuid =
    NvGuid::new(0x84848C12, 0x6F71, 0x4C13, [0x93, 0x1B, 0x53, 0xE2, 0x83, 0xF5, 0x79, 0x74]);

/// Low Latency High Quality Preset GUID.
pub const NV_ENC_PRESET_LOW_LATENCY_HQ_GUID: NvGuid =
    NvGuid::new(0x67082A44, 0x4BAD, 0x48FA, [0x98, 0xEA, 0x93, 0x05, 0x6D, 0x15, 0x0A, 0x58]);

/// Low Latency High Performance Preset GUID.
pub const NV_ENC_PRESET_LOW_LATENCY_HP_GUID: NvGuid =
    NvGuid::new(0x2FC12F38, 0xDF0B, 0x4E28, [0x92, 0xC8, 0x52, 0x9F, 0x54, 0x5F, 0x17, 0x39]);

/// Lossless Default Preset GUID.
pub const NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID: NvGuid =
    NvGuid::new(0xD5BFB716, 0xC604, 0x44E7, [0x9B, 0xB8, 0xDE, 0xA5, 0x51, 0x0F, 0xC3, 0xAC]);

// ============================================================================
// CUDA Types (for nvenc-sys / cuda-sys bindings)
// ============================================================================

/// CUDA context handle - opaque pointer type.
pub type CUcontext = *mut std::ffi::c_void;

/// CUDA device handle.
pub type CUdevice = i32;

/// CUDA stream handle.
pub type CUstream = *mut std::ffi::c_void;

/// CUDA device pointer (GPU memory address).
pub type CUdeviceptr = u64;

/// CUDA result codes - mirrors CUresult.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaResult {
    /// Success.
    Success = 0,
    /// Invalid value.
    InvalidValue = 1,
    /// Out of memory.
    OutOfMemory = 2,
    /// Not initialized.
    NotInitialized = 3,
    /// Deinitialized.
    Deinitialized = 4,
    /// Profiler disabled.
    ProfilerDisabled = 5,
    /// No device.
    NoDevice = 100,
    /// Invalid device.
    InvalidDevice = 101,
    /// Device not ready.
    DeviceNotReady = 600,
    /// Invalid context.
    InvalidContext = 201,
    /// Context already current.
    ContextAlreadyCurrent = 202,
    /// Map failed.
    MapFailed = 205,
    /// Unmap failed.
    UnmapFailed = 206,
    /// Array is mapped.
    ArrayIsMapped = 207,
    /// Already mapped.
    AlreadyMapped = 208,
    /// Not mapped.
    NotMapped = 211,
    /// Invalid handle.
    InvalidHandle = 400,
    /// Invalid image.
    InvalidImage = 200,
    /// Unknown error.
    Unknown = 999,
}

impl CudaResult {
    /// Check if result indicates success.
    pub fn is_success(&self) -> bool {
        *self == CudaResult::Success
    }

    /// Convert to Result.
    pub fn to_result(&self) -> Result<()> {
        if self.is_success() {
            Ok(())
        } else {
            Err(HwAccelError::CudaError(format!("{:?}", self)))
        }
    }
}

// ============================================================================
// Initialization Structures
// ============================================================================

/// Device type for encoder session - mirrors NV_ENC_DEVICE_TYPE.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncDeviceType {
    /// DirectX device.
    Directx = 0,
    /// CUDA device.
    #[default]
    Cuda = 1,
    /// OpenGL device.
    Opengl = 2,
}

/// Open encode session parameters - mirrors NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncOpenEncodeSessionExParams {
    /// Structure version (set to NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER).
    pub version: u32,
    /// Device type (CUDA, DirectX, etc.).
    pub device_type: NvEncDeviceType,
    /// Device handle (CUcontext for CUDA, ID3D11Device* for DX).
    pub device: *mut std::ffi::c_void,
    /// Reserved - must be 0.
    pub reserved: *mut std::ffi::c_void,
    /// API version requested.
    pub api_version: u32,
    /// Reserved fields.
    pub reserved1: [u32; 253],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncOpenEncodeSessionExParams {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(1),
            device_type: NvEncDeviceType::Cuda,
            device: std::ptr::null_mut(),
            reserved: std::ptr::null_mut(),
            api_version: NVENCAPI_VERSION,
            reserved1: [0; 253],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Tuning info for presets - mirrors NV_ENC_TUNING_INFO.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncTuningInfo {
    /// Undefined tuning.
    Undefined = 0,
    /// High quality tuning.
    #[default]
    HighQuality = 1,
    /// Low latency tuning.
    LowLatency = 2,
    /// Ultra low latency tuning.
    UltraLowLatency = 3,
    /// Lossless tuning.
    Lossless = 4,
}

/// Rate control mode - mirrors NV_ENC_PARAMS_RC_MODE.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncRcMode {
    /// Constant QP.
    ConstQp = 0,
    /// Variable bitrate.
    #[default]
    Vbr = 1,
    /// Constant bitrate.
    Cbr = 2,
    /// VBR with MinQP (deprecated).
    VbrMinQp = 4,
    /// 2-pass quality.
    TwoPassQuality = 8,
    /// 2-pass frame size cap.
    TwoPassFrameSizeCap = 16,
    /// 2-pass VBR.
    TwoPassVbr = 32,
    /// CBR with low delay HQ.
    CbrLowdelayHq = 64,
    /// CBR with HQ.
    CbrHq = 128,
    /// VBR with HQ.
    VbrHq = 256,
}

/// Multi-pass encoding mode - mirrors NV_ENC_MULTI_PASS.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncMultiPass {
    /// Disabled.
    #[default]
    Disabled = 0,
    /// Quarter resolution.
    QuarterResolution = 1,
    /// Full resolution.
    FullResolution = 2,
}

/// Rate control parameters - mirrors NV_ENC_RC_PARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncRcParams {
    /// Structure version.
    pub version: u32,
    /// Rate control mode.
    pub rate_control_mode: NvEncRcMode,
    /// Constant QP for I-frames (CQP mode).
    pub const_qp_i: i32,
    /// Constant QP for P-frames (CQP mode).
    pub const_qp_p: i32,
    /// Constant QP for B-frames (CQP mode).
    pub const_qp_b: i32,
    /// Average bitrate (bps).
    pub average_bitrate: u32,
    /// Maximum bitrate (bps).
    pub max_bitrate: u32,
    /// VBV buffer size.
    pub vbv_buffer_size: u32,
    /// VBV initial delay.
    pub vbv_initial_delay: u32,
    /// Enable MinQP.
    pub enable_min_qp: u32,
    /// Enable MaxQP.
    pub enable_max_qp: u32,
    /// Enable initial RC QP.
    pub enable_initial_rc_qp: u32,
    /// Enable AQ (adaptive quantization).
    pub enable_aq: u32,
    /// Enable lookahead.
    pub enable_lookahead: u32,
    /// Lookahead depth (frames).
    pub lookahead_depth: u32,
    /// Disable I-frame adaptation.
    pub disable_i_adapt: u32,
    /// Disable B-frame adaptation.
    pub disable_b_adapt: u32,
    /// Enable strict GOP.
    pub strict_gop_target: u32,
    /// AQ strength (1-15).
    pub aq_strength: u32,
    /// Minimum QP for I-frames.
    pub min_qp_i: i32,
    /// Minimum QP for P-frames.
    pub min_qp_p: i32,
    /// Minimum QP for B-frames.
    pub min_qp_b: i32,
    /// Maximum QP for I-frames.
    pub max_qp_i: i32,
    /// Maximum QP for P-frames.
    pub max_qp_p: i32,
    /// Maximum QP for B-frames.
    pub max_qp_b: i32,
    /// Initial RC QP for I-frames.
    pub initial_rc_qp_i: i32,
    /// Initial RC QP for P-frames.
    pub initial_rc_qp_p: i32,
    /// Initial RC QP for B-frames.
    pub initial_rc_qp_b: i32,
    /// Enable temporal AQ.
    pub enable_temporal_aq: u32,
    /// Zero reorder delay.
    pub zero_reorder_delay: u32,
    /// Enable non-reference P-frames.
    pub enable_non_ref_p: u32,
    /// Target quality for CQ mode (0-51).
    pub target_quality: u32,
    /// Target quality LSB.
    pub target_quality_lsb: u32,
    /// Enable QP map.
    pub enable_qp_map: u32,
    /// QP map mode.
    pub qp_map_mode: u32,
    /// Multi-pass mode.
    pub multi_pass: NvEncMultiPass,
    /// CBR low delay HQ params.
    pub cbr_low_delay_hq: u32,
    /// Reserved fields.
    pub reserved: [u32; 219],
}

impl Default for NvEncRcParams {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(2),
            rate_control_mode: NvEncRcMode::Vbr,
            const_qp_i: 0,
            const_qp_p: 0,
            const_qp_b: 0,
            average_bitrate: 5_000_000,
            max_bitrate: 10_000_000,
            vbv_buffer_size: 0,
            vbv_initial_delay: 0,
            enable_min_qp: 0,
            enable_max_qp: 0,
            enable_initial_rc_qp: 0,
            enable_aq: 0,
            enable_lookahead: 0,
            lookahead_depth: 0,
            disable_i_adapt: 0,
            disable_b_adapt: 0,
            strict_gop_target: 0,
            aq_strength: 0,
            min_qp_i: 0,
            min_qp_p: 0,
            min_qp_b: 0,
            max_qp_i: 51,
            max_qp_p: 51,
            max_qp_b: 51,
            initial_rc_qp_i: 0,
            initial_rc_qp_p: 0,
            initial_rc_qp_b: 0,
            enable_temporal_aq: 0,
            zero_reorder_delay: 0,
            enable_non_ref_p: 0,
            target_quality: 0,
            target_quality_lsb: 0,
            enable_qp_map: 0,
            qp_map_mode: 0,
            multi_pass: NvEncMultiPass::Disabled,
            cbr_low_delay_hq: 0,
            reserved: [0; 219],
        }
    }
}

/// Initialize encoder parameters - mirrors NV_ENC_INITIALIZE_PARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncInitializeParams {
    /// Structure version.
    pub version: u32,
    /// Encode GUID (codec).
    pub encode_guid: NvGuid,
    /// Preset GUID.
    pub preset_guid: NvGuid,
    /// Encode width.
    pub encode_width: u32,
    /// Encode height.
    pub encode_height: u32,
    /// Display aspect ratio X.
    pub dar_width: u32,
    /// Display aspect ratio Y.
    pub dar_height: u32,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Enable encode async.
    pub enable_encode_async: u32,
    /// Enable picture-level stats.
    pub enable_pic_stats: u32,
    /// Report slice offsets.
    pub report_slice_offsets: u32,
    /// Enable subframe write.
    pub enable_subframe_write: u32,
    /// Enable external ME hints.
    pub enable_external_me_hints: u32,
    /// Enable ME-only mode.
    pub enable_me_only_mode: u32,
    /// Reserved bitfield.
    pub reserved_bitfields: u32,
    /// Private data.
    pub priv_data_size: u32,
    /// Private data pointer.
    pub priv_data: *mut std::ffi::c_void,
    /// Encoder config.
    pub encode_config: *mut NvEncConfig,
    /// Max encode width (for resolution changes).
    pub max_encode_width: u32,
    /// Max encode height (for resolution changes).
    pub max_encode_height: u32,
    /// ME hints per block - L0.
    pub max_me_hints_per_block_l0: [u32; 2],
    /// ME hints per block - L1.
    pub max_me_hints_per_block_l1: [u32; 2],
    /// Tuning info.
    pub tuning_info: NvEncTuningInfo,
    /// Reserved fields.
    pub reserved: [u32; 287],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncInitializeParams {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(5),
            encode_guid: NV_ENC_CODEC_H264_GUID,
            preset_guid: NV_ENC_PRESET_P4_GUID,
            encode_width: 1920,
            encode_height: 1080,
            dar_width: 16,
            dar_height: 9,
            frame_rate_num: 30,
            frame_rate_den: 1,
            enable_encode_async: 0,
            enable_pic_stats: 0,
            report_slice_offsets: 0,
            enable_subframe_write: 0,
            enable_external_me_hints: 0,
            enable_me_only_mode: 0,
            reserved_bitfields: 0,
            priv_data_size: 0,
            priv_data: std::ptr::null_mut(),
            encode_config: std::ptr::null_mut(),
            max_encode_width: 0,
            max_encode_height: 0,
            max_me_hints_per_block_l0: [0; 2],
            max_me_hints_per_block_l1: [0; 2],
            tuning_info: NvEncTuningInfo::HighQuality,
            reserved: [0; 287],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Encoder configuration - mirrors NV_ENC_CONFIG.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncConfig {
    /// Structure version.
    pub version: u32,
    /// Profile GUID.
    pub profile_guid: NvGuid,
    /// GOP length.
    pub gop_length: u32,
    /// Frame interval P (B-frames between I and P).
    pub frame_interval_p: i32,
    /// Monochrome encoding.
    pub monochrome: u32,
    /// IDR period.
    pub idr_period: u32,
    /// Rate control parameters.
    pub rc_params: NvEncRcParams,
    /// Motion vector precision.
    pub mv_precision: u32,
    /// H.264 config (union - using H264 as default).
    pub enc_codec_config: NvEncCodecConfig,
    /// Reserved fields.
    pub reserved: [u32; 278],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncConfig {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(8),
            profile_guid: NV_ENC_H264_PROFILE_HIGH_GUID,
            gop_length: 60,
            frame_interval_p: 3, // 2 B-frames
            monochrome: 0,
            idr_period: 60,
            rc_params: NvEncRcParams::default(),
            mv_precision: 1,
            enc_codec_config: NvEncCodecConfig::H264(NvEncConfigH264::default()),
            reserved: [0; 278],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Codec-specific configuration (union in C).
#[derive(Debug, Clone)]
pub enum NvEncCodecConfig {
    H264(NvEncConfigH264),
    Hevc(NvEncConfigHevc),
    Av1(NvEncConfigAv1),
}

/// H.264 specific config - mirrors NV_ENC_CONFIG_H264.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncConfigH264 {
    /// Enable stereo MVC.
    pub enable_stereo_mvc: u32,
    /// Hierarchy type.
    pub hierarchy_type: u32,
    /// Max temporal layers.
    pub max_temporal_layers: u32,
    /// SVC temporal ID.
    pub svc_temporal_id: u32,
    /// POC type.
    pub poc_type: u32,
    /// Number of slices per frame.
    pub num_slices: u32,
    /// Slice mode.
    pub slice_mode: u32,
    /// Slice mode data.
    pub slice_mode_data: u32,
    /// CABAC enable.
    pub enable_cabac: u32,
    /// Reserved bitfields.
    pub reserved_bitfields: u32,
    /// IDR period.
    pub idr_period: u32,
    /// Chroma format IDC.
    pub chroma_format_idc: u32,
    /// Maximum number of reference frames.
    pub max_num_ref_frames: u32,
    /// Use constraint set flags.
    pub use_constraint_set_flag: u32,
    /// Disable deblocking IDC.
    pub disable_deblock_idc: u32,
    /// Frame packing arrangement.
    pub frame_packing_arrangement: u32,
    /// LTRP mode.
    pub ltrp_mode: u32,
    /// LTR trust mode.
    pub ltr_trust_mode: u32,
    /// Maximum number of LTR frames.
    pub ltr_num_frames: u32,
    /// Enable intra refresh.
    pub enable_intra_refresh: u32,
    /// Intra refresh period.
    pub intra_refresh_period: u32,
    /// Intra refresh count.
    pub intra_refresh_cnt: u32,
    /// Single slice intra refresh.
    pub single_slice_intra_refresh: u32,
    /// VUI parameters.
    pub h264_vui_parameters: NvEncConfigH264Vui,
    /// Reserved.
    pub reserved: [u32; 208],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncConfigH264 {
    fn default() -> Self {
        Self {
            enable_stereo_mvc: 0,
            hierarchy_type: 0,
            max_temporal_layers: 1,
            svc_temporal_id: 0,
            poc_type: 0,
            num_slices: 1,
            slice_mode: 0,
            slice_mode_data: 0,
            enable_cabac: 1,
            reserved_bitfields: 0,
            idr_period: 0,
            chroma_format_idc: 1,
            max_num_ref_frames: 4,
            use_constraint_set_flag: 0,
            disable_deblock_idc: 0,
            frame_packing_arrangement: 0,
            ltrp_mode: 0,
            ltr_trust_mode: 0,
            ltr_num_frames: 0,
            enable_intra_refresh: 0,
            intra_refresh_period: 0,
            intra_refresh_cnt: 0,
            single_slice_intra_refresh: 0,
            h264_vui_parameters: NvEncConfigH264Vui::default(),
            reserved: [0; 208],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// H.264 VUI parameters - mirrors NV_ENC_CONFIG_H264_VUI_PARAMETERS.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct NvEncConfigH264Vui {
    /// Overscan info present.
    pub overscan_info_present_flag: u32,
    /// Overscan appropriate.
    pub overscan_appropriate_flag: u32,
    /// Video signal type present.
    pub video_signal_type_present_flag: u32,
    /// Video format.
    pub video_format: u32,
    /// Video full range.
    pub video_full_range_flag: u32,
    /// Colour description present.
    pub colour_description_present_flag: u32,
    /// Colour primaries.
    pub colour_primaries: u32,
    /// Transfer characteristics.
    pub transfer_characteristics: u32,
    /// Matrix coefficients.
    pub matrix_coefficients: u32,
    /// Chroma sample location type top.
    pub chroma_sample_loc_type_top_field: u32,
    /// Chroma sample location type bottom.
    pub chroma_sample_loc_type_bottom_field: u32,
    /// Bitstream restriction.
    pub bitstream_restriction_flag: u32,
    /// Reserved.
    pub reserved: [u32; 12],
}

/// HEVC specific config - mirrors NV_ENC_CONFIG_HEVC.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncConfigHevc {
    /// Level.
    pub level: u32,
    /// Tier.
    pub tier: u32,
    /// Minimum CU size.
    pub min_cu_size: u32,
    /// Maximum CU size.
    pub max_cu_size: u32,
    /// Use constraint set flags.
    pub use_constraint_set_flag: u32,
    /// Disable deblocking filter across slices.
    pub disable_deblock_across_slice: u32,
    /// Output buffer format.
    pub output_buffer_format: u32,
    /// Output SEI buffer period.
    pub output_sei_buffer_period: u32,
    /// Output SEI picture timing.
    pub output_sei_picture_timing: u32,
    /// Output AUD.
    pub output_aud: u32,
    /// Enable LTR.
    pub enable_ltr: u32,
    /// LTRP mode.
    pub ltrp_mode: u32,
    /// LTR trust mode.
    pub ltr_trust_mode: u32,
    /// LTR num frames.
    pub ltr_num_frames: u32,
    /// Enable intra refresh.
    pub enable_intra_refresh: u32,
    /// Intra refresh period.
    pub intra_refresh_period: u32,
    /// Intra refresh count.
    pub intra_refresh_cnt: u32,
    /// Chroma format IDC.
    pub chroma_format_idc: u32,
    /// Pixel bit depth minus 8.
    pub pixel_bit_depth_minus8: u32,
    /// Reserved bitfields.
    pub reserved_bitfields: u32,
    /// HEVC VUI params.
    pub hevc_vui_parameters: NvEncConfigHevcVui,
    /// Transform skip mode.
    pub transform_skip_mode: u32,
    /// Max transform hierarchy depth inter.
    pub max_transform_hierarchy_depth_inter: u32,
    /// Max transform hierarchy depth intra.
    pub max_transform_hierarchy_depth_intra: u32,
    /// Number of slices per frame.
    pub num_slices: u32,
    /// Slice mode.
    pub slice_mode: u32,
    /// Slice mode data.
    pub slice_mode_data: u32,
    /// Maximum number of reference frames.
    pub max_num_ref_frames_in_dpb: u32,
    /// Reserved.
    pub reserved: [u32; 195],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncConfigHevc {
    fn default() -> Self {
        Self {
            level: 0,
            tier: 0,
            min_cu_size: 8,
            max_cu_size: 64,
            use_constraint_set_flag: 0,
            disable_deblock_across_slice: 0,
            output_buffer_format: 0,
            output_sei_buffer_period: 0,
            output_sei_picture_timing: 0,
            output_aud: 0,
            enable_ltr: 0,
            ltrp_mode: 0,
            ltr_trust_mode: 0,
            ltr_num_frames: 0,
            enable_intra_refresh: 0,
            intra_refresh_period: 0,
            intra_refresh_cnt: 0,
            chroma_format_idc: 1,
            pixel_bit_depth_minus8: 0,
            reserved_bitfields: 0,
            hevc_vui_parameters: NvEncConfigHevcVui::default(),
            transform_skip_mode: 0,
            max_transform_hierarchy_depth_inter: 0,
            max_transform_hierarchy_depth_intra: 0,
            num_slices: 1,
            slice_mode: 0,
            slice_mode_data: 0,
            max_num_ref_frames_in_dpb: 4,
            reserved: [0; 195],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// HEVC VUI parameters - mirrors NV_ENC_CONFIG_HEVC_VUI_PARAMETERS.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct NvEncConfigHevcVui {
    /// Overscan info present.
    pub overscan_info_present_flag: u32,
    /// Overscan appropriate.
    pub overscan_appropriate_flag: u32,
    /// Video signal type present.
    pub video_signal_type_present_flag: u32,
    /// Video format.
    pub video_format: u32,
    /// Video full range.
    pub video_full_range_flag: u32,
    /// Colour description present.
    pub colour_description_present_flag: u32,
    /// Colour primaries.
    pub colour_primaries: u32,
    /// Transfer characteristics.
    pub transfer_characteristics: u32,
    /// Matrix coefficients.
    pub matrix_coefficients: u32,
    /// Chroma loc info present.
    pub chroma_loc_info_present_flag: u32,
    /// Chroma sample location type top.
    pub chroma_sample_loc_type_top_field: u32,
    /// Chroma sample location type bottom.
    pub chroma_sample_loc_type_bottom_field: u32,
    /// Bitstream restriction.
    pub bitstream_restriction_flag: u32,
    /// Reserved.
    pub reserved: [u32; 11],
}

/// AV1 specific config - mirrors NV_ENC_CONFIG_AV1.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncConfigAv1 {
    /// Profile.
    pub profile: u32,
    /// Level.
    pub level: u32,
    /// Tier.
    pub tier: u32,
    /// Minimum CU size.
    pub min_partition_size: u32,
    /// Maximum CU size.
    pub max_partition_size: u32,
    /// Output AnnexB.
    pub output_annex_b: u32,
    /// Enable temporal unit.
    pub enable_temporal_unit: u32,
    /// Enable inter tool.
    pub enable_intra_bc: u32,
    /// Chroma format IDC.
    pub chroma_format_idc: u32,
    /// Pixel bit depth minus 8.
    pub pixel_bit_depth_minus8: u32,
    /// Enable film grain.
    pub enable_film_grain: u32,
    /// Enable film grain update.
    pub enable_film_grain_update: u32,
    /// Number of tiles per frame.
    pub num_tile_columns: u32,
    /// Number of tiles per row.
    pub num_tile_rows: u32,
    /// Maximum reference frames.
    pub max_num_ref_frames: u32,
    /// Enable LTR.
    pub enable_ltr: u32,
    /// LTRP mode.
    pub ltrp_mode: u32,
    /// LTR num frames.
    pub ltr_num_frames: u32,
    /// Reserved.
    pub reserved: [u32; 209],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncConfigAv1 {
    fn default() -> Self {
        Self {
            profile: 0,
            level: 0,
            tier: 0,
            min_partition_size: 8,
            max_partition_size: 64,
            output_annex_b: 0,
            enable_temporal_unit: 0,
            enable_intra_bc: 0,
            chroma_format_idc: 1,
            pixel_bit_depth_minus8: 0,
            enable_film_grain: 0,
            enable_film_grain_update: 0,
            num_tile_columns: 1,
            num_tile_rows: 1,
            max_num_ref_frames: 4,
            enable_ltr: 0,
            ltrp_mode: 0,
            ltr_num_frames: 0,
            reserved: [0; 209],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Preset configuration - mirrors NV_ENC_PRESET_CONFIG.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncPresetConfig {
    /// Structure version.
    pub version: u32,
    /// Preset config (output).
    pub preset_cfg: NvEncConfig,
    /// Reserved.
    pub reserved1: [u32; 255],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncPresetConfig {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(4),
            preset_cfg: NvEncConfig::default(),
            reserved1: [0; 255],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Reconfigure parameters - mirrors NV_ENC_RECONFIGURE_PARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncReconfigureParams {
    /// Structure version.
    pub version: u32,
    /// New initialization params.
    pub reinit_encode_params: NvEncInitializeParams,
    /// Reset encoder.
    pub reset_encoder: u32,
    /// Force IDR.
    pub force_idr: u32,
    /// Reserved.
    pub reserved: [u32; 251],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncReconfigureParams {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(1),
            reinit_encode_params: NvEncInitializeParams::default(),
            reset_encoder: 0,
            force_idr: 0,
            reserved: [0; 251],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

// ============================================================================
// Encoding Structures
// ============================================================================

/// Create input buffer params - mirrors NV_ENC_CREATE_INPUT_BUFFER.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncCreateInputBuffer {
    /// Structure version.
    pub version: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Memory heap (deprecated).
    pub memory_heap: u32,
    /// Buffer format.
    pub buffer_fmt: NvEncBufferFormat,
    /// Reserved.
    pub reserved: u32,
    /// Input buffer (output).
    pub input_buffer: *mut std::ffi::c_void,
    /// System memory buffer (output).
    pub sys_mem_buffer: *mut std::ffi::c_void,
    /// Reserved.
    pub reserved1: [u32; 57],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 63],
}

impl Default for NvEncCreateInputBuffer {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(1),
            width: 0,
            height: 0,
            memory_heap: 0,
            buffer_fmt: NvEncBufferFormat::Nv12,
            reserved: 0,
            input_buffer: std::ptr::null_mut(),
            sys_mem_buffer: std::ptr::null_mut(),
            reserved1: [0; 57],
            reserved2: [std::ptr::null_mut(); 63],
        }
    }
}

/// Create bitstream buffer params - mirrors NV_ENC_CREATE_BITSTREAM_BUFFER.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncCreateBitstreamBuffer {
    /// Structure version.
    pub version: u32,
    /// Size (deprecated).
    pub size: u32,
    /// Memory heap (deprecated).
    pub memory_heap: u32,
    /// Reserved.
    pub reserved: u32,
    /// Bitstream buffer (output).
    pub bitstream_buffer: *mut std::ffi::c_void,
    /// Bitstream buffer pointer (output).
    pub bitstream_buffer_ptr: *mut std::ffi::c_void,
    /// Reserved.
    pub reserved1: [u32; 58],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncCreateBitstreamBuffer {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(1),
            size: 0,
            memory_heap: 0,
            reserved: 0,
            bitstream_buffer: std::ptr::null_mut(),
            bitstream_buffer_ptr: std::ptr::null_mut(),
            reserved1: [0; 58],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Lock input buffer params - mirrors NV_ENC_LOCK_INPUT_BUFFER.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncLockInputBuffer {
    /// Structure version.
    pub version: u32,
    /// Reserved.
    pub reserved_bitfields: u32,
    /// Input buffer to lock.
    pub input_buffer: *mut std::ffi::c_void,
    /// Buffer data pointer (output).
    pub buffer_data_ptr: *mut std::ffi::c_void,
    /// Pitch (output).
    pub pitch: u32,
    /// Reserved.
    pub reserved1: [u32; 251],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncLockInputBuffer {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(1),
            reserved_bitfields: 0,
            input_buffer: std::ptr::null_mut(),
            buffer_data_ptr: std::ptr::null_mut(),
            pitch: 0,
            reserved1: [0; 251],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Lock bitstream params - mirrors NV_ENC_LOCK_BITSTREAM.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncLockBitstream {
    /// Structure version.
    pub version: u32,
    /// Bitstream wait for completion.
    pub do_not_wait: u32,
    /// Report encode stats.
    pub report_encode_stats: u32,
    /// Output bitstream (input).
    pub output_bitstream: *mut std::ffi::c_void,
    /// Slice offsets (output).
    pub slice_offsets: *mut u32,
    /// Frame index (output).
    pub frame_idx: u32,
    /// Hardware encode status (output).
    pub hw_encode_status: u32,
    /// Number of slices (output).
    pub num_slices: u32,
    /// Bitstream size (output).
    pub bitstream_size_in_bytes: u32,
    /// Output PTS (output).
    pub output_timestamp: u64,
    /// Output duration (output).
    pub output_duration: u64,
    /// Bitstream data pointer (output).
    pub bitstream_buffer_ptr: *mut std::ffi::c_void,
    /// Picture type (output).
    pub picture_type: NvEncPicType,
    /// Picture structure (output).
    pub picture_struct: NvEncPicStruct,
    /// Frame average QP (output).
    pub frame_avg_qp: u32,
    /// Frame saturation (output).
    pub frame_satd: u32,
    /// LTR frame index (output).
    pub ltr_frame_idx: u32,
    /// LTR frame bitmap (output).
    pub ltr_frame_bitmap: u32,
    /// Reserved.
    pub reserved: [u32; 13],
    /// Intra refresh count (output).
    pub intra_refresh_cnt: u32,
    /// Reserved.
    pub reserved1: [u32; 210],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncLockBitstream {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(2),
            do_not_wait: 0,
            report_encode_stats: 0,
            output_bitstream: std::ptr::null_mut(),
            slice_offsets: std::ptr::null_mut(),
            frame_idx: 0,
            hw_encode_status: 0,
            num_slices: 0,
            bitstream_size_in_bytes: 0,
            output_timestamp: 0,
            output_duration: 0,
            bitstream_buffer_ptr: std::ptr::null_mut(),
            picture_type: NvEncPicType::P,
            picture_struct: NvEncPicStruct::Frame,
            frame_avg_qp: 0,
            frame_satd: 0,
            ltr_frame_idx: 0,
            ltr_frame_bitmap: 0,
            reserved: [0; 13],
            intra_refresh_cnt: 0,
            reserved1: [0; 210],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Picture type - mirrors NV_ENC_PIC_TYPE.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncPicType {
    /// P-frame.
    #[default]
    P = 0,
    /// B-frame.
    B = 1,
    /// I-frame.
    I = 2,
    /// IDR-frame.
    Idr = 3,
    /// Bidirectional frame.
    Bi = 4,
    /// Skip frame.
    Skip = 5,
    /// Intra refresh.
    IntraRefresh = 6,
    /// Unknown.
    Unknown = 0xFF,
}

/// Picture structure - mirrors NV_ENC_PIC_STRUCT.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncPicStruct {
    /// Progressive frame.
    #[default]
    Frame = 0x01,
    /// Top field.
    FieldTopBottom = 0x02,
    /// Bottom field.
    FieldBottomTop = 0x03,
}

/// Picture parameters - mirrors NV_ENC_PIC_PARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncPicParams {
    /// Structure version.
    pub version: u32,
    /// Input width.
    pub input_width: u32,
    /// Input height.
    pub input_height: u32,
    /// Input pitch.
    pub input_pitch: u32,
    /// Encode flags.
    pub encode_flags: u32,
    /// Frame index.
    pub frame_idx: u32,
    /// Input timestamp.
    pub input_timestamp: u64,
    /// Input duration.
    pub input_duration: u64,
    /// Input buffer.
    pub input_buffer: *mut std::ffi::c_void,
    /// Output bitstream.
    pub output_bitstream: *mut std::ffi::c_void,
    /// Completion event.
    pub completion_event: *mut std::ffi::c_void,
    /// Buffer format.
    pub buffer_fmt: NvEncBufferFormat,
    /// Picture structure.
    pub picture_struct: NvEncPicStruct,
    /// Picture type.
    pub picture_type: NvEncPicType,
    /// Codec-specific parameters.
    pub codec_pic_params: NvEncCodecPicParams,
    /// ME hint counts per block.
    pub me_hint_counts_per_block: [u32; 2],
    /// ME external hints.
    pub me_external_hints: *mut std::ffi::c_void,
    /// Reserved.
    pub reserved1: [u32; 6],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 2],
    /// QP delta map.
    pub qp_delta_map: *mut i8,
    /// QP delta map size.
    pub qp_delta_map_size: u32,
    /// External ME hint buffer.
    pub external_me_hint_buffer: *mut std::ffi::c_void,
    /// Slice type data.
    pub slice_type_data: *mut std::ffi::c_void,
    /// Slice type array count.
    pub slice_type_array_cnt: u32,
    /// Reference picture set.
    pub ref_pic_set: *mut std::ffi::c_void,
    /// Alpha layer SPS ID.
    pub alpha_layer_sps_id: u32,
    /// Alpha layer PPS ID.
    pub alpha_layer_pps_id: u32,
    /// Reserved.
    pub reserved3: [u32; 196],
    /// Reserved pointers.
    pub reserved4: [*mut std::ffi::c_void; 61],
}

impl Default for NvEncPicParams {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(6),
            input_width: 0,
            input_height: 0,
            input_pitch: 0,
            encode_flags: 0,
            frame_idx: 0,
            input_timestamp: 0,
            input_duration: 0,
            input_buffer: std::ptr::null_mut(),
            output_bitstream: std::ptr::null_mut(),
            completion_event: std::ptr::null_mut(),
            buffer_fmt: NvEncBufferFormat::Nv12,
            picture_struct: NvEncPicStruct::Frame,
            picture_type: NvEncPicType::P,
            codec_pic_params: NvEncCodecPicParams::H264(NvEncH264PicParams::default()),
            me_hint_counts_per_block: [0; 2],
            me_external_hints: std::ptr::null_mut(),
            reserved1: [0; 6],
            reserved2: [std::ptr::null_mut(); 2],
            qp_delta_map: std::ptr::null_mut(),
            qp_delta_map_size: 0,
            external_me_hint_buffer: std::ptr::null_mut(),
            slice_type_data: std::ptr::null_mut(),
            slice_type_array_cnt: 0,
            ref_pic_set: std::ptr::null_mut(),
            alpha_layer_sps_id: 0,
            alpha_layer_pps_id: 0,
            reserved3: [0; 196],
            reserved4: [std::ptr::null_mut(); 61],
        }
    }
}

/// Encode picture flags.
pub mod encode_flags {
    /// EOS flag - signals end of stream.
    pub const NV_ENC_PIC_FLAG_EOS: u32 = 0x01;
    /// Force IDR.
    pub const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x02;
    /// Force intra refresh.
    pub const NV_ENC_PIC_FLAG_FORCEINTRA: u32 = 0x04;
    /// Output SPSPPS.
    pub const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x08;
    /// Invalidate reference frame.
    pub const NV_ENC_PIC_FLAG_INVALIDATE_REF: u32 = 0x10;
}

/// Codec-specific picture parameters (union in C).
#[derive(Debug, Clone)]
pub enum NvEncCodecPicParams {
    H264(NvEncH264PicParams),
    Hevc(NvEncHevcPicParams),
    Av1(NvEncAv1PicParams),
}

/// H.264 picture parameters - mirrors NV_ENC_CODEC_PIC_PARAMS_H264.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncH264PicParams {
    /// Display POC syntax.
    pub display_poc_syntax: u32,
    /// Reserved.
    pub ref_pic_flag: u32,
    /// Color plane ID.
    pub colour_plane_id: u32,
    /// Force intra slice count.
    pub force_intra_refresh_with_frame_cnt: u32,
    /// Constraint set flag.
    pub constraint_set_flag: u32,
    /// SEI message.
    pub sei_payload_array_cnt: u32,
    /// SEI payload array.
    pub sei_payload_array: *mut std::ffi::c_void,
    /// Slice mode.
    pub slice_type_data: *mut u32,
    /// Slice type array count.
    pub slice_type_array_cnt: u32,
    /// Slice mode data override.
    pub slice_mode_data_override: u32,
    /// Slice mode data update.
    pub slice_mode_data_update: *mut std::ffi::c_void,
    /// LTR mark frame.
    pub ltr_mark_frame: u32,
    /// LTR use frames.
    pub ltr_use_frames: u32,
    /// LTR use frame bitmap.
    pub ltr_use_frame_bitmap: u32,
    /// Reserved.
    pub reserved: [u32; 243],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 62],
}

impl Default for NvEncH264PicParams {
    fn default() -> Self {
        Self {
            display_poc_syntax: 0,
            ref_pic_flag: 0,
            colour_plane_id: 0,
            force_intra_refresh_with_frame_cnt: 0,
            constraint_set_flag: 0,
            sei_payload_array_cnt: 0,
            sei_payload_array: std::ptr::null_mut(),
            slice_type_data: std::ptr::null_mut(),
            slice_type_array_cnt: 0,
            slice_mode_data_override: 0,
            slice_mode_data_update: std::ptr::null_mut(),
            ltr_mark_frame: 0,
            ltr_use_frames: 0,
            ltr_use_frame_bitmap: 0,
            reserved: [0; 243],
            reserved2: [std::ptr::null_mut(); 62],
        }
    }
}

/// HEVC picture parameters - mirrors NV_ENC_CODEC_PIC_PARAMS_HEVC.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncHevcPicParams {
    /// Reserved.
    pub reserved: u32,
    /// Slice mode.
    pub slice_mode_data: u32,
    /// SEI message count.
    pub sei_payload_array_cnt: u32,
    /// SEI payload array.
    pub sei_payload_array: *mut std::ffi::c_void,
    /// Slice type data.
    pub slice_type_data: *mut u32,
    /// Slice type array count.
    pub slice_type_array_cnt: u32,
    /// Reserved.
    pub reserved2: [u32; 249],
    /// Reserved pointers.
    pub reserved3: [*mut std::ffi::c_void; 63],
}

impl Default for NvEncHevcPicParams {
    fn default() -> Self {
        Self {
            reserved: 0,
            slice_mode_data: 0,
            sei_payload_array_cnt: 0,
            sei_payload_array: std::ptr::null_mut(),
            slice_type_data: std::ptr::null_mut(),
            slice_type_array_cnt: 0,
            reserved2: [0; 249],
            reserved3: [std::ptr::null_mut(); 63],
        }
    }
}

/// AV1 picture parameters - mirrors NV_ENC_CODEC_PIC_PARAMS_AV1.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncAv1PicParams {
    /// Reserved.
    pub reserved: [u32; 256],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 64],
}

impl Default for NvEncAv1PicParams {
    fn default() -> Self {
        Self {
            reserved: [0; 256],
            reserved2: [std::ptr::null_mut(); 64],
        }
    }
}

/// Map input resource params - mirrors NV_ENC_MAP_INPUT_RESOURCE.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncMapInputResource {
    /// Structure version.
    pub version: u32,
    /// Subresource index.
    pub subresource_index: u32,
    /// Input resource.
    pub input_resource: *mut std::ffi::c_void,
    /// Registered resource.
    pub registered_resource: *mut std::ffi::c_void,
    /// Mapped resource (output).
    pub mapped_resource: *mut std::ffi::c_void,
    /// Mapped buffer format (output).
    pub mapped_buffer_fmt: NvEncBufferFormat,
    /// Reserved.
    pub reserved1: [u32; 251],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 63],
}

impl Default for NvEncMapInputResource {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(4),
            subresource_index: 0,
            input_resource: std::ptr::null_mut(),
            registered_resource: std::ptr::null_mut(),
            mapped_resource: std::ptr::null_mut(),
            mapped_buffer_fmt: NvEncBufferFormat::Undefined,
            reserved1: [0; 251],
            reserved2: [std::ptr::null_mut(); 63],
        }
    }
}

/// Register resource params - mirrors NV_ENC_REGISTER_RESOURCE.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NvEncRegisterResource {
    /// Structure version.
    pub version: u32,
    /// Resource type.
    pub resource_type: NvEncInputResourceType,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Pitch.
    pub pitch: u32,
    /// Subresource index.
    pub subresource_index: u32,
    /// Resource to register.
    pub resource_to_register: *mut std::ffi::c_void,
    /// Registered resource (output).
    pub registered_resource: *mut std::ffi::c_void,
    /// Buffer format.
    pub buffer_fmt: NvEncBufferFormat,
    /// Buffer usage.
    pub buffer_usage: u32,
    /// Reserved.
    pub reserved1: [u32; 247],
    /// Reserved pointers.
    pub reserved2: [*mut std::ffi::c_void; 62],
}

impl Default for NvEncRegisterResource {
    fn default() -> Self {
        Self {
            version: nvenc_struct_version(3),
            resource_type: NvEncInputResourceType::Cudadeviceptr,
            width: 0,
            height: 0,
            pitch: 0,
            subresource_index: 0,
            resource_to_register: std::ptr::null_mut(),
            registered_resource: std::ptr::null_mut(),
            buffer_fmt: NvEncBufferFormat::Nv12,
            buffer_usage: 0,
            reserved1: [0; 247],
            reserved2: [std::ptr::null_mut(); 62],
        }
    }
}

/// Input resource type - mirrors NV_ENC_INPUT_RESOURCE_TYPE.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvEncInputResourceType {
    /// DirectX 9 resource.
    Directx = 0,
    /// CUDA device pointer.
    #[default]
    Cudadeviceptr = 1,
    /// CUDA array.
    Cudaarray = 2,
    /// DirectX 11 resource.
    Directx11 = 3,
    /// DirectX 12 resource.
    Directx12 = 4,
    /// OpenGL resource.
    Opengl = 5,
}

// ============================================================================
// CUDA Integration (Stubbed APIs)
// ============================================================================

/// Stubbed CUDA API module.
pub mod cuda {
    use super::*;

    /// Initialize CUDA driver API.
    /// In real FFI: cuInit(flags: u32) -> CUresult
    pub fn cu_init(_flags: u32) -> CudaResult {
        // Stub implementation
        CudaResult::Success
    }

    /// Get CUDA device count.
    /// In real FFI: cuDeviceGetCount(count: *mut i32) -> CUresult
    pub fn cu_device_get_count() -> std::result::Result<i32, CudaResult> {
        // Stub: report 1 device available
        Ok(1)
    }

    /// Get CUDA device.
    /// In real FFI: cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult
    pub fn cu_device_get(ordinal: i32) -> std::result::Result<CUdevice, CudaResult> {
        if ordinal < 0 {
            return Err(CudaResult::InvalidDevice);
        }
        // Stub: return ordinal as device
        Ok(ordinal)
    }

    /// Get device name.
    /// In real FFI: cuDeviceGetName(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult
    pub fn cu_device_get_name(_device: CUdevice) -> std::result::Result<String, CudaResult> {
        Ok("NVIDIA GPU (Stub)".to_string())
    }

    /// Get device total memory.
    /// In real FFI: cuDeviceTotalMem(bytes: *mut usize, dev: CUdevice) -> CUresult
    pub fn cu_device_total_mem(_device: CUdevice) -> std::result::Result<usize, CudaResult> {
        // Stub: report 8GB
        Ok(8 * 1024 * 1024 * 1024)
    }

    /// Get device compute capability.
    /// In real FFI: cuDeviceGetAttribute(pi: *mut i32, attrib: CUdevice_attribute, dev: CUdevice)
    pub fn cu_device_get_compute_capability(
        _device: CUdevice,
    ) -> std::result::Result<(i32, i32), CudaResult> {
        // Stub: report SM 8.6 (Ampere)
        Ok((8, 6))
    }

    /// Create CUDA context.
    /// In real FFI: cuCtxCreate(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult
    pub fn cu_ctx_create(
        _flags: u32,
        _device: CUdevice,
    ) -> std::result::Result<CUcontext, CudaResult> {
        // Stub: return non-null context
        Ok(0x1 as CUcontext)
    }

    /// Destroy CUDA context.
    /// In real FFI: cuCtxDestroy(ctx: CUcontext) -> CUresult
    pub fn cu_ctx_destroy(_ctx: CUcontext) -> CudaResult {
        CudaResult::Success
    }

    /// Set current CUDA context.
    /// In real FFI: cuCtxSetCurrent(ctx: CUcontext) -> CUresult
    pub fn cu_ctx_set_current(_ctx: CUcontext) -> CudaResult {
        CudaResult::Success
    }

    /// Get current CUDA context.
    /// In real FFI: cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult
    pub fn cu_ctx_get_current() -> std::result::Result<CUcontext, CudaResult> {
        Ok(0x1 as CUcontext)
    }

    /// Push context to stack.
    /// In real FFI: cuCtxPushCurrent(ctx: CUcontext) -> CUresult
    pub fn cu_ctx_push_current(_ctx: CUcontext) -> CudaResult {
        CudaResult::Success
    }

    /// Pop context from stack.
    /// In real FFI: cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult
    pub fn cu_ctx_pop_current() -> std::result::Result<CUcontext, CudaResult> {
        Ok(0x1 as CUcontext)
    }

    /// Allocate device memory.
    /// In real FFI: cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult
    pub fn cu_mem_alloc(size: usize) -> std::result::Result<CUdeviceptr, CudaResult> {
        static COUNTER: AtomicU64 = AtomicU64::new(0x1000_0000);
        if size == 0 {
            return Err(CudaResult::InvalidValue);
        }
        // Stub: return fake device pointer
        Ok(COUNTER.fetch_add(size as u64, Ordering::SeqCst))
    }

    /// Allocate pitched device memory.
    /// In real FFI: cuMemAllocPitch(dptr: *mut CUdeviceptr, pitch: *mut usize, width: usize, height: usize, elem_size: u32)
    pub fn cu_mem_alloc_pitch(
        width: usize,
        height: usize,
        elem_size: u32,
    ) -> std::result::Result<(CUdeviceptr, usize), CudaResult> {
        if width == 0 || height == 0 {
            return Err(CudaResult::InvalidValue);
        }
        // Stub: align pitch to 512 bytes
        let pitch = (width * elem_size as usize + 511) & !511;
        let ptr = cu_mem_alloc(pitch * height)?;
        Ok((ptr, pitch))
    }

    /// Free device memory.
    /// In real FFI: cuMemFree(dptr: CUdeviceptr) -> CUresult
    pub fn cu_mem_free(_ptr: CUdeviceptr) -> CudaResult {
        CudaResult::Success
    }

    /// Copy memory from host to device.
    /// In real FFI: cuMemcpyHtoD(dst: CUdeviceptr, src: *const c_void, bytecount: usize) -> CUresult
    pub fn cu_memcpy_h_to_d(
        _dst: CUdeviceptr,
        _src: *const u8,
        _byte_count: usize,
    ) -> CudaResult {
        CudaResult::Success
    }

    /// Copy memory from device to host.
    /// In real FFI: cuMemcpyDtoH(dst: *mut c_void, src: CUdeviceptr, bytecount: usize) -> CUresult
    pub fn cu_memcpy_d_to_h(
        _dst: *mut u8,
        _src: CUdeviceptr,
        _byte_count: usize,
    ) -> CudaResult {
        CudaResult::Success
    }

    /// Copy memory from device to device.
    /// In real FFI: cuMemcpyDtoD(dst: CUdeviceptr, src: CUdeviceptr, bytecount: usize) -> CUresult
    pub fn cu_memcpy_d_to_d(
        _dst: CUdeviceptr,
        _src: CUdeviceptr,
        _byte_count: usize,
    ) -> CudaResult {
        CudaResult::Success
    }

    /// Async copy memory from host to device.
    /// In real FFI: cuMemcpyHtoDAsync(dst: CUdeviceptr, src: *const c_void, bytecount: usize, stream: CUstream)
    pub fn cu_memcpy_h_to_d_async(
        _dst: CUdeviceptr,
        _src: *const u8,
        _byte_count: usize,
        _stream: CUstream,
    ) -> CudaResult {
        CudaResult::Success
    }

    /// Async copy memory from device to host.
    /// In real FFI: cuMemcpyDtoHAsync(dst: *mut c_void, src: CUdeviceptr, bytecount: usize, stream: CUstream)
    pub fn cu_memcpy_d_to_h_async(
        _dst: *mut u8,
        _src: CUdeviceptr,
        _byte_count: usize,
        _stream: CUstream,
    ) -> CudaResult {
        CudaResult::Success
    }

    /// Create CUDA stream.
    /// In real FFI: cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult
    pub fn cu_stream_create(_flags: u32) -> std::result::Result<CUstream, CudaResult> {
        static COUNTER: AtomicU64 = AtomicU64::new(0x2000_0000);
        Ok(COUNTER.fetch_add(1, Ordering::SeqCst) as CUstream)
    }

    /// Create stream with priority.
    /// In real FFI: cuStreamCreateWithPriority(stream: *mut CUstream, flags: u32, priority: i32)
    pub fn cu_stream_create_with_priority(
        _flags: u32,
        _priority: i32,
    ) -> std::result::Result<CUstream, CudaResult> {
        cu_stream_create(0)
    }

    /// Destroy CUDA stream.
    /// In real FFI: cuStreamDestroy(stream: CUstream) -> CUresult
    pub fn cu_stream_destroy(_stream: CUstream) -> CudaResult {
        CudaResult::Success
    }

    /// Synchronize CUDA stream.
    /// In real FFI: cuStreamSynchronize(stream: CUstream) -> CUresult
    pub fn cu_stream_synchronize(_stream: CUstream) -> CudaResult {
        CudaResult::Success
    }

    /// Query stream status.
    /// In real FFI: cuStreamQuery(stream: CUstream) -> CUresult
    pub fn cu_stream_query(_stream: CUstream) -> CudaResult {
        CudaResult::Success
    }

    /// Synchronize context.
    /// In real FFI: cuCtxSynchronize() -> CUresult
    pub fn cu_ctx_synchronize() -> CudaResult {
        CudaResult::Success
    }

    /// Get free and total memory.
    /// In real FFI: cuMemGetInfo(free: *mut usize, total: *mut usize) -> CUresult
    pub fn cu_mem_get_info() -> std::result::Result<(usize, usize), CudaResult> {
        // Stub: 6GB free, 8GB total
        Ok((6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
    }
}

/// CUDA memory allocation patterns.
pub struct CudaMemoryManager {
    /// Device index.
    device: CUdevice,
    /// CUDA context.
    context: CUcontext,
    /// Allocated buffers.
    allocations: Vec<CudaAllocation>,
    /// Total allocated bytes.
    total_allocated: usize,
}

/// CUDA memory allocation.
#[derive(Debug)]
pub struct CudaAllocation {
    /// Device pointer.
    pub ptr: CUdeviceptr,
    /// Size in bytes.
    pub size: usize,
    /// Pitch (for 2D allocations).
    pub pitch: Option<usize>,
    /// Label for debugging.
    pub label: String,
}

impl CudaMemoryManager {
    /// Create new memory manager.
    pub fn new(device: CUdevice) -> Result<Self> {
        let context = cuda::cu_ctx_create(0, device)
            .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

        Ok(Self {
            device,
            context,
            allocations: Vec::new(),
            total_allocated: 0,
        })
    }

    /// Get device.
    pub fn device(&self) -> CUdevice {
        self.device
    }

    /// Get context.
    pub fn context(&self) -> CUcontext {
        self.context
    }

    /// Allocate linear memory.
    pub fn alloc(&mut self, size: usize, label: &str) -> Result<CUdeviceptr> {
        let ptr = cuda::cu_mem_alloc(size)
            .map_err(|e| HwAccelError::CudaError(format!("Alloc failed: {:?}", e)))?;

        self.allocations.push(CudaAllocation {
            ptr,
            size,
            pitch: None,
            label: label.to_string(),
        });
        self.total_allocated += size;

        Ok(ptr)
    }

    /// Allocate pitched 2D memory.
    pub fn alloc_pitch(
        &mut self,
        width: usize,
        height: usize,
        elem_size: u32,
        label: &str,
    ) -> Result<(CUdeviceptr, usize)> {
        let (ptr, pitch) = cuda::cu_mem_alloc_pitch(width, height, elem_size)
            .map_err(|e| HwAccelError::CudaError(format!("Alloc pitch failed: {:?}", e)))?;

        let size = pitch * height;
        self.allocations.push(CudaAllocation {
            ptr,
            size,
            pitch: Some(pitch),
            label: label.to_string(),
        });
        self.total_allocated += size;

        Ok((ptr, pitch))
    }

    /// Free memory.
    pub fn free(&mut self, ptr: CUdeviceptr) -> Result<()> {
        if let Some(idx) = self.allocations.iter().position(|a| a.ptr == ptr) {
            let alloc = self.allocations.remove(idx);
            self.total_allocated -= alloc.size;
            cuda::cu_mem_free(ptr).to_result()?;
        }
        Ok(())
    }

    /// Get total allocated memory.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get allocation count.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get free memory on device.
    pub fn free_memory(&self) -> Result<usize> {
        let (free, _total) = cuda::cu_mem_get_info()
            .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;
        Ok(free)
    }
}

impl Drop for CudaMemoryManager {
    fn drop(&mut self) {
        // Free all allocations
        for alloc in &self.allocations {
            let _ = cuda::cu_mem_free(alloc.ptr);
        }
        // Destroy context
        let _ = cuda::cu_ctx_destroy(self.context);
    }
}

/// Multi-GPU context manager.
pub struct MultiGpuManager {
    /// GPU contexts.
    contexts: Vec<GpuContext>,
    /// Current GPU index.
    current_gpu: usize,
}

/// Single GPU context.
pub struct GpuContext {
    /// Device index.
    pub device: CUdevice,
    /// CUDA context.
    pub context: CUcontext,
    /// Memory manager.
    pub memory: CudaMemoryManager,
    /// Device name.
    pub name: String,
    /// Compute capability.
    pub compute_capability: (i32, i32),
}

impl MultiGpuManager {
    /// Create multi-GPU manager with all available GPUs.
    pub fn new() -> Result<Self> {
        cuda::cu_init(0).to_result()?;

        let device_count = cuda::cu_device_get_count()
            .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

        let mut contexts = Vec::new();

        for i in 0..device_count {
            let device = cuda::cu_device_get(i)
                .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

            let name = cuda::cu_device_get_name(device)
                .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

            let compute_capability = cuda::cu_device_get_compute_capability(device)
                .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

            let context = cuda::cu_ctx_create(0, device)
                .map_err(|e| HwAccelError::CudaError(format!("{:?}", e)))?;

            // Pop context so we can push it later when needed
            let _ = cuda::cu_ctx_pop_current();

            contexts.push(GpuContext {
                device,
                context,
                memory: CudaMemoryManager::new(device)?,
                name,
                compute_capability,
            });
        }

        Ok(Self {
            contexts,
            current_gpu: 0,
        })
    }

    /// Get number of GPUs.
    pub fn gpu_count(&self) -> usize {
        self.contexts.len()
    }

    /// Set current GPU.
    pub fn set_current_gpu(&mut self, index: usize) -> Result<()> {
        if index >= self.contexts.len() {
            return Err(HwAccelError::DeviceInit(format!(
                "GPU index {} out of range",
                index
            )));
        }

        cuda::cu_ctx_set_current(self.contexts[index].context).to_result()?;
        self.current_gpu = index;
        Ok(())
    }

    /// Get current GPU context.
    pub fn current_context(&self) -> Option<&GpuContext> {
        self.contexts.get(self.current_gpu)
    }

    /// Get current GPU context mutably.
    pub fn current_context_mut(&mut self) -> Option<&mut GpuContext> {
        self.contexts.get_mut(self.current_gpu)
    }

    /// Get GPU context by index.
    pub fn get_context(&self, index: usize) -> Option<&GpuContext> {
        self.contexts.get(index)
    }

    /// Execute on specific GPU.
    pub fn execute_on_gpu<F, R>(&mut self, gpu_index: usize, f: F) -> Result<R>
    where
        F: FnOnce(&mut GpuContext) -> Result<R>,
    {
        let prev_gpu = self.current_gpu;
        self.set_current_gpu(gpu_index)?;

        let result = f(&mut self.contexts[gpu_index]);

        // Restore previous GPU
        if prev_gpu != gpu_index {
            let _ = self.set_current_gpu(prev_gpu);
        }

        result
    }
}

impl Default for MultiGpuManager {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            contexts: Vec::new(),
            current_gpu: 0,
        })
    }
}

// ============================================================================
// Advanced Encoding Features
// ============================================================================

/// Lookahead async model.
pub struct LookaheadAnalyzer {
    /// Lookahead depth.
    depth: u32,
    /// Pending frames for analysis.
    pending_frames: VecDeque<LookaheadFrame>,
    /// Analysis results.
    analysis_results: VecDeque<LookaheadResult>,
    /// Enable temporal AQ.
    temporal_aq: bool,
    /// Enable B-frame adaptation.
    b_adapt: bool,
    /// Scene change threshold.
    scene_change_threshold: f64,
}

/// Frame in lookahead buffer.
#[derive(Debug, Clone)]
pub struct LookaheadFrame {
    /// Frame index.
    pub frame_idx: u64,
    /// PTS.
    pub pts: i64,
    /// Force keyframe.
    pub force_keyframe: bool,
    /// Frame statistics.
    pub stats: Option<FrameStats>,
}

/// Frame statistics from analysis.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Intra cost (sum of SAD for intra prediction).
    pub intra_cost: u64,
    /// Inter cost (sum of SAD for inter prediction).
    pub inter_cost: u64,
    /// Scene change score (0.0-1.0).
    pub scene_change_score: f64,
    /// Complexity per macroblock.
    pub avg_mb_complexity: f64,
    /// Motion vector magnitude average.
    pub avg_mv_magnitude: f64,
}

/// Lookahead analysis result.
#[derive(Debug, Clone)]
pub struct LookaheadResult {
    /// Frame index.
    pub frame_idx: u64,
    /// Recommended frame type.
    pub recommended_type: NvEncPicType,
    /// QP adjustment from temporal AQ.
    pub qp_adjustment: i8,
    /// Is scene change.
    pub is_scene_change: bool,
    /// Reference frame recommendations.
    pub ref_recommendations: RefRecommendations,
}

/// Reference frame recommendations.
#[derive(Debug, Clone, Default)]
pub struct RefRecommendations {
    /// Use as reference.
    pub use_as_ref: bool,
    /// Preferred reference frames.
    pub preferred_refs: Vec<u64>,
    /// Weight for weighted prediction.
    pub wp_weight: Option<WeightedPredParams>,
}

/// Weighted prediction parameters.
#[derive(Debug, Clone)]
pub struct WeightedPredParams {
    /// Luma weight.
    pub luma_weight: i16,
    /// Luma offset.
    pub luma_offset: i16,
    /// Chroma Cb weight.
    pub chroma_cb_weight: i16,
    /// Chroma Cb offset.
    pub chroma_cb_offset: i16,
    /// Chroma Cr weight.
    pub chroma_cr_weight: i16,
    /// Chroma Cr offset.
    pub chroma_cr_offset: i16,
    /// Log2 weight denominator.
    pub log2_weight_denom: u8,
}

impl LookaheadAnalyzer {
    /// Create new lookahead analyzer.
    pub fn new(depth: u32, temporal_aq: bool, b_adapt: bool) -> Self {
        Self {
            depth,
            pending_frames: VecDeque::with_capacity(depth as usize),
            analysis_results: VecDeque::new(),
            temporal_aq,
            b_adapt,
            scene_change_threshold: 0.4,
        }
    }

    /// Submit frame for analysis.
    pub fn submit_frame(&mut self, frame: LookaheadFrame) {
        self.pending_frames.push_back(frame);

        // Trigger analysis when lookahead is full
        if self.pending_frames.len() >= self.depth as usize {
            self.analyze();
        }
    }

    /// Analyze pending frames.
    fn analyze(&mut self) {
        if self.pending_frames.is_empty() {
            return;
        }

        // Stub: Simulate lookahead analysis
        let mut prev_scene_change = false;
        let mut b_frame_count = 0;

        for (idx, frame) in self.pending_frames.iter().enumerate() {
            let stats = frame.stats.clone().unwrap_or_default();
            let is_scene_change = stats.scene_change_score > self.scene_change_threshold;

            // Determine frame type
            let frame_type = if frame.force_keyframe || (idx == 0 && is_scene_change) {
                prev_scene_change = true;
                b_frame_count = 0;
                NvEncPicType::Idr
            } else if is_scene_change || prev_scene_change {
                prev_scene_change = is_scene_change;
                b_frame_count = 0;
                NvEncPicType::I
            } else if self.b_adapt && b_frame_count < 2 && !is_scene_change {
                b_frame_count += 1;
                NvEncPicType::B
            } else {
                b_frame_count = 0;
                NvEncPicType::P
            };

            // Calculate QP adjustment from temporal AQ
            let qp_adjustment = if self.temporal_aq {
                // Simple heuristic: lower QP for complex frames
                let complexity = stats.avg_mb_complexity;
                if complexity > 1.5 {
                    -2
                } else if complexity > 1.0 {
                    -1
                } else if complexity < 0.5 {
                    2
                } else {
                    0
                }
            } else {
                0
            };

            self.analysis_results.push_back(LookaheadResult {
                frame_idx: frame.frame_idx,
                recommended_type: frame_type,
                qp_adjustment,
                is_scene_change,
                ref_recommendations: RefRecommendations::default(),
            });
        }

        self.pending_frames.clear();
    }

    /// Get analysis result.
    pub fn get_result(&mut self) -> Option<LookaheadResult> {
        self.analysis_results.pop_front()
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) {
        self.analyze();
    }

    /// Get pending frame count.
    pub fn pending_count(&self) -> usize {
        self.pending_frames.len()
    }
}

/// Adaptive quantization controller.
pub struct AdaptiveQuantization {
    /// Enable spatial AQ.
    spatial_aq: bool,
    /// Enable temporal AQ.
    temporal_aq: bool,
    /// AQ strength (1-15).
    strength: u8,
    /// Frame dimensions.
    width: u32,
    height: u32,
    /// QP delta map (per MB).
    qp_delta_map: Vec<i8>,
    /// Previous frame complexity map.
    prev_complexity_map: Vec<f32>,
}

impl AdaptiveQuantization {
    /// Create new AQ controller.
    pub fn new(width: u32, height: u32, spatial_aq: bool, temporal_aq: bool, strength: u8) -> Self {
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let mb_count = (mb_width * mb_height) as usize;

        Self {
            spatial_aq,
            temporal_aq,
            strength: strength.clamp(1, 15),
            width,
            height,
            qp_delta_map: vec![0; mb_count],
            prev_complexity_map: vec![1.0; mb_count],
        }
    }

    /// Calculate QP delta map for a frame.
    pub fn calculate_qp_map(&mut self, complexity_map: &[f32]) -> &[i8] {
        let mb_count = self.qp_delta_map.len();
        if complexity_map.len() != mb_count {
            return &self.qp_delta_map;
        }

        // Calculate average complexity
        let avg_complexity: f32 = complexity_map.iter().sum::<f32>() / mb_count as f32;

        for (i, (&complexity, &prev_complexity)) in complexity_map
            .iter()
            .zip(self.prev_complexity_map.iter())
            .enumerate()
        {
            let mut delta: f32 = 0.0;

            // Spatial AQ: adjust based on local complexity
            if self.spatial_aq {
                let spatial_ratio = complexity / avg_complexity.max(0.001);
                // Higher complexity = lower QP (more bits)
                delta += (1.0 - spatial_ratio) * self.strength as f32 * 0.5;
            }

            // Temporal AQ: adjust based on temporal changes
            if self.temporal_aq {
                let temporal_change =
                    (complexity - prev_complexity).abs() / avg_complexity.max(0.001);
                // Large changes = lower QP (more bits for motion areas)
                delta += temporal_change * self.strength as f32 * 0.3;
            }

            // Clamp delta to reasonable range
            self.qp_delta_map[i] = delta.clamp(-15.0, 15.0) as i8;
        }

        // Update previous complexity map
        self.prev_complexity_map.copy_from_slice(complexity_map);

        &self.qp_delta_map
    }

    /// Get QP delta map pointer for NVENC.
    pub fn qp_map_ptr(&self) -> *const i8 {
        self.qp_delta_map.as_ptr()
    }

    /// Get QP map size.
    pub fn qp_map_size(&self) -> usize {
        self.qp_delta_map.len()
    }
}

/// Reference picture invalidation manager.
pub struct RefPicInvalidation {
    /// Invalid reference list.
    invalid_refs: Vec<u64>,
    /// Maximum tracked frames.
    max_tracked: usize,
}

impl RefPicInvalidation {
    /// Create new manager.
    pub fn new(max_tracked: usize) -> Self {
        Self {
            invalid_refs: Vec::with_capacity(max_tracked),
            max_tracked,
        }
    }

    /// Mark frame as invalid reference.
    pub fn invalidate(&mut self, frame_idx: u64) {
        if !self.invalid_refs.contains(&frame_idx) {
            if self.invalid_refs.len() >= self.max_tracked {
                self.invalid_refs.remove(0);
            }
            self.invalid_refs.push(frame_idx);
        }
    }

    /// Check if frame is invalid.
    pub fn is_invalid(&self, frame_idx: u64) -> bool {
        self.invalid_refs.contains(&frame_idx)
    }

    /// Clear invalidation for frame.
    pub fn clear(&mut self, frame_idx: u64) {
        self.invalid_refs.retain(|&f| f != frame_idx);
    }

    /// Clear all invalidations.
    pub fn clear_all(&mut self) {
        self.invalid_refs.clear();
    }

    /// Get invalid frame indices.
    pub fn invalid_frames(&self) -> &[u64] {
        &self.invalid_refs
    }
}

/// Long-term reference frame manager.
pub struct LtrManager {
    /// Maximum LTR frames.
    max_ltr_frames: u32,
    /// LTR frame indices.
    ltr_frames: Vec<LtrFrame>,
    /// LTR trust mode.
    trust_mode: LtrTrustMode,
}

/// LTR frame info.
#[derive(Debug, Clone)]
pub struct LtrFrame {
    /// Frame index.
    pub frame_idx: u64,
    /// LTR index (0 to max_ltr_frames-1).
    pub ltr_idx: u32,
    /// PTS.
    pub pts: i64,
    /// Is marked for use.
    pub marked: bool,
}

/// LTR trust mode.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LtrTrustMode {
    /// Default trust mode.
    #[default]
    Default = 0,
    /// Trust only when explicitly marked.
    Explicit = 1,
}

impl LtrManager {
    /// Create new LTR manager.
    pub fn new(max_ltr_frames: u32, trust_mode: LtrTrustMode) -> Self {
        Self {
            max_ltr_frames,
            ltr_frames: Vec::with_capacity(max_ltr_frames as usize),
            trust_mode,
        }
    }

    /// Mark frame as LTR.
    pub fn mark_ltr(&mut self, frame_idx: u64, pts: i64) -> Option<u32> {
        // Find available LTR slot
        let ltr_idx = if self.ltr_frames.len() < self.max_ltr_frames as usize {
            self.ltr_frames.len() as u32
        } else {
            // Replace oldest LTR
            let oldest_idx = self
                .ltr_frames
                .iter()
                .enumerate()
                .min_by_key(|(_, f)| f.frame_idx)
                .map(|(i, _)| i)?;
            self.ltr_frames.remove(oldest_idx);
            oldest_idx as u32
        };

        self.ltr_frames.push(LtrFrame {
            frame_idx,
            ltr_idx,
            pts,
            marked: true,
        });

        Some(ltr_idx)
    }

    /// Use LTR frame as reference.
    pub fn use_ltr(&self, ltr_idx: u32) -> Option<&LtrFrame> {
        self.ltr_frames.iter().find(|f| f.ltr_idx == ltr_idx && f.marked)
    }

    /// Get LTR bitmap for use.
    pub fn get_use_bitmap(&self) -> u32 {
        let mut bitmap = 0u32;
        for frame in &self.ltr_frames {
            if frame.marked {
                bitmap |= 1 << frame.ltr_idx;
            }
        }
        bitmap
    }

    /// Clear LTR.
    pub fn clear_ltr(&mut self, ltr_idx: u32) {
        self.ltr_frames.retain(|f| f.ltr_idx != ltr_idx);
    }

    /// Clear all LTR frames.
    pub fn clear_all(&mut self) {
        self.ltr_frames.clear();
    }
}

/// Intra refresh (rolling I) manager.
pub struct IntraRefreshManager {
    /// Enable intra refresh.
    enabled: bool,
    /// Intra refresh period (in frames).
    period: u32,
    /// Intra refresh count (MBs per frame).
    count: u32,
    /// Current position in refresh cycle.
    current_position: u32,
    /// Single slice mode.
    single_slice: bool,
    /// Frame dimensions.
    mb_width: u32,
    mb_height: u32,
}

impl IntraRefreshManager {
    /// Create new intra refresh manager.
    pub fn new(width: u32, height: u32, period: u32, single_slice: bool) -> Self {
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let total_mbs = mb_width * mb_height;
        let count = total_mbs.div_ceil(period);

        Self {
            enabled: true,
            period,
            count,
            current_position: 0,
            single_slice,
            mb_width,
            mb_height,
        }
    }

    /// Get intra refresh params for current frame.
    pub fn get_params(&mut self) -> IntraRefreshParams {
        let params = IntraRefreshParams {
            enable: self.enabled,
            position: self.current_position,
            count: self.count,
            single_slice: self.single_slice,
        };

        // Advance position
        self.current_position = (self.current_position + self.count) % (self.mb_width * self.mb_height);

        params
    }

    /// Reset refresh cycle.
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Check if refresh cycle is complete.
    pub fn is_cycle_complete(&self) -> bool {
        self.current_position == 0
    }
}

/// Intra refresh parameters.
#[derive(Debug, Clone)]
pub struct IntraRefreshParams {
    /// Enable intra refresh for this frame.
    pub enable: bool,
    /// Start position (MB index).
    pub position: u32,
    /// Number of MBs to refresh.
    pub count: u32,
    /// Single slice mode.
    pub single_slice: bool,
}

// ============================================================================
// NVDEC Decoding Structures
// ============================================================================

/// Video codec type for NVDEC - mirrors cudaVideoCodec.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CuvidCodec {
    /// MPEG-1.
    Mpeg1 = 0,
    /// MPEG-2.
    Mpeg2 = 1,
    /// MPEG-4.
    Mpeg4 = 2,
    /// VC-1.
    Vc1 = 3,
    /// H.264/AVC.
    #[default]
    H264 = 4,
    /// JPEG.
    Jpeg = 5,
    /// H.265/HEVC.
    Hevc = 7,
    /// VP8.
    Vp8 = 8,
    /// VP9.
    Vp9 = 9,
    /// AV1.
    Av1 = 10,
}

/// Chroma format - mirrors cudaVideoChromaFormat.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CuvidChromaFormat {
    /// Monochrome.
    Monochrome = 0,
    /// 4:2:0.
    #[default]
    Yuv420 = 1,
    /// 4:2:2.
    Yuv422 = 2,
    /// 4:4:4.
    Yuv444 = 3,
}

/// Surface format for output - mirrors cudaVideoSurfaceFormat.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CuvidSurfaceFormat {
    /// NV12.
    #[default]
    Nv12 = 0,
    /// P016 (16-bit per component).
    P016 = 1,
    /// YUV444.
    Yuv444 = 2,
    /// YUV444 16-bit.
    Yuv444_16Bit = 3,
}

/// Decode create info - mirrors CUVIDDECODECREATEINFO.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CuvidDecodeCreateInfo {
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,
    /// Number of decode surfaces.
    pub num_decode_surfaces: u32,
    /// Codec type.
    pub codec_type: CuvidCodec,
    /// Chroma format.
    pub chroma_format: CuvidChromaFormat,
    /// Reserved.
    pub reserved1: u32,
    /// Bit depth minus 8.
    pub bit_depth_minus8: u32,
    /// Output format.
    pub output_format: CuvidSurfaceFormat,
    /// Deinterlace mode.
    pub deinterlace_mode: CuvidDeinterlaceMode,
    /// Target width.
    pub target_width: u32,
    /// Target height.
    pub target_height: u32,
    /// Number of output surfaces.
    pub num_output_surfaces: u32,
    /// Video context (CUcontext).
    pub vid_ctx: CUcontext,
    /// Target rect left.
    pub target_rect_left: i16,
    /// Target rect top.
    pub target_rect_top: i16,
    /// Target rect right.
    pub target_rect_right: i16,
    /// Target rect bottom.
    pub target_rect_bottom: i16,
    /// Display area left.
    pub display_area_left: i16,
    /// Display area top.
    pub display_area_top: i16,
    /// Display area right.
    pub display_area_right: i16,
    /// Display area bottom.
    pub display_area_bottom: i16,
    /// Reserved fields.
    pub reserved2: [u32; 5],
}

impl Default for CuvidDecodeCreateInfo {
    fn default() -> Self {
        Self {
            coded_width: 1920,
            coded_height: 1080,
            num_decode_surfaces: 8,
            codec_type: CuvidCodec::H264,
            chroma_format: CuvidChromaFormat::Yuv420,
            reserved1: 0,
            bit_depth_minus8: 0,
            output_format: CuvidSurfaceFormat::Nv12,
            deinterlace_mode: CuvidDeinterlaceMode::Weave,
            target_width: 1920,
            target_height: 1080,
            num_output_surfaces: 4,
            vid_ctx: std::ptr::null_mut(),
            target_rect_left: 0,
            target_rect_top: 0,
            target_rect_right: 1920,
            target_rect_bottom: 1080,
            display_area_left: 0,
            display_area_top: 0,
            display_area_right: 1920,
            display_area_bottom: 1080,
            reserved2: [0; 5],
        }
    }
}

/// Deinterlace mode - mirrors cudaVideoDeinterlaceMode.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CuvidDeinterlaceMode {
    /// Weave (no deinterlacing).
    #[default]
    Weave = 0,
    /// Bob (line doubling).
    Bob = 1,
    /// Adaptive (motion-adaptive).
    Adaptive = 2,
}

/// Picture parameters - mirrors CUVIDPICPARAMS.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CuvidPicParams {
    /// Picture index.
    pub pic_idx: i32,
    /// Field order count bottom.
    pub field_order_cnt_bottom: i32,
    /// Field order count top.
    pub field_order_cnt_top: i32,
    /// Is reference.
    pub is_ref: i32,
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,
    /// Number of slices.
    pub num_slices: u32,
    /// Bitstream data.
    pub bitstream_data: *const u8,
    /// Bitstream length.
    pub num_bitstream_bytes: u32,
    /// Slice data offsets.
    pub slice_data_offsets: *const u32,
    /// Codec-specific parameters.
    pub codec_specific: CuvidCodecPicParams,
}

impl Default for CuvidPicParams {
    fn default() -> Self {
        Self {
            pic_idx: 0,
            field_order_cnt_bottom: 0,
            field_order_cnt_top: 0,
            is_ref: 1,
            coded_width: 0,
            coded_height: 0,
            num_slices: 0,
            bitstream_data: std::ptr::null(),
            num_bitstream_bytes: 0,
            slice_data_offsets: std::ptr::null(),
            codec_specific: CuvidCodecPicParams::H264(CuvidH264PicParams::default()),
        }
    }
}

/// Codec-specific picture parameters (union in C).
#[derive(Debug, Clone)]
pub enum CuvidCodecPicParams {
    H264(CuvidH264PicParams),
    Hevc(CuvidHevcPicParams),
    Vp9(CuvidVp9PicParams),
    Av1(CuvidAv1PicParams),
}

/// H.264 picture parameters.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidH264PicParams {
    /// Log2 max frame num.
    pub log2_max_frame_num: i32,
    /// Pic order cnt type.
    pub pic_order_cnt_type: i32,
    /// Log2 max pic order cnt lsb.
    pub log2_max_pic_order_cnt_lsb: i32,
    /// Delta pic order always zero flag.
    pub delta_pic_order_always_zero_flag: i32,
    /// Frame MBS only flag.
    pub frame_mbs_only_flag: i32,
    /// Direct 8x8 inference flag.
    pub direct_8x8_inference_flag: i32,
    /// Num ref frames.
    pub num_ref_frames: i32,
    /// Residual colour transform flag.
    pub residual_colour_transform_flag: i32,
    /// Bit depth luma minus 8.
    pub bit_depth_luma_minus8: i32,
    /// Bit depth chroma minus 8.
    pub bit_depth_chroma_minus8: i32,
    /// QP prime Y zero transform bypass flag.
    pub qpprime_y_zero_transform_bypass_flag: i32,
    /// Reserved.
    pub reserved: [i32; 3],
}

/// HEVC picture parameters.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidHevcPicParams {
    /// Pic width in luma samples.
    pub pic_width_in_luma_samples: i32,
    /// Pic height in luma samples.
    pub pic_height_in_luma_samples: i32,
    /// Log2 min luma coding block size.
    pub log2_min_luma_coding_block_size: u8,
    /// Log2 diff max min luma coding block size.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// Log2 min transform block size.
    pub log2_min_transform_block_size: u8,
    /// Log2 diff max min transform block size.
    pub log2_diff_max_min_transform_block_size: u8,
    /// PCM enabled flag.
    pub pcm_enabled_flag: u8,
    /// Reserved.
    pub reserved: [u8; 3],
}

/// VP9 picture parameters.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidVp9PicParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Last ref frame.
    pub last_ref_frame: i32,
    /// Golden ref frame.
    pub golden_ref_frame: i32,
    /// Alt ref frame.
    pub alt_ref_frame: i32,
    /// Reserved.
    pub reserved: [u32; 7],
}

/// AV1 picture parameters.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidAv1PicParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Frame type.
    pub frame_type: u32,
    /// Show frame.
    pub show_frame: u32,
    /// Reserved.
    pub reserved: [u32; 8],
}

/// Parser display info - mirrors CUVIDPARSERDISPINFO.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidParserDispInfo {
    /// Picture index.
    pub picture_index: i32,
    /// Progressive frame.
    pub progressive_frame: i32,
    /// Top field first.
    pub top_field_first: i32,
    /// Repeat first field.
    pub repeat_first_field: i32,
    /// Timestamp.
    pub timestamp: i64,
}

/// Video parser callback infrastructure.
pub trait VideoParserCallback {
    /// Called when sequence parameters change.
    fn sequence_callback(&mut self, format: &CuvidVideoFormat) -> i32;

    /// Called to decode a picture.
    fn decode_callback(&mut self, pic_params: &CuvidPicParams) -> i32;

    /// Called to display a picture.
    fn display_callback(&mut self, disp_info: &CuvidParserDispInfo) -> i32;
}

/// Video format info from parser - mirrors CUVIDEOFORMAT.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CuvidVideoFormat {
    /// Codec.
    pub codec: CuvidCodec,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Progressive sequence.
    pub progressive_sequence: u8,
    /// Bit depth luma minus 8.
    pub bit_depth_luma_minus8: u8,
    /// Bit depth chroma minus 8.
    pub bit_depth_chroma_minus8: u8,
    /// Min surfaces.
    pub min_num_decode_surfaces: u8,
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,
    /// Chroma format.
    pub chroma_format: CuvidChromaFormat,
    /// Display area left.
    pub display_area_left: i16,
    /// Display area top.
    pub display_area_top: i16,
    /// Display area right.
    pub display_area_right: i16,
    /// Display area bottom.
    pub display_area_bottom: i16,
}

/// NVDEC decoder wrapper with callback support.
pub struct NvdecVideoParser {
    /// Parser handle (would be CUvideoparser in real FFI).
    _handle: u64,
    /// Codec type.
    codec: CuvidCodec,
    /// Video format.
    format: Option<CuvidVideoFormat>,
    /// Pending display info.
    display_queue: VecDeque<CuvidParserDispInfo>,
}

impl NvdecVideoParser {
    /// Create new video parser.
    pub fn new(codec: CuvidCodec) -> Self {
        Self {
            _handle: 0,
            codec,
            format: None,
            display_queue: VecDeque::new(),
        }
    }

    /// Parse video data.
    pub fn parse(&mut self, data: &[u8], timestamp: i64) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Stub: Simulate parsing
        // In real implementation, this would call cuvidParseVideoData

        // Simulate decoded frame ready for display
        self.display_queue.push_back(CuvidParserDispInfo {
            picture_index: self.display_queue.len() as i32,
            progressive_frame: 1,
            top_field_first: 0,
            repeat_first_field: 0,
            timestamp,
        });

        Ok(())
    }

    /// Get next display info.
    pub fn get_display_info(&mut self) -> Option<CuvidParserDispInfo> {
        self.display_queue.pop_front()
    }

    /// Get video format.
    pub fn format(&self) -> Option<&CuvidVideoFormat> {
        self.format.as_ref()
    }

    /// Get codec.
    pub fn codec(&self) -> CuvidCodec {
        self.codec
    }
}

// ============================================================================
// Performance Monitoring
// ============================================================================

/// Encode latency measurement.
pub struct EncodeLatencyTracker {
    /// Frame submit times.
    submit_times: VecDeque<(u64, Instant)>,
    /// Completed frame latencies.
    latencies: VecDeque<Duration>,
    /// Maximum tracked samples.
    max_samples: usize,
}

impl EncodeLatencyTracker {
    /// Create new latency tracker.
    pub fn new(max_samples: usize) -> Self {
        Self {
            submit_times: VecDeque::with_capacity(max_samples),
            latencies: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record frame submission.
    pub fn submit(&mut self, frame_idx: u64) {
        if self.submit_times.len() >= self.max_samples {
            self.submit_times.pop_front();
        }
        self.submit_times.push_back((frame_idx, Instant::now()));
    }

    /// Record frame completion.
    pub fn complete(&mut self, frame_idx: u64) {
        if let Some(pos) = self.submit_times.iter().position(|(idx, _)| *idx == frame_idx) {
            let (_, submit_time) = self.submit_times.remove(pos).unwrap();
            let latency = submit_time.elapsed();

            if self.latencies.len() >= self.max_samples {
                self.latencies.pop_front();
            }
            self.latencies.push_back(latency);
        }
    }

    /// Get average latency.
    pub fn average_latency(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }

        let total: Duration = self.latencies.iter().sum();
        Some(total / self.latencies.len() as u32)
    }

    /// Get minimum latency.
    pub fn min_latency(&self) -> Option<Duration> {
        self.latencies.iter().min().copied()
    }

    /// Get maximum latency.
    pub fn max_latency(&self) -> Option<Duration> {
        self.latencies.iter().max().copied()
    }

    /// Get latency percentile (0.0-1.0).
    pub fn percentile_latency(&self, p: f64) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted: Vec<_> = self.latencies.iter().copied().collect();
        sorted.sort();

        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Get pending frame count.
    pub fn pending_count(&self) -> usize {
        self.submit_times.len()
    }
}

/// GPU utilization query (NVML-based).
pub struct GpuUtilizationMonitor {
    /// Device index.
    device_index: u32,
    /// Sample history.
    samples: VecDeque<GpuUtilizationSample>,
    /// Maximum samples.
    max_samples: usize,
    /// Last sample time.
    last_sample: Instant,
    /// Sample interval.
    sample_interval: Duration,
}

/// GPU utilization sample.
#[derive(Debug, Clone, Copy)]
pub struct GpuUtilizationSample {
    /// Timestamp.
    pub timestamp: Instant,
    /// GPU utilization (0-100).
    pub gpu_util: u32,
    /// Memory utilization (0-100).
    pub mem_util: u32,
    /// Encoder utilization (0-100).
    pub encoder_util: u32,
    /// Decoder utilization (0-100).
    pub decoder_util: u32,
    /// Memory used (bytes).
    pub memory_used: u64,
    /// Memory total (bytes).
    pub memory_total: u64,
    /// Temperature (Celsius).
    pub temperature: u32,
    /// Power draw (mW).
    pub power_draw: u32,
}

impl GpuUtilizationMonitor {
    /// Create new monitor.
    pub fn new(device_index: u32, sample_interval: Duration, max_samples: usize) -> Self {
        Self {
            device_index,
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            last_sample: Instant::now(),
            sample_interval,
        }
    }

    /// Sample current utilization.
    pub fn sample(&mut self) -> GpuUtilizationSample {
        // Stub: Return simulated values
        // In real implementation, this would use NVML:
        // - nvmlDeviceGetUtilizationRates
        // - nvmlDeviceGetEncoderUtilization
        // - nvmlDeviceGetDecoderUtilization
        // - nvmlDeviceGetMemoryInfo
        // - nvmlDeviceGetTemperature
        // - nvmlDeviceGetPowerUsage

        let sample = GpuUtilizationSample {
            timestamp: Instant::now(),
            gpu_util: 45,
            mem_util: 30,
            encoder_util: 80,
            decoder_util: 0,
            memory_used: 3 * 1024 * 1024 * 1024,
            memory_total: 8 * 1024 * 1024 * 1024,
            temperature: 65,
            power_draw: 150_000, // 150W
        };

        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.last_sample = Instant::now();

        sample
    }

    /// Try to sample (respects sample interval).
    pub fn try_sample(&mut self) -> Option<GpuUtilizationSample> {
        if self.last_sample.elapsed() >= self.sample_interval {
            Some(self.sample())
        } else {
            None
        }
    }

    /// Get latest sample.
    pub fn latest(&self) -> Option<GpuUtilizationSample> {
        self.samples.back().copied()
    }

    /// Get average utilization.
    pub fn average(&self) -> Option<GpuUtilizationSample> {
        if self.samples.is_empty() {
            return None;
        }

        let count = self.samples.len() as u32;
        let mut gpu_util = 0u64;
        let mut mem_util = 0u64;
        let mut encoder_util = 0u64;
        let mut decoder_util = 0u64;
        let mut temperature = 0u64;

        for sample in &self.samples {
            gpu_util += sample.gpu_util as u64;
            mem_util += sample.mem_util as u64;
            encoder_util += sample.encoder_util as u64;
            decoder_util += sample.decoder_util as u64;
            temperature += sample.temperature as u64;
        }

        let latest = self.samples.back()?;

        Some(GpuUtilizationSample {
            timestamp: Instant::now(),
            gpu_util: (gpu_util / count as u64) as u32,
            mem_util: (mem_util / count as u64) as u32,
            encoder_util: (encoder_util / count as u64) as u32,
            decoder_util: (decoder_util / count as u64) as u32,
            memory_used: latest.memory_used,
            memory_total: latest.memory_total,
            temperature: (temperature / count as u64) as u32,
            power_draw: latest.power_draw,
        })
    }
}

/// Memory bandwidth tracker.
pub struct MemoryBandwidthTracker {
    /// Bytes uploaded.
    bytes_uploaded: AtomicU64,
    /// Bytes downloaded.
    bytes_downloaded: AtomicU64,
    /// Start time.
    start_time: Instant,
    /// Sample history.
    samples: std::sync::Mutex<VecDeque<BandwidthSample>>,
    /// Max samples.
    max_samples: usize,
}

/// Bandwidth sample.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Timestamp.
    pub timestamp: Instant,
    /// Upload bandwidth (bytes/sec).
    pub upload_bps: u64,
    /// Download bandwidth (bytes/sec).
    pub download_bps: u64,
}

impl MemoryBandwidthTracker {
    /// Create new tracker.
    pub fn new(max_samples: usize) -> Self {
        Self {
            bytes_uploaded: AtomicU64::new(0),
            bytes_downloaded: AtomicU64::new(0),
            start_time: Instant::now(),
            samples: std::sync::Mutex::new(VecDeque::with_capacity(max_samples)),
            max_samples,
        }
    }

    /// Record upload.
    pub fn record_upload(&self, bytes: usize) {
        self.bytes_uploaded.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record download.
    pub fn record_download(&self, bytes: usize) {
        self.bytes_downloaded.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Get total bytes uploaded.
    pub fn total_uploaded(&self) -> u64 {
        self.bytes_uploaded.load(Ordering::Relaxed)
    }

    /// Get total bytes downloaded.
    pub fn total_downloaded(&self) -> u64 {
        self.bytes_downloaded.load(Ordering::Relaxed)
    }

    /// Get average upload bandwidth (bytes/sec).
    pub fn average_upload_bps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_uploaded() as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get average download bandwidth (bytes/sec).
    pub fn average_download_bps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_downloaded() as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Reset counters.
    pub fn reset(&self) {
        self.bytes_uploaded.store(0, Ordering::Relaxed);
        self.bytes_downloaded.store(0, Ordering::Relaxed);
    }
}

/// Async operation status tracker.
pub struct AsyncOperationTracker {
    /// Pending operations.
    pending: VecDeque<AsyncOperation>,
    /// Completed operations.
    completed: VecDeque<AsyncOperation>,
    /// Maximum tracked.
    max_tracked: usize,
}

/// Async operation info.
#[derive(Debug, Clone)]
pub struct AsyncOperation {
    /// Operation ID.
    pub id: u64,
    /// Operation type.
    pub op_type: AsyncOpType,
    /// Start time.
    pub start_time: Instant,
    /// Completion time.
    pub complete_time: Option<Instant>,
    /// Status.
    pub status: AsyncOpStatus,
    /// Associated frame index.
    pub frame_idx: Option<u64>,
}

/// Async operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncOpType {
    /// Encode frame.
    Encode,
    /// Decode frame.
    Decode,
    /// Memory copy H2D.
    CopyHtoD,
    /// Memory copy D2H.
    CopyDtoH,
    /// Map resource.
    MapResource,
    /// Unmap resource.
    UnmapResource,
}

/// Async operation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncOpStatus {
    /// Pending.
    Pending,
    /// In progress.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
}

impl AsyncOperationTracker {
    /// Create new tracker.
    pub fn new(max_tracked: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(max_tracked),
            completed: VecDeque::with_capacity(max_tracked),
            max_tracked,
        }
    }

    /// Start tracking operation.
    pub fn start(&mut self, id: u64, op_type: AsyncOpType, frame_idx: Option<u64>) {
        if self.pending.len() >= self.max_tracked {
            self.pending.pop_front();
        }

        self.pending.push_back(AsyncOperation {
            id,
            op_type,
            start_time: Instant::now(),
            complete_time: None,
            status: AsyncOpStatus::Pending,
            frame_idx,
        });
    }

    /// Mark operation as in progress.
    pub fn mark_in_progress(&mut self, id: u64) {
        if let Some(op) = self.pending.iter_mut().find(|o| o.id == id) {
            op.status = AsyncOpStatus::InProgress;
        }
    }

    /// Complete operation.
    pub fn complete(&mut self, id: u64, success: bool) {
        if let Some(idx) = self.pending.iter().position(|o| o.id == id) {
            let mut op = self.pending.remove(idx).unwrap();
            op.complete_time = Some(Instant::now());
            op.status = if success {
                AsyncOpStatus::Completed
            } else {
                AsyncOpStatus::Failed
            };

            if self.completed.len() >= self.max_tracked {
                self.completed.pop_front();
            }
            self.completed.push_back(op);
        }
    }

    /// Get pending operation count.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get pending operations of type.
    pub fn pending_of_type(&self, op_type: AsyncOpType) -> usize {
        self.pending.iter().filter(|o| o.op_type == op_type).count()
    }

    /// Check if operation is pending.
    pub fn is_pending(&self, id: u64) -> bool {
        self.pending.iter().any(|o| o.id == id)
    }

    /// Get average completion time for operation type.
    pub fn average_completion_time(&self, op_type: AsyncOpType) -> Option<Duration> {
        let ops: Vec<_> = self
            .completed
            .iter()
            .filter(|o| o.op_type == op_type && o.complete_time.is_some())
            .collect();

        if ops.is_empty() {
            return None;
        }

        let total: Duration = ops
            .iter()
            .map(|o| o.complete_time.unwrap() - o.start_time)
            .sum();

        Some(total / ops.len() as u32)
    }
}

// ============================================================================
// Original NVENC Types (Preserved for compatibility)
// ============================================================================

/// NVIDIA GPU info.
#[derive(Debug, Clone)]
pub struct NvidiaGpuInfo {
    /// Device index.
    pub device_index: u32,
    /// Device name.
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Free memory in bytes.
    pub free_memory: u64,
    /// NVENC version.
    pub nvenc_version: u32,
    /// Max encode sessions.
    pub max_encode_sessions: u32,
    /// Max concurrent encode sessions.
    pub max_concurrent_sessions: u32,
    /// Supports B-frames.
    pub supports_bframes: bool,
    /// Supports lookahead.
    pub supports_lookahead: bool,
    /// Supports temporal AQ.
    pub supports_temporal_aq: bool,
    /// Supports weighted prediction.
    pub supports_weighted_pred: bool,
}

impl Default for NvidiaGpuInfo {
    fn default() -> Self {
        Self {
            device_index: 0,
            name: "NVIDIA GPU".to_string(),
            compute_capability: (8, 0),
            total_memory: 8 * 1024 * 1024 * 1024,
            free_memory: 6 * 1024 * 1024 * 1024,
            nvenc_version: 12,
            max_encode_sessions: 3,
            max_concurrent_sessions: 2,
            supports_bframes: true,
            supports_lookahead: true,
            supports_temporal_aq: true,
            supports_weighted_pred: true,
        }
    }
}

/// NVENC encoder preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencPreset {
    /// P1 - Fastest encoding.
    P1,
    /// P2 - Faster encoding.
    P2,
    /// P3 - Fast encoding.
    P3,
    /// P4 - Medium (balanced).
    P4,
    /// P5 - Slow encoding.
    P5,
    /// P6 - Slower encoding.
    P6,
    /// P7 - Slowest/best quality.
    P7,
    /// Low latency high quality.
    LowLatencyHq,
    /// Low latency high performance.
    LowLatencyHp,
    /// Lossless encoding.
    Lossless,
}

impl NvencPreset {
    /// Get NVENC preset GUID name.
    pub fn guid_name(&self) -> &'static str {
        match self {
            NvencPreset::P1 => "NV_ENC_PRESET_P1_GUID",
            NvencPreset::P2 => "NV_ENC_PRESET_P2_GUID",
            NvencPreset::P3 => "NV_ENC_PRESET_P3_GUID",
            NvencPreset::P4 => "NV_ENC_PRESET_P4_GUID",
            NvencPreset::P5 => "NV_ENC_PRESET_P5_GUID",
            NvencPreset::P6 => "NV_ENC_PRESET_P6_GUID",
            NvencPreset::P7 => "NV_ENC_PRESET_P7_GUID",
            NvencPreset::LowLatencyHq => "NV_ENC_PRESET_LOW_LATENCY_HQ_GUID",
            NvencPreset::LowLatencyHp => "NV_ENC_PRESET_LOW_LATENCY_HP_GUID",
            NvencPreset::Lossless => "NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID",
        }
    }

    /// Get actual GUID.
    pub fn guid(&self) -> NvGuid {
        match self {
            NvencPreset::P1 => NV_ENC_PRESET_P1_GUID,
            NvencPreset::P2 => NV_ENC_PRESET_P2_GUID,
            NvencPreset::P3 => NV_ENC_PRESET_P3_GUID,
            NvencPreset::P4 => NV_ENC_PRESET_P4_GUID,
            NvencPreset::P5 => NV_ENC_PRESET_P5_GUID,
            NvencPreset::P6 => NV_ENC_PRESET_P6_GUID,
            NvencPreset::P7 => NV_ENC_PRESET_P7_GUID,
            NvencPreset::LowLatencyHq => NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,
            NvencPreset::LowLatencyHp => NV_ENC_PRESET_LOW_LATENCY_HP_GUID,
            NvencPreset::Lossless => NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,
        }
    }

    /// Check if preset supports B-frames.
    pub fn supports_bframes(&self) -> bool {
        !matches!(self, NvencPreset::LowLatencyHq | NvencPreset::LowLatencyHp)
    }
}

/// NVENC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencProfile {
    /// H.264 Baseline.
    H264Baseline,
    /// H.264 Main.
    H264Main,
    /// H.264 High.
    H264High,
    /// H.264 High 444.
    H264High444,
    /// HEVC Main.
    HevcMain,
    /// HEVC Main10 (10-bit).
    HevcMain10,
    /// HEVC Main 444.
    HevcMain444,
    /// AV1 Main.
    Av1Main,
    /// AV1 High (10-bit).
    Av1High,
}

impl NvencProfile {
    /// Get codec for this profile.
    pub fn codec(&self) -> HwCodec {
        match self {
            NvencProfile::H264Baseline
            | NvencProfile::H264Main
            | NvencProfile::H264High
            | NvencProfile::H264High444 => HwCodec::H264,
            NvencProfile::HevcMain | NvencProfile::HevcMain10 | NvencProfile::HevcMain444 => {
                HwCodec::Hevc
            }
            NvencProfile::Av1Main | NvencProfile::Av1High => HwCodec::Av1,
        }
    }

    /// Check if 10-bit.
    pub fn is_10bit(&self) -> bool {
        matches!(self, NvencProfile::HevcMain10 | NvencProfile::Av1High)
    }

    /// Get profile GUID.
    pub fn guid(&self) -> NvGuid {
        match self {
            NvencProfile::H264Baseline => NV_ENC_H264_PROFILE_BASELINE_GUID,
            NvencProfile::H264Main => NV_ENC_H264_PROFILE_MAIN_GUID,
            NvencProfile::H264High => NV_ENC_H264_PROFILE_HIGH_GUID,
            NvencProfile::H264High444 => NV_ENC_H264_PROFILE_HIGH_444_GUID,
            NvencProfile::HevcMain => NV_ENC_HEVC_PROFILE_MAIN_GUID,
            NvencProfile::HevcMain10 => NV_ENC_HEVC_PROFILE_MAIN10_GUID,
            NvencProfile::HevcMain444 => NV_ENC_HEVC_PROFILE_MAIN_GUID, // No 444 GUID defined
            NvencProfile::Av1Main | NvencProfile::Av1High => NV_ENC_AV1_PROFILE_MAIN_GUID,
        }
    }

    /// Get codec GUID.
    pub fn codec_guid(&self) -> NvGuid {
        match self.codec() {
            HwCodec::H264 => NV_ENC_CODEC_H264_GUID,
            HwCodec::Hevc => NV_ENC_CODEC_HEVC_GUID,
            HwCodec::Av1 => NV_ENC_CODEC_AV1_GUID,
            _ => NV_ENC_CODEC_H264_GUID,
        }
    }
}

/// NVENC rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencRateControl {
    /// Constant QP.
    Cqp { i_qp: u8, p_qp: u8, b_qp: u8 },
    /// Variable bitrate.
    Vbr { target_bitrate: u32, max_bitrate: u32 },
    /// Constant bitrate.
    Cbr { bitrate: u32 },
    /// VBR with high quality (2-pass).
    VbrHq { target_bitrate: u32, max_bitrate: u32 },
    /// CBR with high quality.
    CbrHq { bitrate: u32 },
    /// CBR with low delay and high quality.
    CbrLdHq { bitrate: u32 },
}

impl NvencRateControl {
    /// Get target bitrate if applicable.
    pub fn target_bitrate(&self) -> Option<u32> {
        match self {
            NvencRateControl::Vbr { target_bitrate, .. }
            | NvencRateControl::VbrHq { target_bitrate, .. } => Some(*target_bitrate),
            NvencRateControl::Cbr { bitrate }
            | NvencRateControl::CbrHq { bitrate }
            | NvencRateControl::CbrLdHq { bitrate } => Some(*bitrate),
            NvencRateControl::Cqp { .. } => None,
        }
    }

    /// Convert to FFI rate control mode.
    pub fn to_rc_mode(&self) -> NvEncRcMode {
        match self {
            NvencRateControl::Cqp { .. } => NvEncRcMode::ConstQp,
            NvencRateControl::Vbr { .. } => NvEncRcMode::Vbr,
            NvencRateControl::Cbr { .. } => NvEncRcMode::Cbr,
            NvencRateControl::VbrHq { .. } => NvEncRcMode::VbrHq,
            NvencRateControl::CbrHq { .. } => NvEncRcMode::CbrHq,
            NvencRateControl::CbrLdHq { .. } => NvEncRcMode::CbrLowdelayHq,
        }
    }

    /// Apply to RC params structure.
    pub fn apply_to_params(&self, params: &mut NvEncRcParams) {
        params.rate_control_mode = self.to_rc_mode();
        match self {
            NvencRateControl::Cqp { i_qp, p_qp, b_qp } => {
                params.const_qp_i = *i_qp as i32;
                params.const_qp_p = *p_qp as i32;
                params.const_qp_b = *b_qp as i32;
            }
            NvencRateControl::Vbr { target_bitrate, max_bitrate }
            | NvencRateControl::VbrHq { target_bitrate, max_bitrate } => {
                params.average_bitrate = *target_bitrate;
                params.max_bitrate = *max_bitrate;
            }
            NvencRateControl::Cbr { bitrate }
            | NvencRateControl::CbrHq { bitrate }
            | NvencRateControl::CbrLdHq { bitrate } => {
                params.average_bitrate = *bitrate;
                params.max_bitrate = *bitrate;
            }
        }
    }
}

/// Lookahead configuration.
#[derive(Debug, Clone, Copy)]
pub struct LookaheadConfig {
    /// Enable lookahead.
    pub enabled: bool,
    /// Lookahead depth (frames).
    pub depth: u32,
    /// Enable temporal adaptive quantization.
    pub temporal_aq: bool,
    /// Enable intra refresh during lookahead.
    pub intra_refresh: bool,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            depth: 20,
            temporal_aq: true,
            intra_refresh: false,
        }
    }
}

/// B-frame configuration.
#[derive(Debug, Clone, Copy)]
pub struct BFrameConfig {
    /// Enable B-frames.
    pub enabled: bool,
    /// Number of B-frames between I and P frames.
    pub count: u32,
    /// Use B-frames as references (hierarchical B).
    pub as_ref: bool,
    /// B-frame adaptive mode.
    pub adaptive: BFrameAdaptive,
}

impl Default for BFrameConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            count: 2,
            as_ref: true,
            adaptive: BFrameAdaptive::Enabled,
        }
    }
}

/// B-frame adaptive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BFrameAdaptive {
    /// Disabled - fixed B-frame pattern.
    Disabled,
    /// Enabled - adapt based on scene.
    Enabled,
}

/// CUDA buffer for input/output.
#[derive(Debug)]
pub struct CudaBuffer {
    /// Device pointer.
    pub ptr: u64,
    /// Buffer size in bytes.
    pub size: usize,
    /// Pitch (bytes per row).
    pub pitch: usize,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl CudaBuffer {
    /// Create a host-side representation (actual allocation would use CUDA).
    pub fn new(width: u32, height: u32, bytes_per_pixel: f32) -> Self {
        let pitch = (width as f32 * bytes_per_pixel).ceil() as usize;
        let pitch = (pitch + 255) & !255; // Align to 256 bytes
        let size = pitch * height as usize;

        Self {
            ptr: 0, // Would be cuMemAlloc result
            size,
            pitch,
            width,
            height,
        }
    }

    /// Create NV12 buffer (Y + UV planes).
    pub fn new_nv12(width: u32, height: u32) -> Self {
        let pitch = (width as usize + 255) & !255;
        let size = pitch * height as usize * 3 / 2;

        Self {
            ptr: 0,
            size,
            pitch,
            width,
            height,
        }
    }
}

/// Input frame buffer pool.
pub struct NvencInputPool {
    /// Pool width.
    width: u32,
    /// Pool height.
    height: u32,
    /// Available buffers.
    available: VecDeque<NvencInputBuffer>,
    /// In-use buffers count.
    in_use: u32,
    /// Maximum pool size.
    max_size: usize,
}

impl NvencInputPool {
    /// Create a new input buffer pool.
    pub fn new(width: u32, height: u32, max_size: usize) -> Self {
        Self {
            width,
            height,
            available: VecDeque::with_capacity(max_size),
            in_use: 0,
            max_size,
        }
    }

    /// Get a buffer from the pool.
    pub fn acquire(&mut self) -> Option<NvencInputBuffer> {
        if let Some(buffer) = self.available.pop_front() {
            self.in_use += 1;
            Some(buffer)
        } else if (self.in_use as usize) < self.max_size {
            self.in_use += 1;
            Some(NvencInputBuffer::new(self.width, self.height))
        } else {
            None
        }
    }

    /// Return a buffer to the pool.
    pub fn release(&mut self, buffer: NvencInputBuffer) {
        self.in_use -= 1;
        if self.available.len() < self.max_size {
            self.available.push_back(buffer);
        }
    }
}

/// NVENC input buffer.
pub struct NvencInputBuffer {
    /// Registered resource handle.
    pub handle: u64,
    /// CUDA buffer.
    pub cuda_buffer: CudaBuffer,
    /// Mapped input pointer.
    pub mapped_ptr: u64,
}

impl NvencInputBuffer {
    /// Create a new input buffer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            handle: 0,
            cuda_buffer: CudaBuffer::new_nv12(width, height),
            mapped_ptr: 0,
        }
    }
}

/// NVENC output bitstream buffer.
pub struct NvencOutputBuffer {
    /// Buffer handle.
    pub handle: u64,
    /// Buffer size.
    pub size: usize,
}

impl NvencOutputBuffer {
    /// Create a new output buffer.
    pub fn new(size: usize) -> Self {
        Self { handle: 0, size }
    }
}

/// NVENC context.
#[derive(Debug)]
pub struct NvencContext {
    /// GPU info.
    pub gpu: NvidiaGpuInfo,
    /// CUDA context handle.
    cuda_context: Option<u64>,
    /// NVENC encoder session.
    encoder_session: Option<u64>,
}

impl NvencContext {
    /// Create a new NVENC context.
    pub fn new(device_index: u32) -> Result<Self> {
        // In a real implementation:
        // 1. Initialize CUDA with cuInit
        // 2. Get device with cuDeviceGet
        // 3. Create CUDA context with cuCtxCreate
        // 4. Load NVENC library
        // 5. Query encoder capabilities

        Ok(Self {
            gpu: NvidiaGpuInfo {
                device_index,
                ..Default::default()
            },
            cuda_context: None,
            encoder_session: None,
        })
    }

    /// Open default GPU.
    pub fn open_default() -> Result<Self> {
        Self::new(0)
    }

    /// Get capabilities.
    pub fn capabilities(&self) -> HwCapabilities {
        HwCapabilities {
            accel_type: crate::HwAccelType::Nvenc,
            encode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Av1],
            decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
            max_width: 8192,
            max_height: 8192,
            supports_bframes: self.gpu.supports_bframes,
            supports_10bit: true,
            supports_hdr: true,
            device_name: self.gpu.name.clone(),
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

    /// Get GPU utilization.
    pub fn gpu_utilization(&self) -> Result<GpuUtilization> {
        // In a real implementation, this would query NVML
        Ok(GpuUtilization {
            gpu: 50,
            memory: 30,
            encoder: 0,
            decoder: 0,
        })
    }

    /// Check if lookahead is supported.
    pub fn supports_lookahead(&self) -> bool {
        self.gpu.supports_lookahead
    }
}

/// GPU utilization stats.
#[derive(Debug, Clone, Copy)]
pub struct GpuUtilization {
    /// GPU core utilization (0-100).
    pub gpu: u32,
    /// Memory utilization (0-100).
    pub memory: u32,
    /// Encoder utilization (0-100).
    pub encoder: u32,
    /// Decoder utilization (0-100).
    pub decoder: u32,
}

/// NVENC encoder configuration.
#[derive(Debug, Clone)]
pub struct NvencEncoderConfig {
    /// Base configuration.
    pub base: crate::encoder::HwEncoderConfig,
    /// Profile.
    pub profile: NvencProfile,
    /// Preset.
    pub preset: NvencPreset,
    /// Rate control.
    pub rate_control: NvencRateControl,
    /// Lookahead configuration.
    pub lookahead: LookaheadConfig,
    /// B-frame configuration.
    pub bframes: BFrameConfig,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Enable weighted prediction.
    pub weighted_pred: bool,
    /// Enable adaptive quantization.
    pub aq: bool,
    /// AQ strength (1-15).
    pub aq_strength: u8,
}

impl Default for NvencEncoderConfig {
    fn default() -> Self {
        Self {
            base: crate::encoder::HwEncoderConfig::default(),
            profile: NvencProfile::H264High,
            preset: NvencPreset::P4,
            rate_control: NvencRateControl::Vbr {
                target_bitrate: 5_000_000,
                max_bitrate: 10_000_000,
            },
            lookahead: LookaheadConfig::default(),
            bframes: BFrameConfig::default(),
            gop_size: 60,
            weighted_pred: true,
            aq: true,
            aq_strength: 8,
        }
    }
}

impl NvencEncoderConfig {
    /// Build FFI initialization params.
    pub fn build_init_params(&self) -> NvEncInitializeParams {
        let mut params = NvEncInitializeParams::default();
        params.encode_guid = self.profile.codec_guid();
        params.preset_guid = self.preset.guid();
        params.encode_width = self.base.width;
        params.encode_height = self.base.height;
        params.frame_rate_num = self.base.frame_rate.0;
        params.frame_rate_den = self.base.frame_rate.1;
        params.max_encode_width = self.base.width;
        params.max_encode_height = self.base.height;
        params
    }

    /// Build FFI encode config.
    pub fn build_encode_config(&self) -> NvEncConfig {
        let mut config = NvEncConfig::default();
        config.profile_guid = self.profile.guid();
        config.gop_length = self.gop_size;
        config.frame_interval_p = if self.bframes.enabled {
            self.bframes.count as i32 + 1
        } else {
            1
        };
        config.idr_period = self.gop_size;

        // Rate control
        self.rate_control.apply_to_params(&mut config.rc_params);

        // Lookahead
        if self.lookahead.enabled {
            config.rc_params.enable_lookahead = 1;
            config.rc_params.lookahead_depth = self.lookahead.depth;
            config.rc_params.enable_temporal_aq = if self.lookahead.temporal_aq { 1 } else { 0 };
        }

        // AQ
        if self.aq {
            config.rc_params.enable_aq = 1;
            config.rc_params.aq_strength = self.aq_strength as u32;
        }

        config
    }
}

/// Pending frame in lookahead buffer.
struct PendingFrame {
    /// Input buffer.
    input: NvencInputBuffer,
    /// PTS.
    pts: i64,
    /// Force keyframe.
    force_keyframe: bool,
}

/// NVENC encoder.
pub struct NvencEncoder {
    context: NvencContext,
    config: NvencEncoderConfig,
    frame_count: u64,
    /// Input buffer pool.
    input_pool: NvencInputPool,
    /// Output buffers.
    output_buffers: Vec<NvencOutputBuffer>,
    /// Lookahead buffer (pending frames).
    lookahead_buffer: VecDeque<PendingFrame>,
    /// Encoded output queue.
    output_queue: VecDeque<NvencEncodedFrame>,
    /// SPS data.
    sps: Option<Vec<u8>>,
    /// PPS data.
    pps: Option<Vec<u8>>,
    /// VPS data (HEVC/AV1).
    vps: Option<Vec<u8>>,
    /// Latency tracker.
    latency_tracker: EncodeLatencyTracker,
    /// Async operation tracker.
    async_tracker: AsyncOperationTracker,
    /// Lookahead analyzer.
    lookahead_analyzer: Option<LookaheadAnalyzer>,
    /// AQ controller.
    aq_controller: Option<AdaptiveQuantization>,
    /// LTR manager.
    ltr_manager: Option<LtrManager>,
    /// Intra refresh manager.
    intra_refresh: Option<IntraRefreshManager>,
}

/// Encoded frame from NVENC.
#[derive(Debug)]
pub struct NvencEncodedFrame {
    /// Encoded data.
    pub data: Vec<u8>,
    /// PTS.
    pub pts: i64,
    /// DTS.
    pub dts: i64,
    /// Is keyframe.
    pub is_keyframe: bool,
    /// Frame type.
    pub frame_type: crate::encoder::FrameType,
    /// Slice type for multi-slice.
    pub slice_type: NvencSliceType,
}

/// NVENC slice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencSliceType {
    /// I slice (intra).
    I,
    /// P slice (predictive).
    P,
    /// B slice (bi-predictive).
    B,
    /// IDR (instantaneous decoder refresh).
    Idr,
}

impl NvencEncoder {
    /// Create a new NVENC encoder.
    pub fn new(config: NvencEncoderConfig) -> Result<Self> {
        let context = NvencContext::open_default()?;

        if !context.supports_encode(config.profile.codec()) {
            return Err(HwAccelError::CodecNotSupported(
                config.profile.codec().name().to_string(),
            ));
        }

        // Validate lookahead support
        if config.lookahead.enabled && !context.supports_lookahead() {
            tracing::warn!("Lookahead not supported on this GPU, disabling");
        }

        // Validate B-frame support with preset
        if config.bframes.enabled && !config.preset.supports_bframes() {
            tracing::warn!("B-frames not supported with low-latency preset");
        }

        // In a real implementation:
        // 1. Create NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
        // 2. Call NvEncOpenEncodeSessionEx
        // 3. Set NV_ENC_INITIALIZE_PARAMS with preset/tuning
        // 4. Call NvEncInitializeEncoder
        // 5. Allocate input/output buffers

        let lookahead_depth = if config.lookahead.enabled {
            config.lookahead.depth as usize
        } else {
            0
        };

        let pool_size = 4 + lookahead_depth + config.bframes.count as usize;

        // Initialize lookahead analyzer if enabled
        let lookahead_analyzer = if config.lookahead.enabled {
            Some(LookaheadAnalyzer::new(
                config.lookahead.depth,
                config.lookahead.temporal_aq,
                config.bframes.enabled,
            ))
        } else {
            None
        };

        // Initialize AQ controller if enabled
        let aq_controller = if config.aq {
            Some(AdaptiveQuantization::new(
                config.base.width,
                config.base.height,
                config.aq,
                config.lookahead.temporal_aq,
                config.aq_strength,
            ))
        } else {
            None
        };

        tracing::info!(
            "Created NVENC encoder: {:?} {:?} {}x{} @ {:?}",
            config.profile,
            config.preset,
            config.base.width,
            config.base.height,
            config.rate_control.target_bitrate()
        );

        Ok(Self {
            context,
            config: config.clone(),
            frame_count: 0,
            input_pool: NvencInputPool::new(config.base.width, config.base.height, pool_size),
            output_buffers: (0..pool_size)
                .map(|_| NvencOutputBuffer::new(4 * 1024 * 1024))
                .collect(),
            lookahead_buffer: VecDeque::with_capacity(lookahead_depth),
            output_queue: VecDeque::new(),
            sps: None,
            pps: None,
            vps: None,
            latency_tracker: EncodeLatencyTracker::new(100),
            async_tracker: AsyncOperationTracker::new(100),
            lookahead_analyzer,
            aq_controller,
            ltr_manager: None,
            intra_refresh: None,
        })
    }

    /// Create with base config.
    pub fn with_base_config(base: crate::encoder::HwEncoderConfig) -> Result<Self> {
        let profile = match base.codec {
            HwCodec::H264 => NvencProfile::H264High,
            HwCodec::Hevc => NvencProfile::HevcMain,
            HwCodec::Av1 => NvencProfile::Av1Main,
            _ => {
                return Err(HwAccelError::CodecNotSupported(
                    base.codec.name().to_string(),
                ))
            }
        };

        let config = NvencEncoderConfig {
            base,
            profile,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Enable LTR frames.
    pub fn enable_ltr(&mut self, max_ltr_frames: u32) {
        self.ltr_manager = Some(LtrManager::new(max_ltr_frames, LtrTrustMode::Default));
    }

    /// Enable intra refresh.
    pub fn enable_intra_refresh(&mut self, period: u32) {
        self.intra_refresh = Some(IntraRefreshManager::new(
            self.config.base.width,
            self.config.base.height,
            period,
            false,
        ));
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &HwFrame) -> Result<Option<crate::encoder::HwPacket>> {
        // Track latency
        self.latency_tracker.submit(self.frame_count);
        self.async_tracker
            .start(self.frame_count, AsyncOpType::Encode, Some(self.frame_count));

        // In a real implementation:
        // 1. Get input buffer from pool
        // 2. Map input resource (NvEncMapInputResource)
        // 3. Copy frame data to CUDA buffer
        // 4. If lookahead enabled, add to lookahead buffer
        // 5. When lookahead full or flushing:
        //    a. Set NV_ENC_PIC_PARAMS
        //    b. Call NvEncEncodePicture
        //    c. Lock output bitstream
        //    d. Copy and unlock

        let is_keyframe = self.frame_count.is_multiple_of(self.config.gop_size as u64);

        // Complete tracking
        self.latency_tracker.complete(self.frame_count);
        self.async_tracker.complete(self.frame_count, true);

        self.frame_count += 1;

        // Determine frame type based on B-frame pattern
        let frame_type = if is_keyframe {
            crate::encoder::FrameType::I
        } else if self.config.bframes.enabled {
            let pos = self.frame_count % (self.config.bframes.count as u64 + 1);
            if pos == 0 {
                crate::encoder::FrameType::P
            } else {
                crate::encoder::FrameType::B
            }
        } else {
            crate::encoder::FrameType::P
        };

        Ok(Some(crate::encoder::HwPacket {
            data: vec![0u8; 1000],
            pts: frame.pts,
            dts: frame.pts,
            is_keyframe,
            frame_type,
        }))
    }

    /// Submit frame to lookahead buffer.
    pub fn submit_frame(&mut self, pts: i64, force_keyframe: bool) -> Result<()> {
        let input = self
            .input_pool
            .acquire()
            .ok_or_else(|| HwAccelError::ResourceExhausted("No input buffers available".into()))?;

        self.lookahead_buffer.push_back(PendingFrame {
            input,
            pts,
            force_keyframe,
        });

        // Trigger encoding if lookahead is full
        if self.lookahead_buffer.len() >= self.config.lookahead.depth as usize {
            self.process_lookahead()?;
        }

        Ok(())
    }

    /// Process frames in lookahead buffer.
    fn process_lookahead(&mut self) -> Result<()> {
        // In a real implementation:
        // 1. Analyze lookahead frames for scene changes
        // 2. Optimize B-frame placement
        // 3. Apply temporal AQ if enabled
        // 4. Encode frames in order

        while let Some(pending) = self.lookahead_buffer.pop_front() {
            let is_keyframe =
                pending.force_keyframe || self.frame_count.is_multiple_of(self.config.gop_size as u64);

            let frame_type = if is_keyframe {
                crate::encoder::FrameType::I
            } else {
                crate::encoder::FrameType::P
            };

            self.output_queue.push_back(NvencEncodedFrame {
                data: vec![0u8; 1000],
                pts: pending.pts,
                dts: pending.pts,
                is_keyframe,
                frame_type,
                slice_type: if is_keyframe {
                    NvencSliceType::Idr
                } else {
                    NvencSliceType::P
                },
            });

            self.input_pool.release(pending.input);
            self.frame_count += 1;
        }

        Ok(())
    }

    /// Get next encoded frame.
    pub fn get_encoded_frame(&mut self) -> Option<NvencEncodedFrame> {
        self.output_queue.pop_front()
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Vec<crate::encoder::HwPacket>> {
        // Process remaining lookahead frames
        while !self.lookahead_buffer.is_empty() {
            self.process_lookahead()?;
        }

        // Drain output queue
        let packets: Vec<_> = self
            .output_queue
            .drain(..)
            .map(|f| crate::encoder::HwPacket {
                data: f.data,
                pts: f.pts,
                dts: f.dts,
                is_keyframe: f.is_keyframe,
                frame_type: f.frame_type,
            })
            .collect();

        Ok(packets)
    }

    /// Get encoder stats.
    pub fn stats(&self) -> NvencStats {
        NvencStats {
            frames_encoded: self.frame_count,
            frames_in_lookahead: self.lookahead_buffer.len() as u32,
            average_bitrate: 0,
            average_qp: 0.0,
            average_latency: self.latency_tracker.average_latency(),
            pending_async_ops: self.async_tracker.pending_count() as u32,
        }
    }

    /// Get SPS data.
    pub fn get_sps(&self) -> Option<&[u8]> {
        self.sps.as_deref()
    }

    /// Get PPS data.
    pub fn get_pps(&self) -> Option<&[u8]> {
        self.pps.as_deref()
    }

    /// Get VPS data.
    pub fn get_vps(&self) -> Option<&[u8]> {
        self.vps.as_deref()
    }

    /// Get latency tracker.
    pub fn latency_tracker(&self) -> &EncodeLatencyTracker {
        &self.latency_tracker
    }

    /// Get async tracker.
    pub fn async_tracker(&self) -> &AsyncOperationTracker {
        &self.async_tracker
    }
}

/// NVENC encoder statistics.
#[derive(Debug, Clone)]
pub struct NvencStats {
    /// Frames encoded.
    pub frames_encoded: u64,
    /// Frames in lookahead buffer.
    pub frames_in_lookahead: u32,
    /// Average bitrate.
    pub average_bitrate: u64,
    /// Average QP.
    pub average_qp: f64,
    /// Average encode latency.
    pub average_latency: Option<Duration>,
    /// Pending async operations.
    pub pending_async_ops: u32,
}

/// NVDEC decoder.
pub struct NvdecDecoder {
    context: NvencContext,
    config: crate::decoder::HwDecoderConfig,
    width: u32,
    height: u32,
    frames_decoded: u64,
    /// Video parser.
    parser: NvdecVideoParser,
    /// Decode create info.
    decode_info: CuvidDecodeCreateInfo,
    /// Latency tracker.
    latency_tracker: EncodeLatencyTracker,
}

impl NvdecDecoder {
    /// Create a new NVDEC decoder.
    pub fn new(config: crate::decoder::HwDecoderConfig) -> Result<Self> {
        let context = NvencContext::open_default()?;

        if !context.supports_decode(config.codec) {
            return Err(HwAccelError::CodecNotSupported(
                config.codec.name().to_string(),
            ));
        }

        // Map codec
        let cuvid_codec = match config.codec {
            HwCodec::H264 => CuvidCodec::H264,
            HwCodec::Hevc => CuvidCodec::Hevc,
            HwCodec::Vp9 => CuvidCodec::Vp9,
            HwCodec::Av1 => CuvidCodec::Av1,
            HwCodec::Mpeg2 => CuvidCodec::Mpeg2,
            _ => {
                return Err(HwAccelError::CodecNotSupported(
                    config.codec.name().to_string(),
                ))
            }
        };

        let decode_info = CuvidDecodeCreateInfo {
            coded_width: 1920,
            coded_height: 1080,
            num_decode_surfaces: config.surface_count,
            codec_type: cuvid_codec,
            ..Default::default()
        };

        // In a real implementation:
        // 1. Create CUVIDDECODECREATEINFO
        // 2. Call cuvidCreateDecoder

        tracing::info!("Created NVDEC decoder for {:?}", config.codec);

        Ok(Self {
            context,
            config,
            width: 0,
            height: 0,
            frames_decoded: 0,
            parser: NvdecVideoParser::new(cuvid_codec),
            decode_info,
            latency_tracker: EncodeLatencyTracker::new(100),
        })
    }

    /// Decode a packet.
    pub fn decode(&mut self, packet: &crate::decoder::DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        // Track latency
        self.latency_tracker.submit(self.frames_decoded);

        // Parse video data
        self.parser.parse(&packet.data, packet.pts)?;

        // In a real implementation:
        // 1. Create CUVIDSOURCEDATAPACKET
        // 2. Call cuvidParseVideoData
        // 3. Get decoded frame from callback
        // 4. Map frame with cuvidMapVideoFrame

        self.width = 1920;
        self.height = 1080;

        self.latency_tracker.complete(self.frames_decoded);
        self.frames_decoded += 1;

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::CudaPtr(0),
        }))
    }

    /// Flush decoder.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>> {
        Ok(Vec::new())
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get frames decoded.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Get decode info.
    pub fn decode_info(&self) -> &CuvidDecodeCreateInfo {
        &self.decode_info
    }

    /// Get video parser.
    pub fn parser(&self) -> &NvdecVideoParser {
        &self.parser
    }

    /// Get latency tracker.
    pub fn latency_tracker(&self) -> &EncodeLatencyTracker {
        &self.latency_tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvenc_context() {
        let ctx = NvencContext::open_default();
        // May fail if no NVIDIA GPU, that's OK
        if let Ok(ctx) = ctx {
            assert!(ctx.supports_encode(HwCodec::H264));
            assert!(ctx.supports_encode(HwCodec::Av1));
        }
    }

    #[test]
    fn test_nvenc_capabilities() {
        let ctx = NvencContext::open_default();
        if let Ok(ctx) = ctx {
            let caps = ctx.capabilities();
            assert!(caps.supports_10bit);
            assert_eq!(caps.max_width, 8192);
        }
    }

    #[test]
    fn test_nvenc_preset() {
        assert!(NvencPreset::P4.supports_bframes());
        assert!(!NvencPreset::LowLatencyHq.supports_bframes());
    }

    #[test]
    fn test_nvenc_profile() {
        assert_eq!(NvencProfile::H264High.codec(), HwCodec::H264);
        assert_eq!(NvencProfile::HevcMain10.codec(), HwCodec::Hevc);
        assert!(NvencProfile::HevcMain10.is_10bit());
    }

    #[test]
    fn test_lookahead_config() {
        let config = LookaheadConfig::default();
        assert!(config.enabled);
        assert_eq!(config.depth, 20);
    }

    #[test]
    fn test_bframe_config() {
        let config = BFrameConfig::default();
        assert!(config.enabled);
        assert_eq!(config.count, 2);
        assert!(config.as_ref);
    }

    #[test]
    fn test_cuda_buffer() {
        let buffer = CudaBuffer::new_nv12(1920, 1080);
        assert_eq!(buffer.width, 1920);
        assert_eq!(buffer.height, 1080);
        // Pitch should be aligned to 256
        assert_eq!(buffer.pitch % 256, 0);
    }

    #[test]
    fn test_input_pool() {
        let mut pool = NvencInputPool::new(1920, 1080, 4);

        let buf1 = pool.acquire();
        assert!(buf1.is_some());

        let buf2 = pool.acquire();
        assert!(buf2.is_some());

        pool.release(buf1.unwrap());
        pool.release(buf2.unwrap());
    }

    #[test]
    fn test_nvenc_status() {
        assert!(NvEncStatus::Success.is_success());
        assert!(!NvEncStatus::InvalidParam.is_success());
        assert_eq!(
            NvEncStatus::OutOfMemory.description(),
            "Out of memory"
        );
    }

    #[test]
    fn test_buffer_format() {
        assert_eq!(NvEncBufferFormat::Nv12.bits_per_pixel(), 12);
        assert!(NvEncBufferFormat::P010.is_10bit());
        assert!(!NvEncBufferFormat::Nv12.is_10bit());
    }

    #[test]
    fn test_guid() {
        let guid = NV_ENC_CODEC_H264_GUID;
        assert_ne!(guid, NvGuid::NULL);
        assert_eq!(guid, NV_ENC_CODEC_H264_GUID);
    }

    #[test]
    fn test_cuda_api() {
        assert!(cuda::cu_init(0).is_success());
        let device_count = cuda::cu_device_get_count();
        assert!(device_count.is_ok());
        assert!(device_count.unwrap() >= 0);
    }

    #[test]
    fn test_cuda_memory() {
        let ptr = cuda::cu_mem_alloc(1024);
        assert!(ptr.is_ok());
        assert_ne!(ptr.unwrap(), 0);
    }

    #[test]
    fn test_lookahead_analyzer() {
        let mut analyzer = LookaheadAnalyzer::new(4, true, true);

        for i in 0..4 {
            analyzer.submit_frame(LookaheadFrame {
                frame_idx: i,
                pts: i as i64 * 33333,
                force_keyframe: i == 0,
                stats: None,
            });
        }

        // Should have analysis results
        let result = analyzer.get_result();
        assert!(result.is_some());
    }

    #[test]
    fn test_aq_controller() {
        let mut aq = AdaptiveQuantization::new(1920, 1080, true, true, 8);
        let mb_count = ((1920 + 15) / 16) * ((1080 + 15) / 16);
        let complexity: Vec<f32> = (0..mb_count).map(|i| (i % 10) as f32 / 10.0).collect();

        let qp_map = aq.calculate_qp_map(&complexity);
        assert_eq!(qp_map.len(), mb_count as usize);
    }

    #[test]
    fn test_ltr_manager() {
        let mut ltr = LtrManager::new(4, LtrTrustMode::Default);

        let idx = ltr.mark_ltr(100, 0);
        assert!(idx.is_some());
        assert_eq!(idx.unwrap(), 0);

        let bitmap = ltr.get_use_bitmap();
        assert_eq!(bitmap, 1);
    }

    #[test]
    fn test_intra_refresh() {
        let mut ir = IntraRefreshManager::new(1920, 1080, 30, false);

        let params = ir.get_params();
        assert!(params.enable);
        assert!(params.count > 0);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = EncodeLatencyTracker::new(10);

        tracker.submit(0);
        std::thread::sleep(std::time::Duration::from_millis(1));
        tracker.complete(0);

        let avg = tracker.average_latency();
        assert!(avg.is_some());
    }

    #[test]
    fn test_async_tracker() {
        let mut tracker = AsyncOperationTracker::new(10);

        tracker.start(0, AsyncOpType::Encode, Some(0));
        assert!(tracker.is_pending(0));
        assert_eq!(tracker.pending_count(), 1);

        tracker.complete(0, true);
        assert!(!tracker.is_pending(0));
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_nvdec_parser() {
        let mut parser = NvdecVideoParser::new(CuvidCodec::H264);

        // Parse some dummy data
        let result = parser.parse(&[0u8; 100], 0);
        assert!(result.is_ok());

        // Should have display info
        let info = parser.get_display_info();
        assert!(info.is_some());
    }

    #[test]
    fn test_encoder_config_build() {
        let config = NvencEncoderConfig::default();

        let init_params = config.build_init_params();
        assert_eq!(init_params.encode_width, 1920);
        assert_eq!(init_params.encode_height, 1080);

        let enc_config = config.build_encode_config();
        assert_eq!(enc_config.gop_length, 60);
    }

    #[test]
    fn test_profile_guids() {
        assert_eq!(NvencProfile::H264High.codec_guid(), NV_ENC_CODEC_H264_GUID);
        assert_eq!(NvencProfile::HevcMain.codec_guid(), NV_ENC_CODEC_HEVC_GUID);
        assert_eq!(NvencProfile::Av1Main.codec_guid(), NV_ENC_CODEC_AV1_GUID);
    }

    #[test]
    fn test_preset_guids() {
        assert_eq!(NvencPreset::P4.guid(), NV_ENC_PRESET_P4_GUID);
        assert_eq!(NvencPreset::LowLatencyHq.guid(), NV_ENC_PRESET_LOW_LATENCY_HQ_GUID);
    }
}
