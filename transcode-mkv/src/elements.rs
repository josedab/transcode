//! Matroska element definitions and codec ID mappings.
//!
//! This module defines all the Matroska element IDs and their types,
//! as well as codec ID mappings between Matroska and transcode_core.

use transcode_core::{AudioCodec, VideoCodec};

// =============================================================================
// EBML Header Elements
// =============================================================================

/// EBML Header element.
pub const EBML: u32 = 0x1A45DFA3;
/// EBML Version.
pub const EBML_VERSION: u32 = 0x4286;
/// EBML Read Version.
pub const EBML_READ_VERSION: u32 = 0x42F7;
/// EBML Max ID Length.
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
/// EBML Max Size Length.
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
/// EBML Doc Type.
pub const DOC_TYPE: u32 = 0x4282;
/// EBML Doc Type Version.
pub const DOC_TYPE_VERSION: u32 = 0x4287;
/// EBML Doc Type Read Version.
pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;

// =============================================================================
// Segment Elements
// =============================================================================

/// Segment (the root container for all Matroska data).
pub const SEGMENT: u32 = 0x18538067;

// =============================================================================
// Meta Seek Information
// =============================================================================

/// SeekHead (index of top-level elements).
pub const SEEK_HEAD: u32 = 0x114D9B74;
/// Seek entry.
pub const SEEK: u32 = 0x4DBB;
/// Seek ID.
pub const SEEK_ID: u32 = 0x53AB;
/// Seek Position.
pub const SEEK_POSITION: u32 = 0x53AC;

// =============================================================================
// Segment Information
// =============================================================================

/// Segment Info.
pub const INFO: u32 = 0x1549A966;
/// Segment UID.
pub const SEGMENT_UID: u32 = 0x73A4;
/// Segment Filename.
pub const SEGMENT_FILENAME: u32 = 0x7384;
/// Previous UID.
pub const PREV_UID: u32 = 0x3CB923;
/// Previous Filename.
pub const PREV_FILENAME: u32 = 0x3C83AB;
/// Next UID.
pub const NEXT_UID: u32 = 0x3EB923;
/// Next Filename.
pub const NEXT_FILENAME: u32 = 0x3E83BB;
/// Segment Family.
pub const SEGMENT_FAMILY: u32 = 0x4444;
/// Timecode Scale (nanoseconds per timecode unit, default 1000000 = 1ms).
pub const TIMECODE_SCALE: u32 = 0x2AD7B1;
/// Duration (in timecode units).
pub const DURATION: u32 = 0x4489;
/// Date UTC (nanoseconds since 2001-01-01).
pub const DATE_UTC: u32 = 0x4461;
/// Title.
pub const TITLE: u32 = 0x7BA9;
/// Muxing App.
pub const MUXING_APP: u32 = 0x4D80;
/// Writing App.
pub const WRITING_APP: u32 = 0x5741;

// =============================================================================
// Cluster Elements
// =============================================================================

/// Cluster (contains blocks of media data).
pub const CLUSTER: u32 = 0x1F43B675;
/// Cluster Timestamp.
pub const TIMESTAMP: u32 = 0xE7;
/// Silent Tracks.
pub const SILENT_TRACKS: u32 = 0x5854;
/// Silent Track Number.
pub const SILENT_TRACK_NUMBER: u32 = 0x58D7;
/// Position (cluster position in segment).
pub const POSITION: u32 = 0xA7;
/// Previous Size (size of previous cluster).
pub const PREV_SIZE: u32 = 0xAB;
/// SimpleBlock (basic block without lacing info).
pub const SIMPLE_BLOCK: u32 = 0xA3;
/// BlockGroup (block with additional info).
pub const BLOCK_GROUP: u32 = 0xA0;
/// Block.
pub const BLOCK: u32 = 0xA1;
/// Block Duration.
pub const BLOCK_DURATION: u32 = 0x9B;
/// Reference Priority.
pub const REFERENCE_PRIORITY: u32 = 0xFA;
/// Reference Block (timestamp offset to reference frame).
pub const REFERENCE_BLOCK: u32 = 0xFB;
/// Codec State.
pub const CODEC_STATE: u32 = 0xA4;
/// Discard Padding.
pub const DISCARD_PADDING: u32 = 0x75A2;

// =============================================================================
// Track Elements
// =============================================================================

/// Tracks.
pub const TRACKS: u32 = 0x1654AE6B;
/// Track Entry.
pub const TRACK_ENTRY: u32 = 0xAE;
/// Track Number.
pub const TRACK_NUMBER: u32 = 0xD7;
/// Track UID.
pub const TRACK_UID: u32 = 0x73C5;
/// Track Type.
pub const TRACK_TYPE: u32 = 0x83;
/// Flag Enabled.
pub const FLAG_ENABLED: u32 = 0xB9;
/// Flag Default.
pub const FLAG_DEFAULT: u32 = 0x88;
/// Flag Forced.
pub const FLAG_FORCED: u32 = 0x55AA;
/// Flag Lacing.
pub const FLAG_LACING: u32 = 0x9C;
/// Min Cache.
pub const MIN_CACHE: u32 = 0x6DE7;
/// Max Cache.
pub const MAX_CACHE: u32 = 0x6DF8;
/// Default Duration.
pub const DEFAULT_DURATION: u32 = 0x23E383;
/// Default Decoded Field Duration.
pub const DEFAULT_DECODED_FIELD_DURATION: u32 = 0x234E7A;
/// Track Timecode Scale (deprecated).
pub const TRACK_TIMECODE_SCALE: u32 = 0x23314F;
/// Max Block Addition ID.
pub const MAX_BLOCK_ADDITION_ID: u32 = 0x55EE;
/// Name.
pub const NAME: u32 = 0x536E;
/// Language.
pub const LANGUAGE: u32 = 0x22B59C;
/// Language IETF.
pub const LANGUAGE_IETF: u32 = 0x22B59D;
/// Codec ID.
pub const CODEC_ID: u32 = 0x86;
/// Codec Private.
pub const CODEC_PRIVATE: u32 = 0x63A2;
/// Codec Name.
pub const CODEC_NAME: u32 = 0x258688;
/// Attachment Link.
pub const ATTACHMENT_LINK: u32 = 0x7446;
/// Codec Decode All.
pub const CODEC_DECODE_ALL: u32 = 0xAA;
/// Track Overlay.
pub const TRACK_OVERLAY: u32 = 0x6FAB;
/// Codec Delay.
pub const CODEC_DELAY: u32 = 0x56AA;
/// Seek Pre-Roll.
pub const SEEK_PRE_ROLL: u32 = 0x56BB;

// =============================================================================
// Video Elements
// =============================================================================

/// Video settings.
pub const VIDEO: u32 = 0xE0;
/// Flag Interlaced.
pub const FLAG_INTERLACED: u32 = 0x9A;
/// Field Order.
pub const FIELD_ORDER: u32 = 0x9D;
/// Stereo Mode.
pub const STEREO_MODE: u32 = 0x53B8;
/// Alpha Mode.
pub const ALPHA_MODE: u32 = 0x53C0;
/// Pixel Width.
pub const PIXEL_WIDTH: u32 = 0xB0;
/// Pixel Height.
pub const PIXEL_HEIGHT: u32 = 0xBA;
/// Pixel Crop Bottom.
pub const PIXEL_CROP_BOTTOM: u32 = 0x54AA;
/// Pixel Crop Top.
pub const PIXEL_CROP_TOP: u32 = 0x54BB;
/// Pixel Crop Left.
pub const PIXEL_CROP_LEFT: u32 = 0x54CC;
/// Pixel Crop Right.
pub const PIXEL_CROP_RIGHT: u32 = 0x54DD;
/// Display Width.
pub const DISPLAY_WIDTH: u32 = 0x54B0;
/// Display Height.
pub const DISPLAY_HEIGHT: u32 = 0x54BA;
/// Display Unit.
pub const DISPLAY_UNIT: u32 = 0x54B2;
/// Aspect Ratio Type.
pub const ASPECT_RATIO_TYPE: u32 = 0x54B3;
/// Colour Space.
pub const COLOUR_SPACE: u32 = 0x2EB524;
/// Colour.
pub const COLOUR: u32 = 0x55B0;
/// Matrix Coefficients.
pub const MATRIX_COEFFICIENTS: u32 = 0x55B1;
/// Bits Per Channel.
pub const BITS_PER_CHANNEL: u32 = 0x55B2;
/// Chroma Subsampling Horz.
pub const CHROMA_SUBSAMPLING_HORZ: u32 = 0x55B3;
/// Chroma Subsampling Vert.
pub const CHROMA_SUBSAMPLING_VERT: u32 = 0x55B4;
/// Cb Subsampling Horz.
pub const CB_SUBSAMPLING_HORZ: u32 = 0x55B5;
/// Cb Subsampling Vert.
pub const CB_SUBSAMPLING_VERT: u32 = 0x55B6;
/// Chroma Siting Horz.
pub const CHROMA_SITING_HORZ: u32 = 0x55B7;
/// Chroma Siting Vert.
pub const CHROMA_SITING_VERT: u32 = 0x55B8;
/// Range.
pub const RANGE: u32 = 0x55B9;
/// Transfer Characteristics.
pub const TRANSFER_CHARACTERISTICS: u32 = 0x55BA;
/// Primaries.
pub const PRIMARIES: u32 = 0x55BB;
/// Max CLL.
pub const MAX_CLL: u32 = 0x55BC;
/// Max FALL.
pub const MAX_FALL: u32 = 0x55BD;
/// Mastering Metadata.
pub const MASTERING_METADATA: u32 = 0x55D0;
/// Primary R Chromaticity X.
pub const PRIMARY_R_CHROMATICITY_X: u32 = 0x55D1;
/// Primary R Chromaticity Y.
pub const PRIMARY_R_CHROMATICITY_Y: u32 = 0x55D2;
/// Primary G Chromaticity X.
pub const PRIMARY_G_CHROMATICITY_X: u32 = 0x55D3;
/// Primary G Chromaticity Y.
pub const PRIMARY_G_CHROMATICITY_Y: u32 = 0x55D4;
/// Primary B Chromaticity X.
pub const PRIMARY_B_CHROMATICITY_X: u32 = 0x55D5;
/// Primary B Chromaticity Y.
pub const PRIMARY_B_CHROMATICITY_Y: u32 = 0x55D6;
/// White Point Chromaticity X.
pub const WHITE_POINT_CHROMATICITY_X: u32 = 0x55D7;
/// White Point Chromaticity Y.
pub const WHITE_POINT_CHROMATICITY_Y: u32 = 0x55D8;
/// Luminance Max.
pub const LUMINANCE_MAX: u32 = 0x55D9;
/// Luminance Min.
pub const LUMINANCE_MIN: u32 = 0x55DA;

// =============================================================================
// Audio Elements
// =============================================================================

/// Audio settings.
pub const AUDIO: u32 = 0xE1;
/// Sampling Frequency.
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
/// Output Sampling Frequency.
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
/// Channels.
pub const CHANNELS: u32 = 0x9F;
/// Bit Depth.
pub const BIT_DEPTH: u32 = 0x6264;

// =============================================================================
// Content Encoding (Compression/Encryption)
// =============================================================================

/// Content Encodings.
pub const CONTENT_ENCODINGS: u32 = 0x6D80;
/// Content Encoding.
pub const CONTENT_ENCODING: u32 = 0x6240;
/// Content Encoding Order.
pub const CONTENT_ENCODING_ORDER: u32 = 0x5031;
/// Content Encoding Scope.
pub const CONTENT_ENCODING_SCOPE: u32 = 0x5032;
/// Content Encoding Type.
pub const CONTENT_ENCODING_TYPE: u32 = 0x5033;
/// Content Compression.
pub const CONTENT_COMPRESSION: u32 = 0x5034;
/// Content Comp Algo.
pub const CONTENT_COMP_ALGO: u32 = 0x4254;
/// Content Comp Settings.
pub const CONTENT_COMP_SETTINGS: u32 = 0x4255;
/// Content Encryption.
pub const CONTENT_ENCRYPTION: u32 = 0x5035;
/// Content Enc Algo.
pub const CONTENT_ENC_ALGO: u32 = 0x47E1;
/// Content Enc Key ID.
pub const CONTENT_ENC_KEY_ID: u32 = 0x47E2;

// =============================================================================
// Cueing Data
// =============================================================================

/// Cues.
pub const CUES: u32 = 0x1C53BB6B;
/// Cue Point.
pub const CUE_POINT: u32 = 0xBB;
/// Cue Time.
pub const CUE_TIME: u32 = 0xB3;
/// Cue Track Positions.
pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
/// Cue Track.
pub const CUE_TRACK: u32 = 0xF7;
/// Cue Cluster Position.
pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
/// Cue Relative Position.
pub const CUE_RELATIVE_POSITION: u32 = 0xF0;
/// Cue Duration.
pub const CUE_DURATION: u32 = 0xB2;
/// Cue Block Number.
pub const CUE_BLOCK_NUMBER: u32 = 0x5378;

// =============================================================================
// Attachment Elements
// =============================================================================

/// Attachments.
pub const ATTACHMENTS: u32 = 0x1941A469;
/// Attached File.
pub const ATTACHED_FILE: u32 = 0x61A7;
/// File Description.
pub const FILE_DESCRIPTION: u32 = 0x467E;
/// File Name.
pub const FILE_NAME: u32 = 0x466E;
/// File Media Type.
pub const FILE_MEDIA_TYPE: u32 = 0x4660;
/// File Data.
pub const FILE_DATA: u32 = 0x465C;
/// File UID.
pub const FILE_UID: u32 = 0x46AE;

// =============================================================================
// Chapters
// =============================================================================

/// Chapters.
pub const CHAPTERS: u32 = 0x1043A770;
/// Edition Entry.
pub const EDITION_ENTRY: u32 = 0x45B9;
/// Edition UID.
pub const EDITION_UID: u32 = 0x45BC;
/// Edition Flag Hidden.
pub const EDITION_FLAG_HIDDEN: u32 = 0x45BD;
/// Edition Flag Default.
pub const EDITION_FLAG_DEFAULT: u32 = 0x45DB;
/// Edition Flag Ordered.
pub const EDITION_FLAG_ORDERED: u32 = 0x45DD;
/// Chapter Atom.
pub const CHAPTER_ATOM: u32 = 0xB6;
/// Chapter UID.
pub const CHAPTER_UID: u32 = 0x73C4;
/// Chapter String UID.
pub const CHAPTER_STRING_UID: u32 = 0x5654;
/// Chapter Time Start.
pub const CHAPTER_TIME_START: u32 = 0x91;
/// Chapter Time End.
pub const CHAPTER_TIME_END: u32 = 0x92;
/// Chapter Flag Hidden.
pub const CHAPTER_FLAG_HIDDEN: u32 = 0x98;
/// Chapter Flag Enabled.
pub const CHAPTER_FLAG_ENABLED: u32 = 0x4598;
/// Chapter Segment UID.
pub const CHAPTER_SEGMENT_UID: u32 = 0x6E67;
/// Chapter Display.
pub const CHAPTER_DISPLAY: u32 = 0x80;
/// Chap String.
pub const CHAP_STRING: u32 = 0x85;
/// Chap Language.
pub const CHAP_LANGUAGE: u32 = 0x437C;
/// Chap Country.
pub const CHAP_COUNTRY: u32 = 0x437E;

// =============================================================================
// Tagging
// =============================================================================

/// Tags.
pub const TAGS: u32 = 0x1254C367;
/// Tag.
pub const TAG: u32 = 0x7373;
/// Targets.
pub const TARGETS: u32 = 0x63C0;
/// Target Type Value.
pub const TARGET_TYPE_VALUE: u32 = 0x68CA;
/// Target Type.
pub const TARGET_TYPE: u32 = 0x63CA;
/// Tag Track UID.
pub const TAG_TRACK_UID: u32 = 0x63C5;
/// Tag Edition UID.
pub const TAG_EDITION_UID: u32 = 0x63C9;
/// Tag Chapter UID.
pub const TAG_CHAPTER_UID: u32 = 0x63C4;
/// Tag Attachment UID.
pub const TAG_ATTACHMENT_UID: u32 = 0x63C6;
/// Simple Tag.
pub const SIMPLE_TAG: u32 = 0x67C8;
/// Tag Name.
pub const TAG_NAME: u32 = 0x45A3;
/// Tag Language.
pub const TAG_LANGUAGE: u32 = 0x447A;
/// Tag Default.
pub const TAG_DEFAULT: u32 = 0x4484;
/// Tag String.
pub const TAG_STRING: u32 = 0x4487;
/// Tag Binary.
pub const TAG_BINARY: u32 = 0x4485;

// =============================================================================
// Void and CRC
// =============================================================================

/// Void (padding).
pub const VOID: u32 = 0xEC;
/// CRC-32.
pub const CRC32: u32 = 0xBF;

// =============================================================================
// Track Types
// =============================================================================

/// Track type: Video.
pub const TRACK_TYPE_VIDEO: u8 = 1;
/// Track type: Audio.
pub const TRACK_TYPE_AUDIO: u8 = 2;
/// Track type: Complex (combined audio/video).
pub const TRACK_TYPE_COMPLEX: u8 = 3;
/// Track type: Logo.
pub const TRACK_TYPE_LOGO: u8 = 16;
/// Track type: Subtitle.
pub const TRACK_TYPE_SUBTITLE: u8 = 17;
/// Track type: Buttons.
pub const TRACK_TYPE_BUTTONS: u8 = 18;
/// Track type: Control.
pub const TRACK_TYPE_CONTROL: u8 = 32;
/// Track type: Metadata.
pub const TRACK_TYPE_METADATA: u8 = 33;

// =============================================================================
// Codec IDs
// =============================================================================

/// Matroska codec ID definitions.
pub mod codec_ids {
    // Video codecs
    /// VP8 video codec.
    pub const V_VP8: &str = "V_VP8";
    /// VP9 video codec.
    pub const V_VP9: &str = "V_VP9";
    /// AV1 video codec.
    pub const V_AV1: &str = "V_AV1";
    /// H.264/AVC video codec.
    pub const V_MPEG4_ISO_AVC: &str = "V_MPEG4/ISO/AVC";
    /// H.265/HEVC video codec.
    pub const V_MPEGH_ISO_HEVC: &str = "V_MPEGH/ISO/HEVC";
    /// MPEG-4 Simple Profile.
    pub const V_MPEG4_ISO_SP: &str = "V_MPEG4/ISO/SP";
    /// MPEG-4 Advanced Simple Profile.
    pub const V_MPEG4_ISO_ASP: &str = "V_MPEG4/ISO/ASP";
    /// MPEG-4 Advanced Profile.
    pub const V_MPEG4_ISO_AP: &str = "V_MPEG4/ISO/AP";
    /// MPEG-4 Microsoft V3.
    pub const V_MPEG4_MS_V3: &str = "V_MPEG4/MS/V3";
    /// MPEG-1 video.
    pub const V_MPEG1: &str = "V_MPEG1";
    /// MPEG-2 video.
    pub const V_MPEG2: &str = "V_MPEG2";
    /// Motion JPEG.
    pub const V_MJPEG: &str = "V_MJPEG";
    /// Uncompressed video.
    pub const V_UNCOMPRESSED: &str = "V_UNCOMPRESSED";
    /// Theora video.
    pub const V_THEORA: &str = "V_THEORA";

    // Audio codecs
    /// Opus audio codec.
    pub const A_OPUS: &str = "A_OPUS";
    /// Vorbis audio codec.
    pub const A_VORBIS: &str = "A_VORBIS";
    /// FLAC audio codec.
    pub const A_FLAC: &str = "A_FLAC";
    /// AAC audio codec (generic).
    pub const A_AAC: &str = "A_AAC";
    /// AAC MPEG-2 Main Profile.
    pub const A_AAC_MPEG2_MAIN: &str = "A_AAC/MPEG2/MAIN";
    /// AAC MPEG-2 Low Complexity.
    pub const A_AAC_MPEG2_LC: &str = "A_AAC/MPEG2/LC";
    /// AAC MPEG-2 Low Complexity with SBR.
    pub const A_AAC_MPEG2_LC_SBR: &str = "A_AAC/MPEG2/LC/SBR";
    /// AAC MPEG-2 Scalable Sample Rate.
    pub const A_AAC_MPEG2_SSR: &str = "A_AAC/MPEG2/SSR";
    /// AAC MPEG-4 Main Profile.
    pub const A_AAC_MPEG4_MAIN: &str = "A_AAC/MPEG4/MAIN";
    /// AAC MPEG-4 Low Complexity.
    pub const A_AAC_MPEG4_LC: &str = "A_AAC/MPEG4/LC";
    /// AAC MPEG-4 Low Complexity with SBR.
    pub const A_AAC_MPEG4_LC_SBR: &str = "A_AAC/MPEG4/LC/SBR";
    /// AAC MPEG-4 Scalable Sample Rate.
    pub const A_AAC_MPEG4_SSR: &str = "A_AAC/MPEG4/SSR";
    /// AAC MPEG-4 Long Term Prediction.
    pub const A_AAC_MPEG4_LTP: &str = "A_AAC/MPEG4/LTP";
    /// MPEG Layer 3 (MP3).
    pub const A_MPEG_L3: &str = "A_MPEG/L3";
    /// MPEG Layer 2.
    pub const A_MPEG_L2: &str = "A_MPEG/L2";
    /// MPEG Layer 1.
    pub const A_MPEG_L1: &str = "A_MPEG/L1";
    /// AC-3 (Dolby Digital).
    pub const A_AC3: &str = "A_AC3";
    /// AC-3 BSID 9.
    pub const A_AC3_BSID9: &str = "A_AC3/BSID9";
    /// AC-3 BSID 10.
    pub const A_AC3_BSID10: &str = "A_AC3/BSID10";
    /// E-AC-3 (Enhanced AC-3).
    pub const A_EAC3: &str = "A_EAC3";
    /// DTS audio.
    pub const A_DTS: &str = "A_DTS";
    /// DTS Express.
    pub const A_DTS_EXPRESS: &str = "A_DTS/EXPRESS";
    /// DTS Lossless.
    pub const A_DTS_LOSSLESS: &str = "A_DTS/LOSSLESS";
    /// PCM big-endian integer.
    pub const A_PCM_INT_BIG: &str = "A_PCM/INT/BIG";
    /// PCM little-endian integer.
    pub const A_PCM_INT_LIT: &str = "A_PCM/INT/LIT";
    /// PCM IEEE floating point.
    pub const A_PCM_FLOAT_IEEE: &str = "A_PCM/FLOAT/IEEE";
    /// Dolby TrueHD.
    pub const A_TRUEHD: &str = "A_TRUEHD";

    // Subtitle codecs
    /// UTF-8 text subtitles.
    pub const S_TEXT_UTF8: &str = "S_TEXT/UTF8";
    /// SSA subtitles.
    pub const S_TEXT_SSA: &str = "S_TEXT/SSA";
    /// ASS subtitles.
    pub const S_TEXT_ASS: &str = "S_TEXT/ASS";
    /// WebVTT subtitles.
    pub const S_TEXT_WEBVTT: &str = "S_TEXT/WEBVTT";
    /// VobSub subtitles.
    pub const S_VOBSUB: &str = "S_VOBSUB";
    /// HDMV PGS subtitles.
    pub const S_HDMV_PGS: &str = "S_HDMV/PGS";
    /// DVB subtitles.
    pub const S_DVBSUB: &str = "S_DVBSUB";
}

/// Convert a Matroska video codec ID to a transcode_core VideoCodec.
pub fn video_codec_from_mkv_id(codec_id: &str) -> Option<VideoCodec> {
    match codec_id {
        codec_ids::V_VP8 => Some(VideoCodec::Vp8),
        codec_ids::V_VP9 => Some(VideoCodec::Vp9),
        codec_ids::V_AV1 => Some(VideoCodec::Av1),
        codec_ids::V_MPEG4_ISO_AVC => Some(VideoCodec::H264),
        codec_ids::V_MPEGH_ISO_HEVC => Some(VideoCodec::H265),
        codec_ids::V_MJPEG => Some(VideoCodec::Mjpeg),
        codec_ids::V_UNCOMPRESSED => Some(VideoCodec::Raw),
        _ => None,
    }
}

/// Convert a transcode_core VideoCodec to a Matroska codec ID.
pub fn video_codec_to_mkv_id(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::Vp8 => codec_ids::V_VP8,
        VideoCodec::Vp9 => codec_ids::V_VP9,
        VideoCodec::Av1 => codec_ids::V_AV1,
        VideoCodec::H264 => codec_ids::V_MPEG4_ISO_AVC,
        VideoCodec::H265 => codec_ids::V_MPEGH_ISO_HEVC,
        VideoCodec::Mjpeg => codec_ids::V_MJPEG,
        VideoCodec::Raw => codec_ids::V_UNCOMPRESSED,
    }
}

/// Convert a Matroska audio codec ID to a transcode_core AudioCodec.
pub fn audio_codec_from_mkv_id(codec_id: &str) -> Option<AudioCodec> {
    match codec_id {
        codec_ids::A_OPUS => Some(AudioCodec::Opus),
        codec_ids::A_VORBIS => Some(AudioCodec::Vorbis),
        codec_ids::A_FLAC => Some(AudioCodec::Flac),
        codec_ids::A_AAC
        | codec_ids::A_AAC_MPEG2_MAIN
        | codec_ids::A_AAC_MPEG2_LC
        | codec_ids::A_AAC_MPEG2_LC_SBR
        | codec_ids::A_AAC_MPEG2_SSR
        | codec_ids::A_AAC_MPEG4_MAIN
        | codec_ids::A_AAC_MPEG4_LC
        | codec_ids::A_AAC_MPEG4_LC_SBR
        | codec_ids::A_AAC_MPEG4_SSR
        | codec_ids::A_AAC_MPEG4_LTP => Some(AudioCodec::Aac),
        codec_ids::A_MPEG_L3 => Some(AudioCodec::Mp3),
        codec_ids::A_AC3 | codec_ids::A_AC3_BSID9 | codec_ids::A_AC3_BSID10 => Some(AudioCodec::Ac3),
        codec_ids::A_EAC3 => Some(AudioCodec::Eac3),
        codec_ids::A_PCM_INT_BIG | codec_ids::A_PCM_INT_LIT | codec_ids::A_PCM_FLOAT_IEEE => {
            Some(AudioCodec::Pcm)
        }
        _ => None,
    }
}

/// Convert a transcode_core AudioCodec to a Matroska codec ID.
pub fn audio_codec_to_mkv_id(codec: AudioCodec) -> &'static str {
    match codec {
        AudioCodec::Opus => codec_ids::A_OPUS,
        AudioCodec::Vorbis => codec_ids::A_VORBIS,
        AudioCodec::Flac => codec_ids::A_FLAC,
        AudioCodec::Aac => codec_ids::A_AAC_MPEG4_LC,
        AudioCodec::Mp3 => codec_ids::A_MPEG_L3,
        AudioCodec::Ac3 => codec_ids::A_AC3,
        AudioCodec::Eac3 => codec_ids::A_EAC3,
        AudioCodec::Pcm => codec_ids::A_PCM_INT_LIT,
    }
}

/// Check if a codec ID is WebM-compatible.
pub fn is_webm_compatible_codec(codec_id: &str) -> bool {
    matches!(
        codec_id,
        codec_ids::V_VP8
            | codec_ids::V_VP9
            | codec_ids::V_AV1
            | codec_ids::A_OPUS
            | codec_ids::A_VORBIS
    )
}

/// Element type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// Master element (contains other elements).
    Master,
    /// Unsigned integer.
    UnsignedInt,
    /// Signed integer.
    SignedInt,
    /// Floating point.
    Float,
    /// UTF-8 string.
    String,
    /// Binary data.
    Binary,
    /// Date (nanoseconds since 2001-01-01).
    Date,
}

/// Get the type of a known element.
pub fn element_type(id: u32) -> Option<ElementType> {
    match id {
        // Master elements
        EBML | SEGMENT | SEEK_HEAD | SEEK | INFO | TRACKS | TRACK_ENTRY | VIDEO | AUDIO
        | CLUSTER | BLOCK_GROUP | CUES | CUE_POINT | CUE_TRACK_POSITIONS | CHAPTERS
        | EDITION_ENTRY | CHAPTER_ATOM | CHAPTER_DISPLAY | TAGS | TAG | TARGETS | SIMPLE_TAG
        | ATTACHMENTS | ATTACHED_FILE | CONTENT_ENCODINGS | CONTENT_ENCODING
        | CONTENT_COMPRESSION | CONTENT_ENCRYPTION | COLOUR | MASTERING_METADATA
        | SILENT_TRACKS => Some(ElementType::Master),

        // Unsigned integers
        EBML_VERSION | EBML_READ_VERSION | EBML_MAX_ID_LENGTH | EBML_MAX_SIZE_LENGTH
        | DOC_TYPE_VERSION | DOC_TYPE_READ_VERSION | TIMECODE_SCALE | TIMESTAMP | TRACK_NUMBER
        | TRACK_UID | TRACK_TYPE | FLAG_ENABLED | FLAG_DEFAULT | FLAG_FORCED | FLAG_LACING
        | MIN_CACHE | MAX_CACHE | DEFAULT_DURATION | DEFAULT_DECODED_FIELD_DURATION
        | MAX_BLOCK_ADDITION_ID | CODEC_DELAY | SEEK_PRE_ROLL | ATTACHMENT_LINK
        | CODEC_DECODE_ALL | TRACK_OVERLAY | PIXEL_WIDTH | PIXEL_HEIGHT | DISPLAY_WIDTH
        | DISPLAY_HEIGHT | DISPLAY_UNIT | ASPECT_RATIO_TYPE | FLAG_INTERLACED | FIELD_ORDER
        | STEREO_MODE | ALPHA_MODE | PIXEL_CROP_BOTTOM | PIXEL_CROP_TOP | PIXEL_CROP_LEFT
        | PIXEL_CROP_RIGHT | CHANNELS | BIT_DEPTH | CUE_TIME | CUE_TRACK | CUE_CLUSTER_POSITION
        | CUE_RELATIVE_POSITION | CUE_DURATION | CUE_BLOCK_NUMBER | EDITION_UID
        | EDITION_FLAG_HIDDEN | EDITION_FLAG_DEFAULT | EDITION_FLAG_ORDERED | CHAPTER_UID
        | CHAPTER_FLAG_HIDDEN | CHAPTER_FLAG_ENABLED | TARGET_TYPE_VALUE | TAG_TRACK_UID
        | TAG_EDITION_UID | TAG_CHAPTER_UID | TAG_ATTACHMENT_UID | TAG_DEFAULT | FILE_UID
        | SEEK_POSITION | POSITION | PREV_SIZE | BLOCK_DURATION | REFERENCE_PRIORITY
        | CONTENT_ENCODING_ORDER | CONTENT_ENCODING_SCOPE | CONTENT_ENCODING_TYPE
        | CONTENT_COMP_ALGO | CONTENT_ENC_ALGO | MATRIX_COEFFICIENTS | BITS_PER_CHANNEL
        | CHROMA_SUBSAMPLING_HORZ | CHROMA_SUBSAMPLING_VERT | CB_SUBSAMPLING_HORZ
        | CB_SUBSAMPLING_VERT | CHROMA_SITING_HORZ | CHROMA_SITING_VERT | RANGE
        | TRANSFER_CHARACTERISTICS | PRIMARIES | MAX_CLL | MAX_FALL | SILENT_TRACK_NUMBER => {
            Some(ElementType::UnsignedInt)
        }

        // Signed integers
        REFERENCE_BLOCK | DISCARD_PADDING => Some(ElementType::SignedInt),

        // Floats
        DURATION | TRACK_TIMECODE_SCALE | SAMPLING_FREQUENCY | OUTPUT_SAMPLING_FREQUENCY
        | PRIMARY_R_CHROMATICITY_X | PRIMARY_R_CHROMATICITY_Y | PRIMARY_G_CHROMATICITY_X
        | PRIMARY_G_CHROMATICITY_Y | PRIMARY_B_CHROMATICITY_X | PRIMARY_B_CHROMATICITY_Y
        | WHITE_POINT_CHROMATICITY_X | WHITE_POINT_CHROMATICITY_Y | LUMINANCE_MAX
        | LUMINANCE_MIN => Some(ElementType::Float),

        // Strings
        DOC_TYPE | CODEC_ID | CODEC_NAME | NAME | LANGUAGE | LANGUAGE_IETF | TITLE | MUXING_APP
        | WRITING_APP | SEGMENT_FILENAME | PREV_FILENAME | NEXT_FILENAME | FILE_NAME
        | FILE_MEDIA_TYPE | FILE_DESCRIPTION | CHAP_STRING | CHAP_LANGUAGE | CHAP_COUNTRY
        | TARGET_TYPE | TAG_NAME | TAG_LANGUAGE | TAG_STRING | CHAPTER_STRING_UID => {
            Some(ElementType::String)
        }

        // Binary
        SEGMENT_UID | PREV_UID | NEXT_UID | SEGMENT_FAMILY | SEEK_ID | CODEC_PRIVATE
        | COLOUR_SPACE | SIMPLE_BLOCK | BLOCK | CODEC_STATE | CONTENT_COMP_SETTINGS
        | CONTENT_ENC_KEY_ID | FILE_DATA | CHAPTER_SEGMENT_UID | TAG_BINARY | CRC32 | VOID => {
            Some(ElementType::Binary)
        }

        // Date
        DATE_UTC => Some(ElementType::Date),

        // Time elements (unsigned int representing nanoseconds)
        CHAPTER_TIME_START | CHAPTER_TIME_END => Some(ElementType::UnsignedInt),

        _ => None,
    }
}

/// Check if an element is a master element (container).
pub fn is_master_element(id: u32) -> bool {
    element_type(id) == Some(ElementType::Master)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_roundtrip() {
        for codec in [
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::H264,
            VideoCodec::H265,
        ] {
            let mkv_id = video_codec_to_mkv_id(codec);
            let converted = video_codec_from_mkv_id(mkv_id);
            assert_eq!(converted, Some(codec));
        }
    }

    #[test]
    fn test_audio_codec_roundtrip() {
        for codec in [
            AudioCodec::Opus,
            AudioCodec::Vorbis,
            AudioCodec::Flac,
            AudioCodec::Aac,
            AudioCodec::Mp3,
            AudioCodec::Ac3,
            AudioCodec::Eac3,
        ] {
            let mkv_id = audio_codec_to_mkv_id(codec);
            let converted = audio_codec_from_mkv_id(mkv_id);
            assert_eq!(converted, Some(codec));
        }
    }

    #[test]
    fn test_webm_compatible_codecs() {
        assert!(is_webm_compatible_codec(codec_ids::V_VP8));
        assert!(is_webm_compatible_codec(codec_ids::V_VP9));
        assert!(is_webm_compatible_codec(codec_ids::V_AV1));
        assert!(is_webm_compatible_codec(codec_ids::A_OPUS));
        assert!(is_webm_compatible_codec(codec_ids::A_VORBIS));

        assert!(!is_webm_compatible_codec(codec_ids::V_MPEG4_ISO_AVC));
        assert!(!is_webm_compatible_codec(codec_ids::A_AAC));
    }

    #[test]
    fn test_element_types() {
        assert_eq!(element_type(EBML), Some(ElementType::Master));
        assert_eq!(element_type(SEGMENT), Some(ElementType::Master));
        assert_eq!(element_type(TRACK_NUMBER), Some(ElementType::UnsignedInt));
        assert_eq!(element_type(DURATION), Some(ElementType::Float));
        assert_eq!(element_type(CODEC_ID), Some(ElementType::String));
        assert_eq!(element_type(CODEC_PRIVATE), Some(ElementType::Binary));
        assert_eq!(element_type(DATE_UTC), Some(ElementType::Date));
    }

    #[test]
    fn test_is_master_element() {
        assert!(is_master_element(EBML));
        assert!(is_master_element(SEGMENT));
        assert!(is_master_element(TRACKS));
        assert!(!is_master_element(TRACK_NUMBER));
        assert!(!is_master_element(CODEC_ID));
    }

    #[test]
    fn test_element_ids() {
        // Verify some well-known element IDs
        assert_eq!(EBML, 0x1A45DFA3);
        assert_eq!(SEGMENT, 0x18538067);
        assert_eq!(CLUSTER, 0x1F43B675);
        assert_eq!(TRACKS, 0x1654AE6B);
        assert_eq!(CUES, 0x1C53BB6B);
    }
}
