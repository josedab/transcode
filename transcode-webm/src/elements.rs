//! WebM element definitions and constants.
//!
//! This module defines all the EBML element IDs used in WebM files,
//! which is a subset of the Matroska specification.

use transcode_core::{AudioCodec, VideoCodec};

// ============================================================================
// EBML Header Elements (IDs include the VINT marker)
// ============================================================================

/// EBML element (root of EBML header).
pub const EBML: u32 = 0x1A45DFA3;
/// EBML Version.
pub const EBML_VERSION: u32 = 0x4286;
/// EBML Read Version.
pub const EBML_READ_VERSION: u32 = 0x42F7;
/// Maximum ID Length.
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
/// Maximum Size Length.
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
/// Document Type.
pub const DOC_TYPE: u32 = 0x4282;
/// Document Type Version.
pub const DOC_TYPE_VERSION: u32 = 0x4287;
/// Document Type Read Version.
pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;

// ============================================================================
// Segment Elements
// ============================================================================

/// Segment (main container).
pub const SEGMENT: u32 = 0x18538067;

/// SeekHead (index for faster seeking).
pub const SEEK_HEAD: u32 = 0x114D9B74;
/// Seek (single entry in SeekHead).
pub const SEEK: u32 = 0x4DBB;
/// SeekID (element ID being indexed).
pub const SEEK_ID: u32 = 0x53AB;
/// SeekPosition (byte position relative to segment).
pub const SEEK_POSITION: u32 = 0x53AC;

// ============================================================================
// Info Elements (Segment Information)
// ============================================================================

/// Info (segment information).
pub const INFO: u32 = 0x1549A966;
/// Segment UID (unique identifier).
pub const SEGMENT_UID: u32 = 0x73A4;
/// Segment Filename.
pub const SEGMENT_FILENAME: u32 = 0x7384;
/// Title.
pub const TITLE: u32 = 0x7BA9;
/// Muxing Application.
pub const MUXING_APP: u32 = 0x4D80;
/// Writing Application.
pub const WRITING_APP: u32 = 0x5741;
/// Timecode Scale (nanoseconds per tick).
pub const TIMECODE_SCALE: u32 = 0x2AD7B1;
/// Duration (in timecode units).
pub const DURATION: u32 = 0x4489;
/// Date UTC (nanoseconds since 2001-01-01).
pub const DATE_UTC: u32 = 0x4461;

// ============================================================================
// Track Elements
// ============================================================================

/// Tracks container.
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
/// Default Duration (nanoseconds).
pub const DEFAULT_DURATION: u32 = 0x23E383;
/// Track Name.
pub const NAME: u32 = 0x536E;
/// Language (ISO 639-2).
pub const LANGUAGE: u32 = 0x22B59C;
/// Language IETF (BCP 47).
pub const LANGUAGE_IETF: u32 = 0x22B59D;
/// Codec ID.
pub const CODEC_ID: u32 = 0x86;
/// Codec Private.
pub const CODEC_PRIVATE: u32 = 0x63A2;
/// Codec Name.
pub const CODEC_NAME: u32 = 0x258688;
/// Codec Delay (nanoseconds).
pub const CODEC_DELAY: u32 = 0x56AA;
/// Seek Pre-roll (nanoseconds).
pub const SEEK_PRE_ROLL: u32 = 0x56BB;

// Track types
/// Video track type.
pub const TRACK_TYPE_VIDEO: u8 = 1;
/// Audio track type.
pub const TRACK_TYPE_AUDIO: u8 = 2;
/// Subtitle track type.
pub const TRACK_TYPE_SUBTITLE: u8 = 17;

// ============================================================================
// Video Track Elements
// ============================================================================

/// Video settings container.
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
/// Colour element for color metadata.
pub const COLOUR: u32 = 0x55B0;
/// Matrix Coefficients.
pub const MATRIX_COEFFICIENTS: u32 = 0x55B1;
/// Bits Per Channel.
pub const BITS_PER_CHANNEL: u32 = 0x55B2;
/// Chroma Subsampling Horizontal.
pub const CHROMA_SUBSAMPLING_HORZ: u32 = 0x55B3;
/// Chroma Subsampling Vertical.
pub const CHROMA_SUBSAMPLING_VERT: u32 = 0x55B4;
/// Transfer Characteristics.
pub const TRANSFER_CHARACTERISTICS: u32 = 0x55BA;
/// Primaries.
pub const PRIMARIES: u32 = 0x55BB;
/// Range.
pub const RANGE: u32 = 0x55B9;

// ============================================================================
// Audio Track Elements
// ============================================================================

/// Audio settings container.
pub const AUDIO: u32 = 0xE1;
/// Sampling Frequency.
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
/// Output Sampling Frequency (for SBR).
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
/// Channels.
pub const CHANNELS: u32 = 0x9F;
/// Bit Depth.
pub const BIT_DEPTH: u32 = 0x6264;

// ============================================================================
// Cluster Elements
// ============================================================================

/// Cluster (container for frames).
pub const CLUSTER: u32 = 0x1F43B675;
/// Timestamp (cluster timestamp in timecode units).
pub const TIMESTAMP: u32 = 0xE7;
/// Position (byte position of cluster in segment).
pub const POSITION: u32 = 0xA7;
/// Prev Size (size of previous cluster).
pub const PREV_SIZE: u32 = 0xAB;
/// Simple Block (combined block with flags).
pub const SIMPLE_BLOCK: u32 = 0xA3;
/// Block Group.
pub const BLOCK_GROUP: u32 = 0xA0;
/// Block.
pub const BLOCK: u32 = 0xA1;
/// Block Duration.
pub const BLOCK_DURATION: u32 = 0x9B;
/// Reference Block (for P/B frames).
pub const REFERENCE_BLOCK: u32 = 0xFB;
/// Discard Padding.
pub const DISCARD_PADDING: u32 = 0x75A2;

// ============================================================================
// Cues Elements (Seeking Index)
// ============================================================================

/// Cues (seeking index).
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

// ============================================================================
// Tags Elements (not typically used in WebM, but included for completeness)
// ============================================================================

/// Tags container.
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
/// Simple Tag.
pub const SIMPLE_TAG: u32 = 0x67C8;
/// Tag Name.
pub const TAG_NAME: u32 = 0x45A3;
/// Tag Language.
pub const TAG_LANGUAGE: u32 = 0x447A;
/// Tag String.
pub const TAG_STRING: u32 = 0x4487;
/// Tag Binary.
pub const TAG_BINARY: u32 = 0x4485;

// ============================================================================
// Utility Elements
// ============================================================================

/// Void (padding element).
pub const VOID: u32 = 0xEC;
/// CRC-32 (checksum).
pub const CRC32: u32 = 0xBF;

// ============================================================================
// Codec ID Strings
// ============================================================================

/// WebM-compatible codec ID strings.
pub mod codec_ids {
    // Video codecs (WebM-compatible)
    /// VP8 video codec.
    pub const V_VP8: &str = "V_VP8";
    /// VP9 video codec.
    pub const V_VP9: &str = "V_VP9";
    /// AV1 video codec.
    pub const V_AV1: &str = "V_AV1";

    // Audio codecs (WebM-compatible)
    /// Vorbis audio codec.
    pub const A_VORBIS: &str = "A_VORBIS";
    /// Opus audio codec.
    pub const A_OPUS: &str = "A_OPUS";
}

/// Check if a codec ID is WebM-compatible.
pub fn is_webm_compatible_codec(codec_id: &str) -> bool {
    matches!(
        codec_id,
        codec_ids::V_VP8
            | codec_ids::V_VP9
            | codec_ids::V_AV1
            | codec_ids::A_VORBIS
            | codec_ids::A_OPUS
    )
}

/// Convert a VideoCodec enum to a WebM codec ID string.
pub fn video_codec_to_webm_id(codec: VideoCodec) -> Option<&'static str> {
    match codec {
        VideoCodec::Vp8 => Some(codec_ids::V_VP8),
        VideoCodec::Vp9 => Some(codec_ids::V_VP9),
        VideoCodec::Av1 => Some(codec_ids::V_AV1),
        _ => None,
    }
}

/// Convert an AudioCodec enum to a WebM codec ID string.
pub fn audio_codec_to_webm_id(codec: AudioCodec) -> Option<&'static str> {
    match codec {
        AudioCodec::Vorbis => Some(codec_ids::A_VORBIS),
        AudioCodec::Opus => Some(codec_ids::A_OPUS),
        _ => None,
    }
}

/// Convert a WebM video codec ID to a VideoCodec enum.
pub fn video_codec_from_webm_id(codec_id: &str) -> Option<VideoCodec> {
    match codec_id {
        codec_ids::V_VP8 => Some(VideoCodec::Vp8),
        codec_ids::V_VP9 => Some(VideoCodec::Vp9),
        codec_ids::V_AV1 => Some(VideoCodec::Av1),
        _ => None,
    }
}

/// Convert a WebM audio codec ID to an AudioCodec enum.
pub fn audio_codec_from_webm_id(codec_id: &str) -> Option<AudioCodec> {
    match codec_id {
        codec_ids::A_VORBIS => Some(AudioCodec::Vorbis),
        codec_ids::A_OPUS => Some(AudioCodec::Opus),
        _ => None,
    }
}

/// Get a human-readable name for an element ID.
pub fn element_name(id: u32) -> &'static str {
    match id {
        EBML => "EBML",
        EBML_VERSION => "EBMLVersion",
        EBML_READ_VERSION => "EBMLReadVersion",
        EBML_MAX_ID_LENGTH => "EBMLMaxIDLength",
        EBML_MAX_SIZE_LENGTH => "EBMLMaxSizeLength",
        DOC_TYPE => "DocType",
        DOC_TYPE_VERSION => "DocTypeVersion",
        DOC_TYPE_READ_VERSION => "DocTypeReadVersion",
        SEGMENT => "Segment",
        SEEK_HEAD => "SeekHead",
        SEEK => "Seek",
        SEEK_ID => "SeekID",
        SEEK_POSITION => "SeekPosition",
        INFO => "Info",
        SEGMENT_UID => "SegmentUID",
        TITLE => "Title",
        MUXING_APP => "MuxingApp",
        WRITING_APP => "WritingApp",
        TIMECODE_SCALE => "TimecodeScale",
        DURATION => "Duration",
        DATE_UTC => "DateUTC",
        TRACKS => "Tracks",
        TRACK_ENTRY => "TrackEntry",
        TRACK_NUMBER => "TrackNumber",
        TRACK_UID => "TrackUID",
        TRACK_TYPE => "TrackType",
        CODEC_ID => "CodecID",
        CODEC_PRIVATE => "CodecPrivate",
        DEFAULT_DURATION => "DefaultDuration",
        VIDEO => "Video",
        PIXEL_WIDTH => "PixelWidth",
        PIXEL_HEIGHT => "PixelHeight",
        AUDIO => "Audio",
        SAMPLING_FREQUENCY => "SamplingFrequency",
        CHANNELS => "Channels",
        CLUSTER => "Cluster",
        TIMESTAMP => "Timestamp",
        SIMPLE_BLOCK => "SimpleBlock",
        BLOCK_GROUP => "BlockGroup",
        BLOCK => "Block",
        CUES => "Cues",
        CUE_POINT => "CuePoint",
        CUE_TIME => "CueTime",
        CUE_TRACK_POSITIONS => "CueTrackPositions",
        VOID => "Void",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_webm_compatible() {
        assert!(is_webm_compatible_codec(codec_ids::V_VP8));
        assert!(is_webm_compatible_codec(codec_ids::V_VP9));
        assert!(is_webm_compatible_codec(codec_ids::V_AV1));
        assert!(is_webm_compatible_codec(codec_ids::A_VORBIS));
        assert!(is_webm_compatible_codec(codec_ids::A_OPUS));
        assert!(!is_webm_compatible_codec("V_MPEG4/ISO/AVC"));
        assert!(!is_webm_compatible_codec("A_AAC"));
    }

    #[test]
    fn test_video_codec_conversion() {
        assert_eq!(video_codec_to_webm_id(VideoCodec::Vp8), Some(codec_ids::V_VP8));
        assert_eq!(video_codec_to_webm_id(VideoCodec::Vp9), Some(codec_ids::V_VP9));
        assert_eq!(video_codec_to_webm_id(VideoCodec::Av1), Some(codec_ids::V_AV1));
        assert_eq!(video_codec_to_webm_id(VideoCodec::H264), None);

        assert_eq!(video_codec_from_webm_id(codec_ids::V_VP8), Some(VideoCodec::Vp8));
        assert_eq!(video_codec_from_webm_id(codec_ids::V_VP9), Some(VideoCodec::Vp9));
        assert_eq!(video_codec_from_webm_id(codec_ids::V_AV1), Some(VideoCodec::Av1));
        assert_eq!(video_codec_from_webm_id("V_MPEG4/ISO/AVC"), None);
    }

    #[test]
    fn test_audio_codec_conversion() {
        assert_eq!(audio_codec_to_webm_id(AudioCodec::Vorbis), Some(codec_ids::A_VORBIS));
        assert_eq!(audio_codec_to_webm_id(AudioCodec::Opus), Some(codec_ids::A_OPUS));
        assert_eq!(audio_codec_to_webm_id(AudioCodec::Aac), None);

        assert_eq!(audio_codec_from_webm_id(codec_ids::A_VORBIS), Some(AudioCodec::Vorbis));
        assert_eq!(audio_codec_from_webm_id(codec_ids::A_OPUS), Some(AudioCodec::Opus));
        assert_eq!(audio_codec_from_webm_id("A_AAC"), None);
    }

    #[test]
    fn test_element_names() {
        assert_eq!(element_name(EBML), "EBML");
        assert_eq!(element_name(SEGMENT), "Segment");
        assert_eq!(element_name(CLUSTER), "Cluster");
        assert_eq!(element_name(SIMPLE_BLOCK), "SimpleBlock");
        assert_eq!(element_name(0xFFFFFFFF), "Unknown");
    }

    #[test]
    fn test_track_types() {
        assert_eq!(TRACK_TYPE_VIDEO, 1);
        assert_eq!(TRACK_TYPE_AUDIO, 2);
        assert_eq!(TRACK_TYPE_SUBTITLE, 17);
    }
}
