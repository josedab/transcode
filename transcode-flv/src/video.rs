//! FLV video codec support.
//!
//! FLV video tags contain a one-byte header followed by video data:
//! - Frame type (4 bits)
//! - Codec ID (4 bits)
//!
//! For AVC (H.264), additional bytes specify:
//! - AVC packet type (1 byte)
//! - Composition time offset (3 bytes, signed)
//!
//! Enhanced FLV (for HEVC/H.265) uses:
//! - Extended header format with packet type and FourCC

use crate::error::{FlvError, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

/// Codec ID: Sorenson H.263.
pub const CODEC_ID_SORENSON: u8 = 2;
/// Codec ID: Screen video.
pub const CODEC_ID_SCREEN_VIDEO: u8 = 3;
/// Codec ID: VP6.
pub const CODEC_ID_VP6: u8 = 4;
/// Codec ID: VP6 with alpha.
pub const CODEC_ID_VP6_ALPHA: u8 = 5;
/// Codec ID: Screen video version 2.
pub const CODEC_ID_SCREEN_VIDEO_2: u8 = 6;
/// Codec ID: AVC (H.264).
pub const CODEC_ID_AVC: u8 = 7;
/// Codec ID: HEVC (H.265) - Enhanced FLV.
pub const CODEC_ID_HEVC: u8 = 12;

/// Frame type: Keyframe.
pub const FRAME_TYPE_KEYFRAME: u8 = 1;
/// Frame type: Inter frame.
pub const FRAME_TYPE_INTER: u8 = 2;
/// Frame type: Disposable inter frame (H.263 only).
pub const FRAME_TYPE_DISPOSABLE: u8 = 3;
/// Frame type: Generated keyframe (server use only).
pub const FRAME_TYPE_GENERATED_KEYFRAME: u8 = 4;
/// Frame type: Video info/command frame.
pub const FRAME_TYPE_INFO: u8 = 5;

/// AVC packet type: Sequence header (SPS/PPS).
pub const AVC_PACKET_SEQUENCE_HEADER: u8 = 0;
/// AVC packet type: NALU.
pub const AVC_PACKET_NALU: u8 = 1;
/// AVC packet type: End of sequence.
pub const AVC_PACKET_END_OF_SEQUENCE: u8 = 2;

/// Enhanced FLV packet type.
pub const ENHANCED_PACKET_SEQUENCE_START: u8 = 0;
/// Enhanced FLV coded frames.
pub const ENHANCED_PACKET_CODED_FRAMES: u8 = 1;
/// Enhanced FLV sequence end.
pub const ENHANCED_PACKET_SEQUENCE_END: u8 = 2;
/// Enhanced FLV coded frames (no composition time).
pub const ENHANCED_PACKET_CODED_FRAMES_X: u8 = 3;
/// Enhanced FLV metadata.
pub const ENHANCED_PACKET_METADATA: u8 = 4;
/// Enhanced FLV MPEG-2 TS sequence start.
pub const ENHANCED_PACKET_MPEG2TS_SEQUENCE_START: u8 = 5;

/// FourCC for HEVC.
pub const FOURCC_HEVC: [u8; 4] = *b"hvc1";
/// FourCC for AVC.
pub const FOURCC_AVC: [u8; 4] = *b"avc1";
/// FourCC for AV1.
pub const FOURCC_AV1: [u8; 4] = *b"av01";
/// FourCC for VP9.
pub const FOURCC_VP9: [u8; 4] = *b"vp09";

/// Video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VideoCodec {
    /// Sorenson H.263.
    Sorenson = CODEC_ID_SORENSON,
    /// Screen video.
    ScreenVideo = CODEC_ID_SCREEN_VIDEO,
    /// VP6.
    Vp6 = CODEC_ID_VP6,
    /// VP6 with alpha.
    Vp6Alpha = CODEC_ID_VP6_ALPHA,
    /// Screen video version 2.
    ScreenVideo2 = CODEC_ID_SCREEN_VIDEO_2,
    /// AVC (H.264).
    Avc = CODEC_ID_AVC,
    /// HEVC (H.265).
    Hevc = CODEC_ID_HEVC,
}

impl VideoCodec {
    /// Create from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            CODEC_ID_SORENSON => Some(Self::Sorenson),
            CODEC_ID_SCREEN_VIDEO => Some(Self::ScreenVideo),
            CODEC_ID_VP6 => Some(Self::Vp6),
            CODEC_ID_VP6_ALPHA => Some(Self::Vp6Alpha),
            CODEC_ID_SCREEN_VIDEO_2 => Some(Self::ScreenVideo2),
            CODEC_ID_AVC => Some(Self::Avc),
            CODEC_ID_HEVC => Some(Self::Hevc),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the codec name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sorenson => "Sorenson H.263",
            Self::ScreenVideo => "Screen Video",
            Self::Vp6 => "VP6",
            Self::Vp6Alpha => "VP6 Alpha",
            Self::ScreenVideo2 => "Screen Video 2",
            Self::Avc => "H.264/AVC",
            Self::Hevc => "H.265/HEVC",
        }
    }

    /// Check if this codec uses AVC-style packaging.
    pub fn has_avc_packet(self) -> bool {
        matches!(self, Self::Avc | Self::Hevc)
    }

    /// Check if this codec is enhanced FLV only.
    pub fn is_enhanced(self) -> bool {
        self == Self::Hevc
    }
}

/// Video frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// Keyframe (for AVC, IDR frame).
    Keyframe = FRAME_TYPE_KEYFRAME,
    /// Inter frame (for AVC, non-IDR).
    Inter = FRAME_TYPE_INTER,
    /// Disposable inter frame (H.263 only).
    Disposable = FRAME_TYPE_DISPOSABLE,
    /// Generated keyframe (server use only).
    GeneratedKeyframe = FRAME_TYPE_GENERATED_KEYFRAME,
    /// Video info/command frame.
    Info = FRAME_TYPE_INFO,
}

impl FrameType {
    /// Create from raw 4-bit value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value & 0x0F {
            FRAME_TYPE_KEYFRAME => Some(Self::Keyframe),
            FRAME_TYPE_INTER => Some(Self::Inter),
            FRAME_TYPE_DISPOSABLE => Some(Self::Disposable),
            FRAME_TYPE_GENERATED_KEYFRAME => Some(Self::GeneratedKeyframe),
            FRAME_TYPE_INFO => Some(Self::Info),
            _ => None,
        }
    }

    /// Convert to raw 4-bit value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this is a keyframe.
    pub fn is_keyframe(self) -> bool {
        matches!(self, Self::Keyframe | Self::GeneratedKeyframe)
    }
}

/// AVC/HEVC packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AvcPacketType {
    /// Sequence header (AVCDecoderConfigurationRecord or HEVCDecoderConfigurationRecord).
    SequenceHeader = AVC_PACKET_SEQUENCE_HEADER,
    /// One or more NALUs.
    Nalu = AVC_PACKET_NALU,
    /// End of sequence.
    EndOfSequence = AVC_PACKET_END_OF_SEQUENCE,
}

impl AvcPacketType {
    /// Create from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            AVC_PACKET_SEQUENCE_HEADER => Some(Self::SequenceHeader),
            AVC_PACKET_NALU => Some(Self::Nalu),
            AVC_PACKET_END_OF_SEQUENCE => Some(Self::EndOfSequence),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this is a sequence header.
    pub fn is_sequence_header(self) -> bool {
        self == Self::SequenceHeader
    }
}

/// Enhanced FLV packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnhancedPacketType {
    /// Sequence start (decoder configuration).
    SequenceStart = ENHANCED_PACKET_SEQUENCE_START,
    /// Coded frames with composition time.
    CodedFrames = ENHANCED_PACKET_CODED_FRAMES,
    /// Sequence end.
    SequenceEnd = ENHANCED_PACKET_SEQUENCE_END,
    /// Coded frames without composition time.
    CodedFramesX = ENHANCED_PACKET_CODED_FRAMES_X,
    /// Metadata.
    Metadata = ENHANCED_PACKET_METADATA,
    /// MPEG-2 TS sequence start.
    Mpeg2TsSequenceStart = ENHANCED_PACKET_MPEG2TS_SEQUENCE_START,
}

impl EnhancedPacketType {
    /// Create from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            ENHANCED_PACKET_SEQUENCE_START => Some(Self::SequenceStart),
            ENHANCED_PACKET_CODED_FRAMES => Some(Self::CodedFrames),
            ENHANCED_PACKET_SEQUENCE_END => Some(Self::SequenceEnd),
            ENHANCED_PACKET_CODED_FRAMES_X => Some(Self::CodedFramesX),
            ENHANCED_PACKET_METADATA => Some(Self::Metadata),
            ENHANCED_PACKET_MPEG2TS_SEQUENCE_START => Some(Self::Mpeg2TsSequenceStart),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this is a sequence header type.
    pub fn is_sequence_header(self) -> bool {
        matches!(self, Self::SequenceStart | Self::Mpeg2TsSequenceStart)
    }
}

/// FLV video tag header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTagHeader {
    /// Frame type.
    pub frame_type: FrameType,
    /// Video codec.
    pub codec_id: VideoCodec,
    /// AVC packet type (for AVC/HEVC).
    pub avc_packet_type: Option<AvcPacketType>,
    /// Composition time offset in milliseconds (for AVC/HEVC).
    pub composition_time: i32,
    /// Whether this is enhanced FLV format.
    pub is_enhanced: bool,
    /// Enhanced packet type (for enhanced FLV).
    pub enhanced_packet_type: Option<EnhancedPacketType>,
    /// FourCC (for enhanced FLV).
    pub fourcc: Option<[u8; 4]>,
}

impl VideoTagHeader {
    /// Create a new video tag header.
    pub fn new(frame_type: FrameType, codec_id: VideoCodec) -> Self {
        Self {
            frame_type,
            codec_id,
            avc_packet_type: None,
            composition_time: 0,
            is_enhanced: false,
            enhanced_packet_type: None,
            fourcc: None,
        }
    }

    /// Create an AVC sequence header.
    pub fn avc_sequence_header() -> Self {
        Self {
            frame_type: FrameType::Keyframe,
            codec_id: VideoCodec::Avc,
            avc_packet_type: Some(AvcPacketType::SequenceHeader),
            composition_time: 0,
            is_enhanced: false,
            enhanced_packet_type: None,
            fourcc: None,
        }
    }

    /// Create an AVC NALU header.
    pub fn avc_nalu(frame_type: FrameType, composition_time: i32) -> Self {
        Self {
            frame_type,
            codec_id: VideoCodec::Avc,
            avc_packet_type: Some(AvcPacketType::Nalu),
            composition_time,
            is_enhanced: false,
            enhanced_packet_type: None,
            fourcc: None,
        }
    }

    /// Create an AVC keyframe NALU header.
    pub fn avc_keyframe(composition_time: i32) -> Self {
        Self::avc_nalu(FrameType::Keyframe, composition_time)
    }

    /// Create an AVC inter frame NALU header.
    pub fn avc_inter(composition_time: i32) -> Self {
        Self::avc_nalu(FrameType::Inter, composition_time)
    }

    /// Create an HEVC sequence header (enhanced FLV).
    pub fn hevc_sequence_header() -> Self {
        Self {
            frame_type: FrameType::Keyframe,
            codec_id: VideoCodec::Hevc,
            avc_packet_type: None,
            composition_time: 0,
            is_enhanced: true,
            enhanced_packet_type: Some(EnhancedPacketType::SequenceStart),
            fourcc: Some(FOURCC_HEVC),
        }
    }

    /// Create an HEVC coded frames header (enhanced FLV).
    pub fn hevc_coded_frames(frame_type: FrameType, composition_time: i32) -> Self {
        Self {
            frame_type,
            codec_id: VideoCodec::Hevc,
            avc_packet_type: None,
            composition_time,
            is_enhanced: true,
            enhanced_packet_type: Some(EnhancedPacketType::CodedFrames),
            fourcc: Some(FOURCC_HEVC),
        }
    }

    /// Get the header size in bytes.
    pub fn size(&self) -> usize {
        if self.is_enhanced {
            // Enhanced: 1 (frame/type) + 4 (fourcc) + optional composition time
            if self.enhanced_packet_type == Some(EnhancedPacketType::CodedFrames) {
                1 + 4 + 3 // With composition time
            } else {
                1 + 4 // Without composition time
            }
        } else if self.codec_id.has_avc_packet() {
            // AVC: 1 (frame/codec) + 1 (packet type) + 3 (composition time)
            5
        } else {
            // Other: 1 (frame/codec)
            1
        }
    }

    /// Parse from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        let first_byte = reader.read_u8()?;

        let frame_type_value = (first_byte >> 4) & 0x0F;
        let codec_value = first_byte & 0x0F;

        // Check for enhanced FLV (frame type bit 7 set with specific pattern)
        let is_enhanced = frame_type_value & 0x08 != 0;

        if is_enhanced {
            // Enhanced FLV format
            let packet_type_value = codec_value;
            let enhanced_packet_type = EnhancedPacketType::from_u8(packet_type_value);

            // Read FourCC
            let mut fourcc = [0u8; 4];
            reader.read_exact(&mut fourcc)?;

            // Determine codec from FourCC
            let codec_id = match &fourcc {
                b"hvc1" | b"hev1" => VideoCodec::Hevc,
                b"avc1" | b"avc3" => VideoCodec::Avc,
                _ => {
                    return Err(FlvError::InvalidVideoCodec(codec_value));
                }
            };

            // Read composition time if present
            let composition_time =
                if enhanced_packet_type == Some(EnhancedPacketType::CodedFrames) {
                    let ct_bytes = [reader.read_u8()?, reader.read_u8()?, reader.read_u8()?];
                    let ct = ((ct_bytes[0] as i32) << 16)
                        | ((ct_bytes[1] as i32) << 8)
                        | (ct_bytes[2] as i32);
                    // Sign extend if negative
                    if ct & 0x800000 != 0 {
                        ct | !0xFFFFFF
                    } else {
                        ct
                    }
                } else {
                    0
                };

            let frame_type = if frame_type_value & 0x01 != 0 {
                FrameType::Keyframe
            } else {
                FrameType::Inter
            };

            Ok(Self {
                frame_type,
                codec_id,
                avc_packet_type: None,
                composition_time,
                is_enhanced: true,
                enhanced_packet_type,
                fourcc: Some(fourcc),
            })
        } else {
            // Standard FLV format
            let frame_type = FrameType::from_u8(frame_type_value)
                .ok_or(FlvError::InvalidFrameType(frame_type_value))?;
            let codec_id =
                VideoCodec::from_u8(codec_value).ok_or(FlvError::InvalidVideoCodec(codec_value))?;

            let (avc_packet_type, composition_time) = if codec_id.has_avc_packet() {
                let packet_type_byte = reader.read_u8()?;
                let avc_packet_type = AvcPacketType::from_u8(packet_type_byte)
                    .ok_or(FlvError::InvalidAvcPacketType(packet_type_byte))?;

                // Read composition time (24-bit signed)
                let ct_bytes = [reader.read_u8()?, reader.read_u8()?, reader.read_u8()?];
                let ct = ((ct_bytes[0] as i32) << 16)
                    | ((ct_bytes[1] as i32) << 8)
                    | (ct_bytes[2] as i32);
                // Sign extend if negative
                let composition_time = if ct & 0x800000 != 0 {
                    ct | !0xFFFFFF
                } else {
                    ct
                };

                (Some(avc_packet_type), composition_time)
            } else {
                (None, 0)
            };

            Ok(Self {
                frame_type,
                codec_id,
                avc_packet_type,
                composition_time,
                is_enhanced: false,
                enhanced_packet_type: None,
                fourcc: None,
            })
        }
    }

    /// Write to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        if self.is_enhanced {
            // Enhanced FLV format
            let packet_type = self
                .enhanced_packet_type
                .unwrap_or(EnhancedPacketType::CodedFrames)
                .as_u8();
            let frame_flag = if self.frame_type.is_keyframe() {
                0x01
            } else {
                0x00
            };
            let first_byte = 0x80 | (frame_flag << 4) | packet_type;
            writer.write_u8(first_byte)?;

            // Write FourCC
            let fourcc = self.fourcc.unwrap_or(FOURCC_HEVC);
            writer.write_all(&fourcc)?;

            // Write composition time if CodedFrames
            if self.enhanced_packet_type == Some(EnhancedPacketType::CodedFrames) {
                let ct = self.composition_time;
                writer.write_u8(((ct >> 16) & 0xFF) as u8)?;
                writer.write_u8(((ct >> 8) & 0xFF) as u8)?;
                writer.write_u8((ct & 0xFF) as u8)?;
                Ok(8)
            } else {
                Ok(5)
            }
        } else {
            // Standard FLV format
            let first_byte = (self.frame_type.as_u8() << 4) | self.codec_id.as_u8();
            writer.write_u8(first_byte)?;

            if self.codec_id.has_avc_packet() {
                let packet_type = self.avc_packet_type.unwrap_or(AvcPacketType::Nalu);
                writer.write_u8(packet_type.as_u8())?;

                // Write composition time (24-bit)
                let ct = self.composition_time;
                writer.write_u8(((ct >> 16) & 0xFF) as u8)?;
                writer.write_u8(((ct >> 8) & 0xFF) as u8)?;
                writer.write_u8((ct & 0xFF) as u8)?;

                Ok(5)
            } else {
                Ok(1)
            }
        }
    }

    /// Check if this is a sequence header.
    pub fn is_sequence_header(&self) -> bool {
        if self.is_enhanced {
            self.enhanced_packet_type
                .map(|t| t.is_sequence_header())
                .unwrap_or(false)
        } else {
            self.avc_packet_type
                .map(|t| t.is_sequence_header())
                .unwrap_or(false)
        }
    }

    /// Check if this is a keyframe.
    pub fn is_keyframe(&self) -> bool {
        self.frame_type.is_keyframe()
    }
}

/// AVC decoder configuration record (contains SPS/PPS).
#[derive(Debug, Clone)]
pub struct AvcConfig {
    /// Configuration version (always 1).
    pub configuration_version: u8,
    /// AVC profile (from SPS).
    pub avc_profile: u8,
    /// Profile compatibility (from SPS).
    pub profile_compatibility: u8,
    /// AVC level (from SPS).
    pub avc_level: u8,
    /// Length size minus one (typically 3 for 4-byte lengths).
    pub length_size_minus_one: u8,
    /// Sequence Parameter Sets.
    pub sps: Vec<Vec<u8>>,
    /// Picture Parameter Sets.
    pub pps: Vec<Vec<u8>>,
    /// Raw configuration bytes.
    pub raw: Vec<u8>,
}

impl AvcConfig {
    /// Create a new AVC config from SPS and PPS.
    pub fn new(sps: Vec<Vec<u8>>, pps: Vec<Vec<u8>>) -> Result<Self> {
        if sps.is_empty() {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "AVC".to_string(),
                message: "No SPS provided".to_string(),
            });
        }

        let first_sps = &sps[0];
        if first_sps.len() < 4 {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "AVC".to_string(),
                message: "SPS too short".to_string(),
            });
        }

        let avc_profile = first_sps[1];
        let profile_compatibility = first_sps[2];
        let avc_level = first_sps[3];

        let raw = Self::build_raw(&sps, &pps, avc_profile, profile_compatibility, avc_level);

        Ok(Self {
            configuration_version: 1,
            avc_profile,
            profile_compatibility,
            avc_level,
            length_size_minus_one: 3, // 4-byte lengths
            sps,
            pps,
            raw,
        })
    }

    /// Build raw AVCDecoderConfigurationRecord.
    fn build_raw(
        sps_list: &[Vec<u8>],
        pps_list: &[Vec<u8>],
        profile: u8,
        compat: u8,
        level: u8,
    ) -> Vec<u8> {
        let mut raw = vec![
            // configurationVersion
            1,
            // AVCProfileIndication
            profile,
            // profile_compatibility
            compat,
            // AVCLevelIndication
            level,
            // lengthSizeMinusOne (6 reserved bits = 0b111111 + 2 bits for length)
            0xFF, // 0b11111111 = reserved + length_size_minus_one = 3
            // numOfSequenceParameterSets (3 reserved bits + 5 bits for count)
            0xE0 | (sps_list.len() as u8 & 0x1F),
        ];

        for sps in sps_list {
            raw.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            raw.extend_from_slice(sps);
        }

        // numOfPictureParameterSets
        raw.push(pps_list.len() as u8);

        for pps in pps_list {
            raw.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            raw.extend_from_slice(pps);
        }

        raw
    }

    /// Parse from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 7 {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "AVC".to_string(),
                message: "AVCDecoderConfigurationRecord too short".to_string(),
            });
        }

        let configuration_version = data[0];
        let avc_profile = data[1];
        let profile_compatibility = data[2];
        let avc_level = data[3];
        let length_size_minus_one = data[4] & 0x03;

        let num_sps = (data[5] & 0x1F) as usize;
        let mut offset = 6;
        let mut sps = Vec::with_capacity(num_sps);

        for _ in 0..num_sps {
            if offset + 2 > data.len() {
                return Err(FlvError::InvalidSequenceHeader {
                    codec: "AVC".to_string(),
                    message: "SPS length truncated".to_string(),
                });
            }
            let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + sps_len > data.len() {
                return Err(FlvError::InvalidSequenceHeader {
                    codec: "AVC".to_string(),
                    message: "SPS data truncated".to_string(),
                });
            }
            sps.push(data[offset..offset + sps_len].to_vec());
            offset += sps_len;
        }

        if offset >= data.len() {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "AVC".to_string(),
                message: "No PPS count".to_string(),
            });
        }

        let num_pps = data[offset] as usize;
        offset += 1;
        let mut pps = Vec::with_capacity(num_pps);

        for _ in 0..num_pps {
            if offset + 2 > data.len() {
                return Err(FlvError::InvalidSequenceHeader {
                    codec: "AVC".to_string(),
                    message: "PPS length truncated".to_string(),
                });
            }
            let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + pps_len > data.len() {
                return Err(FlvError::InvalidSequenceHeader {
                    codec: "AVC".to_string(),
                    message: "PPS data truncated".to_string(),
                });
            }
            pps.push(data[offset..offset + pps_len].to_vec());
            offset += pps_len;
        }

        Ok(Self {
            configuration_version,
            avc_profile,
            profile_compatibility,
            avc_level,
            length_size_minus_one,
            sps,
            pps,
            raw: data.to_vec(),
        })
    }

    /// Get the NALU length size in bytes.
    pub fn nalu_length_size(&self) -> usize {
        (self.length_size_minus_one + 1) as usize
    }
}

/// HEVC decoder configuration record.
#[derive(Debug, Clone)]
pub struct HevcConfig {
    /// Configuration version.
    pub configuration_version: u8,
    /// General profile space.
    pub general_profile_space: u8,
    /// General tier flag.
    pub general_tier_flag: bool,
    /// General profile IDC.
    pub general_profile_idc: u8,
    /// General level IDC.
    pub general_level_idc: u8,
    /// VPS NAL units.
    pub vps: Vec<Vec<u8>>,
    /// SPS NAL units.
    pub sps: Vec<Vec<u8>>,
    /// PPS NAL units.
    pub pps: Vec<Vec<u8>>,
    /// Raw configuration bytes.
    pub raw: Vec<u8>,
}

impl HevcConfig {
    /// Parse from raw bytes (simplified).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 23 {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "HEVC".to_string(),
                message: "HEVCDecoderConfigurationRecord too short".to_string(),
            });
        }

        let configuration_version = data[0];
        let general_profile_space = (data[1] >> 6) & 0x03;
        let general_tier_flag = (data[1] >> 5) & 0x01 != 0;
        let general_profile_idc = data[1] & 0x1F;
        let general_level_idc = data[12];

        // Simplified: just store raw data
        Ok(Self {
            configuration_version,
            general_profile_space,
            general_tier_flag,
            general_profile_idc,
            general_level_idc,
            vps: Vec::new(),
            sps: Vec::new(),
            pps: Vec::new(),
            raw: data.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_video_codec() {
        assert_eq!(VideoCodec::from_u8(7), Some(VideoCodec::Avc));
        assert_eq!(VideoCodec::from_u8(12), Some(VideoCodec::Hevc));
        assert_eq!(VideoCodec::from_u8(99), None);

        assert_eq!(VideoCodec::Avc.name(), "H.264/AVC");
        assert!(VideoCodec::Avc.has_avc_packet());
    }

    #[test]
    fn test_frame_type() {
        assert!(FrameType::Keyframe.is_keyframe());
        assert!(FrameType::GeneratedKeyframe.is_keyframe());
        assert!(!FrameType::Inter.is_keyframe());
    }

    #[test]
    fn test_video_tag_header_roundtrip() {
        let original = VideoTagHeader::avc_keyframe(100);

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 5);

        let mut cursor = Cursor::new(&buffer);
        let parsed = VideoTagHeader::parse(&mut cursor).unwrap();

        assert_eq!(original.frame_type, parsed.frame_type);
        assert_eq!(original.codec_id, parsed.codec_id);
        assert_eq!(original.avc_packet_type, parsed.avc_packet_type);
        assert_eq!(original.composition_time, parsed.composition_time);
    }

    #[test]
    fn test_avc_sequence_header() {
        let header = VideoTagHeader::avc_sequence_header();
        assert!(header.is_sequence_header());
        assert!(header.is_keyframe());
    }

    #[test]
    fn test_negative_composition_time() {
        let original = VideoTagHeader::avc_inter(-50);

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = VideoTagHeader::parse(&mut cursor).unwrap();

        assert_eq!(parsed.composition_time, -50);
    }

    #[test]
    fn test_avc_config_parse() {
        // Minimal AVCDecoderConfigurationRecord
        let data = [
            0x01, // configurationVersion
            0x64, // AVCProfileIndication (High)
            0x00, // profile_compatibility
            0x1F, // AVCLevelIndication (3.1)
            0xFF, // lengthSizeMinusOne (3)
            0xE1, // numOfSequenceParameterSets (1)
            0x00, 0x04, // SPS length
            0x67, 0x64, 0x00, 0x1F, // SPS data
            0x01, // numOfPictureParameterSets
            0x00, 0x02, // PPS length
            0x68, 0xEF, // PPS data
        ];

        let config = AvcConfig::parse(&data).unwrap();
        assert_eq!(config.configuration_version, 1);
        assert_eq!(config.avc_profile, 0x64);
        assert_eq!(config.avc_level, 0x1F);
        assert_eq!(config.sps.len(), 1);
        assert_eq!(config.pps.len(), 1);
        assert_eq!(config.nalu_length_size(), 4);
    }

    #[test]
    fn test_video_tag_header_size() {
        let avc = VideoTagHeader::avc_keyframe(0);
        assert_eq!(avc.size(), 5);

        let simple = VideoTagHeader::new(FrameType::Keyframe, VideoCodec::Sorenson);
        assert_eq!(simple.size(), 1);
    }

    #[test]
    fn test_enhanced_packet_type() {
        assert!(EnhancedPacketType::SequenceStart.is_sequence_header());
        assert!(!EnhancedPacketType::CodedFrames.is_sequence_header());
    }

    #[test]
    fn test_avc_packet_type() {
        assert!(AvcPacketType::SequenceHeader.is_sequence_header());
        assert!(!AvcPacketType::Nalu.is_sequence_header());
        assert!(!AvcPacketType::EndOfSequence.is_sequence_header());
    }
}
