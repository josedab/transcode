//! CMAF (Common Media Application Format) support.
//!
//! CMAF provides a unified container format compatible with both HLS and DASH.
//! It uses fragmented MP4 (fMP4) as the underlying container format.
//!
//! # Features
//!
//! - Initialization segment (moov box) generation
//! - Media segments (moof + mdat) compatible with HLS and DASH
//! - Low-latency chunk support (CMAF chunks)
//! - Byte-range addressing for efficient seeking
//! - Common Encryption (CENC) placeholder
//! - Track selection and switching support
//!
//! # Example
//!
//! ```no_run
//! use transcode_streaming::{CmafConfig, CmafWriter, Quality};
//!
//! let config = CmafConfig::new("output")
//!     .with_segment_duration(6.0)
//!     .with_chunk_duration(0.5)
//!     .with_quality(Quality::fhd_1080p())
//!     .with_low_latency(true);
//!
//! let mut writer = CmafWriter::new(config)?;
//! // Write segments...
//! # Ok::<(), transcode_streaming::StreamingError>(())
//! ```

use crate::error::StreamingError;
use crate::segment::{Quality, Segment, SegmentNaming, SegmentType};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

// ============================================================================
// CMAF Configuration
// ============================================================================

/// CMAF configuration.
#[derive(Debug, Clone)]
pub struct CmafConfig {
    /// Output directory.
    pub output_dir: PathBuf,
    /// Segment duration in seconds.
    pub segment_duration: f64,
    /// Chunk duration for low-latency streaming (in seconds).
    pub chunk_duration: f64,
    /// Quality levels.
    pub qualities: Vec<Quality>,
    /// Segment naming strategy.
    pub naming: SegmentNaming,
    /// Enable low-latency mode.
    pub low_latency: bool,
    /// Output format compatibility.
    pub output_format: CmafOutputFormat,
    /// Enable common encryption.
    pub encryption: Option<CmafEncryption>,
    /// Track configuration.
    pub tracks: Vec<CmafTrackConfig>,
    /// Use byte-range addressing instead of separate segment files.
    pub byte_range_mode: bool,
    /// Fragment type for HLS.
    pub fragment_type: FragmentType,
    /// Timescale for media timing.
    pub timescale: u32,
}

/// CMAF output format compatibility mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmafOutputFormat {
    /// HLS only (using fMP4 segments).
    HlsOnly,
    /// DASH only.
    DashOnly,
    /// Compatible with both HLS and DASH.
    HlsAndDash,
}

/// Fragment type for HLS compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentType {
    /// Regular segments.
    Regular,
    /// Low-latency fMP4 (LL-HLS parts).
    LowLatency,
    /// Independent segments with sync samples.
    Independent,
}

impl CmafConfig {
    /// Create new CMAF configuration.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            segment_duration: 6.0,
            chunk_duration: 0.5,
            qualities: vec![Quality::fhd_1080p()],
            naming: SegmentNaming::Sequential,
            low_latency: false,
            output_format: CmafOutputFormat::HlsAndDash,
            encryption: None,
            tracks: Vec::new(),
            byte_range_mode: false,
            fragment_type: FragmentType::Regular,
            timescale: 90000,
        }
    }

    /// Set segment duration.
    pub fn with_segment_duration(mut self, duration: f64) -> Self {
        self.segment_duration = duration.max(0.5);
        self
    }

    /// Set chunk duration for low-latency mode.
    pub fn with_chunk_duration(mut self, duration: f64) -> Self {
        self.chunk_duration = duration.max(0.033); // Min ~33ms (1 frame at 30fps)
        self
    }

    /// Add a quality level.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.qualities.push(quality);
        self
    }

    /// Set qualities (replaces existing).
    pub fn with_qualities(mut self, qualities: Vec<Quality>) -> Self {
        self.qualities = qualities;
        self
    }

    /// Enable low-latency mode.
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        if enabled {
            self.fragment_type = FragmentType::LowLatency;
        }
        self
    }

    /// Set output format compatibility.
    pub fn with_output_format(mut self, format: CmafOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Enable byte-range addressing.
    pub fn with_byte_range_mode(mut self, enabled: bool) -> Self {
        self.byte_range_mode = enabled;
        self
    }

    /// Set encryption configuration.
    pub fn with_encryption(mut self, encryption: CmafEncryption) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Add a track configuration.
    pub fn with_track(mut self, track: CmafTrackConfig) -> Self {
        self.tracks.push(track);
        self
    }

    /// Set naming strategy.
    pub fn with_naming(mut self, naming: SegmentNaming) -> Self {
        self.naming = naming;
        self
    }

    /// Set timescale.
    pub fn with_timescale(mut self, timescale: u32) -> Self {
        self.timescale = timescale;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.qualities.is_empty() {
            return Err(StreamingError::InvalidConfig(
                "At least one quality level is required".into(),
            ));
        }

        if self.segment_duration < 0.5 {
            return Err(StreamingError::InvalidConfig(
                "Segment duration must be at least 0.5 seconds".into(),
            ));
        }

        if self.low_latency && self.chunk_duration >= self.segment_duration {
            return Err(StreamingError::InvalidConfig(
                "Chunk duration must be less than segment duration".into(),
            ));
        }

        if self.chunk_duration < 0.033 {
            return Err(StreamingError::InvalidConfig(
                "Chunk duration must be at least 33ms".into(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// CMAF Track Configuration
// ============================================================================

/// Track type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Text/subtitle track.
    Text,
}

/// CMAF track configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmafTrackConfig {
    /// Track ID.
    pub id: u32,
    /// Track type.
    pub track_type: TrackType,
    /// Codec string (e.g., "avc1.64001f", "mp4a.40.2").
    pub codec: String,
    /// Language code (e.g., "en", "es").
    pub language: Option<String>,
    /// Track name/label.
    pub name: String,
    /// Whether this is the default track for its type.
    pub is_default: bool,
    /// Enable this track for output.
    pub enabled: bool,
    /// Track-specific timescale.
    pub timescale: Option<u32>,
    /// Video width (for video tracks).
    pub width: Option<u32>,
    /// Video height (for video tracks).
    pub height: Option<u32>,
    /// Audio sample rate (for audio tracks).
    pub sample_rate: Option<u32>,
    /// Audio channels (for audio tracks).
    pub channels: Option<u32>,
}

impl CmafTrackConfig {
    /// Create a new video track.
    pub fn video(id: u32, codec: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            id,
            track_type: TrackType::Video,
            codec: codec.into(),
            language: None,
            name: "Video".to_string(),
            is_default: true,
            enabled: true,
            timescale: Some(90000),
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
        }
    }

    /// Create a new audio track.
    pub fn audio(id: u32, codec: impl Into<String>, sample_rate: u32, channels: u32) -> Self {
        Self {
            id,
            track_type: TrackType::Audio,
            codec: codec.into(),
            language: Some("en".to_string()),
            name: "Audio".to_string(),
            is_default: true,
            enabled: true,
            timescale: Some(sample_rate),
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
        }
    }

    /// Create a new text track.
    pub fn text(id: u32, codec: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            id,
            track_type: TrackType::Text,
            codec: codec.into(),
            language: Some(language.into()),
            name: "Subtitles".to_string(),
            is_default: false,
            enabled: true,
            timescale: Some(1000),
            width: None,
            height: None,
            sample_rate: None,
            channels: None,
        }
    }

    /// Set track name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set as default track.
    pub fn with_default(mut self, is_default: bool) -> Self {
        self.is_default = is_default;
        self
    }

    /// Enable or disable track.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ============================================================================
// CMAF Encryption (CENC placeholder)
// ============================================================================

/// Encryption scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionScheme {
    /// CENC (Common Encryption) - AES-CTR.
    Cenc,
    /// CBCS (Common Block Cipher) - AES-CBC with pattern encryption.
    Cbcs,
    /// Cens - CTR mode with subsample pattern.
    Cens,
    /// Cbc1 - CBC mode.
    Cbc1,
}

/// DRM system identifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmSystem {
    /// System ID (UUID format).
    pub system_id: String,
    /// System-specific data (base64 encoded).
    pub pssh_data: Option<String>,
    /// License server URL.
    pub license_url: Option<String>,
}

impl DrmSystem {
    /// Create Widevine DRM system.
    pub fn widevine() -> Self {
        Self {
            system_id: "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed".to_string(),
            pssh_data: None,
            license_url: None,
        }
    }

    /// Create PlayReady DRM system.
    pub fn playready() -> Self {
        Self {
            system_id: "9a04f079-9840-4286-ab92-e65be0885f95".to_string(),
            pssh_data: None,
            license_url: None,
        }
    }

    /// Create FairPlay DRM system.
    pub fn fairplay() -> Self {
        Self {
            system_id: "94ce86fb-07ff-4f43-adb8-93d2fa968ca2".to_string(),
            pssh_data: None,
            license_url: None,
        }
    }

    /// Set PSSH data.
    pub fn with_pssh_data(mut self, data: impl Into<String>) -> Self {
        self.pssh_data = Some(data.into());
        self
    }

    /// Set license URL.
    pub fn with_license_url(mut self, url: impl Into<String>) -> Self {
        self.license_url = Some(url.into());
        self
    }
}

/// CMAF encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmafEncryption {
    /// Encryption scheme.
    pub scheme: EncryptionScheme,
    /// Key ID (16 bytes as hex string).
    pub key_id: String,
    /// Content encryption key (16 bytes as hex string).
    /// NOTE: In production, this should be securely managed.
    pub key: Option<String>,
    /// Initialization vector (8 or 16 bytes as hex string).
    pub iv: Option<String>,
    /// DRM systems.
    pub drm_systems: Vec<DrmSystem>,
    /// Default key ID for multi-key scenarios.
    pub default_kid: Option<String>,
    /// Encrypt audio.
    pub encrypt_audio: bool,
    /// Encrypt video.
    pub encrypt_video: bool,
}

impl CmafEncryption {
    /// Create new encryption configuration with CENC scheme.
    pub fn new_cenc(key_id: impl Into<String>) -> Self {
        Self {
            scheme: EncryptionScheme::Cenc,
            key_id: key_id.into(),
            key: None,
            iv: None,
            drm_systems: Vec::new(),
            default_kid: None,
            encrypt_audio: true,
            encrypt_video: true,
        }
    }

    /// Create new encryption configuration with CBCS scheme (Apple-compatible).
    pub fn new_cbcs(key_id: impl Into<String>) -> Self {
        Self {
            scheme: EncryptionScheme::Cbcs,
            key_id: key_id.into(),
            key: None,
            iv: None,
            drm_systems: Vec::new(),
            default_kid: None,
            encrypt_audio: true,
            encrypt_video: true,
        }
    }

    /// Set encryption key for content protection.
    ///
    /// # Security Considerations
    ///
    /// **WARNING:** This method stores the encryption key in memory as a plain string.
    /// For production deployments, consider the following:
    ///
    /// - **Use a Key Management System (KMS):** Integrate with AWS KMS, Google Cloud KMS,
    ///   Azure Key Vault, or HashiCorp Vault for secure key storage and rotation.
    /// - **Key Rotation:** Implement regular key rotation policies.
    /// - **Memory Protection:** Keys in memory may be vulnerable to memory dumps or
    ///   side-channel attacks. Consider using secure memory allocators for sensitive data.
    /// - **Access Control:** Ensure only authorized processes can access encryption keys.
    /// - **Audit Logging:** Log key access events for security auditing.
    ///
    /// This API is designed for flexibility and testing. For production DRM workflows,
    /// use [`Self::with_drm_system`] with a properly configured DRM provider that handles
    /// key management securely.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For testing only - never hardcode keys in production!
    /// let config = CmafEncryptionConfig::new()
    ///     .with_key("0123456789abcdef0123456789abcdef");
    /// ```
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Set initialization vector.
    pub fn with_iv(mut self, iv: impl Into<String>) -> Self {
        self.iv = Some(iv.into());
        self
    }

    /// Add DRM system.
    pub fn with_drm_system(mut self, system: DrmSystem) -> Self {
        self.drm_systems.push(system);
        self
    }

    /// Enable Widevine DRM.
    pub fn with_widevine(mut self) -> Self {
        self.drm_systems.push(DrmSystem::widevine());
        self
    }

    /// Enable PlayReady DRM.
    pub fn with_playready(mut self) -> Self {
        self.drm_systems.push(DrmSystem::playready());
        self
    }

    /// Enable FairPlay DRM.
    pub fn with_fairplay(mut self) -> Self {
        self.drm_systems.push(DrmSystem::fairplay());
        self
    }

    /// Generate PSSH box data (placeholder).
    pub fn generate_pssh(&self) -> Vec<u8> {
        // This is a placeholder for actual PSSH generation.
        // In production, this would create proper PSSH boxes for each DRM system.
        let mut data = Vec::new();

        // PSSH box header placeholder
        data.extend_from_slice(&[0, 0, 0, 0]); // Size (to be filled)
        data.extend_from_slice(b"pssh"); // Type
        data.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // System ID would go here (16 bytes)
        // KID count and KIDs would follow
        // PSSH data would follow

        // Update size
        let size = data.len() as u32;
        data[0..4].copy_from_slice(&size.to_be_bytes());

        data
    }
}

// ============================================================================
// CMAF Segment Structures
// ============================================================================

/// CMAF initialization segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmafInitSegment {
    /// Track ID.
    pub track_id: u32,
    /// Quality name.
    pub quality_name: String,
    /// Codec string.
    pub codec: String,
    /// Width (for video).
    pub width: Option<u32>,
    /// Height (for video).
    pub height: Option<u32>,
    /// Sample rate (for audio).
    pub sample_rate: Option<u32>,
    /// Channels (for audio).
    pub channels: Option<u32>,
    /// Timescale.
    pub timescale: u32,
    /// File path.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Raw fMP4 data (ftyp + moov).
    #[serde(skip)]
    pub data: Vec<u8>,
}

impl CmafInitSegment {
    /// Create a video initialization segment.
    pub fn video(
        track_id: u32,
        quality_name: impl Into<String>,
        codec: impl Into<String>,
        width: u32,
        height: u32,
        timescale: u32,
    ) -> Self {
        Self {
            track_id,
            quality_name: quality_name.into(),
            codec: codec.into(),
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
            timescale,
            path: String::new(),
            size: 0,
            data: Vec::new(),
        }
    }

    /// Create an audio initialization segment.
    pub fn audio(
        track_id: u32,
        quality_name: impl Into<String>,
        codec: impl Into<String>,
        sample_rate: u32,
        channels: u32,
        timescale: u32,
    ) -> Self {
        Self {
            track_id,
            quality_name: quality_name.into(),
            codec: codec.into(),
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            timescale,
            path: String::new(),
            size: 0,
            data: Vec::new(),
        }
    }

    /// Generate fMP4 initialization segment data.
    pub fn generate_fmp4(&mut self) -> Vec<u8> {
        let mut data = Vec::new();

        // ftyp box
        data.extend(self.generate_ftyp());

        // moov box
        data.extend(self.generate_moov());

        self.data = data.clone();
        self.size = data.len() as u64;

        data
    }

    /// Generate ftyp box.
    fn generate_ftyp(&self) -> Vec<u8> {
        let mut ftyp = Vec::new();

        // Box size (will be updated)
        ftyp.extend_from_slice(&[0, 0, 0, 0]);
        // Box type
        ftyp.extend_from_slice(b"ftyp");
        // Major brand - iso6 for CMAF
        ftyp.extend_from_slice(b"iso6");
        // Minor version
        ftyp.extend_from_slice(&[0, 0, 0, 1]);
        // Compatible brands
        ftyp.extend_from_slice(b"iso6"); // ISO Base Media
        ftyp.extend_from_slice(b"cmfc"); // CMAF
        ftyp.extend_from_slice(b"msdh"); // Media Segment
        ftyp.extend_from_slice(b"msix"); // Media Segment Index
        ftyp.extend_from_slice(b"dash"); // DASH
        ftyp.extend_from_slice(b"hlsf"); // HLS fMP4

        // Update size
        let size = ftyp.len() as u32;
        ftyp[0..4].copy_from_slice(&size.to_be_bytes());

        ftyp
    }

    /// Generate moov box (simplified).
    fn generate_moov(&self) -> Vec<u8> {
        let mut moov = Vec::new();

        // Box header
        moov.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        moov.extend_from_slice(b"moov");

        // mvhd (movie header)
        moov.extend(self.generate_mvhd());

        // trak (track)
        moov.extend(self.generate_trak());

        // mvex (movie extends - for fragmented MP4)
        moov.extend(self.generate_mvex());

        // Update moov size
        let size = moov.len() as u32;
        moov[0..4].copy_from_slice(&size.to_be_bytes());

        moov
    }

    /// Generate mvhd box.
    fn generate_mvhd(&self) -> Vec<u8> {
        let mut mvhd = Vec::new();

        mvhd.extend_from_slice(&[0, 0, 0, 108]); // Size
        mvhd.extend_from_slice(b"mvhd");
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Creation time
        mvhd.extend_from_slice(&[0, 0, 0, 0]);
        // Modification time
        mvhd.extend_from_slice(&[0, 0, 0, 0]);
        // Timescale
        mvhd.extend_from_slice(&self.timescale.to_be_bytes());
        // Duration (0 for fragmented)
        mvhd.extend_from_slice(&[0, 0, 0, 0]);

        // Rate (1.0 = 0x00010000)
        mvhd.extend_from_slice(&[0, 1, 0, 0]);
        // Volume (1.0 = 0x0100)
        mvhd.extend_from_slice(&[1, 0]);
        // Reserved
        mvhd.extend_from_slice(&[0, 0]);
        mvhd.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

        // Matrix (identity)
        mvhd.extend_from_slice(&[0, 1, 0, 0]); // a = 1.0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // b = 0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // u = 0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // c = 0
        mvhd.extend_from_slice(&[0, 1, 0, 0]); // d = 1.0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // v = 0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // x = 0
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // y = 0
        mvhd.extend_from_slice(&[0x40, 0, 0, 0]); // w = 1.0

        // Pre-defined
        mvhd.extend_from_slice(&[0; 24]);

        // Next track ID
        mvhd.extend_from_slice(&(self.track_id + 1).to_be_bytes());

        mvhd
    }

    /// Generate trak box (simplified).
    fn generate_trak(&self) -> Vec<u8> {
        let mut trak = Vec::new();

        trak.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        trak.extend_from_slice(b"trak");

        // tkhd (track header)
        trak.extend(self.generate_tkhd());

        // mdia (media)
        trak.extend(self.generate_mdia());

        // Update size
        let size = trak.len() as u32;
        trak[0..4].copy_from_slice(&size.to_be_bytes());

        trak
    }

    /// Generate tkhd box.
    fn generate_tkhd(&self) -> Vec<u8> {
        let mut tkhd = Vec::new();

        tkhd.extend_from_slice(&[0, 0, 0, 92]); // Size
        tkhd.extend_from_slice(b"tkhd");
        tkhd.extend_from_slice(&[0, 0, 0, 7]); // Version 0, flags (enabled, in_movie, in_preview)

        // Creation/modification time
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);

        // Track ID
        tkhd.extend_from_slice(&self.track_id.to_be_bytes());

        // Reserved
        tkhd.extend_from_slice(&[0, 0, 0, 0]);

        // Duration (0 for fragmented)
        tkhd.extend_from_slice(&[0, 0, 0, 0]);

        // Reserved
        tkhd.extend_from_slice(&[0; 8]);

        // Layer and alternate group
        tkhd.extend_from_slice(&[0, 0]);
        tkhd.extend_from_slice(&[0, 0]);

        // Volume (for audio: 0x0100, for video: 0)
        if self.sample_rate.is_some() {
            tkhd.extend_from_slice(&[1, 0]);
        } else {
            tkhd.extend_from_slice(&[0, 0]);
        }

        // Reserved
        tkhd.extend_from_slice(&[0, 0]);

        // Matrix (identity)
        tkhd.extend_from_slice(&[0, 1, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 1, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0]);
        tkhd.extend_from_slice(&[0x40, 0, 0, 0]);

        // Width and height (16.16 fixed point)
        let width = self.width.unwrap_or(0);
        let height = self.height.unwrap_or(0);
        tkhd.extend_from_slice(&(width << 16).to_be_bytes());
        tkhd.extend_from_slice(&(height << 16).to_be_bytes());

        tkhd
    }

    /// Generate mdia box (simplified).
    fn generate_mdia(&self) -> Vec<u8> {
        let mut mdia = Vec::new();

        mdia.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        mdia.extend_from_slice(b"mdia");

        // mdhd (media header)
        mdia.extend(self.generate_mdhd());

        // hdlr (handler)
        mdia.extend(self.generate_hdlr());

        // minf (media info)
        mdia.extend(self.generate_minf());

        // Update size
        let size = mdia.len() as u32;
        mdia[0..4].copy_from_slice(&size.to_be_bytes());

        mdia
    }

    /// Generate mdhd box.
    fn generate_mdhd(&self) -> Vec<u8> {
        let mut mdhd = Vec::new();

        mdhd.extend_from_slice(&[0, 0, 0, 32]); // Size
        mdhd.extend_from_slice(b"mdhd");
        mdhd.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Creation/modification time
        mdhd.extend_from_slice(&[0, 0, 0, 0]);
        mdhd.extend_from_slice(&[0, 0, 0, 0]);

        // Timescale
        mdhd.extend_from_slice(&self.timescale.to_be_bytes());

        // Duration (0 for fragmented)
        mdhd.extend_from_slice(&[0, 0, 0, 0]);

        // Language (und = undetermined)
        mdhd.extend_from_slice(&[0x55, 0xc4]); // Packed ISO 639-2 "und"

        // Pre-defined
        mdhd.extend_from_slice(&[0, 0]);

        mdhd
    }

    /// Generate hdlr box.
    fn generate_hdlr(&self) -> Vec<u8> {
        let mut hdlr = Vec::new();

        let handler_type = if self.width.is_some() { b"vide" } else { b"soun" };
        let name = if self.width.is_some() {
            b"VideoHandler\0"
        } else {
            b"SoundHandler\0"
        };

        let size = 32 + name.len();

        hdlr.extend_from_slice(&(size as u32).to_be_bytes());
        hdlr.extend_from_slice(b"hdlr");
        hdlr.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Pre-defined
        hdlr.extend_from_slice(&[0, 0, 0, 0]);

        // Handler type
        hdlr.extend_from_slice(handler_type);

        // Reserved
        hdlr.extend_from_slice(&[0; 12]);

        // Name
        hdlr.extend_from_slice(name);

        hdlr
    }

    /// Generate minf box (simplified).
    fn generate_minf(&self) -> Vec<u8> {
        let mut minf = Vec::new();

        minf.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        minf.extend_from_slice(b"minf");

        // vmhd/smhd (video/sound media header)
        if self.width.is_some() {
            minf.extend(self.generate_vmhd());
        } else {
            minf.extend(self.generate_smhd());
        }

        // dinf (data information)
        minf.extend(self.generate_dinf());

        // stbl (sample table)
        minf.extend(self.generate_stbl());

        // Update size
        let size = minf.len() as u32;
        minf[0..4].copy_from_slice(&size.to_be_bytes());

        minf
    }

    /// Generate vmhd box.
    fn generate_vmhd(&self) -> Vec<u8> {
        let mut vmhd = Vec::new();

        vmhd.extend_from_slice(&[0, 0, 0, 20]); // Size
        vmhd.extend_from_slice(b"vmhd");
        vmhd.extend_from_slice(&[0, 0, 0, 1]); // Version 0, flags = 1

        // Graphics mode and opcolor
        vmhd.extend_from_slice(&[0, 0]); // graphicsmode
        vmhd.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // opcolor

        vmhd
    }

    /// Generate smhd box.
    fn generate_smhd(&self) -> Vec<u8> {
        let mut smhd = Vec::new();

        smhd.extend_from_slice(&[0, 0, 0, 16]); // Size
        smhd.extend_from_slice(b"smhd");
        smhd.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Balance and reserved
        smhd.extend_from_slice(&[0, 0]); // balance
        smhd.extend_from_slice(&[0, 0]); // reserved

        smhd
    }

    /// Generate dinf box.
    fn generate_dinf(&self) -> Vec<u8> {
        let mut dinf = Vec::new();

        dinf.extend_from_slice(&[0, 0, 0, 36]); // Size
        dinf.extend_from_slice(b"dinf");

        // dref (data reference)
        dinf.extend_from_slice(&[0, 0, 0, 28]); // Size
        dinf.extend_from_slice(b"dref");
        dinf.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        dinf.extend_from_slice(&[0, 0, 0, 1]); // Entry count

        // url entry
        dinf.extend_from_slice(&[0, 0, 0, 12]); // Size
        dinf.extend_from_slice(b"url ");
        dinf.extend_from_slice(&[0, 0, 0, 1]); // Version 0, flags = 1 (self-contained)

        dinf
    }

    /// Generate stbl box (empty for fragmented MP4).
    fn generate_stbl(&self) -> Vec<u8> {
        let mut stbl = Vec::new();

        stbl.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        stbl.extend_from_slice(b"stbl");

        // stsd (sample description)
        stbl.extend(self.generate_stsd());

        // Empty required boxes for fragmented MP4
        // stts
        stbl.extend_from_slice(&[0, 0, 0, 16]); // Size
        stbl.extend_from_slice(b"stts");
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Entry count

        // stsc
        stbl.extend_from_slice(&[0, 0, 0, 16]); // Size
        stbl.extend_from_slice(b"stsc");
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Entry count

        // stsz
        stbl.extend_from_slice(&[0, 0, 0, 20]); // Size
        stbl.extend_from_slice(b"stsz");
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Sample size
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Sample count

        // stco
        stbl.extend_from_slice(&[0, 0, 0, 16]); // Size
        stbl.extend_from_slice(b"stco");
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        stbl.extend_from_slice(&[0, 0, 0, 0]); // Entry count

        // Update size
        let size = stbl.len() as u32;
        stbl[0..4].copy_from_slice(&size.to_be_bytes());

        stbl
    }

    /// Generate stsd box (sample description).
    fn generate_stsd(&self) -> Vec<u8> {
        let mut stsd = Vec::new();

        stsd.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        stsd.extend_from_slice(b"stsd");
        stsd.extend_from_slice(&[0, 0, 0, 0]); // Version and flags
        stsd.extend_from_slice(&[0, 0, 0, 1]); // Entry count

        // Sample entry (avc1 for H.264 video, mp4a for AAC audio)
        if self.width.is_some() {
            stsd.extend(self.generate_avc1());
        } else {
            stsd.extend(self.generate_mp4a());
        }

        // Update size
        let size = stsd.len() as u32;
        stsd[0..4].copy_from_slice(&size.to_be_bytes());

        stsd
    }

    /// Generate avc1 sample entry (simplified).
    fn generate_avc1(&self) -> Vec<u8> {
        let mut avc1 = Vec::new();

        avc1.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        avc1.extend_from_slice(b"avc1");

        // Reserved
        avc1.extend_from_slice(&[0; 6]);
        // Data reference index
        avc1.extend_from_slice(&[0, 1]);

        // Pre-defined + reserved
        avc1.extend_from_slice(&[0; 16]);

        // Width and height
        let width = self.width.unwrap_or(1920);
        let height = self.height.unwrap_or(1080);
        avc1.extend_from_slice(&(width as u16).to_be_bytes());
        avc1.extend_from_slice(&(height as u16).to_be_bytes());

        // Horizontal/vertical resolution (72 dpi)
        avc1.extend_from_slice(&[0, 0x48, 0, 0]); // 72.0
        avc1.extend_from_slice(&[0, 0x48, 0, 0]); // 72.0

        // Reserved
        avc1.extend_from_slice(&[0, 0, 0, 0]);

        // Frame count
        avc1.extend_from_slice(&[0, 1]);

        // Compressor name (32 bytes, padded)
        let mut compressor = [0u8; 32];
        compressor[0..4].copy_from_slice(b"AVC1");
        avc1.extend_from_slice(&compressor);

        // Depth
        avc1.extend_from_slice(&[0, 24]);

        // Pre-defined
        avc1.extend_from_slice(&[0xff, 0xff]);

        // avcC box (placeholder - would need actual SPS/PPS in production)
        avc1.extend(self.generate_avcc_placeholder());

        // Update size
        let size = avc1.len() as u32;
        avc1[0..4].copy_from_slice(&size.to_be_bytes());

        avc1
    }

    /// Generate avcC box placeholder.
    fn generate_avcc_placeholder(&self) -> Vec<u8> {
        let mut avcc = Vec::new();

        // Minimal avcC structure
        avcc.extend_from_slice(&[0, 0, 0, 19]); // Size
        avcc.extend_from_slice(b"avcC");

        // Version
        avcc.push(1);
        // Profile (Main = 77)
        avcc.push(77);
        // Profile compatibility
        avcc.push(0);
        // Level (3.1)
        avcc.push(31);
        // Length size minus one (3 = 4 bytes)
        avcc.push(0xff);
        // Number of SPS (0 for now)
        avcc.push(0xe0);
        // Number of PPS (0 for now)
        avcc.push(0);

        avcc
    }

    /// Generate mp4a sample entry (simplified).
    fn generate_mp4a(&self) -> Vec<u8> {
        let mut mp4a = Vec::new();

        mp4a.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        mp4a.extend_from_slice(b"mp4a");

        // Reserved
        mp4a.extend_from_slice(&[0; 6]);
        // Data reference index
        mp4a.extend_from_slice(&[0, 1]);

        // Reserved
        mp4a.extend_from_slice(&[0; 8]);

        // Channel count
        let channels = self.channels.unwrap_or(2);
        mp4a.extend_from_slice(&(channels as u16).to_be_bytes());

        // Sample size (16 bits)
        mp4a.extend_from_slice(&[0, 16]);

        // Pre-defined + reserved
        mp4a.extend_from_slice(&[0, 0, 0, 0]);

        // Sample rate (16.16 fixed point)
        let sample_rate = self.sample_rate.unwrap_or(48000);
        mp4a.extend_from_slice(&(sample_rate as u16).to_be_bytes());
        mp4a.extend_from_slice(&[0, 0]);

        // esds box (placeholder)
        mp4a.extend(self.generate_esds_placeholder());

        // Update size
        let size = mp4a.len() as u32;
        mp4a[0..4].copy_from_slice(&size.to_be_bytes());

        mp4a
    }

    /// Generate esds box placeholder.
    fn generate_esds_placeholder(&self) -> Vec<u8> {
        let mut esds = Vec::new();

        esds.extend_from_slice(&[0, 0, 0, 39]); // Size
        esds.extend_from_slice(b"esds");
        esds.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // ES Descriptor
        esds.push(0x03); // ES_DescrTag
        esds.push(23); // Length
        esds.extend_from_slice(&[0, 0]); // ES_ID
        esds.push(0); // Flags

        // DecoderConfigDescriptor
        esds.push(0x04); // DecoderConfigDescrTag
        esds.push(15); // Length
        esds.push(0x40); // ObjectTypeIndication (AAC)
        esds.push(0x15); // StreamType (AudioStream)
        esds.extend_from_slice(&[0, 0, 0]); // BufferSizeDB
        esds.extend_from_slice(&[0, 0, 0, 0]); // MaxBitrate
        esds.extend_from_slice(&[0, 0, 0, 0]); // AvgBitrate

        // DecoderSpecificInfo (placeholder)
        esds.push(0x05); // DecSpecificInfoTag
        esds.push(2); // Length
        esds.extend_from_slice(&[0x11, 0x90]); // AAC-LC config

        // SLConfigDescriptor
        esds.push(0x06); // SLConfigDescrTag
        esds.push(1); // Length
        esds.push(0x02); // Predefined

        esds
    }

    /// Generate mvex box (movie extends for fragmented MP4).
    fn generate_mvex(&self) -> Vec<u8> {
        let mut mvex = Vec::new();

        mvex.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        mvex.extend_from_slice(b"mvex");

        // trex (track extends)
        mvex.extend(self.generate_trex());

        // Update size
        let size = mvex.len() as u32;
        mvex[0..4].copy_from_slice(&size.to_be_bytes());

        mvex
    }

    /// Generate trex box.
    fn generate_trex(&self) -> Vec<u8> {
        let mut trex = Vec::new();

        trex.extend_from_slice(&[0, 0, 0, 32]); // Size
        trex.extend_from_slice(b"trex");
        trex.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Track ID
        trex.extend_from_slice(&self.track_id.to_be_bytes());

        // Default sample description index
        trex.extend_from_slice(&[0, 0, 0, 1]);
        // Default sample duration
        trex.extend_from_slice(&[0, 0, 0, 0]);
        // Default sample size
        trex.extend_from_slice(&[0, 0, 0, 0]);
        // Default sample flags
        trex.extend_from_slice(&[0, 0, 0, 0]);

        trex
    }
}

/// CMAF media segment (moof + mdat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmafMediaSegment {
    /// Segment sequence number.
    pub sequence: u64,
    /// Duration in timescale units.
    pub duration_ticks: u64,
    /// Duration in seconds.
    pub duration: f64,
    /// Start time in seconds.
    pub start_time: f64,
    /// Decode time in timescale units.
    pub decode_time: u64,
    /// Track ID.
    pub track_id: u32,
    /// Quality name.
    pub quality_name: String,
    /// File path.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Whether this starts with a sync sample (keyframe).
    pub starts_with_sap: bool,
    /// SAP type (1-6).
    pub sap_type: u8,
    /// Chunks in this segment (for low-latency).
    pub chunks: Vec<CmafChunk>,
    /// Byte range if using byte-range addressing.
    pub byte_range: Option<ByteRange>,
    /// Raw fMP4 data.
    #[serde(skip)]
    pub data: Vec<u8>,
}

impl CmafMediaSegment {
    /// Create a new media segment.
    pub fn new(
        sequence: u64,
        track_id: u32,
        quality_name: impl Into<String>,
        duration: f64,
        start_time: f64,
        timescale: u32,
    ) -> Self {
        Self {
            sequence,
            duration_ticks: (duration * timescale as f64) as u64,
            duration,
            start_time,
            decode_time: (start_time * timescale as f64) as u64,
            track_id,
            quality_name: quality_name.into(),
            path: String::new(),
            size: 0,
            starts_with_sap: true,
            sap_type: 1,
            chunks: Vec::new(),
            byte_range: None,
            data: Vec::new(),
        }
    }

    /// Set as starting with sync sample.
    pub fn with_sap(mut self, starts_with_sap: bool, sap_type: u8) -> Self {
        self.starts_with_sap = starts_with_sap;
        self.sap_type = sap_type;
        self
    }

    /// Generate fMP4 media segment data.
    pub fn generate_fmp4(&mut self, samples: &[SampleInfo]) -> Vec<u8> {
        let mut data = Vec::new();

        // styp box (segment type)
        data.extend(self.generate_styp());

        // sidx box (segment index) - optional but recommended
        data.extend(self.generate_sidx(samples));

        // moof box (movie fragment)
        let moof = self.generate_moof(samples);
        let moof_size = moof.len();
        data.extend(moof);

        // mdat box (media data)
        let mdat_offset = data.len();
        data.extend(self.generate_mdat(samples, moof_size, mdat_offset));

        self.data = data.clone();
        self.size = data.len() as u64;

        data
    }

    /// Generate styp box.
    fn generate_styp(&self) -> Vec<u8> {
        let mut styp = Vec::new();

        styp.extend_from_slice(&[0, 0, 0, 28]); // Size
        styp.extend_from_slice(b"styp");
        styp.extend_from_slice(b"msdh"); // Major brand
        styp.extend_from_slice(&[0, 0, 0, 0]); // Minor version
        styp.extend_from_slice(b"msdh"); // Compatible brands
        styp.extend_from_slice(b"msix");
        styp.extend_from_slice(b"cmfc");

        styp
    }

    /// Generate sidx box.
    fn generate_sidx(&self, samples: &[SampleInfo]) -> Vec<u8> {
        let mut sidx = Vec::new();

        sidx.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        sidx.extend_from_slice(b"sidx");
        sidx.extend_from_slice(&[1, 0, 0, 0]); // Version 1, flags

        // Reference ID
        sidx.extend_from_slice(&self.track_id.to_be_bytes());

        // Timescale (assuming 90000 for video)
        sidx.extend_from_slice(&90000u32.to_be_bytes());

        // Earliest presentation time (64-bit for version 1)
        sidx.extend_from_slice(&self.decode_time.to_be_bytes());

        // First offset (0)
        sidx.extend_from_slice(&0u64.to_be_bytes());

        // Reserved
        sidx.extend_from_slice(&[0, 0]);

        // Reference count
        sidx.extend_from_slice(&1u16.to_be_bytes());

        // Reference entry
        let total_size: u32 = samples.iter().map(|s| s.size).sum();
        // Reference type (0 = media) + referenced size
        sidx.extend_from_slice(&total_size.to_be_bytes());

        // Subsegment duration
        sidx.extend_from_slice(&(self.duration_ticks as u32).to_be_bytes());

        // starts_with_SAP + SAP_type + SAP_delta_time
        let sap_flags = if self.starts_with_sap {
            0x80000000 | ((self.sap_type as u32) << 28)
        } else {
            0
        };
        sidx.extend_from_slice(&sap_flags.to_be_bytes());

        // Update size
        let size = sidx.len() as u32;
        sidx[0..4].copy_from_slice(&size.to_be_bytes());

        sidx
    }

    /// Generate moof box.
    fn generate_moof(&self, samples: &[SampleInfo]) -> Vec<u8> {
        let mut moof = Vec::new();

        moof.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        moof.extend_from_slice(b"moof");

        // mfhd (movie fragment header)
        moof.extend(self.generate_mfhd());

        // traf (track fragment)
        moof.extend(self.generate_traf(samples));

        // Update size
        let size = moof.len() as u32;
        moof[0..4].copy_from_slice(&size.to_be_bytes());

        moof
    }

    /// Generate mfhd box.
    fn generate_mfhd(&self) -> Vec<u8> {
        let mut mfhd = Vec::new();

        mfhd.extend_from_slice(&[0, 0, 0, 16]); // Size
        mfhd.extend_from_slice(b"mfhd");
        mfhd.extend_from_slice(&[0, 0, 0, 0]); // Version and flags

        // Sequence number
        mfhd.extend_from_slice(&(self.sequence as u32).to_be_bytes());

        mfhd
    }

    /// Generate traf box.
    fn generate_traf(&self, samples: &[SampleInfo]) -> Vec<u8> {
        let mut traf = Vec::new();

        traf.extend_from_slice(&[0, 0, 0, 0]); // Size placeholder
        traf.extend_from_slice(b"traf");

        // tfhd (track fragment header)
        traf.extend(self.generate_tfhd());

        // tfdt (track fragment decode time)
        traf.extend(self.generate_tfdt());

        // trun (track run)
        traf.extend(self.generate_trun(samples));

        // Update size
        let size = traf.len() as u32;
        traf[0..4].copy_from_slice(&size.to_be_bytes());

        traf
    }

    /// Generate tfhd box.
    fn generate_tfhd(&self) -> Vec<u8> {
        let mut tfhd = Vec::new();

        tfhd.extend_from_slice(&[0, 0, 0, 16]); // Size
        tfhd.extend_from_slice(b"tfhd");
        // Version 0, flags: default-base-is-moof (0x020000)
        tfhd.extend_from_slice(&[0, 0x02, 0, 0]);

        // Track ID
        tfhd.extend_from_slice(&self.track_id.to_be_bytes());

        tfhd
    }

    /// Generate tfdt box (version 1 for 64-bit decode time).
    fn generate_tfdt(&self) -> Vec<u8> {
        let mut tfdt = Vec::new();

        tfdt.extend_from_slice(&[0, 0, 0, 20]); // Size
        tfdt.extend_from_slice(b"tfdt");
        tfdt.extend_from_slice(&[1, 0, 0, 0]); // Version 1, flags

        // Base media decode time (64-bit)
        tfdt.extend_from_slice(&self.decode_time.to_be_bytes());

        tfdt
    }

    /// Generate trun box.
    fn generate_trun(&self, samples: &[SampleInfo]) -> Vec<u8> {
        let mut trun = Vec::new();

        // Calculate size based on samples
        // Header (12) + data offset (4) + per-sample data
        let per_sample_size = 4 + 4 + 4 + 4; // duration + size + flags + cts_offset
        let size = 12 + 4 + (samples.len() * per_sample_size);

        trun.extend_from_slice(&(size as u32).to_be_bytes());
        trun.extend_from_slice(b"trun");

        // Version 0, flags:
        // 0x001 = data-offset-present
        // 0x100 = sample-duration-present
        // 0x200 = sample-size-present
        // 0x400 = sample-flags-present
        // 0x800 = sample-composition-time-offsets-present
        let flags = 0x000f01u32;
        trun.push(0); // Version
        trun.extend_from_slice(&flags.to_be_bytes()[1..]);

        // Sample count
        trun.extend_from_slice(&(samples.len() as u32).to_be_bytes());

        // Data offset (will be updated after we know moof size)
        // This is a placeholder - actual calculation happens in generate_mdat
        trun.extend_from_slice(&[0, 0, 0, 8]); // Placeholder offset

        // Per-sample data
        for sample in samples {
            // Sample duration
            trun.extend_from_slice(&sample.duration.to_be_bytes());
            // Sample size
            trun.extend_from_slice(&sample.size.to_be_bytes());
            // Sample flags
            let flags: u32 = if sample.is_sync {
                0x02000000 // is-leading = 0, depends-on = 2 (not I-picture), is-depended-on = 0
            } else {
                0x01010000 // Non-sync sample
            };
            trun.extend_from_slice(&flags.to_be_bytes());
            // Composition time offset
            trun.extend_from_slice(&sample.cts_offset.to_be_bytes());
        }

        trun
    }

    /// Generate mdat box.
    fn generate_mdat(&self, samples: &[SampleInfo], moof_size: usize, mdat_offset: usize) -> Vec<u8> {
        let mut mdat = Vec::new();

        // Calculate total data size
        let data_size: usize = samples.iter().map(|s| s.data.len()).sum();
        let box_size = 8 + data_size;

        mdat.extend_from_slice(&(box_size as u32).to_be_bytes());
        mdat.extend_from_slice(b"mdat");

        // Append sample data
        for sample in samples {
            mdat.extend_from_slice(&sample.data);
        }

        // Note: In a real implementation, we would need to go back and update
        // the data offset in the trun box. The offset should be:
        // moof_size + 8 (mdat header)
        let _ = (moof_size, mdat_offset); // Acknowledge these are used for offset calculation

        mdat
    }

    /// Add a chunk to this segment.
    pub fn add_chunk(&mut self, chunk: CmafChunk) {
        self.chunks.push(chunk);
    }
}

/// Sample information for segment generation.
#[derive(Debug, Clone)]
pub struct SampleInfo {
    /// Sample duration in timescale units.
    pub duration: u32,
    /// Sample size in bytes.
    pub size: u32,
    /// Whether this is a sync sample (keyframe).
    pub is_sync: bool,
    /// Composition time offset.
    pub cts_offset: i32,
    /// Sample data.
    pub data: Vec<u8>,
}

impl SampleInfo {
    /// Create a new sample.
    pub fn new(data: Vec<u8>, duration: u32, is_sync: bool) -> Self {
        Self {
            duration,
            size: data.len() as u32,
            is_sync,
            cts_offset: 0,
            data,
        }
    }

    /// Set composition time offset.
    pub fn with_cts_offset(mut self, offset: i32) -> Self {
        self.cts_offset = offset;
        self
    }
}

/// CMAF chunk (for low-latency streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmafChunk {
    /// Chunk index within segment.
    pub index: u32,
    /// Duration in seconds.
    pub duration: f64,
    /// Start time within segment (seconds).
    pub start_time: f64,
    /// File path (if separate file).
    pub path: Option<String>,
    /// Byte range within segment file.
    pub byte_range: Option<ByteRange>,
    /// Size in bytes.
    pub size: u64,
    /// Whether this is an independent chunk.
    pub independent: bool,
}

impl CmafChunk {
    /// Create a new chunk.
    pub fn new(index: u32, duration: f64, start_time: f64) -> Self {
        Self {
            index,
            duration,
            start_time,
            path: None,
            byte_range: None,
            size: 0,
            independent: false,
        }
    }

    /// Set byte range.
    pub fn with_byte_range(mut self, start: u64, length: u64) -> Self {
        self.byte_range = Some(ByteRange { start, length });
        self
    }

    /// Set as independent.
    pub fn with_independent(mut self, independent: bool) -> Self {
        self.independent = independent;
        self
    }
}

/// Byte range for addressing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ByteRange {
    /// Start offset in bytes.
    pub start: u64,
    /// Length in bytes.
    pub length: u64,
}

impl ByteRange {
    /// Create a new byte range.
    pub fn new(start: u64, length: u64) -> Self {
        Self { start, length }
    }

    /// Get end offset (exclusive).
    pub fn end(&self) -> u64 {
        self.start + self.length
    }

    /// Format as HTTP Range header value.
    pub fn to_range_header(&self) -> String {
        format!("bytes={}-{}", self.start, self.end() - 1)
    }

    /// Format for HLS BYTERANGE tag.
    pub fn to_hls_byterange(&self) -> String {
        format!("{}@{}", self.length, self.start)
    }
}

// ============================================================================
// Track Selection and Switching
// ============================================================================

/// Track selection criteria.
#[derive(Debug, Clone)]
pub struct TrackSelector {
    /// Selected track IDs by type.
    selections: HashMap<TrackType, Vec<u32>>,
    /// Preferred language.
    preferred_language: Option<String>,
    /// Maximum video bitrate.
    max_video_bitrate: Option<u64>,
    /// Minimum video resolution (width).
    min_video_width: Option<u32>,
}

impl TrackSelector {
    /// Create a new track selector.
    pub fn new() -> Self {
        Self {
            selections: HashMap::new(),
            preferred_language: None,
            max_video_bitrate: None,
            min_video_width: None,
        }
    }

    /// Select a specific track by ID.
    pub fn select_track(&mut self, track_type: TrackType, track_id: u32) {
        self.selections
            .entry(track_type)
            .or_default()
            .push(track_id);
    }

    /// Set preferred language for audio/text tracks.
    pub fn with_preferred_language(mut self, language: impl Into<String>) -> Self {
        self.preferred_language = Some(language.into());
        self
    }

    /// Set maximum video bitrate.
    pub fn with_max_video_bitrate(mut self, bitrate: u64) -> Self {
        self.max_video_bitrate = Some(bitrate);
        self
    }

    /// Set minimum video resolution.
    pub fn with_min_video_width(mut self, width: u32) -> Self {
        self.min_video_width = Some(width);
        self
    }

    /// Get selected tracks for a type.
    pub fn get_selected(&self, track_type: TrackType) -> Option<&Vec<u32>> {
        self.selections.get(&track_type)
    }

    /// Check if a track matches selection criteria.
    pub fn matches_track(&self, track: &CmafTrackConfig) -> bool {
        // Check explicit selection
        if let Some(selected) = self.selections.get(&track.track_type) {
            if !selected.contains(&track.id) {
                return false;
            }
        }

        // Check language preference
        if let Some(ref preferred) = self.preferred_language {
            if track.track_type == TrackType::Audio || track.track_type == TrackType::Text {
                if let Some(ref lang) = track.language {
                    if lang != preferred {
                        return false;
                    }
                }
            }
        }

        // Check video constraints
        if track.track_type == TrackType::Video {
            if let Some(min_width) = self.min_video_width {
                if let Some(width) = track.width {
                    if width < min_width {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl Default for TrackSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Track switch point for seamless switching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackSwitchPoint {
    /// Segment sequence where switch can occur.
    pub segment_sequence: u64,
    /// Time position in seconds.
    pub time_position: f64,
    /// From track ID.
    pub from_track_id: u32,
    /// To track ID.
    pub to_track_id: u32,
    /// Whether this is a seamless switch point.
    pub seamless: bool,
}

// ============================================================================
// CMAF Writer
// ============================================================================

/// CMAF writer for generating streaming output.
pub struct CmafWriter {
    /// Configuration.
    config: CmafConfig,
    /// Initialization segments by quality.
    init_segments: HashMap<String, CmafInitSegment>,
    /// Media segments by quality.
    media_segments: HashMap<String, Vec<CmafMediaSegment>>,
    /// Segment counters by quality.
    segment_counters: HashMap<String, u64>,
    /// Current time position by quality.
    current_times: HashMap<String, f64>,
    /// Track selector.
    track_selector: TrackSelector,
    /// Byte-range file handles (for byte-range mode).
    byte_range_files: HashMap<String, (File, u64)>,
    /// Total bytes written.
    total_bytes: u64,
}

impl CmafWriter {
    /// Create a new CMAF writer.
    pub fn new(config: CmafConfig) -> Result<Self> {
        config.validate()?;

        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Create subdirectories for each quality
        for quality in &config.qualities {
            let quality_dir = config.output_dir.join(&quality.name);
            fs::create_dir_all(&quality_dir)?;
        }

        Ok(Self {
            config,
            init_segments: HashMap::new(),
            media_segments: HashMap::new(),
            segment_counters: HashMap::new(),
            current_times: HashMap::new(),
            track_selector: TrackSelector::new(),
            byte_range_files: HashMap::new(),
            total_bytes: 0,
        })
    }

    /// Get configuration.
    pub fn config(&self) -> &CmafConfig {
        &self.config
    }

    /// Set track selector.
    pub fn with_track_selector(mut self, selector: TrackSelector) -> Self {
        self.track_selector = selector;
        self
    }

    /// Write initialization segment for a quality.
    #[allow(clippy::too_many_arguments)]
    pub fn write_init_segment(
        &mut self,
        quality_name: &str,
        track_id: u32,
        codec: &str,
        width: Option<u32>,
        height: Option<u32>,
        sample_rate: Option<u32>,
        channels: Option<u32>,
    ) -> Result<CmafInitSegment> {
        let mut init_segment = if let (Some(w), Some(h)) = (width, height) {
            CmafInitSegment::video(
                track_id,
                quality_name,
                codec,
                w,
                h,
                self.config.timescale,
            )
        } else {
            CmafInitSegment::audio(
                track_id,
                quality_name,
                codec,
                sample_rate.unwrap_or(48000),
                channels.unwrap_or(2),
                sample_rate.unwrap_or(48000),
            )
        };

        // Generate fMP4 data
        let data = init_segment.generate_fmp4();

        // Determine output path
        let filename = "init.mp4";
        let relative_path = format!("{}/{}", quality_name, filename);
        let full_path = self.config.output_dir.join(&relative_path);

        // Write to file
        let mut file = File::create(&full_path)?;
        file.write_all(&data)?;

        init_segment.path = relative_path;

        // Store init segment
        self.init_segments
            .insert(quality_name.to_string(), init_segment.clone());

        // Initialize counters for this quality
        self.segment_counters.insert(quality_name.to_string(), 0);
        self.current_times.insert(quality_name.to_string(), 0.0);
        self.media_segments
            .insert(quality_name.to_string(), Vec::new());

        self.total_bytes += data.len() as u64;

        Ok(init_segment)
    }

    /// Write a media segment.
    pub fn write_media_segment(
        &mut self,
        quality_name: &str,
        samples: &[SampleInfo],
        duration: f64,
    ) -> Result<CmafMediaSegment> {
        let sequence = self
            .segment_counters
            .get(quality_name)
            .copied()
            .unwrap_or(0);
        let start_time = self.current_times.get(quality_name).copied().unwrap_or(0.0);

        // Get track ID from init segment
        let track_id = self
            .init_segments
            .get(quality_name)
            .map(|s| s.track_id)
            .unwrap_or(1);

        let mut segment = CmafMediaSegment::new(
            sequence,
            track_id,
            quality_name,
            duration,
            start_time,
            self.config.timescale,
        );

        // Check if first sample is a sync sample
        if let Some(first_sample) = samples.first() {
            segment = segment.with_sap(first_sample.is_sync, if first_sample.is_sync { 1 } else { 0 });
        }

        // Generate fMP4 data
        let data = segment.generate_fmp4(samples);

        // Write segment
        if self.config.byte_range_mode {
            // Byte-range mode: append to single file
            let (byte_range, written) = self.write_to_byte_range_file(quality_name, &data)?;
            segment.byte_range = Some(byte_range);
            segment.path = format!("{}/media.mp4", quality_name);
            self.total_bytes += written;
        } else {
            // Separate files mode
            let filename = self.config.naming.generate(sequence, "m4s");
            let relative_path = format!("{}/{}", quality_name, filename);
            let full_path = self.config.output_dir.join(&relative_path);

            let mut file = File::create(&full_path)?;
            file.write_all(&data)?;

            segment.path = relative_path;
            self.total_bytes += data.len() as u64;
        }

        // Generate chunks for low-latency mode
        if self.config.low_latency {
            let chunks = self.generate_chunks(&segment, samples);
            for chunk in chunks {
                segment.add_chunk(chunk);
            }
        }

        // Update counters
        *self
            .segment_counters
            .entry(quality_name.to_string())
            .or_insert(0) += 1;
        *self
            .current_times
            .entry(quality_name.to_string())
            .or_insert(0.0) += duration;

        // Store segment
        self.media_segments
            .entry(quality_name.to_string())
            .or_default()
            .push(segment.clone());

        Ok(segment)
    }

    /// Write chunk for low-latency streaming.
    pub fn write_chunk(
        &mut self,
        quality_name: &str,
        chunk_index: u32,
        samples: &[SampleInfo],
        duration: f64,
    ) -> Result<CmafChunk> {
        let sequence = self
            .segment_counters
            .get(quality_name)
            .copied()
            .unwrap_or(0);
        let segment_start = self.current_times.get(quality_name).copied().unwrap_or(0.0);
        let chunk_start = segment_start + (chunk_index as f64 * self.config.chunk_duration);

        let mut chunk = CmafChunk::new(chunk_index, duration, chunk_start);

        // Check if chunk contains independent sample
        chunk.independent = samples.first().map(|s| s.is_sync).unwrap_or(false);

        // Generate chunk data (simplified - in production would be proper moof+mdat)
        let data: Vec<u8> = samples.iter().flat_map(|s| s.data.clone()).collect();
        chunk.size = data.len() as u64;

        if self.config.byte_range_mode {
            // Write to byte-range file
            let (byte_range, written) = self.write_to_byte_range_file(quality_name, &data)?;
            chunk.byte_range = Some(byte_range);
            self.total_bytes += written;
        } else {
            // Write to separate chunk file
            let filename = format!("chunk_{}_{}.m4s", sequence, chunk_index);
            let relative_path = format!("{}/{}", quality_name, filename);
            let full_path = self.config.output_dir.join(&relative_path);

            let mut file = File::create(&full_path)?;
            file.write_all(&data)?;

            chunk.path = Some(relative_path);
            self.total_bytes += data.len() as u64;
        }

        Ok(chunk)
    }

    /// Write to byte-range file.
    fn write_to_byte_range_file(
        &mut self,
        quality_name: &str,
        data: &[u8],
    ) -> Result<(ByteRange, u64)> {
        let entry = self.byte_range_files.entry(quality_name.to_string());

        let (file, offset) = match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let (ref mut file, ref mut offset) = e.get_mut();
                let start = *offset;
                file.write_all(data)?;
                *offset += data.len() as u64;
                (start, data.len() as u64)
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let path = self.config.output_dir.join(quality_name).join("media.mp4");
                let mut file = File::create(&path)?;
                file.write_all(data)?;
                let written = data.len() as u64;
                e.insert((file, written));
                (0, written)
            }
        };

        Ok((ByteRange::new(file, offset), offset))
    }

    /// Generate chunks for a segment.
    fn generate_chunks(&self, segment: &CmafMediaSegment, samples: &[SampleInfo]) -> Vec<CmafChunk> {
        let mut chunks = Vec::new();

        let chunk_duration = self.config.chunk_duration;
        let mut chunk_index = 0u32;
        let mut chunk_start = 0.0;
        let mut chunk_samples_size = 0u64;
        let mut chunk_independent = false;

        let timescale = self.config.timescale as f64;

        for sample in samples {
            let sample_duration = sample.duration as f64 / timescale;

            // Check if we should start a new chunk
            if chunk_start + sample_duration > (chunk_index + 1) as f64 * chunk_duration {
                // Finalize current chunk
                if chunk_samples_size > 0 {
                    let mut chunk = CmafChunk::new(
                        chunk_index,
                        chunk_start,
                        segment.start_time + (chunk_index as f64 * chunk_duration),
                    );
                    chunk.size = chunk_samples_size;
                    chunk.independent = chunk_independent;
                    chunks.push(chunk);
                }

                chunk_index += 1;
                chunk_start = chunk_index as f64 * chunk_duration;
                chunk_samples_size = 0;
                chunk_independent = false;
            }

            chunk_samples_size += sample.size as u64;
            if sample.is_sync {
                chunk_independent = true;
            }
        }

        // Finalize last chunk
        if chunk_samples_size > 0 {
            let mut chunk = CmafChunk::new(
                chunk_index,
                segment.duration - chunk_start,
                segment.start_time + chunk_start,
            );
            chunk.size = chunk_samples_size;
            chunk.independent = chunk_independent;
            chunks.push(chunk);
        }

        chunks
    }

    /// Get initialization segment for a quality.
    pub fn get_init_segment(&self, quality_name: &str) -> Option<&CmafInitSegment> {
        self.init_segments.get(quality_name)
    }

    /// Get media segments for a quality.
    pub fn get_media_segments(&self, quality_name: &str) -> Option<&Vec<CmafMediaSegment>> {
        self.media_segments.get(quality_name)
    }

    /// Get current segment count for a quality.
    pub fn segment_count(&self, quality_name: &str) -> u64 {
        self.segment_counters.get(quality_name).copied().unwrap_or(0)
    }

    /// Get current time for a quality.
    pub fn current_time(&self, quality_name: &str) -> f64 {
        self.current_times.get(quality_name).copied().unwrap_or(0.0)
    }

    /// Get total bytes written.
    pub fn total_bytes_written(&self) -> u64 {
        self.total_bytes
    }

    /// Get output directory.
    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }

    /// Generate HLS playlist for CMAF content.
    pub fn generate_hls_playlist(&self, quality_name: &str) -> Result<String> {
        let mut lines = vec![
            "#EXTM3U".to_string(),
            "#EXT-X-VERSION:7".to_string(),
            format!(
                "#EXT-X-TARGETDURATION:{}",
                self.config.segment_duration.ceil() as u32
            ),
            "#EXT-X-MEDIA-SEQUENCE:0".to_string(),
        ];

        // Add init segment map
        if let Some(init) = self.init_segments.get(quality_name) {
            lines.push(format!("#EXT-X-MAP:URI=\"{}\"", init.path));
        }

        // Add media segments
        if let Some(segments) = self.media_segments.get(quality_name) {
            for segment in segments {
                if let Some(ref byte_range) = segment.byte_range {
                    lines.push(format!(
                        "#EXT-X-BYTERANGE:{}",
                        byte_range.to_hls_byterange()
                    ));
                }
                lines.push(format!("#EXTINF:{:.6},", segment.duration));
                lines.push(segment.path.clone());

                // Add parts for low-latency
                if self.config.low_latency && !segment.chunks.is_empty() {
                    for chunk in &segment.chunks {
                        let mut part_line = format!(
                            "#EXT-X-PART:DURATION={:.6}",
                            chunk.duration
                        );
                        if let Some(ref path) = chunk.path {
                            part_line.push_str(&format!(",URI=\"{}\"", path));
                        }
                        if let Some(ref range) = chunk.byte_range {
                            part_line.push_str(&format!(
                                ",BYTERANGE=\"{}\"",
                                range.to_hls_byterange()
                            ));
                        }
                        if chunk.independent {
                            part_line.push_str(",INDEPENDENT=YES");
                        }
                        lines.push(part_line);
                    }
                }
            }
        }

        Ok(lines.join("\n"))
    }

    /// Generate DASH MPD segment list for CMAF content.
    pub fn generate_dash_segment_list(&self, quality_name: &str) -> Result<String> {
        let mut xml = String::new();

        xml.push_str("<SegmentList timescale=\"90000\">\n");

        // Initialization segment
        if let Some(init) = self.init_segments.get(quality_name) {
            xml.push_str(&format!(
                "  <Initialization sourceURL=\"{}\"/>\n",
                init.path
            ));
        }

        // Media segments
        if let Some(segments) = self.media_segments.get(quality_name) {
            for segment in segments {
                let duration_ticks = (segment.duration * 90000.0) as u64;
                if let Some(ref byte_range) = segment.byte_range {
                    xml.push_str(&format!(
                        "  <SegmentURL media=\"{}\" mediaRange=\"{}-{}\"/>\n",
                        segment.path,
                        byte_range.start,
                        byte_range.end() - 1
                    ));
                } else {
                    xml.push_str(&format!(
                        "  <SegmentURL media=\"{}\" duration=\"{}\"/>\n",
                        segment.path, duration_ticks
                    ));
                }
            }
        }

        xml.push_str("</SegmentList>\n");

        Ok(xml)
    }

    /// Finalize and flush all pending data.
    pub fn finalize(&mut self) -> Result<()> {
        // Flush byte-range files
        for (_, (mut file, _)) in self.byte_range_files.drain() {
            file.flush()?;
        }

        Ok(())
    }

    /// Convert segments to base Segment type for compatibility.
    pub fn to_segments(&self, quality_name: &str) -> Vec<Segment> {
        let mut result = Vec::new();

        // Add init segment
        if let Some(init) = self.init_segments.get(quality_name) {
            result.push(Segment::new(
                0,
                SegmentType::Init,
                0.0,
                0.0,
                &init.path,
                quality_name,
            ).with_size(init.size));
        }

        // Add media segments
        if let Some(segments) = self.media_segments.get(quality_name) {
            for seg in segments {
                result.push(Segment::new(
                    seg.sequence,
                    SegmentType::Media,
                    seg.duration,
                    seg.start_time,
                    &seg.path,
                    quality_name,
                ).with_size(seg.size).with_keyframe(seg.starts_with_sap));
            }
        }

        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cmaf_config_creation() {
        let config = CmafConfig::new("output")
            .with_segment_duration(4.0)
            .with_chunk_duration(0.5)
            .with_low_latency(true);

        assert_eq!(config.segment_duration, 4.0);
        assert_eq!(config.chunk_duration, 0.5);
        assert!(config.low_latency);
        assert_eq!(config.fragment_type, FragmentType::LowLatency);
    }

    #[test]
    fn test_cmaf_config_validation() {
        // Valid config
        let config = CmafConfig::new("output").with_quality(Quality::fhd_1080p());
        assert!(config.validate().is_ok());

        // Invalid: no qualities
        let mut config = CmafConfig::new("output");
        config.qualities.clear();
        assert!(config.validate().is_err());

        // Invalid: chunk duration >= segment duration
        let config = CmafConfig::new("output")
            .with_segment_duration(1.0)
            .with_chunk_duration(2.0)
            .with_low_latency(true);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_track_config_creation() {
        let video_track = CmafTrackConfig::video(1, "avc1.64001f", 1920, 1080);
        assert_eq!(video_track.track_type, TrackType::Video);
        assert_eq!(video_track.width, Some(1920));

        let audio_track = CmafTrackConfig::audio(2, "mp4a.40.2", 48000, 2)
            .with_language("en")
            .with_name("English Audio");
        assert_eq!(audio_track.track_type, TrackType::Audio);
        assert_eq!(audio_track.language, Some("en".to_string()));
    }

    #[test]
    fn test_encryption_config() {
        let encryption = CmafEncryption::new_cenc("00112233445566778899aabbccddeeff")
            .with_widevine()
            .with_playready();

        assert_eq!(encryption.scheme, EncryptionScheme::Cenc);
        assert_eq!(encryption.drm_systems.len(), 2);
    }

    #[test]
    fn test_init_segment_generation() {
        let mut init = CmafInitSegment::video(1, "1080p", "avc1.64001f", 1920, 1080, 90000);

        let data = init.generate_fmp4();

        // Check for ftyp box
        assert!(data.len() > 8);
        assert_eq!(&data[4..8], b"ftyp");

        // Check for moov box
        let moov_pos = data.windows(4).position(|w| w == b"moov");
        assert!(moov_pos.is_some());
    }

    #[test]
    fn test_media_segment_generation() {
        let mut segment = CmafMediaSegment::new(0, 1, "1080p", 6.0, 0.0, 90000);

        let samples = vec![
            SampleInfo::new(vec![0u8; 1000], 3003, true),
            SampleInfo::new(vec![0u8; 500], 3003, false),
        ];

        let data = segment.generate_fmp4(&samples);

        // Check for styp box
        assert!(data.len() > 8);
        assert_eq!(&data[4..8], b"styp");

        // Check for moof box
        let moof_pos = data.windows(4).position(|w| w == b"moof");
        assert!(moof_pos.is_some());

        // Check for mdat box
        let mdat_pos = data.windows(4).position(|w| w == b"mdat");
        assert!(mdat_pos.is_some());
    }

    #[test]
    fn test_byte_range() {
        let range = ByteRange::new(100, 500);

        assert_eq!(range.end(), 600);
        assert_eq!(range.to_range_header(), "bytes=100-599");
        assert_eq!(range.to_hls_byterange(), "500@100");
    }

    #[test]
    fn test_track_selector() {
        let mut selector = TrackSelector::new()
            .with_preferred_language("es")
            .with_max_video_bitrate(5_000_000);

        selector.select_track(TrackType::Video, 1);

        let video_track = CmafTrackConfig::video(1, "avc1.64001f", 1920, 1080);
        assert!(selector.matches_track(&video_track));

        let audio_track = CmafTrackConfig::audio(2, "mp4a.40.2", 48000, 2)
            .with_language("en");
        // Won't match because preferred language is "es"
        assert!(!selector.matches_track(&audio_track));

        let spanish_audio = CmafTrackConfig::audio(3, "mp4a.40.2", 48000, 2)
            .with_language("es");
        assert!(selector.matches_track(&spanish_audio));
    }

    #[test]
    fn test_cmaf_chunk() {
        let chunk = CmafChunk::new(0, 0.5, 0.0)
            .with_byte_range(100, 500)
            .with_independent(true);

        assert_eq!(chunk.index, 0);
        assert!(chunk.independent);
        assert!(chunk.byte_range.is_some());
    }

    #[test]
    fn test_cmaf_writer_creation() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p())
            .with_quality(Quality::hd_720p());

        let writer = CmafWriter::new(config);
        assert!(writer.is_ok());
    }

    #[test]
    fn test_cmaf_writer_init_segment() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p());

        let mut writer = CmafWriter::new(config).unwrap();

        let init = writer.write_init_segment(
            "1080p",
            1,
            "avc1.64001f",
            Some(1920),
            Some(1080),
            None,
            None,
        );

        assert!(init.is_ok());
        let init = init.unwrap();
        assert!(init.size > 0);
        assert!(dir.path().join("1080p/init.mp4").exists());
    }

    #[test]
    fn test_cmaf_writer_media_segment() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p());

        let mut writer = CmafWriter::new(config).unwrap();

        // Write init segment first
        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        // Write media segment
        let samples = vec![
            SampleInfo::new(vec![0u8; 1000], 3003, true),
            SampleInfo::new(vec![0u8; 500], 3003, false),
        ];

        let segment = writer.write_media_segment("1080p", &samples, 0.0667);
        assert!(segment.is_ok());

        let segment = segment.unwrap();
        assert_eq!(segment.sequence, 0);
        assert!(segment.size > 0);
    }

    #[test]
    fn test_cmaf_writer_byte_range_mode() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p())
            .with_byte_range_mode(true);

        let mut writer = CmafWriter::new(config).unwrap();

        // Write init segment
        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        // Write multiple media segments
        for i in 0..3 {
            let samples = vec![SampleInfo::new(vec![i as u8; 1000], 3003, true)];
            let segment = writer.write_media_segment("1080p", &samples, 0.0334).unwrap();
            assert!(segment.byte_range.is_some());
        }
    }

    #[test]
    fn test_cmaf_writer_low_latency() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p())
            .with_segment_duration(2.0)
            .with_chunk_duration(0.5)
            .with_low_latency(true);

        let mut writer = CmafWriter::new(config).unwrap();

        // Write init segment
        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        // Write segment with multiple samples
        let mut samples = Vec::new();
        for i in 0..60 {
            // 2 seconds at 30fps
            samples.push(SampleInfo::new(vec![i as u8; 1000], 3003, i % 30 != 0));
        }

        let segment = writer.write_media_segment("1080p", &samples, 2.0).unwrap();

        // Should have chunks
        assert!(!segment.chunks.is_empty());
    }

    #[test]
    fn test_hls_playlist_generation() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p());

        let mut writer = CmafWriter::new(config).unwrap();

        // Write init and media segments
        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        let samples = vec![SampleInfo::new(vec![0u8; 1000], 90000 * 6, true)];
        writer.write_media_segment("1080p", &samples, 6.0).unwrap();

        // Generate playlist
        let playlist = writer.generate_hls_playlist("1080p").unwrap();

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-VERSION:7"));
        assert!(playlist.contains("#EXT-X-MAP:URI="));
        assert!(playlist.contains("#EXTINF:"));
    }

    #[test]
    fn test_dash_segment_list_generation() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p());

        let mut writer = CmafWriter::new(config).unwrap();

        // Write init and media segments
        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        let samples = vec![SampleInfo::new(vec![0u8; 1000], 90000 * 6, true)];
        writer.write_media_segment("1080p", &samples, 6.0).unwrap();

        // Generate segment list
        let segment_list = writer.generate_dash_segment_list("1080p").unwrap();

        assert!(segment_list.contains("<SegmentList"));
        assert!(segment_list.contains("<Initialization"));
        assert!(segment_list.contains("<SegmentURL"));
    }

    #[test]
    fn test_drm_system_creation() {
        let widevine = DrmSystem::widevine()
            .with_license_url("https://license.example.com/widevine");
        assert!(widevine.system_id.contains("edef8ba9"));
        assert!(widevine.license_url.is_some());

        let playready = DrmSystem::playready();
        assert!(playready.system_id.contains("9a04f079"));

        let fairplay = DrmSystem::fairplay();
        assert!(fairplay.system_id.contains("94ce86fb"));
    }

    #[test]
    fn test_output_format_variants() {
        let config_hls = CmafConfig::new("output")
            .with_output_format(CmafOutputFormat::HlsOnly);
        assert_eq!(config_hls.output_format, CmafOutputFormat::HlsOnly);

        let config_dash = CmafConfig::new("output")
            .with_output_format(CmafOutputFormat::DashOnly);
        assert_eq!(config_dash.output_format, CmafOutputFormat::DashOnly);

        let config_both = CmafConfig::new("output")
            .with_output_format(CmafOutputFormat::HlsAndDash);
        assert_eq!(config_both.output_format, CmafOutputFormat::HlsAndDash);
    }

    #[test]
    fn test_to_segments_conversion() {
        let dir = tempdir().unwrap();
        let config = CmafConfig::new(dir.path())
            .with_quality(Quality::fhd_1080p());

        let mut writer = CmafWriter::new(config).unwrap();

        writer
            .write_init_segment("1080p", 1, "avc1.64001f", Some(1920), Some(1080), None, None)
            .unwrap();

        let samples = vec![SampleInfo::new(vec![0u8; 1000], 90000, true)];
        writer.write_media_segment("1080p", &samples, 1.0).unwrap();

        let segments = writer.to_segments("1080p");
        assert_eq!(segments.len(), 2); // init + media
        assert_eq!(segments[0].segment_type, SegmentType::Init);
        assert_eq!(segments[1].segment_type, SegmentType::Media);
    }
}
