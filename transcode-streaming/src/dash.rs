//! DASH (Dynamic Adaptive Streaming over HTTP) support.

use crate::error::StreamingError;
use crate::segment::{Quality, Segment, SegmentNaming, SegmentType};
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Audio track configuration for DASH.
#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Track ID.
    pub id: String,
    /// Language code (BCP-47, e.g., "en", "es", "fr").
    pub language: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Audio codec string (e.g., "mp4a.40.2" for AAC-LC).
    pub codec: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Bitrate in bits per second.
    pub bitrate: u64,
    /// Whether this is the default track.
    pub default: bool,
    /// Audio channel configuration (e.g., "2" for stereo, "6" for 5.1).
    pub channel_configuration: Option<String>,
}

impl AudioTrackConfig {
    /// Create a new audio track configuration.
    pub fn new(id: impl Into<String>, language: impl Into<String>, codec: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            language: language.into(),
            label: None,
            codec: codec.into(),
            sample_rate: 48000,
            channels: 2,
            bitrate: 128_000,
            default: false,
            channel_configuration: None,
        }
    }

    /// Create AAC-LC stereo track.
    pub fn aac_stereo(id: impl Into<String>, language: impl Into<String>) -> Self {
        Self::new(id, language, "mp4a.40.2")
            .with_channels(2)
            .with_bitrate(128_000)
    }

    /// Create AAC 5.1 surround track.
    pub fn aac_surround(id: impl Into<String>, language: impl Into<String>) -> Self {
        Self::new(id, language, "mp4a.40.2")
            .with_channels(6)
            .with_bitrate(384_000)
            .with_channel_configuration("6")
    }

    /// Create Dolby Digital (AC-3) track.
    pub fn ac3(id: impl Into<String>, language: impl Into<String>) -> Self {
        Self::new(id, language, "ac-3")
            .with_channels(6)
            .with_bitrate(384_000)
    }

    /// Create E-AC-3 (Dolby Digital Plus) track.
    pub fn eac3(id: impl Into<String>, language: impl Into<String>) -> Self {
        Self::new(id, language, "ec-3")
            .with_channels(6)
            .with_bitrate(640_000)
    }

    /// Set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set sample rate.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set channel count.
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = channels;
        self
    }

    /// Set bitrate.
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Mark as default track.
    pub fn as_default(mut self) -> Self {
        self.default = true;
        self
    }

    /// Set channel configuration.
    pub fn with_channel_configuration(mut self, config: impl Into<String>) -> Self {
        self.channel_configuration = Some(config.into());
        self
    }

    /// Get channel configuration string.
    pub fn get_channel_configuration(&self) -> String {
        self.channel_configuration
            .clone()
            .unwrap_or_else(|| self.channels.to_string())
    }
}

/// DASH configuration.
#[derive(Debug, Clone)]
pub struct DashConfig {
    /// Output directory.
    pub output_dir: PathBuf,
    /// Segment duration in seconds.
    pub segment_duration: f64,
    /// Quality levels.
    pub qualities: Vec<Quality>,
    /// Audio tracks.
    pub audio_tracks: Vec<AudioTrackConfig>,
    /// Segment naming strategy.
    pub naming: SegmentNaming,
    /// MPD profile.
    pub profile: DashProfile,
    /// Enable low-latency DASH.
    pub low_latency: bool,
    /// Minimum buffer time in seconds.
    pub min_buffer_time: f64,
    /// Suggested presentation delay (for live).
    pub suggested_presentation_delay: Option<f64>,
    /// Time shift buffer depth (for live).
    pub time_shift_buffer_depth: Option<f64>,
}

/// DASH profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashProfile {
    /// ISO Base Media File Format On Demand.
    IsoOnDemand,
    /// ISO Base Media File Format Live.
    IsoLive,
    /// DASH-IF Simple Live.
    SimpleLive,
}

impl DashProfile {
    /// Get profile URN.
    pub fn urn(&self) -> &str {
        match self {
            Self::IsoOnDemand => "urn:mpeg:dash:profile:isoff-on-demand:2011",
            Self::IsoLive => "urn:mpeg:dash:profile:isoff-live:2011",
            Self::SimpleLive => "urn:mpeg:dash:profile:isoff-live:2011",
        }
    }
}

impl DashConfig {
    /// Create new DASH configuration.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            segment_duration: 6.0,
            qualities: vec![Quality::fhd_1080p()],
            audio_tracks: Vec::new(),
            naming: SegmentNaming::Sequential,
            profile: DashProfile::IsoOnDemand,
            low_latency: false,
            min_buffer_time: 2.0,
            suggested_presentation_delay: None,
            time_shift_buffer_depth: None,
        }
    }

    /// Set segment duration.
    pub fn with_segment_duration(mut self, duration: f64) -> Self {
        self.segment_duration = duration.max(1.0);
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

    /// Add an audio track.
    pub fn with_audio_track(mut self, track: AudioTrackConfig) -> Self {
        self.audio_tracks.push(track);
        self
    }

    /// Set audio tracks (replaces existing).
    pub fn with_audio_tracks(mut self, tracks: Vec<AudioTrackConfig>) -> Self {
        self.audio_tracks = tracks;
        self
    }

    /// Set profile.
    pub fn with_profile(mut self, profile: DashProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Enable low-latency DASH.
    pub fn with_low_latency(mut self) -> Self {
        self.low_latency = true;
        self.profile = DashProfile::SimpleLive;
        self
    }

    /// Set for live streaming.
    pub fn for_live(mut self) -> Self {
        self.profile = DashProfile::IsoLive;
        self.suggested_presentation_delay = Some(10.0);
        self.time_shift_buffer_depth = Some(60.0);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.qualities.is_empty() {
            return Err(StreamingError::InvalidConfig(
                "At least one quality level is required".into(),
            ));
        }

        if self.segment_duration < 1.0 {
            return Err(StreamingError::InvalidConfig(
                "Segment duration must be at least 1 second".into(),
            ));
        }

        Ok(())
    }
}

/// MPD (Media Presentation Description) manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpdManifest {
    /// MPD type.
    pub mpd_type: MpdType,
    /// Total media duration.
    pub media_duration: Option<f64>,
    /// Minimum buffer time.
    pub min_buffer_time: f64,
    /// Profiles.
    pub profiles: String,
    /// Availability start time (for live).
    pub availability_start_time: Option<DateTime<Utc>>,
    /// Suggested presentation delay.
    pub suggested_presentation_delay: Option<f64>,
    /// Time shift buffer depth.
    pub time_shift_buffer_depth: Option<f64>,
    /// Periods.
    pub periods: Vec<Period>,
}

/// MPD type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MpdType {
    /// Static (VOD).
    Static,
    /// Dynamic (Live).
    Dynamic,
}

/// Period in MPD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Period {
    /// Period ID.
    pub id: String,
    /// Period start time.
    pub start: Option<f64>,
    /// Period duration.
    pub duration: Option<f64>,
    /// Adaptation sets.
    pub adaptation_sets: Vec<AdaptationSet>,
}

/// Adaptation set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationSet {
    /// Adaptation set ID.
    pub id: u32,
    /// MIME type.
    pub mime_type: String,
    /// Codec.
    pub codecs: String,
    /// Content type.
    pub content_type: ContentType,
    /// Segment alignment.
    pub segment_alignment: bool,
    /// Language code (BCP-47).
    pub lang: Option<String>,
    /// Human-readable label.
    pub label: Option<String>,
    /// Representations.
    pub representations: Vec<Representation>,
    /// Segment template.
    pub segment_template: Option<SegmentTemplate>,
    /// Audio channel configuration (e.g., "2" for stereo, "6" for 5.1).
    pub audio_channel_configuration: Option<String>,
    /// Content protection elements for DRM.
    pub content_protections: Vec<ContentProtection>,
}

/// Content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Video,
    Audio,
    Text,
}

/// Representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Representation {
    /// Representation ID.
    pub id: String,
    /// Bandwidth.
    pub bandwidth: u64,
    /// Width (for video).
    pub width: Option<u32>,
    /// Height (for video).
    pub height: Option<u32>,
    /// Frame rate (for video).
    pub frame_rate: Option<String>,
    /// Sample rate (for audio).
    pub sample_rate: Option<u32>,
    /// Codec string.
    pub codecs: String,
}

/// Segment template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentTemplate {
    /// Timescale.
    pub timescale: u32,
    /// Duration in timescale units.
    pub duration: u32,
    /// Start number.
    pub start_number: u32,
    /// Initialization template.
    pub initialization: String,
    /// Media template.
    pub media: String,
}

/// Content protection element for DRM signaling in DASH MPD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentProtection {
    /// Scheme ID URI (identifies the DRM system).
    pub scheme_id_uri: String,
    /// Value attribute (e.g., encryption scheme like "cenc" or "cbcs").
    pub value: Option<String>,
    /// Default Key ID (UUID format without hyphens).
    pub default_kid: Option<String>,
    /// PSSH box data (base64 encoded).
    pub pssh: Option<String>,
    /// License acquisition URL.
    pub la_url: Option<String>,
}

impl ContentProtection {
    /// Create common encryption signaling (required for all DRM).
    pub fn cenc(default_kid: impl Into<String>) -> Self {
        Self {
            scheme_id_uri: "urn:mpeg:dash:mp4protection:2011".to_string(),
            value: Some("cenc".to_string()),
            default_kid: Some(default_kid.into()),
            pssh: None,
            la_url: None,
        }
    }

    /// Create CBCS encryption signaling (for FairPlay/HLS).
    pub fn cbcs(default_kid: impl Into<String>) -> Self {
        Self {
            scheme_id_uri: "urn:mpeg:dash:mp4protection:2011".to_string(),
            value: Some("cbcs".to_string()),
            default_kid: Some(default_kid.into()),
            pssh: None,
            la_url: None,
        }
    }

    /// Create Widevine DRM signaling.
    pub fn widevine() -> Self {
        Self {
            scheme_id_uri: "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed".to_string(),
            value: None,
            default_kid: None,
            pssh: None,
            la_url: None,
        }
    }

    /// Create PlayReady DRM signaling.
    pub fn playready() -> Self {
        Self {
            scheme_id_uri: "urn:uuid:9a04f079-9840-4286-ab92-e65be0885f95".to_string(),
            value: None,
            default_kid: None,
            pssh: None,
            la_url: None,
        }
    }

    /// Create FairPlay DRM signaling.
    pub fn fairplay() -> Self {
        Self {
            scheme_id_uri: "urn:uuid:94ce86fb-07ff-4f43-adb8-93d2fa968ca2".to_string(),
            value: None,
            default_kid: None,
            pssh: None,
            la_url: None,
        }
    }

    /// Create ClearKey DRM signaling.
    pub fn clearkey() -> Self {
        Self {
            scheme_id_uri: "urn:uuid:e2719d58-a985-b3c9-781a-b030af78d30e".to_string(),
            value: None,
            default_kid: None,
            pssh: None,
            la_url: None,
        }
    }

    /// Set PSSH data (base64 encoded).
    pub fn with_pssh(mut self, pssh: impl Into<String>) -> Self {
        self.pssh = Some(pssh.into());
        self
    }

    /// Set license acquisition URL.
    pub fn with_la_url(mut self, url: impl Into<String>) -> Self {
        self.la_url = Some(url.into());
        self
    }

    /// Set default Key ID.
    pub fn with_default_kid(mut self, kid: impl Into<String>) -> Self {
        self.default_kid = Some(kid.into());
        self
    }

    /// Render ContentProtection element to XML.
    pub fn render(&self) -> String {
        let mut xml = format!("      <ContentProtection schemeIdUri=\"{}\"", self.scheme_id_uri);

        if let Some(ref value) = self.value {
            xml.push_str(&format!(" value=\"{}\"", value));
        }

        if let Some(ref kid) = self.default_kid {
            xml.push_str(&format!(" cenc:default_KID=\"{}\"", kid));
        }

        // Check if we have child elements
        let has_children = self.pssh.is_some() || self.la_url.is_some();

        if has_children {
            xml.push_str(">\n");

            if let Some(ref pssh) = self.pssh {
                xml.push_str(&format!("        <cenc:pssh>{}</cenc:pssh>\n", pssh));
            }

            if let Some(ref la_url) = self.la_url {
                xml.push_str(&format!("        <ms:laurl licenseUrl=\"{}\"/>\n", la_url));
            }

            xml.push_str("      </ContentProtection>\n");
        } else {
            xml.push_str("/>\n");
        }

        xml
    }
}

/// DRM configuration for DASH streaming.
#[derive(Debug, Clone, Default)]
pub struct DashDrmConfig {
    /// Key ID in UUID format.
    pub key_id: Option<String>,
    /// Encryption scheme (cenc, cbcs).
    pub scheme: Option<String>,
    /// Content protection elements.
    pub content_protections: Vec<ContentProtection>,
}

impl DashDrmConfig {
    /// Create new DRM configuration.
    pub fn new(key_id: impl Into<String>) -> Self {
        Self {
            key_id: Some(key_id.into()),
            scheme: Some("cenc".to_string()),
            content_protections: Vec::new(),
        }
    }

    /// Set encryption scheme.
    pub fn with_scheme(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = Some(scheme.into());
        self
    }

    /// Add Widevine DRM.
    pub fn with_widevine(mut self, pssh: Option<String>, la_url: Option<String>) -> Self {
        let mut cp = ContentProtection::widevine();
        if let Some(pssh_data) = pssh {
            cp = cp.with_pssh(pssh_data);
        }
        if let Some(url) = la_url {
            cp = cp.with_la_url(url);
        }
        self.content_protections.push(cp);
        self
    }

    /// Add PlayReady DRM.
    pub fn with_playready(mut self, pssh: Option<String>, la_url: Option<String>) -> Self {
        let mut cp = ContentProtection::playready();
        if let Some(pssh_data) = pssh {
            cp = cp.with_pssh(pssh_data);
        }
        if let Some(url) = la_url {
            cp = cp.with_la_url(url);
        }
        self.content_protections.push(cp);
        self
    }

    /// Add FairPlay DRM.
    pub fn with_fairplay(mut self, la_url: Option<String>) -> Self {
        let mut cp = ContentProtection::fairplay();
        if let Some(url) = la_url {
            cp = cp.with_la_url(url);
        }
        self.content_protections.push(cp);
        self
    }

    /// Add ClearKey DRM.
    pub fn with_clearkey(mut self, la_url: Option<String>) -> Self {
        let mut cp = ContentProtection::clearkey();
        if let Some(url) = la_url {
            cp = cp.with_la_url(url);
        }
        self.content_protections.push(cp);
        self
    }

    /// Add custom content protection.
    pub fn with_content_protection(mut self, cp: ContentProtection) -> Self {
        self.content_protections.push(cp);
        self
    }

    /// Get all content protection elements including the base scheme.
    pub fn get_content_protections(&self) -> Vec<ContentProtection> {
        let mut result = Vec::new();

        // Add base CENC/CBCS signaling first
        if let Some(ref kid) = self.key_id {
            let scheme = self.scheme.as_deref().unwrap_or("cenc");
            if scheme == "cbcs" {
                result.push(ContentProtection::cbcs(kid));
            } else {
                result.push(ContentProtection::cenc(kid));
            }
        }

        // Add DRM-specific signaling
        result.extend(self.content_protections.clone());

        result
    }
}

impl MpdManifest {
    /// Create a new MPD manifest.
    pub fn new(mpd_type: MpdType) -> Self {
        Self {
            mpd_type,
            media_duration: None,
            min_buffer_time: 2.0,
            profiles: "urn:mpeg:dash:profile:isoff-on-demand:2011".to_string(),
            availability_start_time: None,
            suggested_presentation_delay: None,
            time_shift_buffer_depth: None,
            periods: Vec::new(),
        }
    }

    /// Add a period.
    pub fn add_period(&mut self, period: Period) {
        self.periods.push(period);
    }

    /// Generate XML content.
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();

        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
        xml.push_str("     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        xml.push_str("     xsi:schemaLocation=\"urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd\"\n");

        let type_str = match self.mpd_type {
            MpdType::Static => "static",
            MpdType::Dynamic => "dynamic",
        };
        xml.push_str(&format!("     type=\"{}\"\n", type_str));
        xml.push_str(&format!("     profiles=\"{}\"\n", self.profiles));
        xml.push_str(&format!("     minBufferTime=\"PT{:.1}S\"\n", self.min_buffer_time));

        if let Some(duration) = self.media_duration {
            xml.push_str(&format!("     mediaPresentationDuration=\"PT{:.3}S\"\n", duration));
        }

        if let Some(ref time) = self.availability_start_time {
            xml.push_str(&format!("     availabilityStartTime=\"{}\"\n", time.to_rfc3339()));
        }

        if let Some(delay) = self.suggested_presentation_delay {
            xml.push_str(&format!("     suggestedPresentationDelay=\"PT{:.1}S\"\n", delay));
        }

        if let Some(depth) = self.time_shift_buffer_depth {
            xml.push_str(&format!("     timeShiftBufferDepth=\"PT{:.1}S\"\n", depth));
        }

        xml.push_str(">\n");

        // Periods
        for period in &self.periods {
            xml.push_str(&format!("  <Period id=\"{}\"", period.id));
            if let Some(start) = period.start {
                xml.push_str(&format!(" start=\"PT{:.3}S\"", start));
            }
            if let Some(duration) = period.duration {
                xml.push_str(&format!(" duration=\"PT{:.3}S\"", duration));
            }
            xml.push_str(">\n");

            // Adaptation sets
            for adaptation in &period.adaptation_sets {
                xml.push_str(&format!("    <AdaptationSet id=\"{}\"", adaptation.id));
                xml.push_str(&format!(" mimeType=\"{}\"", adaptation.mime_type));
                xml.push_str(&format!(" codecs=\"{}\"", adaptation.codecs));
                let content_type = match adaptation.content_type {
                    ContentType::Video => "video",
                    ContentType::Audio => "audio",
                    ContentType::Text => "text",
                };
                xml.push_str(&format!(" contentType=\"{}\"", content_type));
                xml.push_str(&format!(" segmentAlignment=\"{}\"", adaptation.segment_alignment));
                if let Some(ref lang) = adaptation.lang {
                    xml.push_str(&format!(" lang=\"{}\"", lang));
                }
                xml.push_str(">\n");

                // Content protection elements (DRM)
                for cp in &adaptation.content_protections {
                    xml.push_str(&cp.render());
                }

                // Label element (child element, not attribute)
                if let Some(ref label) = adaptation.label {
                    xml.push_str(&format!("      <Label>{}</Label>\n", label));
                }

                // Audio channel configuration
                if let Some(ref channel_config) = adaptation.audio_channel_configuration {
                    xml.push_str(&format!(
                        "      <AudioChannelConfiguration schemeIdUri=\"urn:mpeg:dash:23003:3:audio_channel_configuration:2011\" value=\"{}\"/>\n",
                        channel_config
                    ));
                }

                // Segment template
                if let Some(ref template) = adaptation.segment_template {
                    xml.push_str(&format!(
                        "      <SegmentTemplate timescale=\"{}\" duration=\"{}\" startNumber=\"{}\"\n",
                        template.timescale, template.duration, template.start_number
                    ));
                    xml.push_str(&format!("                        initialization=\"{}\" media=\"{}\"/>\n",
                        template.initialization, template.media
                    ));
                }

                // Representations
                for rep in &adaptation.representations {
                    xml.push_str(&format!("      <Representation id=\"{}\"", rep.id));
                    xml.push_str(&format!(" bandwidth=\"{}\"", rep.bandwidth));
                    xml.push_str(&format!(" codecs=\"{}\"", rep.codecs));
                    if let Some(width) = rep.width {
                        xml.push_str(&format!(" width=\"{}\"", width));
                    }
                    if let Some(height) = rep.height {
                        xml.push_str(&format!(" height=\"{}\"", height));
                    }
                    if let Some(ref fr) = rep.frame_rate {
                        xml.push_str(&format!(" frameRate=\"{}\"", fr));
                    }
                    if let Some(sr) = rep.sample_rate {
                        xml.push_str(&format!(" audioSamplingRate=\"{}\"", sr));
                    }
                    xml.push_str("/>\n");
                }

                xml.push_str("    </AdaptationSet>\n");
            }

            xml.push_str("  </Period>\n");
        }

        xml.push_str("</MPD>\n");

        Ok(xml)
    }
}

/// DASH writer for generating streaming output.
pub struct DashWriter {
    /// Configuration.
    config: DashConfig,
    /// MPD manifest.
    manifest: MpdManifest,
    /// Segments for each quality.
    segments: Vec<Vec<Segment>>,
    /// Segment counter for each quality.
    segment_counters: Vec<u64>,
    /// Current time position.
    current_time: f64,
}

impl DashWriter {
    /// Create a new DASH writer.
    pub fn new(config: DashConfig) -> Result<Self> {
        config.validate()?;

        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Create subdirectories for each quality
        for quality in &config.qualities {
            let quality_dir = config.output_dir.join(&quality.name);
            fs::create_dir_all(&quality_dir)?;
        }

        let num_qualities = config.qualities.len();

        // Initialize manifest
        let mpd_type = match config.profile {
            DashProfile::IsoOnDemand => MpdType::Static,
            DashProfile::IsoLive | DashProfile::SimpleLive => MpdType::Dynamic,
        };

        let mut manifest = MpdManifest::new(mpd_type);
        manifest.profiles = config.profile.urn().to_string();
        manifest.min_buffer_time = config.min_buffer_time;
        manifest.suggested_presentation_delay = config.suggested_presentation_delay;
        manifest.time_shift_buffer_depth = config.time_shift_buffer_depth;

        if mpd_type == MpdType::Dynamic {
            manifest.availability_start_time = Some(Utc::now());
        }

        Ok(Self {
            config,
            manifest,
            segments: vec![Vec::new(); num_qualities],
            segment_counters: vec![0; num_qualities],
            current_time: 0.0,
        })
    }

    /// Write a segment for a specific quality.
    pub fn write_segment(
        &mut self,
        quality_index: usize,
        data: &[u8],
        duration: f64,
        keyframe: bool,
    ) -> Result<Segment> {
        if quality_index >= self.config.qualities.len() {
            return Err(StreamingError::InvalidQuality(format!(
                "Quality index {} out of range",
                quality_index
            )));
        }

        let quality = &self.config.qualities[quality_index];
        let sequence = self.segment_counters[quality_index];

        // Generate segment filename
        let filename = self.config.naming.generate(sequence, "m4s");
        let relative_path = format!("{}/{}", quality.name, filename);
        let full_path = self.config.output_dir.join(&relative_path);

        // Write segment data
        let mut file = File::create(&full_path)?;
        file.write_all(data)?;

        // Create segment record
        let segment = Segment::new(
            sequence,
            SegmentType::Media,
            duration,
            self.current_time,
            filename,
            &quality.name,
        )
        .with_size(data.len() as u64)
        .with_keyframe(keyframe);

        // Store segment
        self.segments[quality_index].push(segment.clone());

        // Update counters
        self.segment_counters[quality_index] += 1;
        self.current_time += duration;

        Ok(segment)
    }

    /// Write initialization segment.
    pub fn write_init_segment(&mut self, quality_index: usize, data: &[u8]) -> Result<()> {
        if quality_index >= self.config.qualities.len() {
            return Err(StreamingError::InvalidQuality(format!(
                "Quality index {} out of range",
                quality_index
            )));
        }

        let quality = &self.config.qualities[quality_index];
        let relative_path = format!("{}/init.mp4", quality.name);
        let full_path = self.config.output_dir.join(&relative_path);

        let mut file = File::create(&full_path)?;
        file.write_all(data)?;

        Ok(())
    }

    /// Build and write the MPD manifest.
    pub fn write_manifest(&mut self) -> Result<()> {
        // Build period with adaptation sets
        let mut period = Period {
            id: "0".to_string(),
            start: Some(0.0),
            duration: if self.manifest.mpd_type == MpdType::Static {
                Some(self.current_time)
            } else {
                None
            },
            adaptation_sets: Vec::new(),
        };

        // Video adaptation set
        let timescale = 1000;
        let segment_duration = (self.config.segment_duration * timescale as f64) as u32;

        let mut video_adaptation = AdaptationSet {
            id: 0,
            mime_type: "video/mp4".to_string(),
            codecs: self.config.qualities[0].video_codec.clone(),
            content_type: ContentType::Video,
            segment_alignment: true,
            lang: None,
            label: None,
            representations: Vec::new(),
            segment_template: Some(SegmentTemplate {
                timescale,
                duration: segment_duration,
                start_number: 0,
                initialization: "$RepresentationID$/init.mp4".to_string(),
                media: "$RepresentationID$/segment_$Number%05d$.m4s".to_string(),
            }),
            audio_channel_configuration: None,
            content_protections: Vec::new(),
        };

        for quality in &self.config.qualities {
            video_adaptation.representations.push(Representation {
                id: quality.name.clone(),
                bandwidth: quality.bandwidth(),
                width: Some(quality.width),
                height: Some(quality.height),
                frame_rate: Some(format!("{}/1", quality.framerate as u32)),
                sample_rate: None,
                codecs: quality.video_codec.clone(),
            });
        }

        period.adaptation_sets.push(video_adaptation);

        // Audio adaptation sets
        for (index, audio_track) in self.config.audio_tracks.iter().enumerate() {
            let audio_adaptation = AdaptationSet {
                id: (index + 1) as u32, // Start after video (id=0)
                mime_type: "audio/mp4".to_string(),
                codecs: audio_track.codec.clone(),
                content_type: ContentType::Audio,
                segment_alignment: true,
                lang: Some(audio_track.language.clone()),
                label: audio_track.label.clone(),
                representations: vec![Representation {
                    id: audio_track.id.clone(),
                    bandwidth: audio_track.bitrate,
                    width: None,
                    height: None,
                    frame_rate: None,
                    sample_rate: Some(audio_track.sample_rate),
                    codecs: audio_track.codec.clone(),
                }],
                segment_template: Some(SegmentTemplate {
                    timescale,
                    duration: segment_duration,
                    start_number: 0,
                    initialization: format!("{}/init.mp4", audio_track.id),
                    media: format!("{}/segment_$Number%05d$.m4s", audio_track.id),
                }),
                audio_channel_configuration: Some(audio_track.get_channel_configuration()),
                content_protections: Vec::new(),
            };
            period.adaptation_sets.push(audio_adaptation);
        }

        // Set media duration for static MPD
        if self.manifest.mpd_type == MpdType::Static {
            self.manifest.media_duration = Some(self.current_time);
        }

        self.manifest.periods.clear();
        self.manifest.add_period(period);

        // Write manifest
        let path = self.config.output_dir.join("manifest.mpd");
        let mut file = File::create(&path)?;
        file.write_all(self.manifest.to_xml()?.as_bytes())?;

        Ok(())
    }

    /// Finalize the DASH output.
    pub fn finalize(&mut self) -> Result<()> {
        self.write_manifest()
    }

    /// Get output directory.
    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }

    /// Get total duration.
    pub fn total_duration(&self) -> f64 {
        self.current_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpd_generation() {
        let mut manifest = MpdManifest::new(MpdType::Static);
        manifest.media_duration = Some(120.0);

        let mut period = Period {
            id: "0".to_string(),
            start: Some(0.0),
            duration: Some(120.0),
            adaptation_sets: Vec::new(),
        };

        period.adaptation_sets.push(AdaptationSet {
            id: 0,
            mime_type: "video/mp4".to_string(),
            codecs: "avc1.64001f".to_string(),
            content_type: ContentType::Video,
            segment_alignment: true,
            lang: None,
            label: None,
            representations: vec![Representation {
                id: "1080p".to_string(),
                bandwidth: 5_000_000,
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some("30/1".to_string()),
                sample_rate: None,
                codecs: "avc1.64001f".to_string(),
            }],
            segment_template: None,
            audio_channel_configuration: None,
            content_protections: Vec::new(),
        });

        manifest.add_period(period);

        let xml = manifest.to_xml().unwrap();
        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("type=\"static\""));
        assert!(xml.contains("bandwidth=\"5000000\""));
    }
}
