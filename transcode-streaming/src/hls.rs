//! HLS (HTTP Live Streaming) support with Low-Latency HLS (LL-HLS) extensions.

use crate::error::StreamingError;
use crate::segment::{Quality, Segment, SegmentNaming, SegmentType};
use crate::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// HLS configuration.
#[derive(Debug, Clone)]
pub struct HlsConfig {
    /// Output directory.
    pub output_dir: PathBuf,
    /// Segment duration in seconds.
    pub segment_duration: f64,
    /// Quality levels.
    pub qualities: Vec<Quality>,
    /// Segment naming strategy.
    pub naming: SegmentNaming,
    /// Enable LL-HLS (Low-Latency HLS).
    pub low_latency: bool,
    /// Part duration for LL-HLS (in seconds).
    pub part_duration: f64,
    /// Maximum playlist entries (0 for unlimited).
    pub max_playlist_entries: usize,
    /// Enable EXT-X-PROGRAM-DATE-TIME tags.
    pub program_date_time: bool,
    /// Enable independent segments.
    pub independent_segments: bool,
    /// Custom header lines for playlists.
    pub custom_headers: Vec<String>,
    /// LL-HLS configuration.
    pub ll_hls_config: LowLatencyConfig,
}

/// Low-Latency HLS specific configuration.
#[derive(Debug, Clone)]
pub struct LowLatencyConfig {
    /// Part hold-back in seconds (required for LL-HLS).
    /// Should be at least 3 times the part duration.
    pub part_hold_back: f64,
    /// Hold-back for non-LL-HLS clients in seconds.
    /// Should be at least 3 times the target duration.
    pub hold_back: f64,
    /// Enable blocking playlist reload.
    pub can_block_reload: bool,
    /// Enable skipping old segments in delta updates.
    pub can_skip_until: Option<f64>,
    /// Enable skipping date ranges in delta updates.
    pub can_skip_dateranges: bool,
    /// Maximum number of parts to retain after a segment is complete.
    pub max_parts_after_segment: usize,
    /// Enable rendition reports in playlists.
    pub rendition_reports: bool,
}

impl Default for LowLatencyConfig {
    fn default() -> Self {
        Self {
            part_hold_back: 0.6,  // 3x 0.2s default part duration
            hold_back: 18.0,      // 3x 6s default segment duration
            can_block_reload: true,
            can_skip_until: Some(36.0), // 6x target duration
            can_skip_dateranges: false,
            max_parts_after_segment: 3,
            rendition_reports: true,
        }
    }
}

impl HlsConfig {
    /// Create new HLS configuration.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            segment_duration: 6.0,
            qualities: vec![Quality::fhd_1080p()],
            naming: SegmentNaming::Sequential,
            low_latency: false,
            part_duration: 0.2,
            max_playlist_entries: 0,
            program_date_time: false,
            independent_segments: true,
            custom_headers: Vec::new(),
            ll_hls_config: LowLatencyConfig::default(),
        }
    }

    /// Set segment duration.
    pub fn with_segment_duration(mut self, duration: f64) -> Self {
        self.segment_duration = duration.max(1.0);
        // Update hold-back based on new segment duration
        self.ll_hls_config.hold_back = duration * 3.0;
        self.ll_hls_config.can_skip_until = Some(duration * 6.0);
        self
    }

    /// Add a quality level.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.qualities.push(quality);
        self
    }

    /// Set quality levels (replaces existing).
    pub fn with_qualities(mut self, qualities: Vec<Quality>) -> Self {
        self.qualities = qualities;
        self
    }

    /// Enable low-latency HLS.
    pub fn with_low_latency(mut self, part_duration: f64) -> Self {
        self.low_latency = true;
        self.part_duration = part_duration;
        // Set part-hold-back to 3x part duration as per spec
        self.ll_hls_config.part_hold_back = part_duration * 3.0;
        self
    }

    /// Configure LL-HLS settings.
    pub fn with_ll_hls_config(mut self, config: LowLatencyConfig) -> Self {
        self.ll_hls_config = config;
        self
    }

    /// Set segment naming strategy.
    pub fn with_naming(mut self, naming: SegmentNaming) -> Self {
        self.naming = naming;
        self
    }

    /// Set maximum playlist entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_playlist_entries = max;
        self
    }

    /// Enable program date time tags.
    pub fn with_program_date_time(mut self) -> Self {
        self.program_date_time = true;
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

        if self.low_latency {
            if self.part_duration <= 0.0 {
                return Err(StreamingError::InvalidConfig(
                    "Part duration must be positive for LL-HLS".into(),
                ));
            }

            if self.ll_hls_config.part_hold_back < self.part_duration * 2.0 {
                return Err(StreamingError::InvalidConfig(
                    "Part hold-back should be at least 2x part duration".into(),
                ));
            }
        }

        Ok(())
    }
}

/// Master playlist representation.
#[derive(Debug, Clone)]
pub struct MasterPlaylist {
    /// Quality variants.
    pub variants: Vec<VariantStream>,
    /// Audio groups.
    pub audio_groups: Vec<AudioGroup>,
    /// Whether to use independent segments.
    pub independent_segments: bool,
    /// LL-HLS session data.
    pub session_data: Vec<SessionData>,
}

/// Session data for master playlist.
#[derive(Debug, Clone)]
pub struct SessionData {
    /// Data ID.
    pub data_id: String,
    /// Data value.
    pub value: Option<String>,
    /// Data URI.
    pub uri: Option<String>,
    /// Language.
    pub language: Option<String>,
}

/// Variant stream in master playlist.
#[derive(Debug, Clone)]
pub struct VariantStream {
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Average bandwidth.
    pub average_bandwidth: Option<u64>,
    /// Resolution string.
    pub resolution: String,
    /// Frame rate.
    pub frame_rate: f32,
    /// Codec string.
    pub codecs: String,
    /// Playlist URI.
    pub uri: String,
    /// Audio group ID.
    pub audio_group: Option<String>,
}

/// Audio group for alternate audio tracks.
#[derive(Debug, Clone)]
pub struct AudioGroup {
    /// Group ID.
    pub id: String,
    /// Audio tracks.
    pub tracks: Vec<AudioTrack>,
}

/// Audio track in an audio group.
#[derive(Debug, Clone)]
pub struct AudioTrack {
    /// Track name.
    pub name: String,
    /// Language code.
    pub language: Option<String>,
    /// Whether this is the default track.
    pub default: bool,
    /// Whether this is auto-select.
    pub autoselect: bool,
    /// Playlist URI.
    pub uri: String,
}

impl MasterPlaylist {
    /// Create a new master playlist.
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
            audio_groups: Vec::new(),
            independent_segments: true,
            session_data: Vec::new(),
        }
    }

    /// Add a variant stream.
    pub fn add_variant(&mut self, variant: VariantStream) {
        self.variants.push(variant);
    }

    /// Generate playlist content as M3U8 format.
    pub fn render(&self) -> String {
        let mut lines = vec![
            "#EXTM3U".to_string(),
            "#EXT-X-VERSION:7".to_string(),
        ];

        if self.independent_segments {
            lines.push("#EXT-X-INDEPENDENT-SEGMENTS".to_string());
        }

        // Session data
        for data in &self.session_data {
            let mut attrs = vec![format!("DATA-ID=\"{}\"", data.data_id)];
            if let Some(ref value) = data.value {
                attrs.push(format!("VALUE=\"{}\"", value));
            }
            if let Some(ref uri) = data.uri {
                attrs.push(format!("URI=\"{}\"", uri));
            }
            if let Some(ref lang) = data.language {
                attrs.push(format!("LANGUAGE=\"{}\"", lang));
            }
            lines.push(format!("#EXT-X-SESSION-DATA:{}", attrs.join(",")));
        }

        // Audio groups
        for group in &self.audio_groups {
            for track in &group.tracks {
                let mut attrs = vec![
                    "TYPE=AUDIO".to_string(),
                    format!("GROUP-ID=\"{}\"", group.id),
                    format!("NAME=\"{}\"", track.name),
                ];

                if let Some(ref lang) = track.language {
                    attrs.push(format!("LANGUAGE=\"{}\"", lang));
                }

                if track.default {
                    attrs.push("DEFAULT=YES".to_string());
                }

                if track.autoselect {
                    attrs.push("AUTOSELECT=YES".to_string());
                }

                attrs.push(format!("URI=\"{}\"", track.uri));

                lines.push(format!("#EXT-X-MEDIA:{}", attrs.join(",")));
            }
        }

        // Variant streams
        for variant in &self.variants {
            let mut attrs = vec![
                format!("BANDWIDTH={}", variant.bandwidth),
            ];

            if let Some(avg) = variant.average_bandwidth {
                attrs.push(format!("AVERAGE-BANDWIDTH={}", avg));
            }

            if !variant.resolution.is_empty() {
                attrs.push(format!("RESOLUTION={}", variant.resolution));
            }

            if variant.frame_rate > 0.0 {
                attrs.push(format!("FRAME-RATE={:.3}", variant.frame_rate));
            }

            if !variant.codecs.is_empty() {
                attrs.push(format!("CODECS=\"{}\"", variant.codecs));
            }

            if let Some(ref audio) = variant.audio_group {
                attrs.push(format!("AUDIO=\"{}\"", audio));
            }

            lines.push(format!("#EXT-X-STREAM-INF:{}", attrs.join(",")));
            lines.push(variant.uri.clone());
        }

        lines.join("\n")
    }
}

impl Default for MasterPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

/// Partial segment for LL-HLS (EXT-X-PART).
#[derive(Debug, Clone)]
pub struct PartialSegment {
    /// Part sequence number within the segment.
    pub part_index: u32,
    /// Parent segment sequence number.
    pub segment_sequence: u64,
    /// Duration in seconds.
    pub duration: f64,
    /// URI to the partial segment.
    pub uri: String,
    /// Whether this part contains an independent frame.
    pub independent: bool,
    /// Byte range (if using byte-range addressing).
    pub byte_range: Option<ByteRange>,
    /// Gap indicator.
    pub gap: bool,
}

/// Byte range for partial segments.
#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    /// Length in bytes.
    pub length: u64,
    /// Offset from beginning of resource (optional).
    pub offset: Option<u64>,
}

impl PartialSegment {
    /// Create a new partial segment.
    pub fn new(
        part_index: u32,
        segment_sequence: u64,
        duration: f64,
        uri: impl Into<String>,
    ) -> Self {
        Self {
            part_index,
            segment_sequence,
            duration,
            uri: uri.into(),
            independent: part_index == 0, // First part is typically independent
            byte_range: None,
            gap: false,
        }
    }

    /// Mark as independent (contains keyframe).
    pub fn with_independent(mut self, independent: bool) -> Self {
        self.independent = independent;
        self
    }

    /// Set byte range.
    pub fn with_byte_range(mut self, length: u64, offset: Option<u64>) -> Self {
        self.byte_range = Some(ByteRange { length, offset });
        self
    }

    /// Mark as gap.
    pub fn with_gap(mut self) -> Self {
        self.gap = true;
        self
    }

    /// Render as EXT-X-PART tag.
    pub fn render(&self) -> String {
        let mut attrs = vec![
            format!("DURATION={:.6}", self.duration),
            format!("URI=\"{}\"", self.uri),
        ];

        if self.independent {
            attrs.push("INDEPENDENT=YES".to_string());
        }

        if let Some(ref range) = self.byte_range {
            if let Some(offset) = range.offset {
                attrs.push(format!("BYTERANGE=\"{}@{}\"", range.length, offset));
            } else {
                attrs.push(format!("BYTERANGE=\"{}\"", range.length));
            }
        }

        if self.gap {
            attrs.push("GAP=YES".to_string());
        }

        format!("#EXT-X-PART:{}", attrs.join(","))
    }
}

/// Preload hint for upcoming content (EXT-X-PRELOAD-HINT).
#[derive(Debug, Clone)]
pub struct PreloadHint {
    /// Type of content being preloaded.
    pub hint_type: PreloadHintType,
    /// URI to the content.
    pub uri: String,
    /// Byte range start.
    pub byte_range_start: Option<u64>,
    /// Byte range length.
    pub byte_range_length: Option<u64>,
}

/// Type of preload hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreloadHintType {
    /// Partial segment hint.
    Part,
    /// Map (initialization segment) hint.
    Map,
}

impl PreloadHint {
    /// Create a new preload hint for a partial segment.
    pub fn part(uri: impl Into<String>) -> Self {
        Self {
            hint_type: PreloadHintType::Part,
            uri: uri.into(),
            byte_range_start: None,
            byte_range_length: None,
        }
    }

    /// Create a new preload hint for a map.
    pub fn map(uri: impl Into<String>) -> Self {
        Self {
            hint_type: PreloadHintType::Map,
            uri: uri.into(),
            byte_range_start: None,
            byte_range_length: None,
        }
    }

    /// Set byte range.
    pub fn with_byte_range(mut self, start: u64, length: Option<u64>) -> Self {
        self.byte_range_start = Some(start);
        self.byte_range_length = length;
        self
    }

    /// Render as EXT-X-PRELOAD-HINT tag.
    pub fn render(&self) -> String {
        let type_str = match self.hint_type {
            PreloadHintType::Part => "PART",
            PreloadHintType::Map => "MAP",
        };

        let mut attrs = vec![
            format!("TYPE={}", type_str),
            format!("URI=\"{}\"", self.uri),
        ];

        if let Some(start) = self.byte_range_start {
            attrs.push(format!("BYTERANGE-START={}", start));
        }

        if let Some(length) = self.byte_range_length {
            attrs.push(format!("BYTERANGE-LENGTH={}", length));
        }

        format!("#EXT-X-PRELOAD-HINT:{}", attrs.join(","))
    }
}

/// Server control settings (EXT-X-SERVER-CONTROL).
#[derive(Debug, Clone)]
pub struct ServerControl {
    /// Whether the server supports blocking playlist reload.
    pub can_block_reload: bool,
    /// Part hold-back in seconds.
    pub part_hold_back: Option<f64>,
    /// Hold-back in seconds for non-LL clients.
    pub hold_back: Option<f64>,
    /// Can skip segments up to this duration in seconds.
    pub can_skip_until: Option<f64>,
    /// Can skip date ranges.
    pub can_skip_dateranges: bool,
}

impl ServerControl {
    /// Create new server control with default settings.
    pub fn new() -> Self {
        Self {
            can_block_reload: false,
            part_hold_back: None,
            hold_back: None,
            can_skip_until: None,
            can_skip_dateranges: false,
        }
    }

    /// Enable blocking playlist reload.
    pub fn with_blocking_reload(mut self) -> Self {
        self.can_block_reload = true;
        self
    }

    /// Set part hold-back.
    pub fn with_part_hold_back(mut self, seconds: f64) -> Self {
        self.part_hold_back = Some(seconds);
        self
    }

    /// Set hold-back.
    pub fn with_hold_back(mut self, seconds: f64) -> Self {
        self.hold_back = Some(seconds);
        self
    }

    /// Enable skip until.
    pub fn with_can_skip_until(mut self, seconds: f64) -> Self {
        self.can_skip_until = Some(seconds);
        self
    }

    /// Enable skipping date ranges.
    pub fn with_can_skip_dateranges(mut self) -> Self {
        self.can_skip_dateranges = true;
        self
    }

    /// Render as EXT-X-SERVER-CONTROL tag.
    pub fn render(&self) -> String {
        let mut attrs = Vec::new();

        if self.can_block_reload {
            attrs.push("CAN-BLOCK-RELOAD=YES".to_string());
        }

        if let Some(phb) = self.part_hold_back {
            attrs.push(format!("PART-HOLD-BACK={:.6}", phb));
        }

        if let Some(hb) = self.hold_back {
            attrs.push(format!("HOLD-BACK={:.6}", hb));
        }

        if let Some(skip) = self.can_skip_until {
            attrs.push(format!("CAN-SKIP-UNTIL={:.6}", skip));
        }

        if self.can_skip_dateranges {
            attrs.push("CAN-SKIP-DATERANGES=YES".to_string());
        }

        if attrs.is_empty() {
            String::new()
        } else {
            format!("#EXT-X-SERVER-CONTROL:{}", attrs.join(","))
        }
    }
}

impl Default for ServerControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Skip tag for delta playlist updates (EXT-X-SKIP).
#[derive(Debug, Clone)]
pub struct SkipTag {
    /// Number of segments skipped.
    pub skipped_segments: u64,
    /// Recently removed date ranges (optional).
    pub recently_removed_dateranges: Vec<String>,
}

impl SkipTag {
    /// Create a new skip tag.
    pub fn new(skipped_segments: u64) -> Self {
        Self {
            skipped_segments,
            recently_removed_dateranges: Vec::new(),
        }
    }

    /// Add removed date range IDs.
    pub fn with_removed_dateranges(mut self, ids: Vec<String>) -> Self {
        self.recently_removed_dateranges = ids;
        self
    }

    /// Render as EXT-X-SKIP tag.
    pub fn render(&self) -> String {
        let mut attrs = vec![format!("SKIPPED-SEGMENTS={}", self.skipped_segments)];

        if !self.recently_removed_dateranges.is_empty() {
            let ids = self.recently_removed_dateranges.join("\t");
            attrs.push(format!("RECENTLY-REMOVED-DATERANGES=\"{}\"", ids));
        }

        format!("#EXT-X-SKIP:{}", attrs.join(","))
    }
}

/// Rendition report for faster ABR switching (EXT-X-RENDITION-REPORT).
#[derive(Debug, Clone)]
pub struct RenditionReport {
    /// URI to the rendition playlist.
    pub uri: String,
    /// Last media sequence number.
    pub last_msn: Option<u64>,
    /// Last part index.
    pub last_part: Option<u32>,
}

impl RenditionReport {
    /// Create a new rendition report.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            last_msn: None,
            last_part: None,
        }
    }

    /// Set last media sequence number.
    pub fn with_last_msn(mut self, msn: u64) -> Self {
        self.last_msn = Some(msn);
        self
    }

    /// Set last part index.
    pub fn with_last_part(mut self, part: u32) -> Self {
        self.last_part = Some(part);
        self
    }

    /// Render as EXT-X-RENDITION-REPORT tag.
    pub fn render(&self) -> String {
        let mut attrs = vec![format!("URI=\"{}\"", self.uri)];

        if let Some(msn) = self.last_msn {
            attrs.push(format!("LAST-MSN={}", msn));
        }

        if let Some(part) = self.last_part {
            attrs.push(format!("LAST-PART={}", part));
        }

        format!("#EXT-X-RENDITION-REPORT:{}", attrs.join(","))
    }
}

/// Media playlist representation with LL-HLS support.
#[derive(Debug, Clone)]
pub struct MediaPlaylist {
    /// Target duration.
    pub target_duration: u32,
    /// Media sequence number.
    pub media_sequence: u64,
    /// Discontinuity sequence.
    pub discontinuity_sequence: u64,
    /// Playlist type (VOD or EVENT).
    pub playlist_type: Option<PlaylistType>,
    /// Segments.
    pub segments: Vec<Segment>,
    /// Whether the playlist is ended.
    pub ended: bool,
    /// Part target duration (LL-HLS).
    pub part_target: Option<f64>,
    /// Partial segments grouped by parent segment sequence.
    pub partial_segments: Vec<PartialSegment>,
    /// Server control settings.
    pub server_control: Option<ServerControl>,
    /// Preload hints.
    pub preload_hints: Vec<PreloadHint>,
    /// Rendition reports.
    pub rendition_reports: Vec<RenditionReport>,
    /// Skip tag for delta updates.
    pub skip: Option<SkipTag>,
    /// Program date time for first segment.
    pub program_date_time: Option<DateTime<Utc>>,
}

/// Playlist type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistType {
    /// Video on demand - complete, immutable playlist.
    Vod,
    /// Event - may grow but never remove segments.
    Event,
}

impl MediaPlaylist {
    /// Create a new media playlist.
    pub fn new(target_duration: u32) -> Self {
        Self {
            target_duration,
            media_sequence: 0,
            discontinuity_sequence: 0,
            playlist_type: None,
            segments: Vec::new(),
            ended: false,
            part_target: None,
            partial_segments: Vec::new(),
            server_control: None,
            preload_hints: Vec::new(),
            rendition_reports: Vec::new(),
            skip: None,
            program_date_time: None,
        }
    }

    /// Create a new LL-HLS media playlist.
    pub fn new_ll_hls(target_duration: u32, part_target: f64) -> Self {
        let mut playlist = Self::new(target_duration);
        playlist.part_target = Some(part_target);
        playlist.server_control = Some(ServerControl::new()
            .with_blocking_reload()
            .with_part_hold_back(part_target * 3.0));
        playlist
    }

    /// Add a segment.
    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    /// Add a partial segment.
    pub fn add_partial_segment(&mut self, part: PartialSegment) {
        self.partial_segments.push(part);
    }

    /// Add a preload hint.
    pub fn add_preload_hint(&mut self, hint: PreloadHint) {
        self.preload_hints.push(hint);
    }

    /// Add a rendition report.
    pub fn add_rendition_report(&mut self, report: RenditionReport) {
        self.rendition_reports.push(report);
    }

    /// Set server control.
    pub fn with_server_control(mut self, control: ServerControl) -> Self {
        self.server_control = Some(control);
        self
    }

    /// Mark playlist as ended.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Set playlist type.
    pub fn with_type(mut self, playlist_type: PlaylistType) -> Self {
        self.playlist_type = Some(playlist_type);
        self
    }

    /// Set skip tag for delta updates.
    pub fn with_skip(mut self, skip: SkipTag) -> Self {
        self.skip = Some(skip);
        self
    }

    /// Get the last media sequence number.
    pub fn last_msn(&self) -> u64 {
        self.media_sequence + self.segments.len().saturating_sub(1) as u64
    }

    /// Get the last part index for the current segment.
    pub fn last_part(&self) -> Option<u32> {
        self.partial_segments
            .iter()
            .filter(|p| p.segment_sequence == self.last_msn())
            .map(|p| p.part_index)
            .max()
    }

    /// Generate playlist content as M3U8 format.
    pub fn render(&self) -> String {
        let mut lines = vec![
            "#EXTM3U".to_string(),
            "#EXT-X-VERSION:7".to_string(),
            format!("#EXT-X-TARGETDURATION:{}", self.target_duration),
            format!("#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence),
        ];

        // Part target duration for LL-HLS
        if let Some(part_target) = self.part_target {
            lines.push(format!("#EXT-X-PART-INF:PART-TARGET={:.6}", part_target));
        }

        // Server control
        if let Some(ref control) = self.server_control {
            let control_tag = control.render();
            if !control_tag.is_empty() {
                lines.push(control_tag);
            }
        }

        // Playlist type
        if let Some(playlist_type) = self.playlist_type {
            let type_str = match playlist_type {
                PlaylistType::Vod => "VOD",
                PlaylistType::Event => "EVENT",
            };
            lines.push(format!("#EXT-X-PLAYLIST-TYPE:{}", type_str));
        }

        // Skip tag for delta updates
        if let Some(ref skip) = self.skip {
            lines.push(skip.render());
        }

        // Program date time
        if let Some(ref pdt) = self.program_date_time {
            lines.push(format!("#EXT-X-PROGRAM-DATE-TIME:{}", pdt.to_rfc3339()));
        }

        // Group partial segments by parent segment sequence
        let mut parts_by_segment: HashMap<u64, Vec<&PartialSegment>> = HashMap::new();
        for part in &self.partial_segments {
            parts_by_segment.entry(part.segment_sequence).or_default().push(part);
        }

        // Sort parts within each segment
        for parts in parts_by_segment.values_mut() {
            parts.sort_by_key(|p| p.part_index);
        }

        // Render segments with their parts
        for segment in &self.segments {
            if segment.segment_type == SegmentType::Init {
                lines.push(format!("#EXT-X-MAP:URI=\"{}\"", segment.path));
            } else {
                // Render partial segments for this segment
                if let Some(parts) = parts_by_segment.get(&segment.sequence) {
                    for part in parts {
                        lines.push(part.render());
                    }
                }

                lines.push(format!("#EXTINF:{:.6},", segment.duration));
                lines.push(segment.path.clone());
            }
        }

        // Render partial segments for incomplete (current) segment
        let next_sequence = self.segments
            .iter()
            .filter(|s| s.segment_type != SegmentType::Init)
            .count() as u64 + self.media_sequence;

        if let Some(parts) = parts_by_segment.get(&next_sequence) {
            for part in parts {
                lines.push(part.render());
            }
        }

        // Preload hints
        for hint in &self.preload_hints {
            lines.push(hint.render());
        }

        // Rendition reports
        for report in &self.rendition_reports {
            lines.push(report.render());
        }

        if self.ended {
            lines.push("#EXT-X-ENDLIST".to_string());
        }

        lines.join("\n")
    }

    /// Render a delta playlist (with skipped segments).
    pub fn render_delta(&self, skip_segments: u64) -> String {
        let mut delta = self.clone();
        delta.skip = Some(SkipTag::new(skip_segments));

        // Remove skipped segments from the playlist
        let skip_count = skip_segments as usize;
        if skip_count < delta.segments.len() {
            delta.segments = delta.segments.split_off(skip_count);
            delta.media_sequence += skip_segments;
        }

        delta.render()
    }
}

/// Blocking playlist request parameters.
#[derive(Debug, Clone)]
pub struct BlockingPlaylistRequest {
    /// Requested media sequence number.
    pub msn: Option<u64>,
    /// Requested part index.
    pub part: Option<u32>,
    /// Skip directive.
    pub skip: Option<SkipDirective>,
}

/// Skip directive for delta playlist requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipDirective {
    /// Skip old segments (EXT-X-SKIP).
    Yes,
    /// Skip segments and date ranges.
    V2,
}

impl BlockingPlaylistRequest {
    /// Create a new blocking request.
    pub fn new() -> Self {
        Self {
            msn: None,
            part: None,
            skip: None,
        }
    }

    /// Request a specific media sequence number.
    pub fn with_msn(mut self, msn: u64) -> Self {
        self.msn = Some(msn);
        self
    }

    /// Request a specific part.
    pub fn with_part(mut self, part: u32) -> Self {
        self.part = Some(part);
        self
    }

    /// Request a delta playlist.
    pub fn with_skip(mut self, skip: SkipDirective) -> Self {
        self.skip = Some(skip);
        self
    }

    /// Parse from query string.
    pub fn from_query(query: &str) -> Self {
        let mut request = Self::new();

        for param in query.split('&') {
            let mut parts = param.split('=');
            if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                match key.to_lowercase().as_str() {
                    "_hlsmsn" => {
                        if let Ok(msn) = value.parse() {
                            request.msn = Some(msn);
                        }
                    }
                    "_hlspart" => {
                        if let Ok(part) = value.parse() {
                            request.part = Some(part);
                        }
                    }
                    "_hlsskip" => {
                        request.skip = match value.to_uppercase().as_str() {
                            "YES" => Some(SkipDirective::Yes),
                            "V2" => Some(SkipDirective::V2),
                            _ => None,
                        };
                    }
                    _ => {}
                }
            }
        }

        request
    }
}

impl Default for BlockingPlaylistRequest {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// I-Frame Only Playlist Support
// =============================================================================

/// I-Frame segment for I-frame only playlists.
///
/// I-frame segments contain only keyframes, enabling trick play modes
/// like fast-forward and rewind.
#[derive(Debug, Clone)]
pub struct IFrameSegment {
    /// Sequence number.
    pub sequence: u64,
    /// Duration in seconds.
    pub duration: f64,
    /// URI to the segment.
    pub uri: String,
    /// Byte range for the I-frame within the segment.
    pub byte_range: Option<ByteRange>,
    /// Byte offset from file start.
    pub byte_offset: u64,
}

impl IFrameSegment {
    /// Create a new I-frame segment.
    pub fn new(sequence: u64, duration: f64, uri: impl Into<String>) -> Self {
        Self {
            sequence,
            duration,
            uri: uri.into(),
            byte_range: None,
            byte_offset: 0,
        }
    }

    /// Set byte range.
    pub fn with_byte_range(mut self, length: u64, offset: u64) -> Self {
        self.byte_range = Some(ByteRange {
            length,
            offset: Some(offset),
        });
        self.byte_offset = offset;
        self
    }

    /// Render as EXT-X-BYTERANGE and EXTINF tags.
    pub fn render(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref range) = self.byte_range {
            if let Some(offset) = range.offset {
                lines.push(format!("#EXT-X-BYTERANGE:{}@{}", range.length, offset));
            } else {
                lines.push(format!("#EXT-X-BYTERANGE:{}", range.length));
            }
        }

        lines.push(format!("#EXTINF:{:.6},", self.duration));
        lines.push(self.uri.clone());

        lines.join("\n")
    }
}

/// I-Frame only playlist (EXT-X-I-FRAMES-ONLY).
///
/// Contains only I-frames (keyframes) for trick play modes.
/// Per HLS specification, this playlist uses EXT-X-I-FRAMES-ONLY tag.
#[derive(Debug, Clone)]
pub struct IFramePlaylist {
    /// Target duration in seconds.
    pub target_duration: u32,
    /// Media sequence number.
    pub media_sequence: u64,
    /// I-frame segments.
    pub segments: Vec<IFrameSegment>,
    /// Whether the playlist is ended.
    pub ended: bool,
    /// Init segment (EXT-X-MAP).
    pub init_segment: Option<String>,
    /// Codec string.
    pub codecs: Option<String>,
    /// Resolution string.
    pub resolution: Option<String>,
    /// Bandwidth.
    pub bandwidth: Option<u64>,
}

impl IFramePlaylist {
    /// Create a new I-frame playlist.
    pub fn new(target_duration: u32) -> Self {
        Self {
            target_duration,
            media_sequence: 0,
            segments: Vec::new(),
            ended: false,
            init_segment: None,
            codecs: None,
            resolution: None,
            bandwidth: None,
        }
    }

    /// Add an I-frame segment.
    pub fn add_segment(&mut self, segment: IFrameSegment) {
        self.segments.push(segment);
    }

    /// Set init segment.
    pub fn with_init_segment(mut self, uri: impl Into<String>) -> Self {
        self.init_segment = Some(uri.into());
        self
    }

    /// Set codec string.
    pub fn with_codecs(mut self, codecs: impl Into<String>) -> Self {
        self.codecs = Some(codecs.into());
        self
    }

    /// Set resolution.
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Set bandwidth.
    pub fn with_bandwidth(mut self, bandwidth: u64) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Mark playlist as ended.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Render as M3U8 format.
    pub fn render(&self) -> String {
        let mut lines = vec![
            "#EXTM3U".to_string(),
            "#EXT-X-VERSION:7".to_string(),
            format!("#EXT-X-TARGETDURATION:{}", self.target_duration),
            format!("#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence),
            "#EXT-X-I-FRAMES-ONLY".to_string(),
        ];

        // Init segment
        if let Some(ref init) = self.init_segment {
            lines.push(format!("#EXT-X-MAP:URI=\"{}\"", init));
        }

        // Segments
        for segment in &self.segments {
            lines.push(segment.render());
        }

        if self.ended {
            lines.push("#EXT-X-ENDLIST".to_string());
        }

        lines.join("\n")
    }

    /// Render as EXT-X-I-FRAME-STREAM-INF for master playlist.
    pub fn render_stream_inf(&self, uri: &str) -> String {
        let mut attrs = Vec::new();

        if let Some(bandwidth) = self.bandwidth {
            attrs.push(format!("BANDWIDTH={}", bandwidth));
        }

        if let Some(ref resolution) = self.resolution {
            attrs.push(format!("RESOLUTION={}", resolution));
        }

        if let Some(ref codecs) = self.codecs {
            attrs.push(format!("CODECS=\"{}\"", codecs));
        }

        attrs.push(format!("URI=\"{}\"", uri));

        format!("#EXT-X-I-FRAME-STREAM-INF:{}", attrs.join(","))
    }
}

// =============================================================================
// Subtitle Track Support
// =============================================================================

/// Subtitle group for alternate subtitle tracks.
#[derive(Debug, Clone)]
pub struct SubtitleGroup {
    /// Group ID.
    pub id: String,
    /// Subtitle tracks.
    pub tracks: Vec<SubtitleTrack>,
}

impl SubtitleGroup {
    /// Create a new subtitle group.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tracks: Vec::new(),
        }
    }

    /// Add a subtitle track.
    pub fn add_track(&mut self, track: SubtitleTrack) {
        self.tracks.push(track);
    }

    /// Render EXT-X-MEDIA tags for all tracks.
    pub fn render(&self) -> Vec<String> {
        self.tracks
            .iter()
            .map(|track| track.render(&self.id))
            .collect()
    }
}

/// Subtitle track in a subtitle group.
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// Track name (display name).
    pub name: String,
    /// Language code (ISO 639-1 or 639-2).
    pub language: Option<String>,
    /// Associated language (for accessibility tracks).
    pub assoc_language: Option<String>,
    /// Whether this is the default track.
    pub default: bool,
    /// Whether this is auto-select.
    pub autoselect: bool,
    /// Whether this is forced subtitles (e.g., translations of foreign dialogue).
    pub forced: bool,
    /// Playlist URI.
    pub uri: String,
    /// Characteristics (e.g., "public.accessibility.transcribes-spoken-dialog").
    pub characteristics: Vec<String>,
    /// Instream ID for closed captions (CC1-CC4, SERVICE1-SERVICE63).
    pub instream_id: Option<String>,
}

impl SubtitleTrack {
    /// Create a new subtitle track.
    pub fn new(name: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            language: None,
            assoc_language: None,
            default: false,
            autoselect: false,
            forced: false,
            uri: uri.into(),
            characteristics: Vec::new(),
            instream_id: None,
        }
    }

    /// Create a new closed caption track.
    pub fn closed_caption(name: impl Into<String>, instream_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            language: None,
            assoc_language: None,
            default: false,
            autoselect: false,
            forced: false,
            uri: String::new(),
            characteristics: Vec::new(),
            instream_id: Some(instream_id.into()),
        }
    }

    /// Set language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set associated language.
    pub fn with_assoc_language(mut self, language: impl Into<String>) -> Self {
        self.assoc_language = Some(language.into());
        self
    }

    /// Mark as default track.
    pub fn with_default(mut self) -> Self {
        self.default = true;
        self
    }

    /// Mark as autoselect.
    pub fn with_autoselect(mut self) -> Self {
        self.autoselect = true;
        self
    }

    /// Mark as forced subtitles.
    pub fn with_forced(mut self) -> Self {
        self.forced = true;
        self
    }

    /// Add accessibility characteristic.
    pub fn with_characteristic(mut self, characteristic: impl Into<String>) -> Self {
        self.characteristics.push(characteristic.into());
        self
    }

    /// Render as EXT-X-MEDIA tag.
    pub fn render(&self, group_id: &str) -> String {
        let media_type = if self.instream_id.is_some() {
            "CLOSED-CAPTIONS"
        } else {
            "SUBTITLES"
        };

        let mut attrs = vec![
            format!("TYPE={}", media_type),
            format!("GROUP-ID=\"{}\"", group_id),
            format!("NAME=\"{}\"", self.name),
        ];

        if let Some(ref lang) = self.language {
            attrs.push(format!("LANGUAGE=\"{}\"", lang));
        }

        if let Some(ref assoc) = self.assoc_language {
            attrs.push(format!("ASSOC-LANGUAGE=\"{}\"", assoc));
        }

        if self.default {
            attrs.push("DEFAULT=YES".to_string());
        }

        if self.autoselect {
            attrs.push("AUTOSELECT=YES".to_string());
        }

        if self.forced {
            attrs.push("FORCED=YES".to_string());
        }

        if !self.characteristics.is_empty() {
            attrs.push(format!(
                "CHARACTERISTICS=\"{}\"",
                self.characteristics.join(",")
            ));
        }

        if let Some(ref instream_id) = self.instream_id {
            attrs.push(format!("INSTREAM-ID=\"{}\"", instream_id));
        } else if !self.uri.is_empty() {
            attrs.push(format!("URI=\"{}\"", self.uri));
        }

        format!("#EXT-X-MEDIA:{}", attrs.join(","))
    }
}

// =============================================================================
// HLS DRM Key Support
// =============================================================================

/// HLS encryption method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsKeyMethod {
    /// No encryption.
    None,
    /// AES-128 encryption.
    Aes128,
    /// SAMPLE-AES encryption (for CBCS/FairPlay).
    SampleAes,
    /// SAMPLE-AES-CTR encryption.
    SampleAesCtr,
}

impl HlsKeyMethod {
    /// Get the method string for EXT-X-KEY tag.
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "NONE",
            Self::Aes128 => "AES-128",
            Self::SampleAes => "SAMPLE-AES",
            Self::SampleAesCtr => "SAMPLE-AES-CTR",
        }
    }
}

/// Key format for HLS DRM.
#[derive(Debug, Clone)]
pub enum HlsKeyFormat {
    /// Standard HLS key format.
    Identity,
    /// FairPlay Streaming key format.
    FairPlay,
    /// Widevine key format.
    Widevine,
    /// Custom key format.
    Custom(String),
}

impl HlsKeyFormat {
    /// Get the KEYFORMAT string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Identity => "identity",
            Self::FairPlay => "com.apple.streamingkeydelivery",
            Self::Widevine => "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::Custom(s) => s,
        }
    }

    /// Get the KEYFORMATVERSIONS string.
    pub fn versions(&self) -> Option<&str> {
        match self {
            Self::FairPlay => Some("1"),
            Self::Widevine => Some("1"),
            _ => None,
        }
    }
}

/// HLS EXT-X-KEY tag for content protection.
#[derive(Debug, Clone)]
pub struct HlsKey {
    /// Encryption method.
    pub method: HlsKeyMethod,
    /// Key URI.
    pub uri: Option<String>,
    /// Initialization vector (hex string, 32 characters for 16 bytes).
    pub iv: Option<String>,
    /// Key format.
    pub key_format: Option<HlsKeyFormat>,
    /// Key format versions.
    pub key_format_versions: Option<String>,
}

impl HlsKey {
    /// Create a key tag with no encryption.
    pub fn none() -> Self {
        Self {
            method: HlsKeyMethod::None,
            uri: None,
            iv: None,
            key_format: None,
            key_format_versions: None,
        }
    }

    /// Create an AES-128 key tag.
    pub fn aes128(uri: impl Into<String>) -> Self {
        Self {
            method: HlsKeyMethod::Aes128,
            uri: Some(uri.into()),
            iv: None,
            key_format: None,
            key_format_versions: None,
        }
    }

    /// Create a SAMPLE-AES key tag (for CBCS/FairPlay).
    pub fn sample_aes(uri: impl Into<String>) -> Self {
        Self {
            method: HlsKeyMethod::SampleAes,
            uri: Some(uri.into()),
            iv: None,
            key_format: None,
            key_format_versions: None,
        }
    }

    /// Create a FairPlay key tag.
    pub fn fairplay(uri: impl Into<String>) -> Self {
        Self {
            method: HlsKeyMethod::SampleAes,
            uri: Some(uri.into()),
            iv: None,
            key_format: Some(HlsKeyFormat::FairPlay),
            key_format_versions: Some("1".to_string()),
        }
    }

    /// Create a Widevine key tag for HLS.
    pub fn widevine(uri: impl Into<String>) -> Self {
        Self {
            method: HlsKeyMethod::SampleAesCtr,
            uri: Some(uri.into()),
            iv: None,
            key_format: Some(HlsKeyFormat::Widevine),
            key_format_versions: Some("1".to_string()),
        }
    }

    /// Set the initialization vector.
    pub fn with_iv(mut self, iv: impl Into<String>) -> Self {
        self.iv = Some(iv.into());
        self
    }

    /// Set the key format.
    pub fn with_key_format(mut self, format: HlsKeyFormat) -> Self {
        self.key_format = Some(format);
        self
    }

    /// Set key format versions.
    pub fn with_key_format_versions(mut self, versions: impl Into<String>) -> Self {
        self.key_format_versions = Some(versions.into());
        self
    }

    /// Render as EXT-X-KEY tag.
    pub fn render(&self) -> String {
        let mut attrs = vec![format!("METHOD={}", self.method.as_str())];

        if let Some(ref uri) = self.uri {
            attrs.push(format!("URI=\"{}\"", uri));
        }

        if let Some(ref iv) = self.iv {
            // IV must be prefixed with 0x
            let iv_str = if iv.starts_with("0x") || iv.starts_with("0X") {
                iv.clone()
            } else {
                format!("0x{}", iv)
            };
            attrs.push(format!("IV={}", iv_str));
        }

        if let Some(ref format) = self.key_format {
            attrs.push(format!("KEYFORMAT=\"{}\"", format.as_str()));

            // Add KEYFORMATVERSIONS if available
            if let Some(ref versions) = self.key_format_versions {
                attrs.push(format!("KEYFORMATVERSIONS=\"{}\"", versions));
            } else if let Some(versions) = format.versions() {
                attrs.push(format!("KEYFORMATVERSIONS=\"{}\"", versions));
            }
        }

        format!("#EXT-X-KEY:{}", attrs.join(","))
    }
}

/// HLS EXT-X-SESSION-KEY for master playlist.
#[derive(Debug, Clone)]
pub struct HlsSessionKey {
    /// Inner key definition.
    pub key: HlsKey,
}

impl HlsSessionKey {
    /// Create a new session key.
    pub fn new(key: HlsKey) -> Self {
        Self { key }
    }

    /// Create a FairPlay session key.
    pub fn fairplay(uri: impl Into<String>) -> Self {
        Self::new(HlsKey::fairplay(uri))
    }

    /// Create a Widevine session key.
    pub fn widevine(uri: impl Into<String>) -> Self {
        Self::new(HlsKey::widevine(uri))
    }

    /// Render as EXT-X-SESSION-KEY tag.
    pub fn render(&self) -> String {
        self.key.render().replace("#EXT-X-KEY:", "#EXT-X-SESSION-KEY:")
    }
}

/// HLS DRM configuration.
#[derive(Debug, Clone, Default)]
pub struct HlsDrmConfig {
    /// Session keys for the master playlist.
    pub session_keys: Vec<HlsSessionKey>,
    /// Media key for media playlists.
    pub media_key: Option<HlsKey>,
}

impl HlsDrmConfig {
    /// Create a new DRM configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a FairPlay configuration.
    pub fn with_fairplay(mut self, key_server_uri: impl Into<String>) -> Self {
        self.session_keys.push(HlsSessionKey::fairplay(key_server_uri.into()));
        self
    }

    /// Add a Widevine configuration.
    pub fn with_widevine(mut self, license_uri: impl Into<String>) -> Self {
        self.session_keys.push(HlsSessionKey::widevine(license_uri.into()));
        self
    }

    /// Set the media key.
    pub fn with_media_key(mut self, key: HlsKey) -> Self {
        self.media_key = Some(key);
        self
    }

    /// Add a session key.
    pub fn with_session_key(mut self, key: HlsKey) -> Self {
        self.session_keys.push(HlsSessionKey::new(key));
        self
    }

    /// Render all session keys for master playlist.
    pub fn render_session_keys(&self) -> Vec<String> {
        self.session_keys.iter().map(|k| k.render()).collect()
    }

    /// Render the media key for media playlist.
    pub fn render_media_key(&self) -> Option<String> {
        self.media_key.as_ref().map(|k| k.render())
    }
}

/// HLS writer for generating streaming output with LL-HLS support.
pub struct HlsWriter {
    /// Configuration.
    config: HlsConfig,
    /// Playlists for each quality.
    playlists: Vec<MediaPlaylist>,
    /// Master playlist.
    master: MasterPlaylist,
    /// Segment counter for each quality.
    segment_counters: Vec<u64>,
    /// Part counter for current segment per quality.
    part_counters: Vec<u32>,
    /// Current time position.
    current_time: f64,
    /// Start time for program date time.
    start_time: Option<DateTime<Utc>>,
}

impl HlsWriter {
    /// Create a new HLS writer.
    pub fn new(config: HlsConfig) -> Result<Self> {
        config.validate()?;

        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Create subdirectories for each quality
        for quality in &config.qualities {
            let quality_dir = config.output_dir.join(&quality.name);
            fs::create_dir_all(&quality_dir)?;
        }

        let num_qualities = config.qualities.len();

        // Initialize playlists
        let target_duration = config.segment_duration.ceil() as u32;
        let playlists: Vec<MediaPlaylist> = if config.low_latency {
            (0..num_qualities)
                .map(|_| {
                    let mut playlist = MediaPlaylist::new_ll_hls(
                        target_duration,
                        config.part_duration,
                    );

                    // Configure server control
                    let mut control = ServerControl::new();
                    if config.ll_hls_config.can_block_reload {
                        control = control.with_blocking_reload();
                    }
                    control = control.with_part_hold_back(config.ll_hls_config.part_hold_back);
                    control = control.with_hold_back(config.ll_hls_config.hold_back);
                    if let Some(skip_until) = config.ll_hls_config.can_skip_until {
                        control = control.with_can_skip_until(skip_until);
                    }
                    if config.ll_hls_config.can_skip_dateranges {
                        control = control.with_can_skip_dateranges();
                    }
                    playlist.server_control = Some(control);

                    if config.program_date_time {
                        playlist.program_date_time = Some(Utc::now());
                    }

                    playlist
                })
                .collect()
        } else {
            (0..num_qualities)
                .map(|_| MediaPlaylist::new(target_duration))
                .collect()
        };

        // Initialize master playlist
        let mut master = MasterPlaylist::new();
        master.independent_segments = config.independent_segments;

        for quality in &config.qualities {
            master.add_variant(VariantStream {
                bandwidth: quality.bandwidth(),
                average_bandwidth: Some(quality.bitrate),
                resolution: quality.resolution_string(),
                frame_rate: quality.framerate,
                codecs: quality.codecs_string(),
                uri: format!("{}/playlist.m3u8", quality.name),
                audio_group: None,
            });
        }

        let start_time = if config.program_date_time {
            Some(Utc::now())
        } else {
            None
        };

        Ok(Self {
            config,
            playlists,
            master,
            segment_counters: vec![0; num_qualities],
            part_counters: vec![0; num_qualities],
            current_time: 0.0,
            start_time,
        })
    }

    /// Write a partial segment for LL-HLS.
    pub fn write_partial_segment(
        &mut self,
        quality_index: usize,
        data: &[u8],
        duration: f64,
        independent: bool,
    ) -> Result<PartialSegment> {
        if !self.config.low_latency {
            return Err(StreamingError::InvalidConfig(
                "Partial segments require LL-HLS mode".into(),
            ));
        }

        if quality_index >= self.config.qualities.len() {
            return Err(StreamingError::InvalidQuality(format!(
                "Quality index {} out of range",
                quality_index
            )));
        }

        let quality = &self.config.qualities[quality_index];
        let segment_seq = self.segment_counters[quality_index];
        let part_index = self.part_counters[quality_index];

        // Generate part filename
        let filename = format!(
            "segment_{:05}.{}.part",
            segment_seq,
            part_index
        );
        let relative_path = format!("{}/{}", quality.name, filename);
        let full_path = self.config.output_dir.join(&relative_path);

        // Write part data
        let mut file = File::create(&full_path)?;
        file.write_all(data)?;

        // Create partial segment record
        let part = PartialSegment::new(
            part_index,
            segment_seq,
            duration,
            filename,
        ).with_independent(independent);

        // Add to playlist
        self.playlists[quality_index].add_partial_segment(part.clone());

        // Update part counter
        self.part_counters[quality_index] += 1;

        // Update preload hint for next part
        self.update_preload_hints(quality_index)?;

        // Update playlist file
        self.write_media_playlist(quality_index)?;

        Ok(part)
    }

    /// Update preload hints for the next expected part.
    fn update_preload_hints(&mut self, quality_index: usize) -> Result<()> {
        let segment_seq = self.segment_counters[quality_index];
        let next_part = self.part_counters[quality_index];

        let next_part_uri = format!(
            "segment_{:05}.{}.part",
            segment_seq,
            next_part
        );

        let playlist = &mut self.playlists[quality_index];
        playlist.preload_hints.clear();
        playlist.preload_hints.push(PreloadHint::part(next_part_uri));

        // Add rendition reports for other qualities if enabled
        if self.config.ll_hls_config.rendition_reports {
            self.update_rendition_reports(quality_index);
        }

        Ok(())
    }

    /// Update rendition reports for all qualities.
    fn update_rendition_reports(&mut self, current_quality: usize) {
        let mut reports: Vec<(usize, RenditionReport)> = Vec::new();

        for (i, quality) in self.config.qualities.iter().enumerate() {
            if i != current_quality {
                let playlist = &self.playlists[i];
                let report = RenditionReport::new(format!("../{}/playlist.m3u8", quality.name))
                    .with_last_msn(playlist.last_msn());

                if let Some(last_part) = playlist.last_part() {
                    reports.push((current_quality, report.with_last_part(last_part)));
                } else {
                    reports.push((current_quality, report));
                }
            }
        }

        for (quality_index, report) in reports {
            self.playlists[quality_index].rendition_reports.push(report);
        }
    }

    /// Write a complete segment for a specific quality.
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
        let filename = self.config.naming.generate(sequence, "ts");
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

        // Add to playlist
        self.playlists[quality_index].add_segment(segment.clone());

        // Reset part counter for new segment
        if self.config.low_latency {
            self.part_counters[quality_index] = 0;

            // Clean up old partial segments beyond retention limit
            self.cleanup_old_parts(quality_index);
        }

        // Update counters
        self.segment_counters[quality_index] += 1;
        self.current_time += duration;

        // Update playlist file
        self.write_media_playlist(quality_index)?;

        Ok(segment)
    }

    /// Clean up old partial segments.
    fn cleanup_old_parts(&mut self, quality_index: usize) {
        let max_keep = self.config.ll_hls_config.max_parts_after_segment;
        let current_seq = self.segment_counters[quality_index];

        if current_seq <= max_keep as u64 {
            return;
        }

        let min_keep_seq = current_seq - max_keep as u64;

        self.playlists[quality_index].partial_segments.retain(|part| {
            part.segment_sequence >= min_keep_seq
        });
    }

    /// Write initialization segment for fMP4.
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

        // Add init segment to playlist
        let segment = Segment::new(
            0,
            SegmentType::Init,
            0.0,
            0.0,
            "init.mp4",
            &quality.name,
        )
        .with_size(data.len() as u64);

        self.playlists[quality_index].add_segment(segment);

        Ok(())
    }

    /// Write media playlist for a quality.
    fn write_media_playlist(&self, quality_index: usize) -> Result<()> {
        let quality = &self.config.qualities[quality_index];
        let playlist = &self.playlists[quality_index];
        let path = self.config.output_dir.join(&quality.name).join("playlist.m3u8");

        let mut file = File::create(&path)?;
        file.write_all(playlist.render().as_bytes())?;

        Ok(())
    }

    /// Write master playlist.
    pub fn write_master_playlist(&self) -> Result<()> {
        let path = self.config.output_dir.join("master.m3u8");
        let mut file = File::create(&path)?;
        file.write_all(self.master.render().as_bytes())?;
        Ok(())
    }

    /// Generate a blocking playlist response.
    pub fn get_blocking_playlist(
        &self,
        quality_index: usize,
        request: &BlockingPlaylistRequest,
    ) -> Result<Option<String>> {
        if quality_index >= self.playlists.len() {
            return Err(StreamingError::InvalidQuality(format!(
                "Quality index {} out of range",
                quality_index
            )));
        }

        let playlist = &self.playlists[quality_index];

        // Check if we have the requested content
        if let Some(msn) = request.msn {
            let current_msn = playlist.last_msn();

            if msn > current_msn {
                // Not yet available - return None for blocking
                return Ok(None);
            }

            if let Some(part) = request.part {
                // Check if we have the requested part
                let has_part = playlist.partial_segments.iter()
                    .any(|p| p.segment_sequence == msn && p.part_index == part);

                if !has_part && msn == current_msn {
                    // Part not yet available - return None for blocking
                    return Ok(None);
                }
            }
        }

        // Generate playlist
        let content = match request.skip {
            Some(SkipDirective::Yes) | Some(SkipDirective::V2) => {
                // Calculate how many segments to skip
                if let Some(skip_until) = self.config.ll_hls_config.can_skip_until {
                    let segments_to_keep = (skip_until / self.config.segment_duration).ceil() as u64;
                    let total_segments = playlist.segments.iter()
                        .filter(|s| s.segment_type != SegmentType::Init)
                        .count() as u64;

                    if total_segments > segments_to_keep {
                        let skip_count = total_segments - segments_to_keep;
                        playlist.render_delta(skip_count)
                    } else {
                        playlist.render()
                    }
                } else {
                    playlist.render()
                }
            }
            None => playlist.render(),
        };

        Ok(Some(content))
    }

    /// Finalize all playlists.
    pub fn finalize(&mut self) -> Result<()> {
        // Clear preload hints and mark as ended
        for playlist in &mut self.playlists {
            playlist.preload_hints.clear();
            playlist.rendition_reports.clear();
            playlist.end();
        }

        // Write final media playlists
        for i in 0..self.config.qualities.len() {
            self.write_media_playlist(i)?;
        }

        // Write master playlist
        self.write_master_playlist()?;

        Ok(())
    }

    /// Get current segment count for a quality.
    pub fn segment_count(&self, quality_index: usize) -> u64 {
        self.segment_counters.get(quality_index).copied().unwrap_or(0)
    }

    /// Get current part count for a quality.
    pub fn part_count(&self, quality_index: usize) -> u32 {
        self.part_counters.get(quality_index).copied().unwrap_or(0)
    }

    /// Get total duration.
    pub fn total_duration(&self) -> f64 {
        self.current_time
    }

    /// Get output directory.
    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }

    /// Check if LL-HLS is enabled.
    pub fn is_low_latency(&self) -> bool {
        self.config.low_latency
    }

    /// Get the media playlist for a quality.
    pub fn get_playlist(&self, quality_index: usize) -> Option<&MediaPlaylist> {
        self.playlists.get(quality_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_playlist_generation() {
        let mut master = MasterPlaylist::new();
        master.add_variant(VariantStream {
            bandwidth: 5_500_000,
            average_bandwidth: Some(5_000_000),
            resolution: "1920x1080".to_string(),
            frame_rate: 30.0,
            codecs: "avc1.64001f,mp4a.40.2".to_string(),
            uri: "1080p/playlist.m3u8".to_string(),
            audio_group: None,
        });

        let content = master.render();
        assert!(content.contains("#EXTM3U"));
        assert!(content.contains("BANDWIDTH=5500000"));
        assert!(content.contains("RESOLUTION=1920x1080"));
    }

    #[test]
    fn test_media_playlist_generation() {
        let mut playlist = MediaPlaylist::new(6);
        playlist.add_segment(Segment::new(0, SegmentType::Media, 6.0, 0.0, "seg0.ts", "1080p"));
        playlist.add_segment(Segment::new(1, SegmentType::Media, 6.0, 6.0, "seg1.ts", "1080p"));

        let content = playlist.render();
        assert!(content.contains("#EXT-X-TARGETDURATION:6"));
        assert!(content.contains("#EXTINF:6.000000"));
    }

    #[test]
    fn test_ll_hls_playlist_generation() {
        let mut playlist = MediaPlaylist::new_ll_hls(6, 0.2);

        // Add partial segments
        playlist.add_partial_segment(PartialSegment::new(0, 0, 0.2, "segment_00000.0.part").with_independent(true));
        playlist.add_partial_segment(PartialSegment::new(1, 0, 0.2, "segment_00000.1.part"));
        playlist.add_partial_segment(PartialSegment::new(2, 0, 0.2, "segment_00000.2.part"));

        // Add preload hint
        playlist.add_preload_hint(PreloadHint::part("segment_00000.3.part"));

        let content = playlist.render();

        assert!(content.contains("#EXT-X-PART-INF:PART-TARGET=0.200000"));
        assert!(content.contains("#EXT-X-PART:DURATION=0.200000,URI=\"segment_00000.0.part\",INDEPENDENT=YES"));
        assert!(content.contains("#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"segment_00000.3.part\""));
    }

    #[test]
    fn test_server_control_rendering() {
        let control = ServerControl::new()
            .with_blocking_reload()
            .with_part_hold_back(0.6)
            .with_hold_back(18.0)
            .with_can_skip_until(36.0);

        let tag = control.render();

        assert!(tag.contains("CAN-BLOCK-RELOAD=YES"));
        assert!(tag.contains("PART-HOLD-BACK=0.600000"));
        assert!(tag.contains("HOLD-BACK=18.000000"));
        assert!(tag.contains("CAN-SKIP-UNTIL=36.000000"));
    }

    #[test]
    fn test_partial_segment_rendering() {
        let part = PartialSegment::new(0, 5, 0.2, "segment_00005.0.part")
            .with_independent(true);

        let tag = part.render();

        assert!(tag.contains("DURATION=0.200000"));
        assert!(tag.contains("URI=\"segment_00005.0.part\""));
        assert!(tag.contains("INDEPENDENT=YES"));
    }

    #[test]
    fn test_preload_hint_rendering() {
        let hint = PreloadHint::part("segment_00005.3.part");
        let tag = hint.render();

        assert!(tag.contains("TYPE=PART"));
        assert!(tag.contains("URI=\"segment_00005.3.part\""));

        let map_hint = PreloadHint::map("init.mp4");
        let map_tag = map_hint.render();

        assert!(map_tag.contains("TYPE=MAP"));
        assert!(map_tag.contains("URI=\"init.mp4\""));
    }

    #[test]
    fn test_skip_tag_rendering() {
        let skip = SkipTag::new(5);
        let tag = skip.render();

        assert!(tag.contains("SKIPPED-SEGMENTS=5"));

        let skip_with_ranges = SkipTag::new(3)
            .with_removed_dateranges(vec!["id1".to_string(), "id2".to_string()]);
        let tag2 = skip_with_ranges.render();

        assert!(tag2.contains("SKIPPED-SEGMENTS=3"));
        assert!(tag2.contains("RECENTLY-REMOVED-DATERANGES=\"id1\tid2\""));
    }

    #[test]
    fn test_rendition_report_rendering() {
        let report = RenditionReport::new("../720p/playlist.m3u8")
            .with_last_msn(10)
            .with_last_part(5);

        let tag = report.render();

        assert!(tag.contains("URI=\"../720p/playlist.m3u8\""));
        assert!(tag.contains("LAST-MSN=10"));
        assert!(tag.contains("LAST-PART=5"));
    }

    #[test]
    fn test_delta_playlist_rendering() {
        let mut playlist = MediaPlaylist::new(6);

        for i in 0..10 {
            playlist.add_segment(Segment::new(
                i,
                SegmentType::Media,
                6.0,
                i as f64 * 6.0,
                format!("seg{}.ts", i),
                "1080p",
            ));
        }

        let delta = playlist.render_delta(5);

        assert!(delta.contains("#EXT-X-SKIP:SKIPPED-SEGMENTS=5"));
        assert!(delta.contains("#EXT-X-MEDIA-SEQUENCE:5"));
        assert!(!delta.contains("seg0.ts"));
        assert!(!delta.contains("seg4.ts"));
        assert!(delta.contains("seg5.ts"));
        assert!(delta.contains("seg9.ts"));
    }

    #[test]
    fn test_blocking_request_parsing() {
        let request = BlockingPlaylistRequest::from_query("_HLSmsn=10&_HLSpart=5&_HLSskip=YES");

        assert_eq!(request.msn, Some(10));
        assert_eq!(request.part, Some(5));
        assert_eq!(request.skip, Some(SkipDirective::Yes));
    }

    #[test]
    fn test_low_latency_config() {
        let config = LowLatencyConfig {
            part_hold_back: 0.9,
            hold_back: 18.0,
            can_block_reload: true,
            can_skip_until: Some(36.0),
            can_skip_dateranges: false,
            max_parts_after_segment: 3,
            rendition_reports: true,
        };

        assert!(config.can_block_reload);
        assert_eq!(config.part_hold_back, 0.9);
        assert_eq!(config.can_skip_until, Some(36.0));
    }

    #[test]
    fn test_hls_config_validation() {
        let config = HlsConfig::new("/tmp/test")
            .with_low_latency(0.2);

        assert!(config.validate().is_ok());

        // Segment duration validation - setting directly to avoid the max(1.0) in with_segment_duration
        let mut bad_config = HlsConfig::new("/tmp/test");
        bad_config.segment_duration = 0.5;

        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_partial_segment_with_byte_range() {
        let part = PartialSegment::new(0, 0, 0.2, "segment.ts")
            .with_byte_range(10000, Some(50000));

        let tag = part.render();

        assert!(tag.contains("BYTERANGE=\"10000@50000\""));
    }

    #[test]
    fn test_partial_segment_with_gap() {
        let part = PartialSegment::new(0, 0, 0.2, "gap.part")
            .with_gap();

        let tag = part.render();

        assert!(tag.contains("GAP=YES"));
    }

    #[test]
    fn test_preload_hint_with_byte_range() {
        let hint = PreloadHint::part("segment.part")
            .with_byte_range(1000, Some(5000));

        let tag = hint.render();

        assert!(tag.contains("BYTERANGE-START=1000"));
        assert!(tag.contains("BYTERANGE-LENGTH=5000"));
    }

    #[test]
    fn test_media_playlist_last_msn() {
        let mut playlist = MediaPlaylist::new(6);
        playlist.media_sequence = 100;

        playlist.add_segment(Segment::new(100, SegmentType::Media, 6.0, 0.0, "seg0.ts", "1080p"));
        playlist.add_segment(Segment::new(101, SegmentType::Media, 6.0, 6.0, "seg1.ts", "1080p"));
        playlist.add_segment(Segment::new(102, SegmentType::Media, 6.0, 12.0, "seg2.ts", "1080p"));

        assert_eq!(playlist.last_msn(), 102);
    }

    #[test]
    fn test_media_playlist_last_part() {
        let mut playlist = MediaPlaylist::new_ll_hls(6, 0.2);
        playlist.media_sequence = 5;

        playlist.add_segment(Segment::new(5, SegmentType::Media, 6.0, 0.0, "seg5.ts", "1080p"));

        playlist.add_partial_segment(PartialSegment::new(0, 6, 0.2, "seg6.0.part"));
        playlist.add_partial_segment(PartialSegment::new(1, 6, 0.2, "seg6.1.part"));
        playlist.add_partial_segment(PartialSegment::new(2, 6, 0.2, "seg6.2.part"));

        // last_msn should be 5 (the last complete segment)
        assert_eq!(playlist.last_msn(), 5);

        // last_part for sequence 6 should be 2
        let last_part = playlist.partial_segments
            .iter()
            .filter(|p| p.segment_sequence == 6)
            .map(|p| p.part_index)
            .max();
        assert_eq!(last_part, Some(2));
    }

    #[test]
    fn test_full_ll_hls_playlist() {
        let mut playlist = MediaPlaylist::new_ll_hls(6, 0.2);

        // Configure server control
        playlist.server_control = Some(
            ServerControl::new()
                .with_blocking_reload()
                .with_part_hold_back(0.6)
                .with_hold_back(18.0)
                .with_can_skip_until(36.0)
        );

        // Add init segment
        playlist.add_segment(Segment::new(0, SegmentType::Init, 0.0, 0.0, "init.mp4", "1080p"));

        // Add complete segment with parts
        for i in 0..30 {
            playlist.add_partial_segment(
                PartialSegment::new(i, 0, 0.2, format!("segment_00000.{}.part", i))
                    .with_independent(i == 0)
            );
        }
        playlist.add_segment(Segment::new(0, SegmentType::Media, 6.0, 0.0, "segment_00000.ts", "1080p"));

        // Add partial segments for current segment
        for i in 0..15 {
            playlist.add_partial_segment(
                PartialSegment::new(i, 1, 0.2, format!("segment_00001.{}.part", i))
                    .with_independent(i == 0)
            );
        }

        // Add preload hint
        playlist.add_preload_hint(PreloadHint::part("segment_00001.15.part"));

        // Add rendition reports
        playlist.add_rendition_report(
            RenditionReport::new("../720p/playlist.m3u8")
                .with_last_msn(1)
                .with_last_part(14)
        );

        let content = playlist.render();

        // Verify all LL-HLS elements are present
        assert!(content.contains("#EXT-X-PART-INF:PART-TARGET=0.200000"));
        assert!(content.contains("#EXT-X-SERVER-CONTROL:"));
        assert!(content.contains("CAN-BLOCK-RELOAD=YES"));
        assert!(content.contains("PART-HOLD-BACK=0.600000"));
        assert!(content.contains("CAN-SKIP-UNTIL=36.000000"));
        assert!(content.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(content.contains("#EXT-X-PART:DURATION=0.200000"));
        assert!(content.contains("INDEPENDENT=YES"));
        assert!(content.contains("#EXT-X-PRELOAD-HINT:TYPE=PART"));
        assert!(content.contains("#EXT-X-RENDITION-REPORT:"));
    }

    #[test]
    fn test_session_data() {
        let mut master = MasterPlaylist::new();
        master.session_data.push(SessionData {
            data_id: "com.example.title".to_string(),
            value: Some("Test Stream".to_string()),
            uri: None,
            language: Some("en".to_string()),
        });

        let content = master.render();

        assert!(content.contains("#EXT-X-SESSION-DATA:"));
        assert!(content.contains("DATA-ID=\"com.example.title\""));
        assert!(content.contains("VALUE=\"Test Stream\""));
        assert!(content.contains("LANGUAGE=\"en\""));
    }
}
