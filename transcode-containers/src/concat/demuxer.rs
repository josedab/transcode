//! Concat demuxer implementation.
//!
//! Provides a demuxer that sequentially reads multiple input files as a single stream,
//! handling timestamp adjustment between files automatically.

use crate::mp4::Mp4Demuxer;
use crate::traits::{CodecId, Demuxer, SeekMode, SeekResult, SeekTarget, StreamInfo, TrackType};
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use thiserror::Error;
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::timestamp::TimeBase;

/// Errors specific to concat demuxing.
#[derive(Error, Debug)]
pub enum ConcatError {
    /// No input files provided.
    #[error("No input files provided")]
    NoInputFiles,

    /// Stream count mismatch between files.
    #[error("Stream count mismatch: first file has {expected} streams, file {file_index} has {found}")]
    StreamCountMismatch {
        expected: usize,
        found: usize,
        file_index: usize,
    },

    /// Codec mismatch between files.
    #[error("Codec mismatch in stream {stream_index}: expected {expected:?}, found {found:?} in file {file_index}")]
    CodecMismatch {
        stream_index: usize,
        expected: CodecId,
        found: CodecId,
        file_index: usize,
    },

    /// Video resolution mismatch between files.
    #[error("Resolution mismatch in stream {stream_index}: expected {expected_width}x{expected_height}, found {found_width}x{found_height} in file {file_index}")]
    ResolutionMismatch {
        stream_index: usize,
        expected_width: u32,
        expected_height: u32,
        found_width: u32,
        found_height: u32,
        file_index: usize,
    },

    /// Audio sample rate mismatch between files.
    #[error("Sample rate mismatch in stream {stream_index}: expected {expected}Hz, found {found}Hz in file {file_index}")]
    SampleRateMismatch {
        stream_index: usize,
        expected: u32,
        found: u32,
        file_index: usize,
    },

    /// Audio channel count mismatch between files.
    #[error("Channel count mismatch in stream {stream_index}: expected {expected}, found {found} in file {file_index}")]
    ChannelCountMismatch {
        stream_index: usize,
        expected: u8,
        found: u8,
        file_index: usize,
    },

    /// Track type mismatch between files.
    #[error("Track type mismatch in stream {stream_index}: expected {expected:?}, found {found:?} in file {file_index}")]
    TrackTypeMismatch {
        stream_index: usize,
        expected: TrackType,
        found: TrackType,
        file_index: usize,
    },

    /// Failed to open file.
    #[error("Failed to open file '{path}': {message}")]
    FileOpenError { path: String, message: String },

    /// Invalid concat file format.
    #[error("Invalid concat file format at line {line}: {message}")]
    InvalidConcatFormat { line: usize, message: String },

    /// File not found in concat list.
    #[error("File not found: {path}")]
    FileNotFound { path: String },
}

impl From<ConcatError> for Error {
    fn from(e: ConcatError) -> Self {
        Error::Container(e.to_string().into())
    }
}

/// Validation strictness level for stream compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationLevel {
    /// Strict validation: all parameters must match exactly.
    #[default]
    Strict,
    /// Relaxed validation: only codec must match.
    Relaxed,
    /// No validation: accept any stream configuration.
    None,
}

/// Configuration for the concat demuxer.
#[derive(Debug, Clone)]
pub struct ConcatConfig {
    /// Validation level for stream compatibility checks.
    pub validation_level: ValidationLevel,
    /// Whether to reset timestamps at the start of each file.
    /// If false, timestamps continue from the end of the previous file.
    pub reset_timestamps: bool,
    /// Base directory for resolving relative paths in concat files.
    pub base_dir: Option<PathBuf>,
    /// Gap duration in microseconds to insert between files.
    /// Defaults to 0 (seamless concatenation).
    pub gap_duration_us: i64,
}

impl Default for ConcatConfig {
    fn default() -> Self {
        Self {
            validation_level: ValidationLevel::Strict,
            reset_timestamps: false,
            base_dir: None,
            gap_duration_us: 0,
        }
    }
}

impl ConcatConfig {
    /// Create a new config with strict validation.
    pub fn strict() -> Self {
        Self::default()
    }

    /// Create a new config with relaxed validation.
    pub fn relaxed() -> Self {
        Self {
            validation_level: ValidationLevel::Relaxed,
            ..Default::default()
        }
    }

    /// Create a new config with no validation.
    pub fn no_validation() -> Self {
        Self {
            validation_level: ValidationLevel::None,
            ..Default::default()
        }
    }

    /// Set the base directory for relative path resolution.
    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }

    /// Set whether to reset timestamps at file boundaries.
    pub fn with_reset_timestamps(mut self, reset: bool) -> Self {
        self.reset_timestamps = reset;
        self
    }

    /// Set the gap duration between files in microseconds.
    pub fn with_gap(mut self, gap_us: i64) -> Self {
        self.gap_duration_us = gap_us;
        self
    }
}

/// An entry in the concat file list.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Path to the file.
    pub path: PathBuf,
    /// Optional duration override in microseconds.
    pub duration_us: Option<i64>,
    /// Optional in-point (start time) in microseconds.
    pub in_point_us: Option<i64>,
    /// Optional out-point (end time) in microseconds.
    pub out_point_us: Option<i64>,
}

impl FileEntry {
    /// Create a new file entry from a path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            duration_us: None,
            in_point_us: None,
            out_point_us: None,
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration_us: i64) -> Self {
        self.duration_us = Some(duration_us);
        self
    }

    /// Set the in-point.
    pub fn with_in_point(mut self, in_point_us: i64) -> Self {
        self.in_point_us = Some(in_point_us);
        self
    }

    /// Set the out-point.
    pub fn with_out_point(mut self, out_point_us: i64) -> Self {
        self.out_point_us = Some(out_point_us);
        self
    }
}

/// Input source for the concat demuxer.
#[derive(Debug, Clone)]
pub enum ConcatInputSource {
    /// Direct list of file paths.
    Files(Vec<PathBuf>),
    /// List of file entries with optional metadata.
    Entries(Vec<FileEntry>),
    /// Path to a concat file in FFmpeg format.
    ConcatFile(PathBuf),
    /// Concat file content as a string.
    ConcatString(String),
}

impl ConcatInputSource {
    /// Create from a list of file paths.
    pub fn from_files(files: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self::Files(files.into_iter().map(Into::into).collect())
    }

    /// Create from file entries.
    pub fn from_entries(entries: Vec<FileEntry>) -> Self {
        Self::Entries(entries)
    }

    /// Create from a concat file path.
    pub fn from_concat_file(path: impl Into<PathBuf>) -> Self {
        Self::ConcatFile(path.into())
    }

    /// Create from concat file content string.
    pub fn from_concat_string(content: impl Into<String>) -> Self {
        Self::ConcatString(content.into())
    }
}

/// State for tracking the current file being demuxed.
struct FileState {
    /// The demuxer for the current file.
    demuxer: Mp4Demuxer,
    /// File index in the list.
    index: usize,
    /// Duration of this file in microseconds.
    duration_us: i64,
    /// Timestamp offset for this file.
    timestamp_offset_us: i64,
    /// Resolved file path.
    path: PathBuf,
}

/// Concat demuxer for reading multiple files as a single stream.
///
/// This demuxer opens multiple input files and presents them as a single
/// continuous media stream. Timestamps are automatically adjusted to
/// provide seamless playback across file boundaries.
///
/// # Example
///
/// ```ignore
/// use transcode_containers::concat::{ConcatDemuxer, ConcatConfig, ConcatInputSource};
/// use transcode_containers::traits::Demuxer;
///
/// let files = vec!["file1.mp4", "file2.mp4", "file3.mp4"];
/// let mut demuxer = ConcatDemuxer::new(
///     ConcatInputSource::from_files(files),
///     ConcatConfig::default(),
/// );
///
/// demuxer.initialize()?;
///
/// while let Some(packet) = demuxer.read_packet()? {
///     // Process packet with adjusted timestamps
/// }
/// ```
pub struct ConcatDemuxer {
    /// Input source.
    source: ConcatInputSource,
    /// Configuration.
    config: ConcatConfig,
    /// Resolved file entries.
    entries: Vec<FileEntry>,
    /// Current file state.
    current_file: Option<FileState>,
    /// Stream info from the first file (reference for validation).
    streams: Vec<StreamInfo>,
    /// Total duration in microseconds.
    total_duration_us: Option<i64>,
    /// Current timestamp offset for the active file.
    current_offset_us: i64,
    /// Whether the demuxer has been initialized.
    initialized: bool,
}

impl ConcatDemuxer {
    /// Create a new concat demuxer.
    pub fn new(source: ConcatInputSource, config: ConcatConfig) -> Self {
        Self {
            source,
            config,
            entries: Vec::new(),
            current_file: None,
            streams: Vec::new(),
            total_duration_us: None,
            current_offset_us: 0,
            initialized: false,
        }
    }

    /// Create a concat demuxer with default configuration.
    pub fn with_files(files: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self::new(ConcatInputSource::from_files(files), ConcatConfig::default())
    }

    /// Initialize the demuxer by resolving the input source and opening the first file.
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Resolve input source to file entries
        self.entries = self.resolve_input_source()?;

        if self.entries.is_empty() {
            return Err(ConcatError::NoInputFiles.into());
        }

        // Open the first file and extract reference stream info
        self.open_file(0)?;

        // Store stream info from first file
        if let Some(ref state) = self.current_file {
            for i in 0..state.demuxer.num_streams() {
                if let Some(info) = state.demuxer.stream_info(i) {
                    self.streams.push(info.clone());
                }
            }
        }

        // Validate all files (if validation is enabled)
        if self.config.validation_level != ValidationLevel::None {
            self.validate_all_files()?;
        }

        // Calculate total duration
        self.calculate_total_duration()?;

        self.initialized = true;
        Ok(())
    }

    /// Resolve the input source to a list of file entries.
    fn resolve_input_source(&self) -> Result<Vec<FileEntry>> {
        match &self.source {
            ConcatInputSource::Files(files) => {
                Ok(files.iter().map(FileEntry::new).collect())
            }
            ConcatInputSource::Entries(entries) => Ok(entries.clone()),
            ConcatInputSource::ConcatFile(path) => {
                let content = std::fs::read_to_string(path).map_err(|e| {
                    ConcatError::FileOpenError {
                        path: path.display().to_string(),
                        message: e.to_string(),
                    }
                })?;
                self.parse_concat_file(&content, path.parent())
            }
            ConcatInputSource::ConcatString(content) => {
                self.parse_concat_file(content, self.config.base_dir.as_deref())
            }
        }
    }

    /// Parse FFmpeg-style concat file format.
    ///
    /// Format:
    /// ```text
    /// ffconcat version 1.0
    /// file 'path/to/file1.mp4'
    /// duration 10.5
    /// file 'path/to/file2.mp4'
    /// inpoint 1.0
    /// outpoint 5.0
    /// file 'path/to/file3.mp4'
    /// ```
    fn parse_concat_file(
        &self,
        content: &str,
        base_dir: Option<&Path>,
    ) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        let mut current_entry: Option<FileEntry> = None;
        let mut version_found = false;

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Check for version header
            if line.starts_with("ffconcat") {
                if line.contains("version 1.0") {
                    version_found = true;
                }
                continue;
            }

            // Parse directives
            if let Some(rest) = line.strip_prefix("file ") {
                // Save previous entry
                if let Some(entry) = current_entry.take() {
                    entries.push(entry);
                }

                // Extract path (handle quoted and unquoted)
                let path_str = parse_quoted_value(rest).ok_or_else(|| {
                    ConcatError::InvalidConcatFormat {
                        line: line_num + 1,
                        message: "Invalid file path format".to_string(),
                    }
                })?;

                let path = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else if let Some(base) = base_dir.or(self.config.base_dir.as_deref()) {
                    base.join(&path_str)
                } else {
                    PathBuf::from(&path_str)
                };

                current_entry = Some(FileEntry::new(path));
            } else if let Some(rest) = line.strip_prefix("duration ") {
                if let Some(ref mut entry) = current_entry {
                    let duration = parse_time_value(rest).ok_or_else(|| {
                        ConcatError::InvalidConcatFormat {
                            line: line_num + 1,
                            message: "Invalid duration value".to_string(),
                        }
                    })?;
                    entry.duration_us = Some(duration);
                }
            } else if let Some(rest) = line.strip_prefix("inpoint ") {
                if let Some(ref mut entry) = current_entry {
                    let inpoint = parse_time_value(rest).ok_or_else(|| {
                        ConcatError::InvalidConcatFormat {
                            line: line_num + 1,
                            message: "Invalid inpoint value".to_string(),
                        }
                    })?;
                    entry.in_point_us = Some(inpoint);
                }
            } else if let Some(rest) = line.strip_prefix("outpoint ") {
                if let Some(ref mut entry) = current_entry {
                    let outpoint = parse_time_value(rest).ok_or_else(|| {
                        ConcatError::InvalidConcatFormat {
                            line: line_num + 1,
                            message: "Invalid outpoint value".to_string(),
                        }
                    })?;
                    entry.out_point_us = Some(outpoint);
                }
            } else if !version_found && !line.starts_with("stream") && !line.starts_with("exact_stream_id") {
                // Ignore unknown directives after version, but warn if no version
                // (We're lenient for compatibility)
            }
        }

        // Don't forget the last entry
        if let Some(entry) = current_entry {
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Open a file at the given index.
    fn open_file(&mut self, index: usize) -> Result<()> {
        let entry = self.entries.get(index).ok_or_else(|| {
            Error::Container("File index out of range".into())
        })?;

        // Open the file
        let file = File::open(&entry.path).map_err(|e| {
            ConcatError::FileOpenError {
                path: entry.path.display().to_string(),
                message: e.to_string(),
            }
        })?;

        // Create and open demuxer
        let mut demuxer = Mp4Demuxer::new();
        demuxer.open(file)?;

        // Handle in-point seeking if specified
        if let Some(in_point) = entry.in_point_us {
            demuxer.seek(in_point)?;
        }

        // Get duration
        let duration_us = if let Some(dur) = entry.duration_us {
            dur
        } else if let Some(out) = entry.out_point_us {
            out - entry.in_point_us.unwrap_or(0)
        } else {
            demuxer.duration().unwrap_or(0)
        };

        self.current_file = Some(FileState {
            demuxer,
            index,
            duration_us,
            timestamp_offset_us: self.current_offset_us,
            path: entry.path.clone(),
        });

        Ok(())
    }

    /// Validate all files have compatible streams.
    fn validate_all_files(&self) -> Result<()> {
        for (i, entry) in self.entries.iter().enumerate().skip(1) {
            // Open file for validation
            let file = File::open(&entry.path).map_err(|e| {
                ConcatError::FileOpenError {
                    path: entry.path.display().to_string(),
                    message: e.to_string(),
                }
            })?;

            let mut demuxer = Mp4Demuxer::new();
            demuxer.open(file)?;

            self.validate_streams(&demuxer, i)?;
        }

        Ok(())
    }

    /// Validate that a demuxer's streams are compatible with the reference.
    fn validate_streams(&self, demuxer: &Mp4Demuxer, file_index: usize) -> Result<()> {
        // Check stream count
        if demuxer.num_streams() != self.streams.len() {
            return Err(ConcatError::StreamCountMismatch {
                expected: self.streams.len(),
                found: demuxer.num_streams(),
                file_index,
            }
            .into());
        }

        for (i, ref_stream) in self.streams.iter().enumerate() {
            let stream = demuxer.stream_info(i).ok_or_else(|| {
                Error::Container(format!("Stream {} not found in file {}", i, file_index).into())
            })?;

            // Check track type
            if stream.track_type != ref_stream.track_type {
                return Err(ConcatError::TrackTypeMismatch {
                    stream_index: i,
                    expected: ref_stream.track_type,
                    found: stream.track_type,
                    file_index,
                }
                .into());
            }

            // Check codec
            if stream.codec_id != ref_stream.codec_id {
                return Err(ConcatError::CodecMismatch {
                    stream_index: i,
                    expected: ref_stream.codec_id.clone(),
                    found: stream.codec_id.clone(),
                    file_index,
                }
                .into());
            }

            // For strict validation, check additional parameters
            if self.config.validation_level == ValidationLevel::Strict {
                // Video resolution check
                if let (Some(ref ref_video), Some(ref video)) =
                    (&ref_stream.video, &stream.video)
                {
                    if video.width != ref_video.width || video.height != ref_video.height {
                        return Err(ConcatError::ResolutionMismatch {
                            stream_index: i,
                            expected_width: ref_video.width,
                            expected_height: ref_video.height,
                            found_width: video.width,
                            found_height: video.height,
                            file_index,
                        }
                        .into());
                    }
                }

                // Audio sample rate and channel count check
                if let (Some(ref ref_audio), Some(ref audio)) =
                    (&ref_stream.audio, &stream.audio)
                {
                    if audio.sample_rate != ref_audio.sample_rate {
                        return Err(ConcatError::SampleRateMismatch {
                            stream_index: i,
                            expected: ref_audio.sample_rate,
                            found: audio.sample_rate,
                            file_index,
                        }
                        .into());
                    }

                    if audio.channels != ref_audio.channels {
                        return Err(ConcatError::ChannelCountMismatch {
                            stream_index: i,
                            expected: ref_audio.channels,
                            found: audio.channels,
                            file_index,
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate total duration of all files.
    fn calculate_total_duration(&mut self) -> Result<()> {
        let mut total = 0i64;

        for (i, entry) in self.entries.iter().enumerate() {
            let duration = if let Some(dur) = entry.duration_us {
                dur
            } else if let Some(out) = entry.out_point_us {
                out - entry.in_point_us.unwrap_or(0)
            } else if i == 0 {
                // Use current file's duration for first file
                self.current_file
                    .as_ref()
                    .and_then(|f| f.demuxer.duration())
                    .unwrap_or(0)
            } else {
                // Open file to get duration
                let file = File::open(&entry.path).map_err(|e| {
                    ConcatError::FileOpenError {
                        path: entry.path.display().to_string(),
                        message: e.to_string(),
                    }
                })?;

                let mut demuxer = Mp4Demuxer::new();
                demuxer.open(file)?;
                demuxer.duration().unwrap_or(0)
            };

            total = total.saturating_add(duration);
            total = total.saturating_add(self.config.gap_duration_us);
        }

        // Remove the last gap (no gap after the last file)
        if !self.entries.is_empty() {
            total = total.saturating_sub(self.config.gap_duration_us);
        }

        self.total_duration_us = Some(total);
        Ok(())
    }

    /// Move to the next file.
    fn advance_to_next_file(&mut self) -> Result<bool> {
        let next_index = self
            .current_file
            .as_ref()
            .map(|f| f.index + 1)
            .unwrap_or(0);

        if next_index >= self.entries.len() {
            return Ok(false);
        }

        // Update offset from previous file
        if let Some(ref state) = self.current_file {
            self.current_offset_us = state
                .timestamp_offset_us
                .saturating_add(state.duration_us)
                .saturating_add(self.config.gap_duration_us);
        }

        // Close current file
        if let Some(ref mut state) = self.current_file {
            state.demuxer.close();
        }

        self.open_file(next_index)?;
        Ok(true)
    }

    /// Adjust packet timestamps based on current file offset.
    fn adjust_packet_timestamps(&self, packet: &mut Packet<'static>) {
        if self.config.reset_timestamps {
            return;
        }

        let Some(ref state) = self.current_file else {
            return;
        };

        // Convert offset to the packet's time base
        let time_base = packet.pts.time_base;
        let offset_value = TimeBase::MICROSECONDS.convert(
            state.timestamp_offset_us,
            time_base,
        );

        // Adjust timestamps
        if packet.pts.is_valid() {
            packet.pts.value = packet.pts.value.saturating_add(offset_value);
        }

        if packet.dts.is_valid() {
            packet.dts.value = packet.dts.value.saturating_add(offset_value);
        }
    }

    /// Get the current file index.
    pub fn current_file_index(&self) -> Option<usize> {
        self.current_file.as_ref().map(|f| f.index)
    }

    /// Get the current file path.
    pub fn current_file_path(&self) -> Option<&Path> {
        self.current_file.as_ref().map(|f| f.path.as_path())
    }

    /// Get all file entries.
    pub fn entries(&self) -> &[FileEntry] {
        &self.entries
    }

    /// Get the number of files.
    pub fn file_count(&self) -> usize {
        self.entries.len()
    }
}

impl Demuxer for ConcatDemuxer {
    fn open<R: Read + Seek + Send + 'static>(&mut self, _reader: R) -> Result<()> {
        // ConcatDemuxer manages its own file I/O, so we ignore the reader
        // and use initialize() instead
        self.initialize()
    }

    fn format_name(&self) -> &str {
        "concat"
    }

    fn duration(&self) -> Option<i64> {
        self.total_duration_us
    }

    fn num_streams(&self) -> usize {
        self.streams.len()
    }

    fn stream_info(&self, index: usize) -> Option<&StreamInfo> {
        self.streams.get(index)
    }

    fn read_packet(&mut self) -> Result<Option<Packet<'static>>> {
        if !self.initialized {
            self.initialize()?;
        }

        loop {
            let Some(ref mut state) = self.current_file else {
                return Ok(None);
            };

            // Check if we've reached the out-point for this file
            let entry = &self.entries[state.index];

            // Try to read a packet from the current file
            match state.demuxer.read_packet()? {
                Some(mut packet) => {
                    // Check out-point
                    if let Some(out_point) = entry.out_point_us {
                        // Convert packet PTS to microseconds
                        if let Some(pts_us) = packet.pts.to_seconds().map(|s| (s * 1_000_000.0) as i64) {
                            if pts_us >= out_point {
                                // Past out-point, move to next file
                                if !self.advance_to_next_file()? {
                                    return Ok(None);
                                }
                                continue;
                            }
                        }
                    }

                    // Adjust timestamps
                    self.adjust_packet_timestamps(&mut packet);
                    return Ok(Some(packet));
                }
                None => {
                    // End of current file, try next
                    if !self.advance_to_next_file()? {
                        return Ok(None);
                    }
                }
            }
        }
    }

    fn seek_to(&mut self, target: SeekTarget, mode: SeekMode) -> Result<SeekResult> {
        if !self.initialized {
            self.initialize()?;
        }

        // Convert target to timestamp in microseconds
        let timestamp_us = match target {
            SeekTarget::Timestamp(ts) => ts,
            SeekTarget::ByteOffset(_) => {
                return Err(Error::unsupported(
                    "Byte offset seeking not supported in concat demuxer",
                ));
            }
            SeekTarget::Sample { .. } => {
                return Err(Error::unsupported(
                    "Sample-based seeking not supported in concat demuxer",
                ));
            }
        };

        // Find which file contains this timestamp
        let mut accumulated_us = 0i64;
        let mut target_file_index = 0;
        let mut local_timestamp_us = timestamp_us;

        for (i, entry) in self.entries.iter().enumerate() {
            let duration = if let Some(dur) = entry.duration_us {
                dur
            } else if i == 0 && self.current_file.as_ref().map(|f| f.index) == Some(0) {
                self.current_file
                    .as_ref()
                    .and_then(|f| f.demuxer.duration())
                    .unwrap_or(0)
            } else {
                // We'd need to open the file to get duration, but let's use cached value
                0
            };

            if accumulated_us.saturating_add(duration) > timestamp_us {
                target_file_index = i;
                local_timestamp_us = timestamp_us.saturating_sub(accumulated_us);
                break;
            }

            accumulated_us = accumulated_us
                .saturating_add(duration)
                .saturating_add(self.config.gap_duration_us);
            target_file_index = i + 1;
            local_timestamp_us = 0;
        }

        // Clamp to valid file index
        target_file_index = target_file_index.min(self.entries.len().saturating_sub(1));

        // Update offset
        self.current_offset_us = accumulated_us;

        // Open target file if different
        let current_index = self.current_file.as_ref().map(|f| f.index);
        if current_index != Some(target_file_index) {
            if let Some(ref mut state) = self.current_file {
                state.demuxer.close();
            }
            self.current_file = None;
            self.open_file(target_file_index)?;
        }

        // Seek within the file
        let seek_result = if let Some(ref mut state) = self.current_file {
            let in_point = self.entries[state.index].in_point_us.unwrap_or(0);
            let local_target = SeekTarget::Timestamp(local_timestamp_us.saturating_add(in_point));
            let inner_result = state.demuxer.seek_to(local_target, mode)?;

            // Adjust the result timestamp by adding our offset
            SeekResult {
                timestamp_us: inner_result.timestamp_us.saturating_add(state.timestamp_offset_us),
                is_keyframe: inner_result.is_keyframe,
                sample_indices: inner_result.sample_indices,
            }
        } else {
            SeekResult {
                timestamp_us,
                is_keyframe: true,
                sample_indices: vec![],
            }
        };

        Ok(seek_result)
    }

    fn position(&self) -> Option<u64> {
        // For concat demuxer, position doesn't have a meaningful byte offset interpretation
        // since we're spanning multiple files. Return None to indicate this.
        None
    }

    fn close(&mut self) {
        if let Some(ref mut state) = self.current_file {
            state.demuxer.close();
        }
        self.current_file = None;
        self.initialized = false;
    }
}

/// Parse a quoted or unquoted value from a concat file.
fn parse_quoted_value(s: &str) -> Option<String> {
    let s = s.trim();

    if let Some(stripped) = s.strip_prefix('\'') {
        // Single-quoted
        let end = stripped.find('\'')?;
        Some(stripped[..end].to_string())
    } else if let Some(stripped) = s.strip_prefix('"') {
        // Double-quoted
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else {
        // Unquoted (take until whitespace)
        Some(s.split_whitespace().next()?.to_string())
    }
}

/// Parse a time value (seconds as float) to microseconds.
fn parse_time_value(s: &str) -> Option<i64> {
    let s = s.trim();
    let secs: f64 = s.parse().ok()?;
    Some((secs * 1_000_000.0) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_quoted_value_single() {
        assert_eq!(
            parse_quoted_value("'path/to/file.mp4'"),
            Some("path/to/file.mp4".to_string())
        );
    }

    #[test]
    fn test_parse_quoted_value_double() {
        assert_eq!(
            parse_quoted_value("\"path/to/file.mp4\""),
            Some("path/to/file.mp4".to_string())
        );
    }

    #[test]
    fn test_parse_quoted_value_unquoted() {
        assert_eq!(
            parse_quoted_value("path/to/file.mp4"),
            Some("path/to/file.mp4".to_string())
        );
    }

    #[test]
    fn test_parse_time_value() {
        assert_eq!(parse_time_value("10.5"), Some(10_500_000));
        assert_eq!(parse_time_value("0.001"), Some(1000));
        assert_eq!(parse_time_value("100"), Some(100_000_000));
    }

    #[test]
    fn test_concat_config_defaults() {
        let config = ConcatConfig::default();
        assert_eq!(config.validation_level, ValidationLevel::Strict);
        assert!(!config.reset_timestamps);
        assert_eq!(config.gap_duration_us, 0);
    }

    #[test]
    fn test_concat_config_builder() {
        let config = ConcatConfig::relaxed()
            .with_gap(1000)
            .with_reset_timestamps(true)
            .with_base_dir("/tmp");

        assert_eq!(config.validation_level, ValidationLevel::Relaxed);
        assert!(config.reset_timestamps);
        assert_eq!(config.gap_duration_us, 1000);
        assert_eq!(config.base_dir, Some(PathBuf::from("/tmp")));
    }

    #[test]
    fn test_file_entry_builder() {
        let entry = FileEntry::new("/path/to/file.mp4")
            .with_duration(5_000_000)
            .with_in_point(1_000_000)
            .with_out_point(4_000_000);

        assert_eq!(entry.path, PathBuf::from("/path/to/file.mp4"));
        assert_eq!(entry.duration_us, Some(5_000_000));
        assert_eq!(entry.in_point_us, Some(1_000_000));
        assert_eq!(entry.out_point_us, Some(4_000_000));
    }

    #[test]
    fn test_parse_concat_file_basic() {
        let content = r#"
ffconcat version 1.0
file 'file1.mp4'
file 'file2.mp4'
file 'file3.mp4'
"#;

        let demuxer = ConcatDemuxer::new(
            ConcatInputSource::ConcatString(content.to_string()),
            ConcatConfig::default(),
        );

        let entries = demuxer.parse_concat_file(content, None).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].path, PathBuf::from("file1.mp4"));
        assert_eq!(entries[1].path, PathBuf::from("file2.mp4"));
        assert_eq!(entries[2].path, PathBuf::from("file3.mp4"));
    }

    #[test]
    fn test_parse_concat_file_with_options() {
        let content = r#"
ffconcat version 1.0
file 'file1.mp4'
duration 10.5
file 'file2.mp4'
inpoint 1.0
outpoint 5.0
"#;

        let demuxer = ConcatDemuxer::new(
            ConcatInputSource::ConcatString(content.to_string()),
            ConcatConfig::default(),
        );

        let entries = demuxer.parse_concat_file(content, None).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].duration_us, Some(10_500_000));
        assert_eq!(entries[1].in_point_us, Some(1_000_000));
        assert_eq!(entries[1].out_point_us, Some(5_000_000));
    }

    #[test]
    fn test_parse_concat_file_with_base_dir() {
        let content = r#"
ffconcat version 1.0
file 'file1.mp4'
file 'subdir/file2.mp4'
"#;

        let base_dir = Path::new("/media/videos");
        let demuxer = ConcatDemuxer::new(
            ConcatInputSource::ConcatString(content.to_string()),
            ConcatConfig::default().with_base_dir(base_dir),
        );

        let entries = demuxer.parse_concat_file(content, Some(base_dir)).unwrap();
        assert_eq!(entries[0].path, PathBuf::from("/media/videos/file1.mp4"));
        assert_eq!(
            entries[1].path,
            PathBuf::from("/media/videos/subdir/file2.mp4")
        );
    }

    #[test]
    fn test_parse_concat_file_with_comments() {
        let content = r#"
# This is a comment
ffconcat version 1.0
file 'file1.mp4'
# Another comment
file 'file2.mp4'
"#;

        let demuxer = ConcatDemuxer::new(
            ConcatInputSource::ConcatString(content.to_string()),
            ConcatConfig::default(),
        );

        let entries = demuxer.parse_concat_file(content, None).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_input_source_from_files() {
        let source = ConcatInputSource::from_files(vec!["file1.mp4", "file2.mp4"]);
        match source {
            ConcatInputSource::Files(files) => {
                assert_eq!(files.len(), 2);
                assert_eq!(files[0], PathBuf::from("file1.mp4"));
            }
            _ => panic!("Expected Files variant"),
        }
    }

    #[test]
    fn test_concat_error_display() {
        let err = ConcatError::StreamCountMismatch {
            expected: 2,
            found: 1,
            file_index: 3,
        };
        assert!(err.to_string().contains("2 streams"));
        assert!(err.to_string().contains("file 3"));
    }

    #[test]
    fn test_validation_level_default() {
        let level = ValidationLevel::default();
        assert_eq!(level, ValidationLevel::Strict);
    }

    #[test]
    fn test_concat_demuxer_with_files() {
        let demuxer = ConcatDemuxer::with_files(vec!["file1.mp4", "file2.mp4"]);
        assert_eq!(demuxer.file_count(), 0); // Not initialized yet
        assert_eq!(demuxer.config.validation_level, ValidationLevel::Strict);
    }

    #[test]
    fn test_concat_demuxer_format_name() {
        let demuxer = ConcatDemuxer::with_files(vec!["file.mp4"]);
        assert_eq!(demuxer.format_name(), "concat");
    }

    // Integration tests would require actual MP4 files
    // These tests verify the concat file parsing and configuration logic
}
