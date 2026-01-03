//! Chapter markers support for media containers.
//!
//! This module provides functionality for reading, writing, and managing
//! chapter markers in media containers like MP4 and MKV.
//!
//! # Supported Formats
//!
//! ## MP4/MOV
//! - Nero chapters (chpl atom in udta)
//! - QuickTime text track chapters (tref/chap reference)
//!
//! ## Export Formats
//! - Simple text format (timestamp + title)
//! - WebVTT chapters

use std::fmt;
use std::io::Write;
use std::time::Duration;
use transcode_core::error::{Error, Result};

/// A single chapter marker.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Chapter title.
    pub title: String,
    /// Start time in nanoseconds from the beginning of the media.
    pub start_time_ns: u64,
    /// End time in nanoseconds from the beginning of the media.
    /// If None, the chapter extends to the start of the next chapter or end of media.
    pub end_time_ns: Option<u64>,
    /// ISO 639-2 language code (e.g., "eng", "deu", "fra").
    pub language: Option<String>,
}

impl Chapter {
    /// Create a new chapter with the given title and start time.
    pub fn new(title: impl Into<String>, start_time_ns: u64) -> Self {
        Self {
            title: title.into(),
            start_time_ns,
            end_time_ns: None,
            language: None,
        }
    }

    /// Create a new chapter with start and end times.
    pub fn with_duration(
        title: impl Into<String>,
        start_time_ns: u64,
        end_time_ns: u64,
    ) -> Self {
        Self {
            title: title.into(),
            start_time_ns,
            end_time_ns: Some(end_time_ns),
            language: None,
        }
    }

    /// Set the language for this chapter.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Get start time as Duration.
    pub fn start_time(&self) -> Duration {
        Duration::from_nanos(self.start_time_ns)
    }

    /// Get end time as Duration, if available.
    pub fn end_time(&self) -> Option<Duration> {
        self.end_time_ns.map(Duration::from_nanos)
    }

    /// Get duration of the chapter, if end time is available.
    pub fn duration(&self) -> Option<Duration> {
        self.end_time_ns.map(|end| {
            Duration::from_nanos(end.saturating_sub(self.start_time_ns))
        })
    }

    /// Get start time in milliseconds.
    pub fn start_time_ms(&self) -> u64 {
        self.start_time_ns / 1_000_000
    }

    /// Get end time in milliseconds, if available.
    pub fn end_time_ms(&self) -> Option<u64> {
        self.end_time_ns.map(|ns| ns / 1_000_000)
    }

    /// Get start time in 100-nanosecond units (used by MP4 chpl atom).
    pub fn start_time_100ns(&self) -> u64 {
        self.start_time_ns / 100
    }

    /// Create chapter from 100-nanosecond units.
    pub fn from_100ns(title: impl Into<String>, start_100ns: u64) -> Self {
        Self {
            title: title.into(),
            start_time_ns: start_100ns.saturating_mul(100),
            end_time_ns: None,
            language: None,
        }
    }
}

impl fmt::Display for Chapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let start_ms = self.start_time_ms();
        let hours = start_ms / 3_600_000;
        let minutes = (start_ms % 3_600_000) / 60_000;
        let seconds = (start_ms % 60_000) / 1000;
        let millis = start_ms % 1000;

        write!(
            f,
            "{:02}:{:02}:{:02}.{:03} - {}",
            hours, minutes, seconds, millis, self.title
        )
    }
}

/// A list of chapters with associated metadata.
#[derive(Debug, Clone, Default)]
pub struct ChapterList {
    /// The chapters in order.
    chapters: Vec<Chapter>,
    /// Overall language for the chapter list.
    pub language: Option<String>,
    /// Title/name of the chapter list.
    pub title: Option<String>,
}

impl ChapterList {
    /// Create a new empty chapter list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a chapter list with an initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            chapters: Vec::with_capacity(capacity),
            language: None,
            title: None,
        }
    }

    /// Add a chapter to the list.
    pub fn add(&mut self, chapter: Chapter) {
        self.chapters.push(chapter);
    }

    /// Add a chapter and return self for chaining.
    pub fn with_chapter(mut self, chapter: Chapter) -> Self {
        self.add(chapter);
        self
    }

    /// Get the number of chapters.
    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    /// Check if the chapter list is empty.
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Get a chapter by index.
    pub fn get(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }

    /// Get a mutable reference to a chapter by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Chapter> {
        self.chapters.get_mut(index)
    }

    /// Get all chapters as a slice.
    pub fn chapters(&self) -> &[Chapter] {
        &self.chapters
    }

    /// Get all chapters as a mutable slice.
    pub fn chapters_mut(&mut self) -> &mut [Chapter] {
        &mut self.chapters
    }

    /// Sort chapters by start time.
    pub fn sort_by_time(&mut self) {
        self.chapters.sort_by_key(|c| c.start_time_ns);
    }

    /// Compute and set end times based on the start of the next chapter.
    /// The last chapter's end time is set to `total_duration_ns` if provided.
    pub fn compute_end_times(&mut self, total_duration_ns: Option<u64>) {
        self.sort_by_time();
        let len = self.chapters.len();

        for i in 0..len {
            if i + 1 < len {
                self.chapters[i].end_time_ns = Some(self.chapters[i + 1].start_time_ns);
            } else if let Some(duration) = total_duration_ns {
                self.chapters[i].end_time_ns = Some(duration);
            }
        }
    }

    /// Find the chapter at a given timestamp (in nanoseconds).
    pub fn chapter_at(&self, time_ns: u64) -> Option<&Chapter> {
        // Binary search would be more efficient for large lists,
        // but linear search is fine for typical chapter counts.
        self.chapters
            .iter()
            .rev()
            .find(|chapter| time_ns >= chapter.start_time_ns)
    }

    /// Get total duration covered by chapters (last chapter's end time).
    pub fn total_duration(&self) -> Option<Duration> {
        self.chapters
            .last()
            .and_then(|c| c.end_time_ns)
            .map(Duration::from_nanos)
    }

    /// Create an iterator over the chapters.
    pub fn iter(&self) -> impl Iterator<Item = &Chapter> {
        self.chapters.iter()
    }

    /// Clear all chapters.
    pub fn clear(&mut self) {
        self.chapters.clear();
    }

    // =========================================================================
    // Export formats
    // =========================================================================

    /// Export to simple text format.
    ///
    /// Format: `HH:MM:SS.mmm Title`
    pub fn to_simple_text(&self) -> String {
        let mut output = String::new();

        for chapter in &self.chapters {
            let ms = chapter.start_time_ms();
            let hours = ms / 3_600_000;
            let minutes = (ms % 3_600_000) / 60_000;
            let seconds = (ms % 60_000) / 1000;
            let millis = ms % 1000;

            output.push_str(&format!(
                "{:02}:{:02}:{:02}.{:03} {}\n",
                hours, minutes, seconds, millis, chapter.title
            ));
        }

        output
    }

    /// Export to WebVTT chapter format.
    ///
    /// This generates a valid WebVTT file with chapter cues.
    pub fn to_webvtt(&self) -> String {
        let mut output = String::from("WEBVTT\n\n");

        let chapters: Vec<_> = self.chapters.iter().collect();

        for (i, chapter) in chapters.iter().enumerate() {
            // Chapter number
            output.push_str(&format!("Chapter {}\n", i + 1));

            // Start timestamp
            let start_ms = chapter.start_time_ms();
            let start_str = format_webvtt_timestamp(start_ms);

            // End timestamp (use next chapter start or add 1 second)
            let end_ms = chapter.end_time_ms().unwrap_or_else(|| {
                chapters.get(i + 1)
                    .map(|c| c.start_time_ms())
                    .unwrap_or(start_ms + 1000)
            });
            let end_str = format_webvtt_timestamp(end_ms);

            output.push_str(&format!("{} --> {}\n", start_str, end_str));
            output.push_str(&chapter.title);
            output.push_str("\n\n");
        }

        output
    }

    /// Write to a writer in simple text format.
    pub fn write_simple_text<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(self.to_simple_text().as_bytes())?;
        Ok(())
    }

    /// Write to a writer in WebVTT format.
    pub fn write_webvtt<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(self.to_webvtt().as_bytes())?;
        Ok(())
    }

    // =========================================================================
    // Import/parsing
    // =========================================================================

    /// Parse from simple text format.
    ///
    /// Expected format: `HH:MM:SS.mmm Title` or `HH:MM:SS Title`
    pub fn from_simple_text(text: &str) -> Result<Self> {
        let mut list = ChapterList::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to parse timestamp at the start
            if let Some(chapter) = parse_simple_text_line(line) {
                list.add(chapter);
            }
        }

        list.sort_by_time();
        Ok(list)
    }
}

impl IntoIterator for ChapterList {
    type Item = Chapter;
    type IntoIter = std::vec::IntoIter<Chapter>;

    fn into_iter(self) -> Self::IntoIter {
        self.chapters.into_iter()
    }
}

impl<'a> IntoIterator for &'a ChapterList {
    type Item = &'a Chapter;
    type IntoIter = std::slice::Iter<'a, Chapter>;

    fn into_iter(self) -> Self::IntoIter {
        self.chapters.iter()
    }
}

impl FromIterator<Chapter> for ChapterList {
    fn from_iter<I: IntoIterator<Item = Chapter>>(iter: I) -> Self {
        let chapters: Vec<_> = iter.into_iter().collect();
        ChapterList {
            chapters,
            language: None,
            title: None,
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Format a timestamp in WebVTT format (HH:MM:SS.mmm).
fn format_webvtt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Parse a line in simple text format.
fn parse_simple_text_line(line: &str) -> Option<Chapter> {
    // Try to match patterns like:
    // 00:00:00.000 Title
    // 00:00:00 Title
    // 0:00:00 Title

    let parts: Vec<&str> = line.splitn(2, |c: char| c.is_whitespace()).collect();
    if parts.len() != 2 {
        return None;
    }

    let timestamp_str = parts[0];
    let title = parts[1].trim();

    if title.is_empty() {
        return None;
    }

    let time_ns = parse_timestamp(timestamp_str)?;
    Some(Chapter::new(title, time_ns))
}

/// Parse a timestamp string into nanoseconds.
/// Supports formats: HH:MM:SS.mmm, HH:MM:SS, MM:SS.mmm, MM:SS
fn parse_timestamp(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split(':').collect();

    let (hours, minutes, seconds_part): (u64, u64, &str) = match parts.len() {
        2 => (0u64, parts[0].parse().ok()?, parts[1]),
        3 => (parts[0].parse().ok()?, parts[1].parse().ok()?, parts[2]),
        _ => return None,
    };

    // Parse seconds and optional milliseconds
    let (seconds, millis): (u64, u64) = if let Some(dot_pos) = seconds_part.find('.') {
        let sec: u64 = seconds_part[..dot_pos].parse().ok()?;
        let ms_str = &seconds_part[dot_pos + 1..];
        // Pad or truncate to 3 digits
        let ms: u64 = if ms_str.len() >= 3 {
            ms_str[..3].parse().ok()?
        } else {
            let mut padded = ms_str.to_string();
            while padded.len() < 3 {
                padded.push('0');
            }
            padded.parse().ok()?
        };
        (sec, ms)
    } else {
        (seconds_part.parse().ok()?, 0)
    };

    let total_ms = hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis;
    Some(total_ms * 1_000_000) // Convert to nanoseconds
}

// =============================================================================
// MP4 Chapter Support
// =============================================================================

/// MP4 chapter reader for extracting chapters from MP4/MOV files.
pub struct Mp4ChapterReader;

impl Mp4ChapterReader {
    /// Read chapters from a chpl (Nero chapters) atom data.
    ///
    /// The chpl atom format:
    /// - 4 bytes: version (1 byte) + flags (3 bytes)
    /// - 4 bytes: reserved (for version 0) or 1 byte reserved (version 1)
    /// - 1 byte: chapter count (version 0) or 2 bytes count (version 1)
    /// - For each chapter:
    ///   - 8 bytes: start time in 100-nanosecond units
    ///   - 1 byte: title length
    ///   - N bytes: title (UTF-8)
    pub fn parse_chpl(data: &[u8]) -> Result<ChapterList> {
        if data.len() < 5 {
            return Err(Error::Container("chpl atom too short".into()));
        }

        let version = data[0];
        let (mut offset, chapter_count): (usize, usize) = if version == 0 {
            // Version 0: 4 bytes header + 4 bytes reserved + 1 byte count
            if data.len() < 9 {
                return Err(Error::Container("chpl v0 atom too short".into()));
            }
            (9, data[8] as usize)
        } else {
            // Version 1: 4 bytes header + 1 byte reserved + 2 bytes count
            if data.len() < 7 {
                return Err(Error::Container("chpl v1 atom too short".into()));
            }
            (7, u16::from_be_bytes([data[5], data[6]]) as usize)
        };

        let mut list = ChapterList::with_capacity(chapter_count);

        for _ in 0..chapter_count {
            // Read start time (8 bytes, 100ns units)
            if offset + 9 > data.len() {
                break;
            }

            let start_100ns = u64::from_be_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]);
            offset += 8;

            // Read title length
            let title_len = data[offset] as usize;
            offset += 1;

            // Read title
            if offset + title_len > data.len() {
                break;
            }

            let title = String::from_utf8_lossy(&data[offset..offset + title_len]).to_string();
            offset += title_len;

            list.add(Chapter::from_100ns(title, start_100ns));
        }

        list.sort_by_time();
        Ok(list)
    }

    /// Parse chapters from a text track sample (tx3g format).
    ///
    /// Text track chapters in MP4 store the chapter title as the sample text,
    /// with the timing derived from the sample's PTS.
    pub fn parse_text_sample(data: &[u8], start_time_ns: u64) -> Result<Chapter> {
        if data.len() < 2 {
            return Err(Error::Container("Text sample too short".into()));
        }

        // First 2 bytes are the text length in tx3g format
        let text_len = u16::from_be_bytes([data[0], data[1]]) as usize;

        if data.len() < 2 + text_len {
            return Err(Error::Container("Text sample data truncated".into()));
        }

        let title = String::from_utf8_lossy(&data[2..2 + text_len]).to_string();

        Ok(Chapter::new(title, start_time_ns))
    }

    /// Check if a handler type indicates a chapter track.
    pub fn is_chapter_handler(handler_type: &[u8; 4]) -> bool {
        handler_type == b"text" || handler_type == b"sbtl"
    }
}

/// MP4 chapter writer for embedding chapters in MP4/MOV files.
pub struct Mp4ChapterWriter;

impl Mp4ChapterWriter {
    /// Build a chpl (Nero chapters) atom.
    ///
    /// Returns the complete atom data including the atom header.
    pub fn build_chpl(chapters: &ChapterList) -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags (version 1)
        data.push(1); // version
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Reserved byte
        data.push(0);

        // Chapter count (2 bytes for version 1)
        let count = chapters.len().min(65535) as u16;
        data.extend_from_slice(&count.to_be_bytes());

        // Chapters
        for chapter in chapters.iter() {
            // Start time in 100ns units
            let start_100ns = chapter.start_time_100ns();
            data.extend_from_slice(&start_100ns.to_be_bytes());

            // Title (truncate to 255 bytes max)
            let title_bytes = chapter.title.as_bytes();
            let title_len = title_bytes.len().min(255);
            data.push(title_len as u8);
            data.extend_from_slice(&title_bytes[..title_len]);
        }

        // Build complete atom with header
        let atom_size = (data.len() + 8) as u32;
        let mut atom = Vec::with_capacity(atom_size as usize);
        atom.extend_from_slice(&atom_size.to_be_bytes());
        atom.extend_from_slice(b"chpl");
        atom.extend_from_slice(&data);

        atom
    }

    /// Build a udta atom containing chapters.
    ///
    /// Returns the complete udta atom with the chpl nested inside.
    pub fn build_udta_with_chapters(chapters: &ChapterList) -> Vec<u8> {
        let chpl = Self::build_chpl(chapters);

        // Build udta wrapper
        let udta_size = (chpl.len() + 8) as u32;
        let mut udta = Vec::with_capacity(udta_size as usize);
        udta.extend_from_slice(&udta_size.to_be_bytes());
        udta.extend_from_slice(b"udta");
        udta.extend_from_slice(&chpl);

        udta
    }

    /// Build a text sample for a chapter track.
    ///
    /// This creates a tx3g format text sample.
    pub fn build_text_sample(title: &str) -> Vec<u8> {
        let title_bytes = title.as_bytes();
        let len = title_bytes.len().min(65535);

        let mut data = Vec::with_capacity(2 + len);
        data.extend_from_slice(&(len as u16).to_be_bytes());
        data.extend_from_slice(&title_bytes[..len]);

        data
    }
}

/// Chapter track reference information.
#[derive(Debug, Clone)]
pub struct ChapterTrackRef {
    /// Track ID of the track that references chapters.
    pub track_id: u32,
    /// Track IDs of the chapter tracks.
    pub chapter_track_ids: Vec<u32>,
}

impl ChapterTrackRef {
    /// Parse a tref atom to find chapter track references.
    ///
    /// The tref atom contains track reference type atoms, including 'chap'.
    pub fn parse_tref(data: &[u8], parent_track_id: u32) -> Result<Option<Self>> {
        let mut offset = 0;

        while offset + 8 <= data.len() {
            let atom_size = u32::from_be_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;

            let atom_type = &data[offset + 4..offset + 8];

            if atom_type == b"chap" {
                // Found chapter reference
                let mut chapter_track_ids = Vec::new();
                let content_start = offset + 8;
                let content_end = (offset + atom_size).min(data.len());

                let mut pos = content_start;
                while pos + 4 <= content_end {
                    let track_id = u32::from_be_bytes([
                        data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    ]);
                    chapter_track_ids.push(track_id);
                    pos += 4;
                }

                if !chapter_track_ids.is_empty() {
                    return Ok(Some(ChapterTrackRef {
                        track_id: parent_track_id,
                        chapter_track_ids,
                    }));
                }
            }

            if atom_size == 0 {
                break;
            }
            offset += atom_size;
        }

        Ok(None)
    }

    /// Build a tref atom with chapter reference.
    pub fn build_tref(chapter_track_ids: &[u32]) -> Vec<u8> {
        // Build chap atom
        let chap_content_size = chapter_track_ids.len() * 4;
        let chap_size = (8 + chap_content_size) as u32;

        let mut chap = Vec::with_capacity(chap_size as usize);
        chap.extend_from_slice(&chap_size.to_be_bytes());
        chap.extend_from_slice(b"chap");
        for track_id in chapter_track_ids {
            chap.extend_from_slice(&track_id.to_be_bytes());
        }

        // Build tref wrapper
        let tref_size = (8 + chap.len()) as u32;
        let mut tref = Vec::with_capacity(tref_size as usize);
        tref.extend_from_slice(&tref_size.to_be_bytes());
        tref.extend_from_slice(b"tref");
        tref.extend_from_slice(&chap);

        tref
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_creation() {
        let chapter = Chapter::new("Introduction", 0);
        assert_eq!(chapter.title, "Introduction");
        assert_eq!(chapter.start_time_ns, 0);
        assert!(chapter.end_time_ns.is_none());
        assert!(chapter.language.is_none());
    }

    #[test]
    fn test_chapter_with_duration() {
        let chapter = Chapter::with_duration("Scene 1", 1_000_000_000, 5_000_000_000);
        assert_eq!(chapter.title, "Scene 1");
        assert_eq!(chapter.start_time_ns, 1_000_000_000);
        assert_eq!(chapter.end_time_ns, Some(5_000_000_000));
        assert_eq!(chapter.duration(), Some(Duration::from_secs(4)));
    }

    #[test]
    fn test_chapter_with_language() {
        let chapter = Chapter::new("Einleitung", 0).with_language("deu");
        assert_eq!(chapter.language, Some("deu".to_string()));
    }

    #[test]
    fn test_chapter_time_conversions() {
        // 1 hour, 30 minutes, 45 seconds, 500 milliseconds
        let time_ns = (90 * 60 + 45) * 1_000_000_000u64 + 500_000_000;
        let chapter = Chapter::new("Test", time_ns);

        assert_eq!(chapter.start_time_ms(), 5445500);
        assert_eq!(chapter.start_time(), Duration::from_nanos(time_ns));
    }

    #[test]
    fn test_chapter_100ns_conversion() {
        // 1 second in 100ns units = 10,000,000
        let chapter = Chapter::from_100ns("Test", 10_000_000);
        assert_eq!(chapter.start_time_ns, 1_000_000_000);
        assert_eq!(chapter.start_time_100ns(), 10_000_000);
    }

    #[test]
    fn test_chapter_display() {
        let chapter = Chapter::new("Test Chapter", 3661500_000_000); // 1:01:01.500
        let display = format!("{}", chapter);
        assert_eq!(display, "01:01:01.500 - Test Chapter");
    }

    #[test]
    fn test_chapter_list_basic() {
        let mut list = ChapterList::new();
        assert!(list.is_empty());

        list.add(Chapter::new("Chapter 1", 0));
        list.add(Chapter::new("Chapter 2", 60_000_000_000));

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0).unwrap().title, "Chapter 1");
        assert_eq!(list.get(1).unwrap().title, "Chapter 2");
    }

    #[test]
    fn test_chapter_list_sorting() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Chapter 3", 180_000_000_000));
        list.add(Chapter::new("Chapter 1", 0));
        list.add(Chapter::new("Chapter 2", 60_000_000_000));

        list.sort_by_time();

        assert_eq!(list.get(0).unwrap().title, "Chapter 1");
        assert_eq!(list.get(1).unwrap().title, "Chapter 2");
        assert_eq!(list.get(2).unwrap().title, "Chapter 3");
    }

    #[test]
    fn test_chapter_list_compute_end_times() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Chapter 1", 0));
        list.add(Chapter::new("Chapter 2", 60_000_000_000));
        list.add(Chapter::new("Chapter 3", 120_000_000_000));

        list.compute_end_times(Some(180_000_000_000));

        assert_eq!(list.get(0).unwrap().end_time_ns, Some(60_000_000_000));
        assert_eq!(list.get(1).unwrap().end_time_ns, Some(120_000_000_000));
        assert_eq!(list.get(2).unwrap().end_time_ns, Some(180_000_000_000));
    }

    #[test]
    fn test_chapter_at() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Chapter 1", 0));
        list.add(Chapter::new("Chapter 2", 60_000_000_000));
        list.add(Chapter::new("Chapter 3", 120_000_000_000));
        list.sort_by_time();

        assert_eq!(list.chapter_at(0).unwrap().title, "Chapter 1");
        assert_eq!(list.chapter_at(30_000_000_000).unwrap().title, "Chapter 1");
        assert_eq!(list.chapter_at(60_000_000_000).unwrap().title, "Chapter 2");
        assert_eq!(list.chapter_at(90_000_000_000).unwrap().title, "Chapter 2");
        assert_eq!(list.chapter_at(120_000_000_000).unwrap().title, "Chapter 3");
    }

    #[test]
    fn test_simple_text_export() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Introduction", 0));
        list.add(Chapter::new("Main Content", 60_000_000_000)); // 1 minute
        list.add(Chapter::new("Conclusion", 3661_000_000_000)); // 1:01:01

        let text = list.to_simple_text();

        assert!(text.contains("00:00:00.000 Introduction"));
        assert!(text.contains("00:01:00.000 Main Content"));
        assert!(text.contains("01:01:01.000 Conclusion"));
    }

    #[test]
    fn test_webvtt_export() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Chapter 1", 0));
        list.add(Chapter::new("Chapter 2", 60_000_000_000));

        list.compute_end_times(Some(120_000_000_000));

        let vtt = list.to_webvtt();

        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("Chapter 1"));
        assert!(vtt.contains("00:00:00.000 --> 00:01:00.000"));
        assert!(vtt.contains("Chapter 2"));
    }

    #[test]
    fn test_simple_text_parsing() {
        let text = r#"
00:00:00.000 Introduction
00:01:30.500 First Scene
01:00:00.000 Intermission
"#;

        let list = ChapterList::from_simple_text(text).unwrap();

        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0).unwrap().title, "Introduction");
        assert_eq!(list.get(0).unwrap().start_time_ns, 0);
        assert_eq!(list.get(1).unwrap().title, "First Scene");
        assert_eq!(list.get(1).unwrap().start_time_ms(), 90500);
        assert_eq!(list.get(2).unwrap().title, "Intermission");
        assert_eq!(list.get(2).unwrap().start_time_ms(), 3600000);
    }

    #[test]
    fn test_parse_timestamp() {
        assert_eq!(parse_timestamp("00:00:00.000"), Some(0));
        assert_eq!(parse_timestamp("00:01:00"), Some(60_000_000_000));
        assert_eq!(parse_timestamp("01:00:00.000"), Some(3600_000_000_000));
        assert_eq!(parse_timestamp("1:30.500"), Some(90_500_000_000));
        assert_eq!(parse_timestamp("invalid"), None);
    }

    #[test]
    fn test_chpl_parsing() {
        // Build a simple chpl atom (version 1)
        let mut data = vec![
            1, 0, 0, 0, // version 1, flags
            0,          // reserved
            0, 2,       // 2 chapters
        ];

        // Chapter 1: 0 seconds, "Intro"
        data.extend_from_slice(&0u64.to_be_bytes());
        data.push(5); // title length
        data.extend_from_slice(b"Intro");

        // Chapter 2: 60 seconds (600,000,000 in 100ns units)
        data.extend_from_slice(&600_000_000u64.to_be_bytes());
        data.push(7); // title length
        data.extend_from_slice(b"Scene 1");

        let list = Mp4ChapterReader::parse_chpl(&data).unwrap();

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0).unwrap().title, "Intro");
        assert_eq!(list.get(0).unwrap().start_time_ns, 0);
        assert_eq!(list.get(1).unwrap().title, "Scene 1");
        assert_eq!(list.get(1).unwrap().start_time_ns, 60_000_000_000);
    }

    #[test]
    fn test_chpl_building() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Intro", 0));
        list.add(Chapter::new("Scene 1", 60_000_000_000));

        let chpl = Mp4ChapterWriter::build_chpl(&list);

        // Verify it's a valid atom
        assert!(chpl.len() >= 8);
        assert_eq!(&chpl[4..8], b"chpl");

        // Parse it back
        let parsed = Mp4ChapterReader::parse_chpl(&chpl[8..]).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.get(0).unwrap().title, "Intro");
        assert_eq!(parsed.get(1).unwrap().title, "Scene 1");
    }

    #[test]
    fn test_text_sample_parsing() {
        let mut data = vec![0, 5]; // length = 5
        data.extend_from_slice(b"Hello");

        let chapter = Mp4ChapterReader::parse_text_sample(&data, 1_000_000_000).unwrap();
        assert_eq!(chapter.title, "Hello");
        assert_eq!(chapter.start_time_ns, 1_000_000_000);
    }

    #[test]
    fn test_text_sample_building() {
        let sample = Mp4ChapterWriter::build_text_sample("Test Chapter");

        assert_eq!(sample[0..2], [0, 12]); // length = 12
        assert_eq!(&sample[2..], b"Test Chapter");
    }

    #[test]
    fn test_tref_chap_parsing() {
        // Build a tref with chap reference
        let mut data = vec![
            0, 0, 0, 16, // atom size = 16
            b'c', b'h', b'a', b'p', // type = chap
            0, 0, 0, 2, // track ID 2
            0, 0, 0, 3, // track ID 3
        ];

        let tref = ChapterTrackRef::parse_tref(&data, 1).unwrap().unwrap();
        assert_eq!(tref.track_id, 1);
        assert_eq!(tref.chapter_track_ids, vec![2, 3]);
    }

    #[test]
    fn test_tref_building() {
        let tref = ChapterTrackRef::build_tref(&[2, 3]);

        // Verify structure
        assert!(tref.len() >= 16);
        assert_eq!(&tref[4..8], b"tref");
        assert_eq!(&tref[12..16], b"chap");
    }

    #[test]
    fn test_udta_with_chapters() {
        let mut list = ChapterList::new();
        list.add(Chapter::new("Test", 0));

        let udta = Mp4ChapterWriter::build_udta_with_chapters(&list);

        assert!(udta.len() >= 8);
        assert_eq!(&udta[4..8], b"udta");
        // Verify chpl is nested
        assert!(udta.windows(4).any(|w| w == b"chpl"));
    }

    #[test]
    fn test_chapter_list_iteration() {
        let list = ChapterList::new()
            .with_chapter(Chapter::new("A", 0))
            .with_chapter(Chapter::new("B", 1000))
            .with_chapter(Chapter::new("C", 2000));

        let titles: Vec<_> = list.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_chapter_list_from_iter() {
        let chapters = vec![
            Chapter::new("First", 0),
            Chapter::new("Second", 1000),
        ];

        let list: ChapterList = chapters.into_iter().collect();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_is_chapter_handler() {
        assert!(Mp4ChapterReader::is_chapter_handler(b"text"));
        assert!(Mp4ChapterReader::is_chapter_handler(b"sbtl"));
        assert!(!Mp4ChapterReader::is_chapter_handler(b"vide"));
        assert!(!Mp4ChapterReader::is_chapter_handler(b"soun"));
    }
}
