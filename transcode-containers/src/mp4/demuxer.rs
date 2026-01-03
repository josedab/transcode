//! MP4 demuxer implementation.

use super::atoms::{AtomHeader, FtypAtom, HdlrAtom, MdhdAtom, MvhdAtom, StblInfo, TkhdAtom};
use crate::chapters::{ChapterList, ChapterTrackRef, Mp4ChapterReader};
use crate::traits::{
    AudioStreamInfo, CodecId, Demuxer, SeekMode, SeekResult, SeekTarget, StreamInfo, TrackType,
    VideoStreamInfo,
};
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::rational::Rational;
use transcode_core::timestamp::{TimeBase, Timestamp};
use std::io::{Read, Seek, SeekFrom};

/// Maximum size for atom content allocation (100 MB).
/// Prevents denial of service from malformed files with huge atom sizes.
const MAX_ATOM_CONTENT_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum size for sample data allocation (50 MB).
/// Individual video/audio samples shouldn't exceed this.
const MAX_SAMPLE_SIZE: u32 = 50 * 1024 * 1024;

/// Validate that an allocation size is within acceptable limits.
fn validate_allocation_size(size: u64) -> Result<usize> {
    if size > MAX_ATOM_CONTENT_SIZE {
        return Err(Error::Container(format!(
            "Atom content size {} exceeds maximum allowed size {}",
            size, MAX_ATOM_CONTENT_SIZE
        ).into()));
    }
    Ok(size as usize)
}

/// Track information.
#[derive(Debug)]
struct TrackInfo {
    /// Stream info.
    stream: StreamInfo,
    /// Sample table.
    stbl: StblInfo,
    /// Current sample index.
    current_sample: usize,
    /// Media timescale.
    timescale: u32,
}

impl TrackInfo {
    /// Get sample file offset.
    fn sample_offset(&self, sample_idx: usize) -> Option<u64> {
        // Find chunk containing this sample
        let mut samples_before = 0usize;
        let mut current_stsc_idx = 0;

        for (i, &(first_chunk, samples_per_chunk, _)) in self.stbl.stsc.iter().enumerate() {
            // Use saturating arithmetic to prevent overflow
            let chunk_count = self.stbl.chunk_offsets.len();
            let next_first = self
                .stbl
                .stsc
                .get(i + 1)
                .map(|e| e.0)
                .unwrap_or_else(|| (chunk_count as u32).saturating_add(1));

            let chunks_in_range = next_first.saturating_sub(first_chunk) as usize;
            let samples_in_range = chunks_in_range.checked_mul(samples_per_chunk as usize)?;

            if samples_before.saturating_add(samples_in_range) > sample_idx {
                current_stsc_idx = i;
                break;
            }
            samples_before = samples_before.checked_add(samples_in_range)?;
        }

        let (first_chunk, samples_per_chunk, _) = self.stbl.stsc.get(current_stsc_idx)?;
        let samples_per_chunk_usize = *samples_per_chunk as usize;
        if samples_per_chunk_usize == 0 {
            return None;
        }
        let sample_offset_in_range = sample_idx.checked_sub(samples_before)?;
        let chunk_offset_in_range = (sample_offset_in_range / samples_per_chunk_usize) as u32;
        let chunk_idx = first_chunk.checked_sub(1)?.checked_add(chunk_offset_in_range)?;
        let sample_in_chunk = sample_offset_in_range % samples_per_chunk_usize;

        let chunk_offset = *self.stbl.chunk_offsets.get(chunk_idx as usize)?;

        // Calculate offset within chunk
        let mut offset = chunk_offset;
        let chunk_idx_usize = chunk_idx as usize;
        let first_chunk_usize = *first_chunk as usize;
        for i in 0..sample_in_chunk {
            let chunks_from_first = chunk_idx_usize.checked_sub(first_chunk_usize)?.checked_add(1)?;
            let samples_in_prev_chunks = chunks_from_first.checked_mul(samples_per_chunk_usize)?;
            let sample_global_idx = samples_before.checked_add(samples_in_prev_chunks)?.checked_add(i)?;
            offset = offset.checked_add(*self.stbl.sample_sizes.get(sample_global_idx)? as u64)?;
        }

        Some(offset)
    }

    /// Get sample size.
    fn sample_size(&self, sample_idx: usize) -> Option<u32> {
        self.stbl.sample_sizes.get(sample_idx).copied()
    }

    /// Get sample timestamp.
    /// Returns the timestamp using saturating arithmetic to prevent overflow.
    fn sample_timestamp(&self, sample_idx: usize) -> i64 {
        let mut ts = 0i64;
        let mut sample_count = 0usize;

        for &(count, delta) in &self.stbl.stts {
            let samples_in_entry = count as usize;
            if sample_count.saturating_add(samples_in_entry) > sample_idx {
                // Use saturating arithmetic to prevent overflow
                let remaining = sample_idx.saturating_sub(sample_count);
                let delta_contribution = remaining.saturating_mul(delta as usize) as i64;
                ts = ts.saturating_add(delta_contribution);
                break;
            }
            // Use saturating arithmetic to prevent overflow
            let entry_contribution = samples_in_entry.saturating_mul(delta as usize) as i64;
            ts = ts.saturating_add(entry_contribution);
            sample_count = sample_count.saturating_add(samples_in_entry);
        }

        ts
    }

    /// Get composition time offset.
    fn composition_offset(&self, sample_idx: usize) -> i64 {
        let mut sample_count = 0usize;

        for &(count, offset) in &self.stbl.ctts {
            let samples_in_entry = count as usize;
            if sample_count + samples_in_entry > sample_idx {
                return offset as i64;
            }
            sample_count += samples_in_entry;
        }

        0
    }

    /// Check if sample is a keyframe.
    fn is_keyframe(&self, sample_idx: usize) -> bool {
        if self.stbl.stss.is_empty() {
            // No sync sample table - all samples are sync samples
            true
        } else {
            self.stbl.stss.contains(&(sample_idx as u32 + 1))
        }
    }

    /// Get total number of samples.
    fn sample_count(&self) -> usize {
        self.stbl.sample_sizes.len()
    }

    /// Find the sample index at or before the given timestamp.
    /// Returns (sample_index, sample_timestamp).
    fn find_sample_at_timestamp(&self, timestamp: i64) -> (usize, i64) {
        let mut sample_idx = 0usize;
        let mut sample_ts = 0i64;
        let mut prev_ts = 0i64;

        for &(count, delta) in &self.stbl.stts {
            for _ in 0..count {
                if sample_ts > timestamp {
                    // We've passed the target, return the previous sample
                    return (sample_idx.saturating_sub(1), prev_ts);
                }
                prev_ts = sample_ts;
                sample_ts = sample_ts.saturating_add(delta as i64);
                sample_idx = sample_idx.saturating_add(1);
            }
        }

        // Return the last sample if we didn't find a match
        (sample_idx.saturating_sub(1), prev_ts)
    }

    /// Find the nearest keyframe to the given sample index.
    /// If `forward` is true, find the next keyframe at or after the sample.
    /// If `forward` is false, find the previous keyframe at or before the sample.
    fn find_nearest_keyframe(&self, sample_idx: usize, forward: bool) -> usize {
        if self.stbl.stss.is_empty() {
            // No sync sample table means all samples are keyframes
            return sample_idx;
        }

        let sample_number = (sample_idx + 1) as u32;

        if forward {
            // Find first keyframe at or after sample_idx
            for &sync_sample in &self.stbl.stss {
                if sync_sample >= sample_number {
                    return (sync_sample as usize).saturating_sub(1);
                }
            }
            // If no keyframe found after, return the last keyframe
            self.stbl
                .stss
                .last()
                .map(|&s| (s as usize).saturating_sub(1))
                .unwrap_or(sample_idx)
        } else {
            // Find last keyframe at or before sample_idx
            let mut best_keyframe = 0usize;
            for &sync_sample in &self.stbl.stss {
                if sync_sample <= sample_number {
                    best_keyframe = (sync_sample as usize).saturating_sub(1);
                } else {
                    break;
                }
            }
            best_keyframe
        }
    }

    /// Convert a byte offset to a sample index by searching through sample offsets.
    /// Returns the sample index closest to the given byte offset.
    fn find_sample_at_byte_offset(&self, byte_offset: u64) -> Option<usize> {
        // Build a list of sample offsets and find the one closest to the target
        let sample_count = self.sample_count();
        if sample_count == 0 {
            return None;
        }

        let mut best_sample = 0usize;
        let mut best_distance = u64::MAX;

        for i in 0..sample_count {
            if let Some(offset) = self.sample_offset(i) {
                let distance = offset.abs_diff(byte_offset);

                if distance < best_distance {
                    best_distance = distance;
                    best_sample = i;
                }

                // Early exit if we've passed the target significantly
                if offset > byte_offset && distance > best_distance {
                    break;
                }
            }
        }

        Some(best_sample)
    }

    /// Get the duration in timescale units.
    #[allow(dead_code)]
    fn total_duration(&self) -> i64 {
        self.stbl
            .stts
            .iter()
            .map(|&(count, delta)| (count as i64).saturating_mul(delta as i64))
            .fold(0i64, |acc, x| acc.saturating_add(x))
    }
}

/// MP4 demuxer.
pub struct Mp4Demuxer {
    /// Reader.
    reader: Option<Box<dyn ReadSeek>>,
    /// File type.
    ftyp: Option<FtypAtom>,
    /// Movie header.
    mvhd: Option<MvhdAtom>,
    /// Tracks.
    tracks: Vec<TrackInfo>,
    /// Duration in microseconds.
    duration_us: Option<i64>,
    /// Current track index for interleaved reading.
    #[allow(dead_code)]
    current_track: usize,
    /// Chapters extracted from the file.
    chapters: Option<ChapterList>,
    /// Chapter track references (for text-track-based chapters).
    chapter_track_refs: Vec<ChapterTrackRef>,
}

trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

impl Mp4Demuxer {
    /// Create a new MP4 demuxer.
    pub fn new() -> Self {
        Self {
            reader: None,
            ftyp: None,
            mvhd: None,
            tracks: Vec::new(),
            duration_us: None,
            current_track: 0,
            chapters: None,
            chapter_track_refs: Vec::new(),
        }
    }

    /// Get chapters if present in the file.
    ///
    /// Chapters can come from:
    /// - Nero chapters (chpl atom in udta)
    /// - QuickTime text track chapters (tref/chap reference)
    pub fn chapters(&self) -> Option<&ChapterList> {
        self.chapters.as_ref()
    }

    /// Get mutable reference to chapters.
    pub fn chapters_mut(&mut self) -> Option<&mut ChapterList> {
        self.chapters.as_mut()
    }

    /// Take ownership of chapters, leaving None in the demuxer.
    pub fn take_chapters(&mut self) -> Option<ChapterList> {
        self.chapters.take()
    }

    /// Get chapter track references.
    pub fn chapter_track_refs(&self) -> &[ChapterTrackRef] {
        &self.chapter_track_refs
    }

    /// Parse the file structure.
    fn parse(&mut self) -> Result<()> {
        // Take ownership of reader temporarily to avoid borrow checker issues
        let mut reader = self.reader.take().ok_or(Error::Container("No reader".into()))?;

        // Scan top-level atoms
        reader.seek(SeekFrom::Start(0))?;

        let mut moov_headers = Vec::new();

        while let Some(header) = AtomHeader::read(reader.as_mut())? {
            match &header.atom_type {
                b"ftyp" => {
                    let size = validate_allocation_size(header.content_size())?;
                    let mut content = vec![0u8; size];
                    reader.read_exact(&mut content)?;
                    self.ftyp = Some(FtypAtom::parse(&content)?);
                }
                b"moov" => {
                    moov_headers.push(header);
                }
                b"mdat" => {
                    // Skip media data for now
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
                _ => {
                    // Skip unknown atoms
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
            }
        }

        // Parse moov atoms
        for moov_header in moov_headers {
            self.parse_moov(&mut reader, &moov_header)?;
        }

        // Put reader back
        self.reader = Some(reader);

        // Calculate duration
        if let Some(ref mvhd) = self.mvhd {
            self.duration_us = Some((mvhd.duration_seconds() * 1_000_000.0) as i64);
        }

        Ok(())
    }

    /// Parse moov atom.
    fn parse_moov(&mut self, reader: &mut Box<dyn ReadSeek>, moov_header: &AtomHeader) -> Result<()> {
        let end = moov_header.offset + moov_header.size;

        // Seek to start of moov content
        reader.seek(SeekFrom::Start(moov_header.offset + 8))?;

        let mut trak_headers = Vec::new();
        let mut udta_header: Option<AtomHeader> = None;

        while reader.stream_position()? < end {
            let Some(header) = AtomHeader::read(reader.as_mut())? else {
                break;
            };

            match &header.atom_type {
                b"mvhd" => {
                    let size = validate_allocation_size(header.content_size())?;
                    let mut content = vec![0u8; size];
                    reader.read_exact(&mut content)?;
                    self.mvhd = Some(MvhdAtom::parse(&content)?);
                }
                b"trak" => {
                    trak_headers.push(header);
                }
                b"udta" => {
                    // Store for later parsing (after tracks)
                    udta_header = Some(header.clone());
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
                _ => {
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
            }
        }

        // Parse trak atoms
        for trak_header in trak_headers {
            self.parse_trak(reader, &trak_header)?;
        }

        // Parse udta for Nero chapters (chpl)
        if let Some(udta) = udta_header {
            self.parse_udta(reader, &udta)?;
        }

        Ok(())
    }

    /// Parse udta (user data) atom for chapters.
    fn parse_udta(&mut self, reader: &mut Box<dyn ReadSeek>, udta_header: &AtomHeader) -> Result<()> {
        let end = udta_header.offset + udta_header.size;

        // Seek to start of udta content
        reader.seek(SeekFrom::Start(udta_header.offset + 8))?;

        while reader.stream_position()? < end {
            let Some(header) = AtomHeader::read(reader.as_mut())? else {
                break;
            };

            match &header.atom_type {
                b"chpl" => {
                    // Nero chapters
                    let size = validate_allocation_size(header.content_size())?;
                    let mut content = vec![0u8; size];
                    reader.read_exact(&mut content)?;

                    if let Ok(chapters) = Mp4ChapterReader::parse_chpl(&content) {
                        if !chapters.is_empty() {
                            let mut list = chapters;
                            // Compute end times based on duration
                            if let Some(duration_us) = self.duration_us {
                                let duration_ns = (duration_us as u64) * 1000;
                                list.compute_end_times(Some(duration_ns));
                            }
                            self.chapters = Some(list);
                        }
                    }
                }
                _ => {
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
            }
        }

        Ok(())
    }

    /// Parse trak atom.
    fn parse_trak(&mut self, reader: &mut Box<dyn ReadSeek>, trak_header: &AtomHeader) -> Result<()> {
        let end = trak_header.offset + trak_header.size;

        // Seek to start of trak content
        reader.seek(SeekFrom::Start(trak_header.offset + 8))?;

        let mut tkhd: Option<TkhdAtom> = None;
        let mut mdhd: Option<MdhdAtom> = None;
        let mut hdlr: Option<HdlrAtom> = None;
        let mut stbl: Option<StblInfo> = None;
        let mut extra_data: Option<Vec<u8>> = None;
        let mut tref_data: Option<Vec<u8>> = None;

        while reader.stream_position()? < end {
            let Some(header) = AtomHeader::read(reader.as_mut())? else {
                break;
            };

            match &header.atom_type {
                b"tkhd" => {
                    let size = validate_allocation_size(header.content_size())?;
                    let mut content = vec![0u8; size];
                    reader.read_exact(&mut content)?;
                    tkhd = Some(TkhdAtom::parse(&content)?);
                }
                b"tref" => {
                    // Track reference atom - may contain chapter references
                    let size = validate_allocation_size(header.content_size())?;
                    let mut content = vec![0u8; size];
                    reader.read_exact(&mut content)?;
                    tref_data = Some(content);
                }
                b"mdia" => {
                    // Parse mdia container
                    let mdia_end = header.offset + header.size;
                    while reader.stream_position()? < mdia_end {
                        let Some(mdia_atom) = AtomHeader::read(reader.as_mut())? else {
                            break;
                        };

                        match &mdia_atom.atom_type {
                            b"mdhd" => {
                                let size = validate_allocation_size(mdia_atom.content_size())?;
                                let mut content = vec![0u8; size];
                                reader.read_exact(&mut content)?;
                                mdhd = Some(MdhdAtom::parse(&content)?);
                            }
                            b"hdlr" => {
                                let size = validate_allocation_size(mdia_atom.content_size())?;
                                let mut content = vec![0u8; size];
                                reader.read_exact(&mut content)?;
                                hdlr = Some(HdlrAtom::parse(&content)?);
                            }
                            b"minf" => {
                                // Parse minf to find stbl
                                let minf_end = mdia_atom.offset + mdia_atom.size;
                                while reader.stream_position()? < minf_end {
                                    let Some(minf_atom) = AtomHeader::read(reader.as_mut())? else {
                                        break;
                                    };

                                    if &minf_atom.atom_type == b"stbl" {
                                        stbl = Some(StblInfo::parse(reader.as_mut(), minf_atom.content_size())?);
                                    } else {
                                        reader.seek(SeekFrom::Start(minf_atom.offset + minf_atom.size))?;
                                    }
                                }
                            }
                            _ => {
                                reader.seek(SeekFrom::Start(mdia_atom.offset + mdia_atom.size))?;
                            }
                        }
                    }
                }
                _ => {
                    reader.seek(SeekFrom::Start(header.offset + header.size))?;
                }
            }
        }

        // Parse tref for chapter references
        if let (Some(ref tkhd_atom), Some(ref tref)) = (&tkhd, &tref_data) {
            if let Ok(Some(chapter_ref)) = ChapterTrackRef::parse_tref(tref, tkhd_atom.track_id) {
                self.chapter_track_refs.push(chapter_ref);
            }
        }

        // Build track info
        if let (Some(_tkhd), Some(mdhd), Some(hdlr), Some(stbl)) = (tkhd, mdhd, hdlr, stbl) {
            let track_type = if hdlr.is_video() {
                TrackType::Video
            } else if hdlr.is_audio() {
                TrackType::Audio
            } else {
                TrackType::Unknown
            };

            let (codec_id, video_info, audio_info) = if let Some(entry) = stbl.sample_entries.first() {
                let codec_id = match &entry.entry_type {
                    b"avc1" | b"avc3" => CodecId::H264,
                    b"hev1" | b"hvc1" => CodecId::H265,
                    b"vp09" => CodecId::Vp9,
                    b"av01" => CodecId::Av1,
                    b"mp4a" => CodecId::Aac,
                    b"Opus" => CodecId::Opus,
                    _ => CodecId::Unknown(String::from_utf8_lossy(&entry.entry_type).to_string()),
                };

                let video_info = if track_type == TrackType::Video {
                    Some(VideoStreamInfo {
                        width: entry.width as u32,
                        height: entry.height as u32,
                        frame_rate: None,
                        pixel_aspect_ratio: None,
                        bit_depth: 8,
                    })
                } else {
                    None
                };

                let audio_info = if track_type == TrackType::Audio {
                    Some(AudioStreamInfo {
                        sample_rate: entry.sample_rate >> 16,
                        channels: entry.channel_count as u8,
                        bits_per_sample: entry.sample_size as u8,
                    })
                } else {
                    None
                };

                extra_data = if !entry.codec_data.is_empty() {
                    Some(entry.codec_data.clone())
                } else {
                    None
                };

                (codec_id, video_info, audio_info)
            } else {
                (CodecId::Unknown("unknown".to_string()), None, None)
            };

            let stream = StreamInfo {
                index: self.tracks.len(),
                track_type,
                codec_id,
                time_base: Rational::new(1, mdhd.timescale as i64),
                duration: Some(mdhd.duration as i64),
                extra_data,
                video: video_info,
                audio: audio_info,
            };

            self.tracks.push(TrackInfo {
                stream,
                stbl,
                current_sample: 0,
                timescale: mdhd.timescale,
            });
        }

        Ok(())
    }

    /// Find track with next sample to read.
    fn find_next_track(&self) -> Option<usize> {
        let mut best_track = None;
        let mut best_dts = i64::MAX;

        for (i, track) in self.tracks.iter().enumerate() {
            if track.current_sample < track.sample_count() {
                let dts = track.sample_timestamp(track.current_sample);
                // Use saturating arithmetic to prevent overflow
                let dts_us = dts.saturating_mul(1_000_000) / (track.timescale as i64).max(1);

                if dts_us < best_dts {
                    best_dts = dts_us;
                    best_track = Some(i);
                }
            }
        }

        best_track
    }
}

impl Default for Mp4Demuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for Mp4Demuxer {
    fn open<R: Read + Seek + Send + 'static>(&mut self, reader: R) -> Result<()> {
        self.reader = Some(Box::new(reader));
        self.parse()
    }

    fn format_name(&self) -> &str {
        "mp4"
    }

    fn duration(&self) -> Option<i64> {
        self.duration_us
    }

    fn num_streams(&self) -> usize {
        self.tracks.len()
    }

    fn stream_info(&self, index: usize) -> Option<&StreamInfo> {
        self.tracks.get(index).map(|t| &t.stream)
    }

    fn read_packet(&mut self) -> Result<Option<Packet<'static>>> {
        let track_idx = match self.find_next_track() {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let track = &mut self.tracks[track_idx];
        let sample_idx = track.current_sample;

        let offset = track
            .sample_offset(sample_idx)
            .ok_or_else(|| Error::Container("Invalid sample offset".into()))?;

        let size = track
            .sample_size(sample_idx)
            .ok_or_else(|| Error::Container("Invalid sample size".into()))?;

        // Validate sample size to prevent excessive allocation
        if size > MAX_SAMPLE_SIZE {
            return Err(Error::Container(format!(
                "Sample size {} exceeds maximum allowed size {}",
                size, MAX_SAMPLE_SIZE
            ).into()));
        }

        // Read sample data
        let reader = self.reader.as_mut().ok_or(Error::Container("No reader".into()))?;
        reader.seek(SeekFrom::Start(offset))?;

        let mut data = vec![0u8; size as usize];
        reader.read_exact(&mut data)?;

        // Build packet
        let dts_value = track.sample_timestamp(sample_idx);
        let cts_offset = track.composition_offset(sample_idx);
        let pts_value = dts_value + cts_offset;
        let keyframe = track.is_keyframe(sample_idx);
        let time_base = TimeBase::new(1, track.timescale as i64);

        let mut packet = Packet::new(data);
        packet.stream_index = track_idx as u32;
        packet.pts = Timestamp::new(pts_value, time_base);
        packet.dts = Timestamp::new(dts_value, time_base);
        packet.set_keyframe(keyframe);

        track.current_sample += 1;

        Ok(Some(packet))
    }

    fn seek_to(&mut self, target: SeekTarget, mode: SeekMode) -> Result<SeekResult> {
        // Handle edge case: no tracks
        if self.tracks.is_empty() {
            return Ok(SeekResult {
                timestamp_us: 0,
                is_keyframe: true,
                sample_indices: vec![],
            });
        }

        // Determine the target sample for each track based on the seek target type
        let mut sample_indices: Vec<usize> = Vec::with_capacity(self.tracks.len());
        let mut result_timestamp_us = 0i64;
        let mut result_is_keyframe = true;

        match target {
            SeekTarget::Timestamp(timestamp_us) => {
                // Handle edge cases: seeking before start or past end
                let clamped_timestamp = if timestamp_us < 0 {
                    0i64
                } else if let Some(duration) = self.duration_us {
                    timestamp_us.min(duration)
                } else {
                    timestamp_us
                };

                for track in &mut self.tracks {
                    // Convert timestamp from microseconds to track timescale
                    let timescale = (track.timescale as i64).max(1);
                    let timestamp_in_timescale =
                        clamped_timestamp.saturating_mul(timescale) / 1_000_000;

                    // Find the sample at this timestamp
                    let (sample_idx, sample_ts) =
                        track.find_sample_at_timestamp(timestamp_in_timescale);

                    // Clamp sample index to valid range
                    let sample_count = track.sample_count();
                    let clamped_sample_idx = sample_idx.min(sample_count.saturating_sub(1));

                    // Find the appropriate keyframe based on seek mode
                    let final_sample = match mode {
                        SeekMode::Backward => {
                            track.find_nearest_keyframe(clamped_sample_idx, false)
                        }
                        SeekMode::Forward => track.find_nearest_keyframe(clamped_sample_idx, true),
                        SeekMode::Exact => clamped_sample_idx,
                    };

                    // Clamp final sample to valid range
                    let final_sample = final_sample.min(sample_count.saturating_sub(1));

                    track.current_sample = final_sample;
                    sample_indices.push(final_sample);

                    // Calculate the actual timestamp we landed on
                    let landed_ts = track.sample_timestamp(final_sample);
                    let landed_us = landed_ts.saturating_mul(1_000_000) / timescale;

                    // Use the video track's timestamp as the result (or first track)
                    if track.stream.track_type == TrackType::Video || result_timestamp_us == 0 {
                        result_timestamp_us = landed_us;
                        result_is_keyframe = track.is_keyframe(final_sample);
                    }

                    // For exact mode on non-video tracks, also update timestamp
                    if mode == SeekMode::Exact {
                        result_timestamp_us =
                            result_timestamp_us.max(sample_ts.saturating_mul(1_000_000) / timescale);
                    }
                }
            }

            SeekTarget::ByteOffset(byte_offset) => {
                // For byte offset seeking, we find the closest sample to the offset
                for track in &mut self.tracks {
                    let sample_idx = track
                        .find_sample_at_byte_offset(byte_offset)
                        .unwrap_or(0);

                    // Apply keyframe seeking based on mode
                    let final_sample = match mode {
                        SeekMode::Backward => track.find_nearest_keyframe(sample_idx, false),
                        SeekMode::Forward => track.find_nearest_keyframe(sample_idx, true),
                        SeekMode::Exact => sample_idx,
                    };

                    // Clamp to valid range
                    let sample_count = track.sample_count();
                    let final_sample = final_sample.min(sample_count.saturating_sub(1));

                    track.current_sample = final_sample;
                    sample_indices.push(final_sample);

                    // Calculate timestamp
                    let timescale = (track.timescale as i64).max(1);
                    let landed_ts = track.sample_timestamp(final_sample);
                    let landed_us = landed_ts.saturating_mul(1_000_000) / timescale;

                    if track.stream.track_type == TrackType::Video || result_timestamp_us == 0 {
                        result_timestamp_us = landed_us;
                        result_is_keyframe = track.is_keyframe(final_sample);
                    }
                }
            }

            SeekTarget::Sample {
                track_index,
                sample_number,
            } => {
                // Validate track index
                if track_index >= self.tracks.len() {
                    return Err(Error::Container(
                        format!("Invalid track index: {}", track_index).into(),
                    ));
                }

                // First, seek the specified track to the sample
                let target_track = &mut self.tracks[track_index];
                let sample_count = target_track.sample_count();

                // Clamp sample number to valid range
                let clamped_sample = sample_number.min(sample_count.saturating_sub(1));

                // Apply keyframe seeking based on mode
                let final_sample = match mode {
                    SeekMode::Backward => target_track.find_nearest_keyframe(clamped_sample, false),
                    SeekMode::Forward => target_track.find_nearest_keyframe(clamped_sample, true),
                    SeekMode::Exact => clamped_sample,
                };

                let final_sample = final_sample.min(sample_count.saturating_sub(1));
                let timescale = (target_track.timescale as i64).max(1);
                let target_ts = target_track.sample_timestamp(final_sample);
                let target_us = target_ts.saturating_mul(1_000_000) / timescale;

                result_timestamp_us = target_us;
                result_is_keyframe = target_track.is_keyframe(final_sample);

                // Now seek all tracks to the same timestamp
                for (idx, track) in self.tracks.iter_mut().enumerate() {
                    if idx == track_index {
                        track.current_sample = final_sample;
                        sample_indices.push(final_sample);
                    } else {
                        // Convert the target timestamp to this track's timescale
                        let track_timescale = (track.timescale as i64).max(1);
                        let ts_in_track = target_us.saturating_mul(track_timescale) / 1_000_000;

                        let (sample_idx, _) = track.find_sample_at_timestamp(ts_in_track);
                        let sample_count = track.sample_count();
                        let clamped = sample_idx.min(sample_count.saturating_sub(1));

                        // Apply keyframe seeking
                        let final_sample = match mode {
                            SeekMode::Backward => track.find_nearest_keyframe(clamped, false),
                            SeekMode::Forward => track.find_nearest_keyframe(clamped, true),
                            SeekMode::Exact => clamped,
                        };

                        let final_sample = final_sample.min(sample_count.saturating_sub(1));
                        track.current_sample = final_sample;
                        sample_indices.push(final_sample);
                    }
                }
            }
        }

        Ok(SeekResult {
            timestamp_us: result_timestamp_us,
            is_keyframe: result_is_keyframe,
            sample_indices,
        })
    }

    fn position(&self) -> Option<u64> {
        // We can't call stream_position on a trait object directly,
        // so we return the approximate position based on current sample
        if self.reader.is_some() {
            if let Some(track) = self.tracks.first() {
                track.sample_offset(track.current_sample)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn close(&mut self) {
        self.reader = None;
        self.tracks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a mock StblInfo with sample timing data for testing.
    fn create_test_stbl(
        sample_count: usize,
        delta: u32,
        keyframe_interval: usize,
    ) -> StblInfo {
        let mut stbl = StblInfo::default();

        // Create sample sizes (all same size for simplicity)
        stbl.sample_sizes = vec![1000; sample_count];

        // Create stts entry (all samples have same delta)
        stbl.stts = vec![(sample_count as u32, delta)];

        // Create stsc entry (one sample per chunk)
        stbl.stsc = vec![(1, 1, 1)];

        // Create chunk offsets
        stbl.chunk_offsets = (0..sample_count)
            .map(|i| (i * 1000) as u64 + 1000) // Start at offset 1000
            .collect();

        // Create sync samples (keyframes)
        if keyframe_interval > 0 {
            stbl.stss = (1..=sample_count)
                .filter(|&i| (i - 1) % keyframe_interval == 0)
                .map(|i| i as u32)
                .collect();
        }
        // If keyframe_interval is 0, leave stss empty (all frames are keyframes)

        stbl
    }

    /// Create a TrackInfo for testing.
    fn create_test_track(
        sample_count: usize,
        delta: u32,
        timescale: u32,
        keyframe_interval: usize,
    ) -> TrackInfo {
        TrackInfo {
            stream: StreamInfo {
                index: 0,
                track_type: TrackType::Video,
                codec_id: CodecId::H264,
                time_base: Rational::new(1, timescale as i64),
                duration: Some((sample_count as i64) * (delta as i64)),
                extra_data: None,
                video: None,
                audio: None,
            },
            stbl: create_test_stbl(sample_count, delta, keyframe_interval),
            current_sample: 0,
            timescale,
        }
    }

    #[test]
    fn test_track_sample_timestamp() {
        let track = create_test_track(100, 1000, 30000, 30);

        // First sample should be at timestamp 0
        assert_eq!(track.sample_timestamp(0), 0);

        // 10th sample should be at 10 * 1000 = 10000
        assert_eq!(track.sample_timestamp(10), 10000);

        // 50th sample should be at 50 * 1000 = 50000
        assert_eq!(track.sample_timestamp(50), 50000);
    }

    #[test]
    fn test_track_is_keyframe() {
        let track = create_test_track(100, 1000, 30000, 30);

        // Sample 0 (sample number 1) should be a keyframe
        assert!(track.is_keyframe(0));

        // Sample 29 should not be a keyframe
        assert!(!track.is_keyframe(29));

        // Sample 30 (sample number 31) should be a keyframe
        assert!(track.is_keyframe(30));

        // Sample 60 should be a keyframe
        assert!(track.is_keyframe(60));
    }

    #[test]
    fn test_track_all_keyframes_when_no_stss() {
        let track = create_test_track(100, 1000, 30000, 0);

        // All samples should be keyframes when stss is empty
        assert!(track.is_keyframe(0));
        assert!(track.is_keyframe(15));
        assert!(track.is_keyframe(50));
        assert!(track.is_keyframe(99));
    }

    #[test]
    fn test_find_sample_at_timestamp() {
        let track = create_test_track(100, 1000, 30000, 30);

        // Timestamp 0 should return sample 0
        let (idx, ts) = track.find_sample_at_timestamp(0);
        assert_eq!(idx, 0);
        assert_eq!(ts, 0);

        // Timestamp 10000 should return sample 10
        let (idx, ts) = track.find_sample_at_timestamp(10000);
        assert_eq!(idx, 10);
        assert_eq!(ts, 10000);

        // Timestamp 10500 (between samples) should return sample 10
        let (idx, ts) = track.find_sample_at_timestamp(10500);
        assert_eq!(idx, 10);
        assert_eq!(ts, 10000);

        // Timestamp past the end should return the last sample
        let (idx, _) = track.find_sample_at_timestamp(1_000_000);
        assert_eq!(idx, 99);
    }

    #[test]
    fn test_find_nearest_keyframe_backward() {
        let track = create_test_track(100, 1000, 30000, 30);

        // At keyframe, should return same sample
        assert_eq!(track.find_nearest_keyframe(0, false), 0);
        assert_eq!(track.find_nearest_keyframe(30, false), 30);

        // Between keyframes, should return previous keyframe
        assert_eq!(track.find_nearest_keyframe(15, false), 0);
        assert_eq!(track.find_nearest_keyframe(45, false), 30);
        assert_eq!(track.find_nearest_keyframe(75, false), 60);
    }

    #[test]
    fn test_find_nearest_keyframe_forward() {
        let track = create_test_track(100, 1000, 30000, 30);

        // At keyframe, should return same sample
        assert_eq!(track.find_nearest_keyframe(0, true), 0);
        assert_eq!(track.find_nearest_keyframe(30, true), 30);

        // Between keyframes, should return next keyframe
        assert_eq!(track.find_nearest_keyframe(15, true), 30);
        assert_eq!(track.find_nearest_keyframe(45, true), 60);
    }

    #[test]
    fn test_find_sample_at_byte_offset() {
        let track = create_test_track(100, 1000, 30000, 30);

        // Offset 1000 is sample 0
        assert_eq!(track.find_sample_at_byte_offset(1000), Some(0));

        // Offset 1500 is closer to sample 0
        assert_eq!(track.find_sample_at_byte_offset(1500), Some(0));

        // Offset 2000 is sample 1
        assert_eq!(track.find_sample_at_byte_offset(2000), Some(1));

        // Offset 50000 is closer to sample 49
        assert_eq!(track.find_sample_at_byte_offset(50000), Some(49));
    }

    #[test]
    fn test_seek_target_constructors() {
        let ts = SeekTarget::from_micros(1_000_000);
        assert_eq!(ts, SeekTarget::Timestamp(1_000_000));

        let ts = SeekTarget::from_millis(1000);
        assert_eq!(ts, SeekTarget::Timestamp(1_000_000));

        let ts = SeekTarget::from_secs(1.0);
        assert_eq!(ts, SeekTarget::Timestamp(1_000_000));

        let ts = SeekTarget::from_byte_offset(12345);
        assert_eq!(ts, SeekTarget::ByteOffset(12345));

        let ts = SeekTarget::from_sample(0, 50);
        assert_eq!(
            ts,
            SeekTarget::Sample {
                track_index: 0,
                sample_number: 50
            }
        );
    }

    #[test]
    fn test_seek_mode_default() {
        let mode = SeekMode::default();
        assert_eq!(mode, SeekMode::Backward);
    }

    #[test]
    fn test_total_duration() {
        let track = create_test_track(100, 1000, 30000, 30);
        assert_eq!(track.total_duration(), 100_000);
    }

    #[test]
    fn test_sample_count() {
        let track = create_test_track(100, 1000, 30000, 30);
        assert_eq!(track.sample_count(), 100);
    }

    // Integration tests for Mp4Demuxer seeking would require actual MP4 files
    // or a more sophisticated mock. The following tests verify the edge case handling.

    #[test]
    fn test_demuxer_seek_to_empty_tracks() {
        let mut demuxer = Mp4Demuxer::new();
        // Demuxer without tracks should handle seek gracefully
        let result = demuxer.seek_to(SeekTarget::Timestamp(1_000_000), SeekMode::Backward);
        assert!(result.is_ok());
        let seek_result = result.unwrap();
        assert_eq!(seek_result.timestamp_us, 0);
        assert!(seek_result.is_keyframe);
        assert!(seek_result.sample_indices.is_empty());
    }

    #[test]
    fn test_demuxer_position_without_reader() {
        let demuxer = Mp4Demuxer::new();
        assert!(demuxer.position().is_none());
    }

    #[test]
    fn test_demuxer_can_seek() {
        let demuxer = Mp4Demuxer::new();
        assert!(demuxer.can_seek());
    }

    #[test]
    fn test_seek_result_fields() {
        let result = SeekResult {
            timestamp_us: 1_000_000,
            is_keyframe: true,
            sample_indices: vec![30, 45],
        };

        assert_eq!(result.timestamp_us, 1_000_000);
        assert!(result.is_keyframe);
        assert_eq!(result.sample_indices.len(), 2);
        assert_eq!(result.sample_indices[0], 30);
        assert_eq!(result.sample_indices[1], 45);
    }

    // Test edge cases for timestamp clamping
    #[test]
    fn test_find_sample_at_negative_timestamp() {
        let track = create_test_track(100, 1000, 30000, 30);

        // Negative timestamp should still work (will compare against 0)
        let (idx, ts) = track.find_sample_at_timestamp(-1000);
        // Since -1000 < 0, first iteration finds sample_ts=0 > -1000,
        // so it returns the previous sample which is saturating_sub(1) from 0 = 0
        assert_eq!(idx, 0);
        assert_eq!(ts, 0);
    }

    #[test]
    fn test_keyframe_lookup_at_boundaries() {
        let track = create_test_track(100, 1000, 30000, 30);

        // First sample
        assert_eq!(track.find_nearest_keyframe(0, false), 0);
        assert_eq!(track.find_nearest_keyframe(0, true), 0);

        // Last sample (99)
        // Backward should find last keyframe at 90
        assert_eq!(track.find_nearest_keyframe(99, false), 90);
        // Forward should return 90 (last keyframe) since there's none after 99
        assert_eq!(track.find_nearest_keyframe(99, true), 90);
    }

    #[test]
    fn test_stts_with_multiple_entries() {
        let mut track = create_test_track(100, 1000, 30000, 30);

        // Modify stts to have multiple entries
        track.stbl.stts = vec![
            (50, 1000), // First 50 samples have delta 1000
            (50, 2000), // Next 50 samples have delta 2000
        ];

        // Sample 0: timestamp 0
        assert_eq!(track.sample_timestamp(0), 0);

        // Sample 25: timestamp 25000
        assert_eq!(track.sample_timestamp(25), 25000);

        // Sample 50: timestamp 50000 (end of first entry)
        assert_eq!(track.sample_timestamp(50), 50000);

        // Sample 75: timestamp 50000 + 25*2000 = 100000
        assert_eq!(track.sample_timestamp(75), 100000);
    }
}
