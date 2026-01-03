//! MP4 atom (box) parsing and writing.

// Allow dead code for atom structures that are parsed but not yet fully utilized.
// These are part of the public API and will be used as the muxer is developed.
#![allow(dead_code)]

use super::{read_u32_be, read_u64_be, write_u32_be, write_u64_be};
use transcode_core::error::{Error, Result};
use std::io::{Read, Seek, SeekFrom, Write};

/// Atom header.
#[derive(Debug, Clone)]
pub struct AtomHeader {
    /// Atom type (4 bytes).
    pub atom_type: [u8; 4],
    /// Atom size (including header).
    pub size: u64,
    /// Header size (8 or 16 bytes).
    pub header_size: u8,
    /// Offset in file.
    pub offset: u64,
}

impl AtomHeader {
    /// Read atom header from reader.
    pub fn read<R: Read + Seek + ?Sized>(reader: &mut R) -> Result<Option<Self>> {
        let offset = reader.stream_position()?;

        let mut header = [0u8; 8];
        match reader.read_exact(&mut header) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let atom_type = [header[4], header[5], header[6], header[7]];

        let (size, header_size) = if size == 1 {
            // Extended size
            let mut ext_size = [0u8; 8];
            reader.read_exact(&mut ext_size)?;
            (u64::from_be_bytes(ext_size), 16)
        } else if size == 0 {
            // Size extends to end of file
            let current = reader.stream_position()?;
            let end = reader.seek(SeekFrom::End(0))?;
            reader.seek(SeekFrom::Start(current))?;
            (end - offset, 8)
        } else {
            (size as u64, 8)
        };

        Ok(Some(Self {
            atom_type,
            size,
            header_size,
            offset,
        }))
    }

    /// Write atom header.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        if self.size > u32::MAX as u64 {
            // Extended size
            writer.write_all(&[0, 0, 0, 1])?;
            writer.write_all(&self.atom_type)?;
            writer.write_all(&write_u64_be(self.size))?;
        } else {
            writer.write_all(&write_u32_be(self.size as u32))?;
            writer.write_all(&self.atom_type)?;
        }
        Ok(())
    }

    /// Get content size (size - header).
    pub fn content_size(&self) -> u64 {
        self.size.saturating_sub(self.header_size as u64)
    }

    /// Get content offset.
    pub fn content_offset(&self) -> u64 {
        self.offset + self.header_size as u64
    }

    /// Check if this is a container atom.
    pub fn is_container(&self) -> bool {
        matches!(
            &self.atom_type,
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"dinf" | b"udta"
        )
    }
}

/// File type atom (ftyp).
#[derive(Debug, Clone)]
pub struct FtypAtom {
    /// Major brand.
    pub major_brand: [u8; 4],
    /// Minor version.
    pub minor_version: u32,
    /// Compatible brands.
    pub compatible_brands: Vec<[u8; 4]>,
}

impl FtypAtom {
    /// Parse ftyp atom.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::Container("ftyp atom too short".into()));
        }

        let major_brand = [data[0], data[1], data[2], data[3]];
        let minor_version = read_u32_be(&data[4..8])?;

        let mut compatible_brands = Vec::new();
        let mut offset = 8;
        while offset + 4 <= data.len() {
            let brand = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
            compatible_brands.push(brand);
            offset += 4;
        }

        Ok(Self {
            major_brand,
            minor_version,
            compatible_brands,
        })
    }

    /// Write ftyp atom.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        let size = 8 + 8 + self.compatible_brands.len() * 4;

        writer.write_all(&write_u32_be(size as u32))?;
        writer.write_all(b"ftyp")?;
        writer.write_all(&self.major_brand)?;
        writer.write_all(&write_u32_be(self.minor_version))?;

        for brand in &self.compatible_brands {
            writer.write_all(brand)?;
        }

        Ok(())
    }
}

/// Movie header atom (mvhd).
#[derive(Debug, Clone)]
pub struct MvhdAtom {
    /// Version.
    pub version: u8,
    /// Flags.
    pub flags: u32,
    /// Creation time.
    pub creation_time: u64,
    /// Modification time.
    pub modification_time: u64,
    /// Timescale (units per second).
    pub timescale: u32,
    /// Duration.
    pub duration: u64,
    /// Rate (fixed-point 16.16).
    pub rate: u32,
    /// Volume (fixed-point 8.8).
    pub volume: u16,
    /// Next track ID.
    pub next_track_id: u32,
}

impl MvhdAtom {
    /// Parse mvhd atom.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Container("mvhd atom empty".into()));
        }

        let version = data[0];
        let flags = read_u32_be(&data[0..4])? & 0x00FFFFFF;

        let (creation_time, modification_time, timescale, duration, offset) = if version == 1 {
            if data.len() < 32 {
                return Err(Error::Container("mvhd v1 atom too short".into()));
            }
            (
                read_u64_be(&data[4..12])?,
                read_u64_be(&data[12..20])?,
                read_u32_be(&data[20..24])?,
                read_u64_be(&data[24..32])?,
                32,
            )
        } else {
            if data.len() < 20 {
                return Err(Error::Container("mvhd v0 atom too short".into()));
            }
            (
                read_u32_be(&data[4..8])? as u64,
                read_u32_be(&data[8..12])? as u64,
                read_u32_be(&data[12..16])?,
                read_u32_be(&data[16..20])? as u64,
                20,
            )
        };

        let rate = read_u32_be(&data[offset..offset + 4])?;
        let volume = u16::from_be_bytes([data[offset + 4], data[offset + 5]]);

        // Skip reserved and matrix
        let next_track_offset = offset + 76;
        let next_track_id = if data.len() >= next_track_offset + 4 {
            read_u32_be(&data[next_track_offset..next_track_offset + 4])?
        } else {
            1
        };

        Ok(Self {
            version,
            flags,
            creation_time,
            modification_time,
            timescale,
            duration,
            rate,
            volume,
            next_track_id,
        })
    }

    /// Get duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.timescale > 0 {
            self.duration as f64 / self.timescale as f64
        } else {
            0.0
        }
    }
}

/// Track header atom (tkhd).
#[derive(Debug, Clone)]
pub struct TkhdAtom {
    /// Version.
    pub version: u8,
    /// Flags.
    pub flags: u32,
    /// Track ID.
    pub track_id: u32,
    /// Duration.
    pub duration: u64,
    /// Width (fixed-point 16.16).
    pub width: u32,
    /// Height (fixed-point 16.16).
    pub height: u32,
}

impl TkhdAtom {
    /// Parse tkhd atom.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Container("tkhd atom empty".into()));
        }

        let version = data[0];
        let flags = read_u32_be(&data[0..4])? & 0x00FFFFFF;

        let (track_id, duration, offset) = if version == 1 {
            if data.len() < 32 {
                return Err(Error::Container("tkhd v1 atom too short".into()));
            }
            (
                read_u32_be(&data[20..24])?,
                read_u64_be(&data[28..36])?,
                36,
            )
        } else {
            if data.len() < 20 {
                return Err(Error::Container("tkhd v0 atom too short".into()));
            }
            (
                read_u32_be(&data[12..16])?,
                read_u32_be(&data[20..24])? as u64,
                24,
            )
        };

        // Skip to width/height (offset + 52 for matrix and other fields)
        let dim_offset = offset + 52;
        let (width, height) = if data.len() >= dim_offset + 8 {
            (
                read_u32_be(&data[dim_offset..dim_offset + 4])?,
                read_u32_be(&data[dim_offset + 4..dim_offset + 8])?,
            )
        } else {
            (0, 0)
        };

        Ok(Self {
            version,
            flags,
            track_id,
            duration,
            width,
            height,
        })
    }

    /// Get width in pixels.
    pub fn width_pixels(&self) -> u32 {
        self.width >> 16
    }

    /// Get height in pixels.
    pub fn height_pixels(&self) -> u32 {
        self.height >> 16
    }

    /// Check if track is enabled.
    pub fn is_enabled(&self) -> bool {
        (self.flags & 0x01) != 0
    }
}

/// Media header atom (mdhd).
#[derive(Debug, Clone)]
pub struct MdhdAtom {
    /// Version.
    pub version: u8,
    /// Timescale.
    pub timescale: u32,
    /// Duration.
    pub duration: u64,
    /// Language code.
    pub language: u16,
}

impl MdhdAtom {
    /// Parse mdhd atom.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Container("mdhd atom empty".into()));
        }

        let version = data[0];

        let (timescale, duration, lang_offset) = if version == 1 {
            if data.len() < 28 {
                return Err(Error::Container("mdhd v1 atom too short".into()));
            }
            (
                read_u32_be(&data[20..24])?,
                read_u64_be(&data[24..32])?,
                32,
            )
        } else {
            if data.len() < 20 {
                return Err(Error::Container("mdhd v0 atom too short".into()));
            }
            (
                read_u32_be(&data[12..16])?,
                read_u32_be(&data[16..20])? as u64,
                20,
            )
        };

        let language = if data.len() >= lang_offset + 2 {
            u16::from_be_bytes([data[lang_offset], data[lang_offset + 1]])
        } else {
            0
        };

        Ok(Self {
            version,
            timescale,
            duration,
            language,
        })
    }
}

/// Handler reference atom (hdlr).
#[derive(Debug, Clone)]
pub struct HdlrAtom {
    /// Handler type.
    pub handler_type: [u8; 4],
    /// Name.
    pub name: String,
}

impl HdlrAtom {
    /// Parse hdlr atom.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(Error::Container("hdlr atom too short".into()));
        }

        let handler_type = [data[8], data[9], data[10], data[11]];

        // Name starts at offset 24
        let name = if data.len() > 24 {
            String::from_utf8_lossy(&data[24..]).trim_end_matches('\0').to_string()
        } else {
            String::new()
        };

        Ok(Self { handler_type, name })
    }

    /// Check if this is a video handler.
    pub fn is_video(&self) -> bool {
        &self.handler_type == b"vide"
    }

    /// Check if this is an audio handler.
    pub fn is_audio(&self) -> bool {
        &self.handler_type == b"soun"
    }
}

/// Sample table box (stbl) contents.
#[derive(Debug, Clone, Default)]
pub struct StblInfo {
    /// Sample descriptions.
    pub sample_entries: Vec<SampleEntry>,
    /// Sample sizes.
    pub sample_sizes: Vec<u32>,
    /// Sample to chunk mapping.
    pub stsc: Vec<(u32, u32, u32)>, // (first_chunk, samples_per_chunk, description_index)
    /// Chunk offsets.
    pub chunk_offsets: Vec<u64>,
    /// Time to sample mapping.
    pub stts: Vec<(u32, u32)>, // (sample_count, sample_delta)
    /// Composition time offsets.
    pub ctts: Vec<(u32, i32)>, // (sample_count, offset)
    /// Sync samples (keyframes).
    pub stss: Vec<u32>,
}

/// Sample entry description.
#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Entry type.
    pub entry_type: [u8; 4],
    /// Data reference index.
    pub data_reference_index: u16,
    /// Video width.
    pub width: u16,
    /// Video height.
    pub height: u16,
    /// Audio sample rate (fixed-point 16.16).
    pub sample_rate: u32,
    /// Audio channel count.
    pub channel_count: u16,
    /// Audio sample size.
    pub sample_size: u16,
    /// Codec-specific data.
    pub codec_data: Vec<u8>,
}

impl StblInfo {
    /// Parse stbl contents.
    pub fn parse<R: Read + Seek + ?Sized>(reader: &mut R, stbl_size: u64) -> Result<Self> {
        let start = reader.stream_position()?;
        let end = start + stbl_size;
        let mut info = StblInfo::default();

        while reader.stream_position()? < end {
            let Some(header) = AtomHeader::read(reader)? else {
                break;
            };

            let content_size = header.content_size() as usize;
            let mut content = vec![0u8; content_size];
            reader.read_exact(&mut content)?;

            match &header.atom_type {
                b"stsd" => {
                    info.sample_entries = Self::parse_stsd(&content)?;
                }
                b"stsz" | b"stz2" => {
                    info.sample_sizes = Self::parse_stsz(&content)?;
                }
                b"stsc" => {
                    info.stsc = Self::parse_stsc(&content)?;
                }
                b"stco" => {
                    info.chunk_offsets = Self::parse_stco(&content)?;
                }
                b"co64" => {
                    info.chunk_offsets = Self::parse_co64(&content)?;
                }
                b"stts" => {
                    info.stts = Self::parse_stts(&content)?;
                }
                b"ctts" => {
                    info.ctts = Self::parse_ctts(&content)?;
                }
                b"stss" => {
                    info.stss = Self::parse_stss(&content)?;
                }
                _ => {}
            }
        }

        Ok(info)
    }

    fn parse_stsd(data: &[u8]) -> Result<Vec<SampleEntry>> {
        if data.len() < 8 {
            return Err(Error::Container("stsd too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])?;
        let mut entries = Vec::with_capacity(entry_count as usize);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 8 > data.len() {
                break;
            }

            let entry_size = read_u32_be(&data[offset..offset + 4])? as usize;
            let entry_type = [
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ];

            if offset + entry_size > data.len() {
                break;
            }

            let entry_data = &data[offset..offset + entry_size];

            let entry = if entry_type[0..4] == *b"avc1"
                || entry_type[0..4] == *b"hev1"
                || entry_type[0..4] == *b"hvc1"
                || entry_type[0..4] == *b"vp09"
            {
                // Video sample entry
                Self::parse_video_entry(entry_data)?
            } else if entry_type[0..4] == *b"mp4a" || entry_type[0..4] == *b"Opus" {
                // Audio sample entry
                Self::parse_audio_entry(entry_data)?
            } else {
                SampleEntry {
                    entry_type,
                    data_reference_index: 0,
                    width: 0,
                    height: 0,
                    sample_rate: 0,
                    channel_count: 0,
                    sample_size: 0,
                    codec_data: entry_data[16..].to_vec(),
                }
            };

            entries.push(entry);
            offset += entry_size;
        }

        Ok(entries)
    }

    fn parse_video_entry(data: &[u8]) -> Result<SampleEntry> {
        if data.len() < 78 {
            return Err(Error::Container("Video sample entry too short".into()));
        }

        let entry_type = [data[4], data[5], data[6], data[7]];
        let data_reference_index = u16::from_be_bytes([data[14], data[15]]);
        let width = u16::from_be_bytes([data[32], data[33]]);
        let height = u16::from_be_bytes([data[34], data[35]]);

        // Codec-specific data starts at offset 78
        let codec_data = if data.len() > 78 { data[78..].to_vec() } else { Vec::new() };

        Ok(SampleEntry {
            entry_type,
            data_reference_index,
            width,
            height,
            sample_rate: 0,
            channel_count: 0,
            sample_size: 0,
            codec_data,
        })
    }

    fn parse_audio_entry(data: &[u8]) -> Result<SampleEntry> {
        if data.len() < 36 {
            return Err(Error::Container("Audio sample entry too short".into()));
        }

        let entry_type = [data[4], data[5], data[6], data[7]];
        let data_reference_index = u16::from_be_bytes([data[14], data[15]]);
        let channel_count = u16::from_be_bytes([data[24], data[25]]);
        let sample_size = u16::from_be_bytes([data[26], data[27]]);
        let sample_rate = read_u32_be(&data[32..36])?;

        let codec_data = if data.len() > 36 { data[36..].to_vec() } else { Vec::new() };

        Ok(SampleEntry {
            entry_type,
            data_reference_index,
            width: 0,
            height: 0,
            sample_rate,
            channel_count,
            sample_size,
            codec_data,
        })
    }

    fn parse_stsz(data: &[u8]) -> Result<Vec<u32>> {
        if data.len() < 12 {
            return Err(Error::Container("stsz too short".into()));
        }

        let sample_size = read_u32_be(&data[4..8])?;
        let sample_count = read_u32_be(&data[8..12])? as usize;

        if sample_size != 0 {
            // Constant sample size
            Ok(vec![sample_size; sample_count])
        } else {
            // Variable sample sizes
            let mut sizes = Vec::with_capacity(sample_count);
            let mut offset = 12;

            for _ in 0..sample_count {
                if offset + 4 > data.len() {
                    break;
                }
                sizes.push(read_u32_be(&data[offset..offset + 4])?);
                offset += 4;
            }

            Ok(sizes)
        }
    }

    fn parse_stsc(data: &[u8]) -> Result<Vec<(u32, u32, u32)>> {
        if data.len() < 8 {
            return Err(Error::Container("stsc too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 12 > data.len() {
                break;
            }
            let first_chunk = read_u32_be(&data[offset..offset + 4])?;
            let samples_per_chunk = read_u32_be(&data[offset + 4..offset + 8])?;
            let description_index = read_u32_be(&data[offset + 8..offset + 12])?;
            entries.push((first_chunk, samples_per_chunk, description_index));
            offset += 12;
        }

        Ok(entries)
    }

    fn parse_stco(data: &[u8]) -> Result<Vec<u64>> {
        if data.len() < 8 {
            return Err(Error::Container("stco too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut offsets = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 4 > data.len() {
                break;
            }
            offsets.push(read_u32_be(&data[offset..offset + 4])? as u64);
            offset += 4;
        }

        Ok(offsets)
    }

    fn parse_co64(data: &[u8]) -> Result<Vec<u64>> {
        if data.len() < 8 {
            return Err(Error::Container("co64 too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut offsets = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 8 > data.len() {
                break;
            }
            offsets.push(read_u64_be(&data[offset..offset + 8])?);
            offset += 8;
        }

        Ok(offsets)
    }

    fn parse_stts(data: &[u8]) -> Result<Vec<(u32, u32)>> {
        if data.len() < 8 {
            return Err(Error::Container("stts too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 8 > data.len() {
                break;
            }
            let sample_count = read_u32_be(&data[offset..offset + 4])?;
            let sample_delta = read_u32_be(&data[offset + 4..offset + 8])?;
            entries.push((sample_count, sample_delta));
            offset += 8;
        }

        Ok(entries)
    }

    fn parse_ctts(data: &[u8]) -> Result<Vec<(u32, i32)>> {
        if data.len() < 8 {
            return Err(Error::Container("ctts too short".into()));
        }

        let version = data[0];
        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 8 > data.len() {
                break;
            }
            let sample_count = read_u32_be(&data[offset..offset + 4])?;
            // Note: Version 1 uses signed 32-bit, version 0 uses unsigned 32-bit,
            // but both are read the same way for this purpose
            let _ = version; // Acknowledge version for future compatibility
            let sample_offset = read_u32_be(&data[offset + 4..offset + 8])? as i32;
            entries.push((sample_count, sample_offset));
            offset += 8;
        }

        Ok(entries)
    }

    fn parse_stss(data: &[u8]) -> Result<Vec<u32>> {
        if data.len() < 8 {
            return Err(Error::Container("stss too short".into()));
        }

        let entry_count = read_u32_be(&data[4..8])? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        let mut offset = 8;

        for _ in 0..entry_count {
            if offset + 4 > data.len() {
                break;
            }
            entries.push(read_u32_be(&data[offset..offset + 4])?);
            offset += 4;
        }

        Ok(entries)
    }
}
