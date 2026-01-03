//! FLAC decoder implementation with streaming support

#![allow(clippy::needless_range_loop)]
#![allow(clippy::enum_variant_names)]

use std::io::Read;
use crate::{
    FlacError, Result, FlacMetadata, StreamInfo, VorbisComment, Picture,
    SeekPoint, AudioFrame, MetadataBlockType, ChannelAssignment, SubframeType,
};

/// CRC-8 polynomial for FLAC (x^8 + x^2 + x^1 + x^0)
const CRC8_POLY: u8 = 0x07;

/// CRC-16 polynomial for FLAC (x^16 + x^15 + x^2 + x^0)
const CRC16_POLY: u16 = 0x8005;

/// Precomputed CRC-8 table
fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for i in 0..256 {
        let mut crc = i as u8;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ CRC8_POLY;
            } else {
                crc <<= 1;
            }
        }
        table[i] = crc;
    }
    table
}

/// Precomputed CRC-16 table
fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    for i in 0..256 {
        let mut crc = (i as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ CRC16_POLY;
            } else {
                crc <<= 1;
            }
        }
        table[i] = crc;
    }
    table
}

/// Calculate CRC-8
fn calculate_crc8(data: &[u8]) -> u8 {
    let table = crc8_table();
    let mut crc = 0u8;
    for &byte in data {
        crc = table[(crc ^ byte) as usize];
    }
    crc
}

/// Calculate CRC-16
fn calculate_crc16(data: &[u8]) -> u16 {
    let table = crc16_table();
    let mut crc = 0u16;
    for &byte in data {
        crc = (crc << 8) ^ table[((crc >> 8) as u8 ^ byte) as usize];
    }
    crc
}

/// Bitstream reader for FLAC decoding
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    #[allow(dead_code)]
    fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            0
        } else {
            (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
        }
    }

    fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(FlacError::InvalidSubframe);
        }

        let mut result = 0u32;
        let mut bits_left = n;

        while bits_left > 0 {
            if self.byte_pos >= self.data.len() {
                return Err(FlacError::UnexpectedEof);
            }

            let bits_available = 8 - self.bit_pos;
            let bits_to_read = bits_left.min(bits_available);

            // Create mask for available bits (handle 8 bits case)
            let mask = if bits_available == 8 {
                0xFF
            } else {
                (1u8 << bits_available) - 1
            };
            let byte_val = (self.data[self.byte_pos] & mask) >> (bits_available - bits_to_read);

            result = (result << bits_to_read) | (byte_val as u32);

            self.bit_pos += bits_to_read;
            bits_left -= bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? == 1)
    }

    fn read_unary(&mut self) -> Result<u32> {
        let mut count = 0u32;
        while !self.read_bit()? {
            count += 1;
            if count > 32 {
                return Err(FlacError::InvalidRicePartition);
            }
        }
        Ok(count)
    }

    fn read_signed(&mut self, n: u8) -> Result<i32> {
        let val = self.read_bits(n)? as i32;
        let sign_bit = 1i32 << (n - 1);
        if val & sign_bit != 0 {
            Ok(val | !((1i32 << n) - 1))
        } else {
            Ok(val)
        }
    }

    fn read_rice_signed(&mut self, param: u8) -> Result<i32> {
        let msb = self.read_unary()?;
        let lsb = if param > 0 {
            self.read_bits(param)?
        } else {
            0
        };
        let val = (msb << param) | lsb;
        // Fold negative values
        if val & 1 != 0 {
            Ok(-((val >> 1) as i32) - 1)
        } else {
            Ok((val >> 1) as i32)
        }
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    fn read_utf8_coded(&mut self) -> Result<u64> {
        let first = self.read_bits(8)? as u8;

        if first & 0x80 == 0 {
            return Ok(first as u64);
        }

        let leading_ones = first.leading_ones() as usize;
        if !(2..=7).contains(&leading_ones) {
            return Err(FlacError::InvalidFrameHeader);
        }

        let mut value = (first & (0xFF >> (leading_ones + 1))) as u64;

        for _ in 1..leading_ones {
            let byte = self.read_bits(8)? as u8;
            if byte & 0xC0 != 0x80 {
                return Err(FlacError::InvalidFrameHeader);
            }
            value = (value << 6) | ((byte & 0x3F) as u64);
        }

        Ok(value)
    }

    fn position(&self) -> usize {
        self.byte_pos
    }
}

/// FLAC decoder with full metadata and frame decoding support
pub struct FlacDecoder<R: Read> {
    reader: R,
    metadata: FlacMetadata,
    buffer: Vec<u8>,
    #[allow(dead_code)]
    stream_started: bool,
}

impl<R: Read> FlacDecoder<R> {
    /// Create a new FLAC decoder
    pub fn new(mut reader: R) -> Result<Self> {
        // Read and verify fLaC marker
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker)?;
        if &marker != b"fLaC" {
            return Err(FlacError::InvalidMarker);
        }

        let mut decoder = Self {
            reader,
            metadata: FlacMetadata::default(),
            buffer: Vec::with_capacity(65536),
            stream_started: false,
        };

        // Parse all metadata blocks
        decoder.parse_metadata()?;

        Ok(decoder)
    }

    /// Get the parsed metadata
    pub fn metadata(&self) -> &FlacMetadata {
        &self.metadata
    }

    /// Get stream info
    pub fn stream_info(&self) -> Option<&StreamInfo> {
        self.metadata.stream_info.as_ref()
    }

    fn parse_metadata(&mut self) -> Result<()> {
        loop {
            let mut header = [0u8; 4];
            self.reader.read_exact(&mut header)?;

            let is_last = header[0] & 0x80 != 0;
            let block_type = MetadataBlockType::from(header[0] & 0x7F);
            let length = ((header[1] as u32) << 16) | ((header[2] as u32) << 8) | (header[3] as u32);

            let mut data = vec![0u8; length as usize];
            self.reader.read_exact(&mut data)?;

            match block_type {
                MetadataBlockType::StreamInfo => {
                    self.metadata.stream_info = Some(Self::parse_stream_info(&data)?);
                }
                MetadataBlockType::VorbisComment => {
                    self.metadata.vorbis_comment = Some(Self::parse_vorbis_comment(&data)?);
                }
                MetadataBlockType::Picture => {
                    self.metadata.pictures.push(Self::parse_picture(&data)?);
                }
                MetadataBlockType::SeekTable => {
                    self.metadata.seek_table = Self::parse_seek_table(&data)?;
                }
                MetadataBlockType::Application => {
                    if data.len() >= 4 {
                        let id = String::from_utf8_lossy(&data[0..4]).to_string();
                        self.metadata.application_data.push((id, data[4..].to_vec()));
                    }
                }
                _ => {} // Skip padding and unknown blocks
            }

            if is_last {
                break;
            }
        }

        if self.metadata.stream_info.is_none() {
            return Err(FlacError::InvalidMetadata);
        }

        Ok(())
    }

    fn parse_stream_info(data: &[u8]) -> Result<StreamInfo> {
        if data.len() < 34 {
            return Err(FlacError::InvalidMetadata);
        }

        let min_block_size = ((data[0] as u16) << 8) | (data[1] as u16);
        let max_block_size = ((data[2] as u16) << 8) | (data[3] as u16);
        let min_frame_size = ((data[4] as u32) << 16) | ((data[5] as u32) << 8) | (data[6] as u32);
        let max_frame_size = ((data[7] as u32) << 16) | ((data[8] as u32) << 8) | (data[9] as u32);

        // Sample rate: 20 bits
        let sample_rate = ((data[10] as u32) << 12) | ((data[11] as u32) << 4) | ((data[12] as u32) >> 4);

        // Channels: 3 bits (stored as channels - 1)
        let channels = ((data[12] >> 1) & 0x07) + 1;

        // Bits per sample: 5 bits (stored as bits - 1)
        let bits_per_sample = (((data[12] & 0x01) << 4) | ((data[13] >> 4) & 0x0F)) + 1;

        // Total samples: 36 bits
        let total_samples = ((data[13] as u64 & 0x0F) << 32)
            | ((data[14] as u64) << 24)
            | ((data[15] as u64) << 16)
            | ((data[16] as u64) << 8)
            | (data[17] as u64);

        let mut md5_signature = [0u8; 16];
        md5_signature.copy_from_slice(&data[18..34]);

        Ok(StreamInfo {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5_signature,
        })
    }

    fn parse_vorbis_comment(data: &[u8]) -> Result<VorbisComment> {
        if data.len() < 8 {
            return Err(FlacError::InvalidMetadata);
        }

        let mut pos = 0;

        // Vendor string (little-endian length)
        let vendor_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;

        if pos + vendor_len > data.len() {
            return Err(FlacError::InvalidMetadata);
        }
        let vendor = String::from_utf8_lossy(&data[pos..pos+vendor_len]).to_string();
        pos += vendor_len;

        // Number of comments
        if pos + 4 > data.len() {
            return Err(FlacError::InvalidMetadata);
        }
        let num_comments = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;

        let mut comments = Vec::with_capacity(num_comments);
        for _ in 0..num_comments {
            if pos + 4 > data.len() {
                break;
            }
            let comment_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;

            if pos + comment_len > data.len() {
                break;
            }
            let comment = String::from_utf8_lossy(&data[pos..pos+comment_len]).to_string();
            pos += comment_len;

            if let Some(eq_pos) = comment.find('=') {
                let key = comment[..eq_pos].to_uppercase();
                let value = comment[eq_pos+1..].to_string();
                comments.push((key, value));
            }
        }

        Ok(VorbisComment { vendor, comments })
    }

    fn parse_picture(data: &[u8]) -> Result<Picture> {
        if data.len() < 32 {
            return Err(FlacError::InvalidMetadata);
        }

        let mut pos = 0;

        let picture_type = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;

        let mime_len = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let mime_type = String::from_utf8_lossy(&data[pos..pos+mime_len]).to_string();
        pos += mime_len;

        let desc_len = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let description = String::from_utf8_lossy(&data[pos..pos+desc_len]).to_string();
        pos += desc_len;

        let width = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;
        let height = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;
        let color_depth = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;
        let colors = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;

        let data_len = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let picture_data = data[pos..pos+data_len].to_vec();

        Ok(Picture {
            picture_type,
            mime_type,
            description,
            width,
            height,
            color_depth,
            colors,
            data: picture_data,
        })
    }

    fn parse_seek_table(data: &[u8]) -> Result<Vec<SeekPoint>> {
        let num_points = data.len() / 18;
        let mut points = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let offset = i * 18;
            let sample_number = u64::from_be_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3],
                data[offset+4], data[offset+5], data[offset+6], data[offset+7],
            ]);

            // Skip placeholder points
            if sample_number == 0xFFFFFFFFFFFFFFFF {
                continue;
            }

            let stream_offset = u64::from_be_bytes([
                data[offset+8], data[offset+9], data[offset+10], data[offset+11],
                data[offset+12], data[offset+13], data[offset+14], data[offset+15],
            ]);
            let frame_samples = u16::from_be_bytes([data[offset+16], data[offset+17]]);

            points.push(SeekPoint {
                sample_number,
                stream_offset,
                frame_samples,
            });
        }

        Ok(points)
    }

    /// Decode the next audio frame
    pub fn next_frame(&mut self) -> Result<Option<AudioFrame>> {
        let stream_info = self.metadata.stream_info.as_ref()
            .ok_or(FlacError::InvalidMetadata)?;

        // Read frame sync code
        loop {
            let mut sync = [0u8; 2];
            match self.reader.read_exact(&mut sync) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(None);
                }
                Err(e) => return Err(FlacError::Io(e)),
            }

            // Check for sync code (0xFF 0xF8 or 0xFF 0xF9)
            if sync[0] == 0xFF && (sync[1] & 0xFC) == 0xF8 {
                self.buffer.clear();
                self.buffer.extend_from_slice(&sync);
                break;
            }
        }

        // Read the rest of the frame header and data
        // We need to buffer the entire frame for CRC validation
        let mut total_read = 2;

        // Read more data until we find the next sync or EOF
        loop {
            let mut byte = [0u8; 1];
            match self.reader.read_exact(&mut byte) {
                Ok(_) => {
                    self.buffer.push(byte[0]);
                    total_read += 1;

                    // Check for next frame sync (end of current frame)
                    if self.buffer.len() >= 4 {
                        let len = self.buffer.len();
                        if self.buffer[len-2] == 0xFF && (self.buffer[len-1] & 0xFC) == 0xF8 {
                            // Remove the sync bytes - they belong to next frame
                            self.buffer.truncate(len - 2);
                            // Put them back for next call (we'd need to refactor to handle this properly)
                            break;
                        }
                    }

                    if total_read > 65536 {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => return Err(FlacError::Io(e)),
            }
        }

        self.decode_frame(&self.buffer.clone(), stream_info)
    }

    fn decode_frame(&self, data: &[u8], stream_info: &StreamInfo) -> Result<Option<AudioFrame>> {
        if data.len() < 6 {
            return Err(FlacError::InvalidFrameHeader);
        }

        let mut reader = BitReader::new(data);

        // Sync code (already verified)
        let _sync = reader.read_bits(14)?;

        // Reserved bit
        let _reserved = reader.read_bit()?;

        // Blocking strategy (0 = fixed block size, 1 = variable)
        let _blocking_strategy = reader.read_bit()?;

        // Block size
        let block_size_code = reader.read_bits(4)? as u8;
        let block_size = match block_size_code {
            0 => return Err(FlacError::InvalidFrameHeader),
            1 => 192,
            2..=5 => 576 * (1 << (block_size_code - 2)),
            6 => 0, // Read 8-bit value later
            7 => 0, // Read 16-bit value later
            _ => 256 * (1 << (block_size_code - 8)),
        };

        // Sample rate
        let sample_rate_code = reader.read_bits(4)? as u8;
        let sample_rate = match sample_rate_code {
            0 => stream_info.sample_rate,
            1 => 88200,
            2 => 176400,
            3 => 192000,
            4 => 8000,
            5 => 16000,
            6 => 22050,
            7 => 24000,
            8 => 32000,
            9 => 44100,
            10 => 48000,
            11 => 96000,
            12 => 0, // Read 8-bit kHz later
            13 => 0, // Read 16-bit Hz later
            14 => 0, // Read 16-bit tens of Hz later
            15 => return Err(FlacError::InvalidFrameHeader),
            _ => unreachable!(),
        };

        // Channel assignment
        let channel_code = reader.read_bits(4)? as u8;
        let channel_assignment = match channel_code {
            0..=7 => ChannelAssignment::Independent(channel_code + 1),
            8 => ChannelAssignment::LeftSide,
            9 => ChannelAssignment::RightSide,
            10 => ChannelAssignment::MidSide,
            _ => return Err(FlacError::InvalidFrameHeader),
        };

        let channels = match channel_assignment {
            ChannelAssignment::Independent(n) => n,
            _ => 2,
        };

        // Sample size
        let sample_size_code = reader.read_bits(3)? as u8;
        let bits_per_sample = match sample_size_code {
            0 => stream_info.bits_per_sample,
            1 => 8,
            2 => 12,
            3 => return Err(FlacError::InvalidFrameHeader),
            4 => 16,
            5 => 20,
            6 => 24,
            7 => 32,
            _ => unreachable!(),
        };

        // Reserved bit
        let _reserved = reader.read_bit()?;

        // Frame/sample number (UTF-8 coded)
        let frame_number = reader.read_utf8_coded()?;

        // Extended block size
        let block_size = if block_size_code == 6 {
            reader.read_bits(8)? + 1
        } else if block_size_code == 7 {
            reader.read_bits(16)? + 1
        } else {
            block_size
        };

        // Extended sample rate
        let sample_rate = if sample_rate_code == 12 {
            reader.read_bits(8)? * 1000
        } else if sample_rate_code == 13 {
            reader.read_bits(16)?
        } else if sample_rate_code == 14 {
            reader.read_bits(16)? * 10
        } else {
            sample_rate
        };

        // CRC-8 of frame header
        let header_end = reader.position();
        let expected_crc8 = reader.read_bits(8)? as u8;
        let actual_crc8 = calculate_crc8(&data[0..header_end]);

        if expected_crc8 != actual_crc8 {
            return Err(FlacError::CrcMismatch {
                expected: expected_crc8 as u16,
                actual: actual_crc8 as u16,
            });
        }

        // Decode subframes
        let mut all_samples = Vec::with_capacity((block_size * channels as u32) as usize);
        let mut channel_samples = Vec::with_capacity(channels as usize);

        for ch in 0..channels {
            let bps = match channel_assignment {
                ChannelAssignment::LeftSide if ch == 1 => bits_per_sample + 1,
                ChannelAssignment::RightSide if ch == 0 => bits_per_sample + 1,
                ChannelAssignment::MidSide if ch == 1 => bits_per_sample + 1,
                _ => bits_per_sample,
            };

            let samples = self.decode_subframe(&mut reader, block_size as usize, bps)?;
            channel_samples.push(samples);
        }

        // Apply stereo decorrelation
        match channel_assignment {
            ChannelAssignment::LeftSide => {
                // Side = Left - Right, so Right = Left - Side
                for i in 0..block_size as usize {
                    channel_samples[1][i] = channel_samples[0][i] - channel_samples[1][i];
                }
            }
            ChannelAssignment::RightSide => {
                // Side = Left - Right, so Left = Right + Side
                for i in 0..block_size as usize {
                    channel_samples[0][i] += channel_samples[1][i];
                }
            }
            ChannelAssignment::MidSide => {
                // Mid = (Left + Right) / 2, Side = Left - Right
                // Left = Mid + Side/2, Right = Mid - Side/2
                for i in 0..block_size as usize {
                    let mid = channel_samples[0][i];
                    let side = channel_samples[1][i];
                    channel_samples[0][i] = mid + ((side + 1) >> 1);
                    channel_samples[1][i] = mid - (side >> 1);
                }
            }
            _ => {}
        }

        // Interleave samples
        for i in 0..block_size as usize {
            for ch in 0..channels as usize {
                all_samples.push(channel_samples[ch][i]);
            }
        }

        // Align to byte boundary
        reader.align_to_byte();

        // CRC-16 of entire frame (except the CRC-16 itself)
        let frame_end = reader.position();
        if frame_end + 2 <= data.len() {
            let expected_crc16 = ((data[frame_end] as u16) << 8) | (data[frame_end + 1] as u16);
            let actual_crc16 = calculate_crc16(&data[0..frame_end]);

            if expected_crc16 != actual_crc16 {
                return Err(FlacError::Crc16Mismatch {
                    expected: expected_crc16,
                    actual: actual_crc16,
                });
            }
        }

        Ok(Some(AudioFrame {
            sample_rate,
            channels,
            bits_per_sample,
            block_size,
            frame_number,
            samples: all_samples,
        }))
    }

    fn decode_subframe(&self, reader: &mut BitReader, block_size: usize, bps: u8) -> Result<Vec<i32>> {
        // Zero bit padding
        let _zero = reader.read_bit()?;

        // Subframe type
        let type_bits = reader.read_bits(6)? as u8;
        let subframe_type = match type_bits {
            0 => SubframeType::Constant,
            1 => SubframeType::Verbatim,
            8..=12 => SubframeType::Fixed(type_bits & 0x07),
            32..=63 => SubframeType::Lpc((type_bits & 0x1F) + 1),
            _ => return Err(FlacError::InvalidSubframe),
        };

        // Wasted bits per sample flag
        let has_wasted_bits = reader.read_bit()?;
        let wasted_bits = if has_wasted_bits {
            reader.read_unary()? as u8 + 1
        } else {
            0
        };

        let effective_bps = bps - wasted_bits;

        let mut samples = match subframe_type {
            SubframeType::Constant => {
                let value = reader.read_signed(effective_bps)?;
                vec![value; block_size]
            }
            SubframeType::Verbatim => {
                let mut samples = Vec::with_capacity(block_size);
                for _ in 0..block_size {
                    samples.push(reader.read_signed(effective_bps)?);
                }
                samples
            }
            SubframeType::Fixed(order) => {
                self.decode_fixed(reader, block_size, effective_bps, order)?
            }
            SubframeType::Lpc(order) => {
                self.decode_lpc(reader, block_size, effective_bps, order)?
            }
        };

        // Apply wasted bits
        if wasted_bits > 0 {
            for sample in &mut samples {
                *sample <<= wasted_bits;
            }
        }

        Ok(samples)
    }

    fn decode_fixed(&self, reader: &mut BitReader, block_size: usize, bps: u8, order: u8) -> Result<Vec<i32>> {
        let mut samples = Vec::with_capacity(block_size);

        // Read warm-up samples
        for _ in 0..order {
            samples.push(reader.read_signed(bps)?);
        }

        // Decode residual
        let residual = self.decode_residual(reader, block_size, order as usize)?;

        // Fixed prediction coefficients
        let coeffs: &[i32] = match order {
            0 => &[],
            1 => &[1],
            2 => &[2, -1],
            3 => &[3, -3, 1],
            4 => &[4, -6, 4, -1],
            _ => return Err(FlacError::InvalidLpcOrder(order)),
        };

        // Apply prediction
        for i in 0..residual.len() {
            let mut prediction = 0i64;
            for (j, &coef) in coeffs.iter().enumerate() {
                prediction += coef as i64 * samples[samples.len() - 1 - j] as i64;
            }
            samples.push(residual[i] + prediction as i32);
        }

        Ok(samples)
    }

    fn decode_lpc(&self, reader: &mut BitReader, block_size: usize, bps: u8, order: u8) -> Result<Vec<i32>> {
        let mut samples = Vec::with_capacity(block_size);

        // Read warm-up samples
        for _ in 0..order {
            samples.push(reader.read_signed(bps)?);
        }

        // LPC coefficient precision
        let precision = reader.read_bits(4)? as u8 + 1;
        if precision > 15 {
            return Err(FlacError::InvalidLpcOrder(order));
        }

        // LPC coefficient shift
        let shift = reader.read_signed(5)?;

        // Read LPC coefficients
        let mut coeffs = Vec::with_capacity(order as usize);
        for _ in 0..order {
            coeffs.push(reader.read_signed(precision)?);
        }

        // Decode residual
        let residual = self.decode_residual(reader, block_size, order as usize)?;

        // Apply LPC prediction
        for i in 0..residual.len() {
            let mut prediction = 0i64;
            for (j, &coef) in coeffs.iter().enumerate() {
                prediction += coef as i64 * samples[samples.len() - 1 - j] as i64;
            }
            let predicted = if shift >= 0 {
                (prediction >> shift) as i32
            } else {
                (prediction << (-shift)) as i32
            };
            samples.push(residual[i] + predicted);
        }

        Ok(samples)
    }

    fn decode_residual(&self, reader: &mut BitReader, block_size: usize, predictor_order: usize) -> Result<Vec<i32>> {
        let coding_method = reader.read_bits(2)? as u8;

        let rice_param_bits = match coding_method {
            0 => 4,
            1 => 5,
            _ => return Err(FlacError::InvalidRicePartition),
        };

        let partition_order = reader.read_bits(4)? as u8;
        let num_partitions = 1usize << partition_order;

        let mut residual = Vec::with_capacity(block_size - predictor_order);

        for partition in 0..num_partitions {
            let samples_in_partition = if partition_order == 0 {
                block_size - predictor_order
            } else if partition == 0 {
                (block_size >> partition_order) - predictor_order
            } else {
                block_size >> partition_order
            };

            let rice_param = reader.read_bits(rice_param_bits)? as u8;

            if rice_param == (1 << rice_param_bits) - 1 {
                // Escape code - raw encoding
                let raw_bits = reader.read_bits(5)? as u8;
                for _ in 0..samples_in_partition {
                    residual.push(reader.read_signed(raw_bits)?);
                }
            } else {
                // Rice coding
                for _ in 0..samples_in_partition {
                    residual.push(reader.read_rice_signed(rice_param)?);
                }
            }
        }

        Ok(residual)
    }
}

/// Streaming FLAC decoder for decoding frames as they arrive
pub struct StreamingDecoder {
    metadata: Option<FlacMetadata>,
    buffer: Vec<u8>,
    state: StreamingState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum StreamingState {
    ReadingMarker,
    ReadingMetadata,
    ReadingFrames,
}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingDecoder {
    /// Create a new streaming decoder
    pub fn new() -> Self {
        Self {
            metadata: None,
            buffer: Vec::with_capacity(65536),
            state: StreamingState::ReadingMarker,
        }
    }

    /// Feed data to the decoder
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Check if metadata has been parsed
    pub fn has_metadata(&self) -> bool {
        self.metadata.is_some()
    }

    /// Get metadata if available
    pub fn metadata(&self) -> Option<&FlacMetadata> {
        self.metadata.as_ref()
    }

    /// Try to decode the next frame from buffered data
    pub fn decode_frame(&mut self) -> Result<Option<AudioFrame>> {
        match self.state {
            StreamingState::ReadingMarker => {
                if self.buffer.len() < 4 {
                    return Ok(None);
                }
                if &self.buffer[0..4] != b"fLaC" {
                    return Err(FlacError::InvalidMarker);
                }
                self.buffer.drain(0..4);
                self.state = StreamingState::ReadingMetadata;
                self.decode_frame()
            }
            StreamingState::ReadingMetadata => {
                self.try_parse_metadata()?;
                if self.metadata.is_some() {
                    self.state = StreamingState::ReadingFrames;
                }
                Ok(None)
            }
            StreamingState::ReadingFrames => {
                self.try_decode_frame()
            }
        }
    }

    fn try_parse_metadata(&mut self) -> Result<()> {
        let mut metadata = FlacMetadata::default();
        let mut offset = 0;

        loop {
            if offset + 4 > self.buffer.len() {
                return Ok(());
            }

            let is_last = self.buffer[offset] & 0x80 != 0;
            let block_type = MetadataBlockType::from(self.buffer[offset] & 0x7F);
            let length = ((self.buffer[offset + 1] as u32) << 16)
                | ((self.buffer[offset + 2] as u32) << 8)
                | (self.buffer[offset + 3] as u32);

            let block_end = offset + 4 + length as usize;
            if block_end > self.buffer.len() {
                return Ok(());
            }

            let data = &self.buffer[offset + 4..block_end];

            match block_type {
                MetadataBlockType::StreamInfo => {
                    metadata.stream_info = Some(FlacDecoder::<std::io::Empty>::parse_stream_info(data)?);
                }
                MetadataBlockType::VorbisComment => {
                    metadata.vorbis_comment = Some(FlacDecoder::<std::io::Empty>::parse_vorbis_comment(data)?);
                }
                MetadataBlockType::Picture => {
                    metadata.pictures.push(FlacDecoder::<std::io::Empty>::parse_picture(data)?);
                }
                MetadataBlockType::SeekTable => {
                    metadata.seek_table = FlacDecoder::<std::io::Empty>::parse_seek_table(data)?;
                }
                _ => {}
            }

            offset = block_end;

            if is_last {
                break;
            }
        }

        self.buffer.drain(0..offset);
        self.metadata = Some(metadata);
        Ok(())
    }

    fn try_decode_frame(&mut self) -> Result<Option<AudioFrame>> {
        // Find frame sync
        let mut sync_pos = None;
        for i in 0..self.buffer.len().saturating_sub(1) {
            if self.buffer[i] == 0xFF && (self.buffer[i + 1] & 0xFC) == 0xF8 {
                sync_pos = Some(i);
                break;
            }
        }

        let start = match sync_pos {
            Some(pos) => {
                if pos > 0 {
                    self.buffer.drain(0..pos);
                }
                0
            }
            None => return Ok(None),
        };

        // Find next sync (end of frame)
        let mut end = None;
        for i in (start + 2)..self.buffer.len().saturating_sub(1) {
            if self.buffer[i] == 0xFF && (self.buffer[i + 1] & 0xFC) == 0xF8 {
                end = Some(i);
                break;
            }
        }

        let frame_data = match end {
            Some(end_pos) => &self.buffer[start..end_pos],
            None => {
                // Not enough data yet
                if self.buffer.len() > 65536 {
                    // Buffer too large without finding frame end - likely corrupted
                    self.buffer.clear();
                    return Err(FlacError::InvalidFrameHeader);
                }
                return Ok(None);
            }
        };

        let stream_info = self.metadata.as_ref()
            .and_then(|m| m.stream_info.as_ref())
            .ok_or(FlacError::InvalidMetadata)?;

        // Create a temporary decoder to parse the frame
        let decoder = FlacDecoder::<std::io::Empty> {
            reader: std::io::empty(),
            metadata: self.metadata.clone().unwrap_or_default(),
            buffer: Vec::new(),
            stream_started: true,
        };

        let result = decoder.decode_frame(frame_data, stream_info);

        if let Some(end_pos) = end {
            self.buffer.drain(0..end_pos);
        }

        result
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.metadata = None;
        self.buffer.clear();
        self.state = StreamingState::ReadingMarker;
    }

    /// Clear the buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc8() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let crc = calculate_crc8(&data);
        assert_eq!(crc, 0x00);
    }

    #[test]
    fn test_crc16() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let crc = calculate_crc16(&data);
        assert_eq!(crc, 0x0000);
    }

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b01100001];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01100001);
    }

    #[test]
    fn test_streaming_decoder_new() {
        let decoder = StreamingDecoder::new();
        assert!(!decoder.has_metadata());
        assert_eq!(decoder.buffer_size(), 0);
    }

    #[test]
    fn test_streaming_decoder_feed() {
        let mut decoder = StreamingDecoder::new();
        decoder.feed(b"fLaC");
        assert_eq!(decoder.buffer_size(), 4);
    }

    #[test]
    fn test_metadata_block_type_from() {
        assert_eq!(MetadataBlockType::from(0), MetadataBlockType::StreamInfo);
        assert_eq!(MetadataBlockType::from(4), MetadataBlockType::VorbisComment);
        assert_eq!(MetadataBlockType::from(6), MetadataBlockType::Picture);
        assert!(matches!(MetadataBlockType::from(127), MetadataBlockType::Reserved(127)));
    }
}
