//! FLAC encoder implementation with configurable compression levels

#![allow(clippy::needless_range_loop)]

use std::io::Write;
use crate::{FlacError, Result, StreamInfo, VorbisComment};

/// Compression level presets (0 = fastest, 8 = best compression)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Level 0: Fastest, minimal compression
    Fastest,
    /// Level 1-2: Fast compression
    Fast,
    /// Level 3-4: Balanced (default)
    Default,
    /// Level 5-6: Better compression
    Better,
    /// Level 7-8: Best compression, slowest
    Best,
    /// Custom settings
    Custom {
        block_size: u32,
        max_lpc_order: u8,
        rice_parameter_search: u8,
        do_exhaustive_model_search: bool,
        do_qlp_coeff_prec_search: bool,
    },
}

impl CompressionLevel {
    /// Get block size for this compression level
    pub fn block_size(&self) -> u32 {
        match self {
            CompressionLevel::Fastest => 1152,
            CompressionLevel::Fast => 2304,
            CompressionLevel::Default => 4096,
            CompressionLevel::Better => 4608,
            CompressionLevel::Best => 4608,
            CompressionLevel::Custom { block_size, .. } => *block_size,
        }
    }

    /// Get maximum LPC order
    pub fn max_lpc_order(&self) -> u8 {
        match self {
            CompressionLevel::Fastest => 0, // Fixed prediction only
            CompressionLevel::Fast => 6,
            CompressionLevel::Default => 8,
            CompressionLevel::Better => 10,
            CompressionLevel::Best => 12,
            CompressionLevel::Custom { max_lpc_order, .. } => *max_lpc_order,
        }
    }

    /// Get Rice parameter search depth
    pub fn rice_parameter_search(&self) -> u8 {
        match self {
            CompressionLevel::Fastest => 0,
            CompressionLevel::Fast => 0,
            CompressionLevel::Default => 0,
            CompressionLevel::Better => 1,
            CompressionLevel::Best => 2,
            CompressionLevel::Custom { rice_parameter_search, .. } => *rice_parameter_search,
        }
    }

    /// Whether to do exhaustive model search
    pub fn do_exhaustive_model_search(&self) -> bool {
        match self {
            CompressionLevel::Fastest | CompressionLevel::Fast | CompressionLevel::Default => false,
            CompressionLevel::Better | CompressionLevel::Best => true,
            CompressionLevel::Custom { do_exhaustive_model_search, .. } => *do_exhaustive_model_search,
        }
    }
}

impl From<u8> for CompressionLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => CompressionLevel::Fastest,
            1 | 2 => CompressionLevel::Fast,
            3 | 4 => CompressionLevel::Default,
            5 | 6 => CompressionLevel::Better,
            _ => CompressionLevel::Best,
        }
    }
}

/// CRC-8 table
fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for i in 0..256 {
        let mut crc = i as u8;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
        table[i] = crc;
    }
    table
}

/// CRC-16 table
fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    for i in 0..256 {
        let mut crc = (i as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
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

/// Bitstream writer for FLAC encoding
struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        if n == 0 {
            return;
        }

        let mut bits_left = n;
        let mut val = value;

        while bits_left > 0 {
            let bits_available = 8 - self.bit_pos;
            let bits_to_write = bits_left.min(bits_available);

            let shift = bits_left - bits_to_write;
            let mask = ((1u32 << bits_to_write) - 1) as u8;
            let byte_val = ((val >> shift) & mask as u32) as u8;

            self.current_byte |= byte_val << (bits_available - bits_to_write);
            self.bit_pos += bits_to_write;
            bits_left -= bits_to_write;
            val &= (1u32 << shift) - 1;

            if self.bit_pos >= 8 {
                self.buffer.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    fn write_bit(&mut self, bit: bool) {
        self.write_bits(if bit { 1 } else { 0 }, 1);
    }

    fn write_signed(&mut self, value: i32, n: u8) {
        let mask = (1u32 << n) - 1;
        self.write_bits((value as u32) & mask, n);
    }

    fn write_unary(&mut self, value: u32) {
        for _ in 0..value {
            self.write_bit(false);
        }
        self.write_bit(true);
    }

    fn write_rice_signed(&mut self, value: i32, param: u8) {
        // Fold to unsigned
        let unsigned = if value >= 0 {
            (value as u32) << 1
        } else {
            ((-(value + 1)) as u32) << 1 | 1
        };

        let msb = unsigned >> param;
        let lsb = unsigned & ((1 << param) - 1);

        self.write_unary(msb);
        if param > 0 {
            self.write_bits(lsb, param);
        }
    }

    fn write_utf8_coded(&mut self, value: u64) {
        if value < 0x80 {
            self.write_bits(value as u32, 8);
        } else if value < 0x800 {
            self.write_bits(0xC0 | ((value >> 6) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x10000 {
            self.write_bits(0xE0 | ((value >> 12) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x200000 {
            self.write_bits(0xF0 | ((value >> 18) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x4000000 {
            self.write_bits(0xF8 | ((value >> 24) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else if value < 0x80000000 {
            self.write_bits(0xFC | ((value >> 30) as u32), 8);
            self.write_bits(0x80 | (((value >> 24) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        } else {
            self.write_bits(0xFE, 8);
            self.write_bits(0x80 | (((value >> 30) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 24) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 18) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 12) & 0x3F) as u32), 8);
            self.write_bits(0x80 | (((value >> 6) & 0x3F) as u32), 8);
            self.write_bits(0x80 | ((value & 0x3F) as u32), 8);
        }
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.buffer
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    #[allow(dead_code)]
    fn position(&self) -> usize {
        self.buffer.len()
    }
}

/// FLAC encoder
pub struct FlacEncoder<W: Write> {
    writer: W,
    stream_info: StreamInfo,
    compression_level: CompressionLevel,
    frame_number: u64,
    samples_written: u64,
    md5_context: Md5Context,
    finished: bool,
    header_position: u64,
}

/// Simple MD5 context for audio data hashing
struct Md5Context {
    state: [u32; 4],
    count: [u32; 2],
    buffer: [u8; 64],
}

impl Md5Context {
    fn new() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            count: [0, 0],
            buffer: [0; 64],
        }
    }

    fn update(&mut self, input: &[u8]) {
        let mut index = ((self.count[0] >> 3) & 0x3F) as usize;
        let input_len = input.len();

        self.count[0] = self.count[0].wrapping_add((input_len << 3) as u32);
        if self.count[0] < (input_len << 3) as u32 {
            self.count[1] = self.count[1].wrapping_add(1);
        }
        self.count[1] = self.count[1].wrapping_add((input_len >> 29) as u32);

        let part_len = 64 - index;
        let mut i = 0;

        if input_len >= part_len {
            self.buffer[index..64].copy_from_slice(&input[..part_len]);
            self.transform(&self.buffer.clone());

            i = part_len;
            while i + 63 < input_len {
                // Safe: loop condition ensures input[i..i+64] is exactly 64 bytes
                let block: [u8; 64] = input[i..i + 64]
                    .try_into()
                    .expect("slice length is 64 by loop invariant");
                self.transform(&block);
                i += 64;
            }
            index = 0;
        }

        self.buffer[index..index + input_len - i].copy_from_slice(&input[i..]);
    }

    fn transform(&mut self, block: &[u8; 64]) {
        const S: [[u32; 4]; 4] = [
            [7, 12, 17, 22],
            [5, 9, 14, 20],
            [4, 11, 16, 23],
            [6, 10, 15, 21],
        ];

        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
            0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
            0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
            0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
            0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
            0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
            0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
            0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
        ];

        let mut x = [0u32; 16];
        for i in 0..16 {
            x[i] = u32::from_le_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];

        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };

            let temp = d;
            d = c;
            c = b;
            let round = i / 16;
            let shift_idx = i % 4;
            b = b.wrapping_add(
                (a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(x[g]))
                    .rotate_left(S[round][shift_idx]),
            );
            a = temp;
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }

    fn finalize(mut self) -> [u8; 16] {
        let bits = [
            (self.count[0]) as u8,
            (self.count[0] >> 8) as u8,
            (self.count[0] >> 16) as u8,
            (self.count[0] >> 24) as u8,
            (self.count[1]) as u8,
            (self.count[1] >> 8) as u8,
            (self.count[1] >> 16) as u8,
            (self.count[1] >> 24) as u8,
        ];

        let index = ((self.count[0] >> 3) & 0x3F) as usize;
        let pad_len = if index < 56 { 56 - index } else { 120 - index };

        let mut padding = vec![0u8; pad_len];
        padding[0] = 0x80;
        self.update(&padding);
        self.update(&bits);

        let mut result = [0u8; 16];
        for i in 0..4 {
            result[i * 4] = self.state[i] as u8;
            result[i * 4 + 1] = (self.state[i] >> 8) as u8;
            result[i * 4 + 2] = (self.state[i] >> 16) as u8;
            result[i * 4 + 3] = (self.state[i] >> 24) as u8;
        }
        result
    }
}

impl<W: Write> FlacEncoder<W> {
    /// Create a new FLAC encoder
    pub fn new(
        writer: W,
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u8,
        compression_level: CompressionLevel,
    ) -> Result<Self> {
        if !(1..=8).contains(&channels) {
            return Err(FlacError::EncodingError("Channels must be 1-8".into()));
        }
        if !(4..=32).contains(&bits_per_sample) {
            return Err(FlacError::EncodingError("Bits per sample must be 4-32".into()));
        }
        if sample_rate == 0 || sample_rate > 655350 {
            return Err(FlacError::EncodingError("Invalid sample rate".into()));
        }

        let block_size = compression_level.block_size() as u16;

        let stream_info = StreamInfo {
            min_block_size: block_size,
            max_block_size: block_size,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples: 0,
            md5_signature: [0; 16],
        };

        let mut encoder = Self {
            writer,
            stream_info,
            compression_level,
            frame_number: 0,
            samples_written: 0,
            md5_context: Md5Context::new(),
            finished: false,
            header_position: 0,
        };

        encoder.write_header()?;

        Ok(encoder)
    }

    /// Create encoder with custom stream info
    pub fn with_stream_info(
        writer: W,
        stream_info: StreamInfo,
        compression_level: CompressionLevel,
    ) -> Result<Self> {
        let mut encoder = Self {
            writer,
            stream_info,
            compression_level,
            frame_number: 0,
            samples_written: 0,
            md5_context: Md5Context::new(),
            finished: false,
            header_position: 0,
        };

        encoder.write_header()?;

        Ok(encoder)
    }

    fn write_header(&mut self) -> Result<()> {
        // Write fLaC marker
        self.writer.write_all(b"fLaC")?;

        // Write STREAMINFO block (always first, is_last = true for now)
        let stream_info_data = self.encode_stream_info();
        self.writer.write_all(&[0x80, 0x00, 0x00, 0x22])?; // is_last=1, type=0, length=34
        self.header_position = 4; // Position after the block header
        self.writer.write_all(&stream_info_data)?;

        Ok(())
    }

    fn encode_stream_info(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(34);

        // Min/max block size
        data.push((self.stream_info.min_block_size >> 8) as u8);
        data.push(self.stream_info.min_block_size as u8);
        data.push((self.stream_info.max_block_size >> 8) as u8);
        data.push(self.stream_info.max_block_size as u8);

        // Min/max frame size (3 bytes each)
        data.push((self.stream_info.min_frame_size >> 16) as u8);
        data.push((self.stream_info.min_frame_size >> 8) as u8);
        data.push(self.stream_info.min_frame_size as u8);
        data.push((self.stream_info.max_frame_size >> 16) as u8);
        data.push((self.stream_info.max_frame_size >> 8) as u8);
        data.push(self.stream_info.max_frame_size as u8);

        // Sample rate (20 bits) + channels (3 bits) + bits per sample (5 bits) + total samples (36 bits)
        let sample_rate = self.stream_info.sample_rate;
        let channels = self.stream_info.channels - 1;
        let bps = self.stream_info.bits_per_sample - 1;
        let total_samples = self.stream_info.total_samples;

        data.push((sample_rate >> 12) as u8);
        data.push((sample_rate >> 4) as u8);
        data.push(((sample_rate << 4) as u8) | ((channels << 1) & 0x0E) | ((bps >> 4) & 0x01));
        data.push(((bps << 4) & 0xF0) | ((total_samples >> 32) as u8 & 0x0F));
        data.push((total_samples >> 24) as u8);
        data.push((total_samples >> 16) as u8);
        data.push((total_samples >> 8) as u8);
        data.push(total_samples as u8);

        // MD5 signature
        data.extend_from_slice(&self.stream_info.md5_signature);

        data
    }

    /// Write a Vorbis comment metadata block
    pub fn write_vorbis_comment(&mut self, comment: &VorbisComment) -> Result<()> {
        let mut data = Vec::new();

        // Vendor string (little-endian length)
        let vendor_bytes = comment.vendor.as_bytes();
        data.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor_bytes);

        // Number of comments
        data.extend_from_slice(&(comment.comments.len() as u32).to_le_bytes());

        // Comments
        for (key, value) in &comment.comments {
            let entry = format!("{}={}", key, value);
            let entry_bytes = entry.as_bytes();
            data.extend_from_slice(&(entry_bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(entry_bytes);
        }

        // Write block header
        let len = data.len() as u32;
        self.writer.write_all(&[
            0x04, // type = 4 (VORBIS_COMMENT), is_last = 0
            (len >> 16) as u8,
            (len >> 8) as u8,
            len as u8,
        ])?;
        self.writer.write_all(&data)?;

        Ok(())
    }

    /// Encode audio samples (interleaved format)
    pub fn encode_samples(&mut self, samples: &[i32]) -> Result<()> {
        if self.finished {
            return Err(FlacError::EncodingError("Encoder already finished".into()));
        }

        let channels = self.stream_info.channels as usize;
        let block_size = self.stream_info.max_block_size as usize;
        let samples_per_channel = samples.len() / channels;

        // Update MD5 for raw samples
        for &sample in samples {
            let bytes = sample.to_le_bytes();
            let byte_count = (self.stream_info.bits_per_sample as usize).div_ceil(8);
            self.md5_context.update(&bytes[..byte_count]);
        }

        // De-interleave samples
        let mut channel_data: Vec<Vec<i32>> = vec![Vec::with_capacity(samples_per_channel); channels];
        for (i, &sample) in samples.iter().enumerate() {
            channel_data[i % channels].push(sample);
        }

        // Process in blocks
        let mut offset = 0;
        while offset < samples_per_channel {
            let remaining = samples_per_channel - offset;
            let current_block_size = remaining.min(block_size);

            let block: Vec<Vec<i32>> = channel_data.iter()
                .map(|ch| ch[offset..offset + current_block_size].to_vec())
                .collect();

            self.encode_frame(&block)?;
            offset += current_block_size;
        }

        self.samples_written += samples_per_channel as u64;

        Ok(())
    }

    fn encode_frame(&mut self, channel_data: &[Vec<i32>]) -> Result<()> {
        let block_size = channel_data[0].len();
        let channels = channel_data.len();

        let mut writer = BitWriter::new();

        // Frame header
        // Sync code (14 bits)
        writer.write_bits(0x3FFE, 14);

        // Reserved (1 bit)
        writer.write_bit(false);

        // Blocking strategy (1 bit): 0 = fixed block size
        writer.write_bit(false);

        // Block size (4 bits)
        let block_size_code = match block_size {
            192 => 1,
            576 => 2,
            1152 => 3,
            2304 => 4,
            4608 => 5,
            256 => 8,
            512 => 9,
            1024 => 10,
            2048 => 11,
            4096 => 12,
            8192 => 13,
            16384 => 14,
            32768 => 15,
            _ if block_size <= 256 => 6, // 8-bit block size
            _ => 7, // 16-bit block size
        };
        writer.write_bits(block_size_code, 4);

        // Sample rate (4 bits)
        let sample_rate_code = match self.stream_info.sample_rate {
            88200 => 1,
            176400 => 2,
            192000 => 3,
            8000 => 4,
            16000 => 5,
            22050 => 6,
            24000 => 7,
            32000 => 8,
            44100 => 9,
            48000 => 10,
            96000 => 11,
            _ => 0, // Use streaminfo
        };
        writer.write_bits(sample_rate_code, 4);

        // Channel assignment (4 bits)
        let channel_code = if channels == 2 {
            // Try stereo decorrelation
            10 // Mid/side
        } else {
            (channels - 1) as u32
        };
        writer.write_bits(channel_code, 4);

        // Sample size (3 bits)
        let sample_size_code = match self.stream_info.bits_per_sample {
            8 => 1,
            12 => 2,
            16 => 4,
            20 => 5,
            24 => 6,
            32 => 7,
            _ => 0, // Use streaminfo
        };
        writer.write_bits(sample_size_code, 3);

        // Reserved (1 bit)
        writer.write_bit(false);

        // Frame number (UTF-8 coded)
        writer.write_utf8_coded(self.frame_number);

        // Extended block size
        if block_size_code == 6 {
            writer.write_bits((block_size - 1) as u32, 8);
        } else if block_size_code == 7 {
            writer.write_bits((block_size - 1) as u32, 16);
        }

        // CRC-8 of header
        writer.align_to_byte();
        let header_bytes = writer.as_bytes();
        let crc8 = calculate_crc8(header_bytes);
        writer.write_bits(crc8 as u32, 8);

        // Encode subframes
        if channels == 2 && channel_code == 10 {
            // Mid/side encoding
            let left = &channel_data[0];
            let right = &channel_data[1];

            let mut mid = Vec::with_capacity(block_size);
            let mut side = Vec::with_capacity(block_size);

            for i in 0..block_size {
                mid.push((left[i] + right[i]) >> 1);
                side.push(left[i] - right[i]);
            }

            self.encode_subframe(&mut writer, &mid, self.stream_info.bits_per_sample)?;
            self.encode_subframe(&mut writer, &side, self.stream_info.bits_per_sample + 1)?;
        } else {
            for ch_data in channel_data {
                self.encode_subframe(&mut writer, ch_data, self.stream_info.bits_per_sample)?;
            }
        }

        // Align to byte boundary
        writer.align_to_byte();

        // CRC-16 of entire frame
        let frame_bytes = writer.as_bytes();
        let crc16 = calculate_crc16(frame_bytes);
        writer.write_bits((crc16 >> 8) as u32, 8);
        writer.write_bits((crc16 & 0xFF) as u32, 8);

        // Write frame
        self.writer.write_all(&writer.into_bytes())?;
        self.frame_number += 1;

        Ok(())
    }

    fn encode_subframe(&self, writer: &mut BitWriter, samples: &[i32], bps: u8) -> Result<()> {
        // Zero bit padding
        writer.write_bit(false);

        // Determine best encoding method
        let max_lpc_order = self.compression_level.max_lpc_order() as usize;

        // Try constant encoding first
        if samples.iter().all(|&s| s == samples[0]) {
            // Constant subframe
            writer.write_bits(0, 6); // Type = 0 (constant)
            writer.write_bit(false); // No wasted bits
            writer.write_signed(samples[0], bps);
            return Ok(());
        }

        // Try fixed prediction
        let (best_fixed_order, fixed_residual) = self.find_best_fixed_order(samples);
        let fixed_bits = self.estimate_rice_bits(&fixed_residual);

        // Try LPC if enabled
        let (lpc_order, lpc_coeffs, lpc_shift, lpc_residual) = if max_lpc_order > 0 {
            self.compute_lpc(samples, max_lpc_order)
        } else {
            (0, vec![], 0, vec![])
        };

        let lpc_bits = if lpc_order > 0 {
            self.estimate_rice_bits(&lpc_residual) + lpc_order * 16
        } else {
            usize::MAX
        };

        // Choose best method
        if lpc_order > 0 && lpc_bits < fixed_bits {
            // LPC subframe
            writer.write_bits(32 + (lpc_order - 1) as u32, 6);
            writer.write_bit(false); // No wasted bits

            // Warm-up samples
            for i in 0..lpc_order {
                writer.write_signed(samples[i], bps);
            }

            // LPC precision (4 bits)
            let precision = 15u8; // Use full precision
            writer.write_bits((precision - 1) as u32, 4);

            // LPC shift (5 bits, signed)
            writer.write_signed(lpc_shift, 5);

            // LPC coefficients
            for &coef in &lpc_coeffs {
                writer.write_signed(coef, precision);
            }

            // Encode residual
            self.encode_residual(writer, &lpc_residual, lpc_order)?;
        } else if best_fixed_order > 0 {
            // Fixed subframe
            writer.write_bits(8 + best_fixed_order as u32, 6);
            writer.write_bit(false); // No wasted bits

            // Warm-up samples
            for i in 0..best_fixed_order {
                writer.write_signed(samples[i], bps);
            }

            // Encode residual
            self.encode_residual(writer, &fixed_residual, best_fixed_order)?;
        } else {
            // Verbatim subframe
            writer.write_bits(1, 6);
            writer.write_bit(false); // No wasted bits

            for &sample in samples {
                writer.write_signed(sample, bps);
            }
        }

        Ok(())
    }

    fn find_best_fixed_order(&self, samples: &[i32]) -> (usize, Vec<i32>) {
        let mut best_order = 0;
        let mut best_residual = samples.to_vec();
        let mut best_bits = self.estimate_rice_bits(&best_residual);

        for order in 1..=4 {
            if order >= samples.len() {
                break;
            }

            let residual = self.compute_fixed_residual(samples, order);
            let bits = self.estimate_rice_bits(&residual);

            if bits < best_bits {
                best_bits = bits;
                best_order = order;
                best_residual = residual;
            }
        }

        (best_order, best_residual)
    }

    fn compute_fixed_residual(&self, samples: &[i32], order: usize) -> Vec<i32> {
        let coeffs: &[i64] = match order {
            1 => &[1],
            2 => &[2, -1],
            3 => &[3, -3, 1],
            4 => &[4, -6, 4, -1],
            _ => return samples.to_vec(),
        };

        let mut residual = Vec::with_capacity(samples.len() - order);

        for i in order..samples.len() {
            let mut prediction = 0i64;
            for (j, &coef) in coeffs.iter().enumerate() {
                prediction += coef * samples[i - 1 - j] as i64;
            }
            residual.push(samples[i] - prediction as i32);
        }

        residual
    }

    fn compute_lpc(&self, samples: &[i32], max_order: usize) -> (usize, Vec<i32>, i32, Vec<i32>) {
        if samples.len() <= max_order {
            return (0, vec![], 0, vec![]);
        }

        // Compute autocorrelation
        let order = max_order.min(samples.len() - 1).min(32);
        let mut autocorr = vec![0.0f64; order + 1];

        for i in 0..=order {
            for j in 0..samples.len() - i {
                autocorr[i] += samples[j] as f64 * samples[j + i] as f64;
            }
        }

        if autocorr[0] == 0.0 {
            return (0, vec![], 0, vec![]);
        }

        // Levinson-Durbin algorithm
        let mut lpc = vec![0.0f64; order];
        let mut error = autocorr[0];

        for i in 0..order {
            let mut lambda = 0.0f64;
            for j in 0..i {
                lambda -= lpc[j] * autocorr[i - j];
            }
            lambda -= autocorr[i + 1];
            lambda /= error;

            for j in 0..i / 2 {
                let tmp = lpc[j];
                lpc[j] -= lambda * lpc[i - 1 - j];
                lpc[i - 1 - j] -= lambda * tmp;
            }
            if i % 2 == 0 {
                lpc[i / 2] -= lambda * lpc[i / 2];
            }
            lpc[i] = -lambda;

            error *= 1.0 - lambda * lambda;
            if error <= 0.0 {
                break;
            }
        }

        // Find best order
        let mut best_order = order;
        for o in 1..=order {
            // Simple heuristic: use full order
            best_order = o;
        }

        // Quantize LPC coefficients
        let lpc_precision = 15;
        let mut max_coef = 0.0f64;
        for &c in &lpc[..best_order] {
            max_coef = max_coef.max(c.abs());
        }

        let shift = if max_coef > 0.0 {
            let ideal_shift = lpc_precision as f64 - 1.0 - max_coef.log2();
            ideal_shift.floor() as i32
        } else {
            0
        };
        let shift = shift.clamp(-16, 15);

        let scale = (1i64 << shift.abs()) as f64;
        let mut quantized_lpc = Vec::with_capacity(best_order);
        for &c in &lpc[..best_order] {
            let q = if shift >= 0 {
                (c * scale).round() as i32
            } else {
                (c / scale).round() as i32
            };
            quantized_lpc.push(q);
        }

        // Compute residual with quantized coefficients
        let mut residual = Vec::with_capacity(samples.len() - best_order);
        for i in best_order..samples.len() {
            let mut prediction = 0i64;
            for (j, &coef) in quantized_lpc.iter().enumerate() {
                prediction += coef as i64 * samples[i - 1 - j] as i64;
            }
            let predicted = if shift >= 0 {
                (prediction >> shift) as i32
            } else {
                (prediction << (-shift)) as i32
            };
            residual.push(samples[i] - predicted);
        }

        (best_order, quantized_lpc, shift, residual)
    }

    fn estimate_rice_bits(&self, residual: &[i32]) -> usize {
        if residual.is_empty() {
            return 0;
        }

        // Estimate bits using average absolute value
        let sum: u64 = residual.iter().map(|&r| r.unsigned_abs() as u64).sum();
        let avg = (sum as f64 / residual.len() as f64).max(1.0);

        // Estimated Rice parameter
        let k = (avg.log2()).ceil() as usize;
        let k = k.min(14);

        // Estimated bits: (avg/2^k + 1) + k per sample
        let unary_avg = (avg / (1 << k) as f64) + 1.0;
        ((unary_avg + k as f64) * residual.len() as f64) as usize
    }

    fn encode_residual(&self, writer: &mut BitWriter, residual: &[i32], predictor_order: usize) -> Result<()> {
        // Coding method: 0 = Rice (4-bit param)
        writer.write_bits(0, 2);

        // Find optimal partition order
        let block_size = residual.len() + predictor_order;
        let max_partition_order = self.find_max_partition_order(block_size, predictor_order);

        // Use partition order 0 for simplicity (can be optimized)
        let partition_order = max_partition_order.min(4);
        writer.write_bits(partition_order as u32, 4);

        let num_partitions = 1 << partition_order;
        let samples_per_partition = block_size >> partition_order;

        let mut offset = 0;
        for partition in 0..num_partitions {
            let partition_samples = if partition == 0 {
                samples_per_partition - predictor_order
            } else {
                samples_per_partition
            };

            let partition_data = &residual[offset..offset + partition_samples];

            // Find optimal Rice parameter
            let rice_param = self.find_optimal_rice_param(partition_data);

            if rice_param < 15 {
                writer.write_bits(rice_param as u32, 4);

                // Encode with Rice
                for &sample in partition_data {
                    writer.write_rice_signed(sample, rice_param);
                }
            } else {
                // Escape code - use raw encoding
                writer.write_bits(15, 4);
                let raw_bits = self.stream_info.bits_per_sample;
                writer.write_bits(raw_bits as u32, 5);

                for &sample in partition_data {
                    writer.write_signed(sample, raw_bits);
                }
            }

            offset += partition_samples;
        }

        Ok(())
    }

    fn find_max_partition_order(&self, block_size: usize, predictor_order: usize) -> usize {
        let mut order = 0;
        while order < 8 {
            let partition_samples = block_size >> (order + 1);
            if partition_samples < predictor_order {
                break;
            }
            order += 1;
        }
        order
    }

    fn find_optimal_rice_param(&self, samples: &[i32]) -> u8 {
        if samples.is_empty() {
            return 0;
        }

        // Calculate sum of absolute values
        let sum: u64 = samples.iter().map(|&s| s.unsigned_abs() as u64).sum();

        if sum == 0 {
            return 0;
        }

        // Estimate optimal parameter
        let avg = sum as f64 / samples.len() as f64;
        let k = (avg.log2()).ceil() as u8;

        k.min(14)
    }

    /// Finish encoding and write final metadata
    pub fn finish(mut self) -> Result<W> {
        if self.finished {
            return Ok(self.writer);
        }

        self.finished = true;
        self.stream_info.total_samples = self.samples_written;
        self.stream_info.md5_signature = self.md5_context.finalize();

        // Note: In a real implementation, we would seek back and update STREAMINFO
        // For now, we just flush

        self.writer.flush()?;
        Ok(self.writer)
    }

    /// Get the current stream info
    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    /// Get number of samples written
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }

    /// Get current frame number
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_level_from() {
        assert_eq!(CompressionLevel::from(0), CompressionLevel::Fastest);
        assert_eq!(CompressionLevel::from(3), CompressionLevel::Default);
        assert_eq!(CompressionLevel::from(8), CompressionLevel::Best);
    }

    #[test]
    fn test_compression_level_block_size() {
        assert_eq!(CompressionLevel::Fastest.block_size(), 1152);
        assert_eq!(CompressionLevel::Default.block_size(), 4096);
        assert_eq!(CompressionLevel::Best.block_size(), 4608);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1010, 4);
        writer.write_bits(0b1100, 4);
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10101100]);
    }

    #[test]
    fn test_bit_writer_unaligned() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b11001, 5);
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b10111001]);
    }

    #[test]
    fn test_encoder_creation() {
        let buffer = Vec::new();
        let encoder = FlacEncoder::new(buffer, 44100, 2, 16, CompressionLevel::Default);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_channels() {
        let buffer = Vec::new();
        let encoder = FlacEncoder::new(buffer, 44100, 0, 16, CompressionLevel::Default);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_rice_encoding() {
        let mut writer = BitWriter::new();

        // Test positive values
        writer.write_rice_signed(5, 2);

        // Test negative values
        writer.write_rice_signed(-3, 2);

        let bytes = writer.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_crc8() {
        let data = [0x00, 0x00];
        let crc = calculate_crc8(&data);
        assert_eq!(crc, 0x00);
    }

    #[test]
    fn test_crc16() {
        let data = [0x00, 0x00];
        let crc = calculate_crc16(&data);
        assert_eq!(crc, 0x0000);
    }

    #[test]
    fn test_md5() {
        let mut ctx = Md5Context::new();
        ctx.update(b"");
        let result = ctx.finalize();
        // MD5 of empty string
        assert_eq!(
            result,
            [0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04,
             0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e]
        );
    }
}
