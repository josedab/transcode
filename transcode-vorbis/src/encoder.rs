//! Vorbis encoder implementation.

use crate::codebook::Codebook;
use crate::error::{VorbisError, Result};
use crate::floor::Floor;
use crate::mdct::Mdct;
use crate::residue::Residue;

/// Vorbis encoder configuration.
#[derive(Debug, Clone)]
pub struct VorbisConfig {
    /// Sample rate (8000-192000).
    pub sample_rate: u32,
    /// Number of channels (1-8).
    pub channels: u8,
    /// Quality level (-2.0 to 10.0).
    pub quality: f32,
    /// Maximum bitrate (0 = unset).
    pub max_bitrate: u32,
    /// Nominal bitrate (0 = unset, use quality).
    pub nominal_bitrate: u32,
    /// Minimum bitrate (0 = unset).
    pub min_bitrate: u32,
    /// Short block size (64-8192, power of 2).
    pub block_size_short: u16,
    /// Long block size (64-8192, power of 2).
    pub block_size_long: u16,
}

impl VorbisConfig {
    /// Create a new configuration.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            quality: 5.0,
            max_bitrate: 0,
            nominal_bitrate: 0,
            min_bitrate: 0,
            block_size_short: 256,
            block_size_long: 2048,
        }
    }

    /// Set quality level (-2.0 to 10.0).
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(-2.0, 10.0);
        self
    }

    /// Set target bitrate for ABR mode.
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.nominal_bitrate = bitrate;
        self
    }

    /// Set bitrate range for constrained VBR.
    pub fn with_bitrate_range(mut self, min: u32, nominal: u32, max: u32) -> Self {
        self.min_bitrate = min;
        self.nominal_bitrate = nominal;
        self.max_bitrate = max;
        self
    }

    /// Set block sizes.
    pub fn with_block_sizes(mut self, short: u16, long: u16) -> Self {
        self.block_size_short = short;
        self.block_size_long = long;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate < 8000 || self.sample_rate > 192000 {
            return Err(VorbisError::UnsupportedSampleRate(self.sample_rate));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(VorbisError::UnsupportedChannels(self.channels));
        }
        if !self.block_size_short.is_power_of_two() || self.block_size_short < 64 {
            return Err(VorbisError::ConfigError("Invalid short block size".into()));
        }
        if !self.block_size_long.is_power_of_two() || self.block_size_long < self.block_size_short {
            return Err(VorbisError::ConfigError("Invalid long block size".into()));
        }
        Ok(())
    }

    /// Estimate bitrate from quality.
    pub fn estimated_bitrate(&self) -> u32 {
        if self.nominal_bitrate > 0 {
            return self.nominal_bitrate;
        }

        // Rough estimate: quality 0 = ~64kbps stereo, quality 10 = ~500kbps stereo
        let base_rate = 64000.0 + (self.quality + 2.0) * 36000.0;
        (base_rate * self.channels as f32 / 2.0) as u32
    }
}

/// Vorbis encoded packet.
#[derive(Debug, Clone)]
pub struct VorbisPacket {
    /// Packet data.
    pub data: Vec<u8>,
    /// Granule position (total samples).
    pub granule_position: u64,
    /// Is this a header packet?
    pub is_header: bool,
    /// Packet duration in samples.
    pub duration: u32,
}

/// Vorbis encoder statistics.
#[derive(Debug, Clone, Default)]
pub struct VorbisEncoderStats {
    /// Total packets encoded.
    pub packets_encoded: u64,
    /// Total bytes encoded.
    pub bytes_encoded: u64,
    /// Total samples encoded.
    pub samples_encoded: u64,
    /// Short blocks used.
    pub short_blocks: u64,
    /// Long blocks used.
    pub long_blocks: u64,
    /// Average bitrate (bits per second).
    pub average_bitrate: f64,
}

/// Vorbis encoder.
#[derive(Debug)]
pub struct VorbisEncoder {
    config: VorbisConfig,
    /// MDCT for short blocks.
    mdct_short: Mdct,
    /// MDCT for long blocks.
    mdct_long: Mdct,
    /// Floor encoder.
    floors: Vec<Floor>,
    /// Residue encoder.
    residues: Vec<Residue>,
    /// Codebooks.
    codebooks: Vec<Codebook>,
    /// Input sample buffer.
    input_buffer: Vec<Vec<f32>>,
    /// Previous window data.
    prev_window: Vec<Vec<f32>>,
    /// Whether headers have been output.
    headers_output: bool,
    /// Current granule position.
    granule_position: u64,
    /// Encoder statistics.
    stats: VorbisEncoderStats,
    /// Previous block was long.
    prev_long_block: bool,
}

impl VorbisEncoder {
    /// Create a new Vorbis encoder.
    pub fn new(config: VorbisConfig) -> Result<Self> {
        config.validate()?;

        let mdct_short = Mdct::new(config.block_size_short as usize);
        let mdct_long = Mdct::new(config.block_size_long as usize);

        let floors = vec![
            Floor::new_type1(config.block_size_short as usize),
            Floor::new_type1(config.block_size_long as usize),
        ];

        let residues = vec![
            Residue::new(2, config.block_size_short as usize),
            Residue::new(2, config.block_size_long as usize),
        ];

        let input_buffer = vec![Vec::new(); config.channels as usize];
        let prev_window = vec![
            vec![0.0f32; config.block_size_long as usize];
            config.channels as usize
        ];

        Ok(Self {
            config,
            mdct_short,
            mdct_long,
            floors,
            residues,
            codebooks: Vec::new(),
            input_buffer,
            prev_window,
            headers_output: false,
            granule_position: 0,
            stats: VorbisEncoderStats::default(),
            prev_long_block: false,
        })
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &VorbisConfig {
        &self.config
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> &VorbisEncoderStats {
        &self.stats
    }

    /// Get headers (ID, comment, setup).
    pub fn get_headers(&mut self) -> Result<Vec<VorbisPacket>> {
        if self.headers_output {
            return Ok(Vec::new());
        }

        let headers = vec![
            self.create_id_header()?,
            self.create_comment_header()?,
            self.create_setup_header()?,
        ];

        self.headers_output = true;
        Ok(headers)
    }

    /// Create identification header.
    fn create_id_header(&self) -> Result<VorbisPacket> {
        let mut data = Vec::with_capacity(30);

        // Packet type (1 = ID header)
        data.push(1);

        // Signature
        data.extend_from_slice(b"vorbis");

        // Version (0)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Channels
        data.push(self.config.channels);

        // Sample rate
        data.extend_from_slice(&self.config.sample_rate.to_le_bytes());

        // Bitrates (0 = unset)
        data.extend_from_slice(&(self.config.max_bitrate as i32).to_le_bytes());
        data.extend_from_slice(&(self.config.nominal_bitrate as i32).to_le_bytes());
        data.extend_from_slice(&(self.config.min_bitrate as i32).to_le_bytes());

        // Block sizes
        let bs0_log = self.config.block_size_short.trailing_zeros() as u8;
        let bs1_log = self.config.block_size_long.trailing_zeros() as u8;
        data.push(bs0_log | (bs1_log << 4));

        // Framing bit
        data.push(1);

        Ok(VorbisPacket {
            data,
            granule_position: 0,
            is_header: true,
            duration: 0,
        })
    }

    /// Create comment header.
    fn create_comment_header(&self) -> Result<VorbisPacket> {
        let mut data = Vec::new();

        // Packet type (3 = comment header)
        data.push(3);

        // Signature
        data.extend_from_slice(b"vorbis");

        // Vendor string
        let vendor = format!("transcode-vorbis {}", env!("CARGO_PKG_VERSION"));
        data.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor.as_bytes());

        // Number of comments (0 for now)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Framing bit
        data.push(1);

        Ok(VorbisPacket {
            data,
            granule_position: 0,
            is_header: true,
            duration: 0,
        })
    }

    /// Create setup header.
    fn create_setup_header(&self) -> Result<VorbisPacket> {
        let mut data = Vec::new();

        // Packet type (5 = setup header)
        data.push(5);

        // Signature
        data.extend_from_slice(b"vorbis");

        // Codebook count (placeholder)
        data.push(1); // 1 codebook

        // Codebook data (simplified)
        data.extend_from_slice(&[0x42, 0x43, 0x56]); // "BCV" sync
        data.extend_from_slice(&[1, 0]); // dimensions = 1
        data.extend_from_slice(&[0, 1, 0]); // entries = 256

        // ... additional setup data would go here

        // Framing bit
        data.push(1);

        Ok(VorbisPacket {
            data,
            granule_position: 0,
            is_header: true,
            duration: 0,
        })
    }

    /// Submit audio samples for encoding.
    pub fn encode(&mut self, samples: &[Vec<f32>]) -> Result<Vec<VorbisPacket>> {
        if samples.len() != self.config.channels as usize {
            return Err(VorbisError::InvalidData(format!(
                "Expected {} channels, got {}",
                self.config.channels,
                samples.len()
            )));
        }

        // Add samples to input buffer
        for (ch, ch_samples) in samples.iter().enumerate() {
            self.input_buffer[ch].extend_from_slice(ch_samples);
        }

        let mut packets = Vec::new();

        // Encode complete blocks
        while self.input_buffer[0].len() >= self.config.block_size_long as usize {
            if let Some(packet) = self.encode_block()? {
                packets.push(packet);
            }
        }

        Ok(packets)
    }

    /// Encode a single block.
    fn encode_block(&mut self) -> Result<Option<VorbisPacket>> {
        // Decide block size based on content (simplified: use long block)
        let use_long_block = self.should_use_long_block();
        let block_size = if use_long_block {
            self.config.block_size_long as usize
        } else {
            self.config.block_size_short as usize
        };

        if self.input_buffer[0].len() < block_size {
            return Ok(None);
        }

        let n2 = block_size / 2;
        let channels = self.config.channels as usize;

        // Get the MDCT and floor/residue for this block size
        let (mdct, floor, residue) = if use_long_block {
            (&self.mdct_long, &self.floors[1], &self.residues[1])
        } else {
            (&self.mdct_short, &self.floors[0], &self.residues[0])
        };

        let mut packet_data = Vec::new();

        // Block type flag
        let block_flag = if use_long_block { 1u8 } else { 0u8 };
        packet_data.push(block_flag << 1);

        // Previous/next block flags (for variable block size)
        if use_long_block {
            packet_data[0] |= (if self.prev_long_block { 1 } else { 0 }) << 2;
        }

        // Encode each channel
        let mut channel_spectra: Vec<Vec<f32>> = Vec::with_capacity(channels);
        let mut channel_floors: Vec<Vec<f32>> = Vec::with_capacity(channels);

        for ch in 0..channels {
            // Extract samples with windowing
            let samples: Vec<f32> = self.input_buffer[ch]
                .drain(..block_size)
                .collect();

            // Apply window
            let window = mdct.window();
            let windowed: Vec<f32> = samples
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward MDCT
            let mut spectrum = vec![0.0f32; n2];
            mdct.forward(&windowed, &mut spectrum);

            // Encode floor
            let floor_curve = floor.encode(&spectrum, &mut packet_data)?;

            channel_spectra.push(spectrum);
            channel_floors.push(floor_curve);
        }

        // Encode residue (after removing floor)
        residue.encode(&channel_spectra, &channel_floors, &mut packet_data)?;

        // Update statistics
        self.stats.packets_encoded += 1;
        self.stats.bytes_encoded += packet_data.len() as u64;
        self.stats.samples_encoded += n2 as u64;

        if use_long_block {
            self.stats.long_blocks += 1;
        } else {
            self.stats.short_blocks += 1;
        }

        self.granule_position += n2 as u64;
        self.prev_long_block = use_long_block;

        // Calculate average bitrate
        if self.stats.samples_encoded > 0 {
            self.stats.average_bitrate = (self.stats.bytes_encoded as f64 * 8.0
                * self.config.sample_rate as f64)
                / self.stats.samples_encoded as f64;
        }

        Ok(Some(VorbisPacket {
            data: packet_data,
            granule_position: self.granule_position,
            is_header: false,
            duration: n2 as u32,
        }))
    }

    /// Decide whether to use long or short block.
    fn should_use_long_block(&self) -> bool {
        // Simplified: analyze transient content
        // A real implementation would look for attacks/transients
        let ch0 = &self.input_buffer[0];
        if ch0.len() < self.config.block_size_long as usize {
            return false;
        }

        // Look for energy spikes that indicate transients
        let half = self.config.block_size_long as usize / 2;
        let first_half_energy: f32 = ch0[..half].iter().map(|s| s * s).sum();
        let second_half_energy: f32 = ch0[half..self.config.block_size_long as usize]
            .iter()
            .map(|s| s * s)
            .sum();

        // If energy ratio is relatively stable, use long block
        let ratio = if first_half_energy > 0.0001 {
            second_half_energy / first_half_energy
        } else {
            1.0
        };

        ratio > 0.3 && ratio < 3.0
    }

    /// Flush remaining samples.
    pub fn flush(&mut self) -> Result<Vec<VorbisPacket>> {
        let mut packets = Vec::new();

        // Pad input buffer to complete a block
        let remaining = self.input_buffer[0].len();
        if remaining > 0 {
            let block_size = self.config.block_size_short as usize;
            let padding = block_size - (remaining % block_size);
            if padding < block_size {
                for ch in &mut self.input_buffer {
                    ch.extend(std::iter::repeat(0.0).take(padding));
                }
            }

            // Encode remaining blocks
            while self.input_buffer[0].len() >= self.config.block_size_short as usize {
                if let Some(packet) = self.encode_block()? {
                    packets.push(packet);
                }
            }
        }

        Ok(packets)
    }

    /// Reset the encoder.
    pub fn reset(&mut self) {
        for ch in &mut self.input_buffer {
            ch.clear();
        }
        for ch in &mut self.prev_window {
            ch.fill(0.0);
        }
        self.granule_position = 0;
        self.stats = VorbisEncoderStats::default();
        self.prev_long_block = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = VorbisConfig::new(44100, 2);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_quality() {
        let config = VorbisConfig::new(44100, 2).with_quality(7.0);
        assert_eq!(config.quality, 7.0);

        let config = VorbisConfig::new(44100, 2).with_quality(15.0);
        assert_eq!(config.quality, 10.0); // Clamped
    }

    #[test]
    fn test_encoder_creation() {
        let config = VorbisConfig::new(44100, 2);
        let encoder = VorbisEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_headers_generation() {
        let config = VorbisConfig::new(44100, 2);
        let mut encoder = VorbisEncoder::new(config).unwrap();

        let headers = encoder.get_headers().unwrap();
        assert_eq!(headers.len(), 3);

        // Check ID header
        assert!(headers[0].is_header);
        assert_eq!(headers[0].data[0], 1);
        assert_eq!(&headers[0].data[1..7], b"vorbis");

        // Check comment header
        assert!(headers[1].is_header);
        assert_eq!(headers[1].data[0], 3);

        // Check setup header
        assert!(headers[2].is_header);
        assert_eq!(headers[2].data[0], 5);
    }

    #[test]
    fn test_encoding() {
        let config = VorbisConfig::new(44100, 2);
        let mut encoder = VorbisEncoder::new(config).unwrap();

        // Generate test samples
        let samples: Vec<Vec<f32>> = vec![
            (0..4096).map(|i| (i as f32 * 0.01).sin()).collect(),
            (0..4096).map(|i| (i as f32 * 0.01).cos()).collect(),
        ];

        let packets = encoder.encode(&samples).unwrap();
        assert!(!packets.is_empty());
    }

    #[test]
    fn test_estimated_bitrate() {
        let config = VorbisConfig::new(44100, 2).with_quality(5.0);
        let bitrate = config.estimated_bitrate();
        assert!(bitrate > 100000); // Should be reasonable for stereo
    }
}
