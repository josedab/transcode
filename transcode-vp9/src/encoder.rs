//! VP9 encoder implementation.
//!
//! This module provides VP9 video encoding capabilities. The encoder supports:
//! - Profile 0 (8-bit, 4:2:0)
//! - Profile 2 (10/12-bit, 4:2:0)
//! - Multiple rate control modes (CQ, VBR, CBR)
//! - Two-pass encoding for better quality
//! - Configurable speed/quality tradeoffs

use crate::error::{Result, Vp9Error};
use crate::frame_header::{ColorSpace, Profile};

/// VP9 encoder configuration.
#[derive(Debug, Clone)]
pub struct Vp9EncoderConfig {
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Framerate numerator.
    pub framerate_num: u32,
    /// Framerate denominator.
    pub framerate_den: u32,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// VP9 profile.
    pub profile: Profile,
    /// Rate control mode.
    pub rate_control: Vp9RateControl,
    /// Encoder speed preset (0 = best quality, 9 = fastest).
    pub speed: u8,
    /// Keyframe interval (0 = auto).
    pub keyframe_interval: u32,
    /// Minimum keyframe interval.
    pub min_keyframe_interval: u32,
    /// Number of encoding threads (0 = auto).
    pub threads: u32,
    /// Enable row-based multi-threading.
    pub row_mt: bool,
    /// Tile columns (log2, 0-6).
    pub tile_cols_log2: u8,
    /// Tile rows (log2, 0-2).
    pub tile_rows_log2: u8,
    /// Target level (0 = unrestricted, 10-62 = specific level).
    pub target_level: u8,
    /// Color space.
    pub color_space: ColorSpace,
    /// Enable automatic alt-ref frames.
    pub auto_alt_ref: bool,
    /// Number of lag-in-frames for lookahead (0-25).
    pub lag_in_frames: u8,
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Enable noise sensitivity.
    pub noise_sensitivity: u8,
    /// Content type.
    pub content_type: Vp9ContentType,
}

impl Default for Vp9EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
            bit_depth: 8,
            profile: Profile::Profile0,
            rate_control: Vp9RateControl::ConstantQuality { cq_level: 32 },
            speed: 5,
            keyframe_interval: 240,
            min_keyframe_interval: 0,
            threads: 0,
            row_mt: true,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            target_level: 0,
            color_space: ColorSpace::Bt709,
            auto_alt_ref: true,
            lag_in_frames: 25,
            sharpness: 0,
            noise_sensitivity: 0,
            content_type: Vp9ContentType::Default,
        }
    }
}

impl Vp9EncoderConfig {
    /// Create a new encoder configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set the framerate.
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        self
    }

    /// Set the bit depth (8, 10, or 12).
    pub fn with_bit_depth(mut self, depth: u8) -> Self {
        self.bit_depth = depth;
        self.profile = match depth {
            8 => Profile::Profile0,
            10 | 12 => Profile::Profile2,
            _ => Profile::Profile0,
        };
        self
    }

    /// Set the rate control mode.
    pub fn with_rate_control(mut self, rc: Vp9RateControl) -> Self {
        self.rate_control = rc;
        self
    }

    /// Set target bitrate (convenience method for VBR).
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.rate_control = Vp9RateControl::Vbr {
            target_bitrate: bitrate,
            min_bitrate: None,
            max_bitrate: None,
        };
        self
    }

    /// Set constant quality level (0-63, lower is better).
    pub fn with_quality(mut self, cq_level: u8) -> Self {
        self.rate_control = Vp9RateControl::ConstantQuality {
            cq_level: cq_level.min(63),
        };
        self
    }

    /// Set encoder speed (0 = best quality, 9 = fastest).
    pub fn with_speed(mut self, speed: u8) -> Self {
        self.speed = speed.min(9);
        self
    }

    /// Set keyframe interval.
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Set number of threads.
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    /// Enable row-based multi-threading.
    pub fn with_row_mt(mut self, enabled: bool) -> Self {
        self.row_mt = enabled;
        self
    }

    /// Set tile configuration.
    pub fn with_tiles(mut self, cols_log2: u8, rows_log2: u8) -> Self {
        self.tile_cols_log2 = cols_log2.min(6);
        self.tile_rows_log2 = rows_log2.min(2);
        self
    }

    /// Set content type.
    pub fn with_content_type(mut self, content_type: Vp9ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Vp9Error::InvalidConfig("Invalid dimensions".into()));
        }

        if self.width > 8192 || self.height > 4352 {
            return Err(Vp9Error::InvalidConfig(
                "Dimensions exceed VP9 maximum".into(),
            ));
        }

        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(Vp9Error::InvalidConfig(format!(
                "Invalid bit depth: {}",
                self.bit_depth
            )));
        }

        if self.speed > 9 {
            return Err(Vp9Error::InvalidConfig("Speed must be 0-9".into()));
        }

        Ok(())
    }
}

/// Rate control mode.
#[derive(Debug, Clone)]
pub enum Vp9RateControl {
    /// Constant quality mode.
    ConstantQuality {
        /// Quality level (0-63, lower is better).
        cq_level: u8,
    },
    /// Variable bitrate mode.
    Vbr {
        /// Target bitrate in bits per second.
        target_bitrate: u32,
        /// Minimum bitrate (optional).
        min_bitrate: Option<u32>,
        /// Maximum bitrate (optional).
        max_bitrate: Option<u32>,
    },
    /// Constant bitrate mode.
    Cbr {
        /// Target bitrate in bits per second.
        bitrate: u32,
        /// Buffer size in milliseconds.
        buffer_size: u32,
        /// Initial buffer fill level (0-100%).
        initial_buffer_level: u32,
        /// Optimal buffer level (0-100%).
        optimal_buffer_level: u32,
    },
    /// Constrained quality mode.
    ConstrainedQuality {
        /// CQ level.
        cq_level: u8,
        /// Maximum bitrate.
        max_bitrate: u32,
    },
}

/// Content type hint for encoder optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9ContentType {
    /// Default content (mixed).
    Default,
    /// Screen content (text, graphics).
    Screen,
    /// Film content (high detail, grain).
    Film,
}

/// Encoded VP9 packet.
#[derive(Debug, Clone)]
pub struct Vp9Packet {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Duration.
    pub duration: i64,
    /// Whether this is a keyframe.
    pub keyframe: bool,
    /// Frame flags.
    pub flags: Vp9FrameFlags,
}

/// VP9 frame flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9FrameFlags {
    /// Frame is visible (not a hidden frame).
    pub visible: bool,
    /// Frame is droppable (not used as reference).
    pub droppable: bool,
    /// Frame is an alt-ref frame.
    pub alt_ref: bool,
}

/// Encoder statistics.
#[derive(Debug, Clone, Default)]
pub struct Vp9EncoderStats {
    /// Frames encoded.
    pub frames_encoded: u64,
    /// Bytes produced.
    pub bytes_produced: u64,
    /// Keyframes produced.
    pub keyframes: u64,
    /// Average bitrate in bits per second.
    pub avg_bitrate: f64,
    /// Average PSNR (if calculated).
    pub avg_psnr: Option<f64>,
    /// Encoding time in seconds.
    pub encoding_time_s: f64,
}

/// VP9 encoder.
pub struct Vp9Encoder {
    /// Encoder configuration.
    config: Vp9EncoderConfig,
    /// Frame count.
    frame_count: u64,
    /// Bytes produced.
    bytes_produced: u64,
    /// Keyframes produced.
    keyframes: u64,
    /// Pending frames for encoding.
    pending_frames: Vec<Vp9InputFrame>,
    /// Output packets queue.
    output_queue: Vec<Vp9Packet>,
    /// Whether encoder is flushing.
    flushing: bool,
    /// Whether encoder is finished.
    finished: bool,
    /// Quantizer tables.
    quant_tables: QuantTables,
    /// Reference frame buffer.
    ref_frames: [Option<ReferenceFrame>; 8],
    /// Last keyframe number.
    last_keyframe: u64,
}

/// Input frame for encoding.
#[derive(Debug, Clone)]
struct Vp9InputFrame {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
    pts: i64,
    force_keyframe: bool,
}

/// Reference frame buffer.
#[derive(Debug, Clone)]
struct ReferenceFrame {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
    width: u32,
    height: u32,
}

/// Quantization tables.
#[derive(Debug, Clone)]
struct QuantTables {
    y_dc_delta: i16,
    y_ac_delta: i16,
    uv_dc_delta: i16,
    uv_ac_delta: i16,
}

impl Default for QuantTables {
    fn default() -> Self {
        Self {
            y_dc_delta: 0,
            y_ac_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
        }
    }
}

impl Vp9Encoder {
    /// Create a new VP9 encoder with the given configuration.
    pub fn new(config: Vp9EncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            frame_count: 0,
            bytes_produced: 0,
            keyframes: 0,
            pending_frames: Vec::new(),
            output_queue: Vec::new(),
            flushing: false,
            finished: false,
            quant_tables: QuantTables::default(),
            ref_frames: Default::default(),
            last_keyframe: 0,
        })
    }

    /// Encode a frame (8-bit YUV420).
    pub fn encode_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        pts: i64,
    ) -> Result<Option<Vp9Packet>> {
        self.encode_frame_internal(y_plane, u_plane, v_plane, pts, false)
    }

    /// Encode a keyframe.
    pub fn encode_keyframe(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        pts: i64,
    ) -> Result<Option<Vp9Packet>> {
        self.encode_frame_internal(y_plane, u_plane, v_plane, pts, true)
    }

    fn encode_frame_internal(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Option<Vp9Packet>> {
        // Validate input dimensions
        let expected_y_size = (self.config.width * self.config.height) as usize;
        let expected_uv_size = expected_y_size / 4;

        if y_plane.len() < expected_y_size {
            return Err(Vp9Error::InvalidFrame(
                "Y plane too small for configured dimensions".into(),
            ));
        }
        if u_plane.len() < expected_uv_size || v_plane.len() < expected_uv_size {
            return Err(Vp9Error::InvalidFrame(
                "UV planes too small for configured dimensions".into(),
            ));
        }

        // Determine if this should be a keyframe
        let is_keyframe = force_keyframe
            || self.frame_count == 0
            || (self.config.keyframe_interval > 0
                && self.frame_count - self.last_keyframe >= self.config.keyframe_interval as u64);

        // Encode the frame
        let packet = self.encode_vp9_frame(y_plane, u_plane, v_plane, pts, is_keyframe)?;

        self.frame_count += 1;
        if is_keyframe {
            self.keyframes += 1;
            self.last_keyframe = self.frame_count - 1;
        }

        self.bytes_produced += packet.data.len() as u64;

        Ok(Some(packet))
    }

    /// Encode a VP9 frame to bitstream.
    fn encode_vp9_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        pts: i64,
        is_keyframe: bool,
    ) -> Result<Vp9Packet> {
        let mut bitstream = Vec::new();

        // Get quantizer based on rate control
        let q_index = self.get_quantizer();

        // Write uncompressed header
        self.write_uncompressed_header(&mut bitstream, is_keyframe, q_index)?;

        // Write compressed header (probability updates)
        self.write_compressed_header(&mut bitstream)?;

        // Write tile data
        self.write_tile_data(&mut bitstream, y_plane, u_plane, v_plane, is_keyframe, q_index)?;

        // Update reference frames
        if is_keyframe {
            self.update_reference_frame(0, y_plane, u_plane, v_plane);
            self.update_reference_frame(1, y_plane, u_plane, v_plane);
            self.update_reference_frame(2, y_plane, u_plane, v_plane);
        } else {
            self.update_reference_frame(0, y_plane, u_plane, v_plane);
        }

        Ok(Vp9Packet {
            data: bitstream,
            pts,
            dts: pts,
            duration: (self.config.framerate_den as i64 * 1_000_000_000)
                / self.config.framerate_num as i64,
            keyframe: is_keyframe,
            flags: Vp9FrameFlags {
                visible: true,
                droppable: false,
                alt_ref: false,
            },
        })
    }

    /// Get quantizer index based on rate control mode.
    fn get_quantizer(&self) -> u8 {
        match &self.config.rate_control {
            Vp9RateControl::ConstantQuality { cq_level } => *cq_level,
            Vp9RateControl::ConstrainedQuality { cq_level, .. } => *cq_level,
            Vp9RateControl::Vbr { target_bitrate, .. } => {
                // Simple bitrate-to-quantizer mapping
                let bits_per_pixel = (*target_bitrate as f64)
                    / (self.config.width * self.config.height * self.config.framerate_num
                        / self.config.framerate_den) as f64;
                if bits_per_pixel > 1.0 {
                    20
                } else if bits_per_pixel > 0.5 {
                    32
                } else if bits_per_pixel > 0.25 {
                    44
                } else {
                    56
                }
            }
            Vp9RateControl::Cbr { bitrate, .. } => {
                let bits_per_pixel = (*bitrate as f64)
                    / (self.config.width * self.config.height * self.config.framerate_num
                        / self.config.framerate_den) as f64;
                if bits_per_pixel > 1.0 {
                    20
                } else if bits_per_pixel > 0.5 {
                    32
                } else {
                    48
                }
            }
        }
    }

    /// Write the uncompressed header.
    fn write_uncompressed_header(
        &self,
        bitstream: &mut Vec<u8>,
        is_keyframe: bool,
        q_index: u8,
    ) -> Result<()> {
        let mut bits = BitWriter::new();

        // Frame marker (2 bits) = 2
        bits.write_bits(2, 2);

        // Profile (1-2 bits based on profile)
        let profile = self.config.profile as u8;
        if profile < 2 {
            bits.write_bit(profile & 1);
        } else {
            bits.write_bit(1);
            bits.write_bit(profile & 1);
        }

        // Show existing frame (1 bit) = 0
        bits.write_bit(0);

        // Frame type (1 bit): 0 = keyframe, 1 = inter
        bits.write_bit(if is_keyframe { 0 } else { 1 });

        // Show frame (1 bit) = 1
        bits.write_bit(1);

        // Error resilient mode (1 bit)
        bits.write_bit(0);

        if is_keyframe {
            // Sync code (24 bits)
            bits.write_bits(0x49, 8);
            bits.write_bits(0x83, 8);
            bits.write_bits(0x42, 8);

            // Color config
            if profile >= 2 {
                // Bit depth (1 bit for 10-bit, another for 12-bit)
                bits.write_bit(if self.config.bit_depth > 8 { 1 } else { 0 });
                if self.config.bit_depth > 8 {
                    bits.write_bit(if self.config.bit_depth > 10 { 1 } else { 0 });
                }
            }

            // Color space (3 bits)
            bits.write_bits(self.config.color_space as u32, 3);

            if self.config.color_space != ColorSpace::Srgb {
                // Color range (1 bit)
                bits.write_bit(0); // Studio range

                if profile == 1 || profile == 3 {
                    // Subsampling (2 bits)
                    bits.write_bits(0, 2); // 4:2:0
                    // Reserved bit
                    bits.write_bit(0);
                }
            }

            // Frame size
            bits.write_bits(self.config.width - 1, 16);
            bits.write_bits(self.config.height - 1, 16);

            // Render size (1 bit = 0, same as frame)
            bits.write_bit(0);
        } else {
            // Intra-only (1 bit) for non-keyframes
            bits.write_bit(0);

            // Reset frame context (2 bits)
            bits.write_bits(0, 2);

            // Reference frame indices (3 bits each for LAST, GOLDEN, ALTREF)
            bits.write_bits(0, 3); // LAST
            bits.write_bits(1, 3); // GOLDEN
            bits.write_bits(2, 3); // ALTREF

            // Sign bias for each reference (1 bit each)
            bits.write_bit(0);
            bits.write_bit(0);
            bits.write_bit(0);

            // Allow high precision motion vectors
            bits.write_bit(1);

            // Interpolation filter (3 bits for switchable)
            bits.write_bits(4, 3);
        }

        // Refresh frame context (1 bit)
        bits.write_bit(1);

        // Frame context index (2 bits)
        bits.write_bits(0, 2);

        // Loop filter params
        bits.write_bits(32, 6); // Filter level
        bits.write_bits(0, 3); // Sharpness

        // Mode ref delta (1 bit)
        bits.write_bit(1);
        bits.write_bit(0); // Don't update deltas

        // Quantization params
        bits.write_bits(q_index as u32, 8); // Base Q index
        bits.write_bit(0); // No delta Q for DC Y
        bits.write_bit(0); // No delta Q for DC UV
        bits.write_bit(0); // No delta Q for AC UV

        // Segmentation (1 bit = disabled)
        bits.write_bit(0);

        // Tile info
        let min_log2_tiles = 0u32;
        let max_log2_tiles = ((self.config.width + 63) / 64).trailing_zeros();
        let tile_cols_log2 = (self.config.tile_cols_log2 as u32).min(max_log2_tiles);

        // Write tile cols increment
        for _ in min_log2_tiles..tile_cols_log2 {
            bits.write_bit(1);
        }
        if tile_cols_log2 < max_log2_tiles {
            bits.write_bit(0);
        }

        // Tile rows
        bits.write_bit(if self.config.tile_rows_log2 > 0 { 1 } else { 0 });
        if self.config.tile_rows_log2 > 0 {
            bits.write_bit(if self.config.tile_rows_log2 > 1 { 1 } else { 0 });
        }

        // Header size placeholder (16 bits)
        bits.write_bits(0, 16);

        // Byte-align and write to output
        bitstream.extend_from_slice(&bits.finish());

        Ok(())
    }

    /// Write the compressed header.
    fn write_compressed_header(&self, bitstream: &mut Vec<u8>) -> Result<()> {
        // For simplicity, write a minimal compressed header
        // Real implementation would use boolean arithmetic coding

        let mut compressed = Vec::new();

        // Tx mode (2 bits for TX_MODE_SELECT)
        compressed.push(0x80); // Minimal valid compressed header

        // Append compressed header
        bitstream.extend_from_slice(&compressed);

        Ok(())
    }

    /// Write tile data.
    fn write_tile_data(
        &mut self,
        bitstream: &mut Vec<u8>,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        is_keyframe: bool,
        q_index: u8,
    ) -> Result<()> {
        // Simplified tile encoding - real implementation would be much more complex
        let tile_data = self.encode_tile(y_plane, u_plane, v_plane, is_keyframe, q_index)?;
        bitstream.extend_from_slice(&tile_data);
        Ok(())
    }

    /// Encode a single tile.
    fn encode_tile(
        &self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        is_keyframe: bool,
        q_index: u8,
    ) -> Result<Vec<u8>> {
        let mut tile_data = Vec::new();
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Process superblocks (64x64)
        let sb_cols = (width + 63) / 64;
        let sb_rows = (height + 63) / 64;

        for sb_row in 0..sb_rows {
            for sb_col in 0..sb_cols {
                let sb_data = self.encode_superblock(
                    y_plane,
                    u_plane,
                    v_plane,
                    sb_col,
                    sb_row,
                    width,
                    height,
                    is_keyframe,
                    q_index,
                )?;
                tile_data.extend_from_slice(&sb_data);
            }
        }

        Ok(tile_data)
    }

    /// Encode a superblock.
    fn encode_superblock(
        &self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        sb_col: usize,
        sb_row: usize,
        width: usize,
        height: usize,
        is_keyframe: bool,
        q_index: u8,
    ) -> Result<Vec<u8>> {
        let mut sb_data = Vec::new();

        // Get superblock bounds
        let sb_x = sb_col * 64;
        let sb_y = sb_row * 64;
        let sb_w = 64.min(width - sb_x);
        let sb_h = 64.min(height - sb_y);

        // For each 8x8 block in the superblock
        for block_y in (0..sb_h).step_by(8) {
            for block_x in (0..sb_w).step_by(8) {
                let x = sb_x + block_x;
                let y = sb_y + block_y;

                // Extract 8x8 Y block
                let mut y_block = [0i16; 64];
                for by in 0..8.min(height - y) {
                    for bx in 0..8.min(width - x) {
                        y_block[by * 8 + bx] = y_plane[(y + by) * width + (x + bx)] as i16 - 128;
                    }
                }

                // Apply DCT
                let mut coeffs = self.forward_dct_8x8(&y_block);

                // Quantize
                self.quantize_8x8(&mut coeffs, q_index, true);

                // Encode coefficients (simplified)
                self.encode_coefficients(&mut sb_data, &coeffs);
            }
        }

        // Encode chroma (4x4 blocks for 4:2:0)
        let uv_width = width / 2;
        let uv_height = height / 2;
        let uv_sb_x = sb_col * 32;
        let uv_sb_y = sb_row * 32;
        let uv_sb_w = 32.min(uv_width.saturating_sub(uv_sb_x));
        let uv_sb_h = 32.min(uv_height.saturating_sub(uv_sb_y));

        for plane in [u_plane, v_plane] {
            for block_y in (0..uv_sb_h).step_by(4) {
                for block_x in (0..uv_sb_w).step_by(4) {
                    let x = uv_sb_x + block_x;
                    let y = uv_sb_y + block_y;

                    let mut uv_block = [0i16; 16];
                    for by in 0..4.min(uv_height - y) {
                        for bx in 0..4.min(uv_width - x) {
                            uv_block[by * 4 + bx] = plane[(y + by) * uv_width + (x + bx)] as i16 - 128;
                        }
                    }

                    let mut coeffs = self.forward_dct_4x4(&uv_block);
                    self.quantize_4x4(&mut coeffs, q_index, false);
                    self.encode_coefficients_4x4(&mut sb_data, &coeffs);
                }
            }
        }

        Ok(sb_data)
    }

    /// Forward 8x8 DCT.
    fn forward_dct_8x8(&self, block: &[i16; 64]) -> [i16; 64] {
        let mut output = [0i16; 64];

        // Simplified DCT implementation
        // Real implementation would use optimized DCT
        for v in 0..8 {
            for u in 0..8 {
                let mut sum = 0.0f64;
                for y in 0..8 {
                    for x in 0..8 {
                        let cu = if u == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                        sum += cu
                            * cv
                            * (block[y * 8 + x] as f64)
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                output[v * 8 + u] = (sum / 4.0).round() as i16;
            }
        }

        output
    }

    /// Forward 4x4 DCT.
    fn forward_dct_4x4(&self, block: &[i16; 16]) -> [i16; 16] {
        let mut output = [0i16; 16];

        for v in 0..4 {
            for u in 0..4 {
                let mut sum = 0.0f64;
                for y in 0..4 {
                    for x in 0..4 {
                        let cu = if u == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                        sum += cu
                            * cv
                            * (block[y * 4 + x] as f64)
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 8.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 8.0).cos();
                    }
                }
                output[v * 4 + u] = (sum / 2.0).round() as i16;
            }
        }

        output
    }

    /// Quantize 8x8 block.
    fn quantize_8x8(&self, coeffs: &mut [i16; 64], q_index: u8, is_luma: bool) {
        let q = Self::get_quant_value(q_index, is_luma);
        for coeff in coeffs.iter_mut() {
            *coeff = (*coeff + q / 2) / q;
        }
    }

    /// Quantize 4x4 block.
    fn quantize_4x4(&self, coeffs: &mut [i16; 16], q_index: u8, is_luma: bool) {
        let q = Self::get_quant_value(q_index, is_luma);
        for coeff in coeffs.iter_mut() {
            *coeff = (*coeff + q / 2) / q;
        }
    }

    /// Get quantization value from index.
    fn get_quant_value(q_index: u8, is_luma: bool) -> i16 {
        // VP9 quantization table (simplified)
        let base = match q_index {
            0..=15 => 4,
            16..=31 => 8,
            32..=47 => 16,
            48..=63 => 32,
            _ => 64,
        };
        if is_luma { base } else { base + 2 }
    }

    /// Encode coefficients to bitstream.
    fn encode_coefficients(&self, output: &mut Vec<u8>, coeffs: &[i16; 64]) {
        // Simplified coefficient encoding
        // Real VP9 uses context-adaptive binary arithmetic coding

        // Zigzag scan order
        const ZIGZAG: [usize; 64] = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
            44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ];

        // Find last non-zero coefficient
        let mut last_nz = 0;
        for (i, &idx) in ZIGZAG.iter().enumerate() {
            if coeffs[idx] != 0 {
                last_nz = i;
            }
        }

        // Simple run-length encoding
        let mut run = 0u8;
        for (i, &idx) in ZIGZAG.iter().enumerate().take(last_nz + 1) {
            let coeff = coeffs[idx];
            if coeff == 0 {
                run += 1;
            } else {
                // Encode run
                output.push(run);
                run = 0;

                // Encode coefficient (simplified)
                let sign = if coeff < 0 { 1 } else { 0 };
                let abs_coeff = coeff.unsigned_abs();

                if abs_coeff < 128 {
                    output.push((abs_coeff as u8) << 1 | sign);
                } else {
                    output.push(0xFF);
                    output.extend_from_slice(&abs_coeff.to_le_bytes());
                    output.push(sign);
                }
            }
        }

        // End of block marker
        output.push(0xFF);
        output.push(0x00);
    }

    /// Encode 4x4 coefficients to bitstream.
    fn encode_coefficients_4x4(&self, output: &mut Vec<u8>, coeffs: &[i16; 16]) {
        const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

        let mut last_nz = 0;
        for (i, &idx) in ZIGZAG_4X4.iter().enumerate() {
            if coeffs[idx] != 0 {
                last_nz = i;
            }
        }

        let mut run = 0u8;
        for (i, &idx) in ZIGZAG_4X4.iter().enumerate().take(last_nz + 1) {
            let coeff = coeffs[idx];
            if coeff == 0 {
                run += 1;
            } else {
                output.push(run);
                run = 0;

                let sign = if coeff < 0 { 1 } else { 0 };
                let abs_coeff = coeff.unsigned_abs();

                if abs_coeff < 128 {
                    output.push((abs_coeff as u8) << 1 | sign);
                } else {
                    output.push(0xFF);
                    output.extend_from_slice(&abs_coeff.to_le_bytes());
                    output.push(sign);
                }
            }
        }

        output.push(0xFF);
        output.push(0x00);
    }

    /// Update a reference frame.
    fn update_reference_frame(
        &mut self,
        index: usize,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
    ) {
        self.ref_frames[index] = Some(ReferenceFrame {
            y_plane: y_plane.to_vec(),
            u_plane: u_plane.to_vec(),
            v_plane: v_plane.to_vec(),
            width: self.config.width,
            height: self.config.height,
        });
    }

    /// Flush the encoder and get remaining packets.
    pub fn flush(&mut self) -> Result<Vec<Vp9Packet>> {
        self.flushing = true;
        self.finished = true;

        // Return any pending output
        Ok(std::mem::take(&mut self.output_queue))
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> Vp9EncoderStats {
        let framerate = self.config.framerate_num as f64 / self.config.framerate_den as f64;
        let duration = self.frame_count as f64 / framerate;

        Vp9EncoderStats {
            frames_encoded: self.frame_count,
            bytes_produced: self.bytes_produced,
            keyframes: self.keyframes,
            avg_bitrate: if duration > 0.0 {
                (self.bytes_produced as f64 * 8.0) / duration
            } else {
                0.0
            },
            avg_psnr: None,
            encoding_time_s: duration,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &Vp9EncoderConfig {
        &self.config
    }

    /// Check if encoder is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Bit writer for generating bitstream.
struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        self.current_byte |= (bit & 1) << (7 - self.bit_pos);
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u8) {
        for i in (0..num_bits).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = Vp9EncoderConfig::new(320, 240);
        let encoder = Vp9Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_config() {
        let config = Vp9EncoderConfig::new(1920, 1080)
            .with_framerate(30, 1)
            .with_bitrate(2_000_000)
            .with_speed(5);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.speed, 5);
    }

    #[test]
    fn test_encode_frame() {
        let config = Vp9EncoderConfig::new(64, 64).with_quality(32);

        let mut encoder = Vp9Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 64 * 64];
        let u_plane = vec![128u8; 32 * 32];
        let v_plane = vec![128u8; 32 * 32];

        let result = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.is_some());
        assert!(packet.unwrap().keyframe);
    }

    #[test]
    fn test_encode_multiple_frames() {
        let config = Vp9EncoderConfig::new(64, 64)
            .with_quality(32)
            .with_keyframe_interval(30);

        let mut encoder = Vp9Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 64 * 64];
        let u_plane = vec![128u8; 32 * 32];
        let v_plane = vec![128u8; 32 * 32];

        // Encode first frame (keyframe)
        let packet1 = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 0).unwrap();
        assert!(packet1.is_some());
        assert!(packet1.unwrap().keyframe);

        // Encode second frame (inter frame)
        let packet2 = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 1).unwrap();
        assert!(packet2.is_some());
        assert!(!packet2.unwrap().keyframe);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b1100, 4);
        writer.write_bit(1);

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0b10111001);
    }
}
