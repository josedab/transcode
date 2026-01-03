//! VP9 decoder implementation.
//!
//! This module provides the main VP9 decoder that implements the VideoDecoder trait.
//! It handles:
//! - Superframe parsing
//! - Frame header parsing
//! - Reference frame management
//! - Tile decoding
//! - Block reconstruction

use std::collections::VecDeque;

use transcode_core::{
    Frame, FrameBuffer, PixelFormat, TimeBase, Timestamp,
    frame::FrameFlags,
};

use crate::entropy::{BoolDecoder, ProbabilityContext};
use crate::error::{Result as Vp9Result, Vp9Error};
use crate::frame_header::{
    ChromaSubsampling, ColorRange, ColorSpace, FrameHeader, Profile,
};
use crate::prediction::{BlockSize, IntraMode, IntraPredictor, LoopFilter, MotionVector};

/// VP9 decoder configuration.
#[derive(Debug, Clone)]
pub struct Vp9DecoderConfig {
    /// Maximum number of threads for parallel decoding.
    pub max_threads: usize,
    /// Enable error concealment.
    pub error_concealment: bool,
    /// Output buffer pool size.
    pub output_pool_size: usize,
    /// Enable loop filter.
    pub enable_loop_filter: bool,
}

impl Default for Vp9DecoderConfig {
    fn default() -> Self {
        Self {
            max_threads: 1,
            error_concealment: true,
            output_pool_size: 8,
            enable_loop_filter: true,
        }
    }
}

/// Reference frame slot.
#[derive(Debug, Clone)]
struct RefFrameSlot {
    /// Frame buffer.
    buffer: Option<FrameBuffer>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Bit depth.
    bit_depth: u8,
    /// Subsampling.
    subsampling: ChromaSubsampling,
}

impl Default for RefFrameSlot {
    fn default() -> Self {
        Self {
            buffer: None,
            width: 0,
            height: 0,
            bit_depth: 8,
            subsampling: ChromaSubsampling::Cs420,
        }
    }
}

impl RefFrameSlot {
    /// Check if the reference frame is valid and has the expected dimensions.
    #[inline]
    pub fn is_valid_for_size(&self, width: u32, height: u32) -> bool {
        self.buffer.is_some() && self.width == width && self.height == height
    }

    /// Check if the reference frame matches the given format.
    #[inline]
    pub fn matches_format(&self, bit_depth: u8, subsampling: ChromaSubsampling) -> bool {
        self.bit_depth == bit_depth && self.subsampling == subsampling
    }

    /// Get the dimensions of this reference frame.
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// VP9 decoder state.
#[derive(Debug, Clone, Default)]
struct DecoderState {
    /// Current frame width.
    width: u32,
    /// Current frame height.
    height: u32,
    /// Current bit depth.
    bit_depth: u8,
    /// Current profile.
    profile: Profile,
    /// Current color space.
    color_space: ColorSpace,
    /// Current color range.
    color_range: ColorRange,
    /// Current subsampling.
    subsampling: ChromaSubsampling,
    /// Frame number counter.
    frame_num: u64,
    /// Show frame counter.
    show_frame_num: u64,
    /// Probability context.
    prob_ctx: ProbabilityContext,
}

/// VP9 decoder.
pub struct Vp9Decoder {
    /// Decoder configuration.
    config: Vp9DecoderConfig,
    /// Decoder state.
    state: DecoderState,
    /// Reference frame slots (8 slots).
    ref_frames: [RefFrameSlot; 8],
    /// Output frame queue.
    output_queue: VecDeque<Frame>,
    /// Time base for output timestamps.
    time_base: TimeBase,
    /// Initialized flag.
    initialized: bool,
    /// Last decoded header.
    last_header: Option<FrameHeader>,
}

impl Vp9Decoder {
    /// Create a new VP9 decoder.
    pub fn new(config: Vp9DecoderConfig) -> Self {
        Self {
            config,
            state: DecoderState::default(),
            ref_frames: Default::default(),
            output_queue: VecDeque::new(),
            time_base: TimeBase::new(1, 90000),
            initialized: false,
            last_header: None,
        }
    }

    /// Create with default configuration.
    pub fn new_default() -> Self {
        Self::new(Vp9DecoderConfig::default())
    }

    /// Set the time base for output timestamps.
    pub fn set_time_base(&mut self, time_base: TimeBase) {
        self.time_base = time_base;
    }

    /// Get the current width.
    pub fn width(&self) -> Option<u32> {
        if self.initialized {
            Some(self.state.width)
        } else {
            None
        }
    }

    /// Get the current height.
    pub fn height(&self) -> Option<u32> {
        if self.initialized {
            Some(self.state.height)
        } else {
            None
        }
    }

    /// Get the current bit depth.
    pub fn bit_depth(&self) -> u8 {
        self.state.bit_depth
    }

    /// Get the current profile.
    pub fn profile(&self) -> Profile {
        self.state.profile
    }

    /// Decode a VP9 frame.
    pub fn decode_frame(&mut self, data: &[u8]) -> Vp9Result<Vec<Frame>> {
        // Handle superframes (multiple frames in one packet)
        let frames = self.parse_superframe(data)?;
        let mut output = Vec::new();

        for frame_data in frames {
            if let Some(frame) = self.decode_single_frame(frame_data)? {
                output.push(frame);
            }
        }

        Ok(output)
    }

    /// Parse superframe index (if present).
    fn parse_superframe<'a>(&self, data: &'a [u8]) -> Vp9Result<Vec<&'a [u8]>> {
        if data.is_empty() {
            return Err(Vp9Error::UnexpectedEndOfStream);
        }

        // Check for superframe marker at end
        let marker = data[data.len() - 1];
        if (marker & 0xE0) != 0xC0 {
            // Not a superframe, return single frame
            return Ok(vec![data]);
        }

        let bytes_per_framesize = ((marker >> 3) & 0x03) as usize + 1;
        let frame_count = (marker & 0x07) as usize + 1;
        let index_size = 2 + frame_count * bytes_per_framesize;

        if data.len() < index_size {
            return Err(Vp9Error::SuperframeError("Invalid superframe index size".into()));
        }

        let index_start = data.len() - index_size;

        // Verify marker at start of index
        if data[index_start] != marker {
            return Err(Vp9Error::SuperframeError("Invalid superframe marker".into()));
        }

        // Parse frame sizes
        let mut frames = Vec::with_capacity(frame_count);
        let mut offset = 0;

        for i in 0..frame_count {
            let size_offset = index_start + 1 + i * bytes_per_framesize;
            let mut frame_size = 0usize;
            for j in 0..bytes_per_framesize {
                frame_size |= (data[size_offset + j] as usize) << (j * 8);
            }

            if offset + frame_size > index_start {
                return Err(Vp9Error::SuperframeError("Frame size exceeds data".into()));
            }

            frames.push(&data[offset..offset + frame_size]);
            offset += frame_size;
        }

        Ok(frames)
    }

    /// Decode a single VP9 frame.
    fn decode_single_frame(&mut self, data: &[u8]) -> Vp9Result<Option<Frame>> {
        // Parse uncompressed header
        let (header, header_bytes) = FrameHeader::parse(data)?;

        // Handle show existing frame
        if header.show_existing_frame {
            return self.show_existing_frame(&header);
        }

        // Update state from header
        self.update_state(&header)?;

        // Parse compressed header
        let compressed_start = header_bytes;
        let compressed_end = compressed_start + header.header_size as usize;

        if compressed_end > data.len() {
            return Err(Vp9Error::UnexpectedEndOfStream);
        }

        header.parse_compressed_header(
            &data[compressed_start..],
            &mut self.state.prob_ctx,
        )?;

        // Decode tile data
        let tile_data = &data[compressed_end..];
        let frame = self.decode_tiles(&header, tile_data)?;

        // Apply loop filter if enabled
        let mut frame = frame;
        if self.config.enable_loop_filter && header.loop_filter.level > 0 {
            self.apply_loop_filter(&mut frame, &header);
        }

        // Update reference frames
        self.update_ref_frames(&header, &frame);

        // Store header
        self.last_header = Some(header.clone());
        self.state.frame_num += 1;

        if header.show_frame {
            self.state.show_frame_num += 1;
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    /// Handle show existing frame.
    fn show_existing_frame(&mut self, header: &FrameHeader) -> Vp9Result<Option<Frame>> {
        let idx = header.frame_to_show_map_idx as usize;
        if idx >= self.ref_frames.len() {
            return Err(Vp9Error::InvalidRefFrameIndex(header.frame_to_show_map_idx));
        }

        let ref_slot = &self.ref_frames[idx];
        if ref_slot.buffer.is_none() {
            return Err(Vp9Error::MissingRefFrame(header.frame_to_show_map_idx));
        }

        // Clone the reference frame buffer
        let buffer = ref_slot.buffer.clone().unwrap();
        let mut frame = Frame::from_buffer(buffer);
        frame.pts = Timestamp::new(self.state.show_frame_num as i64, self.time_base);

        self.state.show_frame_num += 1;
        Ok(Some(frame))
    }

    /// Update decoder state from header.
    fn update_state(&mut self, header: &FrameHeader) -> Vp9Result<()> {
        self.state.width = header.width;
        self.state.height = header.height;
        self.state.bit_depth = header.bit_depth;
        self.state.profile = header.profile;
        self.state.color_space = header.color_space;
        self.state.color_range = header.color_range;
        self.state.subsampling = header.subsampling;

        // Reset probabilities on keyframe
        if header.is_keyframe() {
            self.state.prob_ctx.reset();
        }

        self.initialized = true;
        Ok(())
    }

    /// Decode tiles and reconstruct frame.
    fn decode_tiles(&mut self, header: &FrameHeader, tile_data: &[u8]) -> Vp9Result<Frame> {
        let pixel_format = self.get_pixel_format(header);
        let mut frame = Frame::new(header.width, header.height, pixel_format, self.time_base);

        frame.pts = Timestamp::new(self.state.show_frame_num as i64, self.time_base);
        if header.is_keyframe() {
            frame.flags.insert(FrameFlags::KEYFRAME);
        }

        // Calculate tile dimensions
        let tile_cols = header.tile_info.tile_cols as usize;
        let tile_rows = header.tile_info.tile_rows as usize;
        let sb_cols = header.sb_cols() as usize;
        let sb_rows = header.sb_rows() as usize;

        // Parse tile sizes and decode
        let mut offset = 0;
        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                let is_last = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;

                let tile_size = if is_last {
                    tile_data.len() - offset
                } else {
                    if offset + 4 > tile_data.len() {
                        return Err(Vp9Error::UnexpectedEndOfStream);
                    }
                    let size = u32::from_le_bytes([
                        tile_data[offset],
                        tile_data[offset + 1],
                        tile_data[offset + 2],
                        tile_data[offset + 3],
                    ]) as usize;
                    offset += 4;
                    size
                };

                if offset + tile_size > tile_data.len() {
                    return Err(Vp9Error::UnexpectedEndOfStream);
                }

                self.decode_tile(
                    header,
                    &tile_data[offset..offset + tile_size],
                    &mut frame,
                    tile_row,
                    tile_col,
                    sb_cols,
                    sb_rows,
                )?;

                offset += tile_size;
            }
        }

        Ok(frame)
    }

    /// Decode a single tile.
    #[allow(clippy::too_many_arguments)]
    fn decode_tile(
        &mut self,
        header: &FrameHeader,
        data: &[u8],
        frame: &mut Frame,
        tile_row: usize,
        tile_col: usize,
        sb_cols: usize,
        sb_rows: usize,
    ) -> Vp9Result<()> {
        let tile_cols = header.tile_info.tile_cols as usize;
        let tile_rows = header.tile_info.tile_rows as usize;

        // Calculate tile boundaries
        let sb_col_start = (tile_col * sb_cols) / tile_cols;
        let sb_col_end = ((tile_col + 1) * sb_cols) / tile_cols;
        let sb_row_start = (tile_row * sb_rows) / tile_rows;
        let sb_row_end = ((tile_row + 1) * sb_rows) / tile_rows;

        // Create bool decoder for tile
        let mut decoder = BoolDecoder::new(data)?;

        // Decode superblocks
        for sb_row in sb_row_start..sb_row_end {
            for sb_col in sb_col_start..sb_col_end {
                self.decode_superblock(
                    header,
                    &mut decoder,
                    frame,
                    sb_row,
                    sb_col,
                )?;
            }
        }

        Ok(())
    }

    /// Decode a superblock (64x64).
    fn decode_superblock(
        &mut self,
        header: &FrameHeader,
        decoder: &mut BoolDecoder,
        frame: &mut Frame,
        sb_row: usize,
        sb_col: usize,
    ) -> Vp9Result<()> {
        let x = sb_col * 64;
        let y = sb_row * 64;

        // Decode partition tree recursively
        self.decode_partition(
            header,
            decoder,
            frame,
            x as u32,
            y as u32,
            BlockSize::Block64x64,
        )?;

        Ok(())
    }

    /// Decode partition recursively.
    fn decode_partition(
        &mut self,
        header: &FrameHeader,
        decoder: &mut BoolDecoder,
        frame: &mut Frame,
        x: u32,
        y: u32,
        block_size: BlockSize,
    ) -> Vp9Result<()> {
        // Check bounds
        if x >= header.width || y >= header.height {
            return Ok(());
        }

        let _width = block_size.width();
        let _height = block_size.height();

        // For keyframes, decode intra blocks directly
        if header.is_intra_only() {
            self.decode_intra_block(header, decoder, frame, x, y, block_size)?;
        } else {
            // Inter frame - decode mode and motion vectors
            self.decode_inter_block(header, decoder, frame, x, y, block_size)?;
        }

        Ok(())
    }

    /// Decode an intra block.
    fn decode_intra_block(
        &mut self,
        header: &FrameHeader,
        _decoder: &mut BoolDecoder,
        frame: &mut Frame,
        x: u32,
        y: u32,
        block_size: BlockSize,
    ) -> Vp9Result<()> {
        let width = block_size.width().min(header.width - x) as usize;
        let height = block_size.height().min(header.height - y) as usize;

        // Get prediction mode (simplified - use DC for now)
        let mode = IntraMode::Dc;

        // Get neighbors
        let stride = frame.stride(0);
        let plane = frame.plane_mut(0).ok_or(Vp9Error::NotInitialized)?;

        let have_above = y > 0;
        let have_left = x > 0;

        // Build neighbor arrays
        let mut above = vec![128u8; width + 1];
        let mut left = vec![128u8; height + 1];
        let mut above_left = 128u8;

        if have_above {
            let above_row = (y - 1) as usize * stride + x as usize;
            let copy_width = width.min(header.width as usize - x as usize);
            above[..copy_width].copy_from_slice(&plane[above_row..above_row + copy_width]);
        }

        if have_left {
            for i in 0..height.min(header.height as usize - y as usize) {
                let idx = (y as usize + i) * stride + (x - 1) as usize;
                left[i] = plane[idx];
            }
        }

        if have_above && have_left {
            above_left = plane[(y - 1) as usize * stride + (x - 1) as usize];
        }

        // Create temporary buffer for prediction
        let mut pred = vec![0u8; width * height];

        IntraPredictor::predict(
            &mut pred,
            width,
            &above,
            &left,
            above_left,
            mode,
            width,
            height,
            have_above,
            have_left,
        );

        // Copy prediction to frame
        let plane = frame.plane_mut(0).ok_or(Vp9Error::NotInitialized)?;
        for row in 0..height {
            let dst_offset = (y as usize + row) * stride + x as usize;
            let src_offset = row * width;
            for col in 0..width {
                if x as usize + col < header.width as usize && y as usize + row < header.height as usize {
                    plane[dst_offset + col] = pred[src_offset + col];
                }
            }
        }

        // Handle chroma planes (simplified - copy luma pattern scaled)
        if header.subsampling == ChromaSubsampling::Cs420 {
            let chroma_width = width.div_ceil(2);
            let chroma_height = height.div_ceil(2);
            let chroma_x = (x / 2) as usize;
            let chroma_y = (y / 2) as usize;

            for plane_idx in 1..3 {
                let chroma_stride = frame.stride(plane_idx);
                if let Some(chroma_plane) = frame.plane_mut(plane_idx) {
                    for row in 0..chroma_height {
                        for col in 0..chroma_width {
                            let dst_offset = (chroma_y + row) * chroma_stride + chroma_x + col;
                            if dst_offset < chroma_plane.len() {
                                chroma_plane[dst_offset] = 128; // Neutral chroma
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode an inter block.
    fn decode_inter_block(
        &mut self,
        header: &FrameHeader,
        decoder: &mut BoolDecoder,
        frame: &mut Frame,
        x: u32,
        y: u32,
        block_size: BlockSize,
    ) -> Vp9Result<()> {
        // For inter blocks, we need reference frames
        // Simplified implementation - use zero motion from LAST ref

        let ref_idx = header.ref_frame_idx[0] as usize;
        if ref_idx >= self.ref_frames.len() {
            return Err(Vp9Error::InvalidRefFrameIndex(ref_idx as u8));
        }

        let ref_slot = &self.ref_frames[ref_idx];
        if ref_slot.buffer.is_none() {
            // No reference available, fall back to intra
            return self.decode_intra_block(header, decoder, frame, x, y, block_size);
        }

        // Validate reference frame dimensions and format
        if !ref_slot.is_valid_for_size(header.width, header.height) {
            // Reference has different dimensions, may need scaling
            let (ref_w, ref_h) = ref_slot.dimensions();
            tracing::debug!("Reference frame size {}x{} differs from current {}x{}",
                ref_w, ref_h, header.width, header.height);
        }

        if !ref_slot.matches_format(header.bit_depth, header.subsampling) {
            tracing::warn!("Reference frame format mismatch");
        }

        let ref_buffer = ref_slot.buffer.as_ref().unwrap();
        let width = block_size.width().min(header.width - x) as usize;
        let height = block_size.height().min(header.height - y) as usize;

        // Use zero motion vector
        let _mv = MotionVector::zero();

        // Copy from reference (simplified - no interpolation for zero MV)
        let dst_stride = frame.stride(0);
        let src_stride = ref_buffer.stride(0);

        if let (Some(dst_plane), Some(src_plane)) = (frame.plane_mut(0), ref_buffer.plane(0)) {
            for row in 0..height {
                let dst_offset = (y as usize + row) * dst_stride + x as usize;
                let src_offset = (y as usize + row) * src_stride + x as usize;
                for col in 0..width {
                    if dst_offset + col < dst_plane.len() && src_offset + col < src_plane.len() {
                        dst_plane[dst_offset + col] = src_plane[src_offset + col];
                    }
                }
            }
        }

        // Handle chroma planes
        if header.subsampling == ChromaSubsampling::Cs420 {
            let chroma_width = width.div_ceil(2);
            let chroma_height = height.div_ceil(2);
            let chroma_x = (x / 2) as usize;
            let chroma_y = (y / 2) as usize;

            for plane_idx in 1..3 {
                let dst_stride = frame.stride(plane_idx);
                let src_stride = ref_buffer.stride(plane_idx);

                if let (Some(dst_plane), Some(src_plane)) = (frame.plane_mut(plane_idx), ref_buffer.plane(plane_idx)) {
                    for row in 0..chroma_height {
                        let dst_offset = (chroma_y + row) * dst_stride + chroma_x;
                        let src_offset = (chroma_y + row) * src_stride + chroma_x;
                        for col in 0..chroma_width {
                            if dst_offset + col < dst_plane.len() && src_offset + col < src_plane.len() {
                                dst_plane[dst_offset + col] = src_plane[src_offset + col];
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply loop filter to frame.
    fn apply_loop_filter(&self, frame: &mut Frame, header: &FrameHeader) {
        let level = header.loop_filter.level;
        let sharpness = header.loop_filter.sharpness;

        if level == 0 {
            return;
        }

        let stride = frame.stride(0);
        let width = header.width as usize;
        let height = header.height as usize;

        if let Some(plane) = frame.plane_mut(0) {
            // Apply vertical edge filtering
            for x in (8..width).step_by(8) {
                LoopFilter::filter_vertical_edge(plane, stride, x, 0, height, level, sharpness);
            }

            // Apply horizontal edge filtering
            for y in (8..height).step_by(8) {
                LoopFilter::filter_horizontal_edge(plane, stride, 0, y, width, level, sharpness);
            }
        }
    }

    /// Update reference frames based on header flags.
    fn update_ref_frames(&mut self, header: &FrameHeader, frame: &Frame) {
        for i in 0..8 {
            if (header.refresh_frame_flags >> i) & 1 != 0 {
                self.ref_frames[i] = RefFrameSlot {
                    buffer: Some(frame.buffer().clone()),
                    width: header.width,
                    height: header.height,
                    bit_depth: header.bit_depth,
                    subsampling: header.subsampling,
                };
            }
        }
    }

    /// Get pixel format based on header.
    fn get_pixel_format(&self, header: &FrameHeader) -> PixelFormat {
        match (header.subsampling, header.bit_depth) {
            (ChromaSubsampling::Cs420, 8) => PixelFormat::Yuv420p,
            (ChromaSubsampling::Cs420, 10) => PixelFormat::Yuv420p10le,
            (ChromaSubsampling::Cs422, 8) => PixelFormat::Yuv422p,
            (ChromaSubsampling::Cs422, 10) => PixelFormat::Yuv422p10le,
            (ChromaSubsampling::Cs444, 8) => PixelFormat::Yuv444p,
            (ChromaSubsampling::Cs444, 10) => PixelFormat::Yuv444p10le,
            _ => PixelFormat::Yuv420p, // Fallback
        }
    }

    /// Flush any buffered frames.
    pub fn flush_decoder(&mut self) -> Vp9Result<Vec<Frame>> {
        let frames: Vec<Frame> = self.output_queue.drain(..).collect();
        Ok(frames)
    }

    /// Reset decoder state.
    pub fn reset_decoder(&mut self) {
        self.state = DecoderState::default();
        self.ref_frames = Default::default();
        self.output_queue.clear();
        self.initialized = false;
        self.last_header = None;
    }
}

/// Codec info implementation.
pub fn codec_info() -> transcode_core::format::VideoCodec {
    transcode_core::format::VideoCodec::Vp9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = Vp9Decoder::new_default();
        assert!(decoder.width().is_none());
        assert!(decoder.height().is_none());
        assert!(!decoder.initialized);
    }

    #[test]
    fn test_decoder_config() {
        let config = Vp9DecoderConfig {
            max_threads: 4,
            error_concealment: false,
            output_pool_size: 16,
            enable_loop_filter: false,
        };
        let decoder = Vp9Decoder::new(config.clone());
        assert_eq!(decoder.config.max_threads, 4);
        assert!(!decoder.config.error_concealment);
    }

    #[test]
    fn test_superframe_single() {
        let decoder = Vp9Decoder::new_default();
        // Data without superframe marker
        let data = [0x82, 0x49, 0x83, 0x42]; // Frame marker + sync code partial
        let result = decoder.parse_superframe(&data);
        assert!(result.is_ok());
        let frames = result.unwrap();
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_superframe_empty() {
        let decoder = Vp9Decoder::new_default();
        let result = decoder.parse_superframe(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut decoder = Vp9Decoder::new_default();
        decoder.state.width = 1920;
        decoder.state.height = 1080;
        decoder.initialized = true;

        decoder.reset_decoder();

        assert_eq!(decoder.state.width, 0);
        assert_eq!(decoder.state.height, 0);
        assert!(!decoder.initialized);
    }

    #[test]
    fn test_time_base() {
        let mut decoder = Vp9Decoder::new_default();
        let time_base = TimeBase::new(1, 30);
        decoder.set_time_base(time_base);
        assert_eq!(decoder.time_base, time_base);
    }

    #[test]
    fn test_pixel_format_selection() {
        let decoder = Vp9Decoder::new_default();

        let mut header = FrameHeader::default();
        header.subsampling = ChromaSubsampling::Cs420;
        header.bit_depth = 8;
        assert_eq!(decoder.get_pixel_format(&header), PixelFormat::Yuv420p);

        header.bit_depth = 10;
        assert_eq!(decoder.get_pixel_format(&header), PixelFormat::Yuv420p10le);

        header.subsampling = ChromaSubsampling::Cs444;
        header.bit_depth = 8;
        assert_eq!(decoder.get_pixel_format(&header), PixelFormat::Yuv444p);
    }

    #[test]
    fn test_ref_frame_slot_default() {
        let slot = RefFrameSlot::default();
        assert!(slot.buffer.is_none());
        assert_eq!(slot.width, 0);
        assert_eq!(slot.height, 0);
        assert_eq!(slot.bit_depth, 8);
    }

    #[test]
    fn test_ref_frame_slot_methods() {
        let slot = RefFrameSlot {
            buffer: None,
            width: 1920,
            height: 1080,
            bit_depth: 8,
            subsampling: ChromaSubsampling::Cs420,
        };

        // Test dimensions
        assert_eq!(slot.dimensions(), (1920, 1080));

        // Test is_valid_for_size (false because buffer is None)
        assert!(!slot.is_valid_for_size(1920, 1080));

        // Test matches_format
        assert!(slot.matches_format(8, ChromaSubsampling::Cs420));
        assert!(!slot.matches_format(10, ChromaSubsampling::Cs420));
        assert!(!slot.matches_format(8, ChromaSubsampling::Cs444));
    }

    #[test]
    fn test_decoder_state_default() {
        let state = DecoderState::default();
        assert_eq!(state.width, 0);
        assert_eq!(state.height, 0);
        assert_eq!(state.frame_num, 0);
        assert_eq!(state.profile, Profile::Profile0);
    }
}
