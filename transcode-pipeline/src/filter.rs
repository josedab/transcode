//! Filter abstractions for video and audio processing.

use crate::error::PipelineError;
use crate::Result;
use transcode_core::frame::{Frame, FrameFlags};
use transcode_core::sample::Sample;
use transcode_core::timestamp::TimeBase;

/// Base filter trait.
pub trait Filter: Send {
    /// Get filter name.
    fn name(&self) -> &str;

    /// Check if filter is enabled.
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Video filter trait.
pub trait VideoFilter: Filter {
    /// Process a video frame.
    fn process(&mut self, frame: Frame) -> Result<Frame>;

    /// Flush any buffered frames.
    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Audio filter trait.
pub trait AudioFilter: Filter {
    /// Process audio samples.
    fn process(&mut self, sample: Sample) -> Result<Sample>;

    /// Flush any buffered samples.
    fn flush(&mut self) -> Result<Vec<Sample>> {
        Ok(Vec::new())
    }
}

/// Chain of filters.
pub struct FilterChain<F: ?Sized> {
    filters: Vec<Box<F>>,
}

impl<F: ?Sized> Default for FilterChain<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: ?Sized> FilterChain<F> {
    /// Create a new empty filter chain.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the chain.
    pub fn add(&mut self, filter: Box<F>) {
        self.filters.push(filter);
    }

    /// Get number of filters.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Check if chain is empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl FilterChain<dyn VideoFilter> {
    /// Process a video frame through all filters.
    pub fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        for filter in &mut self.filters {
            if filter.is_enabled() {
                frame = filter.process(frame)?;
            }
        }
        Ok(frame)
    }

    /// Flush all filters.
    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        let mut frames = Vec::new();
        for filter in &mut self.filters {
            frames.extend(filter.flush()?);
        }
        Ok(frames)
    }
}

impl FilterChain<dyn AudioFilter> {
    /// Process audio samples through all filters.
    pub fn process(&mut self, mut sample: Sample) -> Result<Sample> {
        for filter in &mut self.filters {
            if filter.is_enabled() {
                sample = filter.process(sample)?;
            }
        }
        Ok(sample)
    }

    /// Flush all filters.
    pub fn flush(&mut self) -> Result<Vec<Sample>> {
        let mut samples = Vec::new();
        for filter in &mut self.filters {
            samples.extend(filter.flush()?);
        }
        Ok(samples)
    }
}

/// Scale filter for video resizing.
pub struct ScaleFilter {
    name: String,
    target_width: u32,
    target_height: u32,
    enabled: bool,
}

impl ScaleFilter {
    /// Create a new scale filter.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            name: format!("scale_{}x{}", width, height),
            target_width: width,
            target_height: height,
            enabled: true,
        }
    }

    /// Enable or disable the filter.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Filter for ScaleFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl VideoFilter for ScaleFilter {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        let src_width = frame.width() as usize;
        let src_height = frame.height() as usize;
        let dst_width = self.target_width as usize;
        let dst_height = self.target_height as usize;

        if src_width == dst_width && src_height == dst_height {
            return Ok(frame);
        }

        let mut scaled = Frame::new(
            self.target_width,
            self.target_height,
            frame.format(),
            TimeBase::new(1, 1000),
        );

        // Simple nearest-neighbor scaling for Y plane
        if let (Some(src_y), Some(dst_y)) = (frame.plane(0), scaled.plane_mut(0)) {
            for y in 0..dst_height {
                let src_y_pos = y * src_height / dst_height;
                for x in 0..dst_width {
                    let src_x_pos = x * src_width / dst_width;
                    let dst_idx = y * dst_width + x;
                    let src_idx = src_y_pos * src_width + src_x_pos;
                    if dst_idx < dst_y.len() && src_idx < src_y.len() {
                        dst_y[dst_idx] = src_y[src_idx];
                    }
                }
            }
        }

        // Scale chroma planes (if present)
        let num_planes = frame.format().num_planes();
        if num_planes > 1 {
            let chroma_width_src = src_width / 2;
            let chroma_height_src = src_height / 2;
            let chroma_width_dst = dst_width / 2;
            let chroma_height_dst = dst_height / 2;

            for plane_idx in 1..num_planes.min(3) {
                if let (Some(src_plane), Some(dst_plane)) =
                    (frame.plane(plane_idx), scaled.plane_mut(plane_idx))
                {
                    for y in 0..chroma_height_dst {
                        let src_y_pos = y * chroma_height_src / chroma_height_dst;
                        for x in 0..chroma_width_dst {
                            let src_x_pos = x * chroma_width_src / chroma_width_dst;
                            let dst_idx = y * chroma_width_dst + x;
                            let src_idx = src_y_pos * chroma_width_src + src_x_pos;
                            if dst_idx < dst_plane.len() && src_idx < src_plane.len() {
                                dst_plane[dst_idx] = src_plane[src_idx];
                            }
                        }
                    }
                }
            }
        }

        // Copy timing information
        scaled.pts = frame.pts;
        scaled.dts = frame.dts;
        if frame.is_keyframe() {
            scaled.flags |= FrameFlags::KEYFRAME;
        }

        Ok(scaled)
    }
}

/// Volume filter for audio level adjustment.
pub struct VolumeFilter {
    name: String,
    gain: f32,
    enabled: bool,
}

impl VolumeFilter {
    /// Create a new volume filter with gain in dB.
    pub fn new(gain_db: f32) -> Self {
        Self {
            name: format!("volume_{:.1}dB", gain_db),
            gain: 10.0_f32.powf(gain_db / 20.0),
            enabled: true,
        }
    }

    /// Set gain in dB.
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain = 10.0_f32.powf(gain_db / 20.0);
    }

    /// Enable or disable the filter.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Filter for VolumeFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl AudioFilter for VolumeFilter {
    fn process(&mut self, mut sample: Sample) -> Result<Sample> {
        // Apply gain to sample data
        let data = sample.buffer_mut().data_mut();

        // Process as S16 samples if format allows
        if data.len() >= 2 {
            // Validate alignment requirements for safe i16 access
            if data.len() % 2 != 0 {
                return Err(PipelineError::InvalidConfig(
                    "Sample data length must be even for S16 processing".to_string(),
                ));
            }
            let ptr = data.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<i16>()) != 0 {
                return Err(PipelineError::InvalidConfig(
                    "Sample buffer not aligned for i16 access".to_string(),
                ));
            }

            // SAFETY: We verified length is even and pointer is properly aligned for i16
            let samples: &mut [i16] = unsafe {
                std::slice::from_raw_parts_mut(ptr as *mut i16, data.len() / 2)
            };

            for s in samples.iter_mut() {
                *s = ((*s as f32) * self.gain).clamp(-32768.0, 32767.0) as i16;
            }
        }

        Ok(sample)
    }
}

/// Null video filter (pass-through).
pub struct NullVideoFilter {
    name: String,
}

impl Default for NullVideoFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl NullVideoFilter {
    /// Create a new null filter.
    pub fn new() -> Self {
        Self {
            name: "null".to_string(),
        }
    }
}

impl Filter for NullVideoFilter {
    fn name(&self) -> &str {
        &self.name
    }
}

impl VideoFilter for NullVideoFilter {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        Ok(frame)
    }
}

/// Null audio filter (pass-through).
pub struct NullAudioFilter {
    name: String,
}

impl Default for NullAudioFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl NullAudioFilter {
    /// Create a new null filter.
    pub fn new() -> Self {
        Self {
            name: "anull".to_string(),
        }
    }
}

impl Filter for NullAudioFilter {
    fn name(&self) -> &str {
        &self.name
    }
}

impl AudioFilter for NullAudioFilter {
    fn process(&mut self, sample: Sample) -> Result<Sample> {
        Ok(sample)
    }
}
