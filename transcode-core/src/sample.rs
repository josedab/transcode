//! Audio sample buffer abstractions.
//!
//! Provides types for representing decoded audio samples in various formats.

use crate::timestamp::{Duration, TimeBase, Timestamp};
use std::fmt;
use std::sync::Arc;

/// Sample format for audio data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit.
    U8,
    /// Signed 16-bit, native endian.
    S16,
    /// Signed 32-bit, native endian.
    S32,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// Unsigned 8-bit planar.
    U8p,
    /// Signed 16-bit planar.
    S16p,
    /// Signed 32-bit planar.
    S32p,
    /// 32-bit float planar.
    F32p,
    /// 64-bit float planar.
    F64p,
}

impl SampleFormat {
    /// Get the number of bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 | Self::U8p => 1,
            Self::S16 | Self::S16p => 2,
            Self::S32 | Self::S32p | Self::F32 | Self::F32p => 4,
            Self::F64 | Self::F64p => 8,
        }
    }

    /// Check if this is a planar format.
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::U8p | Self::S16p | Self::S32p | Self::F32p | Self::F64p
        )
    }

    /// Check if this is a floating-point format.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F32p | Self::F64p)
    }

    /// Get the packed equivalent of this format.
    pub fn to_packed(&self) -> Self {
        match self {
            Self::U8p => Self::U8,
            Self::S16p => Self::S16,
            Self::S32p => Self::S32,
            Self::F32p => Self::F32,
            Self::F64p => Self::F64,
            other => *other,
        }
    }

    /// Get the planar equivalent of this format.
    pub fn to_planar(&self) -> Self {
        match self {
            Self::U8 => Self::U8p,
            Self::S16 => Self::S16p,
            Self::S32 => Self::S32p,
            Self::F32 => Self::F32p,
            Self::F64 => Self::F64p,
            other => *other,
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8 => write!(f, "u8"),
            Self::S16 => write!(f, "s16"),
            Self::S32 => write!(f, "s32"),
            Self::F32 => write!(f, "flt"),
            Self::F64 => write!(f, "dbl"),
            Self::U8p => write!(f, "u8p"),
            Self::S16p => write!(f, "s16p"),
            Self::S32p => write!(f, "s32p"),
            Self::F32p => write!(f, "fltp"),
            Self::F64p => write!(f, "dblp"),
        }
    }
}

/// Channel layout for audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelLayout {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels: left, right).
    #[default]
    Stereo,
    /// 2.1 (3 channels: left, right, LFE).
    Surround21,
    /// Quad (4 channels: FL, FR, BL, BR).
    Quad,
    /// 5.0 (5 channels: FL, FR, FC, BL, BR).
    Surround50,
    /// 5.1 (6 channels: FL, FR, FC, LFE, BL, BR).
    Surround51,
    /// 7.1 (8 channels: FL, FR, FC, LFE, BL, BR, SL, SR).
    Surround71,
    /// Custom layout with specified channel count.
    Custom(u32),
}

impl ChannelLayout {
    /// Get the number of channels.
    pub fn channels(&self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround21 => 3,
            Self::Quad => 4,
            Self::Surround50 => 5,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(n) => *n,
        }
    }

    /// Create a layout from channel count.
    pub fn from_channels(channels: u32) -> Self {
        match channels {
            1 => Self::Mono,
            2 => Self::Stereo,
            6 => Self::Surround51,
            8 => Self::Surround71,
            n => Self::Custom(n),
        }
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mono => write!(f, "mono"),
            Self::Stereo => write!(f, "stereo"),
            Self::Surround21 => write!(f, "2.1"),
            Self::Quad => write!(f, "quad"),
            Self::Surround50 => write!(f, "5.0"),
            Self::Surround51 => write!(f, "5.1"),
            Self::Surround71 => write!(f, "7.1"),
            Self::Custom(n) => write!(f, "{}ch", n),
        }
    }
}

/// A decoded audio sample buffer.
#[derive(Clone)]
pub struct Sample {
    /// Sample data buffer.
    buffer: SampleBuffer,
    /// Presentation timestamp.
    pub pts: Timestamp,
    /// Duration of this sample buffer.
    pub duration: Duration,
}

impl Sample {
    /// Create a new sample buffer.
    pub fn new(
        num_samples: usize,
        format: SampleFormat,
        layout: ChannelLayout,
        sample_rate: u32,
    ) -> Self {
        Self {
            buffer: SampleBuffer::new(num_samples, format, layout, sample_rate),
            pts: Timestamp::none(),
            duration: Duration::zero(),
        }
    }

    /// Create from an existing buffer.
    pub fn from_buffer(buffer: SampleBuffer) -> Self {
        let duration = buffer.duration();
        Self {
            buffer,
            pts: Timestamp::none(),
            duration,
        }
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.buffer.num_samples
    }

    /// Get the sample format.
    pub fn format(&self) -> SampleFormat {
        self.buffer.format
    }

    /// Get the channel layout.
    pub fn channel_layout(&self) -> ChannelLayout {
        self.buffer.layout
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.buffer.sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u32 {
        self.buffer.layout.channels()
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &SampleBuffer {
        &self.buffer
    }

    /// Get a mutable reference to the buffer.
    pub fn buffer_mut(&mut self) -> &mut SampleBuffer {
        &mut self.buffer
    }

    /// Get a channel's data (for planar formats).
    pub fn channel(&self, index: u32) -> Option<&[u8]> {
        self.buffer.channel(index)
    }

    /// Get a mutable reference to a channel's data.
    pub fn channel_mut(&mut self, index: u32) -> Option<&mut [u8]> {
        self.buffer.channel_mut(index)
    }

    /// Get interleaved data (for packed formats).
    pub fn data(&self) -> &[u8] {
        self.buffer.data()
    }

    /// Get mutable interleaved data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.buffer.data_mut()
    }
}

impl fmt::Debug for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample")
            .field("num_samples", &self.num_samples())
            .field("format", &self.format())
            .field("layout", &self.channel_layout())
            .field("sample_rate", &self.sample_rate())
            .field("pts", &self.pts)
            .finish()
    }
}

/// Buffer for storing audio sample data.
#[derive(Clone)]
pub struct SampleBuffer {
    /// Number of samples per channel.
    pub num_samples: usize,
    /// Sample format.
    pub format: SampleFormat,
    /// Channel layout.
    pub layout: ChannelLayout,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Sample data (planar: one Vec per channel, packed: single Vec).
    data: Vec<Vec<u8>>,
}

impl SampleBuffer {
    /// Create a new sample buffer.
    pub fn new(
        num_samples: usize,
        format: SampleFormat,
        layout: ChannelLayout,
        sample_rate: u32,
    ) -> Self {
        let bytes_per_sample = format.bytes_per_sample();
        let channels = layout.channels() as usize;

        let data = if format.is_planar() {
            // One buffer per channel
            (0..channels)
                .map(|_| vec![0u8; num_samples * bytes_per_sample])
                .collect()
        } else {
            // Single interleaved buffer
            vec![vec![0u8; num_samples * channels * bytes_per_sample]]
        };

        Self {
            num_samples,
            format,
            layout,
            sample_rate,
            data,
        }
    }

    /// Get the duration of this buffer.
    pub fn duration(&self) -> Duration {
        Duration::new(
            self.num_samples as i64,
            TimeBase::new(1, self.sample_rate as i64),
        )
    }

    /// Get the total size in bytes.
    pub fn size(&self) -> usize {
        self.data.iter().map(|d| d.len()).sum()
    }

    /// Get a channel's data (for planar formats).
    pub fn channel(&self, index: u32) -> Option<&[u8]> {
        if self.format.is_planar() {
            self.data.get(index as usize).map(|v| v.as_slice())
        } else {
            None
        }
    }

    /// Get a mutable reference to a channel's data.
    pub fn channel_mut(&mut self, index: u32) -> Option<&mut [u8]> {
        if self.format.is_planar() {
            self.data.get_mut(index as usize).map(|v| v.as_mut_slice())
        } else {
            None
        }
    }

    /// Get interleaved data (for packed formats).
    pub fn data(&self) -> &[u8] {
        &self.data[0]
    }

    /// Get mutable interleaved data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data[0]
    }

    /// Get typed sample data for S16 format.
    ///
    /// Returns `None` if the format is not S16/S16p or if the data is not properly aligned.
    pub fn as_s16(&self) -> Option<&[i16]> {
        if self.format == SampleFormat::S16 || self.format == SampleFormat::S16p {
            let data = &self.data[0];
            let ptr = data.as_ptr();
            // Check alignment before casting
            if ptr.align_offset(std::mem::align_of::<i16>()) != 0 {
                return None;
            }
            // SAFETY: We've verified the format is S16/S16p and the pointer is properly aligned
            Some(unsafe {
                std::slice::from_raw_parts(
                    ptr as *const i16,
                    data.len() / 2,
                )
            })
        } else {
            None
        }
    }

    /// Get typed sample data for F32 format.
    ///
    /// Returns `None` if the format is not F32/F32p or if the data is not properly aligned.
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.format == SampleFormat::F32 || self.format == SampleFormat::F32p {
            let data = &self.data[0];
            let ptr = data.as_ptr();
            // Check alignment before casting
            if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
                return None;
            }
            // SAFETY: We've verified the format is F32/F32p and the pointer is properly aligned
            Some(unsafe {
                std::slice::from_raw_parts(
                    ptr as *const f32,
                    data.len() / 4,
                )
            })
        } else {
            None
        }
    }

    /// Fill all channels with silence.
    pub fn silence(&mut self) {
        let silence_value = match self.format {
            SampleFormat::U8 | SampleFormat::U8p => 128,
            _ => 0,
        };
        for channel in &mut self.data {
            channel.fill(silence_value);
        }
    }
}

impl fmt::Debug for SampleBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampleBuffer")
            .field("num_samples", &self.num_samples)
            .field("format", &self.format)
            .field("layout", &self.layout)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

/// A reference-counted sample for efficient sharing.
pub type SharedSample = Arc<Sample>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format() {
        assert_eq!(SampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::F32.bytes_per_sample(), 4);
        assert!(!SampleFormat::S16.is_planar());
        assert!(SampleFormat::S16p.is_planar());
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(ChannelLayout::Stereo.channels(), 2);
        assert_eq!(ChannelLayout::Surround51.channels(), 6);
        assert_eq!(ChannelLayout::from_channels(2), ChannelLayout::Stereo);
    }

    #[test]
    fn test_sample_buffer_creation() {
        let buffer = SampleBuffer::new(1024, SampleFormat::S16, ChannelLayout::Stereo, 48000);
        assert_eq!(buffer.num_samples, 1024);
        assert_eq!(buffer.size(), 1024 * 2 * 2); // 1024 samples * 2 bytes * 2 channels
    }

    #[test]
    fn test_planar_buffer() {
        let buffer = SampleBuffer::new(1024, SampleFormat::F32p, ChannelLayout::Stereo, 48000);
        assert!(buffer.channel(0).is_some());
        assert!(buffer.channel(1).is_some());
        assert!(buffer.channel(2).is_none());
    }
}
