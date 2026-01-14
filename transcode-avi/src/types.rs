//! AVI type definitions

/// AVI main header (avih chunk)
#[derive(Debug, Clone)]
pub struct AviHeader {
    /// Microseconds per frame
    pub microseconds_per_frame: u32,
    /// Maximum bytes per second
    pub max_bytes_per_sec: u32,
    /// Padding granularity
    pub padding_granularity: u32,
    /// AVI flags
    pub flags: AviFlags,
    /// Total number of frames
    pub total_frames: u32,
    /// Initial frames (for interleaved files)
    pub initial_frames: u32,
    /// Number of streams
    pub streams: u32,
    /// Suggested buffer size
    pub suggested_buffer_size: u32,
    /// Video width
    pub width: u32,
    /// Video height
    pub height: u32,
}

impl AviHeader {
    /// Calculate frame rate in fps
    pub fn frame_rate(&self) -> f64 {
        if self.microseconds_per_frame > 0 {
            1_000_000.0 / self.microseconds_per_frame as f64
        } else {
            0.0
        }
    }

    /// Calculate duration in seconds
    pub fn duration(&self) -> f64 {
        (self.total_frames as f64 * self.microseconds_per_frame as f64) / 1_000_000.0
    }
}

impl Default for AviHeader {
    fn default() -> Self {
        AviHeader {
            microseconds_per_frame: 33333, // ~30 fps
            max_bytes_per_sec: 10_000_000,
            padding_granularity: 0,
            flags: AviFlags::default(),
            total_frames: 0,
            initial_frames: 0,
            streams: 0,
            suggested_buffer_size: 1_000_000,
            width: 0,
            height: 0,
        }
    }
}

/// AVI header flags
#[derive(Debug, Clone, Copy, Default)]
pub struct AviFlags {
    /// File has an index
    pub has_index: bool,
    /// File must use index
    pub must_use_index: bool,
    /// File is interleaved
    pub is_interleaved: bool,
    /// Trust chunk type for seeking
    pub trust_chunk_type: bool,
    /// File was captured
    pub was_captured: bool,
    /// File is copyrighted
    pub is_copyrighted: bool,
}

impl AviFlags {
    pub fn from_u32(value: u32) -> Self {
        AviFlags {
            has_index: (value & 0x10) != 0,
            must_use_index: (value & 0x20) != 0,
            is_interleaved: (value & 0x100) != 0,
            trust_chunk_type: (value & 0x800) != 0,
            was_captured: (value & 0x10000) != 0,
            is_copyrighted: (value & 0x20000) != 0,
        }
    }

    pub fn to_u32(self) -> u32 {
        let mut value = 0u32;
        if self.has_index {
            value |= 0x10;
        }
        if self.must_use_index {
            value |= 0x20;
        }
        if self.is_interleaved {
            value |= 0x100;
        }
        if self.trust_chunk_type {
            value |= 0x800;
        }
        if self.was_captured {
            value |= 0x10000;
        }
        if self.is_copyrighted {
            value |= 0x20000;
        }
        value
    }
}

/// Stream header (strh chunk)
#[derive(Debug, Clone)]
pub struct StreamHeader {
    /// Stream type (vids, auds, txts, mids)
    pub stream_type: StreamType,
    /// FourCC handler/codec
    pub handler: [u8; 4],
    /// Stream flags
    pub flags: u32,
    /// Priority
    pub priority: u16,
    /// Language
    pub language: u16,
    /// Initial frames
    pub initial_frames: u32,
    /// Time scale
    pub scale: u32,
    /// Rate (samples per second = rate/scale)
    pub rate: u32,
    /// Start time
    pub start: u32,
    /// Length (number of frames or audio samples)
    pub length: u32,
    /// Suggested buffer size
    pub suggested_buffer_size: u32,
    /// Quality (0-10000)
    pub quality: u32,
    /// Sample size (0 for variable)
    pub sample_size: u32,
    /// Frame rectangle
    pub frame: Rect,
}

impl Default for StreamHeader {
    fn default() -> Self {
        StreamHeader {
            stream_type: StreamType::Video,
            handler: [0; 4],
            flags: 0,
            priority: 0,
            language: 0,
            initial_frames: 0,
            scale: 1,
            rate: 30,
            start: 0,
            length: 0,
            suggested_buffer_size: 1_000_000,
            quality: 0,
            sample_size: 0,
            frame: Rect::default(),
        }
    }
}

/// Stream type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Video,
    Audio,
    Text,
    Midi,
    Unknown([u8; 4]),
}

impl StreamType {
    pub fn from_fourcc(fourcc: &[u8; 4]) -> Self {
        match fourcc {
            b"vids" => StreamType::Video,
            b"auds" => StreamType::Audio,
            b"txts" => StreamType::Text,
            b"mids" => StreamType::Midi,
            _ => StreamType::Unknown(*fourcc),
        }
    }

    pub fn to_fourcc(self) -> [u8; 4] {
        match self {
            StreamType::Video => *b"vids",
            StreamType::Audio => *b"auds",
            StreamType::Text => *b"txts",
            StreamType::Midi => *b"mids",
            StreamType::Unknown(fourcc) => fourcc,
        }
    }
}

/// Rectangle structure
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub left: i16,
    pub top: i16,
    pub right: i16,
    pub bottom: i16,
}

/// Video format (BITMAPINFOHEADER)
#[derive(Debug, Clone)]
pub struct VideoFormat {
    /// Structure size
    pub size: u32,
    /// Width in pixels
    pub width: i32,
    /// Height in pixels (negative for top-down)
    pub height: i32,
    /// Number of planes (always 1)
    pub planes: u16,
    /// Bits per pixel
    pub bit_count: u16,
    /// Compression FourCC
    pub compression: [u8; 4],
    /// Image size in bytes
    pub image_size: u32,
    /// Horizontal resolution
    pub x_pels_per_meter: i32,
    /// Vertical resolution
    pub y_pels_per_meter: i32,
    /// Colors used
    pub colors_used: u32,
    /// Important colors
    pub colors_important: u32,
}

impl Default for VideoFormat {
    fn default() -> Self {
        VideoFormat {
            size: 40,
            width: 0,
            height: 0,
            planes: 1,
            bit_count: 24,
            compression: *b"DIB ",
            image_size: 0,
            x_pels_per_meter: 0,
            y_pels_per_meter: 0,
            colors_used: 0,
            colors_important: 0,
        }
    }
}

impl VideoFormat {
    /// Get absolute height (handles negative for top-down)
    pub fn abs_height(&self) -> u32 {
        self.height.unsigned_abs()
    }

    /// Check if image is top-down
    pub fn is_top_down(&self) -> bool {
        self.height < 0
    }

    /// Get codec FourCC as string
    pub fn codec_string(&self) -> String {
        String::from_utf8_lossy(&self.compression).trim().to_string()
    }
}

/// Audio format (WAVEFORMATEX)
#[derive(Debug, Clone)]
pub struct AudioFormat {
    /// Format tag
    pub format_tag: u16,
    /// Number of channels
    pub channels: u16,
    /// Samples per second
    pub samples_per_sec: u32,
    /// Average bytes per second
    pub avg_bytes_per_sec: u32,
    /// Block alignment
    pub block_align: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Extra data size
    pub extra_size: u16,
    /// Extra codec-specific data
    pub extra_data: Vec<u8>,
}

impl Default for AudioFormat {
    fn default() -> Self {
        AudioFormat {
            format_tag: 1, // PCM
            channels: 2,
            samples_per_sec: 44100,
            avg_bytes_per_sec: 176400,
            block_align: 4,
            bits_per_sample: 16,
            extra_size: 0,
            extra_data: Vec::new(),
        }
    }
}

impl AudioFormat {
    /// Get format name
    pub fn format_name(&self) -> &'static str {
        match self.format_tag {
            0x0001 => "PCM",
            0x0003 => "IEEE Float",
            0x0006 => "A-Law",
            0x0007 => "μ-Law",
            0x0055 => "MP3",
            0x00FF => "AAC",
            0x0161 => "WMA v2",
            0x0162 => "WMA Pro",
            0x2000 => "AC-3",
            0xFFFE => "Extensible",
            _ => "Unknown",
        }
    }
}

/// Common AVI codec FourCCs
#[allow(dead_code)]
pub mod codec {
    /// Uncompressed RGB
    pub const DIB: [u8; 4] = *b"DIB ";
    /// Uncompressed packed YUV
    pub const YUY2: [u8; 4] = *b"YUY2";
    pub const UYVY: [u8; 4] = *b"UYVY";
    pub const YV12: [u8; 4] = *b"YV12";
    /// MPEG-4 variants
    pub const DIVX: [u8; 4] = *b"DIVX";
    pub const XVID: [u8; 4] = *b"XVID";
    pub const DX50: [u8; 4] = *b"DX50";
    pub const FMP4: [u8; 4] = *b"FMP4";
    /// H.264
    pub const H264: [u8; 4] = *b"H264";
    pub const AVC1: [u8; 4] = *b"avc1";
    pub const X264: [u8; 4] = *b"X264";
    /// Motion JPEG
    pub const MJPG: [u8; 4] = *b"MJPG";
    /// Huffyuv
    pub const HFYU: [u8; 4] = *b"HFYU";
    /// FFV1
    pub const FFV1: [u8; 4] = *b"FFV1";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avi_header_frame_rate() {
        let header = AviHeader {
            microseconds_per_frame: 33333,
            ..Default::default()
        };
        let fps = header.frame_rate();
        assert!((fps - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_avi_flags() {
        let flags = AviFlags::from_u32(0x110);
        assert!(flags.has_index);
        assert!(flags.is_interleaved);
        assert!(!flags.was_captured);

        assert_eq!(flags.to_u32(), 0x110);
    }

    #[test]
    fn test_stream_type() {
        assert_eq!(StreamType::from_fourcc(b"vids"), StreamType::Video);
        assert_eq!(StreamType::from_fourcc(b"auds"), StreamType::Audio);
        assert_eq!(StreamType::Video.to_fourcc(), *b"vids");
    }

    #[test]
    fn test_video_format() {
        let fmt = VideoFormat {
            height: -480,
            ..Default::default()
        };
        assert!(fmt.is_top_down());
        assert_eq!(fmt.abs_height(), 480);
    }

    #[test]
    fn test_audio_format_name() {
        let pcm = AudioFormat::default();
        assert_eq!(pcm.format_name(), "PCM");

        let mp3 = AudioFormat {
            format_tag: 0x0055,
            ..Default::default()
        };
        assert_eq!(mp3.format_name(), "MP3");
    }
}
