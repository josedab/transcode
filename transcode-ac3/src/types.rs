//! AC-3 and E-AC-3 data types.

use std::fmt;

/// AC-3 sync frame information.
#[derive(Debug, Clone, PartialEq)]
pub struct Ac3SyncFrame {
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bitrate in bits per second.
    pub bitrate: u32,
    /// Bitstream identification (bsid).
    pub bsid: u8,
    /// Audio coding mode (channel configuration).
    pub acmod: AudioCodingMode,
    /// Low frequency effects channel present.
    pub lfe_on: bool,
    /// Number of audio channels (including LFE).
    pub channels: u8,
    /// Dialogue normalization value (-1 to -31 dB).
    pub dialnorm: i8,
    /// Compression gain word exists.
    pub compre: bool,
    /// Compression gain word value.
    pub compr: Option<u8>,
    /// Language code exists.
    pub langcode: bool,
    /// Language code value.
    pub langcod: Option<u8>,
    /// Audio production information exists.
    pub audprodie: bool,
    /// Mixing level.
    pub mixlevel: Option<u8>,
    /// Room type.
    pub roomtyp: Option<RoomType>,
    /// Frame sample rate code.
    pub fscod: u8,
    /// Frame size code.
    pub frmsizecod: u8,
    /// Bit stream mode.
    pub bsmod: BitstreamMode,
    /// CRC word 1.
    pub crc1: u16,
    /// CRC word 2.
    pub crc2: u16,
}

impl Ac3SyncFrame {
    /// Create a new empty sync frame.
    pub fn new() -> Self {
        Self {
            frame_size: 0,
            sample_rate: 48000,
            bitrate: 384000,
            bsid: 8,
            acmod: AudioCodingMode::Stereo,
            lfe_on: false,
            channels: 2,
            dialnorm: -31,
            compre: false,
            compr: None,
            langcode: false,
            langcod: None,
            audprodie: false,
            mixlevel: None,
            roomtyp: None,
            fscod: 0,
            frmsizecod: 0,
            bsmod: BitstreamMode::CompleteMain,
            crc1: 0,
            crc2: 0,
        }
    }

    /// Get duration of this frame in samples.
    pub fn samples(&self) -> usize {
        crate::AC3_SAMPLES_PER_FRAME
    }

    /// Get duration of this frame in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples() as f64 / self.sample_rate as f64
    }

    /// Get channel layout description.
    pub fn channel_layout(&self) -> String {
        let mut layout = self.acmod.to_string();
        if self.lfe_on {
            layout.push_str(".1");
        }
        layout
    }
}

impl Default for Ac3SyncFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// E-AC-3 sync frame information.
#[derive(Debug, Clone, PartialEq)]
pub struct Eac3SyncFrame {
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio blocks (1, 2, 3, or 6).
    pub num_blocks: u8,
    /// Bitstream identification (bsid).
    pub bsid: u8,
    /// Audio coding mode (channel configuration).
    pub acmod: AudioCodingMode,
    /// Low frequency effects channel present.
    pub lfe_on: bool,
    /// Number of audio channels (including LFE).
    pub channels: u8,
    /// Stream type.
    pub stream_type: Eac3StreamType,
    /// Substream ID.
    pub substreamid: u8,
    /// Dialogue normalization value.
    pub dialnorm: i8,
    /// Frame sample rate code.
    pub fscod: u8,
    /// Frame sample rate code 2 (for 24kHz etc).
    pub fscod2: Option<u8>,
    /// Bit stream mode.
    pub bsmod: BitstreamMode,
    /// Channel coupling used.
    pub chanmape: bool,
    /// Number of independent substreams.
    pub num_ind_sub: u8,
    /// Number of dependent substreams.
    pub num_dep_sub: Option<u8>,
    /// Enhanced channel configuration.
    pub chan_loc: Option<u16>,
    /// Calculated bitrate (approximate).
    pub bitrate: u32,
}

impl Eac3SyncFrame {
    /// Create a new empty E-AC-3 sync frame.
    pub fn new() -> Self {
        Self {
            frame_size: 0,
            sample_rate: 48000,
            num_blocks: 6,
            bsid: 16,
            acmod: AudioCodingMode::Stereo,
            lfe_on: false,
            channels: 2,
            stream_type: Eac3StreamType::Independent,
            substreamid: 0,
            dialnorm: -31,
            fscod: 0,
            fscod2: None,
            bsmod: BitstreamMode::CompleteMain,
            chanmape: false,
            num_ind_sub: 1,
            num_dep_sub: None,
            chan_loc: None,
            bitrate: 0,
        }
    }

    /// Get number of samples in this frame.
    pub fn samples(&self) -> usize {
        self.num_blocks as usize * crate::AC3_SAMPLES_PER_BLOCK
    }

    /// Get duration of this frame in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples() as f64 / self.sample_rate as f64
    }

    /// Check if this is a Dolby Atmos stream.
    pub fn is_atmos(&self) -> bool {
        // Dolby Atmos uses object-based audio signaled via JOC
        self.chan_loc.is_some() || self.num_dep_sub.unwrap_or(0) > 0
    }
}

impl Default for Eac3SyncFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio coding mode (acmod).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AudioCodingMode {
    /// 1+1: Dual mono (Ch1, Ch2).
    DualMono = 0,
    /// 1/0: Mono (C).
    Mono = 1,
    /// 2/0: Stereo (L, R).
    Stereo = 2,
    /// 3/0: L, C, R.
    ThreeChannelFront = 3,
    /// 2/1: L, R, S.
    TwoChannelPlusSurround = 4,
    /// 3/1: L, C, R, S.
    ThreeChannelPlusSurround = 5,
    /// 2/2: L, R, SL, SR.
    TwoChannelPlusTwoSurround = 6,
    /// 3/2: L, C, R, SL, SR.
    FiveChannel = 7,
}

impl AudioCodingMode {
    /// Get number of full-bandwidth channels for this mode.
    pub fn num_channels(self) -> u8 {
        match self {
            AudioCodingMode::DualMono => 2,
            AudioCodingMode::Mono => 1,
            AudioCodingMode::Stereo => 2,
            AudioCodingMode::ThreeChannelFront => 3,
            AudioCodingMode::TwoChannelPlusSurround => 3,
            AudioCodingMode::ThreeChannelPlusSurround => 4,
            AudioCodingMode::TwoChannelPlusTwoSurround => 4,
            AudioCodingMode::FiveChannel => 5,
        }
    }

    /// Check if center channel is present.
    pub fn has_center(self) -> bool {
        matches!(
            self,
            AudioCodingMode::ThreeChannelFront
                | AudioCodingMode::ThreeChannelPlusSurround
                | AudioCodingMode::FiveChannel
        )
    }

    /// Check if surround channels are present.
    pub fn has_surround(self) -> bool {
        matches!(
            self,
            AudioCodingMode::TwoChannelPlusSurround
                | AudioCodingMode::ThreeChannelPlusSurround
                | AudioCodingMode::TwoChannelPlusTwoSurround
                | AudioCodingMode::FiveChannel
        )
    }

    /// Parse from acmod value.
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(AudioCodingMode::DualMono),
            1 => Some(AudioCodingMode::Mono),
            2 => Some(AudioCodingMode::Stereo),
            3 => Some(AudioCodingMode::ThreeChannelFront),
            4 => Some(AudioCodingMode::TwoChannelPlusSurround),
            5 => Some(AudioCodingMode::ThreeChannelPlusSurround),
            6 => Some(AudioCodingMode::TwoChannelPlusTwoSurround),
            7 => Some(AudioCodingMode::FiveChannel),
            _ => None,
        }
    }
}

impl fmt::Display for AudioCodingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioCodingMode::DualMono => write!(f, "1+1"),
            AudioCodingMode::Mono => write!(f, "1.0"),
            AudioCodingMode::Stereo => write!(f, "2.0"),
            AudioCodingMode::ThreeChannelFront => write!(f, "3.0"),
            AudioCodingMode::TwoChannelPlusSurround => write!(f, "2.1"),
            AudioCodingMode::ThreeChannelPlusSurround => write!(f, "3.1"),
            AudioCodingMode::TwoChannelPlusTwoSurround => write!(f, "4.0"),
            AudioCodingMode::FiveChannel => write!(f, "5.0"),
        }
    }
}

/// Bitstream mode (bsmod).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BitstreamMode {
    /// Complete main (CM).
    CompleteMain = 0,
    /// Music and effects (ME).
    MusicEffects = 1,
    /// Visually impaired (VI).
    VisuallyImpaired = 2,
    /// Hearing impaired (HI).
    HearingImpaired = 3,
    /// Dialogue (D).
    Dialogue = 4,
    /// Commentary (C).
    Commentary = 5,
    /// Emergency (E).
    Emergency = 6,
    /// Voice over (VO) / Karaoke.
    VoiceOver = 7,
}

impl BitstreamMode {
    /// Parse from bsmod value.
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(BitstreamMode::CompleteMain),
            1 => Some(BitstreamMode::MusicEffects),
            2 => Some(BitstreamMode::VisuallyImpaired),
            3 => Some(BitstreamMode::HearingImpaired),
            4 => Some(BitstreamMode::Dialogue),
            5 => Some(BitstreamMode::Commentary),
            6 => Some(BitstreamMode::Emergency),
            7 => Some(BitstreamMode::VoiceOver),
            _ => None,
        }
    }
}

impl fmt::Display for BitstreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitstreamMode::CompleteMain => write!(f, "Complete Main"),
            BitstreamMode::MusicEffects => write!(f, "Music & Effects"),
            BitstreamMode::VisuallyImpaired => write!(f, "Visually Impaired"),
            BitstreamMode::HearingImpaired => write!(f, "Hearing Impaired"),
            BitstreamMode::Dialogue => write!(f, "Dialogue"),
            BitstreamMode::Commentary => write!(f, "Commentary"),
            BitstreamMode::Emergency => write!(f, "Emergency"),
            BitstreamMode::VoiceOver => write!(f, "Voice Over"),
        }
    }
}

/// E-AC-3 stream type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Eac3StreamType {
    /// Independent stream.
    Independent = 0,
    /// Dependent stream (extension of another stream).
    Dependent = 1,
    /// Independent stream with associated data stream.
    IndependentAssociated = 2,
    /// Reserved.
    Reserved = 3,
}

impl Eac3StreamType {
    /// Parse from strmtyp value.
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(Eac3StreamType::Independent),
            1 => Some(Eac3StreamType::Dependent),
            2 => Some(Eac3StreamType::IndependentAssociated),
            3 => Some(Eac3StreamType::Reserved),
            _ => None,
        }
    }
}

impl fmt::Display for Eac3StreamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Eac3StreamType::Independent => write!(f, "Independent"),
            Eac3StreamType::Dependent => write!(f, "Dependent"),
            Eac3StreamType::IndependentAssociated => write!(f, "Independent+Associated"),
            Eac3StreamType::Reserved => write!(f, "Reserved"),
        }
    }
}

/// Room type for audio production information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoomType {
    /// Not indicated.
    NotIndicated = 0,
    /// Large room.
    LargeRoom = 1,
    /// Small room.
    SmallRoom = 2,
    /// Reserved.
    Reserved = 3,
}

impl RoomType {
    /// Parse from roomtyp value.
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(RoomType::NotIndicated),
            1 => Some(RoomType::LargeRoom),
            2 => Some(RoomType::SmallRoom),
            3 => Some(RoomType::Reserved),
            _ => None,
        }
    }
}

impl fmt::Display for RoomType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoomType::NotIndicated => write!(f, "Not Indicated"),
            RoomType::LargeRoom => write!(f, "Large Room"),
            RoomType::SmallRoom => write!(f, "Small Room"),
            RoomType::Reserved => write!(f, "Reserved"),
        }
    }
}

/// AC-3/E-AC-3 channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ChannelLayout {
    /// Front left channel.
    pub front_left: bool,
    /// Front center channel.
    pub front_center: bool,
    /// Front right channel.
    pub front_right: bool,
    /// Surround left channel.
    pub surround_left: bool,
    /// Surround right channel.
    pub surround_right: bool,
    /// Rear surround left channel.
    pub rear_surround_left: bool,
    /// Rear surround right channel.
    pub rear_surround_right: bool,
    /// Low frequency effects channel.
    pub lfe: bool,
    /// Top front center (Atmos).
    pub top_front_center: bool,
    /// Top surround left (Atmos).
    pub top_surround_left: bool,
    /// Top surround right (Atmos).
    pub top_surround_right: bool,
}

impl ChannelLayout {
    /// Create layout from audio coding mode and LFE flag.
    pub fn from_acmod(acmod: AudioCodingMode, lfe: bool) -> Self {
        let mut layout = Self {
            lfe,
            ..Default::default()
        };

        match acmod {
            AudioCodingMode::DualMono | AudioCodingMode::Stereo => {
                layout.front_left = true;
                layout.front_right = true;
            }
            AudioCodingMode::Mono => {
                layout.front_center = true;
            }
            AudioCodingMode::ThreeChannelFront => {
                layout.front_left = true;
                layout.front_center = true;
                layout.front_right = true;
            }
            AudioCodingMode::TwoChannelPlusSurround => {
                layout.front_left = true;
                layout.front_right = true;
                layout.surround_left = true;
            }
            AudioCodingMode::ThreeChannelPlusSurround => {
                layout.front_left = true;
                layout.front_center = true;
                layout.front_right = true;
                layout.surround_left = true;
            }
            AudioCodingMode::TwoChannelPlusTwoSurround => {
                layout.front_left = true;
                layout.front_right = true;
                layout.surround_left = true;
                layout.surround_right = true;
            }
            AudioCodingMode::FiveChannel => {
                layout.front_left = true;
                layout.front_center = true;
                layout.front_right = true;
                layout.surround_left = true;
                layout.surround_right = true;
            }
        }

        layout
    }

    /// Get total number of channels.
    pub fn num_channels(&self) -> u8 {
        let mut count = 0;
        if self.front_left {
            count += 1;
        }
        if self.front_center {
            count += 1;
        }
        if self.front_right {
            count += 1;
        }
        if self.surround_left {
            count += 1;
        }
        if self.surround_right {
            count += 1;
        }
        if self.rear_surround_left {
            count += 1;
        }
        if self.rear_surround_right {
            count += 1;
        }
        if self.lfe {
            count += 1;
        }
        if self.top_front_center {
            count += 1;
        }
        if self.top_surround_left {
            count += 1;
        }
        if self.top_surround_right {
            count += 1;
        }
        count
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut channels = Vec::new();
        if self.front_left {
            channels.push("FL");
        }
        if self.front_center {
            channels.push("FC");
        }
        if self.front_right {
            channels.push("FR");
        }
        if self.surround_left {
            channels.push("SL");
        }
        if self.surround_right {
            channels.push("SR");
        }
        if self.rear_surround_left {
            channels.push("RL");
        }
        if self.rear_surround_right {
            channels.push("RR");
        }
        if self.top_front_center {
            channels.push("TFC");
        }
        if self.top_surround_left {
            channels.push("TSL");
        }
        if self.top_surround_right {
            channels.push("TSR");
        }
        if self.lfe {
            channels.push("LFE");
        }
        write!(f, "{}", channels.join("+"))
    }
}

/// Decoded audio samples from AC-3/E-AC-3.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Channel layout.
    pub layout: ChannelLayout,
    /// Audio samples (interleaved, normalized to [-1.0, 1.0]).
    pub samples: Vec<f32>,
    /// Number of samples per channel.
    pub samples_per_channel: usize,
}

impl DecodedAudio {
    /// Create new decoded audio buffer.
    pub fn new(sample_rate: u32, channels: u8, layout: ChannelLayout) -> Self {
        Self {
            sample_rate,
            channels,
            layout,
            samples: Vec::new(),
            samples_per_channel: 0,
        }
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }

    /// Get samples for a specific channel.
    pub fn channel_samples(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels as usize {
            return Vec::new();
        }

        self.samples
            .iter()
            .skip(channel)
            .step_by(self.channels as usize)
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_coding_mode_channels() {
        assert_eq!(AudioCodingMode::Mono.num_channels(), 1);
        assert_eq!(AudioCodingMode::Stereo.num_channels(), 2);
        assert_eq!(AudioCodingMode::FiveChannel.num_channels(), 5);
    }

    #[test]
    fn test_audio_coding_mode_has_center() {
        assert!(!AudioCodingMode::Stereo.has_center());
        assert!(AudioCodingMode::FiveChannel.has_center());
    }

    #[test]
    fn test_channel_layout() {
        let layout = ChannelLayout::from_acmod(AudioCodingMode::FiveChannel, true);
        assert_eq!(layout.num_channels(), 6);
        assert!(layout.front_left);
        assert!(layout.front_center);
        assert!(layout.lfe);
    }

    #[test]
    fn test_ac3_sync_frame() {
        let frame = Ac3SyncFrame::new();
        assert_eq!(frame.samples(), 1536);
        assert!((frame.duration_secs() - 0.032).abs() < 0.001);
    }

    #[test]
    fn test_bitstream_mode_parse() {
        assert_eq!(
            BitstreamMode::from_value(0),
            Some(BitstreamMode::CompleteMain)
        );
        assert_eq!(
            BitstreamMode::from_value(7),
            Some(BitstreamMode::VoiceOver)
        );
        assert_eq!(BitstreamMode::from_value(8), None);
    }

    #[test]
    fn test_eac3_stream_type() {
        let frame = Eac3SyncFrame {
            stream_type: Eac3StreamType::Independent,
            ..Default::default()
        };
        assert_eq!(frame.stream_type, Eac3StreamType::Independent);
    }
}
