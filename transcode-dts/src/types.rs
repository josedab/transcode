//! DTS and TrueHD data types.

use std::fmt;

/// DTS sync frame information.
#[derive(Debug, Clone, PartialEq)]
pub struct DtsSyncFrame {
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Audio channel arrangement.
    pub amode: AudioMode,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bit rate in kbps.
    pub bit_rate: u32,
    /// Number of PCM samples per channel.
    pub samples_per_channel: usize,
    /// Number of audio channels.
    pub channels: u8,
    /// LFE (subwoofer) channel present.
    pub lfe: bool,
    /// Source PCM resolution.
    pub pcm_resolution: u8,
    /// Frame checksum present.
    pub crc_present: bool,
    /// Number of PCM sample blocks.
    pub nblks: u8,
    /// Primary frame size.
    pub fsize: u16,
    /// Audio coding mode.
    pub sfreq: u8,
    /// Transmission bit rate.
    pub rate: u8,
    /// Embedded dynamic range flag.
    pub dynf: bool,
    /// Embedded timestamp flag.
    pub timef: bool,
    /// Auxiliary data flag.
    pub auxf: bool,
    /// HDCD master audio.
    pub hdcd: bool,
    /// Extension audio descriptor.
    pub ext_audio_id: ExtensionAudioId,
    /// Extended coding flag.
    pub ext_audio: bool,
    /// Audio sync word insertion flag.
    pub aspf: bool,
    /// Low frequency effects flag.
    pub lff: u8,
    /// Predictor history flag switch.
    pub hflag: bool,
    /// CRC type.
    pub crc_type: u8,
    /// Multirate interpolator switch.
    pub filts: bool,
    /// Encoder software revision.
    pub vernum: u8,
    /// Copy history.
    pub chist: u8,
    /// Dialog normalization.
    pub dialog_norm: i8,
}

impl DtsSyncFrame {
    /// Create a new DTS sync frame with default values.
    pub fn new() -> Self {
        Self {
            frame_size: 0,
            amode: AudioMode::Stereo,
            sample_rate: 48000,
            bit_rate: 768,
            samples_per_channel: 512,
            channels: 2,
            lfe: false,
            pcm_resolution: 16,
            crc_present: false,
            nblks: 7,
            fsize: 0,
            sfreq: 13,
            rate: 0,
            dynf: false,
            timef: false,
            auxf: false,
            hdcd: false,
            ext_audio_id: ExtensionAudioId::None,
            ext_audio: false,
            aspf: false,
            lff: 0,
            hflag: false,
            crc_type: 0,
            filts: false,
            vernum: 0,
            chist: 0,
            dialog_norm: 0,
        }
    }

    /// Check if this is a DTS-HD frame.
    pub fn is_hd(&self) -> bool {
        self.ext_audio && matches!(self.ext_audio_id, ExtensionAudioId::XCh | ExtensionAudioId::Xxch | ExtensionAudioId::X96)
    }

    /// Get the duration of this frame in seconds.
    pub fn duration(&self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }

    /// Get total number of channels including LFE.
    pub fn total_channels(&self) -> u8 {
        self.channels + if self.lfe { 1 } else { 0 }
    }
}

impl Default for DtsSyncFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// DTS-HD extension frame information.
#[derive(Debug, Clone, PartialEq)]
pub struct DtsHdFrame {
    /// Extension type.
    pub extension_type: DtsHdExtension,
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Number of audio assets.
    pub num_assets: u8,
    /// Number of audio presentations.
    pub num_presentations: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bit depth.
    pub bit_depth: u8,
    /// Channel count.
    pub channels: u8,
    /// Lossless extension.
    pub lossless: bool,
    /// Profile (DTS-HD High Resolution or Master Audio).
    pub profile: DtsHdProfile,
    /// DTS:X present.
    pub dtsx: bool,
    /// Number of samples.
    pub num_samples: usize,
}

impl DtsHdFrame {
    /// Create a new DTS-HD frame.
    pub fn new() -> Self {
        Self {
            extension_type: DtsHdExtension::HdMasterAudio,
            frame_size: 0,
            num_assets: 1,
            num_presentations: 1,
            sample_rate: 48000,
            bit_depth: 24,
            channels: 6,
            lossless: false,
            profile: DtsHdProfile::HighResolution,
            dtsx: false,
            num_samples: 0,
        }
    }
}

impl Default for DtsHdFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// TrueHD sync frame information.
#[derive(Debug, Clone, PartialEq)]
pub struct TrueHdSyncFrame {
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels (up to 16).
    pub channels: u8,
    /// Bit depth (16, 20, or 24).
    pub bit_depth: u8,
    /// Number of samples per frame.
    pub samples_per_channel: usize,
    /// Group size in frames.
    pub group_size: u16,
    /// Access unit size in samples.
    pub access_unit_size: u16,
    /// Peak bitrate in kbps.
    pub peak_bitrate: u32,
    /// Channel assignment.
    pub channel_assignment: TrueHdChannelAssignment,
    /// Dolby Atmos present.
    pub atmos: bool,
    /// Substream count.
    pub substream_count: u8,
    /// CRC present.
    pub crc_present: bool,
    /// Major sync interval.
    pub major_sync_interval: u16,
}

impl TrueHdSyncFrame {
    /// Create a new TrueHD sync frame.
    pub fn new() -> Self {
        Self {
            frame_size: 0,
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
            samples_per_channel: 40,
            group_size: 1,
            access_unit_size: 40,
            peak_bitrate: 0,
            channel_assignment: TrueHdChannelAssignment::Stereo,
            atmos: false,
            substream_count: 1,
            crc_present: false,
            major_sync_interval: 0,
        }
    }

    /// Check if this is a Dolby Atmos stream.
    pub fn is_atmos(&self) -> bool {
        self.atmos
    }

    /// Get the duration of this frame in seconds.
    pub fn duration(&self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }
}

impl Default for TrueHdSyncFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// DTS audio mode (channel configuration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum AudioMode {
    /// Mono (1.0).
    Mono = 0,
    /// Dual mono.
    DualMono = 1,
    /// Stereo (2.0).
    Stereo = 2,
    /// Stereo with sum/difference coding.
    StereoSumDiff = 3,
    /// Stereo with total coding.
    StereoTotal = 4,
    /// 3/0 (L, C, R).
    Front3_0 = 5,
    /// 2/1 (L, R, S).
    Front2_Rear1 = 6,
    /// 3/1 (L, C, R, S).
    Front3_Rear1 = 7,
    /// 2/2 (L, R, SL, SR).
    Front2_Rear2 = 8,
    /// 3/2 (L, C, R, SL, SR) - 5.0.
    Front3_Rear2 = 9,
    /// 4/2 with center surround.
    Front4_Rear2 = 10,
    /// 3/3 with center surround.
    Front3_Rear3 = 11,
    /// 4/2/1 with overhead.
    Front4_Rear2_Overhead1 = 12,
    /// 3/2/2 with height.
    Front3_Rear2_Height2 = 13,
    /// User defined.
    UserDefined = 14,
}

impl AudioMode {
    /// Parse from amode value.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => AudioMode::Mono,
            1 => AudioMode::DualMono,
            2 => AudioMode::Stereo,
            3 => AudioMode::StereoSumDiff,
            4 => AudioMode::StereoTotal,
            5 => AudioMode::Front3_0,
            6 => AudioMode::Front2_Rear1,
            7 => AudioMode::Front3_Rear1,
            8 => AudioMode::Front2_Rear2,
            9 => AudioMode::Front3_Rear2,
            10 => AudioMode::Front4_Rear2,
            11 => AudioMode::Front3_Rear3,
            12 => AudioMode::Front4_Rear2_Overhead1,
            13 => AudioMode::Front3_Rear2_Height2,
            _ => AudioMode::UserDefined,
        }
    }

    /// Get the number of channels for this mode.
    pub fn channel_count(&self) -> u8 {
        match self {
            AudioMode::Mono => 1,
            AudioMode::DualMono | AudioMode::Stereo | AudioMode::StereoSumDiff | AudioMode::StereoTotal => 2,
            AudioMode::Front3_0 | AudioMode::Front2_Rear1 => 3,
            AudioMode::Front3_Rear1 | AudioMode::Front2_Rear2 => 4,
            AudioMode::Front3_Rear2 => 5,
            AudioMode::Front4_Rear2 | AudioMode::Front3_Rear3 => 6,
            AudioMode::Front4_Rear2_Overhead1 | AudioMode::Front3_Rear2_Height2 => 7,
            AudioMode::UserDefined => 8,
        }
    }
}

impl fmt::Display for AudioMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioMode::Mono => write!(f, "1.0 (Mono)"),
            AudioMode::DualMono => write!(f, "1+1 (Dual Mono)"),
            AudioMode::Stereo => write!(f, "2.0 (Stereo)"),
            AudioMode::StereoSumDiff => write!(f, "2.0 (Sum/Diff)"),
            AudioMode::StereoTotal => write!(f, "2.0 (Total)"),
            AudioMode::Front3_0 => write!(f, "3.0 (L/C/R)"),
            AudioMode::Front2_Rear1 => write!(f, "2.1 (L/R/S)"),
            AudioMode::Front3_Rear1 => write!(f, "3.1 (L/C/R/S)"),
            AudioMode::Front2_Rear2 => write!(f, "2.2 (Quad)"),
            AudioMode::Front3_Rear2 => write!(f, "5.0"),
            AudioMode::Front4_Rear2 => write!(f, "6.0"),
            AudioMode::Front3_Rear3 => write!(f, "6.0"),
            AudioMode::Front4_Rear2_Overhead1 => write!(f, "7.0"),
            AudioMode::Front3_Rear2_Height2 => write!(f, "7.0"),
            AudioMode::UserDefined => write!(f, "User Defined"),
        }
    }
}

/// Extension audio ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionAudioId {
    /// No extension.
    None,
    /// XCh - 6.1 extension.
    XCh,
    /// XXCh - 7.1 extension.
    Xxch,
    /// X96 - 96kHz extension.
    X96,
}

impl ExtensionAudioId {
    /// Parse from extension audio ID value.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => ExtensionAudioId::XCh,
            2 => ExtensionAudioId::X96,
            6 => ExtensionAudioId::Xxch,
            _ => ExtensionAudioId::None,
        }
    }
}

impl fmt::Display for ExtensionAudioId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtensionAudioId::None => write!(f, "None"),
            ExtensionAudioId::XCh => write!(f, "XCh (6.1)"),
            ExtensionAudioId::Xxch => write!(f, "XXCh (7.1)"),
            ExtensionAudioId::X96 => write!(f, "X96 (96kHz)"),
        }
    }
}

/// DTS-HD extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtsHdExtension {
    /// DTS-HD High Resolution.
    HdHighResolution,
    /// DTS-HD Master Audio.
    HdMasterAudio,
    /// DTS Express.
    Express,
    /// DTS-HD low bit rate extension.
    LbrExtension,
    /// DTS:X.
    DtsX,
}

impl fmt::Display for DtsHdExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DtsHdExtension::HdHighResolution => write!(f, "DTS-HD High Resolution"),
            DtsHdExtension::HdMasterAudio => write!(f, "DTS-HD Master Audio"),
            DtsHdExtension::Express => write!(f, "DTS Express"),
            DtsHdExtension::LbrExtension => write!(f, "DTS-HD LBR"),
            DtsHdExtension::DtsX => write!(f, "DTS:X"),
        }
    }
}

/// DTS-HD profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtsHdProfile {
    /// Core only.
    Core,
    /// High Resolution (lossy extension).
    HighResolution,
    /// Master Audio (lossless).
    MasterAudio,
    /// Express (low bitrate).
    Express,
    /// DTS:X (object-based).
    DtsX,
}

impl fmt::Display for DtsHdProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DtsHdProfile::Core => write!(f, "DTS Core"),
            DtsHdProfile::HighResolution => write!(f, "DTS-HD HR"),
            DtsHdProfile::MasterAudio => write!(f, "DTS-HD MA"),
            DtsHdProfile::Express => write!(f, "DTS Express"),
            DtsHdProfile::DtsX => write!(f, "DTS:X"),
        }
    }
}

/// TrueHD channel assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TrueHdChannelAssignment {
    /// Mono.
    Mono,
    /// Stereo.
    #[default]
    Stereo,
    /// 3.0 (L, C, R).
    Layout3_0,
    /// 4.0 (L, R, Ls, Rs).
    Layout4_0,
    /// 5.0 (L, C, R, Ls, Rs).
    Layout5_0,
    /// 5.1 (L, C, R, Ls, Rs, LFE).
    Layout5_1,
    /// 7.1 (L, C, R, Ls, Rs, Lb, Rb, LFE).
    Layout7_1,
    /// Atmos (object-based).
    Atmos,
    /// Custom layout.
    Custom(u8),
}

impl TrueHdChannelAssignment {
    /// Get the number of channels.
    pub fn channel_count(&self) -> u8 {
        match self {
            TrueHdChannelAssignment::Mono => 1,
            TrueHdChannelAssignment::Stereo => 2,
            TrueHdChannelAssignment::Layout3_0 => 3,
            TrueHdChannelAssignment::Layout4_0 => 4,
            TrueHdChannelAssignment::Layout5_0 => 5,
            TrueHdChannelAssignment::Layout5_1 => 6,
            TrueHdChannelAssignment::Layout7_1 => 8,
            TrueHdChannelAssignment::Atmos => 8, // Minimum for Atmos
            TrueHdChannelAssignment::Custom(n) => *n,
        }
    }
}

impl fmt::Display for TrueHdChannelAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrueHdChannelAssignment::Mono => write!(f, "1.0"),
            TrueHdChannelAssignment::Stereo => write!(f, "2.0"),
            TrueHdChannelAssignment::Layout3_0 => write!(f, "3.0"),
            TrueHdChannelAssignment::Layout4_0 => write!(f, "4.0"),
            TrueHdChannelAssignment::Layout5_0 => write!(f, "5.0"),
            TrueHdChannelAssignment::Layout5_1 => write!(f, "5.1"),
            TrueHdChannelAssignment::Layout7_1 => write!(f, "7.1"),
            TrueHdChannelAssignment::Atmos => write!(f, "7.1 Atmos"),
            TrueHdChannelAssignment::Custom(n) => write!(f, "Custom({})", n),
        }
    }
}

/// Channel layout for decoded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChannelLayout {
    /// Front left.
    pub front_left: bool,
    /// Front right.
    pub front_right: bool,
    /// Front center.
    pub front_center: bool,
    /// LFE (subwoofer).
    pub lfe: bool,
    /// Back (surround) left.
    pub back_left: bool,
    /// Back (surround) right.
    pub back_right: bool,
    /// Back center.
    pub back_center: bool,
    /// Side left.
    pub side_left: bool,
    /// Side right.
    pub side_right: bool,
    /// Top front left.
    pub top_front_left: bool,
    /// Top front right.
    pub top_front_right: bool,
    /// Top front center.
    pub top_front_center: bool,
    /// Top back left.
    pub top_back_left: bool,
    /// Top back right.
    pub top_back_right: bool,
    /// Top back center.
    pub top_back_center: bool,
}

impl ChannelLayout {
    /// Create a mono layout.
    pub fn mono() -> Self {
        Self {
            front_center: true,
            ..Default::default()
        }
    }

    /// Create a stereo layout.
    pub fn stereo() -> Self {
        Self {
            front_left: true,
            front_right: true,
            ..Default::default()
        }
    }

    /// Create a 5.1 layout.
    pub fn surround_5_1() -> Self {
        Self {
            front_left: true,
            front_right: true,
            front_center: true,
            lfe: true,
            back_left: true,
            back_right: true,
            ..Default::default()
        }
    }

    /// Create a 7.1 layout.
    pub fn surround_7_1() -> Self {
        Self {
            front_left: true,
            front_right: true,
            front_center: true,
            lfe: true,
            back_left: true,
            back_right: true,
            side_left: true,
            side_right: true,
            ..Default::default()
        }
    }

    /// Get the total number of channels.
    pub fn channel_count(&self) -> u8 {
        let mut count = 0u8;
        if self.front_left { count += 1; }
        if self.front_right { count += 1; }
        if self.front_center { count += 1; }
        if self.lfe { count += 1; }
        if self.back_left { count += 1; }
        if self.back_right { count += 1; }
        if self.back_center { count += 1; }
        if self.side_left { count += 1; }
        if self.side_right { count += 1; }
        if self.top_front_left { count += 1; }
        if self.top_front_right { count += 1; }
        if self.top_front_center { count += 1; }
        if self.top_back_left { count += 1; }
        if self.top_back_right { count += 1; }
        if self.top_back_center { count += 1; }
        count
    }

    /// Create layout from DTS audio mode.
    pub fn from_audio_mode(mode: AudioMode, lfe: bool) -> Self {
        let mut layout = match mode {
            AudioMode::Mono => Self::mono(),
            AudioMode::Stereo | AudioMode::StereoSumDiff | AudioMode::StereoTotal | AudioMode::DualMono => Self::stereo(),
            AudioMode::Front3_0 => Self {
                front_left: true,
                front_center: true,
                front_right: true,
                ..Default::default()
            },
            AudioMode::Front3_Rear2 => Self {
                front_left: true,
                front_center: true,
                front_right: true,
                back_left: true,
                back_right: true,
                ..Default::default()
            },
            _ => Self::surround_5_1(),
        };
        layout.lfe = lfe;
        layout
    }
}

/// Decoded audio buffer.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Channel layout.
    pub layout: ChannelLayout,
    /// Interleaved sample data (f32).
    pub samples: Vec<f32>,
    /// Number of samples per channel.
    pub samples_per_channel: usize,
    /// Bit depth of original audio.
    pub bit_depth: u8,
}

impl DecodedAudio {
    /// Get the total number of samples.
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    /// Get the duration in seconds.
    pub fn duration(&self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_mode() {
        assert_eq!(AudioMode::from_code(0), AudioMode::Mono);
        assert_eq!(AudioMode::from_code(2), AudioMode::Stereo);
        assert_eq!(AudioMode::from_code(9), AudioMode::Front3_Rear2);
        assert_eq!(AudioMode::Mono.channel_count(), 1);
        assert_eq!(AudioMode::Stereo.channel_count(), 2);
        assert_eq!(AudioMode::Front3_Rear2.channel_count(), 5);
    }

    #[test]
    fn test_dts_sync_frame() {
        let frame = DtsSyncFrame::new();
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.channels, 2);
        assert!(!frame.is_hd());
    }

    #[test]
    fn test_truehd_sync_frame() {
        let frame = TrueHdSyncFrame::new();
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.bit_depth, 24);
        assert!(!frame.is_atmos());
    }

    #[test]
    fn test_channel_layout() {
        let mono = ChannelLayout::mono();
        assert_eq!(mono.channel_count(), 1);

        let stereo = ChannelLayout::stereo();
        assert_eq!(stereo.channel_count(), 2);

        let surround = ChannelLayout::surround_5_1();
        assert_eq!(surround.channel_count(), 6);

        let surround_7_1 = ChannelLayout::surround_7_1();
        assert_eq!(surround_7_1.channel_count(), 8);
    }

    #[test]
    fn test_truehd_channel_assignment() {
        assert_eq!(TrueHdChannelAssignment::Stereo.channel_count(), 2);
        assert_eq!(TrueHdChannelAssignment::Layout5_1.channel_count(), 6);
        assert_eq!(TrueHdChannelAssignment::Layout7_1.channel_count(), 8);
    }

    #[test]
    fn test_dts_hd_profile() {
        let profile = DtsHdProfile::MasterAudio;
        assert_eq!(format!("{}", profile), "DTS-HD MA");
    }

    #[test]
    fn test_extension_audio_id() {
        assert_eq!(ExtensionAudioId::from_code(0), ExtensionAudioId::XCh);
        assert_eq!(ExtensionAudioId::from_code(2), ExtensionAudioId::X96);
    }
}
