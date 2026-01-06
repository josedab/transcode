//! DNxHD/DNxHR profile definitions

use crate::types::{BitDepth, ChromaFormat};

/// DNxHD/DNxHR profile variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnxProfile {
    // DNxHD profiles (HD resolution)
    /// DNxHD 36 - Low bandwidth 1080p/29.97
    Dnxhd36,
    /// DNxHD 45 - Low bandwidth 1080p/25
    Dnxhd45,
    /// DNxHD 90 - Medium quality 8-bit
    Dnxhd90,
    /// DNxHD 90x - Medium quality 10-bit
    Dnxhd90x,
    /// DNxHD 120 - High quality 1080i
    Dnxhd120,
    /// DNxHD 145 - High quality 1080p
    Dnxhd145,
    /// DNxHD 175 - High quality 10-bit
    Dnxhd175,
    /// DNxHD 175x - High quality 10-bit (alternate)
    Dnxhd175x,
    /// DNxHD 220 - Highest quality 8-bit
    Dnxhd220,
    /// DNxHD 220x - Highest quality 10-bit
    Dnxhd220x,

    // DNxHR profiles (higher resolutions)
    /// DNxHR LB - Low Bandwidth (proxy)
    DnxhrLb,
    /// DNxHR SQ - Standard Quality
    DnxhrSq,
    /// DNxHR HQ - High Quality 8-bit
    DnxhrHq,
    /// DNxHR HQX - High Quality 10-bit
    DnxhrHqx,
    /// DNxHR 444 - Full chroma 4:4:4
    Dnxhr444,
}

impl DnxProfile {
    /// Get profile info for this profile
    pub fn info(&self) -> ProfileInfo {
        match self {
            // DNxHD profiles
            DnxProfile::Dnxhd36 => ProfileInfo {
                profile_id: 1235,
                name: "DNxHD 36",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 36,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd45 => ProfileInfo {
                profile_id: 1237,
                name: "DNxHD 45",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 45,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd90 => ProfileInfo {
                profile_id: 1241,
                name: "DNxHD 90",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 90,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd90x => ProfileInfo {
                profile_id: 1242,
                name: "DNxHD 90x",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 90,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd120 => ProfileInfo {
                profile_id: 1243,
                name: "DNxHD 120",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 120,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd145 => ProfileInfo {
                profile_id: 1244,
                name: "DNxHD 145",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 145,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd175 => ProfileInfo {
                profile_id: 1250,
                name: "DNxHD 175",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 175,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd175x => ProfileInfo {
                profile_id: 1251,
                name: "DNxHD 175x",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 175,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd220 => ProfileInfo {
                profile_id: 1252,
                name: "DNxHD 220",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 220,
                is_dnxhr: false,
            },
            DnxProfile::Dnxhd220x => ProfileInfo {
                profile_id: 1253,
                name: "DNxHD 220x",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 220,
                is_dnxhr: false,
            },

            // DNxHR profiles
            DnxProfile::DnxhrLb => ProfileInfo {
                profile_id: 1270,
                name: "DNxHR LB",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 80, // Variable based on resolution
                is_dnxhr: true,
            },
            DnxProfile::DnxhrSq => ProfileInfo {
                profile_id: 1271,
                name: "DNxHR SQ",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 150,
                is_dnxhr: true,
            },
            DnxProfile::DnxhrHq => ProfileInfo {
                profile_id: 1272,
                name: "DNxHR HQ",
                bit_depth: BitDepth::Bit8,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 220,
                is_dnxhr: true,
            },
            DnxProfile::DnxhrHqx => ProfileInfo {
                profile_id: 1273,
                name: "DNxHR HQX",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV422,
                target_bitrate_mbps: 220,
                is_dnxhr: true,
            },
            DnxProfile::Dnxhr444 => ProfileInfo {
                profile_id: 1274,
                name: "DNxHR 444",
                bit_depth: BitDepth::Bit10,
                chroma_format: ChromaFormat::YUV444,
                target_bitrate_mbps: 440,
                is_dnxhr: true,
            },
        }
    }

    /// Get profile from ID
    pub fn from_id(id: u32) -> Option<DnxProfile> {
        match id {
            1235 => Some(DnxProfile::Dnxhd36),
            1237 => Some(DnxProfile::Dnxhd45),
            1241 => Some(DnxProfile::Dnxhd90),
            1242 => Some(DnxProfile::Dnxhd90x),
            1243 => Some(DnxProfile::Dnxhd120),
            1244 => Some(DnxProfile::Dnxhd145),
            1250 => Some(DnxProfile::Dnxhd175),
            1251 => Some(DnxProfile::Dnxhd175x),
            1252 => Some(DnxProfile::Dnxhd220),
            1253 => Some(DnxProfile::Dnxhd220x),
            1270 => Some(DnxProfile::DnxhrLb),
            1271 => Some(DnxProfile::DnxhrSq),
            1272 => Some(DnxProfile::DnxhrHq),
            1273 => Some(DnxProfile::DnxhrHqx),
            1274 => Some(DnxProfile::Dnxhr444),
            _ => None,
        }
    }

    /// Get the profile ID
    pub fn id(&self) -> u32 {
        self.info().profile_id
    }

    /// Get the bit depth for this profile
    pub fn bit_depth(&self) -> BitDepth {
        self.info().bit_depth
    }

    /// Get the chroma format for this profile
    pub fn chroma_format(&self) -> ChromaFormat {
        self.info().chroma_format
    }

    /// Check if this is a DNxHR profile
    pub fn is_dnxhr(&self) -> bool {
        self.info().is_dnxhr
    }

    /// Check if this profile supports 4:4:4 chroma
    pub fn is_444(&self) -> bool {
        matches!(self, DnxProfile::Dnxhr444)
    }

    /// Get quantization scale factor for this profile
    pub fn quant_scale(&self) -> u8 {
        match self {
            DnxProfile::Dnxhd36 | DnxProfile::Dnxhd45 => 4,
            DnxProfile::Dnxhd90 | DnxProfile::Dnxhd90x => 2,
            DnxProfile::DnxhrLb => 4,
            DnxProfile::DnxhrSq => 2,
            _ => 1,
        }
    }
}

/// Profile information
#[derive(Debug, Clone)]
pub struct ProfileInfo {
    /// Profile ID as used in the frame header
    pub profile_id: u32,
    /// Human-readable profile name
    pub name: &'static str,
    /// Bit depth for this profile
    pub bit_depth: BitDepth,
    /// Chroma format for this profile
    pub chroma_format: ChromaFormat,
    /// Target bitrate in Mbps
    pub target_bitrate_mbps: u32,
    /// Whether this is a DNxHR profile (vs DNxHD)
    pub is_dnxhr: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_id() {
        assert_eq!(DnxProfile::from_id(1235), Some(DnxProfile::Dnxhd36));
        assert_eq!(DnxProfile::from_id(1253), Some(DnxProfile::Dnxhd220x));
        assert_eq!(DnxProfile::from_id(1273), Some(DnxProfile::DnxhrHqx));
        assert_eq!(DnxProfile::from_id(9999), None);
    }

    #[test]
    fn test_profile_info() {
        let profile = DnxProfile::Dnxhd220x;
        let info = profile.info();
        assert_eq!(info.name, "DNxHD 220x");
        assert_eq!(info.bit_depth, BitDepth::Bit10);
        assert!(!info.is_dnxhr);
    }

    #[test]
    fn test_dnxhr_profiles() {
        assert!(DnxProfile::DnxhrHq.is_dnxhr());
        assert!(!DnxProfile::Dnxhd220x.is_dnxhr());
    }

    #[test]
    fn test_444_profile() {
        assert!(DnxProfile::Dnxhr444.is_444());
        assert!(!DnxProfile::DnxhrHqx.is_444());
    }

    #[test]
    fn test_all_profiles_have_unique_ids() {
        let profiles = [
            DnxProfile::Dnxhd36,
            DnxProfile::Dnxhd45,
            DnxProfile::Dnxhd90,
            DnxProfile::Dnxhd90x,
            DnxProfile::Dnxhd120,
            DnxProfile::Dnxhd145,
            DnxProfile::Dnxhd175,
            DnxProfile::Dnxhd175x,
            DnxProfile::Dnxhd220,
            DnxProfile::Dnxhd220x,
            DnxProfile::DnxhrLb,
            DnxProfile::DnxhrSq,
            DnxProfile::DnxhrHq,
            DnxProfile::DnxhrHqx,
            DnxProfile::Dnxhr444,
        ];

        let mut ids: Vec<_> = profiles.iter().map(|p| p.id()).collect();
        let unique_count = {
            ids.sort();
            ids.dedup();
            ids.len()
        };

        assert_eq!(unique_count, profiles.len());
    }
}
