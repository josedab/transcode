//! NAL (Network Abstraction Layer) unit parsing.
//!
//! NAL units are the fundamental unit of H.264 bitstreams.

use transcode_core::bitstream::{find_start_code, remove_emulation_prevention, BitReader};
use transcode_core::error::{CodecError, Result};

/// Maximum size for a single NAL unit (10 MB).
/// Prevents excessive memory allocation from malformed length fields.
const MAX_NAL_UNIT_SIZE: usize = 10 * 1024 * 1024;

/// NAL unit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Unspecified.
    Unspecified = 0,
    /// Non-IDR slice.
    Slice = 1,
    /// Slice data partition A.
    SliceDataA = 2,
    /// Slice data partition B.
    SliceDataB = 3,
    /// Slice data partition C.
    SliceDataC = 4,
    /// IDR slice.
    IdrSlice = 5,
    /// Supplemental enhancement information (SEI).
    Sei = 6,
    /// Sequence parameter set (SPS).
    Sps = 7,
    /// Picture parameter set (PPS).
    Pps = 8,
    /// Access unit delimiter.
    Aud = 9,
    /// End of sequence.
    EndOfSequence = 10,
    /// End of stream.
    EndOfStream = 11,
    /// Filler data.
    Filler = 12,
    /// SPS extension.
    SpsExt = 13,
    /// Prefix NAL unit.
    Prefix = 14,
    /// Subset SPS.
    SubsetSps = 15,
    /// Depth parameter set.
    Dps = 16,
    /// Coded slice of an auxiliary coded picture.
    SliceAux = 19,
    /// Coded slice extension.
    SliceExt = 20,
    /// Coded slice extension for depth view.
    SliceExtDepth = 21,
    /// Unknown/reserved type.
    Unknown(u8),
}

impl NalUnitType {
    /// Create from raw NAL unit type value.
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Unspecified,
            1 => Self::Slice,
            2 => Self::SliceDataA,
            3 => Self::SliceDataB,
            4 => Self::SliceDataC,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::Aud,
            10 => Self::EndOfSequence,
            11 => Self::EndOfStream,
            12 => Self::Filler,
            13 => Self::SpsExt,
            14 => Self::Prefix,
            15 => Self::SubsetSps,
            16 => Self::Dps,
            19 => Self::SliceAux,
            20 => Self::SliceExt,
            21 => Self::SliceExtDepth,
            n => Self::Unknown(n),
        }
    }

    /// Get the raw value.
    pub fn to_u8(&self) -> u8 {
        match self {
            Self::Unspecified => 0,
            Self::Slice => 1,
            Self::SliceDataA => 2,
            Self::SliceDataB => 3,
            Self::SliceDataC => 4,
            Self::IdrSlice => 5,
            Self::Sei => 6,
            Self::Sps => 7,
            Self::Pps => 8,
            Self::Aud => 9,
            Self::EndOfSequence => 10,
            Self::EndOfStream => 11,
            Self::Filler => 12,
            Self::SpsExt => 13,
            Self::Prefix => 14,
            Self::SubsetSps => 15,
            Self::Dps => 16,
            Self::SliceAux => 19,
            Self::SliceExt => 20,
            Self::SliceExtDepth => 21,
            Self::Unknown(n) => *n,
        }
    }

    /// Check if this is a VCL (Video Coding Layer) NAL unit.
    pub fn is_vcl(&self) -> bool {
        matches!(
            self,
            Self::Slice
                | Self::SliceDataA
                | Self::SliceDataB
                | Self::SliceDataC
                | Self::IdrSlice
                | Self::SliceAux
                | Self::SliceExt
                | Self::SliceExtDepth
        )
    }

    /// Check if this NAL starts an access unit.
    pub fn starts_access_unit(&self) -> bool {
        matches!(
            self,
            Self::Aud | Self::Sps | Self::Pps | Self::Sei | Self::IdrSlice
        )
    }

    /// Check if this is a reference picture.
    pub fn is_reference(&self) -> bool {
        matches!(
            self,
            Self::Slice
                | Self::SliceDataA
                | Self::IdrSlice
                | Self::Sps
                | Self::Pps
                | Self::SpsExt
                | Self::SubsetSps
        )
    }
}

/// A parsed NAL unit.
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// NAL unit type.
    pub nal_type: NalUnitType,
    /// NAL reference IDC (0-3).
    pub nal_ref_idc: u8,
    /// Raw NAL unit data (RBSP, with emulation prevention bytes removed).
    pub data: Vec<u8>,
}

impl NalUnit {
    /// Parse a NAL unit from raw data (including the NAL header byte).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidNalUnit("Empty NAL unit".into()).into());
        }

        let header = data[0];
        let forbidden_zero_bit = (header >> 7) & 1;
        let nal_ref_idc = (header >> 5) & 3;
        let nal_type = NalUnitType::from_u8(header & 0x1F);

        if forbidden_zero_bit != 0 {
            return Err(CodecError::InvalidNalUnit("Forbidden zero bit is set".into()).into());
        }

        // Remove emulation prevention bytes from the payload
        let rbsp = if data.len() > 1 {
            remove_emulation_prevention(&data[1..])
        } else {
            Vec::new()
        };

        Ok(Self {
            nal_type,
            nal_ref_idc,
            data: rbsp,
        })
    }

    /// Create a BitReader for this NAL unit's RBSP data.
    pub fn bitstream(&self) -> BitReader<'_> {
        BitReader::new(&self.data)
    }

    /// Check if this is an IDR picture.
    pub fn is_idr(&self) -> bool {
        self.nal_type == NalUnitType::IdrSlice
    }

    /// Check if this is a slice (IDR or non-IDR).
    pub fn is_slice(&self) -> bool {
        matches!(self.nal_type, NalUnitType::Slice | NalUnitType::IdrSlice)
    }
}

/// NAL unit iterator for parsing an Annex B bitstream.
pub struct NalIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> NalIterator<'a> {
    /// Create a new NAL iterator for an Annex B bitstream.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> Iterator for NalIterator<'a> {
    type Item = Result<NalUnit>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        // Find the start of this NAL unit
        let remaining = &self.data[self.pos..];
        let start_code = find_start_code(remaining)?;
        let nal_start = self.pos + start_code.0 + start_code.1;

        if nal_start >= self.data.len() {
            self.pos = self.data.len();
            return None;
        }

        // Find the start of the next NAL unit (or end of data)
        let after_start = &self.data[nal_start..];
        let nal_end = if let Some((next_start, _)) = find_start_code(after_start) {
            nal_start + next_start
        } else {
            self.data.len()
        };

        // Update position for next iteration
        self.pos = nal_end;

        // Parse the NAL unit
        let nal_data = &self.data[nal_start..nal_end];
        Some(NalUnit::parse(nal_data))
    }
}

/// Parse NAL units from an Annex B byte stream.
#[allow(dead_code)]
pub fn parse_annex_b(data: &[u8]) -> Vec<Result<NalUnit>> {
    NalIterator::new(data).collect()
}

/// Parse NAL units from AVCC format (length-prefixed).
pub fn parse_avcc(data: &[u8], length_size: usize) -> Vec<Result<NalUnit>> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos + length_size <= data.len() {
        // Read length prefix
        let length = match length_size {
            1 => data[pos] as usize,
            2 => u16::from_be_bytes([data[pos], data[pos + 1]]) as usize,
            4 => u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize,
            _ => {
                result.push(Err(
                    CodecError::InvalidNalUnit(format!("Invalid length size: {}", length_size))
                        .into(),
                ));
                break;
            }
        };

        pos += length_size;

        // Validate NAL unit size to prevent excessive allocation
        if length > MAX_NAL_UNIT_SIZE {
            result.push(Err(
                CodecError::InvalidNalUnit(format!(
                    "NAL unit size {} exceeds maximum {}",
                    length, MAX_NAL_UNIT_SIZE
                ))
                .into(),
            ));
            break;
        }

        if pos + length > data.len() {
            result.push(Err(CodecError::InvalidNalUnit("NAL unit truncated".into()).into()));
            break;
        }

        result.push(NalUnit::parse(&data[pos..pos + length]));
        pos += length;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_type() {
        assert_eq!(NalUnitType::from_u8(7), NalUnitType::Sps);
        assert_eq!(NalUnitType::from_u8(8), NalUnitType::Pps);
        assert_eq!(NalUnitType::from_u8(5), NalUnitType::IdrSlice);
        assert!(NalUnitType::IdrSlice.is_vcl());
        assert!(!NalUnitType::Sps.is_vcl());
    }

    #[test]
    fn test_nal_unit_type_all_values() {
        // Test all known NAL types
        assert_eq!(NalUnitType::from_u8(0), NalUnitType::Unspecified);
        assert_eq!(NalUnitType::from_u8(1), NalUnitType::Slice);
        assert_eq!(NalUnitType::from_u8(2), NalUnitType::SliceDataA);
        assert_eq!(NalUnitType::from_u8(3), NalUnitType::SliceDataB);
        assert_eq!(NalUnitType::from_u8(4), NalUnitType::SliceDataC);
        assert_eq!(NalUnitType::from_u8(6), NalUnitType::Sei);
        assert_eq!(NalUnitType::from_u8(9), NalUnitType::Aud);
        assert_eq!(NalUnitType::from_u8(10), NalUnitType::EndOfSequence);
        assert_eq!(NalUnitType::from_u8(11), NalUnitType::EndOfStream);
        assert_eq!(NalUnitType::from_u8(12), NalUnitType::Filler);

        // Unknown types
        assert!(matches!(NalUnitType::from_u8(30), NalUnitType::Unknown(30)));
    }

    #[test]
    fn test_nal_unit_type_to_u8() {
        assert_eq!(NalUnitType::Sps.to_u8(), 7);
        assert_eq!(NalUnitType::Pps.to_u8(), 8);
        assert_eq!(NalUnitType::IdrSlice.to_u8(), 5);
        assert_eq!(NalUnitType::Unknown(30).to_u8(), 30);
    }

    #[test]
    fn test_nal_unit_type_is_vcl() {
        assert!(NalUnitType::Slice.is_vcl());
        assert!(NalUnitType::SliceDataA.is_vcl());
        assert!(NalUnitType::IdrSlice.is_vcl());
        assert!(!NalUnitType::Sps.is_vcl());
        assert!(!NalUnitType::Pps.is_vcl());
        assert!(!NalUnitType::Sei.is_vcl());
    }

    #[test]
    fn test_nal_unit_type_starts_access_unit() {
        assert!(NalUnitType::Aud.starts_access_unit());
        assert!(NalUnitType::Sps.starts_access_unit());
        assert!(NalUnitType::Pps.starts_access_unit());
        assert!(NalUnitType::IdrSlice.starts_access_unit());
        assert!(!NalUnitType::Slice.starts_access_unit());
    }

    #[test]
    fn test_nal_unit_parse() {
        // NAL header: nal_ref_idc=3, nal_unit_type=7 (SPS)
        let data = [0x67, 0x42, 0x00, 0x1E];
        let nal = NalUnit::parse(&data).unwrap();
        assert_eq!(nal.nal_type, NalUnitType::Sps);
        assert_eq!(nal.nal_ref_idc, 3);
    }

    #[test]
    fn test_nal_unit_parse_empty() {
        let data: [u8; 0] = [];
        assert!(NalUnit::parse(&data).is_err());
    }

    #[test]
    fn test_nal_unit_parse_forbidden_bit() {
        // NAL header with forbidden_zero_bit set
        let data = [0x87, 0x42, 0x00, 0x1E]; // 0x87 = 1000_0111, forbidden bit set
        assert!(NalUnit::parse(&data).is_err());
    }

    #[test]
    fn test_nal_unit_is_idr() {
        let idr_data = [0x65]; // nal_ref_idc=3, nal_unit_type=5 (IDR)
        let nal = NalUnit::parse(&idr_data).unwrap();
        assert!(nal.is_idr());

        let non_idr_data = [0x61]; // nal_ref_idc=3, nal_unit_type=1 (Slice)
        let nal = NalUnit::parse(&non_idr_data).unwrap();
        assert!(!nal.is_idr());
    }

    #[test]
    fn test_nal_unit_is_slice() {
        // IDR slice
        let idr_data = [0x65];
        let nal = NalUnit::parse(&idr_data).unwrap();
        assert!(nal.is_slice());

        // Non-IDR slice
        let slice_data = [0x61];
        let nal = NalUnit::parse(&slice_data).unwrap();
        assert!(nal.is_slice());

        // SPS (not a slice)
        let sps_data = [0x67];
        let nal = NalUnit::parse(&sps_data).unwrap();
        assert!(!nal.is_slice());
    }

    #[test]
    fn test_annex_b_parsing() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, // SPS
            0x00, 0x00, 0x01, 0x68, 0xCE, // PPS
        ];
        let nals: Vec<_> = NalIterator::new(&data).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].as_ref().unwrap().nal_type, NalUnitType::Sps);
        assert_eq!(nals[1].as_ref().unwrap().nal_type, NalUnitType::Pps);
    }

    #[test]
    fn test_annex_b_parsing_empty() {
        let data: [u8; 0] = [];
        let nals: Vec<_> = NalIterator::new(&data).collect();
        assert_eq!(nals.len(), 0);
    }

    #[test]
    fn test_avcc_parsing() {
        // Two NAL units with 4-byte length prefix
        let data = [
            0x00, 0x00, 0x00, 0x02, 0x67, 0x42, // SPS (length=2)
            0x00, 0x00, 0x00, 0x02, 0x68, 0xCE, // PPS (length=2)
        ];
        let nals = parse_avcc(&data, 4);
        assert_eq!(nals.len(), 2);
        assert!(nals[0].is_ok());
        assert!(nals[1].is_ok());
    }

    #[test]
    fn test_avcc_parsing_invalid_length_size() {
        let data = [0x00, 0x00, 0x00, 0x02, 0x67, 0x42];
        let nals = parse_avcc(&data, 3); // Invalid length size
        assert_eq!(nals.len(), 1);
        assert!(nals[0].is_err());
    }

    #[test]
    fn test_avcc_parsing_truncated() {
        // NAL unit length says 10 bytes but only 2 are available
        let data = [
            0x00, 0x00, 0x00, 0x0A, 0x67, 0x42,
        ];
        let nals = parse_avcc(&data, 4);
        assert_eq!(nals.len(), 1);
        assert!(nals[0].is_err());
    }
}
