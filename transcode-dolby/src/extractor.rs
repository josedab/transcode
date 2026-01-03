//! Dolby Vision extraction utilities.
//!
//! This module provides functionality for extracting Dolby Vision data from
//! HEVC streams, including RPU extraction, layer separation, and RPU injection.

use crate::error::{DolbyError, Result};
use crate::profile::DolbyVisionProfile;
use crate::rpu::{Rpu, RPU_NAL_TYPE};

/// NAL unit type for HEVC Unspecified (used for DV RPU).
#[allow(dead_code)]
const HEVC_NAL_UNSPEC62: u8 = 62;

/// HEVC NAL unit start codes.
const NAL_START_CODE_3: [u8; 3] = [0x00, 0x00, 0x01];
const NAL_START_CODE_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Dolby Vision extractor for HEVC streams.
#[derive(Debug)]
pub struct DolbyVisionExtractor {
    /// Extraction mode.
    mode: ExtractionMode,
    /// Buffer for accumulated NAL units.
    nal_buffer: Vec<NalUnit>,
    /// Extracted RPUs.
    rpus: Vec<Rpu>,
    /// Statistics.
    stats: ExtractionStats,
}

impl DolbyVisionExtractor {
    /// Create a new extractor.
    pub fn new(mode: ExtractionMode) -> Self {
        DolbyVisionExtractor {
            mode,
            nal_buffer: Vec::new(),
            rpus: Vec::new(),
            stats: ExtractionStats::default(),
        }
    }

    /// Create an extractor for RPU-only extraction.
    pub fn rpu_only() -> Self {
        DolbyVisionExtractor::new(ExtractionMode::RpuOnly)
    }

    /// Create an extractor for layer separation.
    pub fn layer_separation() -> Self {
        DolbyVisionExtractor::new(ExtractionMode::SeparateLayers)
    }

    /// Process HEVC stream data.
    pub fn process(&mut self, data: &[u8]) -> Result<()> {
        // Find and parse NAL units
        let nal_units = self.find_nal_units(data)?;

        for nal in nal_units {
            self.stats.total_nals += 1;

            if nal.nal_type == RPU_NAL_TYPE {
                self.stats.rpu_nals += 1;

                match Rpu::parse(&nal.data) {
                    Ok(rpu) => {
                        self.rpus.push(rpu);
                        self.stats.parsed_rpus += 1;
                    }
                    Err(e) => {
                        self.stats.parse_errors += 1;
                        tracing::warn!("Failed to parse RPU: {}", e);
                    }
                }
            }

            // Store NAL for layer separation if needed
            if self.mode == ExtractionMode::SeparateLayers {
                self.nal_buffer.push(nal);
            }
        }

        Ok(())
    }

    /// Get extracted RPUs.
    pub fn rpus(&self) -> &[Rpu] {
        &self.rpus
    }

    /// Take extracted RPUs (consumes them).
    pub fn take_rpus(&mut self) -> Vec<Rpu> {
        std::mem::take(&mut self.rpus)
    }

    /// Get extraction statistics.
    pub fn stats(&self) -> &ExtractionStats {
        &self.stats
    }

    /// Separate base layer and enhancement layer NAL units.
    pub fn separate_layers(&self) -> Result<LayerSeparation> {
        if self.mode != ExtractionMode::SeparateLayers {
            return Err(DolbyError::ExtractionError {
                message: "Extractor not configured for layer separation".to_string(),
            });
        }

        let mut base_layer = Vec::new();
        let mut enhancement_layer = Vec::new();
        let mut rpu_nals = Vec::new();

        for nal in &self.nal_buffer {
            match nal.layer_id {
                0 => {
                    if nal.nal_type == RPU_NAL_TYPE {
                        rpu_nals.push(nal.clone());
                    } else {
                        base_layer.push(nal.clone());
                    }
                }
                1 => {
                    enhancement_layer.push(nal.clone());
                }
                _ => {
                    // Unknown layer, put in base
                    base_layer.push(nal.clone());
                }
            }
        }

        Ok(LayerSeparation {
            base_layer,
            enhancement_layer,
            rpu_nals,
        })
    }

    fn find_nal_units(&self, data: &[u8]) -> Result<Vec<NalUnit>> {
        let mut units = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            // Find start code
            let (_start_code_len, nal_start) = if pos + 4 <= data.len()
                && data[pos..pos + 4] == NAL_START_CODE_4
            {
                (4, pos + 4)
            } else if pos + 3 <= data.len() && data[pos..pos + 3] == NAL_START_CODE_3 {
                (3, pos + 3)
            } else {
                pos += 1;
                continue;
            };

            // Find next start code to determine NAL end
            let mut nal_end = data.len();
            for i in nal_start..data.len().saturating_sub(2) {
                if data[i..i + 3] == NAL_START_CODE_3 {
                    nal_end = i;
                    break;
                }
            }

            if nal_start < nal_end {
                let nal_data = &data[nal_start..nal_end];
                if let Some(nal) = self.parse_nal_header(nal_data) {
                    units.push(nal);
                }
            }

            pos = nal_end;
        }

        Ok(units)
    }

    fn parse_nal_header(&self, data: &[u8]) -> Option<NalUnit> {
        if data.len() < 2 {
            return None;
        }

        // HEVC NAL header: forbidden_zero_bit(1) + nal_unit_type(6) + nuh_layer_id(6) + nuh_temporal_id_plus1(3)
        let forbidden = (data[0] >> 7) & 1;
        if forbidden != 0 {
            return None;
        }

        let nal_type = (data[0] >> 1) & 0x3F;
        let layer_id = ((data[0] & 1) << 5) | ((data[1] >> 3) & 0x1F);
        let temporal_id = (data[1] & 0x07).saturating_sub(1);

        Some(NalUnit {
            nal_type,
            layer_id,
            temporal_id,
            data: data.to_vec(),
        })
    }

    /// Reset the extractor state.
    pub fn reset(&mut self) {
        self.nal_buffer.clear();
        self.rpus.clear();
        self.stats = ExtractionStats::default();
    }
}

/// Extraction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMode {
    /// Extract only RPU data.
    RpuOnly,
    /// Separate base and enhancement layers.
    SeparateLayers,
}

/// NAL unit representation.
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// NAL unit type.
    pub nal_type: u8,
    /// Layer ID.
    pub layer_id: u8,
    /// Temporal ID.
    pub temporal_id: u8,
    /// Raw NAL data (including header).
    pub data: Vec<u8>,
}

impl NalUnit {
    /// Check if this is an RPU NAL.
    pub fn is_rpu(&self) -> bool {
        self.nal_type == RPU_NAL_TYPE
    }

    /// Check if this is a VCL NAL (video coded layer).
    pub fn is_vcl(&self) -> bool {
        self.nal_type <= 31
    }

    /// Check if this is from the base layer.
    pub fn is_base_layer(&self) -> bool {
        self.layer_id == 0
    }

    /// Check if this is from the enhancement layer.
    pub fn is_enhancement_layer(&self) -> bool {
        self.layer_id == 1
    }

    /// Get the NAL payload (without header).
    pub fn payload(&self) -> &[u8] {
        if self.data.len() > 2 {
            &self.data[2..]
        } else {
            &[]
        }
    }
}

/// Layer separation result.
#[derive(Debug)]
pub struct LayerSeparation {
    /// Base layer NAL units.
    pub base_layer: Vec<NalUnit>,
    /// Enhancement layer NAL units.
    pub enhancement_layer: Vec<NalUnit>,
    /// RPU NAL units.
    pub rpu_nals: Vec<NalUnit>,
}

impl LayerSeparation {
    /// Check if this has an enhancement layer.
    pub fn has_enhancement_layer(&self) -> bool {
        !self.enhancement_layer.is_empty()
    }

    /// Get the number of RPUs.
    pub fn rpu_count(&self) -> usize {
        self.rpu_nals.len()
    }

    /// Reconstruct base layer stream.
    pub fn reconstruct_base_layer(&self) -> Vec<u8> {
        self.reconstruct_layer(&self.base_layer)
    }

    /// Reconstruct enhancement layer stream.
    pub fn reconstruct_enhancement_layer(&self) -> Vec<u8> {
        self.reconstruct_layer(&self.enhancement_layer)
    }

    fn reconstruct_layer(&self, nals: &[NalUnit]) -> Vec<u8> {
        let mut result = Vec::new();

        for nal in nals {
            result.extend_from_slice(&NAL_START_CODE_4);
            result.extend_from_slice(&nal.data);
        }

        result
    }
}

/// Extraction statistics.
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total NAL units processed.
    pub total_nals: usize,
    /// RPU NAL units found.
    pub rpu_nals: usize,
    /// Successfully parsed RPUs.
    pub parsed_rpus: usize,
    /// Parse errors.
    pub parse_errors: usize,
}

impl ExtractionStats {
    /// Get parse success rate.
    pub fn success_rate(&self) -> f64 {
        if self.rpu_nals == 0 {
            100.0
        } else {
            (self.parsed_rpus as f64 / self.rpu_nals as f64) * 100.0
        }
    }
}

/// RPU injector for adding RPU data to HEVC streams.
#[derive(Debug)]
pub struct RpuInjector {
    /// Injection mode.
    mode: InjectionMode,
    /// RPU queue.
    rpu_queue: Vec<Rpu>,
    /// Current RPU index.
    current_index: usize,
}

impl RpuInjector {
    /// Create a new RPU injector.
    pub fn new(mode: InjectionMode) -> Self {
        RpuInjector {
            mode,
            rpu_queue: Vec::new(),
            current_index: 0,
        }
    }

    /// Add RPUs to inject.
    pub fn add_rpus(&mut self, rpus: Vec<Rpu>) {
        self.rpu_queue.extend(rpus);
    }

    /// Inject RPU after a specific NAL unit.
    pub fn inject_after_nal(&mut self, stream: &[u8], after_nal_type: u8) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(stream.len() * 2);
        let mut pos = 0;

        while pos < stream.len() {
            // Find start code
            let (_start_code_len, nal_start) = if pos + 4 <= stream.len()
                && stream[pos..pos + 4] == NAL_START_CODE_4
            {
                (4, pos + 4)
            } else if pos + 3 <= stream.len() && stream[pos..pos + 3] == NAL_START_CODE_3 {
                (3, pos + 3)
            } else {
                result.push(stream[pos]);
                pos += 1;
                continue;
            };

            // Find NAL end
            let mut nal_end = stream.len();
            for i in nal_start..stream.len().saturating_sub(2) {
                if stream[i..i + 3] == NAL_START_CODE_3 {
                    nal_end = i;
                    break;
                }
            }

            // Copy original NAL
            result.extend_from_slice(&stream[pos..nal_end]);

            // Check if we should inject RPU after this NAL
            if nal_start < stream.len() {
                let nal_type = (stream[nal_start] >> 1) & 0x3F;

                if nal_type == after_nal_type && self.current_index < self.rpu_queue.len() {
                    // Inject RPU
                    if let Ok(rpu_data) = self.rpu_queue[self.current_index].serialize() {
                        result.extend_from_slice(&NAL_START_CODE_4);
                        result.push((RPU_NAL_TYPE << 1) | 1); // NAL header byte 1
                        result.push(0x01); // NAL header byte 2 (layer_id=0, temporal_id=1)
                        result.extend_from_slice(&rpu_data);
                    }
                    self.current_index += 1;
                }
            }

            pos = nal_end;
        }

        Ok(result)
    }

    /// Inject RPU at specific frame positions.
    pub fn inject_at_frames(&mut self, stream: &[u8], frame_positions: &[usize]) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(stream.len() * 2);
        let mut frame_count = 0;
        let mut pos = 0;
        let mut position_index = 0;

        while pos < stream.len() {
            // Find start code
            let (_start_code_len, nal_start) = if pos + 4 <= stream.len()
                && stream[pos..pos + 4] == NAL_START_CODE_4
            {
                (4, pos + 4)
            } else if pos + 3 <= stream.len() && stream[pos..pos + 3] == NAL_START_CODE_3 {
                (3, pos + 3)
            } else {
                result.push(stream[pos]);
                pos += 1;
                continue;
            };

            // Find NAL end
            let mut nal_end = stream.len();
            for i in nal_start..stream.len().saturating_sub(2) {
                if stream[i..i + 3] == NAL_START_CODE_3 {
                    nal_end = i;
                    break;
                }
            }

            // Check if this is a VCL NAL (indicates new frame)
            if nal_start < stream.len() {
                let nal_type = (stream[nal_start] >> 1) & 0x3F;
                let is_vcl = nal_type <= 31;

                if is_vcl {
                    // Check if we should inject at this frame
                    if position_index < frame_positions.len()
                        && frame_count == frame_positions[position_index]
                        && self.current_index < self.rpu_queue.len()
                    {
                        // Inject RPU before this frame
                        if let Ok(rpu_data) = self.rpu_queue[self.current_index].serialize() {
                            result.extend_from_slice(&NAL_START_CODE_4);
                            result.push((RPU_NAL_TYPE << 1) | 1);
                            result.push(0x01);
                            result.extend_from_slice(&rpu_data);
                        }
                        self.current_index += 1;
                        position_index += 1;
                    }
                    frame_count += 1;
                }
            }

            // Copy original NAL
            result.extend_from_slice(&stream[pos..nal_end]);
            pos = nal_end;
        }

        Ok(result)
    }

    /// Get remaining RPUs not yet injected.
    pub fn remaining_rpus(&self) -> &[Rpu] {
        &self.rpu_queue[self.current_index..]
    }

    /// Reset injector state.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

/// Injection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionMode {
    /// Inject after specific NAL type.
    AfterNal,
    /// Inject at frame boundaries.
    AtFrames,
}

/// Extract RPU from a single HEVC access unit.
pub fn extract_rpu_from_au(access_unit: &[u8]) -> Result<Option<Rpu>> {
    let mut extractor = DolbyVisionExtractor::rpu_only();
    extractor.process(access_unit)?;

    let rpus = extractor.take_rpus();
    Ok(rpus.into_iter().next())
}

/// Extract all RPUs from HEVC stream.
pub fn extract_all_rpus(stream: &[u8]) -> Result<Vec<Rpu>> {
    let mut extractor = DolbyVisionExtractor::rpu_only();
    extractor.process(stream)?;
    Ok(extractor.take_rpus())
}

/// Check if stream contains Dolby Vision.
pub fn has_dolby_vision(stream: &[u8]) -> bool {
    // Quick scan for RPU NAL type
    for i in 0..stream.len().saturating_sub(4) {
        if stream[i..i + 3] == NAL_START_CODE_3 {
            let nal_start = i + 3;
            if nal_start < stream.len() {
                let nal_type = (stream[nal_start] >> 1) & 0x3F;
                if nal_type == RPU_NAL_TYPE {
                    return true;
                }
            }
        }
    }
    false
}

/// Detect Dolby Vision profile from stream.
pub fn detect_profile(stream: &[u8]) -> Result<Option<DolbyVisionProfile>> {
    let rpus = extract_all_rpus(stream)?;

    if let Some(rpu) = rpus.first() {
        return Ok(rpu.profile());
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_parsing() {
        let extractor = DolbyVisionExtractor::rpu_only();

        // Test HEVC NAL header parsing
        let data = [
            0x28, 0x01, // NAL type 20 (IDR_W_RADL), layer 0, tid 1
            0xAB, 0xCD, // Payload
        ];

        let nal = extractor.parse_nal_header(&data);
        assert!(nal.is_some());

        let nal = nal.unwrap();
        assert_eq!(nal.nal_type, 20);
        assert_eq!(nal.layer_id, 0);
        assert!(nal.is_vcl());
        assert!(nal.is_base_layer());
    }

    #[test]
    fn test_nal_unit_types() {
        let rpu_nal = NalUnit {
            nal_type: RPU_NAL_TYPE,
            layer_id: 0,
            temporal_id: 0,
            data: vec![],
        };
        assert!(rpu_nal.is_rpu());
        assert!(!rpu_nal.is_vcl());

        let vcl_nal = NalUnit {
            nal_type: 1, // TRAIL_R
            layer_id: 0,
            temporal_id: 0,
            data: vec![],
        };
        assert!(!vcl_nal.is_rpu());
        assert!(vcl_nal.is_vcl());
    }

    #[test]
    fn test_extraction_stats() {
        let stats = ExtractionStats {
            total_nals: 100,
            rpu_nals: 24,
            parsed_rpus: 24,
            parse_errors: 0,
        };

        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_layer_separation() {
        let mut extractor = DolbyVisionExtractor::layer_separation();

        // Create test stream with start codes
        let mut stream = Vec::new();
        stream.extend_from_slice(&NAL_START_CODE_4);
        stream.extend_from_slice(&[0x28, 0x01, 0xAB]); // Base layer NAL

        extractor.process(&stream).unwrap();
        let separation = extractor.separate_layers().unwrap();

        assert!(!separation.base_layer.is_empty());
    }

    #[test]
    fn test_has_dolby_vision() {
        // Stream without DV
        let stream = [0x00, 0x00, 0x01, 0x28, 0x01, 0xAB];
        assert!(!has_dolby_vision(&stream));

        // Stream with DV (NAL type 62)
        let stream = [0x00, 0x00, 0x01, 0x7C, 0x01, 0xAB]; // 0x7C = 62 << 1
        assert!(has_dolby_vision(&stream));
    }

    #[test]
    fn test_rpu_injector() {
        let mut injector = RpuInjector::new(InjectionMode::AfterNal);

        // Add some RPUs
        let rpu = crate::rpu::create_rpu_from_l1(
            crate::metadata::L1Metadata::default(),
            DolbyVisionProfile::Profile8,
        );
        injector.add_rpus(vec![rpu]);

        assert_eq!(injector.remaining_rpus().len(), 1);
    }

    #[test]
    fn test_extractor_reset() {
        let mut extractor = DolbyVisionExtractor::rpu_only();
        extractor.stats.total_nals = 100;
        extractor.rpus.push(crate::rpu::create_rpu_from_l1(
            crate::metadata::L1Metadata::default(),
            DolbyVisionProfile::Profile8,
        ));

        extractor.reset();

        assert_eq!(extractor.stats.total_nals, 0);
        assert!(extractor.rpus.is_empty());
    }
}
