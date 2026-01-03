//! Downmix matrices for spatial audio.
//!
//! This module provides comprehensive downmix support including:
//! - 7.1.4 to 7.1 to 5.1 to stereo conversion
//! - LFE handling
//! - Dialog normalization
//! - Configurable downmix coefficients

use crate::channels::{ChannelLayout, ChannelPosition, StandardLayout};
use crate::error::{DownmixError, Result, SpatialError};
use std::collections::HashMap;

/// Downmix coefficient presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DownmixPreset {
    /// ITU-R BS.775 standard coefficients.
    #[default]
    ItuR,
    /// Dolby Pro Logic II compatible.
    DolbyProLogic2,
    /// Dolby Surround compatible.
    DolbySurround,
    /// ATSC A/52 (AC-3) standard.
    AtscA52,
    /// Film standard.
    Film,
    /// Custom coefficients.
    Custom,
}

/// Downmix coefficients for a specific channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelCoefficient {
    /// Target channel position.
    pub target: ChannelPosition,
    /// Linear coefficient (gain).
    pub coefficient: f32,
}

impl ChannelCoefficient {
    /// Create a new channel coefficient.
    pub fn new(target: ChannelPosition, coefficient: f32) -> Self {
        Self { target, coefficient }
    }

    /// Create from dB value.
    pub fn from_db(target: ChannelPosition, db: f32) -> Self {
        Self {
            target,
            coefficient: 10.0_f32.powf(db / 20.0),
        }
    }

    /// Get coefficient in dB.
    pub fn to_db(&self) -> f32 {
        if self.coefficient > 0.0 {
            20.0 * self.coefficient.log10()
        } else {
            -120.0
        }
    }
}

/// Downmix matrix definition.
#[derive(Debug, Clone)]
pub struct DownmixMatrix {
    /// Source layout.
    source_layout: StandardLayout,
    /// Target layout.
    target_layout: StandardLayout,
    /// Coefficients: source channel -> list of (target, coefficient).
    coefficients: HashMap<ChannelPosition, Vec<ChannelCoefficient>>,
    /// LFE handling mode.
    lfe_mode: LfeMode,
    /// Normalization to prevent clipping.
    normalize: bool,
}

/// LFE (Low Frequency Effects) handling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LfeMode {
    /// Discard LFE in downmix.
    #[default]
    Discard,
    /// Mix LFE to front L/R at -10dB.
    MixToFront,
    /// Mix LFE to all channels equally.
    MixToAll,
    /// Mix LFE to front L/R at specified dB.
    MixToFrontWithLevel(i8),
    /// Keep LFE if target has LFE channel.
    Preserve,
}

impl DownmixMatrix {
    /// Create a new downmix matrix.
    pub fn new(source: StandardLayout, target: StandardLayout) -> Self {
        Self {
            source_layout: source,
            target_layout: target,
            coefficients: HashMap::new(),
            lfe_mode: LfeMode::Discard,
            normalize: true,
        }
    }

    /// Create with preset coefficients.
    pub fn with_preset(source: StandardLayout, target: StandardLayout, preset: DownmixPreset) -> Self {
        let mut matrix = Self::new(source, target);
        matrix.apply_preset(preset);
        matrix
    }

    /// Apply a preset.
    pub fn apply_preset(&mut self, preset: DownmixPreset) {
        match preset {
            DownmixPreset::ItuR => self.apply_itu_coefficients(),
            DownmixPreset::DolbyProLogic2 => self.apply_dolby_pl2_coefficients(),
            DownmixPreset::DolbySurround => self.apply_dolby_surround_coefficients(),
            DownmixPreset::AtscA52 => self.apply_atsc_coefficients(),
            DownmixPreset::Film => self.apply_film_coefficients(),
            DownmixPreset::Custom => {} // Use manually set coefficients
        }
    }

    /// Apply ITU-R BS.775 coefficients.
    fn apply_itu_coefficients(&mut self) {
        self.coefficients.clear();

        // Common coefficients
        let center_coef = 0.707; // -3dB
        let surround_coef = 0.707; // -3dB

        match (self.source_layout, self.target_layout) {
            (StandardLayout::Surround51, StandardLayout::Stereo) => {
                // 5.1 to Stereo (Lo/Ro)
                // L' = L + 0.707*C + 0.707*Ls
                // R' = R + 0.707*C + 0.707*Rs
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontLeft, center_coef);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontRight, center_coef);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontLeft, surround_coef);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontRight, surround_coef);
            }
            (StandardLayout::Surround71, StandardLayout::Surround51) => {
                // 7.1 to 5.1
                // Mix side channels to back channels
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontCenter, 1.0);
                self.set_coefficient(ChannelPosition::Lfe, ChannelPosition::Lfe, 1.0);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::BackLeft, 1.0);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::BackRight, 1.0);
                self.set_coefficient(ChannelPosition::SideLeft, ChannelPosition::BackLeft, 0.707);
                self.set_coefficient(ChannelPosition::SideRight, ChannelPosition::BackRight, 0.707);
            }
            (StandardLayout::Surround71, StandardLayout::Stereo) => {
                // 7.1 to Stereo
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontLeft, center_coef);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontRight, center_coef);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontLeft, surround_coef);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontRight, surround_coef);
                self.set_coefficient(ChannelPosition::SideLeft, ChannelPosition::FrontLeft, surround_coef);
                self.set_coefficient(ChannelPosition::SideRight, ChannelPosition::FrontRight, surround_coef);
            }
            (StandardLayout::Atmos714, StandardLayout::Surround71) => {
                // 7.1.4 to 7.1 (fold down height channels)
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontCenter, 1.0);
                self.set_coefficient(ChannelPosition::Lfe, ChannelPosition::Lfe, 1.0);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::BackLeft, 1.0);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::BackRight, 1.0);
                self.set_coefficient(ChannelPosition::SideLeft, ChannelPosition::SideLeft, 1.0);
                self.set_coefficient(ChannelPosition::SideRight, ChannelPosition::SideRight, 1.0);
                // Height to base layer
                self.set_coefficient(ChannelPosition::TopFrontLeft, ChannelPosition::FrontLeft, 0.707);
                self.set_coefficient(ChannelPosition::TopFrontRight, ChannelPosition::FrontRight, 0.707);
                self.set_coefficient(ChannelPosition::TopBackLeft, ChannelPosition::BackLeft, 0.707);
                self.set_coefficient(ChannelPosition::TopBackRight, ChannelPosition::BackRight, 0.707);
            }
            (StandardLayout::Atmos714, StandardLayout::Surround51) => {
                // 7.1.4 to 5.1
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontCenter, 1.0);
                self.set_coefficient(ChannelPosition::Lfe, ChannelPosition::Lfe, 1.0);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::BackLeft, 1.0);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::BackRight, 1.0);
                self.set_coefficient(ChannelPosition::SideLeft, ChannelPosition::BackLeft, 0.707);
                self.set_coefficient(ChannelPosition::SideRight, ChannelPosition::BackRight, 0.707);
                // Height channels
                self.set_coefficient(ChannelPosition::TopFrontLeft, ChannelPosition::FrontLeft, 0.5);
                self.set_coefficient(ChannelPosition::TopFrontRight, ChannelPosition::FrontRight, 0.5);
                self.set_coefficient(ChannelPosition::TopBackLeft, ChannelPosition::BackLeft, 0.5);
                self.set_coefficient(ChannelPosition::TopBackRight, ChannelPosition::BackRight, 0.5);
            }
            _ => {
                // Default: pass-through matching channels
                self.apply_passthrough();
            }
        }
    }

    /// Apply Dolby Pro Logic II coefficients.
    fn apply_dolby_pl2_coefficients(&mut self) {
        self.coefficients.clear();

        // Different encoding for Pro Logic II matrix
        let center_coef = 0.707;
        let surround_coef = 0.707;

        match (self.source_layout, self.target_layout) {
            (StandardLayout::Surround51, StandardLayout::Stereo) => {
                // Lt = L + 0.707*C - 0.707*Ls - 0.707*Rs
                // Rt = R + 0.707*C + 0.707*Ls + 0.707*Rs
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontLeft, center_coef);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontRight, center_coef);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontLeft, -surround_coef);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontRight, surround_coef);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontLeft, -surround_coef);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontRight, surround_coef);
            }
            _ => self.apply_itu_coefficients(),
        }
    }

    /// Apply Dolby Surround coefficients.
    fn apply_dolby_surround_coefficients(&mut self) {
        self.coefficients.clear();

        match (self.source_layout, self.target_layout) {
            (StandardLayout::Surround51, StandardLayout::Stereo) => {
                // Lt = L + 0.707*C + 0.707*S
                // Rt = R + 0.707*C - 0.707*S
                // (Where S is surround sum)
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontLeft, 0.707);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontRight, 0.707);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontLeft, 0.5);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontRight, -0.5);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontLeft, 0.5);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontRight, -0.5);
            }
            _ => self.apply_itu_coefficients(),
        }
    }

    /// Apply ATSC A/52 coefficients.
    fn apply_atsc_coefficients(&mut self) {
        // ATSC uses same as ITU but with specific clev/slev values
        self.apply_itu_coefficients();
    }

    /// Apply film standard coefficients.
    fn apply_film_coefficients(&mut self) {
        self.coefficients.clear();

        // Film typically uses -3dB for center and surround
        let center_coef = 0.707;
        let surround_coef = 0.707;

        match (self.source_layout, self.target_layout) {
            (StandardLayout::Surround51, StandardLayout::Stereo) => {
                self.set_coefficient(ChannelPosition::FrontLeft, ChannelPosition::FrontLeft, 1.0);
                self.set_coefficient(ChannelPosition::FrontRight, ChannelPosition::FrontRight, 1.0);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontLeft, center_coef);
                self.set_coefficient(ChannelPosition::FrontCenter, ChannelPosition::FrontRight, center_coef);
                self.set_coefficient(ChannelPosition::BackLeft, ChannelPosition::FrontLeft, surround_coef);
                self.set_coefficient(ChannelPosition::BackRight, ChannelPosition::FrontRight, surround_coef);
            }
            _ => self.apply_itu_coefficients(),
        }
    }

    /// Apply passthrough (copy matching channels).
    fn apply_passthrough(&mut self) {
        let source_positions = self.source_layout.positions();
        let target_positions = self.target_layout.positions();

        for pos in source_positions {
            if target_positions.contains(&pos) {
                self.set_coefficient(pos, pos, 1.0);
            }
        }
    }

    /// Set a coefficient.
    pub fn set_coefficient(&mut self, source: ChannelPosition, target: ChannelPosition, coef: f32) {
        let entry = self.coefficients.entry(source).or_insert_with(Vec::new);

        // Update existing or add new
        if let Some(existing) = entry.iter_mut().find(|c| c.target == target) {
            existing.coefficient = coef;
        } else {
            entry.push(ChannelCoefficient::new(target, coef));
        }
    }

    /// Get coefficients for a source channel.
    pub fn get_coefficients(&self, source: ChannelPosition) -> Option<&[ChannelCoefficient]> {
        self.coefficients.get(&source).map(|v| v.as_slice())
    }

    /// Set LFE mode.
    pub fn set_lfe_mode(&mut self, mode: LfeMode) {
        self.lfe_mode = mode;
    }

    /// Enable/disable normalization.
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    /// Get the source layout.
    pub fn source_layout(&self) -> StandardLayout {
        self.source_layout
    }

    /// Get the target layout.
    pub fn target_layout(&self) -> StandardLayout {
        self.target_layout
    }
}

/// Downmixer processor.
#[derive(Debug)]
pub struct Downmixer {
    /// Downmix matrix.
    matrix: DownmixMatrix,
    /// Dialog normalization level in dB.
    dialog_norm_db: f32,
    /// Apply dialog normalization.
    apply_dialog_norm: bool,
    /// Peak limiter enabled.
    limiter_enabled: bool,
    /// Limiter threshold in dB.
    limiter_threshold_db: f32,
}

impl Downmixer {
    /// Create a new downmixer.
    pub fn new(matrix: DownmixMatrix) -> Self {
        Self {
            matrix,
            dialog_norm_db: 0.0,
            apply_dialog_norm: false,
            limiter_enabled: true,
            limiter_threshold_db: -1.0,
        }
    }

    /// Create with preset.
    pub fn with_preset(source: StandardLayout, target: StandardLayout, preset: DownmixPreset) -> Self {
        let matrix = DownmixMatrix::with_preset(source, target, preset);
        Self::new(matrix)
    }

    /// Create 7.1.4 to 7.1 downmixer.
    pub fn atmos_to_71() -> Self {
        Self::with_preset(StandardLayout::Atmos714, StandardLayout::Surround71, DownmixPreset::ItuR)
    }

    /// Create 7.1 to 5.1 downmixer.
    pub fn surround_71_to_51() -> Self {
        Self::with_preset(StandardLayout::Surround71, StandardLayout::Surround51, DownmixPreset::ItuR)
    }

    /// Create 5.1 to stereo downmixer.
    pub fn surround_51_to_stereo() -> Self {
        Self::with_preset(StandardLayout::Surround51, StandardLayout::Stereo, DownmixPreset::ItuR)
    }

    /// Create 7.1 to stereo downmixer.
    pub fn surround_71_to_stereo() -> Self {
        Self::with_preset(StandardLayout::Surround71, StandardLayout::Stereo, DownmixPreset::ItuR)
    }

    /// Set dialog normalization.
    pub fn set_dialog_norm(&mut self, db: f32, enabled: bool) {
        self.dialog_norm_db = db;
        self.apply_dialog_norm = enabled;
    }

    /// Set limiter parameters.
    pub fn set_limiter(&mut self, enabled: bool, threshold_db: f32) {
        self.limiter_enabled = enabled;
        self.limiter_threshold_db = threshold_db;
    }

    /// Get the output channel count.
    pub fn output_channel_count(&self) -> usize {
        self.matrix.target_layout.channel_count() as usize
    }

    /// Get the input channel count.
    pub fn input_channel_count(&self) -> usize {
        self.matrix.source_layout.channel_count() as usize
    }

    /// Downmix audio.
    pub fn process(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let source_positions = self.matrix.source_layout.positions();
        let target_positions = self.matrix.target_layout.positions();

        if input.len() != source_positions.len() {
            return Err(SpatialError::Downmix(DownmixError::InvalidMatrix(format!(
                "Input has {} channels, expected {}",
                input.len(),
                source_positions.len()
            ))));
        }

        let num_samples = input.first().map(|v| v.len()).unwrap_or(0);
        let mut output = vec![vec![0.0; num_samples]; target_positions.len()];

        // Apply downmix matrix
        for (src_idx, src_pos) in source_positions.iter().enumerate() {
            if let Some(coeffs) = self.matrix.get_coefficients(*src_pos) {
                for coeff in coeffs {
                    if let Some(tgt_idx) = target_positions.iter().position(|p| *p == coeff.target) {
                        for (i, &sample) in input[src_idx].iter().enumerate() {
                            output[tgt_idx][i] += sample * coeff.coefficient;
                        }
                    }
                }
            }
        }

        // Handle LFE
        self.apply_lfe_mode(&input, &source_positions, &target_positions, &mut output);

        // Apply dialog normalization
        if self.apply_dialog_norm && self.dialog_norm_db != 0.0 {
            let norm_gain = 10.0_f32.powf(self.dialog_norm_db / 20.0);
            for channel in &mut output {
                for sample in channel {
                    *sample *= norm_gain;
                }
            }
        }

        // Apply limiter
        if self.limiter_enabled {
            let threshold = 10.0_f32.powf(self.limiter_threshold_db / 20.0);
            for channel in &mut output {
                for sample in channel {
                    if *sample > threshold {
                        *sample = threshold;
                    } else if *sample < -threshold {
                        *sample = -threshold;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Apply LFE mode.
    fn apply_lfe_mode(
        &self,
        input: &[Vec<f32>],
        source_positions: &[ChannelPosition],
        target_positions: &[ChannelPosition],
        output: &mut [Vec<f32>],
    ) {
        // Find LFE in source
        let lfe_idx = source_positions.iter().position(|p| p.is_lfe());
        if lfe_idx.is_none() {
            return;
        }
        let lfe_idx = lfe_idx.unwrap();
        let lfe_samples = &input[lfe_idx];

        match self.matrix.lfe_mode {
            LfeMode::Discard => {
                // Do nothing, LFE is discarded
            }
            LfeMode::MixToFront => {
                // Mix LFE to front L/R at -10dB
                let lfe_gain = 0.316; // -10dB
                if let Some(l_idx) = target_positions.iter().position(|p| *p == ChannelPosition::FrontLeft) {
                    for (i, &sample) in lfe_samples.iter().enumerate() {
                        if i < output[l_idx].len() {
                            output[l_idx][i] += sample * lfe_gain;
                        }
                    }
                }
                if let Some(r_idx) = target_positions.iter().position(|p| *p == ChannelPosition::FrontRight) {
                    for (i, &sample) in lfe_samples.iter().enumerate() {
                        if i < output[r_idx].len() {
                            output[r_idx][i] += sample * lfe_gain;
                        }
                    }
                }
            }
            LfeMode::MixToFrontWithLevel(db) => {
                let lfe_gain = 10.0_f32.powf(db as f32 / 20.0);
                if let Some(l_idx) = target_positions.iter().position(|p| *p == ChannelPosition::FrontLeft) {
                    for (i, &sample) in lfe_samples.iter().enumerate() {
                        if i < output[l_idx].len() {
                            output[l_idx][i] += sample * lfe_gain;
                        }
                    }
                }
                if let Some(r_idx) = target_positions.iter().position(|p| *p == ChannelPosition::FrontRight) {
                    for (i, &sample) in lfe_samples.iter().enumerate() {
                        if i < output[r_idx].len() {
                            output[r_idx][i] += sample * lfe_gain;
                        }
                    }
                }
            }
            LfeMode::MixToAll => {
                let num_non_lfe = target_positions.iter().filter(|p| !p.is_lfe()).count();
                let lfe_gain = 0.5 / (num_non_lfe as f32).sqrt();
                for (ch_idx, pos) in target_positions.iter().enumerate() {
                    if !pos.is_lfe() {
                        for (i, &sample) in lfe_samples.iter().enumerate() {
                            if i < output[ch_idx].len() {
                                output[ch_idx][i] += sample * lfe_gain;
                            }
                        }
                    }
                }
            }
            LfeMode::Preserve => {
                // Copy LFE to target if it has LFE
                if let Some(tgt_lfe_idx) = target_positions.iter().position(|p| p.is_lfe()) {
                    for (i, &sample) in lfe_samples.iter().enumerate() {
                        if i < output[tgt_lfe_idx].len() {
                            output[tgt_lfe_idx][i] = sample;
                        }
                    }
                }
            }
        }
    }
}

/// Cascaded downmixer for multi-step downmix.
#[derive(Debug)]
pub struct CascadedDownmixer {
    stages: Vec<Downmixer>,
}

impl CascadedDownmixer {
    /// Create a new cascaded downmixer.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Create 7.1.4 to stereo (via 7.1 and 5.1).
    pub fn atmos_to_stereo() -> Self {
        let mut cascade = Self::new();
        cascade.add_stage(Downmixer::atmos_to_71());
        cascade.add_stage(Downmixer::surround_71_to_51());
        cascade.add_stage(Downmixer::surround_51_to_stereo());
        cascade
    }

    /// Add a downmix stage.
    pub fn add_stage(&mut self, downmixer: Downmixer) {
        self.stages.push(downmixer);
    }

    /// Process through all stages.
    pub fn process(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut current = input.to_vec();

        for stage in &self.stages {
            current = stage.process(&current)?;
        }

        Ok(current)
    }

    /// Get the final output channel count.
    pub fn output_channel_count(&self) -> usize {
        self.stages.last().map(|s| s.output_channel_count()).unwrap_or(0)
    }
}

impl Default for CascadedDownmixer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_coefficient() {
        let coef = ChannelCoefficient::new(ChannelPosition::FrontLeft, 0.707);
        assert!((coef.to_db() - (-3.0)).abs() < 0.1);

        let coef_db = ChannelCoefficient::from_db(ChannelPosition::FrontLeft, -6.0);
        assert!((coef_db.coefficient - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_downmix_matrix_itu() {
        let matrix = DownmixMatrix::with_preset(
            StandardLayout::Surround51,
            StandardLayout::Stereo,
            DownmixPreset::ItuR,
        );

        // Check that center goes to both L and R
        let center_coeffs = matrix.get_coefficients(ChannelPosition::FrontCenter).unwrap();
        assert_eq!(center_coeffs.len(), 2);
    }

    #[test]
    fn test_downmixer_51_to_stereo() {
        let downmixer = Downmixer::surround_51_to_stereo();

        // Create 5.1 input (silence except center)
        let input: Vec<Vec<f32>> = vec![
            vec![0.0; 100], // FL
            vec![0.0; 100], // FR
            vec![1.0; 100], // FC - signal here
            vec![0.0; 100], // LFE
            vec![0.0; 100], // BL
            vec![0.0; 100], // BR
        ];

        let output = downmixer.process(&input).unwrap();
        assert_eq!(output.len(), 2);

        // Center should appear in both L and R at -3dB
        assert!(output[0][0] > 0.6);
        assert!(output[1][0] > 0.6);
    }

    #[test]
    fn test_lfe_modes() {
        let mut matrix = DownmixMatrix::with_preset(
            StandardLayout::Surround51,
            StandardLayout::Stereo,
            DownmixPreset::ItuR,
        );

        matrix.set_lfe_mode(LfeMode::MixToFront);
        let downmixer = Downmixer::new(matrix);

        // Create input with LFE signal
        let mut input: Vec<Vec<f32>> = vec![
            vec![0.0; 100], // FL
            vec![0.0; 100], // FR
            vec![0.0; 100], // FC
            vec![1.0; 100], // LFE - signal here
            vec![0.0; 100], // BL
            vec![0.0; 100], // BR
        ];

        let output = downmixer.process(&input).unwrap();

        // LFE should be mixed to front at -10dB (0.316)
        assert!(output[0][0] > 0.3);
        assert!(output[1][0] > 0.3);
    }

    #[test]
    fn test_atmos_to_71() {
        let downmixer = Downmixer::atmos_to_71();

        // 7.1.4 has 12 channels
        let input: Vec<Vec<f32>> = (0..12).map(|_| vec![0.5; 100]).collect();

        let output = downmixer.process(&input).unwrap();
        assert_eq!(output.len(), 8); // 7.1 output
    }

    #[test]
    fn test_cascaded_downmixer() {
        let cascade = CascadedDownmixer::atmos_to_stereo();

        // 7.1.4 input
        let input: Vec<Vec<f32>> = (0..12).map(|_| vec![0.5; 100]).collect();

        let output = cascade.process(&input).unwrap();
        assert_eq!(output.len(), 2); // Stereo output
    }

    #[test]
    fn test_dialog_normalization() {
        let mut downmixer = Downmixer::surround_51_to_stereo();
        downmixer.set_dialog_norm(-6.0, true);

        let input: Vec<Vec<f32>> = vec![
            vec![1.0; 100], // FL
            vec![1.0; 100], // FR
            vec![0.0; 100], // FC
            vec![0.0; 100], // LFE
            vec![0.0; 100], // BL
            vec![0.0; 100], // BR
        ];

        let output = downmixer.process(&input).unwrap();

        // Should be attenuated by -6dB (factor of 0.5)
        assert!(output[0][0] < 0.6);
        assert!(output[0][0] > 0.4);
    }

    #[test]
    fn test_limiter() {
        let mut downmixer = Downmixer::surround_51_to_stereo();
        downmixer.set_limiter(true, -3.0); // Limit at -3dB (~0.707)

        // Hot signal that would clip
        let input: Vec<Vec<f32>> = vec![
            vec![1.0; 100],  // FL
            vec![1.0; 100],  // FR
            vec![1.0; 100],  // FC (will add to L/R)
            vec![0.0; 100],  // LFE
            vec![1.0; 100],  // BL (will add to L)
            vec![1.0; 100],  // BR (will add to R)
        ];

        let output = downmixer.process(&input).unwrap();

        // Output should be limited to ~0.707
        assert!(output[0][0] <= 0.71);
        assert!(output[1][0] <= 0.71);
    }
}
