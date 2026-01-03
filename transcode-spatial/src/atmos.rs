//! Dolby Atmos support.
//!
//! This module provides Dolby Atmos audio processing including:
//! - ADM (Audio Definition Model) parsing
//! - Object positions and trajectories
//! - Bed representation
//! - E-AC-3 JOC (Joint Object Coding) metadata
//! - Atmos to 5.1/7.1 downmix

use crate::channels::{ChannelLayout, ChannelPosition, StandardLayout};
use crate::error::{AtmosError, Result, SpatialError};
use crate::objectaudio::{AudioBed, AudioObject, ObjectAudioScene, ObjectTrajectory, Position3D};
use std::collections::HashMap;

/// Dolby Atmos content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AtmosContentType {
    /// Music content.
    #[default]
    Music,
    /// Dialog content.
    Dialog,
    /// Effects content.
    Effects,
    /// Ambient/background content.
    Ambient,
    /// Full mix.
    FullMix,
}

/// Dolby Atmos program configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AtmosConfig {
    /// Maximum number of audio objects.
    pub max_objects: u32,
    /// Maximum number of bed channels.
    pub max_beds: u32,
    /// Content type.
    pub content_type: AtmosContentType,
    /// Target loudness in LKFS.
    pub target_loudness: f32,
    /// Dynamic range control profile.
    pub drc_profile: DrcProfile,
    /// Dialog normalization value in dB.
    pub dialog_norm_db: f32,
}

impl Default for AtmosConfig {
    fn default() -> Self {
        Self {
            max_objects: 128,
            max_beds: 10,
            content_type: AtmosContentType::FullMix,
            target_loudness: -24.0,
            drc_profile: DrcProfile::FilmStandard,
            dialog_norm_db: -31.0,
        }
    }
}

/// Dynamic range control profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DrcProfile {
    /// No DRC.
    None,
    /// Film standard.
    #[default]
    FilmStandard,
    /// Film light.
    FilmLight,
    /// Music standard.
    MusicStandard,
    /// Music light.
    MusicLight,
    /// Speech.
    Speech,
}

/// ADM (Audio Definition Model) document.
#[derive(Debug, Clone, Default)]
pub struct AdmDocument {
    /// Audio programs.
    pub programs: Vec<AdmAudioProgram>,
    /// Audio contents.
    pub contents: Vec<AdmAudioContent>,
    /// Audio objects.
    pub objects: Vec<AdmAudioObject>,
    /// Audio pack formats.
    pub pack_formats: Vec<AdmAudioPackFormat>,
    /// Audio channel formats.
    pub channel_formats: Vec<AdmAudioChannelFormat>,
    /// Audio track UIDs.
    pub track_uids: Vec<AdmAudioTrackUid>,
}

impl AdmDocument {
    /// Create a new empty ADM document.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse ADM from XML string.
    pub fn from_xml(xml: &str) -> Result<Self> {
        // Simplified XML parsing - in production would use proper XML parser
        let mut doc = Self::new();

        // Look for audioObject elements
        let mut in_object = false;
        let mut current_object = AdmAudioObject::default();

        for line in xml.lines() {
            let trimmed = line.trim();

            if trimmed.contains("<audioObject") {
                in_object = true;
                current_object = AdmAudioObject::default();

                // Extract ID if present
                if let Some(id_start) = trimmed.find("audioObjectID=\"") {
                    let rest = &trimmed[id_start + 15..];
                    if let Some(id_end) = rest.find('"') {
                        current_object.id = rest[..id_end].to_string();
                    }
                }

                // Check for name on the same line
                if trimmed.contains("<audioObjectName>") {
                    if let Some(name) = extract_element_content(trimmed, "audioObjectName") {
                        current_object.name = name;
                    }
                }

                // Check if object ends on the same line
                if trimmed.contains("</audioObject>") {
                    doc.objects.push(current_object.clone());
                    in_object = false;
                }
            } else if trimmed.contains("</audioObject>") && in_object {
                doc.objects.push(current_object.clone());
                in_object = false;
            } else if in_object {
                if trimmed.contains("<audioObjectName>") {
                    if let Some(name) = extract_element_content(trimmed, "audioObjectName") {
                        current_object.name = name;
                    }
                }
            }
        }

        Ok(doc)
    }

    /// Get object by ID.
    pub fn object(&self, id: &str) -> Option<&AdmAudioObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Convert to ObjectAudioScene.
    pub fn to_scene(&self) -> ObjectAudioScene {
        let mut scene = ObjectAudioScene::new(self.objects.len() as u32);

        for (idx, adm_obj) in self.objects.iter().enumerate() {
            let position = Position3D::new(
                adm_obj.position.x,
                adm_obj.position.y,
                adm_obj.position.z,
            );

            let object = AudioObject::named(idx as u32 + 1, &adm_obj.name, position);
            let _ = scene.add_object(object);
        }

        scene
    }
}

/// Extract content from a simple XML element.
fn extract_element_content(line: &str, element: &str) -> Option<String> {
    let start_tag = format!("<{}>", element);
    let end_tag = format!("</{}>", element);

    if let Some(start) = line.find(&start_tag) {
        let content_start = start + start_tag.len();
        if let Some(end) = line.find(&end_tag) {
            return Some(line[content_start..end].to_string());
        }
    }
    None
}

/// ADM Audio Program.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioProgram {
    /// Program ID.
    pub id: String,
    /// Program name.
    pub name: String,
    /// Program language.
    pub language: Option<String>,
    /// Loudness metadata.
    pub loudness: Option<LoudnessMetadata>,
}

/// ADM Audio Content.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioContent {
    /// Content ID.
    pub id: String,
    /// Content name.
    pub name: String,
    /// Content type.
    pub content_type: AtmosContentType,
}

/// ADM Audio Object.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioObject {
    /// Object ID.
    pub id: String,
    /// Object name.
    pub name: String,
    /// Start time in samples.
    pub start: u64,
    /// Duration in samples.
    pub duration: u64,
    /// Object position.
    pub position: AdmPosition,
    /// Object gain.
    pub gain: f32,
    /// Is dialog.
    pub is_dialog: bool,
    /// Object importance (0-10).
    pub importance: u32,
}

/// ADM Position.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdmPosition {
    /// X coordinate (-1 to 1).
    pub x: f32,
    /// Y coordinate (-1 to 1).
    pub y: f32,
    /// Z coordinate (-1 to 1).
    pub z: f32,
}

impl AdmPosition {
    /// Convert to Position3D.
    pub fn to_position_3d(&self) -> Position3D {
        Position3D::new(self.x, self.y, self.z)
    }
}

/// ADM Audio Pack Format.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioPackFormat {
    /// Pack format ID.
    pub id: String,
    /// Pack format name.
    pub name: String,
    /// Type definition.
    pub type_definition: String,
}

/// ADM Audio Channel Format.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioChannelFormat {
    /// Channel format ID.
    pub id: String,
    /// Channel format name.
    pub name: String,
    /// Block formats (position automation).
    pub blocks: Vec<AdmBlockFormat>,
}

/// ADM Block Format (position at a point in time).
#[derive(Debug, Clone, Default)]
pub struct AdmBlockFormat {
    /// Block ID.
    pub id: String,
    /// Start time in nanoseconds.
    pub start_time_ns: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Position.
    pub position: AdmPosition,
    /// Gain.
    pub gain: f32,
}

/// ADM Audio Track UID.
#[derive(Debug, Clone, Default)]
pub struct AdmAudioTrackUid {
    /// Track UID.
    pub id: String,
    /// Sample rate.
    pub sample_rate: u32,
    /// Bit depth.
    pub bit_depth: u32,
}

/// Loudness metadata.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoudnessMetadata {
    /// Integrated loudness in LKFS.
    pub integrated_loudness: f32,
    /// Loudness range in LU.
    pub loudness_range: f32,
    /// Maximum true peak in dBTP.
    pub max_true_peak: f32,
    /// Maximum momentary loudness in LKFS.
    pub max_momentary: f32,
    /// Maximum short-term loudness in LKFS.
    pub max_short_term: f32,
    /// Dialog loudness in LKFS.
    pub dialog_loudness: Option<f32>,
}

/// E-AC-3 JOC (Joint Object Coding) metadata.
#[derive(Debug, Clone)]
pub struct JocMetadata {
    /// Number of objects.
    pub num_objects: u32,
    /// Object positions.
    pub object_positions: Vec<JocObjectPosition>,
    /// Bed configuration.
    pub bed_config: JocBedConfig,
    /// Dynamic metadata present.
    pub dynamic_metadata: bool,
}

impl Default for JocMetadata {
    fn default() -> Self {
        Self {
            num_objects: 0,
            object_positions: Vec::new(),
            bed_config: JocBedConfig::default(),
            dynamic_metadata: false,
        }
    }
}

/// JOC object position.
#[derive(Debug, Clone, Copy, Default)]
pub struct JocObjectPosition {
    /// Object index.
    pub index: u32,
    /// Azimuth in degrees.
    pub azimuth: f32,
    /// Elevation in degrees.
    pub elevation: f32,
    /// Distance (0-1).
    pub distance: f32,
    /// Gain.
    pub gain: f32,
    /// Size/spread.
    pub size: f32,
}

impl JocObjectPosition {
    /// Convert to Position3D.
    pub fn to_position_3d(&self) -> Position3D {
        Position3D::from_spherical(self.azimuth, self.elevation, self.distance)
    }
}

/// JOC bed configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct JocBedConfig {
    /// Bed channel configuration (0 = none, 1 = 2.0, 2 = 5.1, 3 = 7.1, etc.).
    pub config: u8,
    /// Bed channel count.
    pub channel_count: u32,
    /// Bed start channel.
    pub start_channel: u32,
}

impl JocBedConfig {
    /// Get the standard layout for this bed config.
    pub fn to_standard_layout(&self) -> Option<StandardLayout> {
        match self.config {
            0 => None,
            1 => Some(StandardLayout::Stereo),
            2 => Some(StandardLayout::Surround51),
            3 => Some(StandardLayout::Surround71),
            4 => Some(StandardLayout::Atmos714),
            _ => None,
        }
    }
}

/// Atmos renderer for downmixing and output.
#[derive(Debug)]
pub struct AtmosRenderer {
    /// Configuration.
    config: AtmosConfig,
    /// Output layout.
    output_layout: StandardLayout,
    /// Downmix coefficients.
    downmix_coeffs: DownmixCoefficients,
}

/// Downmix coefficients for Atmos to stereo/5.1/7.1.
#[derive(Debug, Clone)]
struct DownmixCoefficients {
    /// Height to base layer mixing.
    height_to_base: f32,
    /// Surround to front mixing.
    surround_to_front: f32,
    /// Center to L/R mixing.
    center_to_lr: f32,
    /// LFE mixing level.
    lfe_level: f32,
}

impl Default for DownmixCoefficients {
    fn default() -> Self {
        Self {
            height_to_base: 0.707, // -3dB
            surround_to_front: 0.707,
            center_to_lr: 0.707,
            lfe_level: 0.0, // Often muted in downmix
        }
    }
}

impl AtmosRenderer {
    /// Create a new Atmos renderer.
    pub fn new(config: AtmosConfig, output_layout: StandardLayout) -> Self {
        Self {
            config,
            output_layout,
            downmix_coeffs: DownmixCoefficients::default(),
        }
    }

    /// Create renderer for 7.1 output.
    pub fn to_71(config: AtmosConfig) -> Self {
        Self::new(config, StandardLayout::Surround71)
    }

    /// Create renderer for 5.1 output.
    pub fn to_51(config: AtmosConfig) -> Self {
        Self::new(config, StandardLayout::Surround51)
    }

    /// Create renderer for stereo output.
    pub fn to_stereo(config: AtmosConfig) -> Self {
        Self::new(config, StandardLayout::Stereo)
    }

    /// Set custom downmix coefficients.
    pub fn set_downmix_coefficients(
        &mut self,
        height_to_base: f32,
        surround_to_front: f32,
        center_to_lr: f32,
        lfe_level: f32,
    ) {
        self.downmix_coeffs = DownmixCoefficients {
            height_to_base,
            surround_to_front,
            center_to_lr,
            lfe_level,
        };
    }

    /// Get the output channel count.
    pub fn output_channel_count(&self) -> usize {
        self.output_layout.channel_count() as usize
    }

    /// Render Atmos scene to output.
    pub fn render(&self, scene: &ObjectAudioScene, num_samples: usize) -> Vec<Vec<f32>> {
        let output_channels = self.output_channel_count();
        let mut output = vec![vec![0.0; num_samples]; output_channels];

        // First render beds
        for bed in scene.beds() {
            self.render_bed(bed, &mut output, num_samples);
        }

        // Then render objects
        for object in scene.objects() {
            self.render_object(object, &mut output, num_samples);
        }

        // Apply dialog normalization
        self.apply_dialog_norm(&mut output);

        output
    }

    /// Render a bed to output.
    fn render_bed(&self, bed: &AudioBed, output: &mut [Vec<f32>], num_samples: usize) {
        let output_positions = self.output_layout.positions();

        for channel in &bed.channels {
            let gain = channel.gain_linear();
            let samples = channel.samples();

            // Find matching output channel or downmix
            if let Some(out_idx) = output_positions.iter().position(|p| *p == channel.position) {
                // Direct mapping
                for (i, &s) in samples.iter().enumerate().take(num_samples) {
                    output[out_idx][i] += s * gain;
                }
            } else {
                // Need to downmix
                self.downmix_channel(&channel.position, samples, gain, output, num_samples);
            }
        }
    }

    /// Downmix a single channel to output.
    fn downmix_channel(
        &self,
        position: &ChannelPosition,
        samples: &[f32],
        gain: f32,
        output: &mut [Vec<f32>],
        num_samples: usize,
    ) {
        let output_positions = self.output_layout.positions();

        match position {
            // Height channels to base layer
            ChannelPosition::TopFrontLeft => {
                if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::FrontLeft) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                }
            }
            ChannelPosition::TopFrontRight => {
                if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::FrontRight) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                }
            }
            ChannelPosition::TopBackLeft => {
                if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::BackLeft) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                } else if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::SideLeft) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                }
            }
            ChannelPosition::TopBackRight => {
                if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::BackRight) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                } else if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::SideRight) {
                    let coef = gain * self.downmix_coeffs.height_to_base;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[idx][i] += s * coef;
                    }
                }
            }
            // Side/back to stereo
            ChannelPosition::SideLeft | ChannelPosition::BackLeft => {
                if self.output_layout == StandardLayout::Stereo {
                    let coef = gain * self.downmix_coeffs.surround_to_front;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[0][i] += s * coef; // Left
                    }
                }
            }
            ChannelPosition::SideRight | ChannelPosition::BackRight => {
                if self.output_layout == StandardLayout::Stereo {
                    let coef = gain * self.downmix_coeffs.surround_to_front;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[1][i] += s * coef; // Right
                    }
                }
            }
            // Center to L/R
            ChannelPosition::FrontCenter => {
                if self.output_layout == StandardLayout::Stereo {
                    let coef = gain * self.downmix_coeffs.center_to_lr;
                    for (i, &s) in samples.iter().enumerate().take(num_samples) {
                        output[0][i] += s * coef; // Left
                        output[1][i] += s * coef; // Right
                    }
                }
            }
            // LFE handling
            ChannelPosition::Lfe => {
                if self.downmix_coeffs.lfe_level > 0.0 {
                    // Find LFE in output or mix to L/R
                    if let Some(idx) = output_positions.iter().position(|p| *p == ChannelPosition::Lfe) {
                        let coef = gain * self.downmix_coeffs.lfe_level;
                        for (i, &s) in samples.iter().enumerate().take(num_samples) {
                            output[idx][i] += s * coef;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Render an object to output.
    fn render_object(&self, object: &AudioObject, output: &mut [Vec<f32>], num_samples: usize) {
        let (azimuth, elevation, distance) = object.position.to_spherical();
        let gain = object.metadata.gain_linear();

        // Calculate panning gains for output layout
        let gains = self.calculate_object_gains(azimuth, elevation, distance);

        // Apply to output
        for (ch_idx, &ch_gain) in gains.iter().enumerate() {
            for (i, &s) in object.samples().iter().enumerate().take(num_samples) {
                output[ch_idx][i] += s * gain * ch_gain;
            }
        }
    }

    /// Calculate object gains for output layout.
    fn calculate_object_gains(&self, azimuth: f32, elevation: f32, distance: f32) -> Vec<f32> {
        let output_positions = self.output_layout.positions();
        let num_channels = output_positions.len();
        let mut gains = vec![0.0; num_channels];

        // Simple VBAP-like panning
        for (idx, pos) in output_positions.iter().enumerate() {
            if pos.is_lfe() {
                continue;
            }

            let (spk_az, spk_el) = pos.position_degrees();

            // Angular distance
            let az_diff = angular_distance(azimuth, spk_az);
            let el_diff = (elevation - spk_el).abs();
            let angular_dist = (az_diff * az_diff + el_diff * el_diff).sqrt();

            // Gain based on proximity (simple model)
            gains[idx] = if angular_dist < 30.0 {
                1.0 - angular_dist / 60.0
            } else if angular_dist < 90.0 {
                0.5 - (angular_dist - 30.0) / 120.0
            } else {
                0.0
            };

            gains[idx] = gains[idx].max(0.0);
        }

        // Normalize
        let sum_sq: f32 = gains.iter().map(|g| g * g).sum();
        if sum_sq > 0.0 {
            let norm = 1.0 / sum_sq.sqrt();
            for g in &mut gains {
                *g *= norm;
            }
        }

        // Apply distance attenuation
        let dist_atten = if distance > 0.0 { 1.0 / (1.0 + distance) } else { 1.0 };
        for g in &mut gains {
            *g *= dist_atten;
        }

        gains
    }

    /// Apply dialog normalization.
    fn apply_dialog_norm(&self, output: &mut [Vec<f32>]) {
        let norm_linear = 10.0_f32.powf(self.config.dialog_norm_db / 20.0);

        for channel in output.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= norm_linear;
            }
        }
    }
}

/// Calculate angular distance between two azimuth angles.
fn angular_distance(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atmos_config() {
        let config = AtmosConfig::default();
        assert_eq!(config.max_objects, 128);
        assert_eq!(config.target_loudness, -24.0);
    }

    #[test]
    fn test_adm_document() {
        // Use single-line format for the simple parser
        let xml = r#"<audioFormatExtended>
<audioObject audioObjectID="AO_1001"><audioObjectName>Dialog</audioObjectName></audioObject>
<audioObject audioObjectID="AO_1002"><audioObjectName>Effects</audioObjectName></audioObject>
</audioFormatExtended>"#;

        let doc = AdmDocument::from_xml(xml).unwrap();
        assert_eq!(doc.objects.len(), 2);
        assert_eq!(doc.objects[0].name, "Dialog");
        assert_eq!(doc.objects[1].name, "Effects");
    }

    #[test]
    fn test_joc_metadata() {
        let mut joc = JocMetadata::default();
        joc.num_objects = 4;
        joc.object_positions.push(JocObjectPosition {
            index: 0,
            azimuth: 30.0,
            elevation: 0.0,
            distance: 1.0,
            gain: 1.0,
            size: 0.0,
        });

        assert_eq!(joc.object_positions.len(), 1);
        let pos = joc.object_positions[0].to_position_3d();
        assert!(pos.distance() > 0.0);
    }

    #[test]
    fn test_joc_bed_config() {
        let bed_51 = JocBedConfig {
            config: 2,
            channel_count: 6,
            start_channel: 0,
        };

        assert_eq!(bed_51.to_standard_layout(), Some(StandardLayout::Surround51));
    }

    #[test]
    fn test_atmos_renderer() {
        let config = AtmosConfig::default();
        let renderer = AtmosRenderer::to_51(config);

        assert_eq!(renderer.output_channel_count(), 6);
    }

    #[test]
    fn test_atmos_render_scene() {
        let config = AtmosConfig::default();
        let renderer = AtmosRenderer::to_stereo(config);

        let mut scene = ObjectAudioScene::new(10);
        let mut object = AudioObject::new(1, Position3D::new(0.0, 1.0, 0.0));
        object.set_samples(vec![1.0; 100]);
        scene.add_object(object).unwrap();

        let output = renderer.render(&scene, 100);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 100);
    }

    #[test]
    fn test_loudness_metadata() {
        let loudness = LoudnessMetadata {
            integrated_loudness: -24.0,
            loudness_range: 10.0,
            max_true_peak: -1.0,
            max_momentary: -20.0,
            max_short_term: -22.0,
            dialog_loudness: Some(-24.0),
        };

        assert_eq!(loudness.integrated_loudness, -24.0);
        assert!(loudness.dialog_loudness.is_some());
    }

    #[test]
    fn test_adm_position() {
        let adm_pos = AdmPosition { x: 0.5, y: 0.5, z: 0.0 };
        let pos_3d = adm_pos.to_position_3d();

        assert!((pos_3d.x - 0.5).abs() < 0.001);
        assert!((pos_3d.y - 0.5).abs() < 0.001);
    }
}
