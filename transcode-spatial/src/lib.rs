//! # Transcode Spatial
//!
//! Spatial audio support for the Transcode codec library.
//!
//! This crate provides comprehensive spatial audio processing capabilities including:
//!
//! - **Channel Layouts**: Standard layouts (stereo, 5.1, 7.1, 7.1.4 Atmos), channel positions,
//!   custom layouts, and channel order conversion (WAV, SMPTE, Dolby)
//!
//! - **Ambisonics**: First-order ambisonics (FOA) with 4 channels (WXYZ), higher-order
//!   ambisonics (HOA) with ACN/SN3D ordering, ambisonics to binaural conversion, and
//!   ambisonics to speaker array decoding
//!
//! - **Binaural Rendering**: HRTF (Head-Related Transfer Function) application,
//!   convolution-based processing, head tracking support interface, near-field
//!   compensation, and crossfeed for headphones
//!
//! - **Object-Based Audio**: Audio objects with 3D position, object metadata (size,
//!   diffusion), bed channels + objects model, and object rendering to speaker layouts
//!
//! - **Dolby Atmos**: ADM (Audio Definition Model) parsing, object positions and
//!   trajectories, bed representation, E-AC-3 JOC metadata, and Atmos to 5.1/7.1 downmix
//!
//! - **Spatial Rendering**: VBAP (Vector Base Amplitude Panning), distance attenuation,
//!   room acoustics simulation, and multiple output format support
//!
//! - **Downmixing**: 7.1.4 to 7.1 to 5.1 to stereo conversion, LFE handling, dialog
//!   normalization, and configurable downmix coefficients
//!
//! ## Example
//!
//! ```rust
//! use transcode_spatial::{
//!     SpatialConfig, RenderMode,
//!     channels::{StandardLayout, ChannelLayout},
//!     objectaudio::{AudioObject, Position3D, ObjectAudioScene},
//!     renderer::SpatialRenderer,
//! };
//!
//! // Create a spatial audio configuration
//! let config = SpatialConfig::default()
//!     .with_output_layout(StandardLayout::Surround51)
//!     .with_render_mode(RenderMode::Speakers);
//!
//! // Create a scene with audio objects
//! let mut scene = ObjectAudioScene::new(16);
//! let object = AudioObject::new(1, Position3D::new(0.5, 0.8, 0.0));
//! scene.add_object(object).unwrap();
//!
//! // Create a renderer
//! let renderer = SpatialRenderer::new(StandardLayout::Surround51, 48000).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::approx_constant)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::collapsible_if)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod ambisonics;
pub mod atmos;
pub mod binaural;
pub mod channels;
pub mod downmix;
pub mod error;
pub mod objectaudio;
pub mod renderer;

// Re-exports for convenience
pub use ambisonics::{AmbisonicsDecoder, AmbisonicsEncoder, AmbisonicsFormat, BFormat};
pub use atmos::{AdmDocument, AtmosConfig, AtmosRenderer, JocMetadata};
pub use binaural::{BinauralRenderer, CrossfeedProcessor, HeadTracking, HrtfDatabase};
pub use channels::{ChannelLayout, ChannelOrder, ChannelPosition, StandardLayout};
pub use downmix::{CascadedDownmixer, DownmixMatrix, DownmixPreset, Downmixer, LfeMode};
pub use error::{Result, SpatialError};
pub use objectaudio::{AudioBed, AudioObject, ObjectAudioScene, ObjectRenderer, Position3D};
pub use renderer::{DistanceAttenuator, DistanceModel, RoomSimulator, SpatialRenderer, VbapRenderer};

/// Spatial audio rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderMode {
    /// Render to speaker layout.
    #[default]
    Speakers,
    /// Render to binaural (headphones).
    Binaural,
    /// Render to ambisonics.
    Ambisonics,
}

/// Main configuration for spatial audio processing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialConfig {
    /// Output layout for speaker rendering.
    pub output_layout: StandardLayout,
    /// Rendering mode.
    pub render_mode: RenderMode,
    /// Sample rate.
    pub sample_rate: u32,
    /// Enable room simulation.
    pub room_enabled: bool,
    /// Room reverb time (RT60) in seconds.
    pub room_reverb_time: f32,
    /// Enable head tracking for binaural.
    pub head_tracking_enabled: bool,
    /// Ambisonics order for ambisonics output.
    pub ambisonics_order: u32,
    /// Distance attenuation model.
    pub distance_model: DistanceModel,
    /// Reference distance for attenuation.
    pub reference_distance: f32,
    /// Maximum distance for attenuation.
    pub max_distance: f32,
    /// Downmix preset.
    pub downmix_preset: DownmixPreset,
    /// LFE handling mode.
    pub lfe_mode: LfeMode,
    /// Dialog normalization in dB.
    pub dialog_norm_db: f32,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            output_layout: StandardLayout::Stereo,
            render_mode: RenderMode::Speakers,
            sample_rate: 48000,
            room_enabled: false,
            room_reverb_time: 0.5,
            head_tracking_enabled: false,
            ambisonics_order: 1,
            distance_model: DistanceModel::Inverse,
            reference_distance: 1.0,
            max_distance: 100.0,
            downmix_preset: DownmixPreset::ItuR,
            lfe_mode: LfeMode::Preserve,
            dialog_norm_db: 0.0,
        }
    }
}

impl SpatialConfig {
    /// Create a new spatial configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output layout.
    pub fn with_output_layout(mut self, layout: StandardLayout) -> Self {
        self.output_layout = layout;
        self
    }

    /// Set the render mode.
    pub fn with_render_mode(mut self, mode: RenderMode) -> Self {
        self.render_mode = mode;
        self
    }

    /// Set the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Enable room simulation.
    pub fn with_room(mut self, reverb_time: f32) -> Self {
        self.room_enabled = true;
        self.room_reverb_time = reverb_time;
        self
    }

    /// Enable head tracking.
    pub fn with_head_tracking(mut self, enabled: bool) -> Self {
        self.head_tracking_enabled = enabled;
        self
    }

    /// Set ambisonics order.
    pub fn with_ambisonics_order(mut self, order: u32) -> Self {
        self.ambisonics_order = order;
        self
    }

    /// Set distance attenuation parameters.
    pub fn with_distance_attenuation(
        mut self,
        model: DistanceModel,
        reference: f32,
        max: f32,
    ) -> Self {
        self.distance_model = model;
        self.reference_distance = reference;
        self.max_distance = max;
        self
    }

    /// Set downmix parameters.
    pub fn with_downmix(mut self, preset: DownmixPreset, lfe_mode: LfeMode) -> Self {
        self.downmix_preset = preset;
        self.lfe_mode = lfe_mode;
        self
    }

    /// Set dialog normalization.
    pub fn with_dialog_norm(mut self, db: f32) -> Self {
        self.dialog_norm_db = db;
        self
    }

    /// Create a spatial renderer from this configuration.
    pub fn create_renderer(&self) -> Result<SpatialRenderer> {
        let mut renderer = SpatialRenderer::new(self.output_layout, self.sample_rate)?;

        // Configure distance attenuation
        let attenuator = DistanceAttenuator::new(self.distance_model)
            .with_reference_distance(self.reference_distance)
            .with_max_distance(self.max_distance);
        renderer.set_distance_model(attenuator);

        // Configure room simulation
        if self.room_enabled {
            let room = RoomSimulator::new(
                self.sample_rate,
                (8.0, 10.0, 3.5), // Medium room
                self.room_reverb_time,
            );
            renderer.enable_room(room);
        }

        Ok(renderer)
    }

    /// Create a binaural renderer from this configuration.
    pub fn create_binaural_renderer(&self) -> BinauralRenderer {
        let mut renderer = BinauralRenderer::with_synthetic_hrtf(self.sample_rate, 256);
        renderer.enable_head_tracking(self.head_tracking_enabled);
        renderer
    }

    /// Create a downmixer from this configuration.
    pub fn create_downmixer(&self, source: StandardLayout) -> Downmixer {
        let mut matrix = DownmixMatrix::with_preset(source, self.output_layout, self.downmix_preset);
        matrix.set_lfe_mode(self.lfe_mode);

        let mut downmixer = Downmixer::new(matrix);
        if self.dialog_norm_db != 0.0 {
            downmixer.set_dialog_norm(self.dialog_norm_db, true);
        }
        downmixer
    }

    /// Create an ambisonics encoder from this configuration.
    pub fn create_ambisonics_encoder(&self) -> AmbisonicsEncoder {
        let format = AmbisonicsFormat::ambix(self.ambisonics_order);
        AmbisonicsEncoder::new(format)
    }

    /// Create an ambisonics decoder from this configuration.
    pub fn create_ambisonics_decoder(&self) -> Result<AmbisonicsDecoder> {
        let format = AmbisonicsFormat::ambix(self.ambisonics_order);
        let layout = ChannelLayout::from_standard(self.output_layout);
        AmbisonicsDecoder::new(format, layout)
    }
}

/// Presets for common spatial audio configurations.
pub mod presets {
    use super::*;

    /// Home theater 5.1 configuration.
    pub fn home_theater_51() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround51)
            .with_render_mode(RenderMode::Speakers)
            .with_room(0.4)
            .with_distance_attenuation(DistanceModel::Inverse, 1.0, 20.0)
    }

    /// Home theater 7.1 configuration.
    pub fn home_theater_71() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround71)
            .with_render_mode(RenderMode::Speakers)
            .with_room(0.5)
            .with_distance_attenuation(DistanceModel::Inverse, 1.0, 20.0)
    }

    /// Dolby Atmos 7.1.4 configuration.
    pub fn atmos_714() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Atmos714)
            .with_render_mode(RenderMode::Speakers)
            .with_room(0.6)
            .with_distance_attenuation(DistanceModel::Inverse, 1.0, 30.0)
    }

    /// Headphone binaural configuration.
    pub fn headphones() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Stereo)
            .with_render_mode(RenderMode::Binaural)
            .with_head_tracking(true)
            .with_room(0.3)
    }

    /// VR/AR binaural configuration with head tracking.
    pub fn vr_headphones() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Stereo)
            .with_render_mode(RenderMode::Binaural)
            .with_head_tracking(true)
            .with_distance_attenuation(DistanceModel::InverseSquare, 0.5, 50.0)
    }

    /// Ambisonics production configuration.
    pub fn ambisonics_production() -> SpatialConfig {
        SpatialConfig::new()
            .with_render_mode(RenderMode::Ambisonics)
            .with_ambisonics_order(3)
            .with_sample_rate(48000)
    }

    /// Stereo music playback.
    pub fn stereo_music() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Stereo)
            .with_render_mode(RenderMode::Speakers)
    }

    /// Game audio 7.1 configuration.
    pub fn game_audio() -> SpatialConfig {
        SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround71)
            .with_render_mode(RenderMode::Speakers)
            .with_room(0.2)
            .with_distance_attenuation(DistanceModel::InverseSquare, 1.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_config_default() {
        let config = SpatialConfig::default();
        assert_eq!(config.output_layout, StandardLayout::Stereo);
        assert_eq!(config.render_mode, RenderMode::Speakers);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_spatial_config_builder() {
        let config = SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround51)
            .with_render_mode(RenderMode::Binaural)
            .with_sample_rate(44100)
            .with_room(0.8)
            .with_head_tracking(true);

        assert_eq!(config.output_layout, StandardLayout::Surround51);
        assert_eq!(config.render_mode, RenderMode::Binaural);
        assert_eq!(config.sample_rate, 44100);
        assert!(config.room_enabled);
        assert_eq!(config.room_reverb_time, 0.8);
        assert!(config.head_tracking_enabled);
    }

    #[test]
    fn test_create_renderer() {
        let config = SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround51)
            .with_room(0.5);

        let renderer = config.create_renderer();
        assert!(renderer.is_ok());
        assert_eq!(renderer.unwrap().channel_count(), 6);
    }

    #[test]
    fn test_create_binaural_renderer() {
        let config = SpatialConfig::new().with_head_tracking(true);

        let renderer = config.create_binaural_renderer();
        assert_eq!(renderer.sample_rate(), 48000);
    }

    #[test]
    fn test_create_downmixer() {
        let config = SpatialConfig::new()
            .with_output_layout(StandardLayout::Stereo)
            .with_downmix(DownmixPreset::ItuR, LfeMode::MixToFront);

        let downmixer = config.create_downmixer(StandardLayout::Surround51);
        assert_eq!(downmixer.output_channel_count(), 2);
    }

    #[test]
    fn test_presets() {
        let home = presets::home_theater_51();
        assert_eq!(home.output_layout, StandardLayout::Surround51);
        assert!(home.room_enabled);

        let atmos = presets::atmos_714();
        assert_eq!(atmos.output_layout, StandardLayout::Atmos714);

        let headphones = presets::headphones();
        assert_eq!(headphones.render_mode, RenderMode::Binaural);
        assert!(headphones.head_tracking_enabled);

        let vr = presets::vr_headphones();
        assert!(vr.head_tracking_enabled);
        assert_eq!(vr.distance_model, DistanceModel::InverseSquare);
    }

    #[test]
    fn test_ambisonics_config() {
        let config = SpatialConfig::new()
            .with_render_mode(RenderMode::Ambisonics)
            .with_ambisonics_order(2);

        let encoder = config.create_ambisonics_encoder();
        assert_eq!(encoder.format().order, 2);
        assert_eq!(encoder.format().channel_count(), 9);
    }

    #[test]
    fn test_integration_object_to_speakers() {
        // Create scene
        let mut scene = ObjectAudioScene::new(16);
        let mut object = AudioObject::new(1, Position3D::new(0.5, 0.8, 0.0));
        object.set_samples(vec![1.0; 100]);
        scene.add_object(object).unwrap();

        // Create renderer
        let config = SpatialConfig::new()
            .with_output_layout(StandardLayout::Surround51);
        let renderer = ObjectRenderer::new(ChannelLayout::from_standard(StandardLayout::Surround51));

        // Render
        let output = renderer.render_scene(&scene, 100);
        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_integration_downmix_chain() {
        // 7.1.4 to stereo
        let cascade = CascadedDownmixer::atmos_to_stereo();

        // Create 7.1.4 input
        let input: Vec<Vec<f32>> = (0..12).map(|_| vec![0.5; 100]).collect();

        let output = cascade.process(&input).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 100);
    }
}
