//! Spatial audio renderer.
//!
//! This module provides comprehensive spatial audio rendering including:
//! - VBAP (Vector Base Amplitude Panning)
//! - Distance attenuation
//! - Room acoustics simulation (basic reverb)
//! - Multiple output format support

use crate::channels::{ChannelLayout, ChannelPosition, StandardLayout};
use crate::error::{RendererError, Result, SpatialError};
use crate::objectaudio::Position3D;
use std::f32::consts::PI;

/// Speaker configuration for rendering.
#[derive(Debug, Clone)]
pub struct SpeakerConfig {
    /// Speaker position (azimuth, elevation) in degrees.
    pub position: (f32, f32),
    /// Speaker distance from listener in meters.
    pub distance: f32,
    /// Speaker gain adjustment in dB.
    pub gain_db: f32,
    /// Speaker delay in milliseconds.
    pub delay_ms: f32,
    /// Channel position.
    pub channel: ChannelPosition,
}

impl SpeakerConfig {
    /// Create a new speaker configuration.
    pub fn new(channel: ChannelPosition, azimuth: f32, elevation: f32) -> Self {
        Self {
            position: (azimuth, elevation),
            distance: 1.0,
            gain_db: 0.0,
            delay_ms: 0.0,
            channel,
        }
    }

    /// Create from channel position (uses default speaker angles).
    pub fn from_channel(channel: ChannelPosition) -> Self {
        let (azimuth, elevation) = channel.position_degrees();
        Self::new(channel, azimuth, elevation)
    }

    /// Get linear gain.
    pub fn gain_linear(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }

    /// Get 3D unit vector for this speaker.
    pub fn direction_vector(&self) -> (f32, f32, f32) {
        let az_rad = self.position.0.to_radians();
        let el_rad = self.position.1.to_radians();

        (
            az_rad.sin() * el_rad.cos(),
            az_rad.cos() * el_rad.cos(),
            el_rad.sin(),
        )
    }
}

/// VBAP (Vector Base Amplitude Panning) renderer.
#[derive(Debug)]
pub struct VbapRenderer {
    /// Speaker configuration.
    speakers: Vec<SpeakerConfig>,
    /// Precomputed speaker triplets for 3D VBAP.
    triplets: Vec<SpeakerTriplet>,
    /// Use 2D panning (ignore elevation).
    use_2d: bool,
}

/// Speaker triplet for 3D VBAP.
#[derive(Debug, Clone)]
struct SpeakerTriplet {
    /// Speaker indices.
    indices: [usize; 3],
    /// Inverse of the speaker matrix.
    inv_matrix: [[f32; 3]; 3],
}

impl VbapRenderer {
    /// Create a new VBAP renderer for a speaker layout.
    pub fn new(layout: &ChannelLayout) -> Result<Self> {
        let mut speakers = Vec::new();

        for position in layout.positions() {
            if !position.is_lfe() {
                speakers.push(SpeakerConfig::from_channel(*position));
            }
        }

        if speakers.len() < 2 {
            return Err(SpatialError::Renderer(RendererError::SpeakerConfig(
                "Need at least 2 speakers for VBAP".into(),
            )));
        }

        // Determine if we should use 2D or 3D panning
        let has_height = speakers.iter().any(|s| s.position.1.abs() > 10.0);

        let triplets = if has_height && speakers.len() >= 3 {
            Self::compute_triplets(&speakers)?
        } else {
            Vec::new()
        };

        Ok(Self {
            speakers,
            triplets,
            use_2d: !has_height,
        })
    }

    /// Create for a standard layout.
    pub fn for_layout(layout: StandardLayout) -> Result<Self> {
        Self::new(&ChannelLayout::from_standard(layout))
    }

    /// Compute speaker triplets for 3D VBAP.
    fn compute_triplets(speakers: &[SpeakerConfig]) -> Result<Vec<SpeakerTriplet>> {
        let mut triplets = Vec::new();
        let n = speakers.len();

        // For simplicity, create triplets from adjacent speakers
        // A full implementation would use Delaunay triangulation
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if let Some(triplet) = Self::create_triplet(speakers, i, j, k) {
                        triplets.push(triplet);
                    }
                }
            }
        }

        Ok(triplets)
    }

    /// Create a speaker triplet if valid.
    fn create_triplet(
        speakers: &[SpeakerConfig],
        i: usize,
        j: usize,
        k: usize,
    ) -> Option<SpeakerTriplet> {
        let v1 = speakers[i].direction_vector();
        let v2 = speakers[j].direction_vector();
        let v3 = speakers[k].direction_vector();

        // Build the speaker matrix
        let matrix = [
            [v1.0, v2.0, v3.0],
            [v1.1, v2.1, v3.1],
            [v1.2, v2.2, v3.2],
        ];

        // Compute determinant
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

        if det.abs() < 0.0001 {
            return None; // Degenerate triplet
        }

        // Compute inverse matrix
        let inv_det = 1.0 / det;
        let inv_matrix = [
            [
                inv_det * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]),
                inv_det * (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]),
                inv_det * (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]),
            ],
            [
                inv_det * (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]),
                inv_det * (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]),
                inv_det * (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]),
            ],
            [
                inv_det * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]),
                inv_det * (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]),
                inv_det * (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]),
            ],
        ];

        Some(SpeakerTriplet {
            indices: [i, j, k],
            inv_matrix,
        })
    }

    /// Get the number of output channels.
    pub fn channel_count(&self) -> usize {
        self.speakers.len()
    }

    /// Calculate panning gains for a source direction.
    pub fn calculate_gains(&self, azimuth: f32, elevation: f32) -> Vec<f32> {
        let mut gains = vec![0.0; self.speakers.len()];

        if self.use_2d {
            self.calculate_gains_2d(azimuth, &mut gains);
        } else {
            self.calculate_gains_3d(azimuth, elevation, &mut gains);
        }

        // Normalize for power preservation
        let sum_sq: f32 = gains.iter().map(|g| g * g).sum();
        if sum_sq > 0.0 {
            let norm = 1.0 / sum_sq.sqrt();
            for g in &mut gains {
                *g *= norm;
            }
        }

        gains
    }

    /// Calculate 2D panning gains (ignore elevation).
    fn calculate_gains_2d(&self, azimuth: f32, gains: &mut [f32]) {
        // Find the two speakers that bracket the source azimuth
        let n = self.speakers.len();

        // Sort speakers by azimuth
        let mut speaker_azimuths: Vec<(usize, f32)> = self
            .speakers
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.position.0))
            .collect();
        speaker_azimuths.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Find bracketing speakers
        let mut left_idx = 0;
        let mut right_idx = 0;

        for (i, (_, az)) in speaker_azimuths.iter().enumerate() {
            if *az <= azimuth {
                left_idx = i;
            }
            if *az >= azimuth && right_idx == 0 {
                right_idx = i;
            }
        }

        if right_idx == 0 {
            right_idx = speaker_azimuths.len() - 1;
        }

        let (left_speaker, left_az) = speaker_azimuths[left_idx];
        let (right_speaker, right_az) = speaker_azimuths[right_idx];

        if left_speaker == right_speaker {
            gains[left_speaker] = 1.0;
            return;
        }

        // Calculate blend factor
        let span = if right_az > left_az {
            right_az - left_az
        } else {
            360.0 - left_az + right_az
        };

        let pos_in_span = if azimuth >= left_az {
            azimuth - left_az
        } else {
            360.0 - left_az + azimuth
        };

        let blend = if span > 0.0 { pos_in_span / span } else { 0.5 };

        gains[left_speaker] = (1.0 - blend).sqrt();
        gains[right_speaker] = blend.sqrt();
    }

    /// Calculate 3D panning gains using VBAP.
    fn calculate_gains_3d(&self, azimuth: f32, elevation: f32, gains: &mut [f32]) {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        let source_vec = (
            az_rad.sin() * el_rad.cos(),
            az_rad.cos() * el_rad.cos(),
            el_rad.sin(),
        );

        // Find the best triplet
        let mut best_triplet: Option<&SpeakerTriplet> = None;
        let mut best_gains = [0.0f32; 3];

        for triplet in &self.triplets {
            // Calculate gains for this triplet
            let g = [
                triplet.inv_matrix[0][0] * source_vec.0
                    + triplet.inv_matrix[0][1] * source_vec.1
                    + triplet.inv_matrix[0][2] * source_vec.2,
                triplet.inv_matrix[1][0] * source_vec.0
                    + triplet.inv_matrix[1][1] * source_vec.1
                    + triplet.inv_matrix[1][2] * source_vec.2,
                triplet.inv_matrix[2][0] * source_vec.0
                    + triplet.inv_matrix[2][1] * source_vec.1
                    + triplet.inv_matrix[2][2] * source_vec.2,
            ];

            // Check if all gains are non-negative (source is in this triplet's cone)
            if g[0] >= -0.001 && g[1] >= -0.001 && g[2] >= -0.001 {
                best_triplet = Some(triplet);
                best_gains = g;
                break;
            }
        }

        if let Some(triplet) = best_triplet {
            // Normalize gains
            let sum = best_gains[0] + best_gains[1] + best_gains[2];
            if sum > 0.0 {
                for (i, &idx) in triplet.indices.iter().enumerate() {
                    gains[idx] = (best_gains[i] / sum).max(0.0).sqrt();
                }
            }
        } else {
            // Fallback: use nearest speaker
            self.calculate_gains_2d(azimuth, gains);
        }
    }

    /// Render a mono source to speaker outputs.
    pub fn render_source(
        &self,
        input: &[f32],
        azimuth: f32,
        elevation: f32,
        output: &mut [Vec<f32>],
    ) -> Result<()> {
        if output.len() != self.speakers.len() {
            return Err(SpatialError::Renderer(RendererError::BufferSizeMismatch {
                expected: self.speakers.len(),
                actual: output.len(),
            }));
        }

        let gains = self.calculate_gains(azimuth, elevation);

        for (ch_idx, (out_channel, &gain)) in output.iter_mut().zip(gains.iter()).enumerate() {
            let speaker_gain = self.speakers[ch_idx].gain_linear();
            for (i, &sample) in input.iter().enumerate() {
                if i < out_channel.len() {
                    out_channel[i] += sample * gain * speaker_gain;
                }
            }
        }

        Ok(())
    }
}

/// Distance attenuation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DistanceModel {
    /// No distance attenuation.
    None,
    /// Inverse distance (1/d).
    #[default]
    Inverse,
    /// Inverse distance squared (1/d^2).
    InverseSquare,
    /// Linear falloff.
    Linear,
    /// Exponential decay.
    Exponential,
}

/// Distance attenuation calculator.
#[derive(Debug, Clone)]
pub struct DistanceAttenuator {
    /// Attenuation model.
    model: DistanceModel,
    /// Reference distance (gain = 1.0 at this distance).
    reference_distance: f32,
    /// Maximum distance (beyond this, gain is constant).
    max_distance: f32,
    /// Rolloff factor.
    rolloff: f32,
}

impl Default for DistanceAttenuator {
    fn default() -> Self {
        Self {
            model: DistanceModel::Inverse,
            reference_distance: 1.0,
            max_distance: 100.0,
            rolloff: 1.0,
        }
    }
}

impl DistanceAttenuator {
    /// Create a new distance attenuator.
    pub fn new(model: DistanceModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set reference distance.
    pub fn with_reference_distance(mut self, distance: f32) -> Self {
        self.reference_distance = distance.max(0.001);
        self
    }

    /// Set maximum distance.
    pub fn with_max_distance(mut self, distance: f32) -> Self {
        self.max_distance = distance.max(self.reference_distance);
        self
    }

    /// Set rolloff factor.
    pub fn with_rolloff(mut self, rolloff: f32) -> Self {
        self.rolloff = rolloff.max(0.0);
        self
    }

    /// Calculate attenuation for a given distance.
    pub fn calculate(&self, distance: f32) -> f32 {
        let d = distance.max(self.reference_distance);
        let clamped_d = d.min(self.max_distance);

        match self.model {
            DistanceModel::None => 1.0,
            DistanceModel::Inverse => {
                self.reference_distance
                    / (self.reference_distance
                        + self.rolloff * (clamped_d - self.reference_distance))
            }
            DistanceModel::InverseSquare => {
                let ratio = self.reference_distance / clamped_d;
                ratio * ratio
            }
            DistanceModel::Linear => {
                1.0 - self.rolloff * (clamped_d - self.reference_distance)
                    / (self.max_distance - self.reference_distance)
            }
            DistanceModel::Exponential => {
                (-self.rolloff * (clamped_d - self.reference_distance)).exp()
            }
        }
        .max(0.0)
    }
}

/// Simple room acoustics simulator.
#[derive(Debug)]
pub struct RoomSimulator {
    /// Room dimensions (width, depth, height) in meters.
    dimensions: (f32, f32, f32),
    /// Wall absorption coefficient (0-1).
    absorption: f32,
    /// Reverb time (RT60) in seconds.
    reverb_time: f32,
    /// Sample rate.
    sample_rate: u32,
    /// Early reflections delay lines.
    early_delays: Vec<DelayLine>,
    /// Late reverb (simple FDN).
    late_reverb: FeedbackDelayNetwork,
    /// Dry/wet mix.
    wet_mix: f32,
}

/// Simple delay line.
#[derive(Debug, Clone)]
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
    gain: f32,
}

impl DelayLine {
    fn new(max_samples: usize, delay_samples: usize, gain: f32) -> Self {
        Self {
            buffer: vec![0.0; max_samples],
            write_pos: 0,
            delay_samples: delay_samples.min(max_samples - 1),
            gain,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
        let output = self.buffer[read_pos] * self.gain;
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

/// Simple feedback delay network for late reverb.
#[derive(Debug)]
struct FeedbackDelayNetwork {
    delays: Vec<DelayLine>,
    feedback: f32,
    damping: f32,
    prev_output: Vec<f32>,
}

impl FeedbackDelayNetwork {
    fn new(sample_rate: u32, reverb_time: f32, num_delays: usize) -> Self {
        // Prime number delays for good diffusion
        let prime_delays = [
            1423, 1637, 1801, 1949, 2111, 2269, 2423, 2593, 2749, 2909, 3067, 3229,
        ];

        let delays: Vec<DelayLine> = prime_delays
            .iter()
            .take(num_delays)
            .map(|&d| {
                let samples = (d as f32 * sample_rate as f32 / 48000.0) as usize;
                DelayLine::new(samples + 100, samples, 1.0)
            })
            .collect();

        // Calculate feedback from RT60
        let avg_delay = delays.iter().map(|d| d.delay_samples).sum::<usize>() as f32
            / delays.len() as f32;
        let loop_time = avg_delay / sample_rate as f32;
        let feedback = 0.001_f32.powf(loop_time / reverb_time);

        Self {
            delays,
            feedback: feedback.min(0.98),
            damping: 0.3,
            prev_output: vec![0.0; num_delays],
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let mut output = 0.0;
        let n = self.delays.len();

        // Householder feedback matrix (simplified)
        let feedback_sum: f32 = self.prev_output.iter().sum();
        let feedback_term = feedback_sum * 2.0 / n as f32;

        for i in 0..n {
            let delayed = self.delays[i].process(input + (feedback_term - self.prev_output[i]) * self.feedback);

            // Damping (simple lowpass)
            self.prev_output[i] = self.prev_output[i] * self.damping + delayed * (1.0 - self.damping);
            output += self.prev_output[i];
        }

        output / n as f32
    }

    fn reset(&mut self) {
        for delay in &mut self.delays {
            delay.reset();
        }
        self.prev_output.fill(0.0);
    }
}

impl RoomSimulator {
    /// Create a new room simulator.
    pub fn new(sample_rate: u32, dimensions: (f32, f32, f32), reverb_time: f32) -> Self {
        let speed_of_sound = 343.0;

        // Calculate early reflections from room dimensions
        let mut early_delays = Vec::new();

        // First order reflections (6 walls)
        let distances = [
            dimensions.0,     // Left wall
            dimensions.0,     // Right wall
            dimensions.1,     // Front wall
            dimensions.1,     // Back wall
            dimensions.2,     // Floor
            dimensions.2,     // Ceiling
        ];

        for (i, &dist) in distances.iter().enumerate() {
            let delay_samples = (dist * 2.0 / speed_of_sound * sample_rate as f32) as usize;
            let gain = 0.7_f32.powi((i + 1) as i32); // Decay with each reflection order
            early_delays.push(DelayLine::new(delay_samples + 100, delay_samples, gain));
        }

        let late_reverb = FeedbackDelayNetwork::new(sample_rate, reverb_time, 8);

        Self {
            dimensions,
            absorption: 0.3,
            reverb_time,
            sample_rate,
            early_delays,
            late_reverb,
            wet_mix: 0.3,
        }
    }

    /// Create a small room preset.
    pub fn small_room(sample_rate: u32) -> Self {
        Self::new(sample_rate, (4.0, 5.0, 2.5), 0.4)
    }

    /// Create a medium room preset.
    pub fn medium_room(sample_rate: u32) -> Self {
        Self::new(sample_rate, (8.0, 10.0, 3.5), 0.8)
    }

    /// Create a large hall preset.
    pub fn large_hall(sample_rate: u32) -> Self {
        Self::new(sample_rate, (20.0, 30.0, 12.0), 2.0)
    }

    /// Set wet/dry mix (0 = dry, 1 = wet).
    pub fn set_wet_mix(&mut self, mix: f32) {
        self.wet_mix = mix.clamp(0.0, 1.0);
    }

    /// Process a sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Early reflections
        let mut early = 0.0;
        for delay in &mut self.early_delays {
            early += delay.process(input);
        }
        early /= self.early_delays.len() as f32;

        // Late reverb
        let late = self.late_reverb.process(input + early * 0.5);

        // Mix
        let wet = early * 0.5 + late * 0.5;
        input * (1.0 - self.wet_mix) + wet * self.wet_mix
    }

    /// Process a buffer.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (in_sample, out_sample) in input.iter().zip(output.iter_mut()) {
            *out_sample = self.process_sample(*in_sample);
        }
    }

    /// Reset the simulator state.
    pub fn reset(&mut self) {
        for delay in &mut self.early_delays {
            delay.reset();
        }
        self.late_reverb.reset();
    }
}

/// Complete spatial renderer combining all components.
#[derive(Debug)]
pub struct SpatialRenderer {
    /// VBAP panner.
    panner: VbapRenderer,
    /// Distance attenuator.
    distance: DistanceAttenuator,
    /// Room simulator (optional).
    room: Option<RoomSimulator>,
    /// Output layout.
    layout: StandardLayout,
    /// Sample rate.
    sample_rate: u32,
}

impl SpatialRenderer {
    /// Create a new spatial renderer.
    pub fn new(layout: StandardLayout, sample_rate: u32) -> Result<Self> {
        let channel_layout = ChannelLayout::from_standard(layout);
        let panner = VbapRenderer::new(&channel_layout)?;

        Ok(Self {
            panner,
            distance: DistanceAttenuator::default(),
            room: None,
            layout,
            sample_rate,
        })
    }

    /// Enable room simulation.
    pub fn enable_room(&mut self, room: RoomSimulator) {
        self.room = Some(room);
    }

    /// Disable room simulation.
    pub fn disable_room(&mut self) {
        self.room = None;
    }

    /// Set distance attenuation model.
    pub fn set_distance_model(&mut self, attenuator: DistanceAttenuator) {
        self.distance = attenuator;
    }

    /// Get the output channel count.
    pub fn channel_count(&self) -> usize {
        self.layout.channel_count() as usize
    }

    /// Render a source to output.
    pub fn render(
        &mut self,
        input: &[f32],
        position: Position3D,
        output: &mut [Vec<f32>],
    ) -> Result<()> {
        let (azimuth, elevation, distance) = position.to_spherical();

        // Apply distance attenuation
        let dist_gain = self.distance.calculate(distance);

        // Apply room simulation if enabled
        let processed = if let Some(room) = &mut self.room {
            let mut processed = vec![0.0; input.len()];
            room.process(input, &mut processed);
            for sample in &mut processed {
                *sample *= dist_gain;
            }
            processed
        } else {
            input.iter().map(|&s| s * dist_gain).collect()
        };

        // Render to speakers
        self.panner.render_source(&processed, azimuth, elevation, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_config() {
        let speaker = SpeakerConfig::from_channel(ChannelPosition::FrontLeft);
        assert!((speaker.position.0 - 30.0).abs() < 0.1);
        assert!((speaker.gain_linear() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vbap_renderer() {
        let layout = ChannelLayout::from_standard(StandardLayout::Surround51);
        let renderer = VbapRenderer::new(&layout).unwrap();

        // Front center source
        let gains = renderer.calculate_gains(0.0, 0.0);
        assert_eq!(gains.len(), 5); // 6 channels minus LFE

        // Should have some gains
        let sum: f32 = gains.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_distance_attenuator() {
        let attenuator = DistanceAttenuator::new(DistanceModel::Inverse)
            .with_reference_distance(1.0)
            .with_rolloff(1.0);

        // At reference distance, gain should be 1.0
        assert!((attenuator.calculate(1.0) - 1.0).abs() < 0.01);

        // At double distance, gain should be ~0.5
        assert!(attenuator.calculate(2.0) < 0.6);
        assert!(attenuator.calculate(2.0) > 0.4);
    }

    #[test]
    fn test_room_simulator() {
        let mut room = RoomSimulator::small_room(48000);
        room.set_wet_mix(0.5);

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 4];

        room.process(&input, &mut output);

        // Output should be non-zero
        let sum: f32 = output.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_spatial_renderer() {
        let mut renderer = SpatialRenderer::new(StandardLayout::Surround51, 48000).unwrap();
        assert_eq!(renderer.channel_count(), 6);

        let input = vec![1.0; 100];
        let mut output: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0; 100]).collect();

        let position = Position3D::new(0.5, 0.5, 0.0);
        let result = renderer.render(&input, position, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(100, 10, 0.5);

        // Feed impulse
        let out1 = delay.process(1.0);
        assert!((out1 - 0.0).abs() < 0.001); // No output yet

        // Process zeros
        for _ in 0..9 {
            delay.process(0.0);
        }

        // Now should get the delayed impulse
        let out2 = delay.process(0.0);
        assert!((out2 - 0.5).abs() < 0.001);
    }
}
