//! Object-based audio support.
//!
//! This module provides object-based audio handling including:
//! - Audio objects with 3D position (x, y, z)
//! - Object metadata (size, diffusion)
//! - Bed channels + objects model
//! - Object rendering to speaker layouts

use crate::channels::{ChannelLayout, ChannelPosition, StandardLayout};
use crate::error::{ObjectAudioError, Result, SpatialError};
use std::collections::HashMap;

/// 3D position for an audio object.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Position3D {
    /// X coordinate (-1.0 = left, 1.0 = right).
    pub x: f32,
    /// Y coordinate (-1.0 = back, 1.0 = front).
    pub y: f32,
    /// Z coordinate (-1.0 = below, 1.0 = above).
    pub z: f32,
}

impl Position3D {
    /// Create a new 3D position.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create a position at the origin (center).
    pub fn origin() -> Self {
        Self::default()
    }

    /// Create position from spherical coordinates.
    /// Azimuth: 0 = front, positive = left.
    /// Elevation: 0 = horizontal, positive = up.
    /// Distance: 0 = center, 1 = unit sphere.
    pub fn from_spherical(azimuth: f32, elevation: f32, distance: f32) -> Self {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        Self {
            x: distance * az_rad.sin() * el_rad.cos(),
            y: distance * az_rad.cos() * el_rad.cos(),
            z: distance * el_rad.sin(),
        }
    }

    /// Convert to spherical coordinates (azimuth, elevation, distance).
    pub fn to_spherical(&self) -> (f32, f32, f32) {
        let distance = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        if distance < 0.0001 {
            return (0.0, 0.0, 0.0);
        }

        let azimuth = self.x.atan2(self.y).to_degrees();
        let elevation = (self.z / distance).asin().to_degrees();

        (azimuth, elevation, distance)
    }

    /// Calculate distance from origin.
    pub fn distance(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Calculate distance to another position.
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit sphere.
    pub fn normalize(&self) -> Self {
        let d = self.distance();
        if d < 0.0001 {
            Self::origin()
        } else {
            Self {
                x: self.x / d,
                y: self.y / d,
                z: self.z / d,
            }
        }
    }

    /// Clamp to valid range (-1 to 1 for each axis).
    pub fn clamp(&self) -> Self {
        Self {
            x: self.x.clamp(-1.0, 1.0),
            y: self.y.clamp(-1.0, 1.0),
            z: self.z.clamp(-1.0, 1.0),
        }
    }

    /// Linear interpolation to another position.
    pub fn lerp(&self, other: &Position3D, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

/// Metadata for an audio object.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectMetadata {
    /// Object size/spread (0.0 = point source, 1.0 = fully diffuse).
    pub size: f32,
    /// Diffusion amount (0.0 = focused, 1.0 = enveloping).
    pub diffusion: f32,
    /// Gain in dB.
    pub gain_db: f32,
    /// Priority (higher = more important, used for object limiting).
    pub priority: u32,
    /// Object is dialog (for dialog normalization).
    pub is_dialog: bool,
    /// Object is ambient/environmental.
    pub is_ambient: bool,
}

impl Default for ObjectMetadata {
    fn default() -> Self {
        Self {
            size: 0.0,
            diffusion: 0.0,
            gain_db: 0.0,
            priority: 0,
            is_dialog: false,
            is_ambient: false,
        }
    }
}

impl ObjectMetadata {
    /// Create new object metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata for a dialog object.
    pub fn dialog() -> Self {
        Self {
            is_dialog: true,
            priority: 100,
            ..Default::default()
        }
    }

    /// Create metadata for an ambient object.
    pub fn ambient() -> Self {
        Self {
            size: 0.5,
            diffusion: 0.5,
            is_ambient: true,
            ..Default::default()
        }
    }

    /// Get linear gain from dB.
    pub fn gain_linear(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }

    /// Set gain from linear value.
    pub fn set_gain_linear(&mut self, linear: f32) {
        self.gain_db = if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            -120.0
        };
    }
}

/// An audio object with position and metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioObject {
    /// Unique object ID.
    pub id: u32,
    /// Object name.
    pub name: Option<String>,
    /// Current 3D position.
    pub position: Position3D,
    /// Object metadata.
    pub metadata: ObjectMetadata,
    /// Audio data (samples).
    #[cfg_attr(feature = "serde", serde(skip))]
    samples: Vec<f32>,
}

impl AudioObject {
    /// Create a new audio object.
    pub fn new(id: u32, position: Position3D) -> Self {
        Self {
            id,
            name: None,
            position,
            metadata: ObjectMetadata::default(),
            samples: Vec::new(),
        }
    }

    /// Create a named audio object.
    pub fn named(id: u32, name: impl Into<String>, position: Position3D) -> Self {
        Self {
            id,
            name: Some(name.into()),
            position,
            metadata: ObjectMetadata::default(),
            samples: Vec::new(),
        }
    }

    /// Set the audio samples.
    pub fn set_samples(&mut self, samples: Vec<f32>) {
        self.samples = samples;
    }

    /// Get the audio samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get mutable audio samples.
    pub fn samples_mut(&mut self) -> &mut Vec<f32> {
        &mut self.samples
    }

    /// Update position with interpolation.
    pub fn move_to(&mut self, target: Position3D, steps: usize) -> Vec<Position3D> {
        let mut trajectory = Vec::with_capacity(steps);
        for i in 0..steps {
            let t = i as f32 / steps as f32;
            trajectory.push(self.position.lerp(&target, t));
        }
        self.position = target;
        trajectory
    }

    /// Get spherical coordinates of position.
    pub fn spherical_position(&self) -> (f32, f32, f32) {
        self.position.to_spherical()
    }
}

/// Object trajectory for animation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectTrajectory {
    /// Object ID.
    pub object_id: u32,
    /// Keyframes: (time_ms, position).
    pub keyframes: Vec<(u64, Position3D)>,
    /// Interpolation type.
    pub interpolation: TrajectoryInterpolation,
}

/// Trajectory interpolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrajectoryInterpolation {
    /// No interpolation (step).
    Step,
    /// Linear interpolation.
    #[default]
    Linear,
    /// Cubic spline interpolation.
    Cubic,
}

impl ObjectTrajectory {
    /// Create a new trajectory.
    pub fn new(object_id: u32) -> Self {
        Self {
            object_id,
            keyframes: Vec::new(),
            interpolation: TrajectoryInterpolation::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time_ms: u64, position: Position3D) {
        self.keyframes.push((time_ms, position));
        self.keyframes.sort_by_key(|(t, _)| *t);
    }

    /// Get position at a specific time.
    pub fn position_at(&self, time_ms: u64) -> Option<Position3D> {
        if self.keyframes.is_empty() {
            return None;
        }

        // Find surrounding keyframes
        let mut prev = &self.keyframes[0];
        let mut next = prev;

        for kf in &self.keyframes {
            if kf.0 <= time_ms {
                prev = kf;
            }
            if kf.0 >= time_ms {
                next = kf;
                break;
            }
        }

        if prev.0 == next.0 {
            return Some(prev.1);
        }

        match self.interpolation {
            TrajectoryInterpolation::Step => Some(prev.1),
            TrajectoryInterpolation::Linear => {
                let t = (time_ms - prev.0) as f32 / (next.0 - prev.0) as f32;
                Some(prev.1.lerp(&next.1, t))
            }
            TrajectoryInterpolation::Cubic => {
                // Simplified cubic (actually Hermite) interpolation
                let t = (time_ms - prev.0) as f32 / (next.0 - prev.0) as f32;
                let t2 = t * t;
                let t3 = t2 * t;
                let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h2 = -2.0 * t3 + 3.0 * t2;

                Some(Position3D {
                    x: prev.1.x * h1 + next.1.x * h2,
                    y: prev.1.y * h1 + next.1.y * h2,
                    z: prev.1.z * h1 + next.1.z * h2,
                })
            }
        }
    }

    /// Get the duration of the trajectory.
    pub fn duration_ms(&self) -> u64 {
        if self.keyframes.is_empty() {
            0
        } else {
            self.keyframes.last().unwrap().0 - self.keyframes.first().unwrap().0
        }
    }
}

/// Bed channel in object-based audio.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BedChannel {
    /// Channel position.
    pub position: ChannelPosition,
    /// Gain in dB.
    pub gain_db: f32,
    /// Audio samples.
    #[cfg_attr(feature = "serde", serde(skip))]
    samples: Vec<f32>,
}

impl BedChannel {
    /// Create a new bed channel.
    pub fn new(position: ChannelPosition) -> Self {
        Self {
            position,
            gain_db: 0.0,
            samples: Vec::new(),
        }
    }

    /// Set samples.
    pub fn set_samples(&mut self, samples: Vec<f32>) {
        self.samples = samples;
    }

    /// Get samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get linear gain.
    pub fn gain_linear(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }
}

/// Audio bed (fixed channel layout).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioBed {
    /// Bed ID.
    pub id: u32,
    /// Bed name.
    pub name: Option<String>,
    /// Channel layout.
    pub layout: StandardLayout,
    /// Bed channels.
    pub channels: Vec<BedChannel>,
}

impl AudioBed {
    /// Create a new audio bed.
    pub fn new(id: u32, layout: StandardLayout) -> Self {
        let positions = layout.positions();
        let channels = positions
            .into_iter()
            .map(BedChannel::new)
            .collect();

        Self {
            id,
            name: None,
            layout,
            channels,
        }
    }

    /// Create a named audio bed.
    pub fn named(id: u32, name: impl Into<String>, layout: StandardLayout) -> Self {
        let mut bed = Self::new(id, layout);
        bed.name = Some(name.into());
        bed
    }

    /// Get channel by position.
    pub fn channel(&self, position: ChannelPosition) -> Option<&BedChannel> {
        self.channels.iter().find(|c| c.position == position)
    }

    /// Get mutable channel by position.
    pub fn channel_mut(&mut self, position: ChannelPosition) -> Option<&mut BedChannel> {
        self.channels.iter_mut().find(|c| c.position == position)
    }

    /// Get the channel layout.
    pub fn channel_layout(&self) -> ChannelLayout {
        ChannelLayout::from_standard(self.layout)
    }
}

/// Object-based audio scene.
#[derive(Debug, Clone, Default)]
pub struct ObjectAudioScene {
    /// Audio objects.
    objects: HashMap<u32, AudioObject>,
    /// Audio beds.
    beds: HashMap<u32, AudioBed>,
    /// Object trajectories.
    trajectories: HashMap<u32, ObjectTrajectory>,
    /// Maximum number of objects.
    max_objects: u32,
    /// Next object ID.
    next_id: u32,
}

impl ObjectAudioScene {
    /// Create a new scene.
    pub fn new(max_objects: u32) -> Self {
        Self {
            max_objects,
            ..Default::default()
        }
    }

    /// Add an audio object.
    pub fn add_object(&mut self, mut object: AudioObject) -> Result<u32> {
        if self.objects.len() >= self.max_objects as usize {
            return Err(SpatialError::ObjectAudio(ObjectAudioError::MaxObjectsExceeded {
                count: self.objects.len() as u32 + 1,
                max: self.max_objects,
            }));
        }

        let id = if object.id == 0 {
            self.next_id += 1;
            object.id = self.next_id;
            self.next_id
        } else {
            object.id
        };

        self.objects.insert(id, object);
        Ok(id)
    }

    /// Remove an audio object.
    pub fn remove_object(&mut self, id: u32) -> Option<AudioObject> {
        self.trajectories.remove(&id);
        self.objects.remove(&id)
    }

    /// Get an audio object.
    pub fn object(&self, id: u32) -> Option<&AudioObject> {
        self.objects.get(&id)
    }

    /// Get a mutable audio object.
    pub fn object_mut(&mut self, id: u32) -> Option<&mut AudioObject> {
        self.objects.get_mut(&id)
    }

    /// Get all objects.
    pub fn objects(&self) -> impl Iterator<Item = &AudioObject> {
        self.objects.values()
    }

    /// Get object count.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Add an audio bed.
    pub fn add_bed(&mut self, bed: AudioBed) -> u32 {
        let id = bed.id;
        self.beds.insert(id, bed);
        id
    }

    /// Get an audio bed.
    pub fn bed(&self, id: u32) -> Option<&AudioBed> {
        self.beds.get(&id)
    }

    /// Get a mutable audio bed.
    pub fn bed_mut(&mut self, id: u32) -> Option<&mut AudioBed> {
        self.beds.get_mut(&id)
    }

    /// Get all beds.
    pub fn beds(&self) -> impl Iterator<Item = &AudioBed> {
        self.beds.values()
    }

    /// Add a trajectory for an object.
    pub fn add_trajectory(&mut self, trajectory: ObjectTrajectory) -> Result<()> {
        if !self.objects.contains_key(&trajectory.object_id) {
            return Err(SpatialError::ObjectAudio(ObjectAudioError::ObjectNotFound {
                id: trajectory.object_id,
            }));
        }
        self.trajectories.insert(trajectory.object_id, trajectory);
        Ok(())
    }

    /// Update object positions from trajectories.
    pub fn update_trajectories(&mut self, time_ms: u64) {
        for (id, trajectory) in &self.trajectories {
            if let Some(position) = trajectory.position_at(time_ms) {
                if let Some(object) = self.objects.get_mut(id) {
                    object.position = position;
                }
            }
        }
    }

    /// Get objects sorted by priority.
    pub fn objects_by_priority(&self) -> Vec<&AudioObject> {
        let mut objects: Vec<_> = self.objects.values().collect();
        objects.sort_by(|a, b| b.metadata.priority.cmp(&a.metadata.priority));
        objects
    }

    /// Limit objects to maximum count, keeping highest priority.
    pub fn limit_objects(&mut self, max: usize) {
        if self.objects.len() <= max {
            return;
        }

        let mut ids: Vec<_> = self.objects.values().map(|o| (o.id, o.metadata.priority)).collect();
        ids.sort_by(|a, b| b.1.cmp(&a.1));

        let to_remove: Vec<_> = ids.iter().skip(max).map(|(id, _)| *id).collect();
        for id in to_remove {
            self.objects.remove(&id);
        }
    }
}

/// Object renderer for converting objects to speaker feeds.
#[derive(Debug)]
pub struct ObjectRenderer {
    /// Output speaker layout.
    layout: ChannelLayout,
    /// Speaker positions as (azimuth, elevation, distance).
    speaker_positions: Vec<(f32, f32, f32)>,
}

impl ObjectRenderer {
    /// Create a new object renderer.
    pub fn new(layout: ChannelLayout) -> Self {
        let speaker_positions = layout
            .positions()
            .iter()
            .map(|p| {
                let (az, el) = p.position_degrees();
                (az, el, 1.0)
            })
            .collect();

        Self {
            layout,
            speaker_positions,
        }
    }

    /// Create renderer for 5.1 output.
    pub fn surround_51() -> Self {
        Self::new(ChannelLayout::from_standard(StandardLayout::Surround51))
    }

    /// Create renderer for 7.1.4 Atmos output.
    pub fn atmos_714() -> Self {
        Self::new(ChannelLayout::from_standard(StandardLayout::Atmos714))
    }

    /// Get the output channel count.
    pub fn channel_count(&self) -> usize {
        self.layout.channel_count() as usize
    }

    /// Render a single object to speaker feeds.
    pub fn render_object(&self, object: &AudioObject) -> Vec<f32> {
        let (az, el, dist) = object.position.to_spherical();
        let num_speakers = self.speaker_positions.len();

        // Calculate gains using simple panning
        let mut gains = self.calculate_panning_gains(az, el, object.metadata.size);

        // Apply object gain
        let gain = object.metadata.gain_linear();
        for g in &mut gains {
            *g *= gain;
        }

        // Apply distance attenuation
        if dist > 0.0 {
            let attenuation = 1.0 / (1.0 + dist);
            for g in &mut gains {
                *g *= attenuation;
            }
        }

        gains
    }

    /// Calculate panning gains for a direction.
    fn calculate_panning_gains(&self, azimuth: f32, elevation: f32, size: f32) -> Vec<f32> {
        let num_speakers = self.speaker_positions.len();
        let mut gains = vec![0.0; num_speakers];

        if size >= 1.0 {
            // Fully diffuse: equal distribution
            let equal_gain = 1.0 / (num_speakers as f32).sqrt();
            for g in &mut gains {
                *g = equal_gain;
            }
            return gains;
        }

        // Calculate distance to each speaker
        let mut total_weight = 0.0;
        let mut weights = vec![0.0; num_speakers];

        for (i, &(spk_az, spk_el, _)) in self.speaker_positions.iter().enumerate() {
            // Skip LFE
            if self.layout.positions()[i].is_lfe() {
                continue;
            }

            // Angular distance
            let az_diff = angular_distance(azimuth, spk_az);
            let el_diff = (elevation - spk_el).abs();
            let angular_dist = (az_diff * az_diff + el_diff * el_diff).sqrt();

            // Weight based on proximity
            let weight = if angular_dist < 1.0 {
                1.0
            } else {
                1.0 / (1.0 + angular_dist * 0.1)
            };

            weights[i] = weight;
            total_weight += weight;
        }

        // Normalize weights to gains
        if total_weight > 0.0 {
            for (i, &weight) in weights.iter().enumerate() {
                // Apply size-based spreading
                let base_gain = weight / total_weight;
                let spread_gain = 1.0 / (num_speakers as f32).sqrt();
                gains[i] = base_gain * (1.0 - size) + spread_gain * size;
            }
        }

        // Normalize for energy preservation
        let sum_sq: f32 = gains.iter().map(|g| g * g).sum();
        if sum_sq > 0.0 {
            let norm = 1.0 / sum_sq.sqrt();
            for g in &mut gains {
                *g *= norm;
            }
        }

        gains
    }

    /// Render an entire scene to output buffers.
    pub fn render_scene(
        &self,
        scene: &ObjectAudioScene,
        num_samples: usize,
    ) -> Vec<Vec<f32>> {
        let num_channels = self.channel_count();
        let mut output = vec![vec![0.0; num_samples]; num_channels];

        // Render beds first
        for bed in scene.beds() {
            for (ch_idx, channel) in bed.channels.iter().enumerate() {
                if let Some(out_idx) = self.layout.index_of(channel.position) {
                    let gain = channel.gain_linear();
                    for (i, &sample) in channel.samples().iter().enumerate().take(num_samples) {
                        output[out_idx][i] += sample * gain;
                    }
                }
            }
        }

        // Render objects
        for object in scene.objects() {
            let gains = self.render_object(object);
            for (ch_idx, &gain) in gains.iter().enumerate() {
                for (i, &sample) in object.samples().iter().enumerate().take(num_samples) {
                    output[ch_idx][i] += sample * gain;
                }
            }
        }

        output
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
    fn test_position_3d() {
        let pos = Position3D::new(0.5, 0.5, 0.0);
        assert!((pos.distance() - 0.707107).abs() < 0.001);

        let origin = Position3D::origin();
        assert!((origin.distance() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_position_spherical_conversion() {
        let pos = Position3D::from_spherical(45.0, 30.0, 1.0);
        let (az, el, dist) = pos.to_spherical();

        assert!((az - 45.0).abs() < 0.1);
        assert!((el - 30.0).abs() < 0.1);
        assert!((dist - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_object_metadata() {
        let meta = ObjectMetadata::dialog();
        assert!(meta.is_dialog);
        assert_eq!(meta.priority, 100);

        let ambient = ObjectMetadata::ambient();
        assert!(ambient.is_ambient);
        assert!(ambient.size > 0.0);
    }

    #[test]
    fn test_audio_object() {
        let mut obj = AudioObject::new(1, Position3D::new(0.5, 0.5, 0.0));
        obj.set_samples(vec![1.0, 0.5, 0.25]);

        assert_eq!(obj.samples().len(), 3);
        assert_eq!(obj.id, 1);
    }

    #[test]
    fn test_object_trajectory() {
        let mut trajectory = ObjectTrajectory::new(1);
        trajectory.add_keyframe(0, Position3D::new(0.0, 1.0, 0.0));
        trajectory.add_keyframe(1000, Position3D::new(1.0, 0.0, 0.0));

        // Midpoint should interpolate
        let pos = trajectory.position_at(500).unwrap();
        assert!((pos.x - 0.5).abs() < 0.01);
        assert!((pos.y - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_audio_bed() {
        let bed = AudioBed::new(1, StandardLayout::Surround51);
        assert_eq!(bed.channels.len(), 6);
        assert!(bed.channel(ChannelPosition::Lfe).is_some());
    }

    #[test]
    fn test_object_audio_scene() {
        let mut scene = ObjectAudioScene::new(10);

        let obj1 = AudioObject::new(1, Position3D::new(0.5, 0.5, 0.0));
        let obj2 = AudioObject::new(2, Position3D::new(-0.5, 0.5, 0.0));

        scene.add_object(obj1).unwrap();
        scene.add_object(obj2).unwrap();

        assert_eq!(scene.object_count(), 2);
        assert!(scene.object(1).is_some());
    }

    #[test]
    fn test_scene_max_objects() {
        let mut scene = ObjectAudioScene::new(2);

        scene.add_object(AudioObject::new(1, Position3D::origin())).unwrap();
        scene.add_object(AudioObject::new(2, Position3D::origin())).unwrap();

        // Third should fail
        let result = scene.add_object(AudioObject::new(3, Position3D::origin()));
        assert!(result.is_err());
    }

    #[test]
    fn test_object_renderer() {
        let renderer = ObjectRenderer::surround_51();
        assert_eq!(renderer.channel_count(), 6);

        let object = AudioObject::new(1, Position3D::new(0.0, 1.0, 0.0)); // Front center
        let gains = renderer.render_object(&object);

        assert_eq!(gains.len(), 6);
        // Front center should have highest gain for center speaker
    }
}
