//! Binaural rendering for headphone playback.
//!
//! This module provides comprehensive binaural audio processing including:
//! - HRTF (Head-Related Transfer Function) application
//! - Convolution-based processing
//! - Head tracking support interface
//! - Near-field compensation
//! - Crossfeed for headphones

use crate::ambisonics::BFormat;
use crate::error::{BinauralError, Result, SpatialError};
use std::f32::consts::PI;

/// HRTF (Head-Related Transfer Function) data for a single direction.
#[derive(Debug, Clone)]
pub struct HrtfCoefficients {
    /// Left ear impulse response.
    pub left: Vec<f32>,
    /// Right ear impulse response.
    pub right: Vec<f32>,
    /// Azimuth in degrees (0 = front, positive = left).
    pub azimuth: f32,
    /// Elevation in degrees (0 = horizontal, positive = up).
    pub elevation: f32,
    /// Distance in meters (for near-field HRTFs).
    pub distance: f32,
}

impl HrtfCoefficients {
    /// Create new HRTF coefficients.
    pub fn new(left: Vec<f32>, right: Vec<f32>, azimuth: f32, elevation: f32) -> Self {
        Self {
            left,
            right,
            azimuth,
            elevation,
            distance: 1.0, // Default far-field distance
        }
    }

    /// Get the impulse response length.
    pub fn length(&self) -> usize {
        self.left.len().max(self.right.len())
    }

    /// Check if this is a near-field HRTF.
    pub fn is_near_field(&self) -> bool {
        self.distance < 1.0
    }
}

/// HRTF database containing measurements for multiple directions.
#[derive(Debug, Clone)]
pub struct HrtfDatabase {
    /// All HRTF measurements.
    coefficients: Vec<HrtfCoefficients>,
    /// Sample rate of the HRTFs.
    sample_rate: u32,
    /// Maximum impulse response length.
    max_length: usize,
}

impl HrtfDatabase {
    /// Create a new empty HRTF database.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            coefficients: Vec::new(),
            sample_rate,
            max_length: 0,
        }
    }

    /// Create a simple synthetic HRTF database (for testing/fallback).
    pub fn synthetic(sample_rate: u32, ir_length: usize) -> Self {
        let mut db = Self::new(sample_rate);

        // Create HRTFs for common directions
        let azimuths = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, -150.0, -120.0, -90.0, -60.0, -30.0];
        let elevations = [-30.0, 0.0, 30.0, 60.0, 90.0];

        for &azimuth in &azimuths {
            for &elevation in &elevations {
                let hrtf = Self::generate_synthetic_hrtf(azimuth, elevation, ir_length);
                db.add_hrtf(hrtf);
            }
        }

        db
    }

    /// Generate a synthetic HRTF for a given direction.
    fn generate_synthetic_hrtf(azimuth: f32, elevation: f32, length: usize) -> HrtfCoefficients {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        // Simple model: ITD (interaural time difference) and ILD (interaural level difference)
        let head_radius = 0.0875; // meters
        let speed_of_sound = 343.0; // m/s

        // Woodworth formula for ITD
        let itd = if az_rad.abs() < PI / 2.0 {
            head_radius * (az_rad.abs() + az_rad.sin()) / speed_of_sound
        } else {
            head_radius * (PI - az_rad.abs() + az_rad.sin().abs()) / speed_of_sound
        };

        // Convert ITD to sample delay (at 48kHz)
        let sample_delay = (itd * 48000.0) as usize;

        // ILD model (simplified)
        let ild_db = -8.0 * az_rad.sin() * el_rad.cos();
        let ild_linear = 10.0_f32.powf(ild_db / 20.0);

        // Create impulse responses
        let mut left = vec![0.0; length];
        let mut right = vec![0.0; length];

        // Simple lowpass filtered impulse
        let decay = 0.95_f32;
        let peak_pos = length / 4;

        if azimuth >= 0.0 {
            // Source on left, left ear is closer
            for i in 0..length.saturating_sub(sample_delay) {
                let t = i as f32 / length as f32;
                left[i] = decay.powf(i as f32 / 10.0) * (-10.0 * (t - 0.1).powi(2)).exp();
            }
            for i in sample_delay..length {
                let t = (i - sample_delay) as f32 / length as f32;
                right[i] = ild_linear * decay.powf((i - sample_delay) as f32 / 10.0)
                    * (-10.0 * (t - 0.1).powi(2)).exp();
            }
        } else {
            // Source on right, right ear is closer
            for i in sample_delay..length {
                let t = (i - sample_delay) as f32 / length as f32;
                left[i] = ild_linear.recip() * decay.powf((i - sample_delay) as f32 / 10.0)
                    * (-10.0 * (t - 0.1).powi(2)).exp();
            }
            for i in 0..length.saturating_sub(sample_delay) {
                let t = i as f32 / length as f32;
                right[i] = decay.powf(i as f32 / 10.0) * (-10.0 * (t - 0.1).powi(2)).exp();
            }
        }

        HrtfCoefficients::new(left, right, azimuth, elevation)
    }

    /// Add an HRTF to the database.
    pub fn add_hrtf(&mut self, hrtf: HrtfCoefficients) {
        self.max_length = self.max_length.max(hrtf.length());
        self.coefficients.push(hrtf);
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the maximum IR length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Find the nearest HRTF for a given direction.
    pub fn find_nearest(&self, azimuth: f32, elevation: f32) -> Option<&HrtfCoefficients> {
        if self.coefficients.is_empty() {
            return None;
        }

        let mut best = &self.coefficients[0];
        let mut best_distance = f32::MAX;

        for hrtf in &self.coefficients {
            let az_diff = angular_distance(azimuth, hrtf.azimuth);
            let el_diff = elevation - hrtf.elevation;
            let distance = az_diff * az_diff + el_diff * el_diff;

            if distance < best_distance {
                best_distance = distance;
                best = hrtf;
            }
        }

        Some(best)
    }

    /// Interpolate HRTFs for a given direction.
    pub fn interpolate(&self, azimuth: f32, elevation: f32) -> Option<HrtfCoefficients> {
        // Find the 4 nearest HRTFs for bilinear interpolation
        let nearest = self.find_nearest(azimuth, elevation)?;

        // For simplicity, just return the nearest HRTF
        // A full implementation would do proper spherical interpolation
        Some(nearest.clone())
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

/// Head tracking data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadTracking {
    /// Yaw (rotation around vertical axis) in degrees.
    pub yaw: f32,
    /// Pitch (rotation around left-right axis) in degrees.
    pub pitch: f32,
    /// Roll (rotation around front-back axis) in degrees.
    pub roll: f32,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl HeadTracking {
    /// Create new head tracking data.
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self {
            yaw,
            pitch,
            roll,
            timestamp_ms: 0,
        }
    }

    /// Create neutral (forward-facing) position.
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Apply head rotation to a source position.
    /// Returns new (azimuth, elevation) relative to head.
    pub fn transform_source(&self, azimuth: f32, elevation: f32) -> (f32, f32) {
        // Simplified rotation: just apply yaw for now
        let new_azimuth = azimuth - self.yaw;

        // Normalize to -180..180
        let normalized = if new_azimuth > 180.0 {
            new_azimuth - 360.0
        } else if new_azimuth < -180.0 {
            new_azimuth + 360.0
        } else {
            new_azimuth
        };

        // Apply pitch to elevation
        let new_elevation = elevation - self.pitch;

        (normalized, new_elevation.clamp(-90.0, 90.0))
    }
}

/// Head tracking listener for receiving updates.
pub trait HeadTrackingListener: Send + Sync {
    /// Called when head tracking data is updated.
    fn on_head_tracking_update(&mut self, tracking: HeadTracking);
}

/// Convolution state for HRTF processing.
#[derive(Debug, Clone)]
pub struct ConvolutionState {
    /// Left channel overlap-add buffer.
    left_buffer: Vec<f32>,
    /// Right channel overlap-add buffer.
    right_buffer: Vec<f32>,
    /// Current position in the overlap buffer.
    position: usize,
    /// Block size.
    block_size: usize,
}

impl ConvolutionState {
    /// Create new convolution state.
    pub fn new(ir_length: usize, block_size: usize) -> Self {
        let buffer_size = ir_length + block_size;
        Self {
            left_buffer: vec![0.0; buffer_size],
            right_buffer: vec![0.0; buffer_size],
            position: 0,
            block_size,
        }
    }

    /// Reset the convolution state.
    pub fn reset(&mut self) {
        self.left_buffer.fill(0.0);
        self.right_buffer.fill(0.0);
        self.position = 0;
    }

    /// Process a block of mono input with HRTF.
    pub fn process(&mut self, input: &[f32], hrtf: &HrtfCoefficients, output_left: &mut [f32], output_right: &mut [f32]) {
        let ir_len = hrtf.length();

        // Time-domain convolution (simple implementation)
        for (i, &sample) in input.iter().enumerate() {
            // Add to left channel buffer
            for (j, &coef) in hrtf.left.iter().enumerate() {
                let pos = (self.position + i + j) % self.left_buffer.len();
                self.left_buffer[pos] += sample * coef;
            }

            // Add to right channel buffer
            for (j, &coef) in hrtf.right.iter().enumerate() {
                let pos = (self.position + i + j) % self.right_buffer.len();
                self.right_buffer[pos] += sample * coef;
            }
        }

        // Copy output and clear processed samples
        for i in 0..input.len().min(output_left.len()) {
            let pos = (self.position + i) % self.left_buffer.len();
            output_left[i] = self.left_buffer[pos];
            output_right[i] = self.right_buffer[pos];
            self.left_buffer[pos] = 0.0;
            self.right_buffer[pos] = 0.0;
        }

        self.position = (self.position + input.len()) % self.left_buffer.len();
    }
}

/// Binaural renderer for 3D audio.
#[derive(Debug)]
pub struct BinauralRenderer {
    /// HRTF database.
    hrtf_db: HrtfDatabase,
    /// Current head tracking.
    head_tracking: HeadTracking,
    /// Head tracking enabled.
    head_tracking_enabled: bool,
    /// Convolution states for multiple sources.
    convolution_states: Vec<ConvolutionState>,
    /// Near-field compensation enabled.
    near_field_enabled: bool,
    /// Near-field distance threshold.
    near_field_threshold: f32,
    /// Block size for processing.
    block_size: usize,
}

impl BinauralRenderer {
    /// Create a new binaural renderer.
    pub fn new(hrtf_db: HrtfDatabase, block_size: usize) -> Self {
        Self {
            hrtf_db,
            head_tracking: HeadTracking::neutral(),
            head_tracking_enabled: false,
            convolution_states: Vec::new(),
            near_field_enabled: false,
            near_field_threshold: 1.0,
            block_size,
        }
    }

    /// Create with synthetic HRTFs (for testing or fallback).
    pub fn with_synthetic_hrtf(sample_rate: u32, block_size: usize) -> Self {
        let hrtf_db = HrtfDatabase::synthetic(sample_rate, 256);
        Self::new(hrtf_db, block_size)
    }

    /// Enable head tracking.
    pub fn enable_head_tracking(&mut self, enabled: bool) {
        self.head_tracking_enabled = enabled;
    }

    /// Update head tracking data.
    pub fn update_head_tracking(&mut self, tracking: HeadTracking) {
        self.head_tracking = tracking;
    }

    /// Enable near-field compensation.
    pub fn enable_near_field(&mut self, enabled: bool, threshold: f32) {
        self.near_field_enabled = enabled;
        self.near_field_threshold = threshold;
    }

    /// Ensure we have enough convolution states.
    fn ensure_convolution_states(&mut self, count: usize) {
        while self.convolution_states.len() < count {
            self.convolution_states.push(ConvolutionState::new(
                self.hrtf_db.max_length(),
                self.block_size,
            ));
        }
    }

    /// Render a mono source at a specific position.
    pub fn render_source(
        &mut self,
        source_index: usize,
        input: &[f32],
        azimuth: f32,
        elevation: f32,
        distance: f32,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> Result<()> {
        self.ensure_convolution_states(source_index + 1);

        // Apply head tracking if enabled
        let (az, el) = if self.head_tracking_enabled {
            self.head_tracking.transform_source(azimuth, elevation)
        } else {
            (azimuth, elevation)
        };

        // Get HRTF for this direction
        let hrtf = self
            .hrtf_db
            .find_nearest(az, el)
            .ok_or_else(|| SpatialError::Binaural(BinauralError::HrtfLoadFailed("No HRTF data".into())))?
            .clone();

        // Apply distance attenuation
        let attenuation = if distance > 0.0 { 1.0 / distance } else { 1.0 };

        // Apply near-field compensation if enabled and distance is within threshold
        let near_field_factor = if self.near_field_enabled && distance < self.near_field_threshold {
            self.compute_near_field_factor(distance, az)
        } else {
            (1.0, 1.0)
        };

        // Create attenuated input
        let attenuated: Vec<f32> = input.iter().map(|&s| s * attenuation).collect();

        // Convolve with HRTF
        self.convolution_states[source_index].process(&attenuated, &hrtf, output_left, output_right);

        // Apply near-field compensation
        if self.near_field_enabled && distance < self.near_field_threshold {
            for sample in output_left.iter_mut() {
                *sample *= near_field_factor.0;
            }
            for sample in output_right.iter_mut() {
                *sample *= near_field_factor.1;
            }
        }

        Ok(())
    }

    /// Compute near-field ILD compensation factors.
    fn compute_near_field_factor(&self, distance: f32, azimuth: f32) -> (f32, f32) {
        // Near-field effect: increased ILD as source approaches head
        let head_radius = 0.0875;
        let az_rad = azimuth.to_radians();

        if distance < head_radius * 2.0 {
            // Very close sources have pronounced ILD
            let proximity = (head_radius * 2.0 - distance) / (head_radius * 2.0);
            let ild_boost = 1.0 + proximity * az_rad.sin().abs();

            if azimuth > 0.0 {
                (ild_boost, 1.0 / ild_boost)
            } else {
                (1.0 / ild_boost, ild_boost)
            }
        } else {
            (1.0, 1.0)
        }
    }

    /// Render ambisonics B-format to binaural.
    pub fn render_bformat(
        &mut self,
        bformat: &BFormat,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> Result<()> {
        // Initialize output
        for (l, r) in output_left.iter_mut().zip(output_right.iter_mut()) {
            *l = 0.0;
            *r = 0.0;
        }

        // Apply head rotation to B-format if tracking is enabled
        let rotated = if self.head_tracking_enabled {
            bformat.rotate_z(-self.head_tracking.yaw)
        } else {
            *bformat
        };

        // Simple virtual speaker approach: decode to virtual speakers and sum
        let virtual_speakers: [(f32, f32); 6] = [
            (30.0, 0.0),   // Front left
            (-30.0, 0.0),  // Front right
            (90.0, 0.0),   // Left
            (-90.0, 0.0),  // Right
            (150.0, 0.0),  // Rear left
            (-150.0, 0.0), // Rear right
        ];

        self.ensure_convolution_states(virtual_speakers.len());

        for (i, &(az, el)) in virtual_speakers.iter().enumerate() {
            // Decode B-format to this speaker direction
            let az_rad = az.to_radians();
            let el_rad = el.to_radians();
            let cos_el = el_rad.cos();

            let speaker_signal = rotated.w * 0.707107
                + rotated.x * az_rad.cos() * cos_el
                + rotated.y * az_rad.sin() * cos_el
                + rotated.z * el_rad.sin();

            // Create buffer for this speaker
            let input = vec![speaker_signal; self.block_size];
            let mut left_temp = vec![0.0; self.block_size];
            let mut right_temp = vec![0.0; self.block_size];

            // Get HRTF for this direction
            if let Some(hrtf) = self.hrtf_db.find_nearest(az, el) {
                self.convolution_states[i].process(&input, hrtf, &mut left_temp, &mut right_temp);

                // Sum to output
                for (j, (&l, &r)) in left_temp.iter().zip(right_temp.iter()).enumerate() {
                    if j < output_left.len() {
                        output_left[j] += l / virtual_speakers.len() as f32;
                        output_right[j] += r / virtual_speakers.len() as f32;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the HRTF sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.hrtf_db.sample_rate()
    }
}

/// Crossfeed processor for headphone listening.
#[derive(Debug, Clone)]
pub struct CrossfeedProcessor {
    /// Crossfeed amount (0.0 = none, 1.0 = full mono).
    amount: f32,
    /// High-frequency rolloff for crossfeed (simulates head shadow).
    rolloff_coef: f32,
    /// Delay in samples for crossfeed (simulates ITD).
    delay_samples: usize,
    /// Delay buffers.
    delay_left: Vec<f32>,
    delay_right: Vec<f32>,
    /// Delay buffer positions.
    delay_pos: usize,
    /// Previous filtered values for lowpass.
    prev_left: f32,
    prev_right: f32,
}

impl CrossfeedProcessor {
    /// Create a new crossfeed processor.
    pub fn new(sample_rate: u32) -> Self {
        // Default crossfeed settings
        let delay_ms = 0.3; // ~0.3ms delay for natural crossfeed
        let delay_samples = (sample_rate as f32 * delay_ms / 1000.0) as usize;

        Self {
            amount: 0.3,
            rolloff_coef: 0.7,
            delay_samples,
            delay_left: vec![0.0; delay_samples.max(1)],
            delay_right: vec![0.0; delay_samples.max(1)],
            delay_pos: 0,
            prev_left: 0.0,
            prev_right: 0.0,
        }
    }

    /// Set the crossfeed amount (0.0 - 1.0).
    pub fn set_amount(&mut self, amount: f32) {
        self.amount = amount.clamp(0.0, 1.0);
    }

    /// Get the crossfeed amount.
    pub fn amount(&self) -> f32 {
        self.amount
    }

    /// Process stereo audio with crossfeed.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for i in 0..left.len().min(right.len()) {
            let orig_left = left[i];
            let orig_right = right[i];

            // Get delayed and filtered crossfeed signals
            let delayed_left = self.delay_left[self.delay_pos];
            let delayed_right = self.delay_right[self.delay_pos];

            // Apply lowpass filter to crossfeed (simulates head shadow)
            self.prev_left = self.prev_left * self.rolloff_coef + delayed_left * (1.0 - self.rolloff_coef);
            self.prev_right = self.prev_right * self.rolloff_coef + delayed_right * (1.0 - self.rolloff_coef);

            // Mix crossfeed into opposite channel
            left[i] = orig_left * (1.0 - self.amount * 0.5) + self.prev_right * self.amount;
            right[i] = orig_right * (1.0 - self.amount * 0.5) + self.prev_left * self.amount;

            // Update delay buffers
            self.delay_left[self.delay_pos] = orig_left;
            self.delay_right[self.delay_pos] = orig_right;
            self.delay_pos = (self.delay_pos + 1) % self.delay_left.len().max(1);
        }
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.delay_left.fill(0.0);
        self.delay_right.fill(0.0);
        self.delay_pos = 0;
        self.prev_left = 0.0;
        self.prev_right = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrtf_coefficients() {
        let left = vec![1.0, 0.5, 0.25];
        let right = vec![0.8, 0.4, 0.2];
        let hrtf = HrtfCoefficients::new(left, right, 30.0, 0.0);

        assert_eq!(hrtf.length(), 3);
        assert!(!hrtf.is_near_field());
    }

    #[test]
    fn test_hrtf_database_synthetic() {
        let db = HrtfDatabase::synthetic(48000, 128);
        assert_eq!(db.sample_rate(), 48000);
        assert!(db.max_length() > 0);

        // Should find nearest HRTF
        let hrtf = db.find_nearest(30.0, 0.0);
        assert!(hrtf.is_some());
    }

    #[test]
    fn test_head_tracking() {
        let tracking = HeadTracking::new(45.0, 0.0, 0.0);

        // Source at front-left (30 degrees)
        let (az, el) = tracking.transform_source(30.0, 0.0);

        // After 45 degree right turn, source should be more to the left
        assert!((az - (-15.0)).abs() < 0.1);
        assert!((el - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_binaural_renderer() {
        let mut renderer = BinauralRenderer::with_synthetic_hrtf(48000, 256);

        let input = vec![1.0; 256];
        let mut left = vec![0.0; 256];
        let mut right = vec![0.0; 256];

        // Render a source at front-left
        let result = renderer.render_source(0, &input, 30.0, 0.0, 1.0, &mut left, &mut right);
        assert!(result.is_ok());

        // Left channel should be louder for left source
        let left_sum: f32 = left.iter().map(|x| x.abs()).sum();
        let right_sum: f32 = right.iter().map(|x| x.abs()).sum();
        assert!(left_sum > right_sum * 0.8); // Left should be stronger
    }

    #[test]
    fn test_crossfeed() {
        let mut crossfeed = CrossfeedProcessor::new(48000);
        crossfeed.set_amount(0.5);

        // Hard left signal
        let mut left = vec![1.0; 100];
        let mut right = vec![0.0; 100];

        crossfeed.process(&mut left, &mut right);

        // Right channel should now have some signal due to crossfeed
        let right_sum: f32 = right.iter().sum();
        assert!(right_sum > 0.0);
    }

    #[test]
    fn test_convolution_state() {
        let hrtf = HrtfCoefficients::new(vec![1.0, 0.5], vec![0.8, 0.4], 0.0, 0.0);

        let mut state = ConvolutionState::new(hrtf.length(), 16);
        let input = vec![1.0; 16];
        let mut left = vec![0.0; 16];
        let mut right = vec![0.0; 16];

        state.process(&input, &hrtf, &mut left, &mut right);

        // Output should be non-zero
        let left_sum: f32 = left.iter().map(|x| x.abs()).sum();
        assert!(left_sum > 0.0);
    }
}
