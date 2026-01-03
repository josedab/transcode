//! Psychoacoustic model for AAC encoding.

use std::f32::consts::PI;

/// Psychoacoustic model configuration.
#[derive(Debug, Clone)]
pub struct PsyModelConfig {
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bit rate.
    pub bitrate: u32,
    /// Quality factor (0.0 - 1.0).
    pub quality: f32,
}

impl Default for PsyModelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bitrate: 128000,
            quality: 0.5,
        }
    }
}

/// Psychoacoustic analysis for a channel.
#[derive(Debug, Clone)]
pub struct PsyAnalysis {
    /// Masking thresholds per band.
    pub thresholds: Vec<f32>,
    /// Signal-to-mask ratio per band.
    pub smr: Vec<f32>,
    /// Perceptual entropy per band.
    pub pe: Vec<f32>,
    /// Total perceptual entropy.
    pub total_pe: f32,
    /// Window decision.
    pub use_short_windows: bool,
    /// Attack detected.
    pub attack: bool,
}

impl Default for PsyAnalysis {
    fn default() -> Self {
        Self {
            thresholds: vec![0.0; 64],
            smr: vec![0.0; 64],
            pe: vec![0.0; 64],
            total_pe: 0.0,
            use_short_windows: false,
            attack: false,
        }
    }
}

/// Psychoacoustic model (simplified MPEG-1 Layer 3 model).
pub struct PsyModel {
    /// Configuration.
    config: PsyModelConfig,
    /// Number of scalefactor bands.
    num_bands: usize,
    /// Scalefactor band boundaries.
    band_boundaries: Vec<usize>,
    /// Absolute threshold of hearing per band.
    ath: Vec<f32>,
    /// Spreading function matrix.
    spreading: Vec<Vec<f32>>,
    /// Previous block energy.
    prev_energy: Vec<f32>,
    /// Previous-previous block energy.
    prev_prev_energy: Vec<f32>,
}

impl PsyModel {
    /// Create a new psychoacoustic model.
    pub fn new(config: PsyModelConfig) -> Self {
        let num_bands = 49; // Typical number for AAC

        // Simplified band boundaries for 44100 Hz
        let band_boundaries: Vec<usize> = (0..=num_bands)
            .map(|i| ((i as f32 / num_bands as f32).powf(1.7) * 512.0) as usize)
            .collect();

        // Absolute threshold of hearing
        let ath = Self::compute_ath(&band_boundaries, config.sample_rate);

        // Spreading function
        let spreading = Self::compute_spreading(num_bands);

        Self {
            config,
            num_bands,
            band_boundaries,
            ath,
            spreading,
            prev_energy: vec![0.0; num_bands],
            prev_prev_energy: vec![0.0; num_bands],
        }
    }

    /// Compute absolute threshold of hearing for each band.
    fn compute_ath(boundaries: &[usize], sample_rate: u32) -> Vec<f32> {
        let num_bands = boundaries.len() - 1;
        let mut ath = Vec::with_capacity(num_bands);

        for b in 0..num_bands {
            let freq_start = boundaries[b] as f32 * sample_rate as f32 / 1024.0;
            let freq_end = boundaries[b + 1] as f32 * sample_rate as f32 / 1024.0;
            let freq = (freq_start + freq_end) / 2.0;

            // ISO 226 equal-loudness contour approximation (dB SPL)
            let f_khz = freq / 1000.0;
            let ath_db = 3.64 * f_khz.powf(-0.8)
                - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
                + 1e-3 * f_khz.powi(4);

            // Convert to power
            ath.push(10.0f32.powf(ath_db / 10.0));
        }

        ath
    }

    /// Compute spreading function matrix.
    fn compute_spreading(num_bands: usize) -> Vec<Vec<f32>> {
        let mut spreading = vec![vec![0.0f32; num_bands]; num_bands];

        for i in 0..num_bands {
            for j in 0..num_bands {
                let bark_diff = (i as f32 - j as f32) * 0.5; // Approximate bark difference

                // Simplified spreading function
                let spread = if bark_diff <= -1.0 {
                    27.0 * bark_diff + 20.0
                } else if bark_diff < 0.0 {
                    20.0 * bark_diff
                } else if bark_diff < 1.0 {
                    -20.0 * bark_diff
                } else {
                    -5.0 * bark_diff
                };

                spreading[i][j] = 10.0f32.powf(spread / 10.0).max(1e-10);
            }
        }

        spreading
    }

    /// Analyze a frame of audio.
    pub fn analyze(&mut self, spectrum: &[f32; 1024]) -> PsyAnalysis {
        let mut analysis = PsyAnalysis::default();

        // Compute energy per band
        let mut energy = vec![0.0f32; self.num_bands];
        for b in 0..self.num_bands {
            let start = self.band_boundaries[b];
            let end = self.band_boundaries[b + 1].min(512);

            for i in start..end {
                if i < 512 {
                    energy[b] += spectrum[i] * spectrum[i];
                }
            }
            energy[b] = energy[b].max(1e-10);
        }

        // Apply spreading function to get excitation
        let mut excitation = vec![0.0f32; self.num_bands];
        for i in 0..self.num_bands {
            for j in 0..self.num_bands {
                excitation[i] += energy[j] * self.spreading[i][j];
            }
        }

        // Compute masking thresholds
        let offset = 5.0; // Masking offset in dB
        let offset_factor = 10.0f32.powf(-offset / 10.0);

        for b in 0..self.num_bands {
            analysis.thresholds[b] = (excitation[b] * offset_factor).max(self.ath[b]);
        }

        // Compute SMR and PE
        for b in 0..self.num_bands {
            analysis.smr[b] = 10.0 * (energy[b] / analysis.thresholds[b]).log10();

            let pe_factor = if analysis.smr[b] > 0.0 {
                self.config.quality * analysis.smr[b]
            } else {
                0.0
            };
            analysis.pe[b] = pe_factor;
            analysis.total_pe += pe_factor;
        }

        // Attack detection for window switching
        analysis.attack = self.detect_attack(&energy);
        analysis.use_short_windows = analysis.attack;

        // Update energy history
        self.prev_prev_energy = std::mem::take(&mut self.prev_energy);
        self.prev_energy = energy;

        analysis
    }

    /// Detect transient/attack in the signal.
    fn detect_attack(&self, energy: &[f32]) -> bool {
        // Compare current energy to previous frames
        let mut attack_ratio = 0.0f32;

        for b in 0..self.num_bands {
            let prev_max = self.prev_energy[b].max(self.prev_prev_energy[b]).max(1e-10);
            let ratio = energy[b] / prev_max;
            attack_ratio = attack_ratio.max(ratio);
        }

        // Threshold for attack detection
        attack_ratio > 8.0
    }

    /// Get recommended bit allocation for bands.
    pub fn allocate_bits(&self, analysis: &PsyAnalysis, available_bits: u32) -> Vec<u32> {
        let mut allocation = vec![0u32; self.num_bands];

        // Simple proportional allocation based on PE
        if analysis.total_pe > 0.0 {
            for b in 0..self.num_bands {
                let proportion = analysis.pe[b] / analysis.total_pe;
                allocation[b] = (proportion * available_bits as f32) as u32;
            }
        }

        allocation
    }

    /// Reset model state.
    pub fn reset(&mut self) {
        self.prev_energy.fill(0.0);
        self.prev_prev_energy.fill(0.0);
    }
}

/// Compute FFT for psychoacoustic analysis.
pub fn compute_fft_1024(input: &[f32; 1024]) -> [f32; 1024] {
    let mut output = [0.0f32; 1024];

    // Simple DFT (would use FFT in production)
    for k in 0..512 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;

        for n in 0..1024 {
            let angle = -2.0 * PI * (k as f32) * (n as f32) / 1024.0;
            re += input[n] * angle.cos();
            im += input[n] * angle.sin();
        }

        output[k] = (re * re + im * im).sqrt() / 512.0;
    }

    // Mirror for display
    for k in 512..1024 {
        output[k] = output[1023 - k];
    }

    output
}
