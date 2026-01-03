//! Audio classification

use crate::{AudioBuffer, Result};

/// Audio content type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioContentType {
    /// Speech/dialogue
    Speech,
    /// Music
    Music,
    /// Mixed speech and music
    Mixed,
    /// Ambient/effects
    Ambient,
    /// Silence
    Silence,
}

/// Audio classification result
#[derive(Debug, Clone)]
pub struct AudioClassification {
    /// Primary content type
    pub content_type: AudioContentType,
    /// Confidence scores
    pub scores: ClassificationScores,
    /// Speech presence probability
    pub speech_probability: f32,
    /// Music presence probability
    pub music_probability: f32,
}

/// Classification confidence scores
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassificationScores {
    pub speech: f32,
    pub music: f32,
    pub ambient: f32,
    pub silence: f32,
}

/// Audio classifier
pub struct AudioClassifier {
    /// Minimum RMS for non-silence
    silence_threshold: f32,
}

impl AudioClassifier {
    /// Create a new classifier
    pub fn new() -> Self {
        Self {
            silence_threshold: 0.01,
        }
    }

    /// Classify audio buffer
    pub fn classify(&self, buffer: &AudioBuffer) -> Result<AudioClassification> {
        // Calculate basic features
        let rms = self.calculate_rms(buffer);
        let zcr = self.calculate_zero_crossing_rate(buffer);
        let spectral_centroid = self.estimate_spectral_centroid(buffer);

        // Simple rule-based classification
        let mut scores = ClassificationScores::default();

        // Silence detection
        if rms < self.silence_threshold {
            scores.silence = 1.0;
            return Ok(AudioClassification {
                content_type: AudioContentType::Silence,
                scores,
                speech_probability: 0.0,
                music_probability: 0.0,
            });
        }

        // Speech vs music heuristics
        // Speech: higher ZCR, moderate spectral centroid
        // Music: lower ZCR, variable spectral centroid

        let speech_indicators = (zcr > 0.05) as u32 + (spectral_centroid < 3000.0) as u32;
        let music_indicators = (zcr < 0.1) as u32 + (spectral_centroid > 1000.0) as u32;

        scores.speech = speech_indicators as f32 / 3.0;
        scores.music = music_indicators as f32 / 3.0;
        scores.ambient = 0.2;

        let content_type = if scores.speech > scores.music + 0.2 {
            AudioContentType::Speech
        } else if scores.music > scores.speech + 0.2 {
            AudioContentType::Music
        } else {
            AudioContentType::Mixed
        };

        Ok(AudioClassification {
            content_type,
            scores,
            speech_probability: scores.speech,
            music_probability: scores.music,
        })
    }

    fn calculate_rms(&self, buffer: &AudioBuffer) -> f32 {
        let sum_squares: f32 = buffer.samples.iter().map(|s| s * s).sum();
        (sum_squares / buffer.samples.len() as f32).sqrt()
    }

    fn calculate_zero_crossing_rate(&self, buffer: &AudioBuffer) -> f32 {
        let mut crossings = 0;
        let samples = &buffer.samples;

        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / samples.len() as f32
    }

    fn estimate_spectral_centroid(&self, buffer: &AudioBuffer) -> f32 {
        // Simplified spectral centroid using ZCR as proxy
        let zcr = self.calculate_zero_crossing_rate(buffer);
        zcr * buffer.sample_rate as f32 / 2.0
    }
}

impl Default for AudioClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_detection() {
        let classifier = AudioClassifier::new();
        let buffer = AudioBuffer::from_samples(vec![0.0; 44100], 1, 44100);

        let result = classifier.classify(&buffer).unwrap();
        assert_eq!(result.content_type, AudioContentType::Silence);
    }

    #[test]
    fn test_classification() {
        let classifier = AudioClassifier::new();

        // Generate a simple tone
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 0.02).sin() * 0.5)
            .collect();
        let buffer = AudioBuffer::from_samples(samples, 1, 44100);

        let result = classifier.classify(&buffer).unwrap();
        assert_ne!(result.content_type, AudioContentType::Silence);
    }
}
