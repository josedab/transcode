//! Audio/video synchronization.

use crate::Result;
use std::collections::VecDeque;
use transcode_core::frame::Frame;
use transcode_core::packet::Packet;
use transcode_core::rational::Rational;
use transcode_core::sample::Sample;

/// Synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// No synchronization - process packets as they come.
    None,
    /// Audio is master - video syncs to audio.
    #[default]
    AudioMaster,
    /// Video is master - audio syncs to video.
    VideoMaster,
    /// External clock master.
    ExternalClock,
}

/// Synchronization configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Sync mode.
    pub mode: SyncMode,
    /// Maximum audio drift before correction (microseconds).
    pub max_audio_drift_us: i64,
    /// Maximum video drift before frame drop/dup (microseconds).
    pub max_video_drift_us: i64,
    /// Audio buffer size in samples.
    pub audio_buffer_size: usize,
    /// Video buffer size in frames.
    pub video_buffer_size: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            mode: SyncMode::AudioMaster,
            max_audio_drift_us: 50_000,  // 50ms
            max_video_drift_us: 100_000, // 100ms
            audio_buffer_size: 4096,
            video_buffer_size: 4,
        }
    }
}

/// Stream timing information.
#[derive(Debug, Clone)]
struct StreamTiming {
    /// Time base.
    time_base: Rational,
    /// Current timestamp in stream time base.
    current_ts: i64,
    /// Start timestamp.
    start_ts: Option<i64>,
}

impl StreamTiming {
    fn new(time_base: Rational) -> Self {
        Self {
            time_base,
            current_ts: 0,
            start_ts: None,
        }
    }

    /// Convert timestamp to microseconds.
    fn ts_to_us(&self, ts: i64) -> i64 {
        if self.time_base.den == 0 {
            return ts;
        }
        ts * 1_000_000 * self.time_base.num / self.time_base.den
    }

    /// Update current timestamp.
    fn update(&mut self, ts: i64) {
        if self.start_ts.is_none() {
            self.start_ts = Some(ts);
        }
        self.current_ts = ts;
    }

    /// Get elapsed time in microseconds.
    fn elapsed_us(&self) -> i64 {
        let start = self.start_ts.unwrap_or(0);
        self.ts_to_us(self.current_ts - start)
    }
}

/// Synchronizer for managing A/V sync.
pub struct Synchronizer {
    config: SyncConfig,
    audio_timing: Option<StreamTiming>,
    video_timing: Option<StreamTiming>,
    audio_buffer: VecDeque<Sample>,
    video_buffer: VecDeque<Frame>,
    master_clock_us: i64,
}

impl Synchronizer {
    /// Create a new synchronizer.
    pub fn new(config: SyncConfig) -> Self {
        Self {
            config,
            audio_timing: None,
            video_timing: None,
            audio_buffer: VecDeque::new(),
            video_buffer: VecDeque::new(),
            master_clock_us: 0,
        }
    }

    /// Configure audio stream.
    pub fn set_audio_stream(&mut self, time_base: Rational) {
        self.audio_timing = Some(StreamTiming::new(time_base));
    }

    /// Configure video stream.
    pub fn set_video_stream(&mut self, time_base: Rational) {
        self.video_timing = Some(StreamTiming::new(time_base));
    }

    /// Process incoming audio samples.
    pub fn push_audio(&mut self, sample: Sample) -> Result<()> {
        if let Some(ref mut timing) = self.audio_timing {
            if sample.pts.is_valid() {
                timing.update(sample.pts.value);
            }
        }

        self.audio_buffer.push_back(sample);

        // Trim buffer if too large
        while self.audio_buffer.len() > self.config.audio_buffer_size {
            self.audio_buffer.pop_front();
        }

        self.update_master_clock();
        Ok(())
    }

    /// Process incoming video frame.
    pub fn push_video(&mut self, frame: Frame) -> Result<()> {
        if let Some(ref mut timing) = self.video_timing {
            if frame.pts.is_valid() {
                timing.update(frame.pts.value);
            }
        }

        self.video_buffer.push_back(frame);

        // Trim buffer if too large
        while self.video_buffer.len() > self.config.video_buffer_size {
            self.video_buffer.pop_front();
        }

        self.update_master_clock();
        Ok(())
    }

    /// Get next audio samples ready for output.
    pub fn pop_audio(&mut self) -> Option<Sample> {
        self.audio_buffer.pop_front()
    }

    /// Get next video frame ready for output.
    pub fn pop_video(&mut self) -> Option<Frame> {
        match self.config.mode {
            SyncMode::None => self.video_buffer.pop_front(),
            SyncMode::AudioMaster => self.pop_video_synced_to_audio(),
            SyncMode::VideoMaster => self.video_buffer.pop_front(),
            SyncMode::ExternalClock => self.pop_video_synced_to_clock(),
        }
    }

    /// Pop video frame synced to audio.
    fn pop_video_synced_to_audio(&mut self) -> Option<Frame> {
        let audio_clock = self.audio_timing.as_ref()?.elapsed_us();

        // Find frame closest to audio clock
        if let Some(frame) = self.video_buffer.front() {
            let video_timing = self.video_timing.as_ref()?;
            let frame_ts = if frame.pts.is_valid() { frame.pts.value } else { 0 };
            let start_ts = video_timing.start_ts.unwrap_or(0);
            let frame_us = video_timing.ts_to_us(frame_ts - start_ts);
            let drift = frame_us - audio_clock;

            if drift > self.config.max_video_drift_us {
                // Frame is too far ahead - wait
                return None;
            } else if drift < -self.config.max_video_drift_us {
                // Frame is too far behind - drop it and try next
                self.video_buffer.pop_front();
                return self.pop_video_synced_to_audio();
            }
        }

        self.video_buffer.pop_front()
    }

    /// Pop video frame synced to external clock.
    fn pop_video_synced_to_clock(&mut self) -> Option<Frame> {
        if let Some(frame) = self.video_buffer.front() {
            let video_timing = self.video_timing.as_ref()?;
            let frame_ts = if frame.pts.is_valid() { frame.pts.value } else { 0 };
            let start_ts = video_timing.start_ts.unwrap_or(0);
            let frame_us = video_timing.ts_to_us(frame_ts - start_ts);
            let drift = frame_us - self.master_clock_us;

            if drift > self.config.max_video_drift_us {
                return None;
            } else if drift < -self.config.max_video_drift_us {
                self.video_buffer.pop_front();
                return self.pop_video_synced_to_clock();
            }
        }

        self.video_buffer.pop_front()
    }

    /// Update the master clock based on sync mode.
    fn update_master_clock(&mut self) {
        self.master_clock_us = match self.config.mode {
            SyncMode::AudioMaster => {
                self.audio_timing.as_ref().map(|t| t.elapsed_us()).unwrap_or(0)
            }
            SyncMode::VideoMaster => {
                self.video_timing.as_ref().map(|t| t.elapsed_us()).unwrap_or(0)
            }
            _ => self.master_clock_us,
        };
    }

    /// Set external clock time.
    pub fn set_clock(&mut self, clock_us: i64) {
        if self.config.mode == SyncMode::ExternalClock {
            self.master_clock_us = clock_us;
        }
    }

    /// Get current master clock time in microseconds.
    pub fn clock(&self) -> i64 {
        self.master_clock_us
    }

    /// Get audio drift in microseconds.
    pub fn audio_drift(&self) -> i64 {
        let audio_clock = self.audio_timing.as_ref().map(|t| t.elapsed_us()).unwrap_or(0);
        audio_clock - self.master_clock_us
    }

    /// Get video drift in microseconds.
    pub fn video_drift(&self) -> i64 {
        let video_clock = self.video_timing.as_ref().map(|t| t.elapsed_us()).unwrap_or(0);
        video_clock - self.master_clock_us
    }

    /// Check if buffers are empty.
    pub fn is_empty(&self) -> bool {
        self.audio_buffer.is_empty() && self.video_buffer.is_empty()
    }

    /// Flush all buffers.
    pub fn flush(&mut self) -> (Vec<Sample>, Vec<Frame>) {
        let audio: Vec<_> = self.audio_buffer.drain(..).collect();
        let video: Vec<_> = self.video_buffer.drain(..).collect();
        (audio, video)
    }

    /// Reset the synchronizer.
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.video_buffer.clear();
        self.audio_timing = None;
        self.video_timing = None;
        self.master_clock_us = 0;
    }
}

/// Packet reordering buffer for handling out-of-order packets.
pub struct ReorderBuffer {
    packets: VecDeque<Packet<'static>>,
    max_size: usize,
    last_dts: Option<i64>,
}

impl ReorderBuffer {
    /// Create a new reorder buffer.
    pub fn new(max_size: usize) -> Self {
        Self {
            packets: VecDeque::with_capacity(max_size),
            max_size,
            last_dts: None,
        }
    }

    /// Push a packet into the buffer.
    pub fn push(&mut self, packet: Packet<'static>) {
        // Insert in DTS order
        let dts = if packet.dts.is_valid() {
            packet.dts.value
        } else if packet.pts.is_valid() {
            packet.pts.value
        } else {
            0
        };

        let pos = self
            .packets
            .iter()
            .position(|p| {
                let p_dts = if p.dts.is_valid() {
                    p.dts.value
                } else if p.pts.is_valid() {
                    p.pts.value
                } else {
                    0
                };
                p_dts > dts
            })
            .unwrap_or(self.packets.len());

        self.packets.insert(pos, packet);

        // Trim if too large
        while self.packets.len() > self.max_size {
            self.packets.pop_front();
        }
    }

    /// Pop the next packet in DTS order.
    pub fn pop(&mut self) -> Option<Packet<'static>> {
        if let Some(packet) = self.packets.pop_front() {
            self.last_dts = if packet.dts.is_valid() {
                Some(packet.dts.value)
            } else if packet.pts.is_valid() {
                Some(packet.pts.value)
            } else {
                None
            };
            Some(packet)
        } else {
            None
        }
    }

    /// Peek at the next packet.
    pub fn peek(&self) -> Option<&Packet<'static>> {
        self.packets.front()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Get number of packets in buffer.
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// Flush all packets.
    pub fn flush(&mut self) -> Vec<Packet<'static>> {
        self.packets.drain(..).collect()
    }
}
