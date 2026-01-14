//! Main transcoding pipeline implementation.

use crate::error::PipelineTrackType;
use crate::filter::{AudioFilter, FilterChain, VideoFilter};
use crate::node::{DecoderNode, DemuxerNode, EncoderNode, MuxerNode, Node, NodeId, NodeOutput};
use crate::sync::{SyncConfig, Synchronizer};
use crate::{PipelineError, Result};
use std::collections::HashMap;
use transcode_core::packet::Packet;
use tracing::{debug, info, trace};

/// Pipeline state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PipelineState {
    /// Pipeline is created but not initialized.
    Created,
    /// Pipeline is initialized and ready to run.
    Ready,
    /// Pipeline is running.
    Running,
    /// Pipeline is paused.
    Paused,
    /// Pipeline has finished.
    Finished,
    /// Pipeline encountered an error.
    Error,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Synchronization config.
    pub sync: SyncConfig,
    /// Maximum packets to buffer per stream.
    pub max_buffer_size: usize,
    /// Enable progress reporting.
    pub report_progress: bool,
    /// Progress reporting interval (packets).
    pub progress_interval: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sync: SyncConfig::default(),
            max_buffer_size: 64,
            report_progress: true,
            progress_interval: 100,
        }
    }
}

/// Stream mapping configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StreamMapping {
    /// Source stream index.
    source_index: usize,
    /// Destination stream index.
    dest_index: usize,
    /// Decoder node ID.
    decoder_id: Option<NodeId>,
    /// Encoder node ID.
    encoder_id: Option<NodeId>,
    /// Track type.
    track_type: PipelineTrackType,
}

/// Transcoding pipeline.
pub struct Pipeline {
    /// Configuration.
    config: PipelineConfig,
    /// Current state.
    state: PipelineState,
    /// Demuxer node.
    demuxer: Option<DemuxerNode>,
    /// Muxer node.
    muxer: Option<MuxerNode>,
    /// Decoder nodes by stream index.
    decoders: HashMap<usize, DecoderNode>,
    /// Encoder nodes by stream index.
    encoders: HashMap<usize, EncoderNode>,
    /// Stream mappings.
    stream_mappings: Vec<StreamMapping>,
    /// Video filter chain.
    video_filters: FilterChain<dyn VideoFilter>,
    /// Audio filter chain.
    audio_filters: FilterChain<dyn AudioFilter>,
    /// Synchronizer.
    synchronizer: Synchronizer,
    /// Packets processed.
    packets_processed: u64,
    /// Duration processed (microseconds).
    duration_processed: i64,
    /// Total duration (microseconds).
    total_duration: Option<i64>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            synchronizer: Synchronizer::new(config.sync.clone()),
            config,
            state: PipelineState::Created,
            demuxer: None,
            muxer: None,
            decoders: HashMap::new(),
            encoders: HashMap::new(),
            stream_mappings: Vec::new(),
            video_filters: FilterChain::new(),
            audio_filters: FilterChain::new(),
            packets_processed: 0,
            duration_processed: 0,
            total_duration: None,
        }
    }

    /// Get current state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Set the demuxer.
    pub fn set_demuxer(&mut self, demuxer: DemuxerNode) {
        self.total_duration = demuxer.duration();
        self.demuxer = Some(demuxer);
    }

    /// Set the muxer.
    pub fn set_muxer(&mut self, muxer: MuxerNode) {
        self.muxer = Some(muxer);
    }

    /// Add a decoder for a stream.
    pub fn add_decoder(&mut self, stream_index: usize, decoder: DecoderNode) {
        self.decoders.insert(stream_index, decoder);
    }

    /// Add an encoder for a stream.
    pub fn add_encoder(&mut self, stream_index: usize, encoder: EncoderNode) {
        self.encoders.insert(stream_index, encoder);
    }

    /// Add a video filter.
    pub fn add_video_filter(&mut self, filter: Box<dyn VideoFilter>) {
        self.video_filters.add(filter);
    }

    /// Add an audio filter.
    pub fn add_audio_filter(&mut self, filter: Box<dyn AudioFilter>) {
        self.audio_filters.add(filter);
    }

    /// Map a source stream to a destination stream.
    pub fn map_stream(&mut self, source_index: usize, dest_index: usize, track_type: PipelineTrackType) {
        let decoder_id = self.decoders.get(&source_index).map(|_| NodeId::new());
        let encoder_id = self.encoders.get(&dest_index).map(|_| NodeId::new());

        self.stream_mappings.push(StreamMapping {
            source_index,
            dest_index,
            decoder_id,
            encoder_id,
            track_type,
        });
    }

    /// Initialize the pipeline.
    pub fn initialize(&mut self) -> Result<()> {
        let demuxer = self
            .demuxer
            .as_ref()
            .ok_or_else(|| PipelineError::InvalidConfig("No demuxer configured".into()))?;

        if self.muxer.is_none() {
            return Err(PipelineError::InvalidConfig("No muxer configured".into()));
        }

        // Setup stream mappings if not already configured
        if self.stream_mappings.is_empty() {
            for i in 0..demuxer.num_streams() {
                if let Some(info) = demuxer.stream_info(i) {
                    self.stream_mappings.push(StreamMapping {
                        source_index: i,
                        dest_index: i,
                        decoder_id: self.decoders.get(&i).map(|_| NodeId::new()),
                        encoder_id: self.encoders.get(&i).map(|_| NodeId::new()),
                        track_type: info.track_type,
                    });
                }
            }
        }

        // Configure synchronizer
        for mapping in &self.stream_mappings {
            if let Some(info) = demuxer.stream_info(mapping.source_index) {
                match mapping.track_type {
                    PipelineTrackType::Video => {
                        self.synchronizer.set_video_stream(info.time_base);
                    }
                    PipelineTrackType::Audio => {
                        self.synchronizer.set_audio_stream(info.time_base);
                    }
                    _ => {}
                }
            }
        }

        self.state = PipelineState::Ready;
        info!("Pipeline initialized with {} stream mappings", self.stream_mappings.len());

        Ok(())
    }

    /// Run one step of the pipeline.
    pub fn step(&mut self) -> Result<bool> {
        if self.state != PipelineState::Ready && self.state != PipelineState::Running {
            return Err(PipelineError::NotInitialized);
        }

        self.state = PipelineState::Running;

        // Read next packet from demuxer
        let demuxer = self.demuxer.as_mut().ok_or(PipelineError::NotInitialized)?;

        let output = demuxer.process()?;

        match output {
            NodeOutput::Packet(packet) => {
                self.process_packet(&packet)?;
                self.packets_processed += 1;

                if self.config.report_progress
                    && self.packets_processed.is_multiple_of(self.config.progress_interval as u64)
                {
                    self.report_progress();
                }

                Ok(true)
            }
            NodeOutput::EndOfStream => {
                self.flush()?;
                self.state = PipelineState::Finished;
                info!("Pipeline finished, processed {} packets", self.packets_processed);
                Ok(false)
            }
            _ => Ok(true),
        }
    }

    /// Process a packet through the pipeline.
    fn process_packet(&mut self, packet: &Packet<'_>) -> Result<()> {
        let stream_index = packet.stream_index as usize;
        trace!("Processing packet for stream {}", stream_index);

        // Find mapping for this stream
        let mapping = self
            .stream_mappings
            .iter()
            .find(|m| m.source_index == stream_index)
            .cloned();

        let Some(mapping) = mapping else {
            debug!("No mapping for stream {}, skipping packet", stream_index);
            return Ok(());
        };

        // Check if we need to decode
        if let Some(decoder) = self.decoders.get_mut(&stream_index) {
            // Decode the packet
            let decode_outputs = decoder.decode_packet(packet)?;

            for decode_output in decode_outputs {
                match decode_output {
                    NodeOutput::VideoFrame(frame) => {
                        // Apply video filters
                        let filtered = self.video_filters.process(frame)?;

                        // Encode if we have an encoder
                        if let Some(encoder) = self.encoders.get_mut(&mapping.dest_index) {
                            let encode_outputs = encoder.encode_frame(&filtered)?;

                            for encode_output in encode_outputs {
                                if let NodeOutput::Packet(encoded_packet) = encode_output {
                                    self.write_packet(&encoded_packet)?;
                                }
                            }
                        }
                    }
                    NodeOutput::AudioSamples(sample) => {
                        // Apply audio filters
                        let filtered = self.audio_filters.process(sample)?;

                        // Encode if we have an encoder
                        if let Some(encoder) = self.encoders.get_mut(&mapping.dest_index) {
                            let encode_outputs = encoder.encode_samples(&filtered)?;

                            for encode_output in encode_outputs {
                                if let NodeOutput::Packet(encoded_packet) = encode_output {
                                    self.write_packet(&encoded_packet)?;
                                }
                            }
                        }
                    }
                    NodeOutput::None => {}
                    _ => {}
                }
            }
        } else {
            // No decoder - pass through (remux mode)
            self.write_packet(packet)?;
        }

        Ok(())
    }

    /// Write a packet to the muxer.
    fn write_packet(&mut self, packet: &Packet<'_>) -> Result<()> {
        if let Some(muxer) = self.muxer.as_mut() {
            muxer.write_packet(packet)?;

            // Update duration processed
            if packet.pts.is_valid() {
                self.duration_processed = self.duration_processed.max(packet.pts.value);
            }
        }
        Ok(())
    }

    /// Flush the pipeline.
    fn flush(&mut self) -> Result<()> {
        info!("Flushing pipeline");

        // Collect decoder outputs first
        let mut decoder_outputs: Vec<(usize, Vec<NodeOutput>)> = Vec::new();
        for (stream_index, decoder) in &mut self.decoders {
            let outputs = decoder.flush()?;
            decoder_outputs.push((*stream_index, outputs));
        }

        // Process decoder outputs
        for (stream_index, outputs) in decoder_outputs {
            for output in outputs {
                match output {
                    NodeOutput::VideoFrame(frame) => {
                        if let Some(encoder) = self.encoders.get_mut(&stream_index) {
                            let filtered = self.video_filters.process(frame)?;
                            let encode_outputs = encoder.encode_frame(&filtered)?;

                            for encode_output in encode_outputs {
                                if let NodeOutput::Packet(packet) = encode_output {
                                    self.write_packet(&packet)?;
                                }
                            }
                        }
                    }
                    NodeOutput::AudioSamples(sample) => {
                        if let Some(encoder) = self.encoders.get_mut(&stream_index) {
                            let filtered = self.audio_filters.process(sample)?;
                            let encode_outputs = encoder.encode_samples(&filtered)?;

                            for encode_output in encode_outputs {
                                if let NodeOutput::Packet(packet) = encode_output {
                                    self.write_packet(&packet)?;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Collect encoder outputs first
        let mut encoder_outputs: Vec<Vec<NodeOutput>> = Vec::new();
        for encoder in self.encoders.values_mut() {
            let outputs = encoder.flush()?;
            encoder_outputs.push(outputs);
        }

        // Write encoder outputs
        for outputs in encoder_outputs {
            for output in outputs {
                if let NodeOutput::Packet(packet) = output {
                    self.write_packet(&packet)?;
                }
            }
        }

        // Flush video filters
        for frame in self.video_filters.flush()? {
            // Process remaining filtered frames through any available encoder
            if let Some(encoder) = self.encoders.values_mut().next() {
                let encode_outputs = encoder.encode_frame(&frame)?;
                for encode_output in encode_outputs {
                    if let NodeOutput::Packet(packet) = encode_output {
                        self.write_packet(&packet)?;
                    }
                }
            }
        }

        // Flush audio filters
        for sample in self.audio_filters.flush()? {
            if let Some(encoder) = self.encoders.values_mut().next() {
                let encode_outputs = encoder.encode_samples(&sample)?;
                for encode_output in encode_outputs {
                    if let NodeOutput::Packet(packet) = encode_output {
                        self.write_packet(&packet)?;
                    }
                }
            }
        }

        // Finalize muxer
        if let Some(muxer) = self.muxer.as_mut() {
            muxer.finalize()?;
        }

        Ok(())
    }

    /// Report progress.
    fn report_progress(&self) {
        if let Some(total) = self.total_duration {
            if total > 0 {
                let progress = (self.duration_processed as f64 / total as f64 * 100.0).min(100.0);
                info!(
                    "Progress: {:.1}% ({} packets processed)",
                    progress, self.packets_processed
                );
            }
        } else {
            info!("Processed {} packets", self.packets_processed);
        }
    }

    /// Run the pipeline to completion.
    pub fn run(&mut self) -> Result<()> {
        self.initialize()?;

        while self.step()? {}

        Ok(())
    }

    /// Get packets processed count.
    pub fn packets_processed(&self) -> u64 {
        self.packets_processed
    }

    /// Get duration processed in microseconds.
    pub fn duration_processed(&self) -> i64 {
        self.duration_processed
    }
}

/// Builder for constructing pipelines.
pub struct PipelineBuilder {
    config: PipelineConfig,
    demuxer: Option<DemuxerNode>,
    muxer: Option<MuxerNode>,
    decoders: HashMap<usize, DecoderNode>,
    encoders: HashMap<usize, EncoderNode>,
    video_filters: Vec<Box<dyn VideoFilter>>,
    audio_filters: Vec<Box<dyn AudioFilter>>,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            demuxer: None,
            muxer: None,
            decoders: HashMap::new(),
            encoders: HashMap::new(),
            video_filters: Vec::new(),
            audio_filters: Vec::new(),
        }
    }

    /// Set configuration.
    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Set demuxer.
    pub fn demuxer(mut self, demuxer: DemuxerNode) -> Self {
        self.demuxer = Some(demuxer);
        self
    }

    /// Set muxer.
    pub fn muxer(mut self, muxer: MuxerNode) -> Self {
        self.muxer = Some(muxer);
        self
    }

    /// Add a decoder.
    pub fn decoder(mut self, stream_index: usize, decoder: DecoderNode) -> Self {
        self.decoders.insert(stream_index, decoder);
        self
    }

    /// Add an encoder.
    pub fn encoder(mut self, stream_index: usize, encoder: EncoderNode) -> Self {
        self.encoders.insert(stream_index, encoder);
        self
    }

    /// Add a video filter.
    pub fn video_filter(mut self, filter: Box<dyn VideoFilter>) -> Self {
        self.video_filters.push(filter);
        self
    }

    /// Add an audio filter.
    pub fn audio_filter(mut self, filter: Box<dyn AudioFilter>) -> Self {
        self.audio_filters.push(filter);
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> Result<Pipeline> {
        let mut pipeline = Pipeline::new(self.config);

        if let Some(demuxer) = self.demuxer {
            pipeline.set_demuxer(demuxer);
        }

        if let Some(muxer) = self.muxer {
            pipeline.set_muxer(muxer);
        }

        for (index, decoder) in self.decoders {
            pipeline.add_decoder(index, decoder);
        }

        for (index, encoder) in self.encoders {
            pipeline.add_encoder(index, encoder);
        }

        for filter in self.video_filters {
            pipeline.add_video_filter(filter);
        }

        for filter in self.audio_filters {
            pipeline.add_audio_filter(filter);
        }

        Ok(pipeline)
    }
}
