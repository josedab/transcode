//! Pipeline integration tests.
//!
//! Tests the transcoding pipeline with mock components to verify
//! proper data flow and state management.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::packet::Packet;
use transcode_core::rational::Rational;
use transcode_core::sample::{ChannelLayout, Sample, SampleFormat};
use transcode_core::timestamp::TimeBase;
use transcode_pipeline::*;

// =============================================================================
// Mock Implementations
// =============================================================================

/// Mock demuxer for testing.
struct MockDemuxer {
    packets: Vec<Packet<'static>>,
    current_index: usize,
    num_streams: usize,
}

impl MockDemuxer {
    fn new(num_packets: usize, num_streams: usize) -> Self {
        let mut packets = Vec::new();
        for i in 0..num_packets {
            let data = vec![0u8; 100];
            let packet = Packet::new(data)
                .with_stream_index((i % num_streams) as u32)
                .into_owned();
            packets.push(packet);
        }
        Self {
            packets,
            current_index: 0,
            num_streams,
        }
    }
}

impl DemuxerWrapper for MockDemuxer {
    fn format_name(&self) -> &str {
        "mock"
    }

    fn duration(&self) -> Option<i64> {
        Some(1_000_000) // 1 second
    }

    fn num_streams(&self) -> usize {
        self.num_streams
    }

    fn stream_info(&self, index: usize) -> Option<PipelineStreamInfo> {
        if index < self.num_streams {
            Some(PipelineStreamInfo {
                index,
                track_type: if index == 0 {
                    PipelineTrackType::Video
                } else {
                    PipelineTrackType::Audio
                },
                time_base: Rational::new(1, 1000),
                duration: Some(1000),
                extra_data: None,
                width: if index == 0 { Some(320) } else { None },
                height: if index == 0 { Some(240) } else { None },
                sample_rate: if index == 1 { Some(44100) } else { None },
                channels: if index == 1 { Some(2) } else { None },
            })
        } else {
            None
        }
    }

    fn read_packet(&mut self) -> Result<Option<Packet<'static>>> {
        if self.current_index < self.packets.len() {
            let packet = self.packets[self.current_index].clone();
            self.current_index += 1;
            Ok(Some(packet))
        } else {
            Ok(None)
        }
    }

    fn seek(&mut self, _timestamp_us: i64) -> Result<()> {
        self.current_index = 0;
        Ok(())
    }
}

/// Mock muxer for testing.
struct MockMuxer {
    packets_written: Arc<AtomicUsize>,
    streams: Vec<PipelineStreamInfo>,
    header_written: bool,
    trailer_written: bool,
}

impl MockMuxer {
    fn new(packets_written: Arc<AtomicUsize>) -> Self {
        Self {
            packets_written,
            streams: Vec::new(),
            header_written: false,
            trailer_written: false,
        }
    }
}

impl MuxerWrapper for MockMuxer {
    fn format_name(&self) -> &str {
        "mock"
    }

    fn add_stream(&mut self, info: PipelineStreamInfo) -> Result<usize> {
        let index = self.streams.len();
        self.streams.push(info);
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, _packet: &Packet<'_>) -> Result<()> {
        self.packets_written.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        self.trailer_written = true;
        Ok(())
    }
}

/// Mock video decoder for testing.
struct MockVideoDecoder {
    frames_decoded: Arc<AtomicUsize>,
}

impl MockVideoDecoder {
    fn new(frames_decoded: Arc<AtomicUsize>) -> Self {
        Self { frames_decoded }
    }
}

impl VideoDecoderWrapper for MockVideoDecoder {
    fn decode(&mut self, _packet: &Packet<'_>) -> Result<Vec<Frame>> {
        self.frames_decoded.fetch_add(1, Ordering::SeqCst);
        let time_base = TimeBase::new(1, 1000);
        let frame = Frame::new(320, 240, PixelFormat::Yuv420p, time_base);
        Ok(vec![frame])
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(vec![])
    }
}

/// Mock audio decoder for testing.
struct MockAudioDecoder {
    samples_decoded: Arc<AtomicUsize>,
}

impl MockAudioDecoder {
    fn new(samples_decoded: Arc<AtomicUsize>) -> Self {
        Self { samples_decoded }
    }
}

impl AudioDecoderWrapper for MockAudioDecoder {
    fn decode(&mut self, _packet: &Packet<'_>) -> Result<Vec<Sample>> {
        self.samples_decoded.fetch_add(1, Ordering::SeqCst);
        let sample = Sample::new(1024, SampleFormat::F32, ChannelLayout::Stereo, 44100);
        Ok(vec![sample])
    }

    fn flush(&mut self) -> Result<Vec<Sample>> {
        Ok(vec![])
    }
}

/// Mock video encoder for testing.
struct MockVideoEncoder {
    packets_encoded: Arc<AtomicUsize>,
}

impl MockVideoEncoder {
    fn new(packets_encoded: Arc<AtomicUsize>) -> Self {
        Self { packets_encoded }
    }
}

impl VideoEncoderWrapper for MockVideoEncoder {
    fn encode(&mut self, _frame: &Frame) -> Result<Vec<Packet<'static>>> {
        self.packets_encoded.fetch_add(1, Ordering::SeqCst);
        let data = vec![0u8; 50];
        let packet = Packet::new(data).into_owned();
        Ok(vec![packet])
    }

    fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        Ok(vec![])
    }
}

/// Mock audio encoder for testing.
struct MockAudioEncoder {
    packets_encoded: Arc<AtomicUsize>,
}

impl MockAudioEncoder {
    fn new(packets_encoded: Arc<AtomicUsize>) -> Self {
        Self { packets_encoded }
    }
}

impl AudioEncoderWrapper for MockAudioEncoder {
    fn encode(&mut self, _sample: &Sample) -> Result<Vec<Packet<'static>>> {
        self.packets_encoded.fetch_add(1, Ordering::SeqCst);
        let data = vec![0u8; 50];
        let packet = Packet::new(data).into_owned();
        Ok(vec![packet])
    }

    fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        Ok(vec![])
    }
}

// =============================================================================
// Pipeline State Tests
// =============================================================================

#[test]
fn test_pipeline_initial_state() {
    let pipeline = Pipeline::default();
    assert_eq!(pipeline.state(), PipelineState::Created);
}

#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::default();
    assert_eq!(config.max_buffer_size, 64);
    assert!(config.report_progress);
    assert_eq!(config.progress_interval, 100);
}

#[test]
fn test_pipeline_builder_creates_pipeline() {
    let result = PipelineBuilder::new().build();
    assert!(result.is_ok());
    let pipeline = result.unwrap();
    assert_eq!(pipeline.state(), PipelineState::Created);
}

#[test]
fn test_pipeline_without_demuxer_fails() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let muxer = MockMuxer::new(packets_written);

    let mut pipeline = PipelineBuilder::new()
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    let result = pipeline.initialize();
    assert!(result.is_err());
}

#[test]
fn test_pipeline_without_muxer_fails() {
    let demuxer = MockDemuxer::new(10, 1);

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .build()
        .unwrap();

    let result = pipeline.initialize();
    assert!(result.is_err());
}

// =============================================================================
// Remux Pipeline Tests (No Decode/Encode)
// =============================================================================

#[test]
fn test_remux_pipeline_runs_to_completion() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let demuxer = MockDemuxer::new(10, 1);
    let muxer = MockMuxer::new(packets_written.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);
    // In remux mode, packets pass through directly
    assert_eq!(packets_written.load(Ordering::SeqCst), 10);
}

#[test]
fn test_remux_pipeline_with_multiple_streams() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let demuxer = MockDemuxer::new(20, 2); // 10 video, 10 audio
    let muxer = MockMuxer::new(packets_written.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);
    assert_eq!(packets_written.load(Ordering::SeqCst), 20);
}

#[test]
fn test_pipeline_step_by_step() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let demuxer = MockDemuxer::new(5, 1);
    let muxer = MockMuxer::new(packets_written.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    pipeline.initialize().unwrap();
    assert_eq!(pipeline.state(), PipelineState::Ready);

    // Process each packet one at a time
    for i in 0..5 {
        let has_more = pipeline.step().unwrap();
        assert!(has_more, "Should have more data at step {}", i);
        assert_eq!(pipeline.state(), PipelineState::Running);
    }

    // Final step should return false (end of stream)
    let has_more = pipeline.step().unwrap();
    assert!(!has_more);
    assert_eq!(pipeline.state(), PipelineState::Finished);
}

// =============================================================================
// Transcode Pipeline Tests (With Decode/Encode)
// =============================================================================

#[test]
fn test_transcode_pipeline_video_only() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let frames_decoded = Arc::new(AtomicUsize::new(0));
    let packets_encoded = Arc::new(AtomicUsize::new(0));

    let demuxer = MockDemuxer::new(10, 1);
    let muxer = MockMuxer::new(packets_written.clone());
    let decoder = MockVideoDecoder::new(frames_decoded.clone());
    let encoder = MockVideoEncoder::new(packets_encoded.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .decoder(0, DecoderNode::new_video("h264_decoder", decoder, 0))
        .encoder(0, EncoderNode::new_video("h264_encoder", encoder, 0))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);

    // Verify data flow
    assert_eq!(frames_decoded.load(Ordering::SeqCst), 10);
    assert_eq!(packets_encoded.load(Ordering::SeqCst), 10);
    assert_eq!(packets_written.load(Ordering::SeqCst), 10);
}

#[test]
fn test_transcode_pipeline_audio_only() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let samples_decoded = Arc::new(AtomicUsize::new(0));
    let packets_encoded = Arc::new(AtomicUsize::new(0));

    // Demuxer with only audio stream (index 0)
    let mut demuxer = MockDemuxer::new(10, 1);
    // Force all packets to have stream index 0
    for packet in &mut demuxer.packets {
        *packet = Packet::new(vec![0u8; 100])
            .with_stream_index(0)
            .into_owned();
    }

    let muxer = MockMuxer::new(packets_written.clone());
    let decoder = MockAudioDecoder::new(samples_decoded.clone());
    let encoder = MockAudioEncoder::new(packets_encoded.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .decoder(0, DecoderNode::new_audio("aac_decoder", decoder, 0))
        .encoder(0, EncoderNode::new_audio("aac_encoder", encoder, 0))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);

    assert_eq!(samples_decoded.load(Ordering::SeqCst), 10);
    assert_eq!(packets_encoded.load(Ordering::SeqCst), 10);
    assert_eq!(packets_written.load(Ordering::SeqCst), 10);
}

#[test]
fn test_transcode_pipeline_mixed_av() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let frames_decoded = Arc::new(AtomicUsize::new(0));
    let samples_decoded = Arc::new(AtomicUsize::new(0));
    let video_encoded = Arc::new(AtomicUsize::new(0));
    let audio_encoded = Arc::new(AtomicUsize::new(0));

    let demuxer = MockDemuxer::new(20, 2); // 10 video (idx 0), 10 audio (idx 1)
    let muxer = MockMuxer::new(packets_written.clone());

    let video_decoder = MockVideoDecoder::new(frames_decoded.clone());
    let audio_decoder = MockAudioDecoder::new(samples_decoded.clone());
    let video_encoder = MockVideoEncoder::new(video_encoded.clone());
    let audio_encoder = MockAudioEncoder::new(audio_encoded.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .decoder(0, DecoderNode::new_video("h264_decoder", video_decoder, 0))
        .decoder(1, DecoderNode::new_audio("aac_decoder", audio_decoder, 1))
        .encoder(0, EncoderNode::new_video("h264_encoder", video_encoder, 0))
        .encoder(1, EncoderNode::new_audio("aac_encoder", audio_encoder, 1))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);

    // Verify correct data flow to each codec
    assert_eq!(frames_decoded.load(Ordering::SeqCst), 10);
    assert_eq!(samples_decoded.load(Ordering::SeqCst), 10);
    assert_eq!(video_encoded.load(Ordering::SeqCst), 10);
    assert_eq!(audio_encoded.load(Ordering::SeqCst), 10);
    assert_eq!(packets_written.load(Ordering::SeqCst), 20);
}

// =============================================================================
// Pipeline Metrics Tests
// =============================================================================

#[test]
fn test_pipeline_packets_processed_count() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let demuxer = MockDemuxer::new(15, 1);
    let muxer = MockMuxer::new(packets_written);

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    pipeline.run().unwrap();

    assert_eq!(pipeline.packets_processed(), 15);
}

#[test]
fn test_pipeline_empty_input() {
    let packets_written = Arc::new(AtomicUsize::new(0));
    let demuxer = MockDemuxer::new(0, 1);
    let muxer = MockMuxer::new(packets_written.clone());

    let mut pipeline = PipelineBuilder::new()
        .demuxer(DemuxerNode::new("input", demuxer))
        .muxer(MuxerNode::new("output", muxer))
        .build()
        .unwrap();

    let result = pipeline.run();
    assert!(result.is_ok());
    assert_eq!(pipeline.state(), PipelineState::Finished);
    assert_eq!(pipeline.packets_processed(), 0);
    assert_eq!(packets_written.load(Ordering::SeqCst), 0);
}

// =============================================================================
// Node Tests
// =============================================================================

#[test]
fn test_node_id_uniqueness() {
    let id1 = NodeId::new();
    let id2 = NodeId::new();
    let id3 = NodeId::new();

    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
}

#[test]
fn test_node_id_display() {
    let id = NodeId::new();
    let display = format!("{}", id);
    assert!(display.starts_with("Node("));
    assert!(display.ends_with(")"));
}

#[test]
fn test_demuxer_node_properties() {
    let demuxer = MockDemuxer::new(5, 2);
    let node = DemuxerNode::new("test_demuxer", demuxer);

    assert_eq!(node.name(), "test_demuxer");
    assert_eq!(node.num_streams(), 2);
    assert_eq!(node.duration(), Some(1_000_000));
    assert!(!node.is_finished());

    // Check stream info
    let video_info = node.stream_info(0).unwrap();
    assert_eq!(video_info.track_type, PipelineTrackType::Video);
    assert_eq!(video_info.width, Some(320));
    assert_eq!(video_info.height, Some(240));

    let audio_info = node.stream_info(1).unwrap();
    assert_eq!(audio_info.track_type, PipelineTrackType::Audio);
    assert_eq!(audio_info.sample_rate, Some(44100));
    assert_eq!(audio_info.channels, Some(2));
}

#[test]
fn test_demuxer_node_process_packets() {
    let demuxer = MockDemuxer::new(3, 1);
    let mut node = DemuxerNode::new("test", demuxer);

    // Process packets
    let output1 = node.process().unwrap();
    assert!(matches!(output1, NodeOutput::Packet(_)));

    let output2 = node.process().unwrap();
    assert!(matches!(output2, NodeOutput::Packet(_)));

    let output3 = node.process().unwrap();
    assert!(matches!(output3, NodeOutput::Packet(_)));

    // End of stream
    let output4 = node.process().unwrap();
    assert!(matches!(output4, NodeOutput::EndOfStream));
    assert!(node.is_finished());
}

#[test]
fn test_decoder_node_stream_index() {
    let frames_decoded = Arc::new(AtomicUsize::new(0));
    let decoder = MockVideoDecoder::new(frames_decoded);
    let node = DecoderNode::new_video("h264", decoder, 5);

    assert_eq!(node.stream_index(), 5);
    assert_eq!(node.name(), "h264");
}

#[test]
fn test_encoder_node_stream_index() {
    let packets_encoded = Arc::new(AtomicUsize::new(0));
    let encoder = MockVideoEncoder::new(packets_encoded);
    let node = EncoderNode::new_video("h264", encoder, 3);

    assert_eq!(node.stream_index(), 3);
    assert_eq!(node.name(), "h264");
}

// =============================================================================
// Filter Chain Tests
// =============================================================================

#[test]
fn test_null_video_filter() {
    let time_base = TimeBase::new(1, 1000);
    let frame = Frame::new(320, 240, PixelFormat::Yuv420p, time_base);

    let mut filter = NullVideoFilter::new();
    let result = filter.process(frame.clone());
    assert!(result.is_ok());

    let filtered = result.unwrap();
    assert_eq!(filtered.width(), frame.width());
    assert_eq!(filtered.height(), frame.height());
}

#[test]
fn test_null_audio_filter() {
    let sample = Sample::new(1024, SampleFormat::F32, ChannelLayout::Stereo, 44100);

    let mut filter = NullAudioFilter::new();
    let result = filter.process(sample.clone());
    assert!(result.is_ok());

    let filtered = result.unwrap();
    assert_eq!(filtered.num_samples(), sample.num_samples());
    assert_eq!(filtered.sample_rate(), sample.sample_rate());
}

#[test]
fn test_filter_chain_empty() {
    let mut chain: FilterChain<dyn VideoFilter> = FilterChain::new();

    let time_base = TimeBase::new(1, 1000);
    let frame = Frame::new(320, 240, PixelFormat::Yuv420p, time_base);
    let result = chain.process(frame.clone());
    assert!(result.is_ok());
}

// =============================================================================
// Synchronization Tests
// =============================================================================

#[test]
fn test_sync_config_default() {
    let config = SyncConfig::default();
    assert_eq!(config.mode, SyncMode::AudioMaster);
}

#[test]
fn test_synchronizer_creation() {
    let config = SyncConfig::default();
    let sync = Synchronizer::new(config);

    // Just verify it creates without panicking
    assert!(true, "Synchronizer created successfully");

    // Can set streams
    let mut sync = sync;
    sync.set_video_stream(Rational::new(1, 30));
    sync.set_audio_stream(Rational::new(1, 44100));
}

// =============================================================================
// Pipeline Stream Info Tests
// =============================================================================

#[test]
fn test_pipeline_stream_info_default() {
    let info = PipelineStreamInfo::default();

    assert_eq!(info.index, 0);
    assert_eq!(info.track_type, PipelineTrackType::Unknown);
    assert_eq!(info.time_base, Rational::new(1, 1000));
    assert!(info.duration.is_none());
    assert!(info.extra_data.is_none());
    assert!(info.width.is_none());
    assert!(info.height.is_none());
    assert!(info.sample_rate.is_none());
    assert!(info.channels.is_none());
}

#[test]
fn test_pipeline_stream_info_video() {
    let info = PipelineStreamInfo {
        index: 0,
        track_type: PipelineTrackType::Video,
        time_base: Rational::new(1, 90000),
        duration: Some(90000 * 60), // 1 minute
        extra_data: Some(vec![0x00, 0x00, 0x01]),
        width: Some(1920),
        height: Some(1080),
        sample_rate: None,
        channels: None,
    };

    assert_eq!(info.track_type, PipelineTrackType::Video);
    assert_eq!(info.width, Some(1920));
    assert_eq!(info.height, Some(1080));
}

#[test]
fn test_pipeline_stream_info_audio() {
    let info = PipelineStreamInfo {
        index: 1,
        track_type: PipelineTrackType::Audio,
        time_base: Rational::new(1, 48000),
        duration: Some(48000 * 60), // 1 minute at 48kHz
        extra_data: None,
        width: None,
        height: None,
        sample_rate: Some(48000),
        channels: Some(6), // 5.1 surround
    };

    assert_eq!(info.track_type, PipelineTrackType::Audio);
    assert_eq!(info.sample_rate, Some(48000));
    assert_eq!(info.channels, Some(6));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_pipeline_config_custom() {
    let config = PipelineConfig {
        sync: SyncConfig::default(),
        max_buffer_size: 128,
        report_progress: false,
        progress_interval: 50,
    };

    assert_eq!(config.max_buffer_size, 128);
    assert!(!config.report_progress);
    assert_eq!(config.progress_interval, 50);
}

#[test]
fn test_reorder_buffer_creation() {
    let buffer = ReorderBuffer::new(32);
    // Just verify it creates without panicking
    assert!(true, "ReorderBuffer created successfully");
    drop(buffer); // Explicitly drop to use the variable
}
