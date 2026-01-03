//! Pipeline node abstractions.

use crate::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use transcode_core::frame::Frame;
use transcode_core::packet::Packet;
use transcode_core::sample::Sample;
use transcode_core::rational::Rational;

/// Unique node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Create a new unique node ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the raw ID value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Pipeline node trait.
pub trait Node: Send {
    /// Get node ID.
    fn id(&self) -> NodeId;

    /// Get node name.
    fn name(&self) -> &str;

    /// Process one step.
    fn process(&mut self) -> Result<NodeOutput>;

    /// Check if node has finished.
    fn is_finished(&self) -> bool;

    /// Flush any buffered data.
    fn flush(&mut self) -> Result<Vec<NodeOutput>>;
}

/// Output from a node.
#[derive(Debug)]
pub enum NodeOutput {
    /// No output available yet.
    None,
    /// Packet output.
    Packet(Packet<'static>),
    /// Video frame output.
    VideoFrame(Frame),
    /// Audio samples output.
    AudioSamples(Sample),
    /// End of stream.
    EndOfStream,
}

/// Stream information for pipeline streams.
#[derive(Debug, Clone)]
pub struct PipelineStreamInfo {
    /// Stream index.
    pub index: usize,
    /// Track type.
    pub track_type: crate::error::PipelineTrackType,
    /// Time base.
    pub time_base: Rational,
    /// Duration in time base units.
    pub duration: Option<i64>,
    /// Codec-specific extra data.
    pub extra_data: Option<Vec<u8>>,
    /// Width (for video).
    pub width: Option<u32>,
    /// Height (for video).
    pub height: Option<u32>,
    /// Sample rate (for audio).
    pub sample_rate: Option<u32>,
    /// Channels (for audio).
    pub channels: Option<u8>,
}

impl Default for PipelineStreamInfo {
    fn default() -> Self {
        Self {
            index: 0,
            track_type: crate::error::PipelineTrackType::Unknown,
            time_base: Rational::new(1, 1000),
            duration: None,
            extra_data: None,
            width: None,
            height: None,
            sample_rate: None,
            channels: None,
        }
    }
}

/// Demuxer wrapper trait for pipeline integration.
pub trait DemuxerWrapper: Send {
    /// Get format name.
    fn format_name(&self) -> &str;
    /// Get duration in microseconds.
    fn duration(&self) -> Option<i64>;
    /// Get number of streams.
    fn num_streams(&self) -> usize;
    /// Get stream info.
    fn stream_info(&self, index: usize) -> Option<PipelineStreamInfo>;
    /// Read next packet.
    fn read_packet(&mut self) -> Result<Option<Packet<'static>>>;
    /// Seek to timestamp.
    fn seek(&mut self, timestamp_us: i64) -> Result<()>;
}

/// Demuxer node wrapping a container demuxer.
pub struct DemuxerNode {
    id: NodeId,
    name: String,
    demuxer: Box<dyn DemuxerWrapper>,
    finished: bool,
}

impl DemuxerNode {
    /// Create a new demuxer node.
    pub fn new<D: DemuxerWrapper + 'static>(name: impl Into<String>, demuxer: D) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            demuxer: Box::new(demuxer),
            finished: false,
        }
    }

    /// Get stream information.
    pub fn stream_info(&self, index: usize) -> Option<PipelineStreamInfo> {
        self.demuxer.stream_info(index)
    }

    /// Get number of streams.
    pub fn num_streams(&self) -> usize {
        self.demuxer.num_streams()
    }

    /// Get duration.
    pub fn duration(&self) -> Option<i64> {
        self.demuxer.duration()
    }

    /// Seek to timestamp.
    pub fn seek(&mut self, timestamp: i64) -> Result<()> {
        self.demuxer.seek(timestamp)
    }
}

impl Node for DemuxerNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self) -> Result<NodeOutput> {
        if self.finished {
            return Ok(NodeOutput::EndOfStream);
        }

        match self.demuxer.read_packet()? {
            Some(packet) => Ok(NodeOutput::Packet(packet)),
            None => {
                self.finished = true;
                Ok(NodeOutput::EndOfStream)
            }
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn flush(&mut self) -> Result<Vec<NodeOutput>> {
        Ok(vec![NodeOutput::EndOfStream])
    }
}

/// Video decoder wrapper trait.
pub trait VideoDecoderWrapper: Send {
    /// Decode a packet into frames.
    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Frame>>;
    /// Flush the decoder.
    fn flush(&mut self) -> Result<Vec<Frame>>;
}

/// Audio decoder wrapper trait.
pub trait AudioDecoderWrapper: Send {
    /// Decode a packet into samples.
    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Sample>>;
    /// Flush the decoder.
    fn flush(&mut self) -> Result<Vec<Sample>>;
}

/// Decoder node.
pub struct DecoderNode {
    id: NodeId,
    name: String,
    video_decoder: Option<Box<dyn VideoDecoderWrapper>>,
    audio_decoder: Option<Box<dyn AudioDecoderWrapper>>,
    stream_index: usize,
    finished: bool,
    /// Buffered output frames/samples.
    video_buffer: Vec<Frame>,
    audio_buffer: Vec<Sample>,
}

impl DecoderNode {
    /// Create a new video decoder node.
    pub fn new_video<D: VideoDecoderWrapper + 'static>(
        name: impl Into<String>,
        decoder: D,
        stream_index: usize,
    ) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            video_decoder: Some(Box::new(decoder)),
            audio_decoder: None,
            stream_index,
            finished: false,
            video_buffer: Vec::new(),
            audio_buffer: Vec::new(),
        }
    }

    /// Create a new audio decoder node.
    pub fn new_audio<D: AudioDecoderWrapper + 'static>(
        name: impl Into<String>,
        decoder: D,
        stream_index: usize,
    ) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            video_decoder: None,
            audio_decoder: Some(Box::new(decoder)),
            stream_index,
            finished: false,
            video_buffer: Vec::new(),
            audio_buffer: Vec::new(),
        }
    }

    /// Get stream index.
    pub fn stream_index(&self) -> usize {
        self.stream_index
    }

    /// Decode a packet, returning decoded outputs.
    pub fn decode_packet(&mut self, packet: &Packet<'_>) -> Result<Vec<NodeOutput>> {
        let mut outputs = Vec::new();

        if let Some(ref mut decoder) = self.video_decoder {
            let frames = decoder.decode(packet)?;
            for frame in frames {
                outputs.push(NodeOutput::VideoFrame(frame));
            }
        }

        if let Some(ref mut decoder) = self.audio_decoder {
            let samples = decoder.decode(packet)?;
            for sample in samples {
                outputs.push(NodeOutput::AudioSamples(sample));
            }
        }

        Ok(outputs)
    }
}

impl Node for DecoderNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self) -> Result<NodeOutput> {
        // Return buffered output if available
        if let Some(frame) = self.video_buffer.pop() {
            return Ok(NodeOutput::VideoFrame(frame));
        }
        if let Some(sample) = self.audio_buffer.pop() {
            return Ok(NodeOutput::AudioSamples(sample));
        }
        Ok(NodeOutput::None)
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn flush(&mut self) -> Result<Vec<NodeOutput>> {
        let mut outputs = Vec::new();

        if let Some(ref mut decoder) = self.video_decoder {
            for frame in decoder.flush()? {
                outputs.push(NodeOutput::VideoFrame(frame));
            }
        }

        if let Some(ref mut decoder) = self.audio_decoder {
            for sample in decoder.flush()? {
                outputs.push(NodeOutput::AudioSamples(sample));
            }
        }

        self.finished = true;
        outputs.push(NodeOutput::EndOfStream);
        Ok(outputs)
    }
}

/// Video encoder wrapper trait.
pub trait VideoEncoderWrapper: Send {
    /// Encode a frame.
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet<'static>>>;
    /// Flush the encoder.
    fn flush(&mut self) -> Result<Vec<Packet<'static>>>;
}

/// Audio encoder wrapper trait.
pub trait AudioEncoderWrapper: Send {
    /// Encode samples.
    fn encode(&mut self, sample: &Sample) -> Result<Vec<Packet<'static>>>;
    /// Flush the encoder.
    fn flush(&mut self) -> Result<Vec<Packet<'static>>>;
}

/// Encoder node.
pub struct EncoderNode {
    id: NodeId,
    name: String,
    video_encoder: Option<Box<dyn VideoEncoderWrapper>>,
    audio_encoder: Option<Box<dyn AudioEncoderWrapper>>,
    stream_index: usize,
    finished: bool,
}

impl EncoderNode {
    /// Create a new video encoder node.
    pub fn new_video<E: VideoEncoderWrapper + 'static>(
        name: impl Into<String>,
        encoder: E,
        stream_index: usize,
    ) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            video_encoder: Some(Box::new(encoder)),
            audio_encoder: None,
            stream_index,
            finished: false,
        }
    }

    /// Create a new audio encoder node.
    pub fn new_audio<E: AudioEncoderWrapper + 'static>(
        name: impl Into<String>,
        encoder: E,
        stream_index: usize,
    ) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            video_encoder: None,
            audio_encoder: Some(Box::new(encoder)),
            stream_index,
            finished: false,
        }
    }

    /// Get stream index.
    pub fn stream_index(&self) -> usize {
        self.stream_index
    }

    /// Encode a video frame.
    pub fn encode_frame(&mut self, frame: &Frame) -> Result<Vec<NodeOutput>> {
        let mut outputs = Vec::new();
        if let Some(ref mut encoder) = self.video_encoder {
            for packet in encoder.encode(frame)? {
                outputs.push(NodeOutput::Packet(packet));
            }
        }
        Ok(outputs)
    }

    /// Encode audio samples.
    pub fn encode_samples(&mut self, sample: &Sample) -> Result<Vec<NodeOutput>> {
        let mut outputs = Vec::new();
        if let Some(ref mut encoder) = self.audio_encoder {
            for packet in encoder.encode(sample)? {
                outputs.push(NodeOutput::Packet(packet));
            }
        }
        Ok(outputs)
    }
}

impl Node for EncoderNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self) -> Result<NodeOutput> {
        Ok(NodeOutput::None)
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn flush(&mut self) -> Result<Vec<NodeOutput>> {
        let mut outputs = Vec::new();

        if let Some(ref mut encoder) = self.video_encoder {
            for packet in encoder.flush()? {
                outputs.push(NodeOutput::Packet(packet));
            }
        }

        if let Some(ref mut encoder) = self.audio_encoder {
            for packet in encoder.flush()? {
                outputs.push(NodeOutput::Packet(packet));
            }
        }

        self.finished = true;
        outputs.push(NodeOutput::EndOfStream);
        Ok(outputs)
    }
}

/// Muxer wrapper trait.
pub trait MuxerWrapper: Send {
    /// Get format name.
    fn format_name(&self) -> &str;
    /// Add a stream.
    fn add_stream(&mut self, info: PipelineStreamInfo) -> Result<usize>;
    /// Write header.
    fn write_header(&mut self) -> Result<()>;
    /// Write a packet.
    fn write_packet(&mut self, packet: &Packet<'_>) -> Result<()>;
    /// Write trailer.
    fn write_trailer(&mut self) -> Result<()>;
}

/// Muxer node.
pub struct MuxerNode {
    id: NodeId,
    name: String,
    muxer: Box<dyn MuxerWrapper>,
    header_written: bool,
    finished: bool,
}

impl MuxerNode {
    /// Create a new muxer node.
    pub fn new<M: MuxerWrapper + 'static>(name: impl Into<String>, muxer: M) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            muxer: Box::new(muxer),
            header_written: false,
            finished: false,
        }
    }

    /// Add a stream to the muxer.
    pub fn add_stream(&mut self, info: PipelineStreamInfo) -> Result<usize> {
        self.muxer.add_stream(info)
    }

    /// Write header if not already written.
    pub fn ensure_header(&mut self) -> Result<()> {
        if !self.header_written {
            self.muxer.write_header()?;
            self.header_written = true;
        }
        Ok(())
    }

    /// Write a packet.
    pub fn write_packet(&mut self, packet: &Packet<'_>) -> Result<()> {
        self.ensure_header()?;
        self.muxer.write_packet(packet)
    }

    /// Finalize the muxer.
    pub fn finalize(&mut self) -> Result<()> {
        if !self.finished {
            self.muxer.write_trailer()?;
            self.finished = true;
        }
        Ok(())
    }
}

impl Node for MuxerNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self) -> Result<NodeOutput> {
        Ok(NodeOutput::None)
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn flush(&mut self) -> Result<Vec<NodeOutput>> {
        self.finalize()?;
        Ok(vec![NodeOutput::EndOfStream])
    }
}
