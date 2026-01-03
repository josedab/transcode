//! # Transcode FFmpeg Compatibility Layer
//!
//! This crate provides FFmpeg-compatible interfaces for the Transcode library,
//! enabling easy migration from FFmpeg-based workflows.
//!
//! ## Features
//!
//! - **Command-line parsing**: Parse FFmpeg-style command-line arguments
//! - **Stream specifiers**: Support for `-c:v`, `-b:a:0`, `0:v:0` syntax
//! - **Filter graphs**: Parse filter expressions like `scale=1920:1080,fps=30`
//! - **Codec mapping**: Convert FFmpeg codec names to internal types
//! - **Preset mapping**: Map FFmpeg presets to encoder settings
//! - **Value parsing**: Handle bitrates (`5M`), times (`1:30`), resolutions (`1080p`)
//!
//! ## Quick Start
//!
//! ```rust
//! use transcode_compat::{FfmpegCommand, CommandBuilder};
//!
//! // Parse FFmpeg-style command-line arguments
//! let args = ["-i", "input.mp4", "-c:v", "libx264", "-b:v", "5M", "output.mp4"];
//! let cmd = FfmpegCommand::parse(&args).unwrap();
//!
//! // Use the builder for convenient access
//! let builder = CommandBuilder::new(cmd);
//! println!("Video codec: {:?}", builder.video_codec());
//! println!("Video bitrate: {:?}", builder.video_bitrate());
//! ```
//!
//! ## Parsing Values
//!
//! ```rust
//! use transcode_compat::options::{Bitrate, TimeValue, Resolution, AspectRatio};
//!
//! // Bitrate parsing
//! let br = Bitrate::parse("5M").unwrap();
//! assert_eq!(br.bps(), 5_000_000);
//!
//! // Time parsing
//! let time = TimeValue::parse("1:30:00").unwrap();
//! assert_eq!(time.seconds(), 5400.0);
//!
//! // Resolution parsing
//! let res = Resolution::parse("1080p").unwrap();
//! assert_eq!(res.width, 1920);
//! assert_eq!(res.height, 1080);
//!
//! // Aspect ratio parsing
//! let ar = AspectRatio::parse("16:9").unwrap();
//! ```
//!
//! ## Filter Graphs
//!
//! ```rust
//! use transcode_compat::filter::{FilterGraph, FilterChain, Filter};
//!
//! // Parse a simple filter chain
//! let chain = FilterChain::parse("scale=1920:1080,fps=30").unwrap();
//! assert_eq!(chain.len(), 2);
//!
//! // Parse a complex filter graph
//! let graph = FilterGraph::parse("[0:v]scale=1280:720[v];[0:a]volume=0.5[a]").unwrap();
//! assert_eq!(graph.chains.len(), 2);
//!
//! // Build filters programmatically
//! use transcode_compat::filter::video;
//! let scale = video::scale(1920, 1080);
//! let fps = video::fps(30.0);
//! ```
//!
//! ## Codec Mapping
//!
//! ```rust
//! use transcode_compat::formats::{VideoCodecName, AudioCodecName, ContainerName};
//! use transcode_core::format::{VideoCodec, AudioCodec};
//!
//! // Map FFmpeg codec names
//! let codec = VideoCodecName::parse("libx264").unwrap();
//! assert_eq!(codec.to_video_codec(), Some(VideoCodec::H264));
//!
//! let codec = AudioCodecName::parse("libopus").unwrap();
//! assert_eq!(codec.to_audio_codec(), Some(AudioCodec::Opus));
//!
//! // Container format from file extension
//! let format = ContainerName::from_extension("mkv").unwrap();
//! ```
//!
//! ## Presets
//!
//! ```rust
//! use transcode_compat::preset::{Preset, Tune, H264Profile, EncoderSettings};
//!
//! // Parse presets
//! let preset = Preset::parse("slow").unwrap();
//! assert_eq!(preset.speed_value(), 3);
//! assert_eq!(preset.quality_value(), 7);
//!
//! // Get codec-specific values
//! assert_eq!(preset.x264_name(), "slow");
//! assert_eq!(preset.av1_cpu_used(), 3);
//!
//! // Build encoder settings
//! let settings = EncoderSettings::new()
//!     .with_preset(Preset::Medium)
//!     .with_crf(23.0);
//! ```
//!
//! ## Stream Specifiers
//!
//! ```rust
//! use transcode_compat::error::{StreamSpecifier, StreamType};
//!
//! // Parse various specifier formats
//! let spec = StreamSpecifier::parse("v:0").unwrap();  // First video stream
//! let spec = StreamSpecifier::parse("0:a:1").unwrap(); // Second audio from first input
//! let spec = StreamSpecifier::parse("0:1").unwrap();   // Stream 1 from first input
//!
//! // Check if a specifier matches
//! let spec = StreamSpecifier::parse("v").unwrap();
//! assert!(spec.matches(0, 0, StreamType::Video));
//! assert!(!spec.matches(0, 0, StreamType::Audio));
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod command;
pub mod error;
pub mod filter;
pub mod filter_exec;
pub mod formats;
pub mod options;
pub mod preset;

// Re-export main types at crate root
pub use command::{
    AudioOutputSettings, CommandBuilder, FfmpegCommand, GlobalOptions, InputFile, OutputFile,
    StreamMapping, VideoOutputSettings,
};
pub use error::{CompatError, Result, StreamSpecifier, StreamType};
pub use filter::{Filter, FilterChain, FilterGraph};
pub use formats::{AudioCodecName, CodecRegistry, ContainerName, PixelFormatName, SampleFormatName, VideoCodecName};
pub use options::{AspectRatio, Bitrate, OptionSet, ParsedOption, Resolution, TimeValue};
pub use preset::{EncoderSettings, H264Profile, H265Profile, Level, Preset, Tune};
pub use filter_exec::{
    // Core types
    AudioFilter, AudioFilterPipeline, AudioFormat, AudioSamples, FilterRegistry,
    PixelFormat, SampleFormat, VideoFilter, VideoFilterPipeline, VideoFormat, VideoFrame,
    // Video filters
    CropFilter, DeinterlaceFilter, DrawtextFilter, EqFilter, FormatFilter, FpsFilter,
    HflipFilter, NoiseFilter, NullVideoFilter, OverlayFilter, PadFilter, ScaleFilter,
    TransposeFilter, UnsharpFilter, VflipFilter,
    // Audio filters
    AformatFilter, AresampleFilter, AtempoFilter, LoudnormFilter, NullAudioFilter, VolumeFilter,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a complete FFmpeg command parsing workflow.
    #[test]
    fn test_complete_workflow() {
        // Simulate: ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 23 \
        //           -vf "scale=1920:1080,fps=30" -c:a aac -b:a 128k output.mp4
        let args = [
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "23",
            "-vf", "scale=1920:1080,fps=30",
            "-c:a", "aac",
            "-b:a", "128k",
            "output.mp4"
        ];

        let cmd = FfmpegCommand::parse(&args).unwrap();

        // Validate inputs
        assert_eq!(cmd.inputs.len(), 1);
        assert_eq!(cmd.inputs[0].path.to_str().unwrap(), "input.mp4");

        // Validate outputs
        assert_eq!(cmd.outputs.len(), 1);
        assert_eq!(cmd.outputs[0].path.to_str().unwrap(), "output.mp4");

        // Validate video settings
        let video = cmd.outputs[0].video.as_ref().unwrap();
        assert_eq!(video.codec, Some(VideoCodecName::H264));
        assert_eq!(video.encoder.preset, Some(Preset::Slow));
        assert_eq!(video.encoder.crf, Some(23.0));

        // Validate video filters
        let vf = &cmd.outputs[0].video_filters;
        assert_eq!(vf.len(), 2);
        assert_eq!(vf.filters[0].name, "scale");
        assert_eq!(vf.filters[0].get_positional(0), Some("1920"));
        assert_eq!(vf.filters[0].get_positional(1), Some("1080"));
        assert_eq!(vf.filters[1].name, "fps");
        assert_eq!(vf.filters[1].get_positional(0), Some("30"));

        // Validate audio settings
        let audio = cmd.outputs[0].audio.as_ref().unwrap();
        assert_eq!(audio.codec, Some(AudioCodecName::Aac));
        assert_eq!(audio.bitrate.unwrap().bps(), 128_000);
    }

    /// Test stream specifier combinations.
    #[test]
    fn test_stream_specifier_combinations() {
        // All video
        let spec = StreamSpecifier::parse("v").unwrap();
        assert!(spec.matches(0, 0, StreamType::Video));
        assert!(spec.matches(1, 5, StreamType::Video));
        assert!(!spec.matches(0, 0, StreamType::Audio));

        // Specific stream from specific input
        let spec = StreamSpecifier::parse("1:a:2").unwrap();
        assert!(spec.matches(1, 2, StreamType::Audio));
        assert!(!spec.matches(0, 2, StreamType::Audio));
        assert!(!spec.matches(1, 0, StreamType::Audio));
        assert!(!spec.matches(1, 2, StreamType::Video));
    }

    /// Test codec conversions.
    #[test]
    fn test_codec_conversions() {
        // Video codec from FFmpeg name to internal type
        let codec = VideoCodecName::parse("libx265").unwrap();
        let internal = codec.to_video_codec().unwrap();
        assert_eq!(internal, transcode_core::format::VideoCodec::H265);

        // Audio codec from FFmpeg name to internal type
        let codec = AudioCodecName::parse("libopus").unwrap();
        let internal = codec.to_audio_codec().unwrap();
        assert_eq!(internal, transcode_core::format::AudioCodec::Opus);

        // Container from extension
        let format = ContainerName::from_extension("webm").unwrap();
        let internal = format.to_container_format().unwrap();
        assert_eq!(internal, transcode_core::format::ContainerFormat::WebM);
    }

    /// Test filter graph with links.
    #[test]
    fn test_filter_graph_with_links() {
        let graph = FilterGraph::parse(
            "[0:v]scale=1280:720,format=yuv420p[scaled];\
             [scaled]fps=24[out]"
        ).unwrap();

        assert_eq!(graph.chains.len(), 2);

        // First chain
        let chain1 = &graph.chains[0];
        assert_eq!(chain1.filters[0].inputs, vec!["0:v"]);
        assert_eq!(chain1.filters[1].outputs, vec!["scaled"]);

        // Second chain
        let chain2 = &graph.chains[1];
        assert_eq!(chain2.filters[0].inputs, vec!["scaled"]);
        assert_eq!(chain2.filters[0].outputs, vec!["out"]);
    }

    /// Test preset quality/speed tradeoffs.
    #[test]
    fn test_preset_quality_speed() {
        let presets = [
            Preset::Ultrafast,
            Preset::Veryfast,
            Preset::Fast,
            Preset::Medium,
            Preset::Slow,
            Preset::Veryslow,
        ];

        // Speed should decrease and quality should increase as we go through the list
        for i in 0..presets.len() - 1 {
            let current = &presets[i];
            let next = &presets[i + 1];
            assert!(
                current.speed_value() > next.speed_value(),
                "{:?} speed {} should be > {:?} speed {}",
                current,
                current.speed_value(),
                next,
                next.speed_value()
            );
            assert!(
                current.quality_value() < next.quality_value(),
                "{:?} quality {} should be < {:?} quality {}",
                current,
                current.quality_value(),
                next,
                next.quality_value()
            );
        }

        // Verify extreme values
        assert_eq!(Preset::Ultrafast.speed_value(), 10);
        assert_eq!(Preset::Ultrafast.quality_value(), 0);
        assert_eq!(Preset::Placebo.speed_value(), 0);
        assert_eq!(Preset::Placebo.quality_value(), 10);
    }

    /// Test time parsing edge cases.
    #[test]
    fn test_time_parsing_edge_cases() {
        // Seconds only
        assert!((TimeValue::parse("0").unwrap().seconds() - 0.0).abs() < 0.001);
        assert!((TimeValue::parse("0.5").unwrap().seconds() - 0.5).abs() < 0.001);
        assert!((TimeValue::parse("59.999").unwrap().seconds() - 59.999).abs() < 0.001);

        // Minutes:seconds
        assert!((TimeValue::parse("0:00").unwrap().seconds() - 0.0).abs() < 0.001);
        assert!((TimeValue::parse("59:59").unwrap().seconds() - 3599.0).abs() < 0.001);

        // Hours:minutes:seconds
        assert!((TimeValue::parse("0:0:0").unwrap().seconds() - 0.0).abs() < 0.001);
        assert!((TimeValue::parse("23:59:59").unwrap().seconds() - 86399.0).abs() < 0.001);
    }

    /// Test bitrate parsing edge cases.
    #[test]
    fn test_bitrate_parsing_edge_cases() {
        // Basic values
        assert_eq!(Bitrate::parse("0").unwrap().bps(), 0);
        assert_eq!(Bitrate::parse("1").unwrap().bps(), 1);

        // With suffixes
        assert_eq!(Bitrate::parse("1k").unwrap().bps(), 1_000);
        assert_eq!(Bitrate::parse("1K").unwrap().bps(), 1_000);
        assert_eq!(Bitrate::parse("1m").unwrap().bps(), 1_000_000);
        assert_eq!(Bitrate::parse("1M").unwrap().bps(), 1_000_000);
        assert_eq!(Bitrate::parse("1g").unwrap().bps(), 1_000_000_000);
        assert_eq!(Bitrate::parse("1G").unwrap().bps(), 1_000_000_000);

        // Fractional
        assert_eq!(Bitrate::parse("1.5M").unwrap().bps(), 1_500_000);
        assert_eq!(Bitrate::parse("2.5k").unwrap().bps(), 2_500);
    }

    /// Test resolution name parsing.
    #[test]
    fn test_resolution_names() {
        let cases = [
            ("720p", 1280, 720),
            ("1080p", 1920, 1080),
            ("4k", 3840, 2160),
            ("vga", 640, 480),
            ("hd720", 1280, 720),
            ("hd1080", 1920, 1080),
        ];

        for (name, expected_w, expected_h) in cases {
            let res = Resolution::parse(name).unwrap();
            assert_eq!(res.width, expected_w, "Failed for {}", name);
            assert_eq!(res.height, expected_h, "Failed for {}", name);
        }
    }

    /// Test the CommandBuilder convenience methods.
    #[test]
    fn test_command_builder_convenience() {
        let args = [
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-b:v", "5M",
            "-preset", "slow",
            "-tune", "film",
            "-crf", "20",
            "-g", "250",
            "-c:a", "copy",
            "-an",
            "output.mp4"
        ];

        let builder = CommandBuilder::from_args(&args).unwrap();

        // Check video settings
        assert_eq!(builder.video_codec(), Some(VideoCodecName::H264));
        assert_eq!(builder.video_bitrate().unwrap().bps(), 5_000_000);
        assert!(!builder.is_video_copy());
        assert!(!builder.is_video_disabled());

        // Check audio settings
        assert!(builder.audio_codec().is_none()); // copy doesn't set codec
        assert!(builder.is_audio_copy());
        assert!(builder.is_audio_disabled());

        // Check encoder settings
        let encoder = builder.encoder_settings().unwrap();
        assert_eq!(encoder.preset, Some(Preset::Slow));
        assert_eq!(encoder.tune, Some(Tune::Film));
        assert_eq!(encoder.crf, Some(20.0));
        assert_eq!(encoder.keyframe_interval, Some(250));
    }

    /// Test filter construction helpers.
    #[test]
    fn test_filter_helpers() {
        use filter::{video, audio};

        // Video filters
        let scale = video::scale(1920, 1080);
        assert_eq!(scale.name, "scale");
        assert_eq!(scale.get_positional(0), Some("1920"));
        assert_eq!(scale.get_positional(1), Some("1080"));

        let fps = video::fps(29.97);
        assert_eq!(fps.name, "fps");

        // Audio filters
        let vol = audio::volume(0.5);
        assert_eq!(vol.name, "volume");
        assert_eq!(vol.get_positional(0), Some("0.5"));

        let vol_db = audio::volume_db(-3.0);
        assert_eq!(vol_db.get_positional(0), Some("-3dB"));
    }

    /// Test error handling.
    #[test]
    fn test_error_handling() {
        // Unknown codec
        let result = VideoCodecName::parse("unknowncodec");
        assert!(matches!(result, Err(CompatError::UnknownCodec(_))));

        // Invalid bitrate
        let result = Bitrate::parse("not_a_bitrate");
        assert!(matches!(result, Err(CompatError::InvalidBitrate(_))));

        // Invalid time
        let result = TimeValue::parse("invalid:time:format:extra");
        assert!(matches!(result, Err(CompatError::InvalidTime(_))));

        // Invalid resolution
        let result = Resolution::parse("not_a_resolution");
        assert!(matches!(result, Err(CompatError::InvalidResolution(_))));

        // Invalid aspect ratio
        let result = AspectRatio::parse("not:valid:aspect");
        assert!(matches!(result, Err(CompatError::InvalidAspectRatio(_))));

        // Invalid stream specifier
        let result = StreamSpecifier::parse("x:y:z:w");
        assert!(matches!(result, Err(CompatError::InvalidStreamSpecifier(_))));
    }

    /// Test two-pass encoding detection.
    #[test]
    fn test_two_pass_detection() {
        let args = [
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-b:v", "5M",
            "-pass", "1",
            "-passlogfile", "ffmpeg2pass",
            "output.mp4"
        ];

        let cmd = FfmpegCommand::parse(&args).unwrap();
        let video = cmd.outputs[0].video.as_ref().unwrap();

        assert_eq!(video.pass, Some(1));
        assert_eq!(video.passlogfile, Some("ffmpeg2pass".to_string()));
    }

    /// Test metadata parsing.
    #[test]
    fn test_metadata_parsing() {
        let args = [
            "-i", "input.mp4",
            "-metadata", "title=My Video",
            "-metadata", "artist=Test Artist",
            "output.mp4"
        ];

        let cmd = FfmpegCommand::parse(&args).unwrap();
        let metadata = &cmd.outputs[0].metadata;

        assert_eq!(metadata.get("title"), Some(&"My Video".to_string()));
        assert_eq!(metadata.get("artist"), Some(&"Test Artist".to_string()));
    }

    /// Test option set conversion.
    #[test]
    fn test_option_set_conversion() {
        let args = [
            "-y",
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-b:v", "5M",
            "-vf", "scale=1920:1080",
            "-c:a", "aac",
            "-b:a", "128k",
            "output.mp4"
        ];

        let cmd = FfmpegCommand::parse(&args).unwrap();
        let opts = cmd.to_option_set();

        assert!(opts.overwrite);
        assert_eq!(opts.inputs.len(), 1);
        assert_eq!(opts.outputs.len(), 1);
        assert!(opts.video_codec.contains_key(&StreamSpecifier::Video));
        assert!(opts.audio_codec.contains_key(&StreamSpecifier::Audio));
        assert!(opts.video_bitrate.get(&StreamSpecifier::Video).is_some());
        assert!(opts.audio_bitrate.get(&StreamSpecifier::Audio).is_some());
        assert_eq!(opts.video_filters.len(), 1);
    }
}
