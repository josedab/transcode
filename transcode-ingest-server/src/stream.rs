//! Stream session management.

use serde::{Deserialize, Serialize};

/// Protocol used by the ingest stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProtocolType {
    Rtmp,
    Srt,
    Whip,
}

impl ProtocolType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rtmp => "RTMP",
            Self::Srt => "SRT",
            Self::Whip => "WHIP",
        }
    }
}

/// Current status of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamStatus {
    Connecting,
    Live,
    Paused,
    Disconnected,
    Error,
}

/// Information about a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    pub stream_key: String,
    pub protocol: ProtocolType,
    pub remote_address: String,
    pub started_at: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub framerate: Option<f64>,
    pub bitrate_bps: Option<u64>,
    pub frames_received: u64,
    pub bytes_received: u64,
}

/// An active stream session.
pub struct StreamSession {
    info: StreamInfo,
    status: StreamStatus,
}

impl StreamSession {
    pub fn new(stream_key: &str, protocol: ProtocolType, remote_addr: &str) -> Self {
        Self {
            info: StreamInfo {
                stream_key: stream_key.into(),
                protocol,
                remote_address: remote_addr.into(),
                started_at: None,
                video_codec: None,
                audio_codec: None,
                width: None,
                height: None,
                framerate: None,
                bitrate_bps: None,
                frames_received: 0,
                bytes_received: 0,
            },
            status: StreamStatus::Connecting,
        }
    }

    pub fn info(&self) -> &StreamInfo {
        &self.info
    }

    pub fn status(&self) -> StreamStatus {
        self.status
    }

    pub fn set_status(&mut self, status: StreamStatus) {
        self.status = status;
    }

    pub fn set_live(&mut self) {
        self.status = StreamStatus::Live;
        self.info.started_at = Some(chrono::Utc::now().to_rfc3339());
    }

    pub fn update_stats(&mut self, frames: u64, bytes: u64) {
        self.info.frames_received += frames;
        self.info.bytes_received += bytes;
    }

    pub fn set_media_info(
        &mut self,
        video_codec: Option<&str>,
        audio_codec: Option<&str>,
        width: Option<u32>,
        height: Option<u32>,
        framerate: Option<f64>,
    ) {
        self.info.video_codec = video_codec.map(String::from);
        self.info.audio_codec = audio_codec.map(String::from);
        self.info.width = width;
        self.info.height = height;
        self.info.framerate = framerate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let mut session = StreamSession::new("key1", ProtocolType::Rtmp, "1.2.3.4:5678");
        assert_eq!(session.status(), StreamStatus::Connecting);

        session.set_live();
        assert_eq!(session.status(), StreamStatus::Live);
        assert!(session.info().started_at.is_some());

        session.update_stats(100, 1_000_000);
        assert_eq!(session.info().frames_received, 100);
        assert_eq!(session.info().bytes_received, 1_000_000);

        session.set_status(StreamStatus::Disconnected);
        assert_eq!(session.status(), StreamStatus::Disconnected);
    }

    #[test]
    fn test_media_info() {
        let mut session = StreamSession::new("key2", ProtocolType::Srt, "5.6.7.8:9000");
        session.set_media_info(Some("h264"), Some("aac"), Some(1920), Some(1080), Some(30.0));
        assert_eq!(session.info().video_codec.as_deref(), Some("h264"));
        assert_eq!(session.info().width, Some(1920));
    }
}
