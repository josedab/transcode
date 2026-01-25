//! SDP (Session Description Protocol) parsing and generation.

use crate::error::{Result, WhipError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Media type in SDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaType {
    Audio,
    Video,
    Application,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Audio => write!(f, "audio"),
            MediaType::Video => write!(f, "video"),
            MediaType::Application => write!(f, "application"),
        }
    }
}

/// Codec information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codec {
    pub payload_type: u8,
    pub name: String,
    pub clock_rate: u32,
    pub channels: Option<u8>,
    pub parameters: HashMap<String, String>,
}

impl Codec {
    pub fn new(payload_type: u8, name: &str, clock_rate: u32) -> Self {
        Self {
            payload_type,
            name: name.to_string(),
            clock_rate,
            channels: None,
            parameters: HashMap::new(),
        }
    }

    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = Some(channels);
        self
    }

    pub fn with_parameter(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }

    /// Common video codecs.
    pub fn h264(payload_type: u8) -> Self {
        Self::new(payload_type, "H264", 90000)
            .with_parameter("profile-level-id", "42e01f")
            .with_parameter("packetization-mode", "1")
    }

    pub fn vp8(payload_type: u8) -> Self {
        Self::new(payload_type, "VP8", 90000)
    }

    pub fn vp9(payload_type: u8) -> Self {
        Self::new(payload_type, "VP9", 90000)
    }

    pub fn av1(payload_type: u8) -> Self {
        Self::new(payload_type, "AV1", 90000)
    }

    /// Common audio codecs.
    pub fn opus(payload_type: u8) -> Self {
        Self::new(payload_type, "opus", 48000)
            .with_channels(2)
            .with_parameter("minptime", "10")
            .with_parameter("useinbandfec", "1")
    }
}

/// ICE candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCandidate {
    pub foundation: String,
    pub component: u32,
    pub protocol: String,
    pub priority: u32,
    pub ip: String,
    pub port: u16,
    pub typ: String,
    pub related_address: Option<String>,
    pub related_port: Option<u16>,
}

impl IceCandidate {
    pub fn to_sdp_line(&self) -> String {
        let mut line = format!(
            "a=candidate:{} {} {} {} {} {} typ {}",
            self.foundation,
            self.component,
            self.protocol,
            self.priority,
            self.ip,
            self.port,
            self.typ
        );
        if let (Some(addr), Some(port)) = (&self.related_address, self.related_port) {
            line.push_str(&format!(" raddr {} rport {}", addr, port));
        }
        line
    }
}

/// Media description in SDP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaDescription {
    pub media_type: MediaType,
    pub port: u16,
    pub protocol: String,
    pub codecs: Vec<Codec>,
    pub direction: MediaDirection,
    pub ice_ufrag: Option<String>,
    pub ice_pwd: Option<String>,
    pub fingerprint: Option<String>,
    pub setup: Option<String>,
    pub mid: Option<String>,
    pub candidates: Vec<IceCandidate>,
    pub rtcp_mux: bool,
    pub rtcp_rsize: bool,
}

impl MediaDescription {
    pub fn new(media_type: MediaType) -> Self {
        Self {
            media_type,
            port: 9,
            protocol: "UDP/TLS/RTP/SAVPF".to_string(),
            codecs: Vec::new(),
            direction: MediaDirection::SendRecv,
            ice_ufrag: None,
            ice_pwd: None,
            fingerprint: None,
            setup: None,
            mid: None,
            candidates: Vec::new(),
            rtcp_mux: true,
            rtcp_rsize: true,
        }
    }

    pub fn with_codec(mut self, codec: Codec) -> Self {
        self.codecs.push(codec);
        self
    }

    pub fn with_direction(mut self, direction: MediaDirection) -> Self {
        self.direction = direction;
        self
    }
}

/// Media direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaDirection {
    SendOnly,
    RecvOnly,
    SendRecv,
    Inactive,
}

impl std::fmt::Display for MediaDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaDirection::SendOnly => write!(f, "sendonly"),
            MediaDirection::RecvOnly => write!(f, "recvonly"),
            MediaDirection::SendRecv => write!(f, "sendrecv"),
            MediaDirection::Inactive => write!(f, "inactive"),
        }
    }
}

/// Session description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDescription {
    pub version: u8,
    pub origin: Origin,
    pub session_name: String,
    pub timing: Timing,
    pub media: Vec<MediaDescription>,
    pub groups: Vec<String>,
    pub ice_options: Vec<String>,
}

impl Default for SessionDescription {
    fn default() -> Self {
        Self {
            version: 0,
            origin: Origin::default(),
            session_name: "-".to_string(),
            timing: Timing::default(),
            media: Vec::new(),
            groups: Vec::new(),
            ice_options: vec!["trickle".to_string()],
        }
    }
}

impl SessionDescription {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_media(mut self, media: MediaDescription) -> Self {
        self.media.push(media);
        self
    }

    /// Parse SDP from string.
    pub fn parse(sdp: &str) -> Result<Self> {
        let mut desc = SessionDescription::default();
        let mut current_media: Option<MediaDescription> = None;
        let mut mids = Vec::new();

        for line in sdp.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.len() < 2 || line.chars().nth(1) != Some('=') {
                continue;
            }

            let (key, value) = (line.chars().next().unwrap(), &line[2..]);

            match key {
                'v' => {
                    desc.version = value.parse().unwrap_or(0);
                }
                'o' => {
                    desc.origin = Origin::parse(value)?;
                }
                's' => {
                    desc.session_name = value.to_string();
                }
                't' => {
                    desc.timing = Timing::parse(value)?;
                }
                'm' => {
                    // Save previous media if exists
                    if let Some(m) = current_media.take() {
                        if let Some(mid) = &m.mid {
                            mids.push(mid.clone());
                        }
                        desc.media.push(m);
                    }
                    current_media = Some(parse_media_line(value)?);
                }
                'a' => {
                    if let Some(ref mut media) = current_media {
                        parse_media_attribute(media, value)?;
                    } else {
                        // Session-level attribute
                        parse_session_attribute(&mut desc, value)?;
                    }
                }
                _ => {}
            }
        }

        // Save last media
        if let Some(m) = current_media {
            if let Some(mid) = &m.mid {
                mids.push(mid.clone());
            }
            desc.media.push(m);
        }

        if !mids.is_empty() {
            desc.groups = vec![format!("BUNDLE {}", mids.join(" "))];
        }

        Ok(desc)
    }

    /// Generate SDP string.
    pub fn to_sdp(&self) -> String {
        let mut lines = Vec::new();

        // Session description
        lines.push(format!("v={}", self.version));
        lines.push(self.origin.to_sdp_line());
        lines.push(format!("s={}", self.session_name));
        lines.push(self.timing.to_sdp_line());

        // Session-level attributes
        for group in &self.groups {
            lines.push(format!("a=group:{}", group));
        }
        if !self.ice_options.is_empty() {
            lines.push(format!("a=ice-options:{}", self.ice_options.join(" ")));
        }

        // Media descriptions
        for media in &self.media {
            lines.push(media_to_sdp(media));
        }

        lines.join("\r\n") + "\r\n"
    }
}

/// SDP origin field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Origin {
    pub username: String,
    pub session_id: u64,
    pub session_version: u64,
    pub net_type: String,
    pub addr_type: String,
    pub address: String,
}

impl Default for Origin {
    fn default() -> Self {
        Self {
            username: "-".to_string(),
            session_id: rand_session_id(),
            session_version: 2,
            net_type: "IN".to_string(),
            addr_type: "IP4".to_string(),
            address: "127.0.0.1".to_string(),
        }
    }
}

impl Origin {
    fn parse(value: &str) -> Result<Self> {
        let parts: Vec<&str> = value.split_whitespace().collect();
        if parts.len() < 6 {
            return Err(WhipError::InvalidSdp("Invalid origin line".to_string()));
        }
        Ok(Self {
            username: parts[0].to_string(),
            session_id: parts[1].parse().unwrap_or(0),
            session_version: parts[2].parse().unwrap_or(0),
            net_type: parts[3].to_string(),
            addr_type: parts[4].to_string(),
            address: parts[5].to_string(),
        })
    }

    fn to_sdp_line(&self) -> String {
        format!(
            "o={} {} {} {} {} {}",
            self.username,
            self.session_id,
            self.session_version,
            self.net_type,
            self.addr_type,
            self.address
        )
    }
}

/// SDP timing field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    pub start: u64,
    pub stop: u64,
}

impl Timing {
    fn parse(value: &str) -> Result<Self> {
        let parts: Vec<&str> = value.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(WhipError::InvalidSdp("Invalid timing line".to_string()));
        }
        Ok(Self {
            start: parts[0].parse().unwrap_or(0),
            stop: parts[1].parse().unwrap_or(0),
        })
    }

    fn to_sdp_line(&self) -> String {
        format!("t={} {}", self.start, self.stop)
    }
}

fn parse_media_line(value: &str) -> Result<MediaDescription> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(WhipError::InvalidSdp("Invalid media line".to_string()));
    }

    let media_type = match parts[0] {
        "audio" => MediaType::Audio,
        "video" => MediaType::Video,
        "application" => MediaType::Application,
        _ => return Err(WhipError::InvalidSdp(format!("Unknown media type: {}", parts[0]))),
    };

    let mut media = MediaDescription::new(media_type);
    media.port = parts[1].parse().unwrap_or(9);
    media.protocol = parts[2].to_string();

    Ok(media)
}

fn parse_media_attribute(media: &mut MediaDescription, value: &str) -> Result<()> {
    if let Some((key, val)) = value.split_once(':') {
        match key {
            "rtpmap" => {
                // Parse codec info: PT codec/clock/channels
                let parts: Vec<&str> = val.split_whitespace().collect();
                if parts.len() >= 2 {
                    let pt: u8 = parts[0].parse().unwrap_or(0);
                    let codec_parts: Vec<&str> = parts[1].split('/').collect();
                    if !codec_parts.is_empty() {
                        let mut codec = Codec::new(
                            pt,
                            codec_parts[0],
                            codec_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(90000),
                        );
                        if let Some(ch) = codec_parts.get(2).and_then(|s| s.parse().ok()) {
                            codec.channels = Some(ch);
                        }
                        media.codecs.push(codec);
                    }
                }
            }
            "fmtp" => {
                // Format parameters
                let parts: Vec<&str> = val.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let pt: u8 = parts[0].parse().unwrap_or(0);
                    if let Some(codec) = media.codecs.iter_mut().find(|c| c.payload_type == pt) {
                        for param in parts[1].split(';') {
                            if let Some((k, v)) = param.trim().split_once('=') {
                                codec.parameters.insert(k.to_string(), v.to_string());
                            }
                        }
                    }
                }
            }
            "ice-ufrag" => media.ice_ufrag = Some(val.to_string()),
            "ice-pwd" => media.ice_pwd = Some(val.to_string()),
            "fingerprint" => media.fingerprint = Some(val.to_string()),
            "setup" => media.setup = Some(val.to_string()),
            "mid" => media.mid = Some(val.to_string()),
            "candidate" => {
                if let Ok(candidate) = parse_candidate(val) {
                    media.candidates.push(candidate);
                }
            }
            _ => {}
        }
    } else {
        // Attribute without value
        match value {
            "sendonly" => media.direction = MediaDirection::SendOnly,
            "recvonly" => media.direction = MediaDirection::RecvOnly,
            "sendrecv" => media.direction = MediaDirection::SendRecv,
            "inactive" => media.direction = MediaDirection::Inactive,
            "rtcp-mux" => media.rtcp_mux = true,
            "rtcp-rsize" => media.rtcp_rsize = true,
            _ => {}
        }
    }
    Ok(())
}

fn parse_session_attribute(desc: &mut SessionDescription, value: &str) -> Result<()> {
    if let Some((key, val)) = value.split_once(':') {
        match key {
            "group" => desc.groups.push(val.to_string()),
            "ice-options" => {
                desc.ice_options = val.split_whitespace().map(String::from).collect();
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn parse_candidate(value: &str) -> Result<IceCandidate> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() < 8 {
        return Err(WhipError::InvalidSdp("Invalid candidate".to_string()));
    }

    Ok(IceCandidate {
        foundation: parts[0].to_string(),
        component: parts[1].parse().unwrap_or(1),
        protocol: parts[2].to_string(),
        priority: parts[3].parse().unwrap_or(0),
        ip: parts[4].to_string(),
        port: parts[5].parse().unwrap_or(0),
        typ: parts[7].to_string(),
        related_address: None,
        related_port: None,
    })
}

fn media_to_sdp(media: &MediaDescription) -> String {
    let mut lines = Vec::new();

    // m= line
    let payload_types: Vec<String> = media.codecs.iter().map(|c| c.payload_type.to_string()).collect();
    lines.push(format!(
        "m={} {} {} {}",
        media.media_type,
        media.port,
        media.protocol,
        payload_types.join(" ")
    ));

    // Connection info
    lines.push("c=IN IP4 0.0.0.0".to_string());

    // ICE credentials
    if let Some(ref ufrag) = media.ice_ufrag {
        lines.push(format!("a=ice-ufrag:{}", ufrag));
    }
    if let Some(ref pwd) = media.ice_pwd {
        lines.push(format!("a=ice-pwd:{}", pwd));
    }

    // DTLS fingerprint
    if let Some(ref fp) = media.fingerprint {
        lines.push(format!("a=fingerprint:{}", fp));
    }
    if let Some(ref setup) = media.setup {
        lines.push(format!("a=setup:{}", setup));
    }

    // MID
    if let Some(ref mid) = media.mid {
        lines.push(format!("a=mid:{}", mid));
    }

    // Direction
    lines.push(format!("a={}", media.direction));

    // RTCP mux
    if media.rtcp_mux {
        lines.push("a=rtcp-mux".to_string());
    }
    if media.rtcp_rsize {
        lines.push("a=rtcp-rsize".to_string());
    }

    // Codec info
    for codec in &media.codecs {
        let rtpmap = if let Some(ch) = codec.channels {
            format!(
                "a=rtpmap:{} {}/{}/{}",
                codec.payload_type, codec.name, codec.clock_rate, ch
            )
        } else {
            format!(
                "a=rtpmap:{} {}/{}",
                codec.payload_type, codec.name, codec.clock_rate
            )
        };
        lines.push(rtpmap);

        if !codec.parameters.is_empty() {
            let params: Vec<String> = codec
                .parameters
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            lines.push(format!("a=fmtp:{} {}", codec.payload_type, params.join(";")));
        }
    }

    // Candidates
    for candidate in &media.candidates {
        lines.push(candidate.to_sdp_line());
    }

    lines.join("\r\n")
}

fn rand_session_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_h264() {
        let codec = Codec::h264(96);
        assert_eq!(codec.payload_type, 96);
        assert_eq!(codec.name, "H264");
        assert_eq!(codec.clock_rate, 90000);
        assert!(codec.parameters.contains_key("profile-level-id"));
    }

    #[test]
    fn test_codec_opus() {
        let codec = Codec::opus(111);
        assert_eq!(codec.payload_type, 111);
        assert_eq!(codec.name, "opus");
        assert_eq!(codec.clock_rate, 48000);
        assert_eq!(codec.channels, Some(2));
    }

    #[test]
    fn test_session_description_roundtrip() {
        let desc = SessionDescription::new()
            .with_media(
                MediaDescription::new(MediaType::Video)
                    .with_codec(Codec::h264(96))
                    .with_direction(MediaDirection::RecvOnly),
            )
            .with_media(
                MediaDescription::new(MediaType::Audio)
                    .with_codec(Codec::opus(111))
                    .with_direction(MediaDirection::RecvOnly),
            );

        let sdp = desc.to_sdp();
        assert!(sdp.contains("m=video"));
        assert!(sdp.contains("m=audio"));
        assert!(sdp.contains("H264"));
        assert!(sdp.contains("opus"));
    }

    #[test]
    fn test_parse_simple_sdp() {
        let sdp = r#"v=0
o=- 123456789 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 H264/90000
a=sendonly
"#;
        let desc = SessionDescription::parse(sdp).unwrap();
        assert_eq!(desc.version, 0);
        assert_eq!(desc.media.len(), 1);
        assert_eq!(desc.media[0].media_type, MediaType::Video);
        assert_eq!(desc.media[0].direction, MediaDirection::SendOnly);
    }
}
