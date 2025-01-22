//! WHIP/WHEP session management.

use crate::error::{Result, WhipError};
use crate::sdp::{MediaType, SessionDescription};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session created, waiting for ICE.
    Created,
    /// ICE negotiation in progress.
    Connecting,
    /// Session connected and active.
    Connected,
    /// Session is disconnecting.
    Disconnecting,
    /// Session closed.
    Closed,
    /// Session failed.
    Failed,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Created => write!(f, "created"),
            SessionState::Connecting => write!(f, "connecting"),
            SessionState::Connected => write!(f, "connected"),
            SessionState::Disconnecting => write!(f, "disconnecting"),
            SessionState::Closed => write!(f, "closed"),
            SessionState::Failed => write!(f, "failed"),
        }
    }
}

/// Session type (WHIP for ingest, WHEP for egress).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionType {
    /// WHIP: WebRTC-HTTP Ingest Protocol (publisher → server).
    Whip,
    /// WHEP: WebRTC-HTTP Egress Protocol (server → viewer).
    Whep,
}

/// Media track information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaTrack {
    pub id: String,
    pub media_type: MediaType,
    pub codec: String,
    pub bitrate: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
}

/// Session statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub packets_received: u64,
    pub packets_sent: u64,
    pub packets_lost: u64,
    pub jitter_ms: f64,
    pub rtt_ms: f64,
    pub bitrate_kbps: f64,
    pub frames_received: u64,
    pub frames_decoded: u64,
    pub keyframes_received: u64,
}

/// WHIP/WHEP session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub session_type: SessionType,
    pub state: SessionState,
    pub stream_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub connected_at: Option<DateTime<Utc>>,
    pub closed_at: Option<DateTime<Utc>>,
    pub local_sdp: Option<String>,
    pub remote_sdp: Option<String>,
    pub tracks: Vec<MediaTrack>,
    pub stats: SessionStats,
    pub metadata: HashMap<String, String>,
    pub etag: String,
}

impl Session {
    /// Create a new session.
    pub fn new(session_type: SessionType) -> Self {
        let id = Uuid::new_v4().to_string();
        Self {
            id: id.clone(),
            session_type,
            state: SessionState::Created,
            stream_id: None,
            created_at: Utc::now(),
            connected_at: None,
            closed_at: None,
            local_sdp: None,
            remote_sdp: None,
            tracks: Vec::new(),
            stats: SessionStats::default(),
            metadata: HashMap::new(),
            etag: generate_etag(&id),
        }
    }

    /// Create a new WHIP (ingest) session.
    pub fn whip() -> Self {
        Self::new(SessionType::Whip)
    }

    /// Create a new WHEP (egress) session.
    pub fn whep() -> Self {
        Self::new(SessionType::Whep)
    }

    /// Set stream ID.
    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = Some(stream_id.into());
        self
    }

    /// Get session duration.
    pub fn duration(&self) -> Option<chrono::Duration> {
        let end = self.closed_at.unwrap_or_else(Utc::now);
        self.connected_at.map(|start| end - start)
    }

    /// Check if session is active.
    pub fn is_active(&self) -> bool {
        matches!(self.state, SessionState::Connected | SessionState::Connecting)
    }

    /// Update session state.
    pub fn set_state(&mut self, state: SessionState) {
        let now = Utc::now();
        match state {
            SessionState::Connected if self.connected_at.is_none() => {
                self.connected_at = Some(now);
            }
            SessionState::Closed | SessionState::Failed if self.closed_at.is_none() => {
                self.closed_at = Some(now);
            }
            _ => {}
        }
        self.state = state;
        self.etag = generate_etag(&self.id);
    }

    /// Set remote SDP (offer from client).
    pub fn set_remote_sdp(&mut self, sdp: String) -> Result<()> {
        // Parse and validate SDP
        let desc = SessionDescription::parse(&sdp)?;

        // Extract track info
        self.tracks.clear();
        for (i, media) in desc.media.iter().enumerate() {
            let codec_name = media
                .codecs
                .first()
                .map(|c| c.name.clone())
                .unwrap_or_default();

            self.tracks.push(MediaTrack {
                id: format!("track-{}", i),
                media_type: media.media_type,
                codec: codec_name,
                bitrate: None,
                width: None,
                height: None,
                sample_rate: media.codecs.first().map(|c| c.clock_rate),
                channels: media.codecs.first().and_then(|c| c.channels),
            });
        }

        self.remote_sdp = Some(sdp);
        self.etag = generate_etag(&self.id);
        Ok(())
    }

    /// Set local SDP (answer to client).
    pub fn set_local_sdp(&mut self, sdp: String) {
        self.local_sdp = Some(sdp);
        self.etag = generate_etag(&self.id);
    }

    /// Update statistics.
    pub fn update_stats(&mut self, stats: SessionStats) {
        self.stats = stats;
    }
}

fn generate_etag(id: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("\"{}:{}\"", &id[..8], ts % 1_000_000)
}

/// Session manager.
#[derive(Debug)]
pub struct SessionManager {
    sessions: RwLock<HashMap<String, Session>>,
    max_sessions: usize,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
        }
    }

    /// Create a new session.
    pub fn create_session(&self, session_type: SessionType) -> Result<Session> {
        let sessions = self.sessions.read();
        if sessions.len() >= self.max_sessions {
            return Err(WhipError::ResourceLimitExceeded(format!(
                "Maximum sessions ({}) reached",
                self.max_sessions
            )));
        }
        drop(sessions);

        let session = Session::new(session_type);
        let id = session.id.clone();

        self.sessions.write().insert(id, session.clone());
        tracing::info!(session_id = %session.id, "Created new {:?} session", session_type);

        Ok(session)
    }

    /// Get session by ID.
    pub fn get_session(&self, id: &str) -> Result<Session> {
        self.sessions
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| WhipError::SessionNotFound(id.to_string()))
    }

    /// Update session.
    pub fn update_session(&self, session: Session) -> Result<()> {
        let mut sessions = self.sessions.write();
        if !sessions.contains_key(&session.id) {
            return Err(WhipError::SessionNotFound(session.id.clone()));
        }
        sessions.insert(session.id.clone(), session);
        Ok(())
    }

    /// Delete session.
    pub fn delete_session(&self, id: &str) -> Result<Session> {
        self.sessions
            .write()
            .remove(id)
            .ok_or_else(|| WhipError::SessionNotFound(id.to_string()))
    }

    /// Get all sessions.
    pub fn list_sessions(&self) -> Vec<Session> {
        self.sessions.read().values().cloned().collect()
    }

    /// Get sessions by stream ID.
    pub fn get_sessions_by_stream(&self, stream_id: &str) -> Vec<Session> {
        self.sessions
            .read()
            .values()
            .filter(|s| s.stream_id.as_deref() == Some(stream_id))
            .cloned()
            .collect()
    }

    /// Get session count.
    pub fn session_count(&self) -> usize {
        self.sessions.read().len()
    }

    /// Get active session count.
    pub fn active_session_count(&self) -> usize {
        self.sessions.read().values().filter(|s| s.is_active()).count()
    }

    /// Clean up stale sessions.
    pub fn cleanup_stale_sessions(&self, max_age: chrono::Duration) -> usize {
        let now = Utc::now();
        let mut sessions = self.sessions.write();
        let initial_count = sessions.len();

        sessions.retain(|_, session| {
            // Keep active sessions
            if session.is_active() {
                return true;
            }
            // Remove closed/failed sessions older than max_age
            if let Some(closed_at) = session.closed_at {
                return now - closed_at < max_age;
            }
            // Remove stale created sessions
            now - session.created_at < max_age
        });

        initial_count - sessions.len()
    }
}

/// Thread-safe session manager handle.
pub type SessionManagerHandle = Arc<SessionManager>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = Session::whip();
        assert_eq!(session.session_type, SessionType::Whip);
        assert_eq!(session.state, SessionState::Created);
        assert!(session.connected_at.is_none());
    }

    #[test]
    fn test_session_state_transitions() {
        let mut session = Session::whip();
        assert_eq!(session.state, SessionState::Created);

        session.set_state(SessionState::Connecting);
        assert_eq!(session.state, SessionState::Connecting);

        session.set_state(SessionState::Connected);
        assert_eq!(session.state, SessionState::Connected);
        assert!(session.connected_at.is_some());

        session.set_state(SessionState::Closed);
        assert_eq!(session.state, SessionState::Closed);
        assert!(session.closed_at.is_some());
    }

    #[test]
    fn test_session_manager() {
        let manager = SessionManager::new(10);

        let session = manager.create_session(SessionType::Whip).unwrap();
        assert_eq!(manager.session_count(), 1);

        let retrieved = manager.get_session(&session.id).unwrap();
        assert_eq!(retrieved.id, session.id);

        manager.delete_session(&session.id).unwrap();
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_session_manager_limit() {
        let manager = SessionManager::new(2);

        manager.create_session(SessionType::Whip).unwrap();
        manager.create_session(SessionType::Whip).unwrap();

        let result = manager.create_session(SessionType::Whip);
        assert!(matches!(result, Err(WhipError::ResourceLimitExceeded(_))));
    }

    #[test]
    fn test_session_duration() {
        let mut session = Session::whip();
        assert!(session.duration().is_none());

        session.set_state(SessionState::Connected);
        std::thread::sleep(std::time::Duration::from_millis(10));

        let duration = session.duration().unwrap();
        assert!(duration.num_milliseconds() >= 10);
    }
}
