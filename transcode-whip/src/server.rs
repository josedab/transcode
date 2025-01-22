//! WHIP/WHEP HTTP server implementation.

use crate::error::{Result, WhipError};
use crate::sdp::{Codec, MediaDescription, MediaDirection, MediaType, SessionDescription};
use crate::session::{Session, SessionManager, SessionManagerHandle, SessionState, SessionType};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, options, patch, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server bind address.
    pub bind_address: String,
    /// Maximum concurrent sessions.
    pub max_sessions: usize,
    /// WHIP endpoint path.
    pub whip_path: String,
    /// WHEP endpoint path.
    pub whep_path: String,
    /// ICE servers for clients.
    pub ice_servers: Vec<IceServer>,
    /// Enable CORS.
    pub enable_cors: bool,
    /// Session timeout in seconds.
    pub session_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8080".to_string(),
            max_sessions: 1000,
            whip_path: "/whip".to_string(),
            whep_path: "/whep".to_string(),
            ice_servers: vec![IceServer {
                urls: vec!["stun:stun.l.google.com:19302".to_string()],
                username: None,
                credential: None,
            }],
            enable_cors: true,
            session_timeout_secs: 30,
        }
    }
}

/// ICE server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceServer {
    pub urls: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

/// Server event.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// New session created.
    SessionCreated { session_id: String, session_type: SessionType },
    /// Session connected.
    SessionConnected { session_id: String },
    /// Session closed.
    SessionClosed { session_id: String },
    /// Media received (WHIP only).
    MediaReceived { session_id: String, track_id: String, bytes: usize },
}

/// Shared server state.
#[derive(Clone)]
pub struct ServerState {
    pub config: ServerConfig,
    pub sessions: SessionManagerHandle,
    pub events: broadcast::Sender<ServerEvent>,
}

/// WHIP/WHEP server.
pub struct WhipServer {
    config: ServerConfig,
    state: ServerState,
    event_rx: broadcast::Receiver<ServerEvent>,
}

impl WhipServer {
    /// Create a new WHIP/WHEP server.
    pub fn new(config: ServerConfig) -> Self {
        let (event_tx, event_rx) = broadcast::channel(1000);
        let sessions = Arc::new(SessionManager::new(config.max_sessions));

        let state = ServerState {
            config: config.clone(),
            sessions,
            events: event_tx,
        };

        Self {
            config,
            state,
            event_rx,
        }
    }

    /// Get the server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Subscribe to server events.
    pub fn subscribe(&self) -> broadcast::Receiver<ServerEvent> {
        self.state.events.subscribe()
    }

    /// Get event receiver (consumes self).
    pub fn into_event_receiver(self) -> broadcast::Receiver<ServerEvent> {
        self.event_rx
    }

    /// Get session manager.
    pub fn sessions(&self) -> &SessionManagerHandle {
        &self.state.sessions
    }

    /// Build the Axum router.
    pub fn router(&self) -> Router {
        let mut app = Router::new()
            // WHIP endpoints
            .route(&self.config.whip_path, post(handle_whip_offer))
            .route(&self.config.whip_path, options(handle_options))
            .route(&format!("{}/:id", self.config.whip_path), patch(handle_ice_trickle))
            .route(&format!("{}/:id", self.config.whip_path), delete(handle_session_delete))
            // WHEP endpoints
            .route(&self.config.whep_path, post(handle_whep_offer))
            .route(&self.config.whep_path, options(handle_options))
            .route(&format!("{}/:id", self.config.whep_path), patch(handle_ice_trickle))
            .route(&format!("{}/:id", self.config.whep_path), delete(handle_session_delete))
            // Management endpoints
            .route("/sessions", get(handle_list_sessions))
            .route("/sessions/:id", get(handle_get_session))
            .route("/health", get(handle_health))
            .with_state(self.state.clone())
            .layer(TraceLayer::new_for_http());

        if self.config.enable_cors {
            app = app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([Method::GET, Method::POST, Method::PATCH, Method::DELETE, Method::OPTIONS])
                    .allow_headers(Any)
                    .expose_headers([
                        header::LOCATION,
                        header::ETAG,
                        header::HeaderName::from_static("link"),
                    ]),
            );
        }

        app
    }

    /// Run the server.
    pub async fn run(self) -> Result<()> {
        let addr: std::net::SocketAddr = self.config.bind_address.parse().map_err(|e| {
            WhipError::ConfigError(format!("Invalid bind address: {}", e))
        })?;

        let router = self.router();

        tracing::info!("WHIP/WHEP server starting on {}", addr);
        tracing::info!("WHIP endpoint: {}", self.config.whip_path);
        tracing::info!("WHEP endpoint: {}", self.config.whep_path);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router)
            .await
            .map_err(|e| WhipError::ServerError(e.to_string()))?;

        Ok(())
    }
}

/// WHIP offer request handler.
async fn handle_whip_offer(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    handle_offer(state, headers, body, SessionType::Whip).await
}

/// WHEP offer request handler.
async fn handle_whep_offer(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    handle_offer(state, headers, body, SessionType::Whep).await
}

/// Common offer handler.
async fn handle_offer(
    state: ServerState,
    headers: HeaderMap,
    body: String,
    session_type: SessionType,
) -> impl IntoResponse {
    // Validate Content-Type
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !content_type.contains("application/sdp") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/sdp",
        )
            .into_response();
    }

    // Create session
    let mut session = match state.sessions.create_session(session_type) {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response();
        }
    };

    // Parse and store remote SDP
    if let Err(e) = session.set_remote_sdp(body) {
        let _ = state.sessions.delete_session(&session.id);
        return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
    }

    // Generate answer SDP
    let answer = generate_answer(&session, &state.config);
    session.set_local_sdp(answer.clone());
    session.set_state(SessionState::Connecting);

    if let Err(e) = state.sessions.update_session(session.clone()) {
        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    // Emit event
    let _ = state.events.send(ServerEvent::SessionCreated {
        session_id: session.id.clone(),
        session_type,
    });

    // Build response
    let endpoint = match session_type {
        SessionType::Whip => &state.config.whip_path,
        SessionType::Whep => &state.config.whep_path,
    };

    let location = format!("{}/{}", endpoint, session.id);
    let link = build_link_header(&state.config.ice_servers);

    Response::builder()
        .status(StatusCode::CREATED)
        .header(header::CONTENT_TYPE, "application/sdp")
        .header(header::LOCATION, &location)
        .header(header::ETAG, &session.etag)
        .header("Link", link)
        .body(Body::from(answer))
        .unwrap()
}

/// Handle ICE trickle (PATCH).
async fn handle_ice_trickle(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Validate ETag
    let etag = headers
        .get(header::IF_MATCH)
        .and_then(|v| v.to_str().ok());

    let session = match state.sessions.get_session(&id) {
        Ok(s) => s,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    if let Some(client_etag) = etag {
        if client_etag != session.etag {
            return StatusCode::PRECONDITION_FAILED.into_response();
        }
    }

    // In a real implementation, we would process the ICE candidate here
    // For now, just acknowledge
    tracing::debug!(session_id = %id, "Received ICE candidate: {}", body);

    StatusCode::NO_CONTENT.into_response()
}

/// Handle session deletion (DELETE).
async fn handle_session_delete(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.sessions.delete_session(&id) {
        Ok(session) => {
            let _ = state.events.send(ServerEvent::SessionClosed {
                session_id: session.id,
            });
            StatusCode::OK.into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Handle OPTIONS for CORS preflight.
async fn handle_options() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::ACCEPT, "application/sdp")
        .header("Accept-Patch", "application/trickle-ice-sdpfrag")
        .body(Body::empty())
        .unwrap()
}

/// List all sessions.
async fn handle_list_sessions(State(state): State<ServerState>) -> impl IntoResponse {
    let sessions: Vec<SessionInfo> = state
        .sessions
        .list_sessions()
        .into_iter()
        .map(|s| SessionInfo {
            id: s.id,
            session_type: s.session_type,
            state: s.state,
            created_at: s.created_at.to_rfc3339(),
            tracks: s.tracks.len(),
        })
        .collect();

    Json(sessions)
}

/// Get single session.
async fn handle_get_session(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.sessions.get_session(&id) {
        Ok(session) => Json(session).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Health check.
async fn handle_health(State(state): State<ServerState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        sessions: state.sessions.session_count(),
        active_sessions: state.sessions.active_session_count(),
    })
}

/// Session info for listing.
#[derive(Serialize)]
struct SessionInfo {
    id: String,
    session_type: SessionType,
    state: SessionState,
    created_at: String,
    tracks: usize,
}

/// Health response.
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    sessions: usize,
    active_sessions: usize,
}

/// Generate SDP answer.
fn generate_answer(session: &Session, _config: &ServerConfig) -> String {
    let mut answer = SessionDescription::new();

    // Generate ICE credentials
    let ice_ufrag = generate_ice_string(8);
    let ice_pwd = generate_ice_string(24);
    let fingerprint = generate_mock_fingerprint();

    // Mirror media sections from offer with appropriate direction
    for track in &session.tracks {
        let direction = match session.session_type {
            SessionType::Whip => MediaDirection::RecvOnly, // Server receives from publisher
            SessionType::Whep => MediaDirection::SendOnly, // Server sends to viewer
        };

        let mut media = MediaDescription::new(track.media_type);
        media.direction = direction;
        media.ice_ufrag = Some(ice_ufrag.clone());
        media.ice_pwd = Some(ice_pwd.clone());
        media.fingerprint = Some(format!("sha-256 {}", fingerprint));
        media.setup = Some("passive".to_string());
        media.mid = Some(track.id.clone());

        // Add codec based on track type
        match track.media_type {
            MediaType::Video => {
                media.codecs.push(Codec::h264(96));
                media.codecs.push(Codec::vp8(97));
            }
            MediaType::Audio => {
                media.codecs.push(Codec::opus(111));
            }
            MediaType::Application => {}
        }

        answer.media.push(media);
    }

    // Set bundle group
    let mids: Vec<String> = session.tracks.iter().map(|t| t.id.clone()).collect();
    if !mids.is_empty() {
        answer.groups = vec![format!("BUNDLE {}", mids.join(" "))];
    }

    answer.to_sdp()
}

/// Generate ICE string.
fn generate_ice_string(len: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| CHARSET[rng.gen_range(0..CHARSET.len())] as char)
        .collect()
}

/// Generate mock DTLS fingerprint.
fn generate_mock_fingerprint() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| format!("{:02X}", rng.gen::<u8>()))
        .collect::<Vec<_>>()
        .join(":")
}

/// Build Link header with ICE servers.
fn build_link_header(ice_servers: &[IceServer]) -> String {
    ice_servers
        .iter()
        .flat_map(|server| {
            server.urls.iter().map(|url| {
                format!("<{}>; rel=\"ice-server\"", url)
            })
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:8080");
        assert_eq!(config.whip_path, "/whip");
        assert_eq!(config.whep_path, "/whep");
        assert!(!config.ice_servers.is_empty());
    }

    #[test]
    fn test_generate_ice_string() {
        let s = generate_ice_string(16);
        assert_eq!(s.len(), 16);
    }

    #[test]
    fn test_build_link_header() {
        let servers = vec![IceServer {
            urls: vec!["stun:stun.example.com:3478".to_string()],
            username: None,
            credential: None,
        }];
        let link = build_link_header(&servers);
        assert!(link.contains("stun:stun.example.com:3478"));
        assert!(link.contains("rel=\"ice-server\""));
    }
}
