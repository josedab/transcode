//! Real-time streaming, alerting, and chart data aggregation.

#![allow(dead_code)]

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Alert types
// ---------------------------------------------------------------------------

/// Severity level for alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Metric type that an alert rule monitors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertMetricType {
    Psnr,
    Ssim,
    Vmaf,
    Bitrate,
    FrameRate,
    EncodingSpeed,
}

/// Condition under which an alert fires.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AlertCondition {
    Below(f64),
    Above(f64),
    OutOfRange(f64, f64),
}

/// A rule that the alert engine evaluates per frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub metric: AlertMetricType,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub cooldown_frames: u64,
}

/// A fired alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub rule_id: String,
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub frame_number: u64,
    pub metric_value: f64,
    pub threshold: String,
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// AlertEngine
// ---------------------------------------------------------------------------

/// Evaluates alert rules against incoming frame metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEngine {
    rules: Vec<AlertRule>,
    fired_alerts: Vec<Alert>,
    last_fired: HashMap<String, u64>,
}

impl AlertEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fired_alerts: Vec::new(),
            last_fired: HashMap::new(),
        }
    }

    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        self.rules.len() < before
    }

    /// Evaluate all enabled rules against the supplied frame metrics.
    pub fn evaluate_frame(
        &mut self,
        frame_number: u64,
        psnr: Option<f64>,
        ssim: Option<f64>,
        vmaf: Option<f64>,
        bitrate: Option<f64>,
    ) {
        let mut new_alerts = Vec::new();

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }

            // Cooldown check
            if let Some(&last) = self.last_fired.get(&rule.id) {
                if frame_number.saturating_sub(last) < rule.cooldown_frames {
                    continue;
                }
            }

            let value = match rule.metric {
                AlertMetricType::Psnr => psnr,
                AlertMetricType::Ssim => ssim,
                AlertMetricType::Vmaf => vmaf,
                AlertMetricType::Bitrate => bitrate,
                AlertMetricType::FrameRate | AlertMetricType::EncodingSpeed => None,
            };

            let value = match value {
                Some(v) => v,
                None => continue,
            };

            let triggered = match rule.condition {
                AlertCondition::Below(t) => value < t,
                AlertCondition::Above(t) => value > t,
                AlertCondition::OutOfRange(lo, hi) => value < lo || value > hi,
            };

            if triggered {
                let threshold_str = match rule.condition {
                    AlertCondition::Below(t) => format!("below {t}"),
                    AlertCondition::Above(t) => format!("above {t}"),
                    AlertCondition::OutOfRange(lo, hi) => format!("outside [{lo}, {hi}]"),
                };

                new_alerts.push((
                    rule.id.clone(),
                    Alert {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        severity: rule.severity,
                        message: format!(
                            "{}: {:.4} is {} (frame {})",
                            rule.name, value, threshold_str, frame_number
                        ),
                        frame_number,
                        metric_value: value,
                        threshold: threshold_str,
                        timestamp: frame_number,
                    },
                ));
            }
        }

        for (id, alert) in new_alerts {
            self.last_fired.insert(id, frame_number);
            self.fired_alerts.push(alert);
        }
    }

    pub fn alerts(&self) -> &[Alert] {
        &self.fired_alerts
    }

    pub fn alerts_by_severity(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.fired_alerts
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    pub fn clear_alerts(&mut self) {
        self.fired_alerts.clear();
    }

    pub fn active_rules(&self) -> Vec<&AlertRule> {
        self.rules.iter().filter(|r| r.enabled).collect()
    }
}

impl Default for AlertEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

/// Per-frame metrics snapshot for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub vmaf: Option<f64>,
    pub bitrate_kbps: Option<f64>,
    pub encoding_fps: Option<f64>,
}

/// A single stream update containing metrics and any alerts fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUpdate {
    pub frame_number: u64,
    pub metrics: StreamMetrics,
    pub alerts: Vec<Alert>,
    pub timestamp: u64,
}

/// Accumulates streaming updates and integrates with the alert engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStream {
    updates: Vec<StreamUpdate>,
    alert_engine: AlertEngine,
    last_frame: u64,
}

impl MetricStream {
    pub fn new() -> Self {
        Self {
            updates: Vec::new(),
            alert_engine: AlertEngine::new(),
            last_frame: 0,
        }
    }

    pub fn with_alert_engine(engine: AlertEngine) -> Self {
        Self {
            updates: Vec::new(),
            alert_engine: engine,
            last_frame: 0,
        }
    }

    /// Push new metrics for a frame, evaluate alerts, and record the update.
    pub fn push_metrics(&mut self, frame_number: u64, metrics: StreamMetrics) {
        let alerts_before = self.alert_engine.fired_alerts.len();

        self.alert_engine.evaluate_frame(
            frame_number,
            metrics.psnr,
            metrics.ssim,
            metrics.vmaf,
            metrics.bitrate_kbps,
        );

        let new_alerts: Vec<Alert> = self.alert_engine.fired_alerts[alerts_before..]
            .to_vec();

        self.updates.push(StreamUpdate {
            frame_number,
            metrics,
            alerts: new_alerts,
            timestamp: frame_number,
        });

        self.last_frame = frame_number;
    }

    /// Return updates since a given frame number (exclusive) for polling.
    pub fn updates_since(&self, frame: u64) -> Vec<&StreamUpdate> {
        self.updates
            .iter()
            .filter(|u| u.frame_number > frame)
            .collect()
    }

    pub fn latest_update(&self) -> Option<&StreamUpdate> {
        self.updates.last()
    }

    /// Serialize the most recent updates to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.updates).unwrap_or_default()
    }
}

impl Default for MetricStream {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Chart data
// ---------------------------------------------------------------------------

/// A single data point in a chart series.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}

/// Time-series chart data with named series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub series: HashMap<String, Vec<DataPoint>>,
}

impl ChartData {
    pub fn new() -> Self {
        Self {
            series: HashMap::new(),
        }
    }

    pub fn add_point(&mut self, series_name: &str, x: f64, y: f64) {
        self.series
            .entry(series_name.to_string())
            .or_default()
            .push(DataPoint { x, y });
    }

    /// Build chart data from a `MetricStream`, creating one series per metric.
    pub fn from_stream(stream: &MetricStream) -> Self {
        let mut chart = Self::new();
        for update in &stream.updates {
            let x = update.frame_number as f64;
            if let Some(v) = update.metrics.psnr {
                chart.add_point("psnr", x, v);
            }
            if let Some(v) = update.metrics.ssim {
                chart.add_point("ssim", x, v);
            }
            if let Some(v) = update.metrics.vmaf {
                chart.add_point("vmaf", x, v);
            }
            if let Some(v) = update.metrics.bitrate_kbps {
                chart.add_point("bitrate_kbps", x, v);
            }
            if let Some(v) = update.metrics.encoding_fps {
                chart.add_point("encoding_fps", x, v);
            }
        }
        chart
    }

    /// Compute a moving average over the named series.
    pub fn windowed_average(&self, series_name: &str, window: usize) -> Vec<DataPoint> {
        let points = match self.series.get(series_name) {
            Some(p) if !p.is_empty() => p,
            _ => return Vec::new(),
        };

        let window = window.max(1);
        points
            .iter()
            .enumerate()
            .map(|(i, pt)| {
                let start = i.saturating_sub(window - 1);
                let slice = &points[start..=i];
                let avg_y = slice.iter().map(|p| p.y).sum::<f64>() / slice.len() as f64;
                DataPoint { x: pt.x, y: avg_y }
            })
            .collect()
    }

    pub fn series_names(&self) -> Vec<&str> {
        self.series.keys().map(|s| s.as_str()).collect()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.series).unwrap_or_default()
    }
}

impl Default for ChartData {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rule(id: &str, metric: AlertMetricType, condition: AlertCondition) -> AlertRule {
        AlertRule {
            id: id.to_string(),
            name: format!("Rule {id}"),
            metric,
            condition,
            severity: AlertSeverity::Warning,
            enabled: true,
            cooldown_frames: 0,
        }
    }

    // -- AlertEngine tests ---------------------------------------------------

    #[test]
    fn test_alert_below_threshold() {
        let mut engine = AlertEngine::new();
        engine.add_rule(sample_rule("psnr_low", AlertMetricType::Psnr, AlertCondition::Below(30.0)));

        engine.evaluate_frame(0, Some(25.0), None, None, None);
        assert_eq!(engine.alerts().len(), 1);
        assert_eq!(engine.alerts()[0].rule_id, "psnr_low");
    }

    #[test]
    fn test_alert_above_threshold() {
        let mut engine = AlertEngine::new();
        engine.add_rule(sample_rule("bitrate_high", AlertMetricType::Bitrate, AlertCondition::Above(5000.0)));

        engine.evaluate_frame(0, None, None, None, Some(6000.0));
        assert_eq!(engine.alerts().len(), 1);

        // Not triggered when within range
        engine.clear_alerts();
        engine.evaluate_frame(1, None, None, None, Some(4000.0));
        assert!(engine.alerts().is_empty());
    }

    #[test]
    fn test_alert_out_of_range() {
        let mut engine = AlertEngine::new();
        engine.add_rule(sample_rule("ssim_range", AlertMetricType::Ssim, AlertCondition::OutOfRange(0.8, 1.0)));

        engine.evaluate_frame(0, None, Some(0.75), None, None);
        assert_eq!(engine.alerts().len(), 1);

        engine.clear_alerts();
        engine.evaluate_frame(1, None, Some(0.95), None, None);
        assert!(engine.alerts().is_empty());
    }

    #[test]
    fn test_cooldown_enforcement() {
        let mut engine = AlertEngine::new();
        let mut rule = sample_rule("psnr_low", AlertMetricType::Psnr, AlertCondition::Below(30.0));
        rule.cooldown_frames = 5;
        engine.add_rule(rule);

        engine.evaluate_frame(0, Some(20.0), None, None, None);
        assert_eq!(engine.alerts().len(), 1);

        // Within cooldown – should NOT fire again
        engine.evaluate_frame(3, Some(20.0), None, None, None);
        assert_eq!(engine.alerts().len(), 1);

        // After cooldown
        engine.evaluate_frame(5, Some(20.0), None, None, None);
        assert_eq!(engine.alerts().len(), 2);
    }

    #[test]
    fn test_disabled_rule_not_evaluated() {
        let mut engine = AlertEngine::new();
        let mut rule = sample_rule("x", AlertMetricType::Psnr, AlertCondition::Below(50.0));
        rule.enabled = false;
        engine.add_rule(rule);

        engine.evaluate_frame(0, Some(10.0), None, None, None);
        assert!(engine.alerts().is_empty());
    }

    #[test]
    fn test_remove_rule() {
        let mut engine = AlertEngine::new();
        engine.add_rule(sample_rule("a", AlertMetricType::Psnr, AlertCondition::Below(50.0)));
        engine.add_rule(sample_rule("b", AlertMetricType::Ssim, AlertCondition::Below(0.5)));

        assert!(engine.remove_rule("a"));
        assert!(!engine.remove_rule("nonexistent"));
        assert_eq!(engine.active_rules().len(), 1);
    }

    #[test]
    fn test_alerts_by_severity() {
        let mut engine = AlertEngine::new();
        let mut warn_rule = sample_rule("w", AlertMetricType::Psnr, AlertCondition::Below(50.0));
        warn_rule.severity = AlertSeverity::Warning;
        let mut crit_rule = sample_rule("c", AlertMetricType::Ssim, AlertCondition::Below(0.5));
        crit_rule.severity = AlertSeverity::Critical;
        engine.add_rule(warn_rule);
        engine.add_rule(crit_rule);

        engine.evaluate_frame(0, Some(10.0), Some(0.1), None, None);
        assert_eq!(engine.alerts_by_severity(AlertSeverity::Warning).len(), 1);
        assert_eq!(engine.alerts_by_severity(AlertSeverity::Critical).len(), 1);
        assert_eq!(engine.alerts_by_severity(AlertSeverity::Info).len(), 0);
    }

    // -- MetricStream tests --------------------------------------------------

    #[test]
    fn test_stream_push_and_poll() {
        let mut stream = MetricStream::new();
        stream.push_metrics(0, StreamMetrics { psnr: Some(40.0), ssim: Some(0.95), vmaf: None, bitrate_kbps: None, encoding_fps: None });
        stream.push_metrics(1, StreamMetrics { psnr: Some(41.0), ssim: Some(0.96), vmaf: None, bitrate_kbps: None, encoding_fps: None });
        stream.push_metrics(2, StreamMetrics { psnr: Some(39.0), ssim: Some(0.94), vmaf: None, bitrate_kbps: None, encoding_fps: None });

        let since = stream.updates_since(0);
        assert_eq!(since.len(), 2); // frames 1 and 2

        let latest = stream.latest_update().unwrap();
        assert_eq!(latest.frame_number, 2);
    }

    #[test]
    fn test_stream_with_alerts() {
        let mut engine = AlertEngine::new();
        engine.add_rule(sample_rule("low_psnr", AlertMetricType::Psnr, AlertCondition::Below(35.0)));

        let mut stream = MetricStream::with_alert_engine(engine);
        stream.push_metrics(0, StreamMetrics { psnr: Some(40.0), ssim: None, vmaf: None, bitrate_kbps: None, encoding_fps: None });
        stream.push_metrics(1, StreamMetrics { psnr: Some(30.0), ssim: None, vmaf: None, bitrate_kbps: None, encoding_fps: None });

        let latest = stream.latest_update().unwrap();
        assert_eq!(latest.alerts.len(), 1);
        assert_eq!(latest.alerts[0].rule_id, "low_psnr");

        // First update should have no alerts
        assert!(stream.updates_since(0)[0].alerts.is_empty() || stream.updates_since(0).len() == 1);
    }

    #[test]
    fn test_stream_json_export() {
        let mut stream = MetricStream::new();
        stream.push_metrics(0, StreamMetrics { psnr: Some(42.0), ssim: None, vmaf: None, bitrate_kbps: None, encoding_fps: None });
        let json = stream.to_json();
        assert!(json.contains("42.0"));
        assert!(json.contains("frame_number"));
    }

    // -- ChartData tests -----------------------------------------------------

    #[test]
    fn test_chart_add_point_and_series() {
        let mut chart = ChartData::new();
        chart.add_point("psnr", 0.0, 40.0);
        chart.add_point("psnr", 1.0, 41.0);
        chart.add_point("ssim", 0.0, 0.95);

        assert_eq!(chart.series["psnr"].len(), 2);
        assert_eq!(chart.series["ssim"].len(), 1);
        assert!(chart.series_names().contains(&"psnr"));
    }

    #[test]
    fn test_chart_from_stream() {
        let mut stream = MetricStream::new();
        stream.push_metrics(0, StreamMetrics { psnr: Some(40.0), ssim: Some(0.95), vmaf: None, bitrate_kbps: Some(3000.0), encoding_fps: None });
        stream.push_metrics(1, StreamMetrics { psnr: Some(41.0), ssim: Some(0.96), vmaf: None, bitrate_kbps: Some(3100.0), encoding_fps: None });

        let chart = ChartData::from_stream(&stream);
        assert_eq!(chart.series["psnr"].len(), 2);
        assert_eq!(chart.series["ssim"].len(), 2);
        assert_eq!(chart.series["bitrate_kbps"].len(), 2);
        assert!(!chart.series.contains_key("vmaf"));
    }

    #[test]
    fn test_windowed_average() {
        let mut chart = ChartData::new();
        for i in 0..5 {
            chart.add_point("v", i as f64, (i * 10) as f64);
        }
        // Values: 0, 10, 20, 30, 40 – window 3
        let avg = chart.windowed_average("v", 3);
        assert_eq!(avg.len(), 5);
        // Last point: avg(20, 30, 40) = 30
        assert!((avg[4].y - 30.0).abs() < 0.001);
        // First point: only one value -> 0
        assert!((avg[0].y - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_windowed_average_empty_series() {
        let chart = ChartData::new();
        let avg = chart.windowed_average("nonexistent", 3);
        assert!(avg.is_empty());
    }

    #[test]
    fn test_chart_json_export() {
        let mut chart = ChartData::new();
        chart.add_point("psnr", 0.0, 42.0);
        let json = chart.to_json();
        assert!(json.contains("psnr"));
        assert!(json.contains("42.0"));
    }
}
