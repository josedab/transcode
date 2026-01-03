//! WASM utility functions.

#![allow(dead_code)]

/// Set panic hook for better error messages in development.
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format duration in milliseconds as human-readable string.
pub fn format_duration_ms(ms: f64) -> String {
    if ms >= 60_000.0 {
        let minutes = (ms / 60_000.0).floor();
        let seconds = ((ms % 60_000.0) / 1000.0).floor();
        format!("{}m {:.0}s", minutes, seconds)
    } else if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.2}ms", ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_duration_ms() {
        assert_eq!(format_duration_ms(500.0), "500.00ms");
        assert_eq!(format_duration_ms(1500.0), "1.50s");
        assert_eq!(format_duration_ms(65000.0), "1m 5s");
    }
}
