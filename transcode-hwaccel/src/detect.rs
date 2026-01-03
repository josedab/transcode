//! Hardware accelerator detection.

use crate::{HwAccelError, HwAccelType, HwCapabilities, HwCodec, Result};

/// Information about an available hardware accelerator.
#[derive(Debug, Clone)]
pub struct HwAccelInfo {
    /// Accelerator type.
    pub accel_type: HwAccelType,
    /// Device index (for multi-GPU systems).
    pub device_index: u32,
    /// Full capabilities.
    pub capabilities: HwCapabilities,
    /// Priority (higher = preferred).
    pub priority: u32,
}

/// Detect all available hardware accelerators.
pub fn detect_accelerators() -> Vec<HwAccelInfo> {
    let mut accelerators = Vec::new();

    // Check platform-specific accelerators
    #[cfg(target_os = "linux")]
    {
        if let Some(info) = detect_vaapi() {
            accelerators.push(info);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(info) = detect_videotoolbox() {
            accelerators.push(info);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(info) = detect_d3d11va() {
            accelerators.push(info);
        }
    }

    // Check for NVIDIA (cross-platform)
    if let Some(info) = detect_nvenc() {
        accelerators.push(info);
    }

    // Check for Intel QSV (cross-platform)
    if let Some(info) = detect_qsv() {
        accelerators.push(info);
    }

    // Sort by priority (descending)
    accelerators.sort_by(|a, b| b.priority.cmp(&a.priority));

    accelerators
}

/// Detect the best available accelerator.
pub fn detect_best() -> Option<HwAccelInfo> {
    detect_accelerators().into_iter().next()
}

/// Check if a specific accelerator is available.
pub fn is_available(accel_type: HwAccelType) -> bool {
    detect_accelerators()
        .iter()
        .any(|a| a.accel_type == accel_type)
}

/// Get capabilities for a specific accelerator.
pub fn get_capabilities(accel_type: HwAccelType) -> Result<HwCapabilities> {
    detect_accelerators()
        .into_iter()
        .find(|a| a.accel_type == accel_type)
        .map(|a| a.capabilities)
        .ok_or(HwAccelError::NotSupported(accel_type.name().to_string()))
}

// Platform-specific detection functions

#[cfg(target_os = "linux")]
fn detect_vaapi() -> Option<HwAccelInfo> {
    // Try to open VA-API device
    // In a real implementation, this would use libva

    // Check if /dev/dri/renderD128 exists
    if std::path::Path::new("/dev/dri/renderD128").exists() {
        Some(HwAccelInfo {
            accel_type: HwAccelType::Vaapi,
            device_index: 0,
            capabilities: HwCapabilities {
                accel_type: HwAccelType::Vaapi,
                encode_codecs: vec![HwCodec::H264, HwCodec::Hevc],
                decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
                max_width: 4096,
                max_height: 2160,
                supports_bframes: true,
                supports_10bit: true,
                supports_hdr: false,
                device_name: "VA-API (Intel/AMD)".to_string(),
            },
            priority: 80,
        })
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn detect_videotoolbox() -> Option<HwAccelInfo> {
    // VideoToolbox is always available on macOS
    Some(HwAccelInfo {
        accel_type: HwAccelType::VideoToolbox,
        device_index: 0,
        capabilities: HwCapabilities {
            accel_type: HwAccelType::VideoToolbox,
            encode_codecs: vec![HwCodec::H264, HwCodec::Hevc],
            decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9],
            max_width: 8192,
            max_height: 4320,
            supports_bframes: true,
            supports_10bit: true,
            supports_hdr: true,
            device_name: "Apple VideoToolbox".to_string(),
        },
        priority: 100,
    })
}

#[cfg(target_os = "windows")]
fn detect_d3d11va() -> Option<HwAccelInfo> {
    // D3D11VA is available on Windows with DirectX 11
    Some(HwAccelInfo {
        accel_type: HwAccelType::D3d11va,
        device_index: 0,
        capabilities: HwCapabilities {
            accel_type: HwAccelType::D3d11va,
            encode_codecs: vec![],
            decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9],
            max_width: 4096,
            max_height: 2160,
            supports_bframes: true,
            supports_10bit: true,
            supports_hdr: true,
            device_name: "D3D11 Video Acceleration".to_string(),
        },
        priority: 70,
    })
}

fn detect_nvenc() -> Option<HwAccelInfo> {
    // Check for NVIDIA GPU
    // In a real implementation, this would use CUDA/NVML

    #[cfg(target_os = "linux")]
    let nvidia_exists = std::path::Path::new("/dev/nvidia0").exists();

    #[cfg(target_os = "windows")]
    let nvidia_exists = true; // Would check registry/driver

    #[cfg(target_os = "macos")]
    let nvidia_exists = false; // No NVIDIA on modern macOS

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    let nvidia_exists = false;

    if nvidia_exists {
        Some(HwAccelInfo {
            accel_type: HwAccelType::Nvenc,
            device_index: 0,
            capabilities: HwCapabilities {
                accel_type: HwAccelType::Nvenc,
                encode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Av1],
                decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
                max_width: 8192,
                max_height: 8192,
                supports_bframes: true,
                supports_10bit: true,
                supports_hdr: true,
                device_name: "NVIDIA NVENC/NVDEC".to_string(),
            },
            priority: 90,
        })
    } else {
        None
    }
}

fn detect_qsv() -> Option<HwAccelInfo> {
    // Check for Intel QSV
    // In a real implementation, this would use the Intel Media SDK

    #[cfg(target_os = "linux")]
    {
        // Check for i915 driver
        let has_intel = std::path::Path::new("/dev/dri/renderD128").exists()
            && std::fs::read_to_string("/sys/class/drm/renderD128/device/vendor")
                .map(|v| v.trim() == "0x8086")
                .unwrap_or(false);

        if has_intel {
            return Some(HwAccelInfo {
                accel_type: HwAccelType::Qsv,
                device_index: 0,
                capabilities: HwCapabilities {
                    accel_type: HwAccelType::Qsv,
                    encode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Av1],
                    decode_codecs: vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1],
                    max_width: 4096,
                    max_height: 2160,
                    supports_bframes: true,
                    supports_10bit: true,
                    supports_hdr: false,
                    device_name: "Intel Quick Sync Video".to_string(),
                },
                priority: 85,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_accelerators() {
        let accels = detect_accelerators();
        // Should at least detect VideoToolbox on macOS
        #[cfg(target_os = "macos")]
        assert!(!accels.is_empty());

        // Print detected accelerators
        for accel in &accels {
            println!(
                "Detected: {} (priority: {})",
                accel.accel_type.name(),
                accel.priority
            );
        }
    }

    #[test]
    fn test_detect_best() {
        if let Some(best) = detect_best() {
            println!("Best accelerator: {}", best.accel_type.name());
        }
    }
}
