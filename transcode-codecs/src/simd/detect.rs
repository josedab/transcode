//! Runtime SIMD feature detection.

/// Detected SIMD capabilities for the current CPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdCapabilities {
    // x86_64 features
    /// SSE4.2 support (baseline for modern x86_64).
    pub sse42: bool,
    /// AVX2 support (256-bit integer SIMD).
    pub avx2: bool,
    /// AVX-512 support (512-bit SIMD).
    pub avx512: bool,
    /// FMA (Fused Multiply-Add) support.
    pub fma: bool,

    // ARM features
    /// NEON support (baseline for AArch64).
    pub neon: bool,
    /// SVE support (Scalable Vector Extension).
    pub sve: bool,

    // WebAssembly
    /// WASM SIMD128 support.
    pub wasm_simd: bool,
}

impl SimdCapabilities {
    /// Check if any SIMD acceleration is available.
    pub fn has_simd(&self) -> bool {
        self.sse42 || self.avx2 || self.avx512 || self.neon || self.sve || self.wasm_simd
    }

    /// Get the best available SIMD level as a string.
    pub fn best_level(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.sse42 {
            "SSE4.2"
        } else if self.sve {
            "SVE"
        } else if self.neon {
            "NEON"
        } else if self.wasm_simd {
            "WASM SIMD"
        } else {
            "Scalar"
        }
    }
}

/// Detect SIMD capabilities at runtime.
#[cfg(target_arch = "x86_64")]
pub fn detect_simd() -> SimdCapabilities {
    let mut caps = SimdCapabilities::default();

    // Use std::arch for runtime detection
    if is_x86_feature_detected!("sse4.2") {
        caps.sse42 = true;
    }
    if is_x86_feature_detected!("avx2") {
        caps.avx2 = true;
    }
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        caps.avx512 = true;
    }
    if is_x86_feature_detected!("fma") {
        caps.fma = true;
    }

    caps
}

/// Detect SIMD capabilities at runtime (ARM).
#[cfg(target_arch = "aarch64")]
pub fn detect_simd() -> SimdCapabilities {
    // NEON is always available on AArch64, SVE detection requires specific OS support
    #[cfg(target_os = "linux")]
    {
        // On Linux, check for SVE via aux vector or hwcap
        // For now, conservatively assume no SVE
        SimdCapabilities {
            neon: true,
            sve: false,
            ..Default::default()
        }
    }

    #[cfg(not(target_os = "linux"))]
    SimdCapabilities {
        neon: true,
        ..Default::default()
    }
}

/// Detect SIMD capabilities at runtime (WebAssembly).
#[cfg(target_arch = "wasm32")]
pub fn detect_simd() -> SimdCapabilities {
    let mut caps = SimdCapabilities::default();

    // WASM SIMD is a compile-time feature
    #[cfg(target_feature = "simd128")]
    {
        caps.wasm_simd = true;
    }

    caps
}

/// Fallback for unsupported architectures.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub fn detect_simd() -> SimdCapabilities {
    SimdCapabilities::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let caps = detect_simd();
        println!("Detected: {:?}", caps);
        println!("Best level: {}", caps.best_level());

        #[cfg(target_arch = "x86_64")]
        {
            // Modern x86_64 should have at least SSE4.2
            assert!(caps.sse42 || !caps.has_simd());
        }

        #[cfg(target_arch = "aarch64")]
        {
            // AArch64 always has NEON
            assert!(caps.neon);
        }
    }
}
