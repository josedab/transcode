//! SIMD capability detection example.
//!
//! This example demonstrates how to detect the SIMD capabilities
//! of the current CPU and understand what optimizations are available.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example simd_detection
//! ```

use transcode_codecs::detect_simd;

fn main() {
    println!("SIMD Capability Detection");
    println!("══════════════════════════════════════════════════════════");
    println!();

    let caps = detect_simd();

    // Display architecture
    #[cfg(target_arch = "x86_64")]
    println!("Architecture: x86_64");
    #[cfg(target_arch = "aarch64")]
    println!("Architecture: aarch64 (ARM64)");
    #[cfg(target_arch = "wasm32")]
    println!("Architecture: wasm32 (WebAssembly)");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    println!("Architecture: Other");

    println!();
    println!("Detected SIMD Features:");
    println!("────────────────────────────────────────────────────────────");

    // x86_64 features
    println!("  SSE4.2:   {}", feature_status(caps.sse42));
    println!("  AVX2:     {}", feature_status(caps.avx2));
    println!("  AVX-512:  {}", feature_status(caps.avx512));
    println!("  FMA:      {}", feature_status(caps.fma));

    // ARM features
    println!("  NEON:     {}", feature_status(caps.neon));
    println!("  SVE:      {}", feature_status(caps.sve));

    println!();
    println!("────────────────────────────────────────────────────────────");
    println!("Best available SIMD level: {}", caps.best_level());
    println!("Has any SIMD support:      {}", if caps.has_simd() { "Yes" } else { "No" });
    println!();

    // Performance implications
    println!("Performance Implications:");
    println!("────────────────────────────────────────────────────────────");

    if caps.avx512 {
        println!("  ✓ AVX-512 available - Maximum x86_64 performance");
        println!("    - 512-bit vector operations");
        println!("    - Best for large batch processing");
    } else if caps.avx2 {
        println!("  ✓ AVX2 available - Excellent x86_64 performance");
        println!("    - 256-bit vector operations");
        println!("    - Great for most workloads");
    } else if caps.sse42 {
        println!("  ○ SSE4.2 available - Good x86_64 performance");
        println!("    - 128-bit vector operations");
        println!("    - Adequate for most tasks");
    }

    if caps.sve {
        println!("  ✓ SVE available - Maximum ARM performance");
        println!("    - Scalable vector length");
        println!("    - Best for modern ARM servers");
    } else if caps.neon {
        println!("  ✓ NEON available - Excellent ARM performance");
        println!("    - 128-bit vector operations");
        println!("    - Standard on modern ARM CPUs");
    }

    if !caps.has_simd() {
        println!("  ⚠ No SIMD support detected");
        println!("    - Using scalar fallback implementations");
        println!("    - Performance will be significantly reduced");
    }

    println!();
    println!("══════════════════════════════════════════════════════════");
}

fn feature_status(available: bool) -> &'static str {
    if available {
        "✓ Available"
    } else {
        "✗ Not available"
    }
}
