# ADR-0010: Multi-Metric Quality Assessment

## Status

Accepted

## Date

2024-06 (inferred from module structure)

## Context

Video transcoding inherently involves quality trade-offs. We need objective metrics to:

1. **Validate encoding quality** against reference
2. **Compare codec/encoder configurations**
3. **Automate quality monitoring** in production
4. **Guide rate control** decisions

Different metrics offer different trade-offs:

| Metric | Speed | Correlation with Human Perception |
|--------|-------|----------------------------------|
| PSNR | Very Fast | Low |
| SSIM | Fast | Medium |
| MS-SSIM | Medium | High |
| VMAF | Slow | Very High |

We need to support multiple metrics to serve different use cases: fast CI checks vs. thorough quality analysis.

## Decision

Implement **four quality metrics** (PSNR, SSIM, MS-SSIM, VMAF) with a **unified assessment API** and **batch processing** support.

### 1. Unified Quality Assessment

Single entry point for all metrics:

```rust
pub struct QualityAssessment {
    config: QualityConfig,
    psnr_calc: Psnr,
    ssim_calc: Ssim,
    ms_ssim_calc: MsSsim,
    vmaf_calc: Vmaf,
}

impl QualityAssessment {
    pub fn assess(&self, reference: &Frame, distorted: &Frame) -> Result<QualityReport> {
        let psnr = if self.config.metrics.psnr {
            Some(self.psnr_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        let ssim = if self.config.metrics.ssim {
            Some(self.ssim_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        // ... MS-SSIM, VMAF

        Ok(QualityReport { psnr, ssim, ms_ssim, vmaf })
    }

    pub fn assess_fast(&self, reference: &Frame, distorted: &Frame) -> Result<QualityReport> {
        // Only PSNR and SSIM for quick checks
    }
}
```

### 2. Configurable Metric Selection

Choose which metrics to compute:

```rust
pub struct QualityMetrics {
    pub psnr: bool,
    pub ssim: bool,
    pub ms_ssim: bool,
    pub vmaf: bool,
}

impl QualityMetrics {
    pub fn all() -> Self {
        Self { psnr: true, ssim: true, ms_ssim: true, vmaf: true }
    }

    pub fn fast() -> Self {
        Self { psnr: true, ssim: true, ms_ssim: false, vmaf: false }
    }
}
```

### 3. PSNR Implementation

Peak Signal-to-Noise Ratio with per-channel support:

```rust
pub struct PsnrResult {
    pub psnr: f64,           // Overall PSNR in dB
    pub psnr_y: f64,         // Luma PSNR
    pub psnr_u: Option<f64>, // Chroma U PSNR
    pub psnr_v: Option<f64>, // Chroma V PSNR
    pub mse: f64,            // Mean Squared Error
}

impl Psnr {
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<PsnrResult> {
        let mse = self.compute_mse(reference, distorted)?;

        let max_value = (1 << self.config.bit_depth) - 1;
        let psnr = if mse > 0.0 {
            10.0 * ((max_value as f64).powi(2) / mse).log10()
        } else {
            f64::INFINITY  // Identical frames
        };

        Ok(PsnrResult { psnr, mse, /* ... */ })
    }
}
```

### 4. SSIM Implementation

Structural Similarity with Gaussian windowing:

```rust
pub struct SsimConfig {
    pub window_size: usize,   // Default: 11
    pub sigma: f64,           // Default: 1.5
    pub k1: f64,              // Default: 0.01
    pub k2: f64,              // Default: 0.03
    pub bit_depth: u8,
}

pub struct SsimResult {
    pub ssim: f64,            // Overall SSIM (-1 to 1)
    pub ssim_map: Option<Vec<f64>>, // Per-pixel SSIM
}

impl Ssim {
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<SsimResult> {
        let window = self.gaussian_window();

        // SSIM = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
        let c1 = (self.config.k1 * max_value).powi(2);
        let c2 = (self.config.k2 * max_value).powi(2);

        // Compute local statistics with Gaussian weighting
        // ...
    }
}
```

### 5. MS-SSIM Implementation

Multi-Scale SSIM for better correlation with perception:

```rust
pub struct MsSsim {
    scales: usize,          // Default: 5
    ssim_calc: Ssim,
    weights: Vec<f64>,      // Per-scale weights
}

impl MsSsim {
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<f64> {
        let mut scores = Vec::new();

        for scale in 0..self.scales {
            let ref_scaled = self.downsample(reference, scale);
            let dist_scaled = self.downsample(distorted, scale);

            let ssim = self.ssim_calc.calculate(&ref_scaled, &dist_scaled)?;
            scores.push(ssim.ssim);
        }

        // Weighted product across scales
        Ok(self.weighted_product(&scores))
    }
}
```

### 6. VMAF Implementation

Video Multi-Method Assessment Fusion (approximation):

```rust
pub struct VmafConfig {
    pub model: VmafModel,
    pub pool_method: PoolMethod,
    pub enable_transform: bool,
}

pub enum VmafModel {
    Default,           // vmaf_v0.6.1
    Phone,             // Mobile-optimized
    Neg,               // Temporal distortion
    Custom(PathBuf),   // Custom model
}

pub struct VmafResult {
    pub score: f64,           // 0-100 scale
    pub vif_score: f64,       // Visual Information Fidelity
    pub dlm_score: f64,       // Detail Loss Metric
    pub motion_score: f64,    // Temporal component
}

impl Vmaf {
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<VmafResult> {
        // Compute feature scores
        let vif = self.compute_vif(reference, distorted)?;
        let dlm = self.compute_dlm(reference, distorted)?;
        let motion = self.compute_motion(reference)?;

        // Fuse features with trained model
        let score = self.model.predict(vif, dlm, motion)?;

        Ok(VmafResult { score, vif_score: vif, dlm_score: dlm, motion_score: motion })
    }
}
```

### 7. Batch Processing

Assess quality across entire videos:

```rust
pub struct BatchQualityAssessment {
    pub frame_scores: Vec<QualityReport>,
}

impl BatchQualityAssessment {
    pub fn average_psnr(&self) -> Option<f64>;
    pub fn average_ssim(&self) -> Option<f64>;
    pub fn average_vmaf(&self) -> Option<f64>;
    pub fn min_scores(&self) -> (Option<f64>, Option<f64>, Option<f64>);

    pub fn summary(&self) -> String {
        // Batch Quality Summary
        // =====================
        // Frames analyzed: 1800
        //
        // Average PSNR: 42.5 dB
        // Average SSIM: 0.9823
        // Average VMAF: 93.2
        //
        // Minimum scores (worst frames):
        //   PSNR: 38.2 dB
        //   SSIM: 0.9512
        //   VMAF: 85.4
    }
}
```

### 8. Quality Reporting

Human-readable quality assessments:

```rust
pub struct QualityReport {
    pub psnr: Option<PsnrResult>,
    pub ssim: Option<SsimResult>,
    pub ms_ssim: Option<f64>,
    pub vmaf: Option<VmafResult>,
}

impl QualityReport {
    pub fn overall_score(&self) -> f64 {
        // Prefer VMAF > SSIM > PSNR for overall quality
        if let Some(ref vmaf) = self.vmaf {
            return vmaf.score;
        }
        if let Some(ref ssim) = self.ssim {
            return ssim.ssim * 100.0;
        }
        // ...
    }

    pub fn rating(&self) -> &'static str {
        let score = self.overall_score();
        if score >= 93.0 { "Excellent" }
        else if score >= 80.0 { "Good" }
        else if score >= 60.0 { "Fair" }
        else { "Poor" }
    }
}
```

## Consequences

### Positive

1. **Comprehensive assessment**: Multiple perspectives on quality

2. **Speed/accuracy trade-off**: Choose metrics based on use case

3. **Industry alignment**: VMAF is Netflix's standard, widely understood

4. **Batch analysis**: Identify quality drops across video duration

5. **Actionable output**: Clear ratings and statistics for decision-making

6. **Automation-friendly**: Consistent numeric scores for CI/CD

### Negative

1. **VMAF complexity**: Full VMAF requires machine learning models

2. **Computational cost**: MS-SSIM and VMAF are CPU-intensive

3. **Reference required**: All metrics need original source for comparison

4. **Perception variance**: No metric perfectly matches human perception

### Mitigations

1. **Fast mode**: Default to PSNR+SSIM for quick checks

2. **Sampling**: Option to assess subset of frames

3. **Caching**: Cache intermediate computations

4. **Parallel processing**: Use SIMD for metric calculations

## Alternatives Considered

### Alternative 1: PSNR Only

Use only PSNR for simplicity.

Rejected because:
- Poor correlation with human perception
- Doesn't detect structural artifacts well
- Industry moving away from PSNR-only assessment

### Alternative 2: External Tools (ffmpeg)

Shell out to ffmpeg for metrics.

Rejected because:
- Process overhead for per-frame metrics
- Less control over implementation
- Dependency on external tool availability

### Alternative 3: Full VMAF Library

Integrate Netflix's libvmaf directly.

Rejected because:
- Large native dependency
- Complex build process
- Reduced portability

### Alternative 4: Machine Learning Only

Use only ML-based metrics like VMAF.

Rejected because:
- Too slow for quick checks
- Requires model files
- Simple metrics still valuable for debugging

## References

- [VMAF by Netflix](https://github.com/Netflix/vmaf)
- [SSIM paper](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)
- [MS-SSIM paper](https://live.ece.utexas.edu/research/Quality/msssim.pdf)
- [PSNR definition](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [Video quality assessment survey](https://arxiv.org/abs/1803.01536)
