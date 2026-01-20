# ADR-0018: Codec Conformance Testing

## Status

Accepted

## Date

2024-08

## Context

Video codec specifications (H.264, HEVC, AV1) are complex documents with subtle requirements. Implementing a codec "from scratch" risks:

1. **Spec deviations**: Misinterpreting specification text
2. **Edge cases**: Missing handling for rare but valid bitstreams
3. **Profile gaps**: Incomplete support for advanced profiles
4. **Interoperability issues**: Output that other decoders can't process

Standards bodies provide **conformance test suites**:
- ITU-T provides H.264/HEVC test vectors
- Alliance for Open Media provides AV1 test vectors
- Each test has known-correct decoded output (MD5 hashes)

Without systematic conformance testing, we cannot claim spec compliance.

## Decision

Implement a **conformance testing framework** in `transcode-conformance` that:

### 1. Manages Test Stream Repository

```rust
pub struct StreamCache {
    cache_dir: PathBuf,
    manifest: StreamManifest,
}

impl StreamCache {
    pub fn get_stream(&self, id: &str) -> Result<PathBuf>;
    pub fn download_if_missing(&self, id: &str) -> Result<PathBuf>;
    pub fn verify_checksum(&self, id: &str) -> Result<bool>;
}
```

### 2. Defines Test Vector Format

```rust
#[derive(Serialize, Deserialize)]
pub struct TestVector {
    pub id: String,
    pub profile: H264Profile,
    pub level: String,
    pub source_url: Option<String>,
    pub local_path: Option<PathBuf>,
    pub frame_checksums: Vec<String>,  // MD5 per frame
    pub expected_frames: usize,
    pub resolution: (u32, u32),
}
```

### 3. Runs Profile-Specific Tests

```rust
pub struct ConformanceRunner {
    decoder: Box<dyn VideoDecoder>,
    report: ConformanceReport,
}

impl ConformanceRunner {
    pub fn run_baseline_suite(&mut self) -> Result<ProfileResult>;
    pub fn run_main_suite(&mut self) -> Result<ProfileResult>;
    pub fn run_high_suite(&mut self) -> Result<ProfileResult>;
    pub fn run_single_test(&mut self, vector: &TestVector) -> Result<TestResult>;
}
```

### 4. Validates Decoded Output

```rust
pub fn validate_frame(frame: &Frame, expected_md5: &str) -> Result<()> {
    let actual_md5 = compute_frame_md5(frame);
    if actual_md5 != expected_md5 {
        return Err(ConformanceError::ChecksumMismatch {
            expected: expected_md5.to_string(),
            actual: actual_md5,
        });
    }
    Ok(())
}

fn compute_frame_md5(frame: &Frame) -> String {
    let mut hasher = Md5::new();
    for plane in frame.planes() {
        hasher.update(plane);
    }
    format!("{:x}", hasher.finalize())
}
```

### 5. Generates Reports

```rust
#[derive(Serialize)]
pub struct ConformanceReport {
    pub timestamp: DateTime<Utc>,
    pub decoder_version: String,
    pub profiles_tested: Vec<ProfileResult>,
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl ConformanceReport {
    pub fn to_html(&self) -> String;
    pub fn to_json(&self) -> String;
    pub fn to_junit_xml(&self) -> String;  // For CI integration
}
```

## Consequences

### Positive

1. **Verified compliance**: Can demonstrate spec conformance objectively

2. **Regression detection**: CI catches spec compliance regressions

3. **Profile coverage**: Know exactly which profiles/levels are supported

4. **Reproducibility**: Tests produce identical results across runs

5. **Documentation**: Test results document actual capabilities

### Negative

1. **Test acquisition**: Some test vectors require licensing/registration

2. **Storage requirements**: Full test suites are multi-gigabyte

3. **Execution time**: Complete suite takes hours to run

4. **Maintenance**: Test vectors updated with spec amendments

### Mitigations

1. **Caching**: Download once, reuse across runs
   ```rust
   // ~/.cache/transcode-conformance/h264/...
   ```

2. **Incremental testing**: Run only changed-codec tests in CI

3. **Profile tiers**: Quick (Baseline), Standard (Main), Full (all profiles)

4. **Skip infrastructure**: `#[ignore]` tests requiring downloads

## Implementation Details

### Test Organization

```
transcode-conformance/
├── tests/
│   ├── h264_baseline.rs
│   ├── h264_main.rs
│   ├── h264_high.rs
│   ├── hevc_main.rs
│   └── av1_main.rs
├── vectors/
│   ├── h264_manifest.json
│   ├── hevc_manifest.json
│   └── av1_manifest.json
└── src/
    ├── cache.rs
    ├── checksum.rs
    ├── download.rs
    ├── runner.rs
    └── report.rs
```

### Manifest Format

```json
{
  "codec": "h264",
  "version": "2024.1",
  "vectors": [
    {
      "id": "BAMQ1_JVC_C",
      "profile": "baseline",
      "level": "1.0",
      "source": "https://standards.iso.org/...",
      "sha256": "abc123...",
      "frames": 300,
      "resolution": [176, 144],
      "checksums": ["d41d8cd98f00b204...", "..."]
    }
  ]
}
```

### CI Integration

```yaml
# .github/workflows/conformance.yml
conformance:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/cache@v3
      with:
        path: ~/.cache/transcode-conformance
        key: conformance-vectors-${{ hashFiles('vectors/*.json') }}

    - name: Run quick conformance
      run: cargo test -p transcode-conformance --features quick

    - name: Upload report
      uses: actions/upload-artifact@v3
      with:
        name: conformance-report
        path: target/conformance-report.html
```

### Test Execution

```rust
#[test]
#[ignore] // Requires test vectors
fn test_h264_baseline_bamq1() {
    let cache = StreamCache::default();
    let stream_path = cache.get_stream("BAMQ1_JVC_C").unwrap();

    let mut decoder = H264Decoder::new().unwrap();
    let mut runner = ConformanceRunner::new(Box::new(decoder));

    let result = runner.run_single_test(&TestVector {
        id: "BAMQ1_JVC_C".into(),
        profile: H264Profile::Baseline,
        // ...
    });

    assert!(result.is_ok(), "Conformance test failed: {:?}", result);
}
```

### Bitstream Validation

Beyond frame output, validate bitstream structure:

```rust
pub struct BitstreamValidator {
    strict_mode: bool,
}

impl BitstreamValidator {
    pub fn validate_nal_structure(&self, data: &[u8]) -> Result<()>;
    pub fn validate_sps(&self, sps: &Sps) -> Result<()>;
    pub fn validate_pps(&self, pps: &Pps) -> Result<()>;
    pub fn validate_slice_header(&self, header: &SliceHeader, sps: &Sps, pps: &Pps) -> Result<()>;
}
```

## Test Coverage Matrix

### H.264 Profiles

| Profile | Status | Test Count | Notes |
|---------|--------|------------|-------|
| Baseline | Complete | 45 | Core use case |
| Constrained Baseline | Complete | 12 | WebRTC common |
| Main | Complete | 38 | B-frames |
| High | Partial | 52 | 8x8 transform |
| High 10 | Partial | 18 | 10-bit |
| High 4:2:2 | Planned | 0 | Professional |
| High 4:4:4 | Planned | 0 | Lossless |

### HEVC Profiles

| Profile | Status | Test Count |
|---------|--------|------------|
| Main | Complete | 30 |
| Main 10 | Partial | 24 |
| Main Still Picture | Complete | 8 |

## Alternatives Considered

### Alternative 1: Visual Comparison

Compare decoded frames visually or via PSNR.

Rejected because:
- Subjective; doesn't prove conformance
- Rounding differences cause false failures
- Slower than checksum comparison

### Alternative 2: Reference Decoder Comparison

Compare against FFmpeg/reference decoder output.

Rejected because:
- Reference may also have bugs
- Bit-exact match unrealistic
- Doesn't use official test vectors

### Alternative 3: No Formal Conformance

Rely on user bug reports for compliance issues.

Rejected because:
- Reactive rather than proactive
- Users may not report subtle issues
- Can't claim compliance without testing

## References

- [ITU-T H.264 Test Sequences](https://www.itu.int/rec/T-REC-H.264.2-201602-I!Amd2)
- [HEVC Conformance Streams](https://www.itu.int/rec/T-REC-H.265)
- [AV1 Test Vectors](https://aomedia.org/av1-features/av1-specification/)
- [JM Reference Software](https://iphome.hhi.de/suehring/tml/)
