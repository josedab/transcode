import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <div className={styles.badges}>
          <a href="https://github.com/transcode/transcode/actions">
            <img src="https://github.com/transcode/transcode/workflows/CI/badge.svg" alt="CI" />
          </a>
          <a href="https://codecov.io/gh/transcode/transcode">
            <img src="https://codecov.io/gh/transcode/transcode/branch/main/graph/badge.svg" alt="codecov" />
          </a>
          <a href="https://crates.io/crates/transcode">
            <img src="https://img.shields.io/crates/v/transcode.svg" alt="Crates.io" />
          </a>
          <a href="https://docs.rs/transcode">
            <img src="https://docs.rs/transcode/badge.svg" alt="Documentation" />
          </a>
          <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License" />
        </div>

        <div className={styles.installCommand}>
          <code>cargo add transcode</code>
        </div>

        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started/installation">
            Get Started
          </Link>
          <Link
            className="button button--outline button--lg"
            to="https://github.com/transcode/transcode">
            View on GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Memory Safe',
    icon: '🛡️',
    description: (
      <>
        Built entirely in Rust, eliminating buffer overflow vulnerabilities
        common in C/C++ codec implementations. No FFmpeg dependency.
      </>
    ),
  },
  {
    title: 'High Performance',
    icon: '⚡',
    description: (
      <>
        SIMD-optimized (AVX2, AVX-512, NEON) with automatic runtime detection.
        GPU acceleration via wgpu compute shaders.
      </>
    ),
  },
  {
    title: 'Universal Codec Support',
    icon: '🎬',
    description: (
      <>
        H.264, H.265, AV1, VP9, ProRes, and more. AAC, Opus, FLAC audio.
        MP4, MKV, HLS, DASH containers.
      </>
    ),
  },
  {
    title: 'AI Enhancement',
    icon: '🤖',
    description: (
      <>
        Neural network-based upscaling, denoising, and frame interpolation.
        Content intelligence for adaptive encoding.
      </>
    ),
  },
  {
    title: 'Distributed Processing',
    icon: '🌐',
    description: (
      <>
        Scale across multiple workers with fault tolerance.
        Coordinator/worker architecture with automatic task distribution.
      </>
    ),
  },
  {
    title: 'Multi-Platform',
    icon: '🔧',
    description: (
      <>
        Works on Linux, macOS, Windows, and WebAssembly.
        Python, Node.js, and C bindings available.
      </>
    ),
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Why Transcode?</Heading>
          <p>A modern, memory-safe alternative to traditional codec libraries</p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function QuickExample(): ReactNode {
  const rustCode = `use transcode::{Transcoder, TranscodeOptions};

fn main() -> transcode::Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)  // 5 Mbps
        .audio_bitrate(128_000);   // 128 kbps

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    println!("Processed {} frames", transcoder.stats().frames_encoded);
    Ok(())
}`;

  const pythonCode = `import transcode_py

# Simple one-liner
stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Processed {stats.frames_encoded} frames")

# With options
options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')
options = options.video_bitrate(5_000_000)
options = options.audio_bitrate(128_000)

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()
print(f"Compression ratio: {stats.compression_ratio:.2f}x")`;

  const cliCode = `# Install the CLI
cargo install transcode-cli

# Basic transcode
transcode -i input.mp4 -o output.mp4

# With options
transcode -i input.mp4 -o output.mp4 \\
  --video-codec h264 \\
  --video-bitrate 5000 \\
  --audio-codec aac \\
  --audio-bitrate 128

# Generate HLS stream
transcode -i input.mp4 --hls output/ \\
  --hls-variant "1920x1080:5000k" \\
  --hls-variant "1280x720:2500k"`;

  const dockerCode = `# Pull the official image
docker pull transcode/transcode:latest

# Basic transcode
docker run -v $(pwd):/data transcode/transcode \\
  -i /data/input.mp4 \\
  -o /data/output.mp4

# With options
docker run -v $(pwd):/data transcode/transcode \\
  -i /data/input.mp4 \\
  -o /data/output.mp4 \\
  --video-codec h264 \\
  --video-bitrate 5000

# Using Docker Compose
docker compose run --rm transcode \\
  -i /data/input.mp4 -o /data/output.mp4`;

  return (
    <section className={styles.quickExample}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Simple, Powerful API</Heading>
          <p>Get started with just a few lines of code in your preferred language</p>
        </div>
        <div className={styles.codeWrapper}>
          <Tabs>
            <TabItem value="rust" label="Rust" default>
              <CodeBlock language="rust" title="basic_transcode.rs">
                {rustCode}
              </CodeBlock>
            </TabItem>
            <TabItem value="python" label="Python">
              <CodeBlock language="python" title="transcode_example.py">
                {pythonCode}
              </CodeBlock>
            </TabItem>
            <TabItem value="cli" label="CLI">
              <CodeBlock language="bash" title="Terminal">
                {cliCode}
              </CodeBlock>
            </TabItem>
            <TabItem value="docker" label="Docker">
              <CodeBlock language="bash" title="Terminal">
                {dockerCode}
              </CodeBlock>
            </TabItem>
          </Tabs>
        </div>
      </div>
    </section>
  );
}

function ArchitectureDiagram(): ReactNode {
  return (
    <section className={styles.architecture}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Modular Architecture</Heading>
          <p>70+ specialized crates organized for flexibility and performance</p>
        </div>
        <div className={styles.architectureContent}>
          <div className={styles.architectureDiagram}>
            <img src="/img/architecture-diagram.svg" alt="Transcode Architecture" />
          </div>
          <div className={styles.architectureHighlights}>
            <div className={styles.highlightItem}>
              <div className={styles.highlightIcon}>📦</div>
              <div className={styles.highlightText}>
                <strong>Core Layer</strong>
                <span>Fundamental types, codecs, containers, and pipeline orchestration</span>
              </div>
            </div>
            <div className={styles.highlightItem}>
              <div className={styles.highlightIcon}>🎥</div>
              <div className={styles.highlightText}>
                <strong>Video Codecs</strong>
                <span>H.264, H.265, AV1, VP9, ProRes, DNxHD, and more</span>
              </div>
            </div>
            <div className={styles.highlightItem}>
              <div className={styles.highlightIcon}>🎵</div>
              <div className={styles.highlightText}>
                <strong>Audio Codecs</strong>
                <span>AAC, Opus, FLAC, AC3, DTS, and lossless formats</span>
              </div>
            </div>
            <div className={styles.highlightItem}>
              <div className={styles.highlightIcon}>⚙️</div>
              <div className={styles.highlightText}>
                <strong>Processing</strong>
                <span>GPU compute, AI enhancement, quality metrics, distributed processing</span>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.architectureCta}>
          <Link to="/docs/core-concepts/architecture" className="button button--primary">
            Explore Architecture
          </Link>
        </div>
      </div>
    </section>
  );
}

function PerformanceSection(): ReactNode {
  return (
    <section className={styles.performance}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Built for Performance</Heading>
          <p>SIMD-optimized processing with automatic hardware detection</p>
        </div>
        <div className={styles.performanceGrid}>
          <div className={styles.performanceCard}>
            <div className={styles.performanceValue}>245</div>
            <div className={styles.performanceUnit}>fps</div>
            <div className={styles.performanceLabel}>H.264 1080p Encoding (AVX2)</div>
          </div>
          <div className={styles.performanceCard}>
            <div className={styles.performanceValue}>180</div>
            <div className={styles.performanceUnit}>MB</div>
            <div className={styles.performanceLabel}>Memory Usage (1080p)</div>
          </div>
          <div className={styles.performanceCard}>
            <div className={styles.performanceValue}>5</div>
            <div className={styles.performanceUnit}>MB</div>
            <div className={styles.performanceLabel}>Core Library Size</div>
          </div>
          <div className={styles.performanceCard}>
            <div className={styles.performanceValue}>0</div>
            <div className={styles.performanceUnit}>deps</div>
            <div className={styles.performanceLabel}>System Libraries Required</div>
          </div>
        </div>
        <div className={styles.performanceFeatures}>
          <div className={styles.performanceFeature}>
            <span className={styles.checkmark}>✓</span>
            <span>AVX2, AVX-512, NEON SIMD optimization</span>
          </div>
          <div className={styles.performanceFeature}>
            <span className={styles.checkmark}>✓</span>
            <span>Zero-copy frame pools</span>
          </div>
          <div className={styles.performanceFeature}>
            <span className={styles.checkmark}>✓</span>
            <span>GPU compute shaders via wgpu</span>
          </div>
          <div className={styles.performanceFeature}>
            <span className={styles.checkmark}>✓</span>
            <span>Multi-threaded encoding/decoding</span>
          </div>
        </div>
        <div className={styles.performanceCta}>
          <Link to="/docs/reference/benchmarks" className="button button--secondary">
            View Benchmarks
          </Link>
        </div>
      </div>
    </section>
  );
}

function CodecMatrix(): ReactNode {
  return (
    <section className={styles.codecMatrix}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Comprehensive Codec Support</Heading>
          <p>Production-ready encoding and decoding for all major formats</p>
        </div>
        <div className={styles.codecTables}>
          <div className={styles.codecTable}>
            <Heading as="h3">Video Codecs</Heading>
            <table>
              <thead>
                <tr>
                  <th>Codec</th>
                  <th>Decode</th>
                  <th>Encode</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>H.264/AVC</td><td>✅</td><td>✅</td></tr>
                <tr><td>H.265/HEVC</td><td>✅</td><td>✅</td></tr>
                <tr><td>AV1</td><td>✅</td><td>✅</td></tr>
                <tr><td>VP9</td><td>✅</td><td>✅</td></tr>
                <tr><td>ProRes</td><td>✅</td><td>✅</td></tr>
                <tr><td>DNxHD/HR</td><td>✅</td><td>✅</td></tr>
              </tbody>
            </table>
          </div>
          <div className={styles.codecTable}>
            <Heading as="h3">Audio Codecs</Heading>
            <table>
              <thead>
                <tr>
                  <th>Codec</th>
                  <th>Decode</th>
                  <th>Encode</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>AAC</td><td>✅</td><td>✅</td></tr>
                <tr><td>Opus</td><td>✅</td><td>✅</td></tr>
                <tr><td>FLAC</td><td>✅</td><td>✅</td></tr>
                <tr><td>AC3/E-AC3</td><td>✅</td><td>✅</td></tr>
                <tr><td>DTS</td><td>✅</td><td>✅</td></tr>
                <tr><td>MP3</td><td>✅</td><td>—</td></tr>
              </tbody>
            </table>
          </div>
          <div className={styles.codecTable}>
            <Heading as="h3">Containers</Heading>
            <table>
              <thead>
                <tr>
                  <th>Format</th>
                  <th>Demux</th>
                  <th>Mux</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>MP4/MOV</td><td>✅</td><td>✅</td></tr>
                <tr><td>MKV/WebM</td><td>✅</td><td>✅</td></tr>
                <tr><td>HLS</td><td>—</td><td>✅</td></tr>
                <tr><td>DASH</td><td>—</td><td>✅</td></tr>
                <tr><td>MPEG-TS</td><td>✅</td><td>✅</td></tr>
                <tr><td>MXF</td><td>✅</td><td>✅</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div className={styles.codecCta}>
          <Link to="/docs/reference/codecs-matrix" className="button button--primary">
            View Full Codec Matrix
          </Link>
        </div>
      </div>
    </section>
  );
}

function Platforms(): ReactNode {
  return (
    <section className={styles.platforms}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Use Anywhere</Heading>
          <p>Native bindings for your language and platform</p>
        </div>
        <div className={styles.platformGrid}>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>🦀</div>
            <Heading as="h4">Rust</Heading>
            <code>cargo add transcode</code>
          </div>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>🐍</div>
            <Heading as="h4">Python</Heading>
            <code>pip install transcode-py</code>
          </div>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>📦</div>
            <Heading as="h4">Node.js</Heading>
            <code>npm install transcode</code>
          </div>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>🌐</div>
            <Heading as="h4">WebAssembly</Heading>
            <code>npm install transcode-wasm</code>
          </div>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>🐳</div>
            <Heading as="h4">Docker</Heading>
            <code>docker pull transcode</code>
          </div>
          <div className={styles.platformItem}>
            <div className={styles.platformIcon}>⚙️</div>
            <Heading as="h4">CLI</Heading>
            <code>cargo install transcode-cli</code>
          </div>
        </div>
      </div>
    </section>
  );
}

function UsedBy(): ReactNode {
  return (
    <section className={styles.usedBy}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Trusted by Developers</Heading>
          <p>Join the growing community building with Transcode</p>
        </div>
        <div className={styles.usedByLogos}>
          <div className={styles.usedByPlaceholder}>
            <span>Your Logo Here</span>
            <p>Are you using Transcode? <Link to="https://github.com/transcode/transcode/discussions">Let us know!</Link></p>
          </div>
        </div>
        <div className={styles.usedByStats}>
          <div className={styles.statItem}>
            <div className={styles.statValue}>70+</div>
            <div className={styles.statLabel}>Specialized Crates</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>12+</div>
            <div className={styles.statLabel}>Video Codecs</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>9+</div>
            <div className={styles.statLabel}>Audio Codecs</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>9+</div>
            <div className={styles.statLabel}>Container Formats</div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Comparison(): ReactNode {
  return (
    <section className={styles.comparison}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">How Transcode Compares</Heading>
          <p>See why developers are choosing Transcode over traditional solutions</p>
        </div>
        <div className={styles.comparisonTable}>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th className={styles.highlight}>Transcode</th>
                <th>FFmpeg</th>
                <th>GStreamer</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Language</td>
                <td className={styles.highlight}>Pure Rust</td>
                <td>C</td>
                <td>C</td>
              </tr>
              <tr>
                <td>Memory Safety</td>
                <td className={styles.highlight}>Guaranteed</td>
                <td>Manual</td>
                <td>Manual</td>
              </tr>
              <tr>
                <td>WebAssembly</td>
                <td className={styles.highlight}>First-class</td>
                <td>Limited</td>
                <td>Not supported</td>
              </tr>
              <tr>
                <td>AI Enhancement</td>
                <td className={styles.highlight}>Built-in</td>
                <td>External</td>
                <td>External</td>
              </tr>
              <tr>
                <td>Binary Size</td>
                <td className={styles.highlight}>~5MB</td>
                <td>~50MB+</td>
                <td>~100MB+</td>
              </tr>
              <tr>
                <td>System Dependencies</td>
                <td className={styles.highlight}>Minimal</td>
                <td>Many</td>
                <td>Many</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className={styles.comparisonCta}>
          <Link to="/docs/getting-started/why-transcode" className="button button--secondary">
            Full Comparison
          </Link>
          <Link to="/docs/comparison/vs-ffmpeg" className="button button--outline">
            Transcode vs FFmpeg
          </Link>
        </div>
      </div>
    </section>
  );
}

function CallToAction(): ReactNode {
  return (
    <section className={styles.cta}>
      <div className="container">
        <Heading as="h2">Ready to get started?</Heading>
        <p>Start transcoding in under 5 minutes with our comprehensive guides.</p>
        <div className={styles.ctaButtons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started/installation">
            Read the Docs
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://github.com/transcode/transcode">
            Star on GitHub
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Memory-safe Media Transcoding"
      description="A memory-safe, high-performance universal codec library written in Rust. No FFmpeg dependency. SIMD optimized. GPU accelerated.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <QuickExample />
        <ArchitectureDiagram />
        <PerformanceSection />
        <CodecMatrix />
        <Comparison />
        <Platforms />
        <UsedBy />
        <CallToAction />
      </main>
    </Layout>
  );
}
