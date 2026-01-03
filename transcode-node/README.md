# @transcode/node

Node.js bindings for the [Transcode](https://github.com/transcode/transcode) codec library - a memory-safe, high-performance universal codec library written in Rust.

## Features

- **High Performance**: Built on Rust with SIMD optimizations
- **Memory Safe**: Leverages Rust's memory safety guarantees
- **Async/Promise API**: Native async support for non-blocking operations
- **Progress Events**: Real-time progress callbacks during transcoding
- **Cross-Platform**: Pre-built binaries for Windows, macOS, and Linux

## Installation

```bash
npm install @transcode/node
```

Pre-built binaries are available for:
- Windows (x64, arm64)
- macOS (x64, arm64)
- Linux (x64, arm64, musl)

## Quick Start

```javascript
const { transcode, probe } = require('@transcode/node');

// Probe a media file
const info = await probe('input.mp4');
console.log(`Duration: ${info.duration}s`);
console.log(`Resolution: ${info.videoStreams[0].width}x${info.videoStreams[0].height}`);

// Transcode with progress callback
await transcode('input.mp4', 'output.mp4', {
  videoBitrate: 5_000_000,
  audioCodec: 'aac',
  onProgress: (p) => console.log(`${p.percent.toFixed(1)}%`)
});
```

## API Reference

### `probe(input: string): Promise<MediaInfo>`

Probe a media file to get information about its streams.

```javascript
const info = await probe('video.mp4');
console.log(info);
// {
//   format: 'mp4',
//   duration: 120.5,
//   size: 52428800,
//   videoStreams: [{
//     index: 0,
//     codec: 'h264',
//     width: 1920,
//     height: 1080,
//     frameRate: 29.97,
//     bitDepth: 8
//   }],
//   audioStreams: [{
//     index: 1,
//     codec: 'aac',
//     sampleRate: 48000,
//     channels: 2,
//     bitsPerSample: 16
//   }],
//   bitrate: 3500000
// }
```

### `transcode(input: string, output: string, options?: TranscodeOptions): Promise<TranscodeStats>`

Transcode a media file.

```javascript
const stats = await transcode('input.mp4', 'output.mp4', {
  videoCodec: 'h264',
  videoBitrate: 5_000_000,
  audioCodec: 'aac',
  audioBitrate: 128_000,
  width: 1920,
  height: 1080,
  overwrite: true
});

console.log(`Compression ratio: ${stats.compressionRatio.toFixed(2)}x`);
```

### `extractThumbnail(input: string, timestamp: number, output?: string): Promise<string>`

Extract a thumbnail from a video file.

```javascript
const thumbnailPath = await extractThumbnail('video.mp4', 10.5);
console.log(`Thumbnail saved to: ${thumbnailPath}`);
```

### `new Transcoder(input: string, output: string, options?: TranscodeOptions)`

Class-based API for more control over transcoding.

```javascript
const { Transcoder } = require('@transcode/node');

const transcoder = new Transcoder('input.mp4', 'output.mp4', {
  videoBitrate: 5_000_000
});

// Run with progress callback
const stats = await transcoder.run((progress) => {
  console.log(`Progress: ${progress.percent}%`);
  console.log(`Speed: ${progress.speed}x`);
  console.log(`ETA: ${progress.eta}s`);
});

// Cancel transcoding
transcoder.cancel();
```

### `detectSimd(): SimdCapabilities`

Detect SIMD capabilities of the current CPU.

```javascript
const simd = detectSimd();
if (simd.avx2) {
  console.log('AVX2 acceleration available');
}
```

### `version(): string`

Get the library version.

```javascript
console.log(version()); // "0.1.0"
```

### `buildInfo(): BuildInfo`

Get build information.

```javascript
const info = buildInfo();
console.log(info);
// { version: "0.1.0", arch: "x64", os: "darwin", debug: false }
```

## TranscodeOptions

| Option | Type | Description |
|--------|------|-------------|
| `videoCodec` | `string` | Video codec (e.g., "h264", "h265", "av1") |
| `audioCodec` | `string` | Audio codec (e.g., "aac", "mp3", "opus") |
| `videoBitrate` | `number` | Video bitrate in bits per second |
| `audioBitrate` | `number` | Audio bitrate in bits per second |
| `width` | `number` | Output width in pixels |
| `height` | `number` | Output height in pixels |
| `frameRate` | `number` | Frame rate |
| `sampleRate` | `number` | Audio sample rate in Hz |
| `channels` | `number` | Number of audio channels |
| `threads` | `number` | Number of encoding threads (0 = auto) |
| `hardwareAcceleration` | `boolean` | Enable hardware acceleration |
| `overwrite` | `boolean` | Overwrite output file if exists |
| `startTime` | `number` | Start time in seconds (for trimming) |
| `duration` | `number` | Duration in seconds (for trimming) |
| `preset` | `string` | Encoder preset (e.g., "ultrafast", "medium", "slow") |
| `crf` | `number` | CRF value for quality-based encoding (0-51) |
| `onProgress` | `function` | Progress callback |

## Building from Source

### Prerequisites

- Node.js >= 14
- Rust toolchain
- napi-rs CLI: `npm install -g @napi-rs/cli`

### Build

```bash
cd transcode-node
npm install
npm run build
```

### Test

```bash
npm test
```

## License

MIT OR Apache-2.0
