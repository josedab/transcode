# transcode-cli

Command-line interface for the Transcode codec library. A memory-safe, high-performance media transcoding tool built in Rust.

## Installation

```bash
cargo install --path .
```

## Usage

```bash
transcode -i <input> -o <output> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-i, --input <FILE>` | Input file path (required) |
| `-o, --output <FILE>` | Output file path (required) |
| `--video-codec <CODEC>` | Video codec: `h264`, `h265` (default: `h264`) |
| `--audio-codec <CODEC>` | Audio codec: `aac`, `mp3` (default: `aac`) |
| `--video-bitrate <KBPS>` | Video bitrate in kbps (e.g., `5000` for 5 Mbps) |
| `--audio-bitrate <KBPS>` | Audio bitrate in kbps (e.g., `128`) |
| `-t, --threads <NUM>` | Number of threads (default: auto-detect) |
| `-y, --overwrite` | Overwrite output file if it exists |
| `-v, --verbose` | Show debug information |
| `-q, --quiet` | Minimal output (only prints output path) |
| `--no-progress` | Disable progress bar |

## Examples

Basic transcoding:
```bash
transcode -i input.mp4 -o output.mp4
```

Set video and audio bitrates:
```bash
transcode -i input.mp4 -o output.mp4 --video-bitrate 5000 --audio-bitrate 128
```

Specify codecs:
```bash
transcode -i input.mp4 -o output.mp4 --video-codec h264 --audio-codec aac
```

Overwrite existing file with verbose output:
```bash
transcode -i input.mp4 -o output.mp4 -y -v
```

Quiet mode for scripting:
```bash
transcode -i input.mp4 -o output.mp4 -q
```

Multi-threaded encoding:
```bash
transcode -i input.mp4 -o output.mp4 -t 8
```

## Features

- **Memory-safe**: Built entirely in Rust with no unsafe FFI bindings
- **SIMD optimized**: Automatic runtime detection for AVX2/NEON acceleration
- **Progress tracking**: Visual progress bar with encoding statistics
- **Flexible output**: Verbose, normal, or quiet modes for different use cases

## Output Statistics

After transcoding, the tool displays:
- Time elapsed
- Packets processed / frames decoded / frames encoded
- Input and output file sizes
- Compression ratio
- Encoding speed (fps)

## License

See the repository root for license information.
