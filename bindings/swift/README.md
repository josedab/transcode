# Transcode Swift SDK

Swift bindings for the Transcode library, providing video/audio transcoding capabilities for macOS, iOS, and tvOS.

## Requirements

- Swift 5.9 or later
- macOS 12+, iOS 15+, or tvOS 15+
- The transcode C API library (`libtranscode_capi`)
- Xcode 15+ (for building)

## Installation

### Swift Package Manager

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/transcode-swift", from: "0.1.0")
]
```

Or add it through Xcode: File > Add Packages... and enter the repository URL.

### Building the C Library

First, build the transcode C API:

```bash
cd /path/to/transcode
cargo build --release -p transcode-capi
```

Set the library path for linking:

```bash
export LIBRARY_PATH=/path/to/transcode/target/release
export PKG_CONFIG_PATH=/path/to/transcode/target/release/pkgconfig
```

## Usage

### Basic Example

```swift
import Transcode

// Print version
print("Transcode version: \(version())")

// Open input file
let context = try TranscodeContext(path: "input.mp4")
defer { context.close() }

// Get stream info
for i in 0..<context.numStreams {
    let info = try context.streamInfo(at: i)
    print("Stream \(i): \(info.type)")

    if info.type == .video {
        print("  Resolution: \(info.width)x\(info.height)")
    }
}
```

### Transcoding

```swift
// Open input
let context = try TranscodeContext(path: "input.mp4")
defer { context.close() }

// Configure output
var config = TranscodeConfig()
config.width = 1280
config.height = 720
config.bitrate = 2_500_000  // 2.5 Mbps
config.quality = 23
config.preset = 5  // Medium

// Open output
try context.openOutput(path: "output.mp4", config: config)

// Process frames
let packet = Packet()
let frame = Frame()

while true {
    do {
        try context.readPacket(packet)
    } catch TranscodeError.endOfStream {
        break
    }

    do {
        try context.decodePacket(packet, into: frame)
        try context.encodeFrame(frame, into: packet)
        try context.writePacket(packet)
    } catch TranscodeError.resourceExhausted {
        continue  // Need more input
    }
}

// Flush encoder
try context.flushEncoder()
```

### Async/Await Support

```swift
// Process packets using async sequence
for try await packet in context.packets() {
    try context.decodePacket(packet, into: frame)
    // Process frame...
}
```

## API Reference

### Types

#### TranscodeContext

The main transcoding context.

```swift
public final class TranscodeContext {
    public init(path: String) throws
    public func close()
    public var numStreams: Int { get }
    public func streamInfo(at index: Int) throws -> StreamInfo
    public func openOutput(path: String, config: TranscodeConfig?) throws
    public func readPacket(_ packet: Packet) throws
    public func writePacket(_ packet: Packet) throws
    public func decodePacket(_ packet: Packet, into frame: Frame) throws
    public func encodeFrame(_ frame: Frame, into packet: Packet) throws
    public func encodeFlush(into packet: Packet) throws
    public func flushDecoder() throws
    public func flushEncoder() throws
    public func seek(to timestamp: Int64, streamIndex: Int, flags: Int) throws
    public func packets() -> AsyncPacketSequence
}
```

#### Packet

Represents an encoded packet of data.

```swift
public final class Packet {
    public init()
    public var size: Int { get }
    public var data: Data? { get }
    public var pts: Int64 { get set }
    public var dts: Int64 { get set }
    public var duration: Int64 { get set }
    public var streamIndex: Int { get set }
    public var flags: UInt32 { get set }
    public var isKeyframe: Bool { get }
}
```

#### Frame

Represents a decoded video frame.

```swift
public final class Frame {
    public init()
    public var width: Int { get }
    public var height: Int { get }
    public var format: PixelFormat { get }
    public var pts: Int64 { get set }
    public var dts: Int64 { get set }
    public var duration: Int64 { get set }
    public var flags: UInt32 { get set }
    public var isKeyframe: Bool { get }
}
```

#### StreamInfo

Information about a media stream.

```swift
public struct StreamInfo {
    public let index: Int
    public let type: StreamType
    public let codecId: UInt32
    public let width: Int          // Video only
    public let height: Int         // Video only
    public let pixelFormat: PixelFormat
    public let sampleRate: Int     // Audio only
    public let channels: Int       // Audio only
    public let bitsPerSample: Int
    public let timeBaseNum: Int32
    public let timeBaseDen: Int32
    public let duration: Int64
    public let bitrate: UInt64
}
```

#### TranscodeConfig

Configuration for encoding/transcoding.

```swift
public struct TranscodeConfig {
    public var width: Int
    public var height: Int
    public var pixelFormat: PixelFormat
    public var bitrate: UInt64
    public var maxBitrate: UInt64
    public var quality: Int32      // 0-51, lower = better
    public var gopSize: Int
    public var bFrames: Int
    public var framerateNum: Int
    public var framerateDen: Int
    public var sampleRate: Int
    public var channels: Int
    public var preset: Int         // 0-9, lower = slower but better
    public var threads: Int
}
```

### Enums

#### StreamType

```swift
public enum StreamType: UInt32 {
    case unknown
    case video
    case audio
    case subtitle
    case data
}
```

#### PixelFormat

```swift
public enum PixelFormat: Int32 {
    case unknown
    case yuv420p
    case yuv422p
    case yuv444p
    case yuv420p10le
    case yuv422p10le
    case yuv444p10le
    case nv12
    case nv21
    case rgb24
    case bgr24
    case rgba
    case bgra
    case gray8
    case gray16
}
```

#### TranscodeError

```swift
public enum TranscodeError: Error {
    case invalidArgument
    case nullPointer
    case endOfStream
    case io(String?)
    case codec(String?)
    case container(String?)
    case resourceExhausted
    case unsupported(String?)
    case cancelled
    case bufferTooSmall
    case invalidState
    case unknown(Int32)
}
```

## Building

### Running Tests

```bash
cd bindings/swift
swift test
```

### Building Examples

```bash
cd bindings/swift
swift build
swift run TranscodeExample input.mp4 output.mp4
```

### Xcode Integration

1. Build the C library:
   ```bash
   cargo build --release -p transcode-capi
   ```

2. In Xcode, set the following in Build Settings:
   - Library Search Paths: `/path/to/transcode/target/release`
   - Header Search Paths: `/path/to/transcode/transcode-capi/include`

3. Add `-ltranscode_capi` to "Other Linker Flags"

## Thread Safety

- `TranscodeContext` is **not** thread-safe. Use one context per thread or protect with a lock.
- `Packet` and `Frame` objects should not be shared between threads.
- The `version()` function is thread-safe.

## Memory Management

Swift's ARC handles memory management automatically:

- `TranscodeContext`, `Packet`, and `Frame` are classes with automatic cleanup in `deinit`
- Use `defer { context.close() }` for explicit early cleanup
- The underlying C resources are freed when Swift objects are deallocated

## Platform Notes

### iOS/tvOS

- Ensure you have the ARM64 version of the C library
- For App Store submission, static linking is recommended

### macOS

- Both Intel and Apple Silicon are supported
- Universal binaries can be created with `lipo`

## License

Part of the Transcode project. See the main repository for license information.
