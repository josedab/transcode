# Transcode Go Bindings

Go bindings for the Transcode library, providing video/audio transcoding capabilities via cgo.

## Requirements

- Go 1.21 or later
- The transcode C API library (`libtranscode_capi`)
- A C compiler (gcc/clang)

## Installation

First, build the transcode C API:

```bash
cd /path/to/transcode
cargo build --release -p transcode-capi
```

Then install the Go package:

```bash
go get github.com/anthropics/transcode-go/transcode
```

## Usage

### Basic Example

```go
package main

import (
    "fmt"
    "github.com/anthropics/transcode-go/transcode"
)

func main() {
    // Print version
    fmt.Println("Version:", transcode.Version())

    // Open input file
    ctx, err := transcode.OpenInput("input.mp4")
    if err != nil {
        panic(err)
    }
    defer ctx.Close()

    // Get stream info
    for i := 0; i < ctx.NumStreams(); i++ {
        info, _ := ctx.StreamInfo(i)
        fmt.Printf("Stream %d: %s (%s)\n", i, info.Type, info.CodecName)
    }
}
```

### Transcoding

```go
// Open input
ctx, _ := transcode.OpenInput("input.mp4")
defer ctx.Close()

// Configure output
config := transcode.Config{
    Width:        1280,
    Height:       720,
    PixelFormat:  transcode.PixelFormatYUV420P,
    Bitrate:      2500000,
    FramerateNum: 30,
    FramerateDen: 1,
}

// Open output
ctx.OpenOutput("output.mp4", &config)

// Process frames
pkt := transcode.NewPacket()
defer pkt.Free()

frm := transcode.NewFrame()
defer frm.Free()

for {
    err := ctx.ReadPacket(pkt)
    if err == transcode.ErrEndOfStream {
        break
    }

    ctx.DecodePacket(pkt, frm)
    ctx.EncodeFrame(frm, pkt)
    ctx.WritePacket(pkt)
}

ctx.FlushEncoder()
```

## API Reference

### Types

#### Context

The main transcoding context. Created with `OpenInput()`.

```go
type Context struct {
    // contains filtered or unexported fields
}

func OpenInput(path string) (*Context, error)
func (c *Context) Close()
func (c *Context) NumStreams() int
func (c *Context) StreamInfo(index int) (*StreamInfo, error)
func (c *Context) OpenOutput(path string, config *Config) error
func (c *Context) ReadPacket(pkt *Packet) error
func (c *Context) WritePacket(pkt *Packet) error
func (c *Context) DecodePacket(pkt *Packet, frame *Frame) error
func (c *Context) EncodeFrame(frame *Frame, pkt *Packet) error
func (c *Context) FlushDecoder() error
func (c *Context) FlushEncoder() error
```

#### Packet

Represents an encoded packet of data.

```go
type Packet struct {
    // contains filtered or unexported fields
}

func NewPacket() *Packet
func (p *Packet) Free()
func (p *Packet) Size() int
func (p *Packet) Data() []byte
func (p *Packet) Pts() int64
func (p *Packet) Dts() int64
func (p *Packet) Duration() int64
func (p *Packet) StreamIndex() int
func (p *Packet) IsKeyframe() bool
func (p *Packet) Flags() int
```

#### Frame

Represents a decoded video/audio frame.

```go
type Frame struct {
    // contains filtered or unexported fields
}

func NewFrame() *Frame
func (f *Frame) Free()
func (f *Frame) Width() int
func (f *Frame) Height() int
func (f *Frame) Format() PixelFormat
func (f *Frame) Pts() int64
func (f *Frame) IsKeyframe() bool
func (f *Frame) Flags() int
```

#### StreamInfo

Information about a media stream.

```go
type StreamInfo struct {
    Index       int
    Type        StreamType
    CodecName   string
    Width       int        // Video only
    Height      int        // Video only
    PixelFormat PixelFormat // Video only
    FramerateNum int       // Video only
    FramerateDen int       // Video only
    SampleRate  int        // Audio only
    Channels    int        // Audio only
    Bitrate     int64
    Duration    float64
}
```

#### Config

Output configuration.

```go
type Config struct {
    Width        int
    Height       int
    PixelFormat  PixelFormat
    Bitrate      int64
    Quality      int
    GOPSize      int
    FramerateNum int
    FramerateDen int
    Preset       int
    Threads      int
}
```

### Constants

#### Stream Types

```go
const (
    StreamTypeUnknown  StreamType = 0
    StreamTypeVideo    StreamType = 1
    StreamTypeAudio    StreamType = 2
    StreamTypeSubtitle StreamType = 3
    StreamTypeData     StreamType = 4
)
```

#### Pixel Formats

```go
const (
    PixelFormatUnknown PixelFormat = -1
    PixelFormatYUV420P PixelFormat = 0
    PixelFormatYUV422P PixelFormat = 4
    PixelFormatYUV444P PixelFormat = 5
    PixelFormatNV12    PixelFormat = 25
    PixelFormatRGB24   PixelFormat = 2
    PixelFormatRGBA    PixelFormat = 26
)
```

#### Color Spaces

```go
const (
    ColorSpaceBT601  ColorSpace = 5
    ColorSpaceBT709  ColorSpace = 1
    ColorSpaceBT2020 ColorSpace = 9
    ColorSpaceSRGB   ColorSpace = 13
)
```

### Errors

```go
var (
    ErrInvalidArgument   = errors.New("invalid argument")
    ErrNullPointer       = errors.New("null pointer")
    ErrEndOfStream       = errors.New("end of stream")
    ErrIO                = errors.New("I/O error")
    ErrCodec             = errors.New("codec error")
    ErrContainer         = errors.New("container error")
    ErrResourceExhausted = errors.New("resource exhausted")
    ErrUnsupported       = errors.New("unsupported operation")
    ErrCancelled         = errors.New("operation cancelled")
    ErrBufferTooSmall    = errors.New("buffer too small")
    ErrInvalidState      = errors.New("invalid state")
    ErrUnknown           = errors.New("unknown error")
)
```

## Building

### CGO Flags

The package expects the transcode library to be available. You may need to set CGO flags:

```bash
export CGO_CFLAGS="-I/path/to/transcode/transcode-capi/include"
export CGO_LDFLAGS="-L/path/to/transcode/target/release -ltranscode_capi"
```

Or use pkg-config if available:

```bash
export PKG_CONFIG_PATH="/path/to/transcode/target/release/pkgconfig"
```

### Running Tests

```bash
cd bindings/go
go test ./transcode/...
```

### Building Examples

```bash
cd bindings/go/examples/basic
go build
./basic input.mp4 output.mp4
```

## Thread Safety

- `Context` is **not** thread-safe. Use one context per goroutine or protect with a mutex.
- `Packet` and `Frame` objects should not be shared between goroutines.
- `Version()` and constant access are thread-safe.

## Memory Management

The bindings manage memory automatically through Go's garbage collector. However, for optimal performance:

- Call `Free()` on `Packet` and `Frame` objects when done
- Use `defer` to ensure cleanup
- `Context.Close()` should always be called

## License

Part of the Transcode project. See the main repository for license information.
