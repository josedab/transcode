# transcode-capi

C API bindings for the transcode library.

## Overview

This crate provides a C-compatible API for the transcode codec library, enabling FFmpeg-like usage patterns from C/C++ applications. All functions are designed to be called from C code with proper null pointer checks and error handling.

## Features

- **FFmpeg-like API**: Familiar patterns for C/C++ developers
- **Full Context Management**: Open, read, seek, decode, encode, write
- **Memory Management**: Explicit allocation/deallocation for packets and frames
- **Error Handling**: Comprehensive error codes with string descriptions
- **Pixel Format Support**: YUV420p, YUV422p, YUV444p, NV12, RGB, RGBA
- **Color Space Support**: BT.601, BT.709, BT.2020, sRGB

## Usage (C)

```c
#include "transcode.h"

int main() {
    TranscodeContext* ctx = NULL;
    TranscodeError err = transcode_open_input("input.mp4", &ctx);
    if (err != TRANSCODE_ERROR_SUCCESS) {
        printf("Error: %s\n", transcode_error_string(err));
        return 1;
    }

    TranscodePacket* packet = transcode_packet_alloc();
    while (transcode_read_packet(ctx, packet) == TRANSCODE_ERROR_SUCCESS) {
        // Process packet...
    }

    transcode_packet_free(packet);
    transcode_close(ctx);
    return 0;
}
```

## API Reference

### Context Management

```c
TranscodeError transcode_open_input(const char* path, TranscodeContext** ctx);
TranscodeError transcode_open_output(TranscodeContext* ctx, const char* path,
                                     const TranscodeConfig* config);
void transcode_close(TranscodeContext* ctx);
```

### Packet Operations

```c
TranscodePacket* transcode_packet_alloc(void);
void transcode_packet_free(TranscodePacket* packet);
TranscodeError transcode_read_packet(TranscodeContext* ctx, TranscodePacket* packet);
TranscodeError transcode_write_packet(TranscodeContext* ctx,
                                      const TranscodePacket* packet);
```

### Frame Operations

```c
TranscodeFrame* transcode_frame_alloc(void);
void transcode_frame_free(TranscodeFrame* frame);
TranscodeError transcode_frame_alloc_buffer(TranscodeFrame* frame,
                                            uint32_t width, uint32_t height,
                                            TranscodePixelFormat format);
TranscodeError transcode_frame_copy(TranscodeFrame* dst,
                                    const TranscodeFrame* src);
```

### Codec Operations

```c
TranscodeError transcode_decode_packet(TranscodeContext* ctx,
                                       const TranscodePacket* packet,
                                       TranscodeFrame* frame);
TranscodeError transcode_encode_frame(TranscodeContext* ctx,
                                      const TranscodeFrame* frame,
                                      TranscodePacket* packet);
```

### Seek and Flush

```c
TranscodeError transcode_seek(TranscodeContext* ctx, int stream_index,
                              int64_t timestamp, int flags);
TranscodeError transcode_flush_decoder(TranscodeContext* ctx);
TranscodeError transcode_flush_encoder(TranscodeContext* ctx);
```

## Error Codes

| Code | Description |
|------|-------------|
| Success | Operation completed successfully |
| InvalidArgument | Invalid argument provided |
| NullPointer | Null pointer where non-null expected |
| EndOfStream | End of stream reached |
| IoError | I/O error occurred |
| CodecError | Codec error occurred |
| ContainerError | Container/format error |
| ResourceExhausted | Out of memory |
| Unsupported | Unsupported feature |

## Building

The crate generates a header file and static/dynamic library:

```bash
cargo build --release -p transcode-capi
```

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
