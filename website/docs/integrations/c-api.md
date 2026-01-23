---
sidebar_position: 4
title: C API
description: Use Transcode from C/C++ applications
---

# C API

Transcode provides a C-compatible API for integration with C, C++, and other languages with C FFI support.

## Overview

The C API provides:

- **Memory-safe bindings** - Rust safety with C compatibility
- **Opaque handles** - Resource management via handles
- **Error handling** - Error codes and messages
- **Full feature access** - All transcoding capabilities

## Installation

### Pre-built Libraries

Download from releases:

```bash
# Linux
curl -LO https://github.com/example/transcode/releases/latest/download/libtranscode-linux-x64.tar.gz
tar xzf libtranscode-linux-x64.tar.gz

# macOS
curl -LO https://github.com/example/transcode/releases/latest/download/libtranscode-macos-arm64.tar.gz
tar xzf libtranscode-macos-arm64.tar.gz

# Windows
curl -LO https://github.com/example/transcode/releases/latest/download/transcode-windows-x64.zip
unzip transcode-windows-x64.zip
```

### Build from Source

```bash
git clone https://github.com/example/transcode
cd transcode
cargo build --release -p transcode-c

# Output: target/release/libtranscode.{so,dylib,dll}
# Header: transcode-c/include/transcode.h
```

## Quick Start

```c
#include <stdio.h>
#include "transcode.h"

int main() {
    // Initialize library
    transcode_init();

    // Create transcoder
    TranscodeOptions opts = {
        .input_path = "input.mp4",
        .output_path = "output.mp4",
        .video_bitrate = 5000000,
        .video_codec = TRANSCODE_CODEC_H264,
        .audio_bitrate = 128000,
        .audio_codec = TRANSCODE_CODEC_AAC,
    };

    TranscoderHandle transcoder = transcode_create(&opts);
    if (!transcoder) {
        fprintf(stderr, "Failed to create transcoder: %s\n", transcode_get_error());
        return 1;
    }

    // Run transcoding
    TranscodeResult result;
    int status = transcode_run(transcoder, &result);

    if (status == TRANSCODE_OK) {
        printf("Transcoded successfully!\n");
        printf("Duration: %.2fs\n", result.duration);
        printf("Output size: %lu bytes\n", result.output_size);
    } else {
        fprintf(stderr, "Transcode failed: %s\n", transcode_get_error());
    }

    // Cleanup
    transcode_destroy(transcoder);
    transcode_shutdown();

    return status;
}
```

## Compilation

### Linux/macOS

```bash
gcc -o transcode_example example.c -L/path/to/lib -ltranscode -lpthread -ldl -lm
```

### Windows

```cmd
cl /Fe:transcode_example.exe example.c /I"path\to\include" /link "path\to\transcode.lib"
```

### CMake

```cmake
cmake_minimum_required(VERSION 3.10)
project(transcode_example)

find_library(TRANSCODE_LIB transcode PATHS /path/to/lib)
include_directories(/path/to/include)

add_executable(transcode_example example.c)
target_link_libraries(transcode_example ${TRANSCODE_LIB} pthread dl m)
```

## API Reference

### Initialization

```c
// Initialize the library (call once at startup)
int transcode_init(void);

// Shutdown the library (call once at exit)
void transcode_shutdown(void);

// Get version string
const char* transcode_version(void);
```

### Transcoder

```c
// Create a transcoder
TranscoderHandle transcode_create(const TranscodeOptions* options);

// Destroy a transcoder
void transcode_destroy(TranscoderHandle transcoder);

// Run transcoding (blocking)
int transcode_run(TranscoderHandle transcoder, TranscodeResult* result);

// Run transcoding with progress callback
int transcode_run_with_progress(
    TranscoderHandle transcoder,
    TranscodeResult* result,
    TranscodeProgressCallback callback,
    void* user_data
);

// Cancel transcoding (from another thread)
void transcode_cancel(TranscoderHandle transcoder);
```

### Options Structure

```c
typedef struct {
    // Input/Output
    const char* input_path;
    const char* output_path;

    // Video settings
    TranscodeCodec video_codec;
    uint32_t video_bitrate;      // bits per second
    uint32_t width;              // 0 = auto
    uint32_t height;             // 0 = auto
    float fps;                   // 0 = auto

    // Audio settings
    TranscodeCodec audio_codec;
    uint32_t audio_bitrate;      // bits per second
    uint32_t sample_rate;        // 0 = auto
    uint8_t channels;            // 0 = auto

    // Advanced
    TranscodePreset preset;
    uint8_t crf;                 // 0 = use bitrate
    bool two_pass;

} TranscodeOptions;
```

### Codecs

```c
typedef enum {
    // Video
    TRANSCODE_CODEC_H264 = 1,
    TRANSCODE_CODEC_H265 = 2,
    TRANSCODE_CODEC_AV1 = 3,
    TRANSCODE_CODEC_VP9 = 4,
    TRANSCODE_CODEC_PRORES = 5,

    // Audio
    TRANSCODE_CODEC_AAC = 100,
    TRANSCODE_CODEC_OPUS = 101,
    TRANSCODE_CODEC_MP3 = 102,
    TRANSCODE_CODEC_FLAC = 103,

} TranscodeCodec;
```

### Presets

```c
typedef enum {
    TRANSCODE_PRESET_ULTRAFAST = 0,
    TRANSCODE_PRESET_SUPERFAST = 1,
    TRANSCODE_PRESET_VERYFAST = 2,
    TRANSCODE_PRESET_FASTER = 3,
    TRANSCODE_PRESET_FAST = 4,
    TRANSCODE_PRESET_MEDIUM = 5,
    TRANSCODE_PRESET_SLOW = 6,
    TRANSCODE_PRESET_SLOWER = 7,
    TRANSCODE_PRESET_VERYSLOW = 8,
} TranscodePreset;
```

### Result Structure

```c
typedef struct {
    double duration;          // Processing time in seconds
    uint64_t output_size;     // Output file size in bytes
    uint64_t frames_encoded;  // Total frames encoded
    double average_fps;       // Average encoding FPS
} TranscodeResult;
```

### Progress Callback

```c
typedef struct {
    uint64_t frames_done;
    uint64_t total_frames;
    double percent;
    double elapsed;
    double estimated_remaining;
    double fps;
} TranscodeProgress;

typedef void (*TranscodeProgressCallback)(
    const TranscodeProgress* progress,
    void* user_data
);
```

### Error Handling

```c
// Error codes
typedef enum {
    TRANSCODE_OK = 0,
    TRANSCODE_ERROR_INVALID_INPUT = 1,
    TRANSCODE_ERROR_INVALID_OUTPUT = 2,
    TRANSCODE_ERROR_CODEC_NOT_SUPPORTED = 3,
    TRANSCODE_ERROR_DECODE_FAILED = 4,
    TRANSCODE_ERROR_ENCODE_FAILED = 5,
    TRANSCODE_ERROR_IO = 6,
    TRANSCODE_ERROR_OUT_OF_MEMORY = 7,
    TRANSCODE_ERROR_CANCELLED = 8,
    TRANSCODE_ERROR_UNKNOWN = 99,
} TranscodeError;

// Get last error message (thread-local)
const char* transcode_get_error(void);

// Get last error code (thread-local)
TranscodeError transcode_get_error_code(void);
```

## Progress Reporting

```c
void progress_callback(const TranscodeProgress* progress, void* user_data) {
    printf("\rProgress: %.1f%% (%llu/%llu frames) @ %.1f fps",
        progress->percent,
        progress->frames_done,
        progress->total_frames,
        progress->fps);
    fflush(stdout);
}

int main() {
    // ... create transcoder ...

    TranscodeResult result;
    int status = transcode_run_with_progress(
        transcoder,
        &result,
        progress_callback,
        NULL  // user_data
    );

    printf("\nDone!\n");
    return status;
}
```

## Media Info

```c
// Probe media file
MediaInfoHandle transcode_probe(const char* path);

// Get media info
int transcode_get_media_info(MediaInfoHandle handle, MediaInfo* info);

// Free media info handle
void transcode_media_info_free(MediaInfoHandle handle);

// Media info structure
typedef struct {
    double duration;
    const char* format;
    uint32_t video_stream_count;
    uint32_t audio_stream_count;
} MediaInfo;

// Get video stream info
int transcode_get_video_stream(
    MediaInfoHandle handle,
    uint32_t index,
    VideoStreamInfo* info
);

typedef struct {
    TranscodeCodec codec;
    uint32_t width;
    uint32_t height;
    float fps;
    uint32_t bitrate;
} VideoStreamInfo;

// Get audio stream info
int transcode_get_audio_stream(
    MediaInfoHandle handle,
    uint32_t index,
    AudioStreamInfo* info
);

typedef struct {
    TranscodeCodec codec;
    uint32_t sample_rate;
    uint8_t channels;
    uint32_t bitrate;
} AudioStreamInfo;
```

### Usage

```c
MediaInfoHandle info = transcode_probe("video.mp4");
if (!info) {
    fprintf(stderr, "Failed to probe: %s\n", transcode_get_error());
    return 1;
}

MediaInfo media;
transcode_get_media_info(info, &media);

printf("Duration: %.2fs\n", media.duration);
printf("Format: %s\n", media.format);

for (uint32_t i = 0; i < media.video_stream_count; i++) {
    VideoStreamInfo video;
    transcode_get_video_stream(info, i, &video);
    printf("Video %d: %dx%d @ %.2f fps\n", i, video.width, video.height, video.fps);
}

for (uint32_t i = 0; i < media.audio_stream_count; i++) {
    AudioStreamInfo audio;
    transcode_get_audio_stream(info, i, &audio);
    printf("Audio %d: %d Hz, %d channels\n", i, audio.sample_rate, audio.channels);
}

transcode_media_info_free(info);
```

## Video Filters

```c
// Add scale filter
int transcode_add_scale_filter(
    TranscoderHandle transcoder,
    uint32_t width,
    uint32_t height,
    TranscodeScaleMode mode
);

typedef enum {
    TRANSCODE_SCALE_NEAREST = 0,
    TRANSCODE_SCALE_BILINEAR = 1,
    TRANSCODE_SCALE_BICUBIC = 2,
    TRANSCODE_SCALE_LANCZOS = 3,
} TranscodeScaleMode;

// Add crop filter
int transcode_add_crop_filter(
    TranscoderHandle transcoder,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height
);

// Add color adjustment filter
int transcode_add_color_filter(
    TranscoderHandle transcoder,
    float brightness,  // -1.0 to 1.0
    float contrast,    // 0.0 to 2.0
    float saturation   // 0.0 to 2.0
);
```

### Usage

```c
TranscoderHandle transcoder = transcode_create(&opts);

// Add filters (order matters)
transcode_add_scale_filter(transcoder, 1920, 1080, TRANSCODE_SCALE_LANCZOS);
transcode_add_color_filter(transcoder, 0.1, 1.1, 1.05);

// Run transcoding
transcode_run(transcoder, &result);
```

## Streaming Output

### HLS

```c
TranscodeHlsOptions hls_opts = {
    .directory = "output/hls",
    .segment_duration = 6,
    .playlist_name = "playlist.m3u8",
};

int transcode_set_hls_output(
    TranscoderHandle transcoder,
    const TranscodeHlsOptions* options
);
```

### DASH

```c
TranscodeDashOptions dash_opts = {
    .directory = "output/dash",
    .segment_duration = 4,
    .manifest_name = "manifest.mpd",
};

int transcode_set_dash_output(
    TranscoderHandle transcoder,
    const TranscodeDashOptions* options
);
```

## Thread Safety

The C API is thread-safe with the following considerations:

- `transcode_init()` and `transcode_shutdown()` are NOT thread-safe
- Each `TranscoderHandle` should only be used from one thread
- Error messages are thread-local
- Progress callbacks may be called from any thread

```c
// Safe: each thread has its own transcoder
void* thread_func(void* arg) {
    TranscodeOptions* opts = (TranscodeOptions*)arg;
    TranscoderHandle transcoder = transcode_create(opts);

    TranscodeResult result;
    transcode_run(transcoder, &result);

    transcode_destroy(transcoder);
    return NULL;
}
```

## C++ Wrapper

For C++ users, a convenience wrapper is available:

```cpp
#include "transcode.hpp"

int main() {
    using namespace transcode;

    try {
        Transcoder transcoder({
            .input = "input.mp4",
            .output = "output.mp4",
            .videoBitrate = 5000000,
            .videoCodec = Codec::H264,
        });

        transcoder.onProgress([](const Progress& p) {
            std::cout << "\rProgress: " << p.percent << "%" << std::flush;
        });

        auto result = transcoder.run();
        std::cout << "\nDone! " << result.outputSize << " bytes\n";

    } catch (const TranscodeError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### RAII Resource Management

```cpp
// TranscoderHandle is automatically freed
{
    Transcoder transcoder(options);
    transcoder.run();
}  // Automatically destroyed here

// MediaInfo is automatically freed
{
    MediaInfo info = probe("video.mp4");
    std::cout << "Duration: " << info.duration() << "s\n";
}  // Automatically freed here
```

## Memory Management

### Buffer Handling

```c
// Transcode to memory buffer
uint8_t* buffer = NULL;
size_t buffer_size = 0;

int transcode_to_buffer(
    TranscoderHandle transcoder,
    uint8_t** buffer,
    size_t* buffer_size
);

// Free buffer when done
void transcode_free_buffer(uint8_t* buffer);
```

### Custom Allocator

```c
typedef void* (*TranscodeAllocFunc)(size_t size);
typedef void (*TranscodeFreeFunc)(void* ptr);

void transcode_set_allocator(
    TranscodeAllocFunc alloc,
    TranscodeFreeFunc free
);
```

## Example: Batch Processing

```c
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include "transcode.h"

int process_file(const char* input, const char* output_dir) {
    char output[256];
    snprintf(output, sizeof(output), "%s/output_%s", output_dir, strrchr(input, '/') + 1);

    TranscodeOptions opts = {
        .input_path = input,
        .output_path = output,
        .video_bitrate = 5000000,
        .video_codec = TRANSCODE_CODEC_H264,
    };

    TranscoderHandle transcoder = transcode_create(&opts);
    if (!transcoder) {
        fprintf(stderr, "Failed to create transcoder for %s: %s\n",
            input, transcode_get_error());
        return 1;
    }

    TranscodeResult result;
    int status = transcode_run(transcoder, &result);

    if (status == TRANSCODE_OK) {
        printf("Transcoded %s -> %s (%.2fs)\n", input, output, result.duration);
    } else {
        fprintf(stderr, "Failed to transcode %s: %s\n", input, transcode_get_error());
    }

    transcode_destroy(transcoder);
    return status;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir>\n", argv[0]);
        return 1;
    }

    transcode_init();

    DIR* dir = opendir(argv[1]);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    struct dirent* entry;
    int processed = 0;
    int failed = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".mp4") || strstr(entry->d_name, ".avi")) {
            char input[256];
            snprintf(input, sizeof(input), "%s/%s", argv[1], entry->d_name);

            if (process_file(input, argv[2]) == TRANSCODE_OK) {
                processed++;
            } else {
                failed++;
            }
        }
    }

    closedir(dir);

    printf("\nProcessed: %d, Failed: %d\n", processed, failed);

    transcode_shutdown();
    return failed > 0 ? 1 : 0;
}
```

## Next Steps

- [Python Integration](/docs/integrations/python) - Python bindings
- [Node.js Integration](/docs/integrations/nodejs) - Node.js bindings
- [CLI Reference](/docs/reference/cli) - Command-line interface
