---
sidebar_position: 2
title: Transcode vs GStreamer
description: Detailed comparison between Transcode and GStreamer
---

# Transcode vs GStreamer

GStreamer is a powerful pipeline-based multimedia framework. This guide compares it with Transcode to help you choose the right tool.

## At a Glance

| Aspect | Transcode | GStreamer |
|--------|-----------|-----------|
| **Language** | Rust | C (with GObject) |
| **Architecture** | Library-first | Plugin-based framework |
| **Primary Use** | Transcoding, encoding | Media playback, streaming |
| **Learning Curve** | Moderate | Steep |
| **Binary Size** | ~5-15 MB | ~100 MB+ with plugins |
| **Plugin System** | Crate features | Dynamic plugins |
| **WebAssembly** | First-class | Not supported |
| **Memory Safety** | Guaranteed | Manual (GObject ref counting) |

## Architectural Differences

### GStreamer: Pipeline Framework

GStreamer is a graph-based framework where you connect "elements" into pipelines:

```
filesrc → demux → decode → convert → encode → mux → filesink
```

```bash
# GStreamer CLI
gst-launch-1.0 filesrc location=input.mp4 ! qtdemux ! h264parse ! \
  avdec_h264 ! videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4
```

**Strengths:**
- Extremely flexible pipeline composition
- Dynamic pipeline reconfiguration
- Strong separation of concerns
- Extensive plugin ecosystem

**Weaknesses:**
- Complex for simple tasks
- Plugin compatibility issues
- Difficult to embed in other languages
- High memory overhead

### Transcode: Transcoding Library

Transcode is purpose-built for media transcoding:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4");

Transcoder::new(options)?.run()?;
```

**Strengths:**
- Simple API for common tasks
- Minimal overhead
- Easy embedding
- Type-safe configuration

**Weaknesses:**
- Less flexible than arbitrary pipelines
- Smaller plugin ecosystem
- Newer, less battle-tested

## When to Choose Transcode

### 1. File-to-File Transcoding

If your primary need is converting video files, Transcode is more direct:

**Transcode:**
```rust
transcode::transcode("input.mp4", "output.mp4")?;
```

**GStreamer:**
```c
GstElement *pipeline = gst_parse_launch(
    "filesrc location=input.mp4 ! qtdemux name=demux "
    "demux.video_0 ! queue ! decodebin ! videoconvert ! x264enc ! "
    "mp4mux name=mux ! filesink location=output.mp4 "
    "demux.audio_0 ! queue ! decodebin ! audioconvert ! audioresample ! "
    "avenc_aac ! mux.",
    NULL
);
```

### 2. Embedding in Applications

Transcode embeds cleanly without framework overhead:

**Transcode (Python):**
```python
import transcode_py

stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Encoded {stats.frames_encoded} frames")
```

**GStreamer (Python):**
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)
pipeline = Gst.parse_launch('...')
pipeline.set_state(Gst.State.PLAYING)
bus = pipeline.get_bus()
# Complex event loop handling...
```

### 3. WebAssembly / Browser

Transcode compiles to WASM:

```javascript
import { WasmTranscoder } from 'transcode-wasm';

const transcoder = new WasmTranscoder();
const result = transcoder.transcode(inputBuffer);
```

GStreamer doesn't support WebAssembly.

### 4. Memory-Constrained Environments

Transcode uses significantly less memory:

| Scenario | Transcode | GStreamer |
|----------|-----------|-----------|
| 1080p transcode | 180 MB | 600+ MB |
| Library load | 5 MB | 100+ MB |
| Container image | 20 MB | 200+ MB |

### 5. Rust/Modern Language Integration

Transcode is native Rust with idiomatic APIs:

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_quality::{Psnr, Ssim};

// Type-safe, compile-time checked
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)
    .on_frame(|frame| {
        // Process each frame
    });
```

## When to Choose GStreamer

### 1. Building Media Players

GStreamer excels at playback applications:

```c
// GStreamer playbin handles everything
GstElement *playbin = gst_element_factory_make("playbin", NULL);
g_object_set(playbin, "uri", "file:///path/to/video.mp4", NULL);
gst_element_set_state(playbin, GST_STATE_PLAYING);
```

Transcode is not designed for real-time playback.

### 2. Complex Pipeline Graphs

GStreamer handles complex scenarios like:

```
         ┌─ video → encode → ─┐
input ──┤                     ├── mux → output
         └─ audio → encode → ─┘
              │
              └── preview → display
```

```bash
gst-launch-1.0 filesrc location=input.mp4 ! qtdemux name=d \
  d.video_0 ! tee name=t \
  t. ! queue ! x264enc ! mux.video_0 \
  t. ! queue ! autovideosink \
  d.audio_0 ! avenc_aac ! mux.audio_0 \
  mp4mux name=mux ! filesink location=output.mp4
```

### 3. Live Streaming / Real-time Processing

GStreamer is built for live scenarios:

```bash
# Capture and stream
gst-launch-1.0 v4l2src ! videoconvert ! x264enc tune=zerolatency ! \
  rtph264pay ! udpsink host=192.168.1.100 port=5000
```

Transcode's live streaming support is still in development.

### 4. GTK/GNOME Integration

GStreamer integrates seamlessly with GTK applications:

```c
GtkWidget *video_widget = gtk_gl_area_new();
gst_video_overlay_set_window_handle(
    GST_VIDEO_OVERLAY(sink),
    GDK_WINDOW_XID(gtk_widget_get_window(video_widget))
);
```

### 5. Hardware-Specific Features

GStreamer has extensive hardware support:

- VA-API elements (`vaapih264enc`, `vaapidecodebin`)
- NVIDIA (`nvenc`, `nvdec`, `cudaupload`)
- Intel Quick Sync (`qsvh264enc`)
- Raspberry Pi (`omxh264enc`)
- Android (`amcviddec`)

Transcode supports hardware acceleration but with fewer specialized elements.

## Feature Comparison

### Codecs

| Codec | Transcode | GStreamer |
|-------|-----------|-----------|
| H.264 | Native | Plugin (x264, openh264, vaapi) |
| H.265 | Native | Plugin (x265, vaapi) |
| AV1 | Native (rav1e/dav1d) | Plugin (aom, svtav1) |
| VP9 | Native | Plugin (vpx) |
| AAC | Native | Plugin (faac, avenc_aac) |
| Opus | Native | Plugin (opusenc) |

### Features

| Feature | Transcode | GStreamer |
|---------|-----------|-----------|
| AI Enhancement | Built-in | Not available |
| Quality Metrics | Built-in | External |
| Distributed Processing | Built-in | External |
| Live Preview | Not yet | Native |
| Network Streaming | HLS/DASH output | Full (RTSP, RTP, WebRTC) |
| Subtitles | Supported | Extensive |
| Dynamic Pipelines | Limited | Native |

## Migration Examples

### Simple Transcode

**GStreamer:**
```bash
gst-launch-1.0 filesrc location=input.mp4 ! qtdemux ! decodebin ! \
  videoconvert ! x264enc bitrate=5000 ! mp4mux ! filesink location=output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --video-codec h264 --video-bitrate 5000
```

### With Scaling

**GStreamer:**
```bash
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! \
  videoscale ! video/x-raw,width=1920,height=1080 ! \
  x264enc ! mp4mux ! filesink location=output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --width 1920 --height 1080
```

### HLS Output

**GStreamer:**
```bash
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! x264enc ! \
  mpegtsmux ! hlssink location=segment%05d.ts playlist-location=playlist.m3u8
```

**Transcode:**
```bash
transcode -i input.mp4 --hls output/ --hls-time 6
```

## API Comparison

### Pipeline Creation

**GStreamer (C):**
```c
GstElement *pipeline, *source, *demux, *decoder, *encoder, *mux, *sink;

pipeline = gst_pipeline_new("transcode");
source = gst_element_factory_make("filesrc", "source");
demux = gst_element_factory_make("qtdemux", "demux");
decoder = gst_element_factory_make("avdec_h264", "decoder");
encoder = gst_element_factory_make("x264enc", "encoder");
mux = gst_element_factory_make("mp4mux", "mux");
sink = gst_element_factory_make("filesink", "sink");

g_object_set(source, "location", "input.mp4", NULL);
g_object_set(sink, "location", "output.mp4", NULL);
g_object_set(encoder, "bitrate", 5000, NULL);

gst_bin_add_many(GST_BIN(pipeline), source, demux, decoder, encoder, mux, sink, NULL);
gst_element_link(source, demux);
// Dynamic pad linking for demux...
gst_element_link_many(decoder, encoder, mux, sink, NULL);

gst_element_set_state(pipeline, GST_STATE_PLAYING);

// Event loop, error handling, cleanup...
```

**Transcode (Rust):**
```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

let mut transcoder = Transcoder::new(options)?;
transcoder.run()?;
```

### Error Handling

**GStreamer:**
```c
GstBus *bus = gst_element_get_bus(pipeline);
GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
    GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

if (msg != NULL) {
    GError *err;
    gchar *debug_info;

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR:
            gst_message_parse_error(msg, &err, &debug_info);
            g_printerr("Error: %s\n", err->message);
            g_error_free(err);
            g_free(debug_info);
            break;
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            break;
    }
    gst_message_unref(msg);
}
```

**Transcode:**
```rust
match transcoder.run() {
    Ok(stats) => println!("Encoded {} frames", stats.frames_encoded),
    Err(Error::Codec(e)) => eprintln!("Codec error: {}", e),
    Err(Error::Io(e)) => eprintln!("IO error: {}", e),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Can They Work Together?

Yes. You can use GStreamer for capture/playback and Transcode for encoding:

```rust
// Use GStreamer to capture from camera
// (via gstreamer-rs bindings)
let pipeline = gst::parse_launch("v4l2src ! appsink name=sink")?;
let appsink = pipeline.by_name("sink").unwrap();

// Process frames with Transcode
let mut encoder = transcode_codecs::video::h264::H264Encoder::new(config)?;

while let Some(sample) = appsink.pull_sample() {
    let frame = convert_gst_to_transcode_frame(sample);
    let packets = encoder.encode(&frame)?;
    // Write packets...
}
```

## Summary

| Use Case | Recommendation |
|----------|----------------|
| File transcoding | **Transcode** |
| Media player | **GStreamer** |
| WebAssembly | **Transcode** |
| Live streaming | **GStreamer** (for now) |
| Rust application | **Transcode** |
| GTK application | **GStreamer** |
| Low memory | **Transcode** |
| Complex pipelines | **GStreamer** |
| AI enhancement | **Transcode** |

---

Ready to try Transcode? See the [Quick Start](/docs/getting-started/quick-start) guide.
