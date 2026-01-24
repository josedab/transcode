# Transcode Playground

An interactive browser-based demo of the transcode-wasm library for client-side video transcoding.

## Features

- Drag & drop video file input
- Multiple quality presets (Web, High Quality, Small File)
- Custom resolution and bitrate controls
- Real-time progress tracking with FPS and ETA
- Video preview and download
- Dark theme UI

## Setup

### Prerequisites

- Rust toolchain
- wasm-pack (`cargo install wasm-pack`)
- A local web server (for CORS headers)

### Build Steps

1. Build the WASM module:

```bash
cd transcode-wasm
wasm-pack build --target web
```

2. Start a local server with proper CORS headers:

```bash
# Using Python
python3 -m http.server 8080

# Or using npx
npx serve -p 8080
```

3. Open the playground:

```
http://localhost:8080/playground/
```

## Project Structure

```
playground/
├── index.html    # Main page with drag-drop UI
├── styles.css    # Dark theme styling
├── app.js        # WASM integration and app logic
└── README.md     # This file
```

## Usage

1. **Drop a video file** onto the drop zone or click to select
2. **Choose a preset** or configure custom settings:
   - Resolution (1080p, 720p, 480p, 360p)
   - Video bitrate (500-50000 kbps)
   - Audio bitrate (64-320 kbps)
3. **Click "Start Transcoding"** to begin
4. **Monitor progress** with real-time FPS and ETA
5. **Preview and download** the transcoded video

## Demo Mode

If the WASM module is not built, the playground runs in demo mode:
- Simulates the transcoding process
- Uses the input file as output (no actual transcoding)
- Useful for testing the UI without building WASM

## Browser Requirements

- WebAssembly support
- Web Workers
- File API
- Streams API (optional, for large files)

Modern browsers (Chrome 80+, Firefox 75+, Safari 14+, Edge 80+) support all required features.

## Customization

### Adding New Presets

Edit the `presets` object in `app.js`:

```javascript
const presets = {
    web: { resolution: '1280x720', videoBitrate: 2500, audioBitrate: 128 },
    // Add your preset here
    mobile: { resolution: '640x360', videoBitrate: 800, audioBitrate: 64 },
};
```

### Styling

The playground uses CSS custom properties for theming. Edit `styles.css` to customize:

```css
:root {
    --bg-primary: #0d1117;
    --accent-primary: #58a6ff;
    /* ... */
}
```

## Troubleshooting

### WASM module not loading

1. Ensure wasm-pack build completed successfully
2. Check that `../pkg/transcode_wasm.js` exists
3. Verify the server sends correct MIME types for `.wasm` files

### SharedArrayBuffer not available

For multi-threaded features, the server must send these headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Large file handling

For files larger than available memory, use the streaming API:

```javascript
const streaming = new wasm.StreamingTranscoder();
// Process in chunks
```

## License

Part of the Transcode project. See the main repository for license information.
