# transcode-tiff

TIFF image codec implementation for the transcode library.

## Overview

Tagged Image File Format (TIFF) is a flexible container format for storing raster graphics images. It's commonly used in archival, professional photography, and publishing workflows due to its flexibility and lossless compression options.

## Features

- **Multiple Compression**: None, LZW, PackBits, Deflate
- **Color Spaces**: RGB, RGBA, Grayscale, CMYK
- **Bit Depths**: 8, 16, and 32 bits per sample
- **Endianness**: Big and little endian support
- **Multi-Page**: Multiple images in a single file
- **BigTIFF**: Support for files larger than 4GB

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-tiff = { path = "../transcode-tiff" }
```

### Decoding

```rust
use transcode_tiff::{TiffDecoder, TiffImage};

let decoder = TiffDecoder::new();
let image = decoder.decode(&tiff_data)?;

println!("Dimensions: {}x{}", image.width, image.height);
println!("Color space: {:?}", image.color_space);
println!("Bits per sample: {}", image.bits_per_sample);
```

### Encoding

```rust
use transcode_tiff::{TiffEncoder, Compression, ColorSpace};

let mut encoder = TiffEncoder::new();
encoder.set_compression(Compression::Lzw);
encoder.set_color_space(ColorSpace::Rgb);

let output = encoder.encode(&image)?;
```

### Multi-Page TIFF

```rust
use transcode_tiff::TiffDecoder;

let decoder = TiffDecoder::new();
let pages = decoder.decode_all_pages(&tiff_data)?;

for (i, page) in pages.iter().enumerate() {
    println!("Page {}: {}x{}", i, page.width, page.height);
}
```

### IFD Access

```rust
use transcode_tiff::{TiffDecoder, Ifd};

let decoder = TiffDecoder::new();
let ifd = decoder.read_ifd(&tiff_data)?;

// Access tags
if let Some(entry) = ifd.get_tag(tags::IMAGE_WIDTH) {
    println!("Width: {:?}", entry.value);
}
```

## Compression Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| None | Uncompressed | Maximum compatibility |
| LZW | Lempel-Ziv-Welch | Good general compression |
| PackBits | RLE variant | Fast, simple |
| Deflate | ZIP/zlib | Best lossless compression |

## Color Spaces

| Space | Description |
|-------|-------------|
| Bilevel | 1-bit black and white |
| Grayscale | 8/16-bit grayscale |
| RGB | Standard RGB color |
| RGBA | RGB with alpha |
| CMYK | Print-oriented color |
| YCbCr | JPEG-style color |

## Common Tags

| Tag | Description |
|-----|-------------|
| 256 | ImageWidth |
| 257 | ImageLength |
| 258 | BitsPerSample |
| 259 | Compression |
| 262 | PhotometricInterpretation |
| 273 | StripOffsets |
| 277 | SamplesPerPixel |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
