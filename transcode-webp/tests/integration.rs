//! Integration tests for transcode-webp

use std::io::Cursor;
use transcode_webp::WebPDecoder;
use transcode_webp::riff::{ChunkType, parse_riff, Vp8xFlags};

/// Helper to create a minimal valid RIFF/WEBP container
fn create_minimal_webp(chunks: &[(ChunkType, &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();

    // Calculate total size
    let chunks_size: usize = chunks.iter()
        .map(|(_, d)| 8 + d.len() + (d.len() % 2))
        .sum();

    // RIFF header
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&((4 + chunks_size) as u32).to_le_bytes());
    data.extend_from_slice(b"WEBP");

    // Add chunks
    for (chunk_type, chunk_data) in chunks {
        data.extend_from_slice(&chunk_type.to_fourcc());
        data.extend_from_slice(&(chunk_data.len() as u32).to_le_bytes());
        data.extend_from_slice(chunk_data);
        if chunk_data.len() % 2 != 0 {
            data.push(0); // Padding
        }
    }

    data
}

/// Create a VP8L header for given dimensions
fn create_vp8l_header(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0x2f]; // VP8L signature

    // Pack width-1 and height-1 into 28 bits
    let w = width - 1;
    let h = height - 1;
    let bits = w | (h << 14) | (1 << 28); // alpha_used = 1

    data.push((bits & 0xFF) as u8);
    data.push(((bits >> 8) & 0xFF) as u8);
    data.push(((bits >> 16) & 0xFF) as u8);
    data.push(((bits >> 24) & 0xFF) as u8);

    // Add minimal bitstream (no transforms, simple huffman)
    data.push(0x00); // No transforms

    data
}

/// Create a VP8 header for given dimensions
fn create_vp8_header(width: u32, height: u32) -> Vec<u8> {
    let mut data = Vec::new();

    // Frame tag: keyframe, version 0, show_frame=1, first_part_size=0
    let frame_tag = 0u32 | (1 << 4);
    data.push((frame_tag & 0xFF) as u8);
    data.push(((frame_tag >> 8) & 0xFF) as u8);
    data.push(((frame_tag >> 16) & 0xFF) as u8);

    // Keyframe signature
    data.push(0x9d);
    data.push(0x01);
    data.push(0x2a);

    // Width and height
    data.push((width & 0xFF) as u8);
    data.push(((width >> 8) & 0x3F) as u8);
    data.push((height & 0xFF) as u8);
    data.push(((height >> 8) & 0x3F) as u8);

    data
}

#[test]
fn test_riff_parsing_valid() {
    let vp8l_data = create_vp8l_header(100, 100);
    let webp = create_minimal_webp(&[(ChunkType::VP8L, &vp8l_data)]);

    let mut cursor = Cursor::new(&webp);
    let container = parse_riff(&mut cursor);
    assert!(container.is_ok());

    let container = container.unwrap();
    assert_eq!(container.chunks.len(), 1);
    assert_eq!(container.chunks[0].chunk_type, ChunkType::VP8L);
}

#[test]
fn test_riff_parsing_invalid_signature() {
    let data = b"NOTARIFF";
    let mut cursor = Cursor::new(&data[..]);
    let result = parse_riff(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_riff_parsing_invalid_webp() {
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(b"NOTW");

    let mut cursor = Cursor::new(&data);
    let result = parse_riff(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_vp8x_flags_parsing() {
    // All flags set
    let flags = Vp8xFlags::from_byte(0x3E);
    assert!(flags.icc);
    assert!(flags.alpha);
    assert!(flags.exif);
    assert!(flags.xmp);
    assert!(flags.animation);

    // No flags
    let flags = Vp8xFlags::from_byte(0x00);
    assert!(!flags.icc);
    assert!(!flags.alpha);
    assert!(!flags.exif);
    assert!(!flags.xmp);
    assert!(!flags.animation);

    // Just alpha
    let flags = Vp8xFlags::from_byte(0x10);
    assert!(flags.alpha);
    assert!(!flags.animation);
}

#[test]
fn test_decoder_dimensions() {
    let vp8l_data = create_vp8l_header(200, 150);
    let webp = create_minimal_webp(&[(ChunkType::VP8L, &vp8l_data)]);

    let cursor = Cursor::new(&webp);
    let mut decoder = WebPDecoder::new(cursor).unwrap();

    let dims = decoder.dimensions();
    assert!(dims.is_ok());
    assert_eq!(dims.unwrap(), (200, 150));
}

#[test]
fn test_decoder_vp8_dimensions() {
    let vp8_data = create_vp8_header(320, 240);
    let webp = create_minimal_webp(&[(ChunkType::VP8, &vp8_data)]);

    let cursor = Cursor::new(&webp);
    let mut decoder = WebPDecoder::new(cursor).unwrap();

    let dims = decoder.dimensions();
    assert!(dims.is_ok());
    assert_eq!(dims.unwrap(), (320, 240));
}

#[test]
fn test_is_animated_false() {
    let vp8l_data = create_vp8l_header(100, 100);
    let webp = create_minimal_webp(&[(ChunkType::VP8L, &vp8l_data)]);

    let cursor = Cursor::new(&webp);
    let mut decoder = WebPDecoder::new(cursor).unwrap();

    assert!(!decoder.is_animated().unwrap());
}

#[test]
fn test_vp8x_with_animation_flag() {
    // Create VP8X chunk with animation flag
    let mut vp8x_data = vec![0u8; 10];
    vp8x_data[0] = 0x02; // Animation flag

    // Width and height (99 + 1 = 100)
    vp8x_data[4] = 99;
    vp8x_data[5] = 0;
    vp8x_data[6] = 0;
    vp8x_data[7] = 99;
    vp8x_data[8] = 0;
    vp8x_data[9] = 0;

    let webp = create_minimal_webp(&[(ChunkType::VP8X, &vp8x_data)]);

    let cursor = Cursor::new(&webp);
    let mut decoder = WebPDecoder::new(cursor).unwrap();

    assert!(decoder.is_animated().unwrap());
}

#[test]
fn test_chunk_type_roundtrip() {
    let chunk_types = [
        ChunkType::VP8,
        ChunkType::VP8L,
        ChunkType::VP8X,
        ChunkType::ALPH,
        ChunkType::ANIM,
        ChunkType::ANMF,
        ChunkType::ICCP,
        ChunkType::EXIF,
        ChunkType::XMP,
    ];

    for ct in chunk_types {
        let fourcc = ct.to_fourcc();
        let parsed = ChunkType::from_fourcc(fourcc);
        assert_eq!(ct, parsed);
    }
}

#[test]
fn test_unknown_chunk_type() {
    let fourcc = *b"UNKN";
    let ct = ChunkType::from_fourcc(fourcc);
    assert_eq!(ct, ChunkType::Unknown(*b"UNKN"));
}

#[test]
fn test_multiple_chunks() {
    let vp8x_data = vec![0u8; 10];
    let exif_data = b"II\x2a\x00\x08\x00\x00\x00";
    let vp8l_data = create_vp8l_header(100, 100);

    let webp = create_minimal_webp(&[
        (ChunkType::VP8X, &vp8x_data),
        (ChunkType::EXIF, exif_data),
        (ChunkType::VP8L, &vp8l_data),
    ]);

    let mut cursor = Cursor::new(&webp);
    let container = parse_riff(&mut cursor).unwrap();

    assert_eq!(container.chunks.len(), 3);
    assert!(container.find_chunk(ChunkType::VP8X).is_some());
    assert!(container.find_chunk(ChunkType::EXIF).is_some());
    assert!(container.find_chunk(ChunkType::VP8L).is_some());
    assert!(container.find_chunk(ChunkType::ANIM).is_none());
}

#[test]
fn test_empty_data() {
    let data: &[u8] = &[];
    let cursor = Cursor::new(data);
    let decoder = WebPDecoder::new(cursor);
    assert!(decoder.is_ok()); // Creation succeeds

    let mut decoder = decoder.unwrap();
    let result = decoder.parse();
    assert!(result.is_err()); // Parsing fails
}

#[test]
fn test_alpha_chunk_presence() {
    let mut vp8x_data = vec![0u8; 10];
    vp8x_data[0] = 0x10; // Alpha flag
    vp8x_data[4] = 99;
    vp8x_data[7] = 99;

    let alpha_data = vec![0x00, 0xFF, 0xFF, 0xFF, 0xFF]; // No compression
    let vp8_data = create_vp8_header(100, 100);

    let webp = create_minimal_webp(&[
        (ChunkType::VP8X, &vp8x_data),
        (ChunkType::ALPH, &alpha_data),
        (ChunkType::VP8, &vp8_data),
    ]);

    let mut cursor = Cursor::new(&webp);
    let container = parse_riff(&mut cursor).unwrap();

    assert!(container.has_alpha());
    assert!(container.find_chunk(ChunkType::ALPH).is_some());
}

#[test]
fn test_metadata_flags() {
    let mut vp8x_data = vec![0u8; 10];
    vp8x_data[0] = 0x2C; // EXIF + XMP + ICC flags
    vp8x_data[4] = 99;
    vp8x_data[7] = 99;

    let webp = create_minimal_webp(&[(ChunkType::VP8X, &vp8x_data)]);

    let mut cursor = Cursor::new(&webp);
    let container = parse_riff(&mut cursor).unwrap();

    assert!(container.has_exif());
    assert!(container.has_xmp());
    assert!(container.has_icc());
}

mod vp8_tests {
    use transcode_webp::Vp8Decoder;

    #[test]
    fn test_vp8_valid_header() {
        let mut data = vec![0u8; 10];
        // Keyframe
        data[0] = 0x00;
        data[1] = 0x00;
        data[2] = 0x00;
        // Signature
        data[3] = 0x9d;
        data[4] = 0x01;
        data[5] = 0x2a;
        // Dimensions: 100x100
        data[6] = 100;
        data[7] = 0;
        data[8] = 100;
        data[9] = 0;

        let decoder = Vp8Decoder::new(&data);
        assert!(decoder.is_ok());
        assert_eq!(decoder.unwrap().dimensions(), (100, 100));
    }

    #[test]
    fn test_vp8_invalid_signature() {
        let mut data = vec![0u8; 10];
        data[3] = 0x00; // Wrong signature
        data[4] = 0x00;
        data[5] = 0x00;

        let decoder = Vp8Decoder::new(&data);
        assert!(decoder.is_err());
    }
}

mod vp8l_tests {
    use transcode_webp::Vp8lDecoder;

    #[test]
    fn test_vp8l_valid_header() {
        let mut data = vec![0x2f]; // Signature

        // 100x100, alpha
        let bits: u32 = 99 | (99 << 14) | (1 << 28);
        data.push((bits & 0xFF) as u8);
        data.push(((bits >> 8) & 0xFF) as u8);
        data.push(((bits >> 16) & 0xFF) as u8);
        data.push(((bits >> 24) & 0xFF) as u8);

        let decoder = Vp8lDecoder::new(&data);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert_eq!(decoder.dimensions(), (100, 100));
        assert!(decoder.has_alpha());
    }

    #[test]
    fn test_vp8l_invalid_signature() {
        let data = vec![0x00, 0x00, 0x00, 0x00, 0x00];
        let decoder = Vp8lDecoder::new(&data);
        assert!(decoder.is_err());
    }
}

mod alpha_tests {
    use transcode_webp::AlphaDecoder;
    use transcode_webp::alpha::{AlphaFilter, AlphaCompression};

    #[test]
    fn test_alpha_no_compression() {
        let data = vec![0x00, 0x80, 0x80, 0x80, 0x80];
        let decoder = AlphaDecoder::new(&data).unwrap();

        assert_eq!(decoder.filter(), AlphaFilter::None);
        assert_eq!(decoder.compression(), AlphaCompression::None);

        let alpha = decoder.decode_alpha(2, 2).unwrap();
        assert_eq!(alpha, vec![0x80, 0x80, 0x80, 0x80]);
    }

    #[test]
    fn test_alpha_horizontal_filter() {
        // Header: horizontal filter (0x10), no compression
        let data = vec![0x10, 10, 10, 10, 10];
        let decoder = AlphaDecoder::new(&data).unwrap();

        assert_eq!(decoder.filter(), AlphaFilter::Horizontal);

        let alpha = decoder.decode_alpha(2, 2).unwrap();
        // After horizontal filter: each row adds left value
        assert_eq!(alpha[0], 10);
        assert_eq!(alpha[1], 20);
        assert_eq!(alpha[2], 10);
        assert_eq!(alpha[3], 20);
    }
}

mod animation_tests {
    use transcode_webp::animation::{BlendingMode, DisposalMethod, AnmfChunk};

    #[test]
    fn test_blending_modes() {
        // Test default values
        assert_eq!(BlendingMode::default(), BlendingMode::AlphaBlending);
        assert_eq!(DisposalMethod::default(), DisposalMethod::None);
    }

    #[test]
    fn test_blending_mode_variants() {
        // Verify variant existence
        let _blend = BlendingMode::AlphaBlending;
        let _no_blend = BlendingMode::NoBlending;
    }

    #[test]
    fn test_disposal_method_variants() {
        // Verify variant existence
        let _none = DisposalMethod::None;
        let _bg = DisposalMethod::Background;
    }

    #[test]
    fn test_anmf_parsing() {
        let mut data = vec![0u8; 16];
        // Width = 50
        data[6] = 49;
        // Height = 50
        data[9] = 49;
        // Duration = 100ms
        data[12] = 100;

        let anmf = AnmfChunk::parse(&data).unwrap();
        assert_eq!(anmf.width, 50);
        assert_eq!(anmf.height, 50);
        assert_eq!(anmf.duration_ms, 100);
    }
}

mod metadata_tests {
    use transcode_webp::metadata::{parse_exif, parse_xmp};

    #[test]
    fn test_exif_little_endian() {
        let mut data = vec![0u8; 14];
        // Little endian marker
        data[0] = 0x49;
        data[1] = 0x49;
        // TIFF magic
        data[2] = 0x2A;
        data[3] = 0x00;
        // IFD offset
        data[4] = 0x08;
        data[5] = 0x00;
        data[6] = 0x00;
        data[7] = 0x00;
        // Number of entries = 0
        data[8] = 0x00;
        data[9] = 0x00;

        let exif = parse_exif(&data);
        assert!(exif.is_ok());
    }

    #[test]
    fn test_exif_big_endian() {
        let mut data = vec![0u8; 14];
        // Big endian marker
        data[0] = 0x4D;
        data[1] = 0x4D;
        // TIFF magic
        data[2] = 0x00;
        data[3] = 0x2A;
        // IFD offset
        data[4] = 0x00;
        data[5] = 0x00;
        data[6] = 0x00;
        data[7] = 0x08;
        // Number of entries = 0
        data[8] = 0x00;
        data[9] = 0x00;

        let exif = parse_exif(&data);
        assert!(exif.is_ok());
    }

    #[test]
    fn test_xmp_parsing() {
        let xmp_str = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description xmp:CreateDate="2024-01-15T10:30:00"/>
                </rdf:RDF>
            </x:xmpmeta>
        <?xpacket end="w"?>"#;

        let xmp = parse_xmp(xmp_str.as_bytes()).unwrap();
        assert!(xmp.raw.contains("xmpmeta"));
        assert_eq!(xmp.properties.get("create_date"), Some(&"2024-01-15T10:30:00".to_string()));
    }
}
