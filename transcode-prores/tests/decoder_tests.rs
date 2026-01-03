//! Comprehensive tests for the ProRes decoder

use transcode_prores::{
    ChromaFormat, ColorPrimaries, InterlaceMode, MatrixCoefficients, ProResDecoder, ProResError,
    ProResProfile, TransferCharacteristic,
};

mod frame_header_tests {
    use super::*;
    use transcode_prores::FrameHeader;

    /// Create a minimal valid ProRes frame header
    fn create_test_frame_header(profile: &[u8; 4], width: u16, height: u16) -> Vec<u8> {
        let mut data = Vec::with_capacity(512);

        // Calculate header size (everything before picture header)
        // frame_size(4) + sig(4) + header_size(2) + version(2) + fourcc(4) +
        // width(2) + height(2) + flags(2) + reserved(4) + color(3) + src_flags(1) +
        // luma_quant(64) + chroma_quant(64) = 158 bytes
        let header_size: u16 = 158;

        // Frame size (placeholder, will update at end)
        data.extend_from_slice(&[0x00, 0x00, 0x02, 0x00]); // 512 bytes placeholder

        // Signature: "icpf"
        data.extend_from_slice(b"icpf");

        // Header size
        data.extend_from_slice(&header_size.to_be_bytes());

        // Version
        data.extend_from_slice(&[0x00, 0x00]);

        // FourCC (profile)
        data.extend_from_slice(profile);

        // Width and height
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());

        // Frame flags (progressive, 4:2:2)
        data.extend_from_slice(&[0x00, 0x02]);

        // Reserved/aspect ratio
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        // Color primaries, transfer, matrix
        data.push(1); // BT.709
        data.push(1); // BT.709
        data.push(1); // BT.709

        // Source format flags (10-bit, no alpha)
        data.push(0x00);

        // Luma quantization matrix (64 bytes)
        for i in 0..64 {
            data.push(4 + (i as u8 % 16));
        }

        // Chroma quantization matrix (64 bytes)
        for i in 0..64 {
            data.push(4 + (i as u8 % 16));
        }

        // Picture header starts at offset 158
        // Picture header size (includes the 4 bytes here + slice index table)
        let mb_height = (height as u32 + 15) / 16;
        let slices_per_row = 8u16;
        let num_slices = slices_per_row as u32 * mb_height;
        let picture_header_size: u8 = 4; // Just the basic header fields

        data.push(picture_header_size);
        data.push(0x00); // Reserved

        // Slice info: high byte is slices per row, low byte is log2 slice MB width
        data.push(slices_per_row as u8); // Slices per row
        data.push(0x00); // Log2 slice MB width (0 = 1 MB per slice)

        // Slice index table - relative offsets from end of picture header
        // Each entry is 2 bytes
        let _slice_index_start = header_size as usize + picture_header_size as usize;
        for i in 0..num_slices {
            // Relative offset for each slice (100 bytes per slice as dummy)
            let offset = ((i + 1) * 100) as u16;
            data.extend_from_slice(&offset.to_be_bytes());
        }

        // Pad with dummy slice data
        while data.len() < 512 {
            data.push(0);
        }

        // Update frame size
        let frame_size = data.len() as u32;
        data[0..4].copy_from_slice(&frame_size.to_be_bytes());

        data
    }

    #[test]
    fn test_parse_valid_hq_header() {
        let data = create_test_frame_header(b"apch", 1920, 1080);
        let header = FrameHeader::parse(&data);

        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.profile, ProResProfile::HQ);
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
    }

    #[test]
    fn test_parse_all_profiles() {
        let profiles = [
            (b"apco", ProResProfile::Proxy),
            (b"apcs", ProResProfile::LT),
            (b"apcn", ProResProfile::Standard),
            (b"apch", ProResProfile::HQ),
        ];

        for (fourcc, expected_profile) in profiles {
            let data = create_test_frame_header(fourcc, 1920, 1080);
            let header = FrameHeader::parse(&data).unwrap();
            assert_eq!(header.profile, expected_profile);
        }
    }

    #[test]
    fn test_invalid_signature() {
        let mut data = create_test_frame_header(b"apch", 1920, 1080);
        // Corrupt signature
        data[4] = b'x';
        data[5] = b'y';
        data[6] = b'z';
        data[7] = b'w';

        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(ProResError::InvalidSignature(_))));
    }

    #[test]
    fn test_unknown_profile() {
        let mut data = create_test_frame_header(b"apch", 1920, 1080);
        // Set unknown FourCC
        data[12] = b'x';
        data[13] = b'x';
        data[14] = b'x';
        data[15] = b'x';

        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(ProResError::UnknownProfile(_))));
    }

    #[test]
    fn test_insufficient_data() {
        let data = [0u8; 10]; // Too short
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(ProResError::InsufficientData { .. })));
    }
}

mod profile_tests {
    use super::*;

    #[test]
    fn test_profile_fourcc() {
        assert_eq!(ProResProfile::Proxy.fourcc(), b"apco");
        assert_eq!(ProResProfile::LT.fourcc(), b"apcs");
        assert_eq!(ProResProfile::Standard.fourcc(), b"apcn");
        assert_eq!(ProResProfile::HQ.fourcc(), b"apch");
        assert_eq!(ProResProfile::P4444.fourcc(), b"ap4h");
        assert_eq!(ProResProfile::P4444XQ.fourcc(), b"ap4x");
    }

    #[test]
    fn test_profile_from_fourcc() {
        assert_eq!(ProResProfile::from_fourcc(b"apco"), Some(ProResProfile::Proxy));
        assert_eq!(ProResProfile::from_fourcc(b"apcs"), Some(ProResProfile::LT));
        assert_eq!(ProResProfile::from_fourcc(b"apcn"), Some(ProResProfile::Standard));
        assert_eq!(ProResProfile::from_fourcc(b"apch"), Some(ProResProfile::HQ));
        assert_eq!(ProResProfile::from_fourcc(b"ap4h"), Some(ProResProfile::P4444));
        assert_eq!(ProResProfile::from_fourcc(b"ap4x"), Some(ProResProfile::P4444XQ));
        assert_eq!(ProResProfile::from_fourcc(b"xxxx"), None);
    }

    #[test]
    fn test_profile_alpha_support() {
        assert!(!ProResProfile::Proxy.supports_alpha());
        assert!(!ProResProfile::LT.supports_alpha());
        assert!(!ProResProfile::Standard.supports_alpha());
        assert!(!ProResProfile::HQ.supports_alpha());
        assert!(ProResProfile::P4444.supports_alpha());
        assert!(ProResProfile::P4444XQ.supports_alpha());
    }

    #[test]
    fn test_profile_is_444() {
        assert!(!ProResProfile::Proxy.is_444());
        assert!(!ProResProfile::LT.is_444());
        assert!(!ProResProfile::Standard.is_444());
        assert!(!ProResProfile::HQ.is_444());
        assert!(ProResProfile::P4444.is_444());
        assert!(ProResProfile::P4444XQ.is_444());
    }

    #[test]
    fn test_profile_default_bit_depth() {
        assert_eq!(ProResProfile::Proxy.default_bit_depth(), 10);
        assert_eq!(ProResProfile::LT.default_bit_depth(), 10);
        assert_eq!(ProResProfile::Standard.default_bit_depth(), 10);
        assert_eq!(ProResProfile::HQ.default_bit_depth(), 10);
        assert_eq!(ProResProfile::P4444.default_bit_depth(), 10);
        assert_eq!(ProResProfile::P4444XQ.default_bit_depth(), 12);
    }
}

mod chroma_format_tests {
    use super::*;

    #[test]
    fn test_chroma_h_shift() {
        assert_eq!(ChromaFormat::YUV422.chroma_h_shift(), 1);
        assert_eq!(ChromaFormat::YUV444.chroma_h_shift(), 0);
    }

    #[test]
    fn test_chroma_v_shift() {
        assert_eq!(ChromaFormat::YUV422.chroma_v_shift(), 0);
        assert_eq!(ChromaFormat::YUV444.chroma_v_shift(), 0);
    }
}

mod interlace_mode_tests {
    use super::*;

    #[test]
    fn test_interlace_from_flags() {
        assert_eq!(InterlaceMode::from_flags(false, false), InterlaceMode::Progressive);
        assert_eq!(InterlaceMode::from_flags(false, true), InterlaceMode::Progressive);
        assert_eq!(InterlaceMode::from_flags(true, true), InterlaceMode::InterlacedTFF);
        assert_eq!(InterlaceMode::from_flags(true, false), InterlaceMode::InterlacedBFF);
    }
}

mod color_tests {
    use super::*;

    #[test]
    fn test_color_primaries_from_code() {
        assert_eq!(ColorPrimaries::from_code(0), ColorPrimaries::Unknown);
        assert_eq!(ColorPrimaries::from_code(1), ColorPrimaries::BT709);
        assert_eq!(ColorPrimaries::from_code(5), ColorPrimaries::BT601NTSC);
        assert_eq!(ColorPrimaries::from_code(9), ColorPrimaries::BT2020);
        assert_eq!(ColorPrimaries::from_code(11), ColorPrimaries::DCIP3);
    }

    #[test]
    fn test_transfer_characteristic_from_code() {
        assert_eq!(TransferCharacteristic::from_code(0), TransferCharacteristic::Unknown);
        assert_eq!(TransferCharacteristic::from_code(1), TransferCharacteristic::BT709);
        assert_eq!(TransferCharacteristic::from_code(16), TransferCharacteristic::PQ);
        assert_eq!(TransferCharacteristic::from_code(18), TransferCharacteristic::HLG);
    }

    #[test]
    fn test_matrix_coefficients_from_code() {
        assert_eq!(MatrixCoefficients::from_code(0), MatrixCoefficients::Unknown);
        assert_eq!(MatrixCoefficients::from_code(1), MatrixCoefficients::BT709);
        assert_eq!(MatrixCoefficients::from_code(5), MatrixCoefficients::BT601);
        assert_eq!(MatrixCoefficients::from_code(9), MatrixCoefficients::BT2020NCL);
    }
}

mod decoder_tests {
    use super::*;
    use transcode_prores::{get_dimensions, get_profile, probe_prores};

    #[test]
    fn test_decoder_new() {
        let decoder = ProResDecoder::new();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = ProResDecoder::new();
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_probe_prores_valid() {
        let data = [
            0x00, 0x00, 0x10, 0x00, // frame size
            b'i', b'c', b'p', b'f', // signature
        ];
        assert!(probe_prores(&data));
    }

    #[test]
    fn test_probe_prores_invalid() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(!probe_prores(&data));
    }

    #[test]
    fn test_probe_prores_short() {
        let data = [b'i', b'c', b'p'];
        assert!(!probe_prores(&data));
    }

    #[test]
    fn test_get_profile_hq() {
        let mut data = vec![0u8; 20];
        data[4..8].copy_from_slice(b"icpf");
        data[12..16].copy_from_slice(b"apch");
        assert_eq!(get_profile(&data), Some(ProResProfile::HQ));
    }

    #[test]
    fn test_get_profile_4444() {
        let mut data = vec![0u8; 20];
        data[4..8].copy_from_slice(b"icpf");
        data[12..16].copy_from_slice(b"ap4h");
        assert_eq!(get_profile(&data), Some(ProResProfile::P4444));
    }

    #[test]
    fn test_get_dimensions() {
        let mut data = vec![0u8; 20];
        data[16] = 0x07;
        data[17] = 0x80; // 1920
        data[18] = 0x04;
        data[19] = 0x38; // 1080
        assert_eq!(get_dimensions(&data), Some((1920, 1080)));
    }

    #[test]
    fn test_get_dimensions_short_data() {
        let data = vec![0u8; 10];
        assert_eq!(get_dimensions(&data), None);
    }
}

mod bitstream_tests {
    #[test]
    fn test_bitstream_read() {
        use transcode_prores::huffman::BitstreamReader;

        let data = [0b10110100, 0b01001011];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01001011);
    }

    #[test]
    fn test_bitstream_peek() {
        use transcode_prores::huffman::BitstreamReader;

        let data = [0b11110000];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.peek_bits(4).unwrap(), 0b1111);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b1111); // Still same
        assert_eq!(reader.read_bits(4).unwrap(), 0b1111);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b0000);
    }

    #[test]
    fn test_bitstream_unary() {
        use transcode_prores::huffman::BitstreamReader;

        let data = [0b11110000]; // 4 ones then zero
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_unary().unwrap(), 4);
    }

    #[test]
    fn test_bitstream_align() {
        use transcode_prores::huffman::BitstreamReader;

        let data = [0xFF, 0x00];
        let mut reader = BitstreamReader::new(&data);

        reader.read_bits(3).unwrap();
        reader.align_to_byte();

        assert_eq!(reader.byte_position(), 1);
        assert_eq!(reader.bit_position(), 0);
    }
}

mod huffman_table_tests {
    use transcode_prores::huffman::{AcHuffmanTable, DcHuffmanTable};

    #[test]
    fn test_dc_table_creation() {
        let table = DcHuffmanTable::new();
        assert_eq!(table.lengths.len(), 12);
        assert!(table.fast_lookup.len() >= 512);
    }

    #[test]
    fn test_ac_table_creation() {
        let table = AcHuffmanTable::new();
        assert_eq!(table.fast_lookup.len(), 1024);
    }
}

mod slice_tests {
    use transcode_prores::SliceHeader;

    #[test]
    fn test_slice_header_parse_422() {
        let data = [0x02, 0x44]; // header_size=2, qscale_y=4, qscale_c=4
        let header = SliceHeader::parse(&data, false, false).unwrap();

        assert_eq!(header.header_size, 2);
        assert_eq!(header.qscale_y, 4);
        assert_eq!(header.qscale_cb, 4);
    }

    #[test]
    fn test_slice_header_parse_444() {
        let data = [
            0x0A, 0x44, // header_size=10, qscale
            0x01, 0x00, // y_size = 256
            0x00, 0x80, // cb_size = 128
            0x00, 0x80, // cr_size = 128
            0x00, 0x40, // alpha_size = 64
        ];
        let header = SliceHeader::parse(&data, true, true).unwrap();

        assert_eq!(header.header_size, 10);
        assert_eq!(header.y_data_size, 256);
        assert_eq!(header.cb_data_size, 128);
    }
}

mod dct_tests {
    use transcode_prores::{dezigzag, zigzag, ZIGZAG_SCAN};

    #[test]
    fn test_zigzag_scan() {
        // First elements should follow zigzag pattern
        assert_eq!(ZIGZAG_SCAN[0], 0);
        assert_eq!(ZIGZAG_SCAN[1], 1);
        assert_eq!(ZIGZAG_SCAN[2], 8);
        assert_eq!(ZIGZAG_SCAN[3], 16);
        assert_eq!(ZIGZAG_SCAN[4], 9);
        assert_eq!(ZIGZAG_SCAN[5], 2);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let original: [i16; 64] = std::array::from_fn(|i| i as i16);
        let zz = zigzag(&original);
        let back = dezigzag(&zz);
        assert_eq!(original, back);
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_decoder_workflow() {
        // Test the basic decoder workflow without actual frame data
        let mut decoder = ProResDecoder::new();
        assert_eq!(decoder.frame_count(), 0);

        // Test configuration
        assert!(!decoder.config().multithreaded);
        assert!(!decoder.config().skip_alpha);

        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_profile_compatibility_matrix() {
        // Verify profile characteristics match Apple specs
        let test_cases = [
            (ProResProfile::Proxy, false, false, 10),
            (ProResProfile::LT, false, false, 10),
            (ProResProfile::Standard, false, false, 10),
            (ProResProfile::HQ, false, false, 10),
            (ProResProfile::P4444, true, true, 10),
            (ProResProfile::P4444XQ, true, true, 12),
        ];

        for (profile, expected_444, expected_alpha, expected_depth) in test_cases {
            assert_eq!(
                profile.is_444(),
                expected_444,
                "Profile {:?} 444 mismatch",
                profile
            );
            assert_eq!(
                profile.supports_alpha(),
                expected_alpha,
                "Profile {:?} alpha mismatch",
                profile
            );
            assert_eq!(
                profile.default_bit_depth(),
                expected_depth,
                "Profile {:?} bit depth mismatch",
                profile
            );
        }
    }
}
