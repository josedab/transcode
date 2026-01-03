#![no_main]

//! Fuzz target for MKV/WebM EBML parsing.
//!
//! Tests EBML variable-length integers (VINT), element IDs, and element headers
//! with arbitrary input to find parsing bugs and panics.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

#[derive(Arbitrary, Debug)]
struct EbmlInput {
    data: Vec<u8>,
    operation: EbmlOperation,
}

#[derive(Arbitrary, Debug)]
enum EbmlOperation {
    /// Parse a variable-length integer (VINT)
    ReadVint,
    /// Parse an element ID
    ReadElementId,
    /// Parse an element size
    ReadElementSize,
    /// Parse a full element header (ID + size)
    ReadElementHeader,
    /// Round-trip encode/decode VINT
    VintRoundtrip { value: u64 },
    /// Round-trip element header
    ElementHeaderRoundtrip { id: u32, size: u64 },
    /// Parse EBML header structure
    ReadEbmlHeader,
}

fuzz_target!(|input: EbmlInput| {
    // Limit input size to prevent excessive memory allocation
    if input.data.len() > 64 * 1024 {
        return;
    }

    match input.operation {
        EbmlOperation::ReadVint => {
            let mut cursor = Cursor::new(&input.data);
            // Should not panic on any input
            let _ = transcode_mkv::ebml::read_vint(&mut cursor);
        }

        EbmlOperation::ReadElementId => {
            let mut cursor = Cursor::new(&input.data);
            // Should not panic on any input
            let _ = transcode_mkv::ebml::read_element_id(&mut cursor);
        }

        EbmlOperation::ReadElementSize => {
            let mut cursor = Cursor::new(&input.data);
            // Should not panic on any input
            let _ = transcode_mkv::ebml::read_element_size(&mut cursor);
        }

        EbmlOperation::ReadElementHeader => {
            let mut cursor = Cursor::new(&input.data);
            // Should not panic on any input
            let _ = transcode_mkv::ElementHeader::read(&mut cursor);
        }

        EbmlOperation::VintRoundtrip { value } => {
            // Limit to values that can be encoded (up to 8 bytes = 56 bits of data)
            let value = value & 0x00FF_FFFF_FFFF_FFFF;

            if let Ok((encoded, len)) = transcode_mkv::ebml::encode_vint(value) {
                let mut cursor = Cursor::new(&encoded[..len]);
                if let Ok((decoded, _)) = transcode_mkv::ebml::read_vint(&mut cursor) {
                    // Values should match after round-trip
                    assert_eq!(value, decoded, "VINT round-trip mismatch");
                }
            }
        }

        EbmlOperation::ElementHeaderRoundtrip { id, size } => {
            // Limit ID to valid range and size to reasonable value
            let id = (id & 0x1FFF_FFFF) | 0x80; // Ensure valid EBML ID format
            let size = size & 0x00FF_FFFF_FFFF; // Limit size to 40 bits

            let header = transcode_mkv::ElementHeader {
                id,
                size: Some(size),
                header_size: 0,
            };

            let mut buffer = Vec::new();
            if header.write(&mut buffer).is_ok() {
                let mut cursor = Cursor::new(&buffer);
                if let Ok(parsed) = transcode_mkv::ElementHeader::read(&mut cursor) {
                    assert_eq!(header.id, parsed.id, "Element ID mismatch");
                    assert_eq!(header.size, parsed.size, "Element size mismatch");
                }
            }
        }

        EbmlOperation::ReadEbmlHeader => {
            let mut cursor = Cursor::new(&input.data);
            // Try to parse as EBML header element
            if let Ok(header) = transcode_mkv::ElementHeader::read(&mut cursor) {
                // Check if this looks like an EBML header element (0x1A45DFA3)
                if header.id == 0x1A45DFA3 {
                    // Try to parse child elements
                    let _ = transcode_mkv::ebml::read_element_id(&mut cursor);
                    let _ = transcode_mkv::ebml::read_element_size(&mut cursor);
                }
            }
        }
    }
});
