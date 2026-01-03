#![no_main]

//! Fuzz target for BitReader edge cases.
//!
//! Tests BitReader with arbitrary sequences of operations to find
//! edge cases in bit manipulation, exp-golomb decoding, and bounds checking.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use transcode_core::bitstream::{BitReader, BitWriter};

#[derive(Arbitrary, Debug)]
struct BitstreamInput {
    data: Vec<u8>,
    operations: Vec<BitOperation>,
    writer_operations: Vec<WriteOperation>,
    test_mode: TestMode,
}

#[derive(Arbitrary, Debug, Clone)]
enum TestMode {
    /// Test BitReader only
    Reader,
    /// Test BitWriter only
    Writer,
    /// Test roundtrip (write then read)
    Roundtrip,
}

#[derive(Arbitrary, Debug, Clone)]
enum BitOperation {
    /// Read a single bit
    ReadBit,
    /// Read n bits (1-32)
    ReadBits(u8),
    /// Read n bits as u64 (1-64)
    ReadBitsU64(u8),
    /// Read unsigned exp-golomb
    ReadUe,
    /// Read signed exp-golomb
    ReadSe,
    /// Read u8
    ReadU8,
    /// Read u16
    ReadU16,
    /// Read u32
    ReadU32,
    /// Skip n bits
    Skip(u8),
    /// Peek at single bit
    PeekBit,
    /// Peek at n bits
    PeekBits(u8),
    /// Align to byte boundary
    AlignToByte,
    /// Check remaining bits
    CheckRemaining,
    /// Check if byte aligned
    CheckByteAligned,
    /// Check if at EOF
    CheckEof,
    /// Get position
    GetPosition,
    /// Get total bits
    GetTotalBits,
    /// Check for more RBSP data
    MoreRbspData,
    /// Get remaining bytes slice
    GetRemainingBytes,
}

#[derive(Arbitrary, Debug)]
enum WriteOperation {
    WriteBit(bool),
    WriteBits { value: u32, n: u8 },
    WriteBitsU64 { value: u64, n: u8 },
    WriteUe(u32),
    WriteSe(i32),
    AlignToByte,
    WriteRbspTrailingBits,
}

fuzz_target!(|input: BitstreamInput| {
    if input.data.is_empty() && matches!(input.test_mode, TestMode::Reader) {
        return;
    }

    // Limit operations to prevent DoS
    if input.operations.len() > 10000 || input.writer_operations.len() > 10000 {
        return;
    }

    match input.test_mode {
        TestMode::Reader => {
            let mut reader = BitReader::new(&input.data);

            for op in input.operations.iter().take(1000) {
                // Stop if we've exhausted the data
                if reader.is_eof() {
                    break;
                }

                match op {
                    BitOperation::ReadBit => {
                        let _ = reader.read_bit();
                    }
                    BitOperation::ReadBits(n) => {
                        // Ensure n is in valid range (1-32)
                        let bits = (*n % 32).max(1);
                        let _ = reader.read_bits(bits);
                    }
                    BitOperation::ReadBitsU64(n) => {
                        // Ensure n is in valid range (1-64)
                        let bits = (*n % 64).max(1);
                        let _ = reader.read_bits_u64(bits);
                    }
                    BitOperation::ReadUe => {
                        let _ = reader.read_ue();
                    }
                    BitOperation::ReadSe => {
                        let _ = reader.read_se();
                    }
                    BitOperation::ReadU8 => {
                        let _ = reader.read_u8();
                    }
                    BitOperation::ReadU16 => {
                        let _ = reader.read_u16();
                    }
                    BitOperation::ReadU32 => {
                        let _ = reader.read_u32();
                    }
                    BitOperation::Skip(n) => {
                        let bits = (*n as usize) % 256;
                        let _ = reader.skip(bits);
                    }
                    BitOperation::PeekBit => {
                        let _ = reader.peek_bit();
                    }
                    BitOperation::PeekBits(n) => {
                        let bits = (*n % 32).max(1);
                        let _ = reader.peek_bits(bits);
                    }
                    BitOperation::AlignToByte => {
                        reader.align_to_byte();
                    }
                    BitOperation::CheckRemaining => {
                        let _ = reader.remaining_bits();
                    }
                    BitOperation::CheckByteAligned => {
                        let _ = reader.is_byte_aligned();
                    }
                    BitOperation::CheckEof => {
                        let _ = reader.is_eof();
                    }
                    BitOperation::GetPosition => {
                        let _ = reader.position();
                    }
                    BitOperation::GetTotalBits => {
                        let _ = reader.total_bits();
                    }
                    BitOperation::MoreRbspData => {
                        let _ = reader.more_rbsp_data();
                    }
                    BitOperation::GetRemainingBytes => {
                        let _ = reader.remaining_bytes();
                    }
                }
            }
        }
        TestMode::Writer => {
            let mut writer = BitWriter::new();

            for op in input.writer_operations.iter().take(1000) {
                match op {
                    WriteOperation::WriteBit(bit) => {
                        let _ = writer.write_bit(*bit);
                    }
                    WriteOperation::WriteBits { value, n } => {
                        let bits = (*n % 32).max(1);
                        let _ = writer.write_bits(*value, bits);
                    }
                    WriteOperation::WriteBitsU64 { value, n } => {
                        let bits = (*n % 64).max(1);
                        let _ = writer.write_bits_u64(*value, bits);
                    }
                    WriteOperation::WriteUe(value) => {
                        // Limit value to prevent excessive writes
                        if *value < 1_000_000 {
                            let _ = writer.write_ue(*value);
                        }
                    }
                    WriteOperation::WriteSe(value) => {
                        // Limit value to prevent excessive writes
                        if value.abs() < 500_000 {
                            let _ = writer.write_se(*value);
                        }
                    }
                    WriteOperation::AlignToByte => {
                        let _ = writer.align_to_byte();
                    }
                    WriteOperation::WriteRbspTrailingBits => {
                        let _ = writer.write_rbsp_trailing_bits();
                    }
                }

                // Stop if writer gets too large
                if writer.data().len() > 1024 * 1024 {
                    break;
                }
            }

            // Test additional writer methods
            let _ = writer.position();
            let _ = writer.is_byte_aligned();
        }
        TestMode::Roundtrip => {
            // Write some data first
            let mut writer = BitWriter::new();

            for op in input.writer_operations.iter().take(100) {
                match op {
                    WriteOperation::WriteBit(bit) => {
                        let _ = writer.write_bit(*bit);
                    }
                    WriteOperation::WriteBits { value, n } => {
                        let bits = (*n % 32).max(1);
                        let _ = writer.write_bits(*value, bits);
                    }
                    WriteOperation::WriteBitsU64 { value, n } => {
                        let bits = (*n % 64).max(1);
                        let _ = writer.write_bits_u64(*value, bits);
                    }
                    WriteOperation::WriteUe(value) => {
                        if *value < 10000 {
                            let _ = writer.write_ue(*value);
                        }
                    }
                    WriteOperation::WriteSe(value) => {
                        if value.abs() < 5000 {
                            let _ = writer.write_se(*value);
                        }
                    }
                    WriteOperation::AlignToByte => {
                        let _ = writer.align_to_byte();
                    }
                    WriteOperation::WriteRbspTrailingBits => {
                        let _ = writer.write_rbsp_trailing_bits();
                    }
                }

                // Stop if writer gets too large
                if writer.data().len() > 64 * 1024 {
                    break;
                }
            }

            // Then read the written data
            let data = writer.data();
            if !data.is_empty() {
                let mut reader = BitReader::new(data);

                // Try to read back some data
                for _ in 0..100 {
                    if reader.is_eof() {
                        break;
                    }
                    let _ = reader.read_bit();
                }
            }
        }
    }
});
