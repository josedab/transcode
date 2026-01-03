#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use transcode_core::bitstream::BitReader;

#[derive(Arbitrary, Debug)]
struct BitReaderInput {
    data: Vec<u8>,
    operations: Vec<BitOperation>,
}

#[derive(Arbitrary, Debug)]
enum BitOperation {
    ReadBit,
    ReadBits(u8),  // 1-32 bits
    ReadUe,        // Unsigned exp-golomb
    ReadSe,        // Signed exp-golomb
    Skip(u8),      // Skip up to 64 bits
    ByteAlign,
}

fuzz_target!(|input: BitReaderInput| {
    if input.data.is_empty() {
        return;
    }

    let mut reader = BitReader::new(&input.data);

    for op in input.operations.iter().take(100) {  // Limit operations
        match op {
            BitOperation::ReadBit => {
                let _ = reader.read_bit();
            }
            BitOperation::ReadBits(n) => {
                let bits = (*n % 32).max(1);  // 1-32 bits
                let _ = reader.read_bits(bits);
            }
            BitOperation::ReadUe => {
                let _ = reader.read_ue();
            }
            BitOperation::ReadSe => {
                let _ = reader.read_se();
            }
            BitOperation::Skip(n) => {
                let bits = (*n % 64) + 1;  // 1-64 bits
                let _ = reader.skip(bits as usize);
            }
            BitOperation::ByteAlign => {
                reader.align_to_byte();
            }
        }

        // Stop if we've exhausted the data
        if reader.remaining_bits() == 0 {
            break;
        }
    }
});
