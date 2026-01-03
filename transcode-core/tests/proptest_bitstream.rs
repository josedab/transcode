//! Property-based tests for bitstream operations.
//!
//! Uses proptest to verify round-trip correctness of BitReader/BitWriter
//! and related encoding/decoding functions.

use proptest::prelude::*;
use transcode_core::bitstream::{
    add_emulation_prevention, remove_emulation_prevention, BitReader, BitWriter,
};

// =============================================================================
// BitReader/BitWriter Round-Trip Tests
// =============================================================================

proptest! {
    /// Test that writing and reading bits produces the same value.
    #[test]
    fn roundtrip_bits_u8(value in 0u8..=255) {
        let mut writer = BitWriter::new();
        writer.write_bits(value as u32, 8).unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_bits(8).unwrap() as u8;

        prop_assert_eq!(value, read_value);
    }

    /// Test that writing and reading arbitrary bit widths works correctly.
    #[test]
    fn roundtrip_bits_variable_width(value in 0u32..=0xFFFF, width in 1u8..=16) {
        // Mask value to the actual width
        let masked_value = value & ((1u32 << width) - 1);

        let mut writer = BitWriter::new();
        writer.write_bits(masked_value, width).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_bits(width).unwrap();

        prop_assert_eq!(masked_value, read_value);
    }

    /// Test that writing and reading 32-bit values works correctly.
    #[test]
    fn roundtrip_bits_u32(value in any::<u32>()) {
        let mut writer = BitWriter::new();
        writer.write_bits(value, 32).unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_bits(32).unwrap();

        prop_assert_eq!(value, read_value);
    }

    /// Test that writing and reading 64-bit values works correctly.
    #[test]
    fn roundtrip_bits_u64(value in any::<u64>()) {
        let mut writer = BitWriter::new();
        writer.write_bits_u64(value, 64).unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_bits_u64(64).unwrap();

        prop_assert_eq!(value, read_value);
    }

    /// Test that writing and reading multiple values works correctly.
    #[test]
    fn roundtrip_multiple_values(
        v1 in 0u32..=0xFF,
        v2 in 0u32..=0xF,
        v3 in 0u32..=0x3F,
        v4 in 0u32..=0x1
    ) {
        let mut writer = BitWriter::new();
        writer.write_bits(v1, 8).unwrap();
        writer.write_bits(v2, 4).unwrap();
        writer.write_bits(v3, 6).unwrap();
        writer.write_bits(v4, 1).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        prop_assert_eq!(reader.read_bits(8).unwrap(), v1);
        prop_assert_eq!(reader.read_bits(4).unwrap(), v2);
        prop_assert_eq!(reader.read_bits(6).unwrap(), v3);
        prop_assert_eq!(reader.read_bits(1).unwrap(), v4);
    }

    /// Test that individual bits round-trip correctly.
    #[test]
    fn roundtrip_individual_bits(bits in prop::collection::vec(any::<bool>(), 1..100)) {
        let mut writer = BitWriter::new();
        for &bit in &bits {
            writer.write_bit(bit).unwrap();
        }
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        for (i, &expected_bit) in bits.iter().enumerate() {
            let read_bit = reader.read_bit().unwrap();
            prop_assert_eq!(expected_bit, read_bit, "Mismatch at bit {}", i);
        }
    }
}

// =============================================================================
// Exp-Golomb Coding Round-Trip Tests
// =============================================================================

proptest! {
    /// Test unsigned Exp-Golomb (ue) encoding round-trip.
    #[test]
    fn roundtrip_exp_golomb_unsigned(value in 0u32..=65534) {
        // Limit to 65534 to avoid overflow issues with very large values
        let mut writer = BitWriter::new();
        writer.write_ue(value).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_ue().unwrap();

        prop_assert_eq!(value, read_value);
    }

    /// Test signed Exp-Golomb (se) encoding round-trip.
    #[test]
    fn roundtrip_exp_golomb_signed(value in -32767i32..=32767) {
        let mut writer = BitWriter::new();
        writer.write_se(value).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_se().unwrap();

        prop_assert_eq!(value, read_value);
    }

    /// Test multiple Exp-Golomb values in sequence.
    #[test]
    fn roundtrip_multiple_exp_golomb(values in prop::collection::vec(0u32..1000, 1..20)) {
        let mut writer = BitWriter::new();
        for &value in &values {
            writer.write_ue(value).unwrap();
        }
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        for (i, &expected) in values.iter().enumerate() {
            let read_value = reader.read_ue().unwrap();
            prop_assert_eq!(expected, read_value, "Mismatch at index {}", i);
        }
    }

    /// Test mixed signed and unsigned Exp-Golomb values.
    #[test]
    fn roundtrip_mixed_exp_golomb(
        ue_values in prop::collection::vec(0u32..1000, 1..10),
        se_values in prop::collection::vec(-500i32..500, 1..10)
    ) {
        let mut writer = BitWriter::new();

        for (&ue, &se) in ue_values.iter().zip(se_values.iter()) {
            writer.write_ue(ue).unwrap();
            writer.write_se(se).unwrap();
        }
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());

        for (i, (&expected_ue, &expected_se)) in ue_values.iter().zip(se_values.iter()).enumerate() {
            let read_ue = reader.read_ue().unwrap();
            let read_se = reader.read_se().unwrap();
            prop_assert_eq!(expected_ue, read_ue, "UE mismatch at index {}", i);
            prop_assert_eq!(expected_se, read_se, "SE mismatch at index {}", i);
        }
    }
}

// =============================================================================
// Emulation Prevention Round-Trip Tests
// =============================================================================

proptest! {
    /// Test emulation prevention byte insertion and removal.
    #[test]
    fn roundtrip_emulation_prevention(data in prop::collection::vec(any::<u8>(), 1..200)) {
        let escaped = add_emulation_prevention(&data);
        let unescaped = remove_emulation_prevention(&escaped);

        prop_assert_eq!(data, unescaped);
    }

    /// Test that emulation prevention never produces start codes.
    #[test]
    fn emulation_prevention_no_start_codes(data in prop::collection::vec(any::<u8>(), 1..200)) {
        let escaped = add_emulation_prevention(&data);

        // Verify no 0x000001 or 0x00000001 sequences exist
        for i in 0..escaped.len().saturating_sub(2) {
            if escaped[i] == 0 && escaped[i + 1] == 0 {
                prop_assert!(
                    escaped[i + 2] != 0 && escaped[i + 2] != 1,
                    "Found potential start code at position {}",
                    i
                );
            }
        }
    }

    /// Test specific byte patterns that require emulation prevention.
    #[test]
    fn emulation_prevention_specific_patterns(
        prefix in prop::collection::vec(any::<u8>(), 0..10),
        suffix in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        // Test with explicit 0x000001 pattern
        let mut data = prefix.clone();
        data.extend_from_slice(&[0x00, 0x00, 0x01]);
        data.extend_from_slice(&suffix);

        let escaped = add_emulation_prevention(&data);

        // Check that the pattern was escaped
        let mut found_unescaped = false;
        for i in 0..escaped.len().saturating_sub(2) {
            if escaped[i] == 0 && escaped[i + 1] == 0 && escaped[i + 2] == 1 {
                found_unescaped = true;
                break;
            }
        }
        prop_assert!(!found_unescaped, "0x000001 pattern should be escaped");

        // Verify round-trip
        let unescaped = remove_emulation_prevention(&escaped);
        prop_assert_eq!(data, unescaped);
    }
}

// =============================================================================
// BitReader Position and State Tests
// =============================================================================

proptest! {
    /// Test that bit position tracking is accurate.
    #[test]
    fn bit_position_tracking(bits_to_read in 1usize..64, data_len in 8usize..32) {
        let data: Vec<u8> = (0..data_len as u8).collect();
        let mut reader = BitReader::new(&data);

        let total_bits = data_len * 8;
        let bits_to_read = bits_to_read.min(total_bits);

        prop_assert_eq!(reader.position(), 0);
        prop_assert_eq!(reader.remaining_bits(), total_bits);

        // Read some bits
        for _ in 0..bits_to_read {
            reader.read_bit().ok();
        }

        prop_assert_eq!(reader.position(), bits_to_read);
        prop_assert_eq!(reader.remaining_bits(), total_bits - bits_to_read);
    }

    /// Test byte alignment behavior.
    #[test]
    fn byte_alignment(initial_bits in 0u8..8, data in prop::collection::vec(any::<u8>(), 2..10)) {
        let mut reader = BitReader::new(&data);

        // Read some initial bits
        for _ in 0..initial_bits {
            let _ = reader.read_bit();
        }

        // Check alignment
        if initial_bits == 0 {
            prop_assert!(reader.is_byte_aligned());
        } else {
            prop_assert!(!reader.is_byte_aligned());
        }

        // Align to byte
        reader.align_to_byte();
        prop_assert!(reader.is_byte_aligned());
    }

    /// Test skip functionality.
    #[test]
    fn skip_bits(skip_count in 1usize..32, data in prop::collection::vec(any::<u8>(), 8..16)) {
        let mut reader = BitReader::new(&data);
        let total_bits = data.len() * 8;
        let skip_count = skip_count.min(total_bits);

        reader.skip(skip_count).unwrap();

        prop_assert_eq!(reader.position(), skip_count);
        prop_assert_eq!(reader.remaining_bits(), total_bits - skip_count);
    }
}

// =============================================================================
// BitWriter State Tests
// =============================================================================

proptest! {
    /// Test BitWriter byte alignment.
    #[test]
    fn writer_byte_alignment(bits in 1u8..8) {
        let mut writer = BitWriter::new();

        prop_assert!(writer.is_byte_aligned());

        // Write some bits
        for _ in 0..bits {
            writer.write_bit(true).unwrap();
        }

        if bits == 8 {
            prop_assert!(writer.is_byte_aligned());
        } else {
            prop_assert!(!writer.is_byte_aligned());
        }

        // Align to byte
        writer.align_to_byte().unwrap();
        prop_assert!(writer.is_byte_aligned());
    }

    /// Test RBSP trailing bits.
    #[test]
    fn rbsp_trailing_bits(initial_bits in 0u8..7) {
        let mut writer = BitWriter::new();

        // Write some initial bits
        for _ in 0..initial_bits {
            writer.write_bit(false).unwrap();
        }

        // Write RBSP trailing bits
        writer.write_rbsp_trailing_bits().unwrap();

        // Should be byte aligned now
        prop_assert!(writer.is_byte_aligned());

        // Verify the data ends with a 1 followed by zeros
        let data = writer.data();
        prop_assert!(!data.is_empty());

        // The last byte should have a 1 bit at some position
        let last_byte = data[data.len() - 1];
        // Find the position of the trailing 1 bit
        let trailing_zeros = last_byte.trailing_zeros();
        let expected_trailing_zeros = (7 - initial_bits) as u32;
        if initial_bits < 7 {
            prop_assert!(trailing_zeros <= expected_trailing_zeros);
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

proptest! {
    /// Test Exp-Golomb with small values (common in H.264).
    #[test]
    fn exp_golomb_small_values(value in 0u32..32) {
        let mut writer = BitWriter::new();
        writer.write_ue(value).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_ue().unwrap();

        prop_assert_eq!(value, read_value);
    }

    /// Test signed Exp-Golomb with typical motion vector ranges.
    #[test]
    fn exp_golomb_motion_vector_range(value in -128i32..128) {
        let mut writer = BitWriter::new();
        writer.write_se(value).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        let read_value = reader.read_se().unwrap();

        prop_assert_eq!(value, read_value);
    }
}

// =============================================================================
// Non-proptest Unit Tests for Edge Cases
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_exp_golomb_zero() {
        let mut writer = BitWriter::new();
        writer.write_ue(0).unwrap();
        writer.align_to_byte().unwrap();

        let mut reader = BitReader::new(writer.data());
        assert_eq!(reader.read_ue().unwrap(), 0);
    }

    #[test]
    fn test_exp_golomb_powers_of_two() {
        for exp in 0..15 {
            let value = (1u32 << exp) - 1;
            let mut writer = BitWriter::new();
            writer.write_ue(value).unwrap();
            writer.align_to_byte().unwrap();

            let mut reader = BitReader::new(writer.data());
            assert_eq!(reader.read_ue().unwrap(), value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_signed_exp_golomb_symmetry() {
        for value in -100..=100 {
            let mut writer = BitWriter::new();
            writer.write_se(value).unwrap();
            writer.align_to_byte().unwrap();

            let mut reader = BitReader::new(writer.data());
            assert_eq!(reader.read_se().unwrap(), value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_emulation_prevention_all_patterns() {
        // Test all critical patterns: 0x000000, 0x000001, 0x000002, 0x000003
        for byte3 in 0u8..=3 {
            let data = vec![0x00, 0x00, byte3];
            let escaped = add_emulation_prevention(&data);
            let unescaped = remove_emulation_prevention(&escaped);
            assert_eq!(data, unescaped, "Failed for pattern 0x0000{:02x}", byte3);
        }
    }

    #[test]
    fn test_emulation_prevention_chained() {
        // Test multiple consecutive patterns
        let data = vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03];
        let escaped = add_emulation_prevention(&data);
        let unescaped = remove_emulation_prevention(&escaped);
        assert_eq!(data, unescaped);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let escaped = add_emulation_prevention(&data);
        let unescaped = remove_emulation_prevention(&escaped);
        assert_eq!(data, unescaped);
    }

    #[test]
    fn test_peek_does_not_consume() {
        let data = [0b10110100];
        let reader = BitReader::new(&data);

        let peek1 = reader.peek_bit().unwrap();
        let peek2 = reader.peek_bit().unwrap();
        assert_eq!(peek1, peek2);
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn test_peek_bits_does_not_consume() {
        let data = [0b10110100, 0b11001010];
        let reader = BitReader::new(&data);

        let peek1 = reader.peek_bits(8).unwrap();
        let peek2 = reader.peek_bits(8).unwrap();
        assert_eq!(peek1, peek2);
        assert_eq!(reader.position(), 0);
    }
}
