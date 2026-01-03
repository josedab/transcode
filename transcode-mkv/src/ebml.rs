//! EBML (Extensible Binary Meta Language) parsing utilities.
//!
//! EBML is the binary format underlying Matroska/WebM. It uses variable-length
//! integers (VINTs) for both element IDs and sizes.

use crate::error::{MkvError, Result};
use std::io::{Read, Seek, SeekFrom, Write};

/// Maximum recursion depth for nested elements.
pub const MAX_RECURSION_DEPTH: u32 = 64;

/// Maximum VINT length in bytes.
pub const MAX_VINT_LENGTH: usize = 8;

/// Special value indicating unknown size (all 1s except marker bit).
pub const UNKNOWN_SIZE: u64 = u64::MAX;

/// Read a variable-length integer (VINT) from a reader.
///
/// EBML VINTs use a leading bit pattern to indicate the length:
/// - 1xxxxxxx: 1 byte (7 bits of data)
/// - 01xxxxxx xxxxxxxx: 2 bytes (14 bits)
/// - 001xxxxx xxxxxxxx xxxxxxxx: 3 bytes (21 bits)
/// - etc.
///
/// Returns the decoded value and the number of bytes read.
pub fn read_vint<R: Read>(reader: &mut R) -> Result<(u64, usize)> {
    let mut first_byte = [0u8; 1];
    reader.read_exact(&mut first_byte)?;

    if first_byte[0] == 0 {
        return Err(MkvError::InvalidVint { offset: 0 });
    }

    // Determine length from leading bit pattern
    let length = first_byte[0].leading_zeros() as usize + 1;
    if length > MAX_VINT_LENGTH {
        return Err(MkvError::VintOverflow);
    }

    // Build the value, masking out the length indicator bits
    let mask = 0xFF >> length;
    let mut value = (first_byte[0] & mask) as u64;

    // Read remaining bytes
    if length > 1 {
        let mut remaining = vec![0u8; length - 1];
        reader.read_exact(&mut remaining)?;

        for byte in remaining {
            value = (value << 8) | byte as u64;
        }
    }

    Ok((value, length))
}

/// Read a VINT as an element ID.
///
/// Element IDs include the VINT marker bits as part of the ID.
pub fn read_element_id<R: Read>(reader: &mut R) -> Result<(u32, usize)> {
    let mut first_byte = [0u8; 1];
    reader.read_exact(&mut first_byte)?;

    if first_byte[0] == 0 {
        return Err(MkvError::InvalidVint { offset: 0 });
    }

    let length = first_byte[0].leading_zeros() as usize + 1;
    if length > 4 {
        return Err(MkvError::InvalidElementId { offset: 0 });
    }

    let mut value = first_byte[0] as u32;

    if length > 1 {
        let mut remaining = vec![0u8; length - 1];
        reader.read_exact(&mut remaining)?;

        for byte in remaining {
            value = (value << 8) | byte as u32;
        }
    }

    Ok((value, length))
}

/// Read an element size (VINT with possible unknown size).
///
/// Returns `None` if the size is unknown (streaming mode).
pub fn read_element_size<R: Read>(reader: &mut R) -> Result<(Option<u64>, usize)> {
    let (value, length) = read_vint(reader)?;

    // Check for unknown size marker (all data bits set to 1)
    let unknown_marker = match length {
        1 => 0x7F,
        2 => 0x3FFF,
        3 => 0x1FFFFF,
        4 => 0x0FFFFFFF,
        5 => 0x07FFFFFFFF,
        6 => 0x03FFFFFFFFFF,
        7 => 0x01FFFFFFFFFFFF,
        8 => 0x00FFFFFFFFFFFFFF,
        _ => return Err(MkvError::VintOverflow),
    };

    if value == unknown_marker {
        Ok((None, length))
    } else {
        Ok((Some(value), length))
    }
}

/// Write a variable-length integer.
pub fn write_vint<W: Write>(writer: &mut W, value: u64) -> Result<usize> {
    let (bytes, length) = encode_vint(value)?;
    writer.write_all(&bytes[..length])?;
    Ok(length)
}

/// Encode a value as a VINT.
///
/// Returns the encoded bytes and the length.
pub fn encode_vint(value: u64) -> Result<([u8; 8], usize)> {
    let length = vint_length(value);
    let mut bytes = [0u8; 8];

    // Set the length marker bit
    let marker = 0x80 >> (length - 1);

    // Encode the value
    let mut v = value;
    for i in (0..length).rev() {
        bytes[i] = (v & 0xFF) as u8;
        v >>= 8;
    }

    // Set the marker bit
    bytes[0] |= marker;

    Ok((bytes, length))
}

/// Calculate the minimum number of bytes needed to encode a value as a VINT.
pub fn vint_length(value: u64) -> usize {
    if value < 0x7F {
        1
    } else if value < 0x3FFF {
        2
    } else if value < 0x1FFFFF {
        3
    } else if value < 0x0FFFFFFF {
        4
    } else if value < 0x07FFFFFFFF {
        5
    } else if value < 0x03FFFFFFFFFF {
        6
    } else if value < 0x01FFFFFFFFFFFF {
        7
    } else {
        8
    }
}

/// Write an element ID.
pub fn write_element_id<W: Write>(writer: &mut W, id: u32) -> Result<usize> {
    let bytes = id.to_be_bytes();

    // Find the first non-zero byte
    let start = bytes.iter().position(|&b| b != 0).unwrap_or(3);
    writer.write_all(&bytes[start..])?;
    Ok(4 - start)
}

/// Write an element with unknown size (streaming mode).
pub fn write_unknown_size<W: Write>(writer: &mut W, length: usize) -> Result<usize> {
    let bytes: &[u8] = match length {
        1 => &[0xFF],
        2 => &[0x7F, 0xFF],
        3 => &[0x3F, 0xFF, 0xFF],
        4 => &[0x1F, 0xFF, 0xFF, 0xFF],
        5 => &[0x0F, 0xFF, 0xFF, 0xFF, 0xFF],
        6 => &[0x07, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        7 => &[0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        8 => &[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        _ => return Err(MkvError::VintOverflow),
    };

    writer.write_all(bytes)?;
    Ok(length)
}

/// An EBML element header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementHeader {
    /// The element ID.
    pub id: u32,
    /// The element size (None for unknown size).
    pub size: Option<u64>,
    /// Total header size in bytes.
    pub header_size: usize,
}

impl ElementHeader {
    /// Read an element header from a reader.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let (id, id_len) = read_element_id(reader)?;
        let (size, size_len) = read_element_size(reader)?;

        Ok(Self {
            id,
            size,
            header_size: id_len + size_len,
        })
    }

    /// Write an element header to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        let id_len = write_element_id(writer, self.id)?;
        let size_len = match self.size {
            Some(size) => write_vint(writer, size)?,
            None => write_unknown_size(writer, 4)?, // Use 4-byte unknown size
        };
        Ok(id_len + size_len)
    }

    /// Get the total size of this element (header + content).
    pub fn total_size(&self) -> Option<u64> {
        self.size.map(|s| s + self.header_size as u64)
    }
}

/// EBML document header information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmlHeader {
    /// EBML version.
    pub version: u64,
    /// EBML read version.
    pub read_version: u64,
    /// Maximum ID length.
    pub max_id_length: u64,
    /// Maximum size length.
    pub max_size_length: u64,
    /// Document type (e.g., "matroska" or "webm").
    pub doc_type: String,
    /// Document type version.
    pub doc_type_version: u64,
    /// Document type read version.
    pub doc_type_read_version: u64,
}

impl Default for EbmlHeader {
    fn default() -> Self {
        Self {
            version: 1,
            read_version: 1,
            max_id_length: 4,
            max_size_length: 8,
            doc_type: "matroska".to_string(),
            doc_type_version: 4,
            doc_type_read_version: 2,
        }
    }
}

impl EbmlHeader {
    /// Create a WebM header.
    pub fn webm() -> Self {
        Self {
            doc_type: "webm".to_string(),
            doc_type_version: 4,
            doc_type_read_version: 2,
            ..Default::default()
        }
    }

    /// Check if this is a WebM document.
    pub fn is_webm(&self) -> bool {
        self.doc_type == "webm"
    }

    /// Check if this is a Matroska document.
    pub fn is_matroska(&self) -> bool {
        self.doc_type == "matroska"
    }
}

/// Read a signed integer from EBML data.
pub fn read_signed_int(data: &[u8]) -> i64 {
    if data.is_empty() {
        return 0;
    }

    // Sign-extend the first byte
    let mut value = if data[0] & 0x80 != 0 {
        -1i64
    } else {
        0i64
    };

    for &byte in data {
        value = (value << 8) | byte as i64;
    }

    value
}

/// Read an unsigned integer from EBML data.
pub fn read_unsigned_int(data: &[u8]) -> u64 {
    let mut value = 0u64;
    for &byte in data {
        value = (value << 8) | byte as u64;
    }
    value
}

/// Read a float from EBML data (4 or 8 bytes).
pub fn read_float(data: &[u8]) -> f64 {
    match data.len() {
        4 => {
            let bits = u32::from_be_bytes(data.try_into().unwrap());
            f32::from_bits(bits) as f64
        }
        8 => {
            let bits = u64::from_be_bytes(data.try_into().unwrap());
            f64::from_bits(bits)
        }
        0 => 0.0,
        _ => f64::NAN,
    }
}

/// Read a UTF-8 string from EBML data.
pub fn read_string(data: &[u8]) -> Result<String> {
    // Find null terminator if present
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    String::from_utf8(data[..end].to_vec())
        .map_err(|e| MkvError::Other(format!("Invalid UTF-8 string: {}", e)))
}

/// Read a date from EBML data (nanoseconds since 2001-01-01).
pub fn read_date(data: &[u8]) -> i64 {
    read_signed_int(data)
}

/// Write a signed integer in minimal bytes.
pub fn write_signed_int<W: Write>(writer: &mut W, value: i64) -> Result<usize> {
    let bytes = value.to_be_bytes();

    // Find the first significant byte
    let start = if value >= 0 {
        bytes.iter().position(|&b| b != 0 || (b & 0x80) != 0).unwrap_or(7)
    } else {
        bytes.iter().position(|&b| b != 0xFF || (b & 0x80) == 0).unwrap_or(7)
    };

    // Ensure at least one byte is written
    let start = start.min(7);
    writer.write_all(&bytes[start..])?;
    Ok(8 - start)
}

/// Write an unsigned integer in minimal bytes.
pub fn write_unsigned_int<W: Write>(writer: &mut W, value: u64) -> Result<usize> {
    if value == 0 {
        writer.write_all(&[0])?;
        return Ok(1);
    }

    let bytes = value.to_be_bytes();
    let start = bytes.iter().position(|&b| b != 0).unwrap_or(7);
    writer.write_all(&bytes[start..])?;
    Ok(8 - start)
}

/// Write a float (always 8 bytes for precision).
pub fn write_float<W: Write>(writer: &mut W, value: f64) -> Result<usize> {
    let bytes = value.to_bits().to_be_bytes();
    writer.write_all(&bytes)?;
    Ok(8)
}

/// Write a UTF-8 string.
pub fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<usize> {
    writer.write_all(value.as_bytes())?;
    Ok(value.len())
}

/// Skip an element's content.
pub fn skip_element<R: Read + Seek>(reader: &mut R, size: u64) -> Result<()> {
    reader.seek(SeekFrom::Current(size as i64))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_vint_1byte() {
        let data = [0x81]; // 1 in 1-byte VINT
        let mut cursor = Cursor::new(&data);
        let (value, len) = read_vint(&mut cursor).unwrap();
        assert_eq!(value, 1);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_read_vint_2byte() {
        let data = [0x40, 0x81]; // 129 in 2-byte VINT
        let mut cursor = Cursor::new(&data);
        let (value, len) = read_vint(&mut cursor).unwrap();
        assert_eq!(value, 129);
        assert_eq!(len, 2);
    }

    #[test]
    fn test_read_vint_3byte() {
        let data = [0x20, 0x40, 0x00]; // 16384 in 3-byte VINT
        let mut cursor = Cursor::new(&data);
        let (value, len) = read_vint(&mut cursor).unwrap();
        assert_eq!(value, 16384);
        assert_eq!(len, 3);
    }

    #[test]
    fn test_read_element_id_1byte() {
        let data = [0xEC]; // Void element ID
        let mut cursor = Cursor::new(&data);
        let (id, len) = read_element_id(&mut cursor).unwrap();
        assert_eq!(id, 0xEC);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_read_element_id_4byte() {
        let data = [0x1A, 0x45, 0xDF, 0xA3]; // EBML header ID
        let mut cursor = Cursor::new(&data);
        let (id, len) = read_element_id(&mut cursor).unwrap();
        assert_eq!(id, 0x1A45DFA3);
        assert_eq!(len, 4);
    }

    #[test]
    fn test_read_unknown_size() {
        let data = [0xFF]; // 1-byte unknown size
        let mut cursor = Cursor::new(&data);
        let (size, len) = read_element_size(&mut cursor).unwrap();
        assert_eq!(size, None);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_encode_vint() {
        let (bytes, len) = encode_vint(1).unwrap();
        assert_eq!(len, 1);
        assert_eq!(bytes[0], 0x81);

        let (bytes, len) = encode_vint(129).unwrap();
        assert_eq!(len, 2);
        assert_eq!(&bytes[..2], &[0x40, 0x81]);
    }

    #[test]
    fn test_vint_roundtrip() {
        for value in [0, 1, 127, 128, 16383, 16384, 1000000] {
            let (encoded, len) = encode_vint(value).unwrap();
            let mut cursor = Cursor::new(&encoded[..len]);
            let (decoded, decoded_len) = read_vint(&mut cursor).unwrap();
            assert_eq!(value, decoded);
            assert_eq!(len, decoded_len);
        }
    }

    #[test]
    fn test_read_signed_int() {
        assert_eq!(read_signed_int(&[0x00]), 0);
        assert_eq!(read_signed_int(&[0x01]), 1);
        assert_eq!(read_signed_int(&[0xFF]), -1);
        assert_eq!(read_signed_int(&[0x00, 0x80]), 128);
        assert_eq!(read_signed_int(&[0xFF, 0x7F]), -129);
    }

    #[test]
    fn test_read_unsigned_int() {
        assert_eq!(read_unsigned_int(&[0x00]), 0);
        assert_eq!(read_unsigned_int(&[0x01]), 1);
        assert_eq!(read_unsigned_int(&[0xFF]), 255);
        assert_eq!(read_unsigned_int(&[0x01, 0x00]), 256);
    }

    #[test]
    fn test_read_float() {
        // 4-byte float
        let data = 1.0f32.to_bits().to_be_bytes();
        assert_eq!(read_float(&data), 1.0);

        // 8-byte float
        let data = 1.0f64.to_bits().to_be_bytes();
        assert_eq!(read_float(&data), 1.0);
    }

    #[test]
    fn test_read_string() {
        let data = b"hello\x00world";
        let s = read_string(data).unwrap();
        assert_eq!(s, "hello");

        let data = b"hello";
        let s = read_string(data).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_element_header_roundtrip() {
        let header = ElementHeader {
            id: 0x1A45DFA3,
            size: Some(100),
            header_size: 0,
        };

        let mut buffer = Vec::new();
        let written = header.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let read_header = ElementHeader::read(&mut cursor).unwrap();

        assert_eq!(header.id, read_header.id);
        assert_eq!(header.size, read_header.size);
        assert_eq!(written, read_header.header_size);
    }

    #[test]
    fn test_ebml_header_default() {
        let header = EbmlHeader::default();
        assert!(header.is_matroska());
        assert!(!header.is_webm());
    }

    #[test]
    fn test_ebml_header_webm() {
        let header = EbmlHeader::webm();
        assert!(header.is_webm());
        assert!(!header.is_matroska());
    }
}
