//! KLV (Key-Length-Value) triplet handling
//!
//! All data in MXF files is encoded as KLV triplets:
//! - Key: 16-byte Universal Label identifying the data
//! - Length: BER-encoded length of the value
//! - Value: The actual data

use crate::error::{MxfError, Result};
use crate::ul::{UniversalLabel, UL};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

/// A KLV triplet
#[derive(Debug, Clone)]
pub struct Klv {
    /// Universal Label (key)
    pub key: UniversalLabel,
    /// Value data
    pub value: Vec<u8>,
    /// Original offset in file
    pub offset: u64,
}

impl Klv {
    /// Create new KLV
    pub fn new(key: UL, value: Vec<u8>) -> Self {
        Klv {
            key: UniversalLabel(key),
            value,
            offset: 0,
        }
    }

    /// Get value length
    pub fn length(&self) -> usize {
        self.value.len()
    }

    /// Total size including key and length encoding
    pub fn total_size(&self) -> usize {
        16 + ber_length_size(self.value.len()) + self.value.len()
    }

    /// Write KLV to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        // Write key
        writer.write_all(self.key.as_bytes())?;

        // Write length (BER encoded)
        let len_bytes = encode_ber_length(self.value.len());
        writer.write_all(&len_bytes)?;

        // Write value
        writer.write_all(&self.value)?;

        Ok(16 + len_bytes.len() + self.value.len())
    }
}

/// KLV reader for parsing MXF data
pub struct KlvReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> KlvReader<'a> {
    /// Create new KLV reader
    pub fn new(data: &'a [u8]) -> Self {
        KlvReader { data, position: 0 }
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Seek to position
    pub fn seek(&mut self, position: usize) {
        self.position = position;
    }

    /// Check if there's more data
    pub fn has_more(&self) -> bool {
        self.position + 16 <= self.data.len()
    }

    /// Read next KLV
    pub fn read_klv(&mut self) -> Result<Option<Klv>> {
        if !self.has_more() {
            return Ok(None);
        }

        let start_offset = self.position as u64;

        // Read key (16 bytes)
        let mut key = [0u8; 16];
        key.copy_from_slice(&self.data[self.position..self.position + 16]);
        self.position += 16;

        // Read length (BER encoded)
        let (length, len_size) = self.read_ber_length()?;
        self.position += len_size;

        // Check if we have enough data
        if self.position + length > self.data.len() {
            return Err(MxfError::InsufficientData {
                needed: length,
                available: self.data.len() - self.position,
            });
        }

        // Read value
        let value = self.data[self.position..self.position + length].to_vec();
        self.position += length;

        Ok(Some(Klv {
            key: UniversalLabel(key),
            value,
            offset: start_offset,
        }))
    }

    /// Skip current KLV (just move past it without copying value)
    pub fn skip_klv(&mut self) -> Result<Option<UniversalLabel>> {
        if !self.has_more() {
            return Ok(None);
        }

        // Read key
        let mut key = [0u8; 16];
        key.copy_from_slice(&self.data[self.position..self.position + 16]);
        self.position += 16;

        // Read length
        let (length, len_size) = self.read_ber_length()?;
        self.position += len_size;

        // Skip value
        if self.position + length > self.data.len() {
            return Err(MxfError::InsufficientData {
                needed: length,
                available: self.data.len() - self.position,
            });
        }
        self.position += length;

        Ok(Some(UniversalLabel(key)))
    }

    /// Read BER-encoded length
    fn read_ber_length(&self) -> Result<(usize, usize)> {
        if self.position >= self.data.len() {
            return Err(MxfError::BerError("No data for length".into()));
        }

        let first_byte = self.data[self.position];

        if first_byte < 0x80 {
            // Short form: length is in the byte itself
            Ok((first_byte as usize, 1))
        } else if first_byte == 0x80 {
            // Indefinite length (not supported for now)
            Err(MxfError::BerError("Indefinite length not supported".into()))
        } else {
            // Long form: first byte indicates number of length bytes
            let num_bytes = (first_byte & 0x7F) as usize;

            if self.position + 1 + num_bytes > self.data.len() {
                return Err(MxfError::BerError("Not enough bytes for length".into()));
            }

            let mut length: usize = 0;
            for i in 0..num_bytes {
                length = (length << 8) | (self.data[self.position + 1 + i] as usize);
            }

            Ok((length, 1 + num_bytes))
        }
    }

    /// Peek at next key without consuming it
    pub fn peek_key(&self) -> Option<UniversalLabel> {
        if self.position + 16 <= self.data.len() {
            let mut key = [0u8; 16];
            key.copy_from_slice(&self.data[self.position..self.position + 16]);
            Some(UniversalLabel(key))
        } else {
            None
        }
    }
}

/// Calculate BER length encoding size
pub fn ber_length_size(length: usize) -> usize {
    if length < 0x80 {
        1
    } else if length <= 0xFF {
        2
    } else if length <= 0xFFFF {
        3
    } else if length <= 0xFFFFFF {
        4
    } else {
        5
    }
}

/// Encode length as BER
pub fn encode_ber_length(length: usize) -> Vec<u8> {
    if length < 0x80 {
        vec![length as u8]
    } else if length <= 0xFF {
        vec![0x81, length as u8]
    } else if length <= 0xFFFF {
        vec![0x82, (length >> 8) as u8, length as u8]
    } else if length <= 0xFFFFFF {
        vec![
            0x83,
            (length >> 16) as u8,
            (length >> 8) as u8,
            length as u8,
        ]
    } else {
        vec![
            0x84,
            (length >> 24) as u8,
            (length >> 16) as u8,
            (length >> 8) as u8,
            length as u8,
        ]
    }
}

/// Decode BER length from bytes
pub fn decode_ber_length(data: &[u8]) -> Result<(usize, usize)> {
    if data.is_empty() {
        return Err(MxfError::BerError("No data".into()));
    }

    let first_byte = data[0];

    if first_byte < 0x80 {
        Ok((first_byte as usize, 1))
    } else if first_byte == 0x80 {
        Err(MxfError::BerError("Indefinite length not supported".into()))
    } else {
        let num_bytes = (first_byte & 0x7F) as usize;

        if data.len() < 1 + num_bytes {
            return Err(MxfError::BerError("Not enough bytes".into()));
        }

        let mut length: usize = 0;
        for byte in data.iter().skip(1).take(num_bytes) {
            length = (length << 8) | (*byte as usize);
        }

        Ok((length, 1 + num_bytes))
    }
}

/// Write a local set (2-byte tag, 2-byte length)
pub fn write_local_set<W: Write>(
    writer: &mut W,
    tag: u16,
    value: &[u8],
) -> Result<usize> {
    writer.write_u16::<BigEndian>(tag)?;
    writer.write_u16::<BigEndian>(value.len() as u16)?;
    writer.write_all(value)?;
    Ok(4 + value.len())
}

/// Read a local set entry
pub fn read_local_set(data: &[u8], offset: usize) -> Result<(u16, Vec<u8>, usize)> {
    if offset + 4 > data.len() {
        return Err(MxfError::InsufficientData {
            needed: 4,
            available: data.len() - offset,
        });
    }

    let mut cursor = Cursor::new(&data[offset..]);
    let tag = cursor.read_u16::<BigEndian>()?;
    let length = cursor.read_u16::<BigEndian>()? as usize;

    if offset + 4 + length > data.len() {
        return Err(MxfError::InsufficientData {
            needed: length,
            available: data.len() - offset - 4,
        });
    }

    let value = data[offset + 4..offset + 4 + length].to_vec();
    Ok((tag, value, 4 + length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ber_length_short() {
        let encoded = encode_ber_length(100);
        assert_eq!(encoded, vec![100]);

        let (decoded, size) = decode_ber_length(&encoded).unwrap();
        assert_eq!(decoded, 100);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_ber_length_long() {
        let encoded = encode_ber_length(1000);
        assert_eq!(encoded, vec![0x82, 0x03, 0xE8]);

        let (decoded, size) = decode_ber_length(&encoded).unwrap();
        assert_eq!(decoded, 1000);
        assert_eq!(size, 3);
    }

    #[test]
    fn test_ber_length_very_long() {
        let encoded = encode_ber_length(0x1234567);
        let (decoded, _) = decode_ber_length(&encoded).unwrap();
        assert_eq!(decoded, 0x1234567);
    }

    #[test]
    fn test_klv_write() {
        let klv = Klv::new(
            [0x06, 0x0E, 0x2B, 0x34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 2, 3, 4],
        );

        let mut buffer = Vec::new();
        let size = klv.write(&mut buffer).unwrap();

        assert_eq!(size, 16 + 1 + 4); // key + short length + value
        assert_eq!(buffer[16], 4); // length byte
        assert_eq!(&buffer[17..21], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_klv_reader() {
        // Create a KLV triplet
        let key = [0x06, 0x0E, 0x2B, 0x34, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let value = vec![0xAA, 0xBB, 0xCC];
        let length = 3u8;

        let mut data = Vec::new();
        data.extend_from_slice(&key);
        data.push(length);
        data.extend_from_slice(&value);

        let mut reader = KlvReader::new(&data);
        let klv = reader.read_klv().unwrap().unwrap();

        assert_eq!(klv.key.as_bytes(), &key);
        assert_eq!(klv.value, value);
    }

    #[test]
    fn test_klv_reader_long_length() {
        let key = [0x06, 0x0E, 0x2B, 0x34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let value = vec![0u8; 200];

        let mut data = Vec::new();
        data.extend_from_slice(&key);
        data.push(0x81); // Long form, 1 byte
        data.push(200);
        data.extend_from_slice(&value);

        let mut reader = KlvReader::new(&data);
        let klv = reader.read_klv().unwrap().unwrap();

        assert_eq!(klv.value.len(), 200);
    }

    #[test]
    fn test_local_set() {
        let mut buffer = Vec::new();
        write_local_set(&mut buffer, 0x1234, &[1, 2, 3]).unwrap();

        let (tag, value, size) = read_local_set(&buffer, 0).unwrap();
        assert_eq!(tag, 0x1234);
        assert_eq!(value, vec![1, 2, 3]);
        assert_eq!(size, 7);
    }
}
