//! TIFF Image File Directory (IFD) handling

use crate::error::{Result, TiffError};
use crate::tags::{data_type, tag_name};
use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};

/// IFD entry value
#[derive(Debug, Clone)]
pub enum IfdValue {
    /// Byte values
    Bytes(Vec<u8>),
    /// ASCII string
    Ascii(String),
    /// Short (u16) values
    Shorts(Vec<u16>),
    /// Long (u32) values
    Longs(Vec<u32>),
    /// Rational (numerator/denominator) values
    Rationals(Vec<(u32, u32)>),
    /// Signed byte values
    SBytes(Vec<i8>),
    /// Undefined bytes
    Undefined(Vec<u8>),
    /// Signed short values
    SShorts(Vec<i16>),
    /// Signed long values
    SLongs(Vec<i32>),
    /// Signed rational values
    SRationals(Vec<(i32, i32)>),
    /// Float values
    Floats(Vec<f32>),
    /// Double values
    Doubles(Vec<f64>),
}

impl IfdValue {
    /// Get as single u16 value
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            IfdValue::Bytes(v) if !v.is_empty() => Some(v[0] as u16),
            IfdValue::Shorts(v) if !v.is_empty() => Some(v[0]),
            IfdValue::Longs(v) if !v.is_empty() => Some(v[0] as u16),
            _ => None,
        }
    }

    /// Get as single u32 value
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            IfdValue::Bytes(v) if !v.is_empty() => Some(v[0] as u32),
            IfdValue::Shorts(v) if !v.is_empty() => Some(v[0] as u32),
            IfdValue::Longs(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    /// Get as vector of u16 values
    pub fn as_u16_vec(&self) -> Option<Vec<u16>> {
        match self {
            IfdValue::Bytes(v) => Some(v.iter().map(|&b| b as u16).collect()),
            IfdValue::Shorts(v) => Some(v.clone()),
            IfdValue::Longs(v) => Some(v.iter().map(|&l| l as u16).collect()),
            _ => None,
        }
    }

    /// Get as vector of u32 values
    pub fn as_u32_vec(&self) -> Option<Vec<u32>> {
        match self {
            IfdValue::Bytes(v) => Some(v.iter().map(|&b| b as u32).collect()),
            IfdValue::Shorts(v) => Some(v.iter().map(|&s| s as u32).collect()),
            IfdValue::Longs(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<String> {
        match self {
            IfdValue::Ascii(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Get as rational (f64)
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            IfdValue::Rationals(v) if !v.is_empty() => {
                let (n, d) = v[0];
                if d == 0 {
                    None
                } else {
                    Some(n as f64 / d as f64)
                }
            }
            IfdValue::SRationals(v) if !v.is_empty() => {
                let (n, d) = v[0];
                if d == 0 {
                    None
                } else {
                    Some(n as f64 / d as f64)
                }
            }
            IfdValue::Floats(v) if !v.is_empty() => Some(v[0] as f64),
            IfdValue::Doubles(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            IfdValue::Bytes(v) => Some(v),
            IfdValue::Undefined(v) => Some(v),
            _ => None,
        }
    }

    /// Get data type ID
    pub fn type_id(&self) -> u16 {
        match self {
            IfdValue::Bytes(_) => data_type::BYTE,
            IfdValue::Ascii(_) => data_type::ASCII,
            IfdValue::Shorts(_) => data_type::SHORT,
            IfdValue::Longs(_) => data_type::LONG,
            IfdValue::Rationals(_) => data_type::RATIONAL,
            IfdValue::SBytes(_) => data_type::SBYTE,
            IfdValue::Undefined(_) => data_type::UNDEFINED,
            IfdValue::SShorts(_) => data_type::SSHORT,
            IfdValue::SLongs(_) => data_type::SLONG,
            IfdValue::SRationals(_) => data_type::SRATIONAL,
            IfdValue::Floats(_) => data_type::FLOAT,
            IfdValue::Doubles(_) => data_type::DOUBLE,
        }
    }

    /// Get count of values
    pub fn count(&self) -> u32 {
        match self {
            IfdValue::Bytes(v) => v.len() as u32,
            IfdValue::Ascii(s) => (s.len() + 1) as u32, // Include null terminator
            IfdValue::Shorts(v) => v.len() as u32,
            IfdValue::Longs(v) => v.len() as u32,
            IfdValue::Rationals(v) => v.len() as u32,
            IfdValue::SBytes(v) => v.len() as u32,
            IfdValue::Undefined(v) => v.len() as u32,
            IfdValue::SShorts(v) => v.len() as u32,
            IfdValue::SLongs(v) => v.len() as u32,
            IfdValue::SRationals(v) => v.len() as u32,
            IfdValue::Floats(v) => v.len() as u32,
            IfdValue::Doubles(v) => v.len() as u32,
        }
    }

    /// Get total byte size
    pub fn byte_size(&self) -> usize {
        let type_size = data_type::size(self.type_id());
        type_size * self.count() as usize
    }
}

/// IFD entry
#[derive(Debug, Clone)]
pub struct IfdEntry {
    /// Tag ID
    pub tag: u16,
    /// Value
    pub value: IfdValue,
}

impl IfdEntry {
    /// Create new entry
    pub fn new(tag: u16, value: IfdValue) -> Self {
        IfdEntry { tag, value }
    }

    /// Create short entry
    pub fn short(tag: u16, value: u16) -> Self {
        IfdEntry {
            tag,
            value: IfdValue::Shorts(vec![value]),
        }
    }

    /// Create long entry
    pub fn long(tag: u16, value: u32) -> Self {
        IfdEntry {
            tag,
            value: IfdValue::Longs(vec![value]),
        }
    }

    /// Create rational entry
    pub fn rational(tag: u16, numerator: u32, denominator: u32) -> Self {
        IfdEntry {
            tag,
            value: IfdValue::Rationals(vec![(numerator, denominator)]),
        }
    }

    /// Create ASCII entry
    pub fn ascii(tag: u16, value: &str) -> Self {
        IfdEntry {
            tag,
            value: IfdValue::Ascii(value.to_string()),
        }
    }
}

/// Image File Directory
#[derive(Debug, Clone)]
pub struct Ifd {
    /// Entries by tag
    entries: BTreeMap<u16, IfdEntry>,
    /// Offset to next IFD (0 if none)
    pub next_ifd_offset: u32,
}

impl Default for Ifd {
    fn default() -> Self {
        Ifd::new()
    }
}

impl Ifd {
    /// Create new empty IFD
    pub fn new() -> Self {
        Ifd {
            entries: BTreeMap::new(),
            next_ifd_offset: 0,
        }
    }

    /// Add entry
    pub fn add(&mut self, entry: IfdEntry) {
        self.entries.insert(entry.tag, entry);
    }

    /// Get entry by tag
    pub fn get(&self, tag: u16) -> Option<&IfdEntry> {
        self.entries.get(&tag)
    }

    /// Get value by tag
    pub fn get_value(&self, tag: u16) -> Option<&IfdValue> {
        self.entries.get(&tag).map(|e| &e.value)
    }

    /// Get required u32 value
    pub fn get_required_u32(&self, tag: u16) -> Result<u32> {
        self.get_value(tag)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| TiffError::MissingTag(tag_name(tag).to_string()))
    }

    /// Get optional u32 value with default
    pub fn get_u32_or(&self, tag: u16, default: u32) -> u32 {
        self.get_value(tag)
            .and_then(|v| v.as_u32())
            .unwrap_or(default)
    }

    /// Get required u16 value
    pub fn get_required_u16(&self, tag: u16) -> Result<u16> {
        self.get_value(tag)
            .and_then(|v| v.as_u16())
            .ok_or_else(|| TiffError::MissingTag(tag_name(tag).to_string()))
    }

    /// Get optional u16 value with default
    pub fn get_u16_or(&self, tag: u16, default: u16) -> u16 {
        self.get_value(tag)
            .and_then(|v| v.as_u16())
            .unwrap_or(default)
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries
    pub fn entries(&self) -> impl Iterator<Item = &IfdEntry> {
        self.entries.values()
    }

    /// Read IFD from reader
    pub fn read<R: Read + Seek, B: ByteOrder + 'static>(reader: &mut R, offset: u32) -> Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let num_entries = reader.read_u16::<B>()?;
        let mut ifd = Ifd::new();

        for _ in 0..num_entries {
            let entry = Self::read_entry::<R, B>(reader)?;
            ifd.add(entry);
        }

        ifd.next_ifd_offset = reader.read_u32::<B>()?;

        Ok(ifd)
    }

    /// Read single entry
    fn read_entry<R: Read + Seek, B: ByteOrder + 'static>(reader: &mut R) -> Result<IfdEntry> {
        let tag = reader.read_u16::<B>()?;
        let type_id = reader.read_u16::<B>()?;
        let count = reader.read_u32::<B>()?;

        let type_size = data_type::size(type_id);
        let total_size = type_size * count as usize;

        // Read value/offset field (4 bytes)
        let mut value_bytes = [0u8; 4];
        reader.read_exact(&mut value_bytes)?;

        let value = if total_size <= 4 {
            // Value fits in the 4-byte field
            Self::parse_value::<B>(type_id, count, &value_bytes)?
        } else {
            // Value is at an offset
            let offset = if std::any::TypeId::of::<B>() == std::any::TypeId::of::<byteorder::LittleEndian>() {
                byteorder::LittleEndian::read_u32(&value_bytes)
            } else {
                byteorder::BigEndian::read_u32(&value_bytes)
            };

            let current_pos = reader.stream_position()?;
            reader.seek(SeekFrom::Start(offset as u64))?;

            let mut data = vec![0u8; total_size];
            reader.read_exact(&mut data)?;

            reader.seek(SeekFrom::Start(current_pos))?;

            Self::parse_value::<B>(type_id, count, &data)?
        };

        Ok(IfdEntry { tag, value })
    }

    /// Parse value from bytes
    fn parse_value<B: ByteOrder>(type_id: u16, count: u32, data: &[u8]) -> Result<IfdValue> {
        let value = match type_id {
            data_type::BYTE => {
                IfdValue::Bytes(data[..count as usize].to_vec())
            }
            data_type::ASCII => {
                let s = String::from_utf8_lossy(&data[..count as usize]);
                IfdValue::Ascii(s.trim_end_matches('\0').to_string())
            }
            data_type::SHORT => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 2;
                    if offset + 2 <= data.len() {
                        values.push(B::read_u16(&data[offset..]));
                    }
                }
                IfdValue::Shorts(values)
            }
            data_type::LONG => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 4;
                    if offset + 4 <= data.len() {
                        values.push(B::read_u32(&data[offset..]));
                    }
                }
                IfdValue::Longs(values)
            }
            data_type::RATIONAL => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 8;
                    if offset + 8 <= data.len() {
                        let n = B::read_u32(&data[offset..]);
                        let d = B::read_u32(&data[offset + 4..]);
                        values.push((n, d));
                    }
                }
                IfdValue::Rationals(values)
            }
            data_type::SBYTE => {
                let values: Vec<i8> = data[..count as usize]
                    .iter()
                    .map(|&b| b as i8)
                    .collect();
                IfdValue::SBytes(values)
            }
            data_type::UNDEFINED => {
                IfdValue::Undefined(data[..count as usize].to_vec())
            }
            data_type::SSHORT => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 2;
                    if offset + 2 <= data.len() {
                        values.push(B::read_i16(&data[offset..]));
                    }
                }
                IfdValue::SShorts(values)
            }
            data_type::SLONG => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 4;
                    if offset + 4 <= data.len() {
                        values.push(B::read_i32(&data[offset..]));
                    }
                }
                IfdValue::SLongs(values)
            }
            data_type::SRATIONAL => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 8;
                    if offset + 8 <= data.len() {
                        let n = B::read_i32(&data[offset..]);
                        let d = B::read_i32(&data[offset + 4..]);
                        values.push((n, d));
                    }
                }
                IfdValue::SRationals(values)
            }
            data_type::FLOAT => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 4;
                    if offset + 4 <= data.len() {
                        values.push(B::read_f32(&data[offset..]));
                    }
                }
                IfdValue::Floats(values)
            }
            data_type::DOUBLE => {
                let mut values = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * 8;
                    if offset + 8 <= data.len() {
                        values.push(B::read_f64(&data[offset..]));
                    }
                }
                IfdValue::Doubles(values)
            }
            _ => {
                // Unknown type - treat as bytes
                IfdValue::Undefined(data[..(count as usize).min(data.len())].to_vec())
            }
        };

        Ok(value)
    }

    /// Write IFD to writer
    pub fn write<W: Write + Seek, B: ByteOrder>(&self, writer: &mut W) -> Result<u32> {
        let ifd_start = writer.stream_position()? as u32;

        // Write number of entries
        writer.write_u16::<B>(self.entries.len() as u16)?;

        // Collect entries for writing
        let entries: Vec<_> = self.entries.values().collect();

        // Calculate offset for values that don't fit in 4 bytes
        let entries_size = 2 + (entries.len() * 12) + 4; // count + entries + next_ifd
        let mut value_offset = ifd_start + entries_size as u32;

        // First pass: write entries and collect large values
        let mut large_values: Vec<(u32, &IfdValue)> = Vec::new();

        for entry in &entries {
            writer.write_u16::<B>(entry.tag)?;
            writer.write_u16::<B>(entry.value.type_id())?;
            writer.write_u32::<B>(entry.value.count())?;

            let byte_size = entry.value.byte_size();
            if byte_size <= 4 {
                // Write value directly
                let mut value_bytes = [0u8; 4];
                Self::write_value_bytes::<B>(&entry.value, &mut value_bytes);
                writer.write_all(&value_bytes)?;
            } else {
                // Write offset
                writer.write_u32::<B>(value_offset)?;
                large_values.push((value_offset, &entry.value));
                value_offset += byte_size as u32;
                // Align to word boundary
                if !value_offset.is_multiple_of(2) {
                    value_offset += 1;
                }
            }
        }

        // Write next IFD offset
        writer.write_u32::<B>(self.next_ifd_offset)?;

        // Second pass: write large values
        for (_offset, value) in large_values {
            Self::write_value::<W, B>(writer, value)?;
            // Align to word boundary
            let pos = writer.stream_position()?;
            if pos % 2 != 0 {
                writer.write_u8(0)?;
            }
        }

        Ok(ifd_start)
    }

    /// Write value bytes for inline storage
    fn write_value_bytes<B: ByteOrder>(value: &IfdValue, bytes: &mut [u8; 4]) {
        match value {
            IfdValue::Bytes(v) => {
                for (i, &b) in v.iter().take(4).enumerate() {
                    bytes[i] = b;
                }
            }
            IfdValue::Shorts(v) => {
                if !v.is_empty() {
                    B::write_u16(&mut bytes[0..], v[0]);
                }
                if v.len() > 1 {
                    B::write_u16(&mut bytes[2..], v[1]);
                }
            }
            IfdValue::Longs(v) => {
                if !v.is_empty() {
                    B::write_u32(bytes, v[0]);
                }
            }
            IfdValue::Ascii(s) => {
                let s_bytes = s.as_bytes();
                for (i, &b) in s_bytes.iter().take(3).enumerate() {
                    bytes[i] = b;
                }
                bytes[s_bytes.len().min(3)] = 0; // null terminator
            }
            _ => {}
        }
    }

    /// Write value to stream
    fn write_value<W: Write, B: ByteOrder>(writer: &mut W, value: &IfdValue) -> Result<()> {
        match value {
            IfdValue::Bytes(v) | IfdValue::Undefined(v) => {
                writer.write_all(v)?;
            }
            IfdValue::Ascii(s) => {
                writer.write_all(s.as_bytes())?;
                writer.write_u8(0)?; // null terminator
            }
            IfdValue::Shorts(v) => {
                for &val in v {
                    writer.write_u16::<B>(val)?;
                }
            }
            IfdValue::Longs(v) => {
                for &val in v {
                    writer.write_u32::<B>(val)?;
                }
            }
            IfdValue::Rationals(v) => {
                for &(n, d) in v {
                    writer.write_u32::<B>(n)?;
                    writer.write_u32::<B>(d)?;
                }
            }
            IfdValue::SBytes(v) => {
                for &val in v {
                    writer.write_i8(val)?;
                }
            }
            IfdValue::SShorts(v) => {
                for &val in v {
                    writer.write_i16::<B>(val)?;
                }
            }
            IfdValue::SLongs(v) => {
                for &val in v {
                    writer.write_i32::<B>(val)?;
                }
            }
            IfdValue::SRationals(v) => {
                for &(n, d) in v {
                    writer.write_i32::<B>(n)?;
                    writer.write_i32::<B>(d)?;
                }
            }
            IfdValue::Floats(v) => {
                for &val in v {
                    writer.write_f32::<B>(val)?;
                }
            }
            IfdValue::Doubles(v) => {
                for &val in v {
                    writer.write_f64::<B>(val)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::tag;

    #[test]
    fn test_ifd_entry_short() {
        let entry = IfdEntry::short(tag::IMAGE_WIDTH, 1920);
        assert_eq!(entry.tag, tag::IMAGE_WIDTH);
        assert_eq!(entry.value.as_u16(), Some(1920));
    }

    #[test]
    fn test_ifd_entry_long() {
        let entry = IfdEntry::long(tag::IMAGE_LENGTH, 1080);
        assert_eq!(entry.tag, tag::IMAGE_LENGTH);
        assert_eq!(entry.value.as_u32(), Some(1080));
    }

    #[test]
    fn test_ifd_entry_rational() {
        let entry = IfdEntry::rational(tag::X_RESOLUTION, 300, 1);
        assert_eq!(entry.tag, tag::X_RESOLUTION);
        assert_eq!(entry.value.as_f64(), Some(300.0));
    }

    #[test]
    fn test_ifd_add_get() {
        let mut ifd = Ifd::new();
        ifd.add(IfdEntry::short(tag::IMAGE_WIDTH, 640));
        ifd.add(IfdEntry::short(tag::IMAGE_LENGTH, 480));

        assert_eq!(ifd.len(), 2);
        assert_eq!(ifd.get_required_u32(tag::IMAGE_WIDTH).unwrap(), 640);
        assert_eq!(ifd.get_required_u32(tag::IMAGE_LENGTH).unwrap(), 480);
    }

    #[test]
    fn test_ifd_value_byte_size() {
        let shorts = IfdValue::Shorts(vec![1, 2, 3]);
        assert_eq!(shorts.byte_size(), 6);

        let longs = IfdValue::Longs(vec![1, 2]);
        assert_eq!(longs.byte_size(), 8);

        let rationals = IfdValue::Rationals(vec![(1, 2), (3, 4)]);
        assert_eq!(rationals.byte_size(), 16);
    }

    #[test]
    fn test_ifd_ascii() {
        let entry = IfdEntry::ascii(tag::SOFTWARE, "TestApp");
        assert_eq!(entry.value.as_string(), Some("TestApp".to_string()));
        assert_eq!(entry.value.count(), 8); // 7 chars + null
    }
}
