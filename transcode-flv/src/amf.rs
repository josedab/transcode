//! AMF0 encoding for FLV metadata.
//!
//! AMF (Action Message Format) is used to serialize ActionScript objects.
//! FLV uses AMF0 for script data tags, particularly for the onMetaData event.
//!
//! ## Supported Types
//!
//! - Number (f64)
//! - Boolean
//! - String (short and long)
//! - Object
//! - ECMA Array (associative array)
//! - Strict Array
//! - Null
//! - Undefined

use crate::error::{FlvError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Write};

/// AMF0 type markers.
pub mod markers {
    /// Number type (f64).
    pub const NUMBER: u8 = 0x00;
    /// Boolean type.
    pub const BOOLEAN: u8 = 0x01;
    /// String type (short, max 65535 bytes).
    pub const STRING: u8 = 0x02;
    /// Object type.
    pub const OBJECT: u8 = 0x03;
    /// MovieClip type (reserved).
    pub const MOVIE_CLIP: u8 = 0x04;
    /// Null type.
    pub const NULL: u8 = 0x05;
    /// Undefined type.
    pub const UNDEFINED: u8 = 0x06;
    /// Reference type.
    pub const REFERENCE: u8 = 0x07;
    /// ECMA Array type (associative array).
    pub const ECMA_ARRAY: u8 = 0x08;
    /// Object end marker.
    pub const OBJECT_END: u8 = 0x09;
    /// Strict Array type.
    pub const STRICT_ARRAY: u8 = 0x0A;
    /// Date type.
    pub const DATE: u8 = 0x0B;
    /// Long String type (max 4GB).
    pub const LONG_STRING: u8 = 0x0C;
    /// Unsupported type.
    pub const UNSUPPORTED: u8 = 0x0D;
    /// RecordSet type (reserved).
    pub const RECORD_SET: u8 = 0x0E;
    /// XML Document type.
    pub const XML_DOCUMENT: u8 = 0x0F;
    /// Typed Object type.
    pub const TYPED_OBJECT: u8 = 0x10;
    /// AMF3 switch marker.
    pub const AMF3_SWITCH: u8 = 0x11;
}

/// AMF0 value.
#[derive(Debug, Clone, PartialEq)]
pub enum AmfValue {
    /// Number (f64).
    Number(f64),
    /// Boolean.
    Boolean(bool),
    /// String.
    String(String),
    /// Object (key-value pairs).
    Object(HashMap<String, AmfValue>),
    /// Null.
    Null,
    /// Undefined.
    Undefined,
    /// ECMA Array (associative array).
    EcmaArray(HashMap<String, AmfValue>),
    /// Strict Array (indexed array).
    StrictArray(Vec<AmfValue>),
    /// Date (milliseconds since epoch + timezone offset).
    Date {
        /// Milliseconds since Unix epoch.
        milliseconds: f64,
        /// Timezone offset in minutes.
        timezone: i16,
    },
}

impl AmfValue {
    /// Create a number value.
    pub fn number(value: f64) -> Self {
        Self::Number(value)
    }

    /// Create a boolean value.
    pub fn boolean(value: bool) -> Self {
        Self::Boolean(value)
    }

    /// Create a string value.
    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Create an object value.
    pub fn object(properties: HashMap<String, AmfValue>) -> Self {
        Self::Object(properties)
    }

    /// Create an empty object.
    pub fn empty_object() -> Self {
        Self::Object(HashMap::new())
    }

    /// Create an ECMA array.
    pub fn ecma_array(properties: HashMap<String, AmfValue>) -> Self {
        Self::EcmaArray(properties)
    }

    /// Create a strict array.
    pub fn strict_array(values: Vec<AmfValue>) -> Self {
        Self::StrictArray(values)
    }

    /// Create a null value.
    pub fn null() -> Self {
        Self::Null
    }

    /// Create an undefined value.
    pub fn undefined() -> Self {
        Self::Undefined
    }

    /// Create a date value.
    pub fn date(milliseconds: f64, timezone: i16) -> Self {
        Self::Date {
            milliseconds,
            timezone,
        }
    }

    /// Get as a number.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as a boolean.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as an object.
    pub fn as_object(&self) -> Option<&HashMap<String, AmfValue>> {
        match self {
            Self::Object(o) | Self::EcmaArray(o) => Some(o),
            _ => None,
        }
    }

    /// Get as a strict array.
    pub fn as_array(&self) -> Option<&[AmfValue]> {
        match self {
            Self::StrictArray(a) => Some(a),
            _ => None,
        }
    }

    /// Check if this is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Check if this is undefined.
    pub fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }

    /// Parse an AMF0 value from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        let marker = reader.read_u8()?;

        match marker {
            markers::NUMBER => {
                let value = reader.read_f64::<BigEndian>()?;
                Ok(Self::Number(value))
            }
            markers::BOOLEAN => {
                let value = reader.read_u8()? != 0;
                Ok(Self::Boolean(value))
            }
            markers::STRING => {
                let value = read_string(reader)?;
                Ok(Self::String(value))
            }
            markers::OBJECT => {
                let properties = read_object_properties(reader)?;
                Ok(Self::Object(properties))
            }
            markers::NULL => Ok(Self::Null),
            markers::UNDEFINED => Ok(Self::Undefined),
            markers::ECMA_ARRAY => {
                // Read approximate count (can be ignored)
                let _count = reader.read_u32::<BigEndian>()?;
                let properties = read_object_properties(reader)?;
                Ok(Self::EcmaArray(properties))
            }
            markers::STRICT_ARRAY => {
                let count = reader.read_u32::<BigEndian>()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(Self::parse(reader)?);
                }
                Ok(Self::StrictArray(values))
            }
            markers::DATE => {
                let milliseconds = reader.read_f64::<BigEndian>()?;
                let timezone = reader.read_i16::<BigEndian>()?;
                Ok(Self::Date {
                    milliseconds,
                    timezone,
                })
            }
            markers::LONG_STRING => {
                let value = read_long_string(reader)?;
                Ok(Self::String(value))
            }
            _ => Err(FlvError::InvalidAmfType(marker)),
        }
    }

    /// Write an AMF0 value to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        match self {
            Self::Number(value) => {
                writer.write_u8(markers::NUMBER)?;
                writer.write_f64::<BigEndian>(*value)?;
                Ok(9)
            }
            Self::Boolean(value) => {
                writer.write_u8(markers::BOOLEAN)?;
                writer.write_u8(if *value { 1 } else { 0 })?;
                Ok(2)
            }
            Self::String(value) => {
                if value.len() > 65535 {
                    // Use long string
                    writer.write_u8(markers::LONG_STRING)?;
                    writer.write_u32::<BigEndian>(value.len() as u32)?;
                    writer.write_all(value.as_bytes())?;
                    Ok(5 + value.len())
                } else {
                    writer.write_u8(markers::STRING)?;
                    write_string(writer, value)?;
                    Ok(3 + value.len())
                }
            }
            Self::Object(properties) => {
                writer.write_u8(markers::OBJECT)?;
                let size = write_object_properties(writer, properties)?;
                Ok(1 + size)
            }
            Self::Null => {
                writer.write_u8(markers::NULL)?;
                Ok(1)
            }
            Self::Undefined => {
                writer.write_u8(markers::UNDEFINED)?;
                Ok(1)
            }
            Self::EcmaArray(properties) => {
                writer.write_u8(markers::ECMA_ARRAY)?;
                writer.write_u32::<BigEndian>(properties.len() as u32)?;
                let size = write_object_properties(writer, properties)?;
                Ok(5 + size)
            }
            Self::StrictArray(values) => {
                writer.write_u8(markers::STRICT_ARRAY)?;
                writer.write_u32::<BigEndian>(values.len() as u32)?;
                let mut size = 5;
                for value in values {
                    size += value.write(writer)?;
                }
                Ok(size)
            }
            Self::Date {
                milliseconds,
                timezone,
            } => {
                writer.write_u8(markers::DATE)?;
                writer.write_f64::<BigEndian>(*milliseconds)?;
                writer.write_i16::<BigEndian>(*timezone)?;
                Ok(11)
            }
        }
    }
}

/// Read a short string (max 65535 bytes).
fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let length = reader.read_u16::<BigEndian>()? as usize;
    let mut buffer = vec![0u8; length];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer).map_err(|e| FlvError::InvalidAmf(format!("Invalid UTF-8: {}", e)))
}

/// Read a long string (max 4GB).
fn read_long_string<R: Read>(reader: &mut R) -> Result<String> {
    let length = reader.read_u32::<BigEndian>()? as usize;
    let mut buffer = vec![0u8; length];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer).map_err(|e| FlvError::InvalidAmf(format!("Invalid UTF-8: {}", e)))
}

/// Write a short string.
fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<usize> {
    if value.len() > 65535 {
        return Err(FlvError::AmfStringTooLong(value.len()));
    }
    writer.write_u16::<BigEndian>(value.len() as u16)?;
    writer.write_all(value.as_bytes())?;
    Ok(2 + value.len())
}

/// Read object properties until OBJECT_END marker.
fn read_object_properties<R: Read>(reader: &mut R) -> Result<HashMap<String, AmfValue>> {
    let mut properties = HashMap::new();

    loop {
        let key = read_string(reader)?;
        if key.is_empty() {
            let end_marker = reader.read_u8()?;
            if end_marker != markers::OBJECT_END {
                return Err(FlvError::InvalidAmf("Expected object end marker".to_string()));
            }
            break;
        }

        let value = AmfValue::parse(reader)?;
        properties.insert(key, value);
    }

    Ok(properties)
}

/// Write object properties with OBJECT_END marker.
fn write_object_properties<W: Write>(
    writer: &mut W,
    properties: &HashMap<String, AmfValue>,
) -> Result<usize> {
    let mut size = 0;

    for (key, value) in properties {
        size += write_string(writer, key)?;
        size += value.write(writer)?;
    }

    // Write empty string + object end marker
    writer.write_u16::<BigEndian>(0)?; // Empty string length
    writer.write_u8(markers::OBJECT_END)?;
    size += 3;

    Ok(size)
}

/// FLV metadata builder.
#[derive(Debug, Default)]
pub struct MetadataBuilder {
    properties: HashMap<String, AmfValue>,
}

impl MetadataBuilder {
    /// Create a new metadata builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a number property.
    pub fn number(mut self, key: &str, value: f64) -> Self {
        self.properties.insert(key.to_string(), AmfValue::Number(value));
        self
    }

    /// Set a boolean property.
    pub fn boolean(mut self, key: &str, value: bool) -> Self {
        self.properties.insert(key.to_string(), AmfValue::Boolean(value));
        self
    }

    /// Set a string property.
    pub fn string(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_string(), AmfValue::String(value.to_string()));
        self
    }

    /// Set duration in seconds.
    pub fn duration(self, seconds: f64) -> Self {
        self.number("duration", seconds)
    }

    /// Set video width.
    pub fn width(self, width: u32) -> Self {
        self.number("width", width as f64)
    }

    /// Set video height.
    pub fn height(self, height: u32) -> Self {
        self.number("height", height as f64)
    }

    /// Set video codec.
    pub fn video_codec_id(self, codec_id: u8) -> Self {
        self.number("videocodecid", codec_id as f64)
    }

    /// Set audio codec.
    pub fn audio_codec_id(self, codec_id: u8) -> Self {
        self.number("audiocodecid", codec_id as f64)
    }

    /// Set video data rate (kbps).
    pub fn video_data_rate(self, kbps: f64) -> Self {
        self.number("videodatarate", kbps)
    }

    /// Set audio data rate (kbps).
    pub fn audio_data_rate(self, kbps: f64) -> Self {
        self.number("audiodatarate", kbps)
    }

    /// Set frame rate.
    pub fn frame_rate(self, fps: f64) -> Self {
        self.number("framerate", fps)
    }

    /// Set audio sample rate.
    pub fn audio_sample_rate(self, rate: u32) -> Self {
        self.number("audiosamplerate", rate as f64)
    }

    /// Set audio sample size.
    pub fn audio_sample_size(self, bits: u8) -> Self {
        self.number("audiosamplesize", bits as f64)
    }

    /// Set stereo flag.
    pub fn stereo(self, stereo: bool) -> Self {
        self.boolean("stereo", stereo)
    }

    /// Set file size.
    pub fn file_size(self, size: u64) -> Self {
        self.number("filesize", size as f64)
    }

    /// Set encoder name.
    pub fn encoder(self, name: &str) -> Self {
        self.string("encoder", name)
    }

    /// Build the metadata as an ECMA array.
    pub fn build(self) -> AmfValue {
        AmfValue::EcmaArray(self.properties)
    }

    /// Build a complete onMetaData script data tag.
    pub fn build_script_data(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Write "onMetaData" string
        AmfValue::String("onMetaData".to_string())
            .write(&mut data)
            .unwrap();

        // Write metadata ECMA array
        self.build().write(&mut data).unwrap();

        data
    }
}

/// Parse onMetaData from script data.
pub fn parse_on_metadata(data: &[u8]) -> Result<HashMap<String, AmfValue>> {
    let mut cursor = std::io::Cursor::new(data);

    // Read event name (should be "onMetaData")
    let event = AmfValue::parse(&mut cursor)?;
    if event.as_str() != Some("onMetaData") {
        return Err(FlvError::InvalidAmf("Expected onMetaData".to_string()));
    }

    // Read metadata object/array
    let metadata = AmfValue::parse(&mut cursor)?;

    match metadata {
        AmfValue::Object(props) | AmfValue::EcmaArray(props) => Ok(props),
        _ => Err(FlvError::InvalidAmf("Expected object or ECMA array".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_number_roundtrip() {
        let value = AmfValue::Number(123.456);

        let mut buffer = Vec::new();
        value.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = AmfValue::parse(&mut cursor).unwrap();

        assert_eq!(value, parsed);
    }

    #[test]
    fn test_boolean_roundtrip() {
        for &b in &[true, false] {
            let value = AmfValue::Boolean(b);

            let mut buffer = Vec::new();
            value.write(&mut buffer).unwrap();

            let mut cursor = Cursor::new(&buffer);
            let parsed = AmfValue::parse(&mut cursor).unwrap();

            assert_eq!(value, parsed);
        }
    }

    #[test]
    fn test_string_roundtrip() {
        let value = AmfValue::String("Hello, World!".to_string());

        let mut buffer = Vec::new();
        value.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = AmfValue::parse(&mut cursor).unwrap();

        assert_eq!(value, parsed);
    }

    #[test]
    fn test_null_undefined() {
        for value in [AmfValue::Null, AmfValue::Undefined] {
            let mut buffer = Vec::new();
            value.write(&mut buffer).unwrap();

            let mut cursor = Cursor::new(&buffer);
            let parsed = AmfValue::parse(&mut cursor).unwrap();

            assert_eq!(value, parsed);
        }
    }

    #[test]
    fn test_strict_array_roundtrip() {
        let value = AmfValue::StrictArray(vec![
            AmfValue::Number(1.0),
            AmfValue::String("two".to_string()),
            AmfValue::Boolean(true),
        ]);

        let mut buffer = Vec::new();
        value.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = AmfValue::parse(&mut cursor).unwrap();

        assert_eq!(value, parsed);
    }

    #[test]
    fn test_ecma_array_roundtrip() {
        let mut props = HashMap::new();
        props.insert("width".to_string(), AmfValue::Number(1920.0));
        props.insert("height".to_string(), AmfValue::Number(1080.0));

        let value = AmfValue::EcmaArray(props);

        let mut buffer = Vec::new();
        value.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = AmfValue::parse(&mut cursor).unwrap();

        // Compare as ECMA arrays
        let parsed_obj = parsed.as_object().unwrap();
        assert_eq!(parsed_obj.get("width").unwrap().as_number(), Some(1920.0));
        assert_eq!(parsed_obj.get("height").unwrap().as_number(), Some(1080.0));
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = MetadataBuilder::new()
            .duration(60.0)
            .width(1920)
            .height(1080)
            .frame_rate(30.0)
            .encoder("transcode-flv")
            .build();

        let props = metadata.as_object().unwrap();
        assert_eq!(props.get("duration").unwrap().as_number(), Some(60.0));
        assert_eq!(props.get("width").unwrap().as_number(), Some(1920.0));
        assert_eq!(props.get("height").unwrap().as_number(), Some(1080.0));
        assert_eq!(props.get("framerate").unwrap().as_number(), Some(30.0));
        assert_eq!(
            props.get("encoder").unwrap().as_str(),
            Some("transcode-flv")
        );
    }

    #[test]
    fn test_build_script_data() {
        let script_data = MetadataBuilder::new()
            .duration(10.0)
            .build_script_data();

        let metadata = parse_on_metadata(&script_data).unwrap();
        assert_eq!(metadata.get("duration").unwrap().as_number(), Some(10.0));
    }

    #[test]
    fn test_value_accessors() {
        let num = AmfValue::Number(42.0);
        assert_eq!(num.as_number(), Some(42.0));
        assert_eq!(num.as_str(), None);

        let s = AmfValue::String("test".to_string());
        assert_eq!(s.as_str(), Some("test"));
        assert_eq!(s.as_number(), None);

        let null = AmfValue::Null;
        assert!(null.is_null());
        assert!(!null.is_undefined());

        let undef = AmfValue::Undefined;
        assert!(undef.is_undefined());
        assert!(!undef.is_null());
    }

    #[test]
    fn test_date_roundtrip() {
        let value = AmfValue::Date {
            milliseconds: 1609459200000.0,
            timezone: 0,
        };

        let mut buffer = Vec::new();
        value.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = AmfValue::parse(&mut cursor).unwrap();

        assert_eq!(value, parsed);
    }
}
