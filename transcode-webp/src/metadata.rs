//! EXIF and XMP metadata extraction for WebP
//!
//! WebP files can contain:
//! - EXIF chunk: EXIF metadata (camera info, orientation, etc.)
//! - XMP chunk: XMP metadata (Adobe's extensible metadata platform)
//! - ICCP chunk: ICC color profile

use std::io::{Read, Seek};
use std::collections::HashMap;

use crate::error::{WebPError, Result};
use crate::riff::{RiffContainer, ChunkType};

/// Metadata extracted from a WebP file
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// EXIF metadata
    pub exif: Option<ExifData>,
    /// XMP metadata
    pub xmp: Option<XmpData>,
    /// ICC color profile
    pub icc_profile: Option<Vec<u8>>,
}

/// Parsed EXIF data
#[derive(Debug, Clone, Default)]
pub struct ExifData {
    /// Raw EXIF bytes
    pub raw: Vec<u8>,
    /// Parsed EXIF tags (tag ID -> value)
    pub tags: HashMap<u16, ExifValue>,
    /// Image orientation (1-8)
    pub orientation: Option<u16>,
    /// Camera make
    pub make: Option<String>,
    /// Camera model
    pub model: Option<String>,
    /// Software used
    pub software: Option<String>,
    /// Date/time original
    pub datetime_original: Option<String>,
    /// GPS latitude
    pub gps_latitude: Option<f64>,
    /// GPS longitude
    pub gps_longitude: Option<f64>,
}

/// EXIF tag value
#[derive(Debug, Clone)]
pub enum ExifValue {
    Byte(Vec<u8>),
    Ascii(String),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(Vec<(u32, u32)>),
    SignedByte(Vec<i8>),
    Undefined(Vec<u8>),
    SignedShort(Vec<i16>),
    SignedLong(Vec<i32>),
    SignedRational(Vec<(i32, i32)>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

/// Parsed XMP data
#[derive(Debug, Clone, Default)]
pub struct XmpData {
    /// Raw XMP string (XML format)
    pub raw: String,
    /// Parsed properties
    pub properties: HashMap<String, String>,
}

// EXIF tag constants
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_MAKE: u16 = 0x010F;
const TAG_MODEL: u16 = 0x0110;
const TAG_SOFTWARE: u16 = 0x0131;
const TAG_DATETIME_ORIGINAL: u16 = 0x9003;
const TAG_GPS_LATITUDE: u16 = 0x0002;
const TAG_GPS_LATITUDE_REF: u16 = 0x0001;
const TAG_GPS_LONGITUDE: u16 = 0x0004;
const TAG_GPS_LONGITUDE_REF: u16 = 0x0003;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;

/// Extract metadata from a parsed RIFF container
pub fn extract_metadata<R: Read + Seek>(
    container: &RiffContainer,
    _reader: &mut R,
) -> Result<Metadata> {
    let mut metadata = Metadata::default();

    // Extract EXIF
    if let Some(exif_chunk) = container.find_chunk(ChunkType::EXIF) {
        metadata.exif = Some(parse_exif(&exif_chunk.data)?);
    }

    // Extract XMP
    if let Some(xmp_chunk) = container.find_chunk(ChunkType::XMP) {
        metadata.xmp = Some(parse_xmp(&xmp_chunk.data)?);
    }

    // Extract ICC profile
    if let Some(icc_chunk) = container.find_chunk(ChunkType::ICCP) {
        metadata.icc_profile = Some(icc_chunk.data.clone());
    }

    Ok(metadata)
}

/// Parse EXIF data from raw bytes
pub fn parse_exif(data: &[u8]) -> Result<ExifData> {
    if data.len() < 8 {
        return Err(WebPError::InvalidMetadata("EXIF data too short".into()));
    }

    let mut exif = ExifData {
        raw: data.to_vec(),
        ..Default::default()
    };

    // Check for TIFF header
    let (is_little_endian, offset) = parse_tiff_header(data)?;

    // Parse IFD0
    if let Ok(tags) = parse_ifd(data, offset as usize, is_little_endian) {
        exif.tags = tags;

        // Extract common tags
        if let Some(ExifValue::Short(v)) = exif.tags.get(&TAG_ORIENTATION) {
            exif.orientation = v.first().copied();
        }

        if let Some(ExifValue::Ascii(s)) = exif.tags.get(&TAG_MAKE) {
            exif.make = Some(s.clone());
        }

        if let Some(ExifValue::Ascii(s)) = exif.tags.get(&TAG_MODEL) {
            exif.model = Some(s.clone());
        }

        if let Some(ExifValue::Ascii(s)) = exif.tags.get(&TAG_SOFTWARE) {
            exif.software = Some(s.clone());
        }

        // Parse EXIF IFD for more tags
        if let Some(ExifValue::Long(v)) = exif.tags.get(&TAG_EXIF_IFD) {
            if let Some(&exif_offset) = v.first() {
                if let Ok(exif_tags) = parse_ifd(data, exif_offset as usize, is_little_endian) {
                    if let Some(ExifValue::Ascii(s)) = exif_tags.get(&TAG_DATETIME_ORIGINAL) {
                        exif.datetime_original = Some(s.clone());
                    }
                    exif.tags.extend(exif_tags);
                }
            }
        }

        // Parse GPS IFD
        if let Some(ExifValue::Long(v)) = exif.tags.get(&TAG_GPS_IFD) {
            if let Some(&gps_offset) = v.first() {
                if let Ok(gps_tags) = parse_ifd(data, gps_offset as usize, is_little_endian) {
                    let lat_ref = gps_tags.get(&TAG_GPS_LATITUDE_REF)
                        .and_then(|v| if let ExifValue::Ascii(s) = v { Some(s.as_str()) } else { None });
                    let lon_ref = gps_tags.get(&TAG_GPS_LONGITUDE_REF)
                        .and_then(|v| if let ExifValue::Ascii(s) = v { Some(s.as_str()) } else { None });

                    if let Some(ExifValue::Rational(rationals)) = gps_tags.get(&TAG_GPS_LATITUDE) {
                        exif.gps_latitude = Some(parse_gps_coord(rationals, lat_ref.unwrap_or("N")));
                    }

                    if let Some(ExifValue::Rational(rationals)) = gps_tags.get(&TAG_GPS_LONGITUDE) {
                        exif.gps_longitude = Some(parse_gps_coord(rationals, lon_ref.unwrap_or("E")));
                    }

                    exif.tags.extend(gps_tags);
                }
            }
        }
    }

    Ok(exif)
}

/// Parse TIFF header, return (is_little_endian, first_ifd_offset)
fn parse_tiff_header(data: &[u8]) -> Result<(bool, u32)> {
    if data.len() < 8 {
        return Err(WebPError::InvalidMetadata("TIFF header too short".into()));
    }

    let is_little_endian = match &data[0..2] {
        [0x49, 0x49] => true,  // "II" - Intel, little endian
        [0x4D, 0x4D] => false, // "MM" - Motorola, big endian
        _ => return Err(WebPError::InvalidMetadata("Invalid TIFF byte order".into())),
    };

    let magic = read_u16(data, 2, is_little_endian);
    if magic != 42 {
        return Err(WebPError::InvalidMetadata("Invalid TIFF magic number".into()));
    }

    let offset = read_u32(data, 4, is_little_endian);
    Ok((is_little_endian, offset))
}

/// Parse an IFD (Image File Directory)
fn parse_ifd(
    data: &[u8],
    offset: usize,
    is_little_endian: bool,
) -> Result<HashMap<u16, ExifValue>> {
    if offset + 2 > data.len() {
        return Err(WebPError::InvalidMetadata("IFD offset out of bounds".into()));
    }

    let num_entries = read_u16(data, offset, is_little_endian) as usize;
    let mut tags = HashMap::new();

    for i in 0..num_entries {
        let entry_offset = offset + 2 + i * 12;
        if entry_offset + 12 > data.len() {
            break;
        }

        let tag_id = read_u16(data, entry_offset, is_little_endian);
        let tag_type = read_u16(data, entry_offset + 2, is_little_endian);
        let count = read_u32(data, entry_offset + 4, is_little_endian) as usize;

        if let Some(value) = parse_tag_value(data, entry_offset + 8, tag_type, count, is_little_endian) {
            tags.insert(tag_id, value);
        }
    }

    Ok(tags)
}

/// Parse a single tag value
fn parse_tag_value(
    data: &[u8],
    value_offset: usize,
    tag_type: u16,
    count: usize,
    is_little_endian: bool,
) -> Option<ExifValue> {
    let type_size = match tag_type {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => return None,
    };

    let total_size = count * type_size;
    let actual_offset = if total_size <= 4 {
        value_offset
    } else {
        read_u32(data, value_offset, is_little_endian) as usize
    };

    if actual_offset + total_size > data.len() {
        return None;
    }

    match tag_type {
        1 => Some(ExifValue::Byte(data[actual_offset..actual_offset + count].to_vec())),
        2 => {
            let s = data[actual_offset..actual_offset + count]
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            Some(ExifValue::Ascii(s))
        }
        3 => {
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                values.push(read_u16(data, actual_offset + i * 2, is_little_endian));
            }
            Some(ExifValue::Short(values))
        }
        4 => {
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                values.push(read_u32(data, actual_offset + i * 4, is_little_endian));
            }
            Some(ExifValue::Long(values))
        }
        5 => {
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                let num = read_u32(data, actual_offset + i * 8, is_little_endian);
                let den = read_u32(data, actual_offset + i * 8 + 4, is_little_endian);
                values.push((num, den));
            }
            Some(ExifValue::Rational(values))
        }
        7 => Some(ExifValue::Undefined(data[actual_offset..actual_offset + count].to_vec())),
        _ => None,
    }
}

/// Parse GPS coordinate from EXIF rational values
fn parse_gps_coord(rationals: &[(u32, u32)], reference: &str) -> f64 {
    if rationals.len() < 3 {
        return 0.0;
    }

    let degrees = rationals[0].0 as f64 / rationals[0].1.max(1) as f64;
    let minutes = rationals[1].0 as f64 / rationals[1].1.max(1) as f64;
    let seconds = rationals[2].0 as f64 / rationals[2].1.max(1) as f64;

    let mut coord = degrees + minutes / 60.0 + seconds / 3600.0;

    if reference == "S" || reference == "W" {
        coord = -coord;
    }

    coord
}

/// Read u16 with specified byte order
fn read_u16(data: &[u8], offset: usize, little_endian: bool) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    if little_endian {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    }
}

/// Read u32 with specified byte order
fn read_u32(data: &[u8], offset: usize, little_endian: bool) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    if little_endian {
        u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
    } else {
        u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
    }
}

/// Parse XMP data from raw bytes
pub fn parse_xmp(data: &[u8]) -> Result<XmpData> {
    let raw = String::from_utf8_lossy(data).to_string();
    let mut xmp = XmpData {
        raw: raw.clone(),
        properties: HashMap::new(),
    };

    // Simple XMP parsing - extract common properties
    // Full XMP parsing would require an XML parser

    // Extract dc:title
    if let Some(title) = extract_xmp_value(&raw, "dc:title") {
        xmp.properties.insert("title".into(), title);
    }

    // Extract dc:description
    if let Some(desc) = extract_xmp_value(&raw, "dc:description") {
        xmp.properties.insert("description".into(), desc);
    }

    // Extract dc:creator
    if let Some(creator) = extract_xmp_value(&raw, "dc:creator") {
        xmp.properties.insert("creator".into(), creator);
    }

    // Extract xmp:CreateDate
    if let Some(date) = extract_xmp_value(&raw, "xmp:CreateDate") {
        xmp.properties.insert("create_date".into(), date);
    }

    // Extract xmp:ModifyDate
    if let Some(date) = extract_xmp_value(&raw, "xmp:ModifyDate") {
        xmp.properties.insert("modify_date".into(), date);
    }

    // Extract photoshop:Credit
    if let Some(credit) = extract_xmp_value(&raw, "photoshop:Credit") {
        xmp.properties.insert("credit".into(), credit);
    }

    Ok(xmp)
}

/// Extract a simple XMP value (very basic parsing)
fn extract_xmp_value(xmp: &str, property: &str) -> Option<String> {
    // Try attribute format: property="value"
    let attr_pattern = format!("{}=\"", property);
    if let Some(start) = xmp.find(&attr_pattern) {
        let value_start = start + attr_pattern.len();
        if let Some(end) = xmp[value_start..].find('"') {
            return Some(xmp[value_start..value_start + end].to_string());
        }
    }

    // Try element format: <property>value</property>
    let open_tag = format!("<{}>", property);
    let close_tag = format!("</{}>", property);
    if let Some(start) = xmp.find(&open_tag) {
        let value_start = start + open_tag.len();
        if let Some(end) = xmp[value_start..].find(&close_tag) {
            let value = xmp[value_start..value_start + end].trim();
            // Handle rdf:li elements
            if value.contains("<rdf:li") {
                if let Some(li_start) = value.find('>') {
                    if let Some(li_end) = value[li_start + 1..].find("</rdf:li>") {
                        return Some(value[li_start + 1..li_start + 1 + li_end].to_string());
                    }
                }
            }
            return Some(value.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tiff_header_little_endian() {
        let data = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        let result = parse_tiff_header(&data);
        assert!(result.is_ok());
        let (is_le, offset) = result.unwrap();
        assert!(is_le);
        assert_eq!(offset, 8);
    }

    #[test]
    fn test_parse_tiff_header_big_endian() {
        let data = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        let result = parse_tiff_header(&data);
        assert!(result.is_ok());
        let (is_le, offset) = result.unwrap();
        assert!(!is_le);
        assert_eq!(offset, 8);
    }

    #[test]
    fn test_parse_gps_coord() {
        let rationals = vec![(37, 1), (46, 1), (2958, 100)];
        let coord = parse_gps_coord(&rationals, "N");
        assert!((coord - 37.7749).abs() < 0.01);

        let coord = parse_gps_coord(&rationals, "S");
        assert!((coord + 37.7749).abs() < 0.01);
    }

    #[test]
    fn test_extract_xmp_value_attribute() {
        let xmp = r#"<x:xmpmeta xmp:CreateDate="2024-01-15T10:30:00">"#;
        let value = extract_xmp_value(xmp, "xmp:CreateDate");
        assert_eq!(value, Some("2024-01-15T10:30:00".into()));
    }

    #[test]
    fn test_extract_xmp_value_element() {
        let xmp = r#"<dc:title><rdf:Alt><rdf:li>Test Image</rdf:li></rdf:Alt></dc:title>"#;
        let value = extract_xmp_value(xmp, "dc:title");
        assert!(value.is_some());
    }

    #[test]
    fn test_read_u16() {
        let data = [0x01, 0x02];
        assert_eq!(read_u16(&data, 0, true), 0x0201);
        assert_eq!(read_u16(&data, 0, false), 0x0102);
    }

    #[test]
    fn test_read_u32() {
        let data = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(read_u32(&data, 0, true), 0x04030201);
        assert_eq!(read_u32(&data, 0, false), 0x01020304);
    }
}
