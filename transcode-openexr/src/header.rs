//! OpenEXR header structure

use crate::channel::{ChannelList, PixelType};
use crate::compression::Compression;
use crate::error::{ExrError, Result};
use crate::types::{
    Box2i, Chromaticities, DataWindow, DisplayWindow, LineOrder, Rational, TileDescription,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Box2i (bounding box)
    Box2i(Box2i),
    /// Channel list
    ChannelList(ChannelList),
    /// Chromaticities
    Chromaticities(Chromaticities),
    /// Compression
    Compression(Compression),
    /// Double (f64)
    Double(f64),
    /// Environment map type
    Envmap(u8),
    /// Float (f32)
    Float(f32),
    /// Int (i32)
    Int(i32),
    /// KeyCode
    KeyCode([i32; 7]),
    /// Line order
    LineOrder(LineOrder),
    /// M33f (3x3 matrix)
    M33f([[f32; 3]; 3]),
    /// M44f (4x4 matrix)
    M44f([[f32; 4]; 4]),
    /// Preview image
    Preview { width: u32, height: u32, data: Vec<u8> },
    /// Rational number
    Rational(Rational),
    /// String
    String(String),
    /// StringVector
    StringVector(Vec<String>),
    /// Tile description
    TileDescription(TileDescription),
    /// TimeCode
    TimeCode(u32, u32),
    /// V2i
    V2i(i32, i32),
    /// V2f
    V2f(f32, f32),
    /// V3i
    V3i(i32, i32, i32),
    /// V3f
    V3f(f32, f32, f32),
    /// Raw bytes (unknown type)
    Raw(Vec<u8>),
}

/// OpenEXR header
#[derive(Debug, Clone)]
pub struct Header {
    /// Attributes
    attributes: HashMap<String, AttributeValue>,
    /// Channel list (cached for convenience)
    channels: ChannelList,
    /// Data window (cached for convenience)
    data_window: Box2i,
    /// Display window (cached for convenience)
    display_window: Box2i,
    /// Compression (cached for convenience)
    compression: Compression,
    /// Line order (cached for convenience)
    line_order: LineOrder,
    /// Pixel aspect ratio (cached)
    pixel_aspect_ratio: f32,
}

impl Default for Header {
    fn default() -> Self {
        Header::new(1920, 1080)
    }
}

impl Header {
    /// Create new header with given dimensions
    pub fn new(width: i32, height: i32) -> Self {
        let data_window = Box2i::from_dimensions(width, height);
        let display_window = data_window;

        let mut header = Header {
            attributes: HashMap::new(),
            channels: ChannelList::rgba(PixelType::Half),
            data_window,
            display_window,
            compression: Compression::Zip,
            line_order: LineOrder::IncreasingY,
            pixel_aspect_ratio: 1.0,
        };

        // Store required attributes
        header.set_attribute("dataWindow", AttributeValue::Box2i(data_window));
        header.set_attribute("displayWindow", AttributeValue::Box2i(display_window));
        header.set_attribute("compression", AttributeValue::Compression(Compression::Zip));
        header.set_attribute("lineOrder", AttributeValue::LineOrder(LineOrder::IncreasingY));
        header.set_attribute("pixelAspectRatio", AttributeValue::Float(1.0));
        header.set_attribute(
            "screenWindowCenter",
            AttributeValue::V2f(0.0, 0.0),
        );
        header.set_attribute("screenWindowWidth", AttributeValue::Float(1.0));

        header
    }

    /// Set channels
    pub fn set_channels(&mut self, channels: ChannelList) {
        self.channels = channels;
    }

    /// Get channels
    pub fn channels(&self) -> &ChannelList {
        &self.channels
    }

    /// Set data window
    pub fn set_data_window(&mut self, window: DataWindow) {
        self.data_window = window;
        self.set_attribute("dataWindow", AttributeValue::Box2i(window));
    }

    /// Get data window
    pub fn data_window(&self) -> &DataWindow {
        &self.data_window
    }

    /// Set display window
    pub fn set_display_window(&mut self, window: DisplayWindow) {
        self.display_window = window;
        self.set_attribute("displayWindow", AttributeValue::Box2i(window));
    }

    /// Get display window
    pub fn display_window(&self) -> &DisplayWindow {
        &self.display_window
    }

    /// Set compression
    pub fn set_compression(&mut self, compression: Compression) {
        self.compression = compression;
        self.set_attribute("compression", AttributeValue::Compression(compression));
    }

    /// Get compression
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Set line order
    pub fn set_line_order(&mut self, line_order: LineOrder) {
        self.line_order = line_order;
        self.set_attribute("lineOrder", AttributeValue::LineOrder(line_order));
    }

    /// Get line order
    pub fn line_order(&self) -> LineOrder {
        self.line_order
    }

    /// Set pixel aspect ratio
    pub fn set_pixel_aspect_ratio(&mut self, ratio: f32) {
        self.pixel_aspect_ratio = ratio;
        self.set_attribute("pixelAspectRatio", AttributeValue::Float(ratio));
    }

    /// Get pixel aspect ratio
    pub fn pixel_aspect_ratio(&self) -> f32 {
        self.pixel_aspect_ratio
    }

    /// Image width
    pub fn width(&self) -> i32 {
        self.data_window.width()
    }

    /// Image height
    pub fn height(&self) -> i32 {
        self.data_window.height()
    }

    /// Set attribute
    pub fn set_attribute(&mut self, name: &str, value: AttributeValue) {
        self.attributes.insert(name.to_string(), value);
    }

    /// Get attribute
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    /// Set chromaticities
    pub fn set_chromaticities(&mut self, chroma: Chromaticities) {
        self.set_attribute("chromaticities", AttributeValue::Chromaticities(chroma));
    }

    /// Get chromaticities
    pub fn chromaticities(&self) -> Option<Chromaticities> {
        if let Some(AttributeValue::Chromaticities(c)) = self.get_attribute("chromaticities") {
            Some(*c)
        } else {
            None
        }
    }

    /// Set owner
    pub fn set_owner(&mut self, owner: &str) {
        self.set_attribute("owner", AttributeValue::String(owner.to_string()));
    }

    /// Set comments
    pub fn set_comments(&mut self, comments: &str) {
        self.set_attribute("comments", AttributeValue::String(comments.to_string()));
    }

    /// Set capture date
    pub fn set_capture_date(&mut self, date: &str) {
        self.set_attribute("capDate", AttributeValue::String(date.to_string()));
    }

    /// Set frames per second
    pub fn set_frames_per_second(&mut self, fps: Rational) {
        self.set_attribute("framesPerSecond", AttributeValue::Rational(fps));
    }

    /// Write header to output
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write magic number and version
        writer.write_u32::<LittleEndian>(crate::EXR_MAGIC)?;
        writer.write_u32::<LittleEndian>(crate::EXR_VERSION)?;

        // Write channels
        self.write_attribute(writer, "channels", &self.channels)?;

        // Write all other attributes
        for (name, value) in &self.attributes {
            self.write_attribute_value(writer, name, value)?;
        }

        // End of header
        writer.write_u8(0)?;

        Ok(())
    }

    /// Write channel list as attribute
    fn write_attribute<W: Write>(&self, writer: &mut W, name: &str, channels: &ChannelList) -> Result<()> {
        // Attribute name (null-terminated)
        writer.write_all(name.as_bytes())?;
        writer.write_u8(0)?;

        // Type name (null-terminated)
        writer.write_all(b"chlist")?;
        writer.write_u8(0)?;

        // Build channel list data
        let mut channel_data = Vec::new();
        for channel in channels.iter() {
            // Channel name (null-terminated)
            channel_data.extend_from_slice(channel.name.as_bytes());
            channel_data.push(0);

            // Pixel type (i32)
            channel_data.write_i32::<LittleEndian>(channel.pixel_type.to_u32() as i32)?;

            // pLinear (u8)
            channel_data.push(if channel.p_linear { 1 } else { 0 });

            // Reserved (3 bytes)
            channel_data.extend_from_slice(&[0, 0, 0]);

            // xSampling, ySampling (i32)
            channel_data.write_i32::<LittleEndian>(channel.x_sampling)?;
            channel_data.write_i32::<LittleEndian>(channel.y_sampling)?;
        }
        // End of channel list
        channel_data.push(0);

        // Size
        writer.write_i32::<LittleEndian>(channel_data.len() as i32)?;

        // Data
        writer.write_all(&channel_data)?;

        Ok(())
    }

    /// Write single attribute
    fn write_attribute_value<W: Write>(
        &self,
        writer: &mut W,
        name: &str,
        value: &AttributeValue,
    ) -> Result<()> {
        // Attribute name
        writer.write_all(name.as_bytes())?;
        writer.write_u8(0)?;

        match value {
            AttributeValue::Box2i(b) => {
                writer.write_all(b"box2i\0")?;
                writer.write_i32::<LittleEndian>(16)?;
                writer.write_i32::<LittleEndian>(b.min.x)?;
                writer.write_i32::<LittleEndian>(b.min.y)?;
                writer.write_i32::<LittleEndian>(b.max.x)?;
                writer.write_i32::<LittleEndian>(b.max.y)?;
            }
            AttributeValue::Compression(c) => {
                writer.write_all(b"compression\0")?;
                writer.write_i32::<LittleEndian>(1)?;
                writer.write_u8(c.to_u8())?;
            }
            AttributeValue::Float(f) => {
                writer.write_all(b"float\0")?;
                writer.write_i32::<LittleEndian>(4)?;
                writer.write_f32::<LittleEndian>(*f)?;
            }
            AttributeValue::Int(i) => {
                writer.write_all(b"int\0")?;
                writer.write_i32::<LittleEndian>(4)?;
                writer.write_i32::<LittleEndian>(*i)?;
            }
            AttributeValue::LineOrder(lo) => {
                writer.write_all(b"lineOrder\0")?;
                writer.write_i32::<LittleEndian>(1)?;
                writer.write_u8(lo.to_u8())?;
            }
            AttributeValue::String(s) => {
                writer.write_all(b"string\0")?;
                writer.write_i32::<LittleEndian>(s.len() as i32)?;
                writer.write_all(s.as_bytes())?;
            }
            AttributeValue::V2f(x, y) => {
                writer.write_all(b"v2f\0")?;
                writer.write_i32::<LittleEndian>(8)?;
                writer.write_f32::<LittleEndian>(*x)?;
                writer.write_f32::<LittleEndian>(*y)?;
            }
            AttributeValue::V2i(x, y) => {
                writer.write_all(b"v2i\0")?;
                writer.write_i32::<LittleEndian>(8)?;
                writer.write_i32::<LittleEndian>(*x)?;
                writer.write_i32::<LittleEndian>(*y)?;
            }
            AttributeValue::Rational(r) => {
                writer.write_all(b"rational\0")?;
                writer.write_i32::<LittleEndian>(8)?;
                writer.write_i32::<LittleEndian>(r.numerator)?;
                writer.write_u32::<LittleEndian>(r.denominator)?;
            }
            AttributeValue::Chromaticities(c) => {
                writer.write_all(b"chromaticities\0")?;
                writer.write_i32::<LittleEndian>(32)?;
                writer.write_f32::<LittleEndian>(c.red.x)?;
                writer.write_f32::<LittleEndian>(c.red.y)?;
                writer.write_f32::<LittleEndian>(c.green.x)?;
                writer.write_f32::<LittleEndian>(c.green.y)?;
                writer.write_f32::<LittleEndian>(c.blue.x)?;
                writer.write_f32::<LittleEndian>(c.blue.y)?;
                writer.write_f32::<LittleEndian>(c.white.x)?;
                writer.write_f32::<LittleEndian>(c.white.y)?;
            }
            _ => {
                // Skip unknown types
            }
        }

        Ok(())
    }

    /// Parse header from data
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut cursor = Cursor::new(data);

        // Read magic number
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != crate::EXR_MAGIC {
            return Err(ExrError::InvalidMagic);
        }

        // Read version
        let version = cursor.read_u32::<LittleEndian>()?;
        let _version_num = version & 0xFF;
        // Ignore version flags for now

        let mut header = Header {
            attributes: HashMap::new(),
            channels: ChannelList::new(),
            data_window: Box2i::default(),
            display_window: Box2i::default(),
            compression: Compression::None,
            line_order: LineOrder::IncreasingY,
            pixel_aspect_ratio: 1.0,
        };

        // Read attributes
        loop {
            let name = Self::read_string(&mut cursor)?;
            if name.is_empty() {
                break;
            }

            let type_name = Self::read_string(&mut cursor)?;
            let size = cursor.read_i32::<LittleEndian>()? as usize;

            let pos_before = cursor.position() as usize;

            match type_name.as_str() {
                "box2i" => {
                    let min_x = cursor.read_i32::<LittleEndian>()?;
                    let min_y = cursor.read_i32::<LittleEndian>()?;
                    let max_x = cursor.read_i32::<LittleEndian>()?;
                    let max_y = cursor.read_i32::<LittleEndian>()?;
                    let b = Box2i::new(
                        crate::types::V2i::new(min_x, min_y),
                        crate::types::V2i::new(max_x, max_y),
                    );

                    if name == "dataWindow" {
                        header.data_window = b;
                    } else if name == "displayWindow" {
                        header.display_window = b;
                    }
                    header.attributes.insert(name, AttributeValue::Box2i(b));
                }
                "chlist" => {
                    let channels = Self::read_channel_list(&mut cursor, size)?;
                    header.channels = channels;
                }
                "compression" => {
                    let c = cursor.read_u8()?;
                    let compression = Compression::from_u8(c)
                        .ok_or(ExrError::UnsupportedCompression(c))?;
                    header.compression = compression;
                    header.attributes.insert(name, AttributeValue::Compression(compression));
                }
                "lineOrder" => {
                    let lo = cursor.read_u8()?;
                    let line_order = LineOrder::from_u8(lo)
                        .ok_or(ExrError::InvalidHeader("Invalid line order".into()))?;
                    header.line_order = line_order;
                    header.attributes.insert(name, AttributeValue::LineOrder(line_order));
                }
                "float" => {
                    let f = cursor.read_f32::<LittleEndian>()?;
                    if name == "pixelAspectRatio" {
                        header.pixel_aspect_ratio = f;
                    }
                    header.attributes.insert(name, AttributeValue::Float(f));
                }
                "int" => {
                    let i = cursor.read_i32::<LittleEndian>()?;
                    header.attributes.insert(name, AttributeValue::Int(i));
                }
                "string" => {
                    let mut s = vec![0u8; size];
                    cursor.read_exact(&mut s)?;
                    let s = String::from_utf8_lossy(&s).to_string();
                    header.attributes.insert(name, AttributeValue::String(s));
                }
                "v2f" => {
                    let x = cursor.read_f32::<LittleEndian>()?;
                    let y = cursor.read_f32::<LittleEndian>()?;
                    header.attributes.insert(name, AttributeValue::V2f(x, y));
                }
                "v2i" => {
                    let x = cursor.read_i32::<LittleEndian>()?;
                    let y = cursor.read_i32::<LittleEndian>()?;
                    header.attributes.insert(name, AttributeValue::V2i(x, y));
                }
                _ => {
                    // Skip unknown attribute
                    let mut raw = vec![0u8; size];
                    cursor.read_exact(&mut raw)?;
                    header.attributes.insert(name, AttributeValue::Raw(raw));
                }
            }

            // Ensure we read exactly 'size' bytes
            let pos_after = cursor.position() as usize;
            if pos_after - pos_before < size {
                cursor.set_position((pos_before + size) as u64);
            }
        }

        Ok((header, cursor.position() as usize))
    }

    /// Read null-terminated string
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let mut bytes = Vec::new();
        loop {
            let b = cursor.read_u8()?;
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Read channel list
    fn read_channel_list(cursor: &mut Cursor<&[u8]>, _size: usize) -> Result<ChannelList> {
        let mut channels = ChannelList::new();

        loop {
            let name = Self::read_string(cursor)?;
            if name.is_empty() {
                break;
            }

            let pixel_type_raw = cursor.read_i32::<LittleEndian>()?;
            let pixel_type = PixelType::from_u32(pixel_type_raw as u32)
                .ok_or(ExrError::InvalidPixelType(pixel_type_raw as u32))?;

            let p_linear = cursor.read_u8()? != 0;

            // Reserved (3 bytes)
            let mut _reserved = [0u8; 3];
            cursor.read_exact(&mut _reserved)?;

            let x_sampling = cursor.read_i32::<LittleEndian>()?;
            let y_sampling = cursor.read_i32::<LittleEndian>()?;

            let mut channel = crate::channel::Channel::new(&name, pixel_type);
            channel.x_sampling = x_sampling;
            channel.y_sampling = y_sampling;
            channel.p_linear = p_linear;

            channels.add(channel);
        }

        Ok(channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_creation() {
        let header = Header::new(1920, 1080);
        assert_eq!(header.width(), 1920);
        assert_eq!(header.height(), 1080);
        assert_eq!(header.compression(), Compression::Zip);
    }

    #[test]
    fn test_header_attributes() {
        let mut header = Header::new(640, 480);
        header.set_owner("Test");
        header.set_comments("Test comment");

        assert!(matches!(
            header.get_attribute("owner"),
            Some(AttributeValue::String(_))
        ));
    }

    #[test]
    fn test_header_write_parse() {
        let mut header = Header::new(100, 100);
        header.set_compression(Compression::Rle);

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        let (parsed, _) = Header::parse(&buffer).unwrap();
        assert_eq!(parsed.width(), 100);
        assert_eq!(parsed.height(), 100);
        assert_eq!(parsed.compression(), Compression::Rle);
    }

    #[test]
    fn test_chromaticities() {
        let mut header = Header::new(100, 100);
        header.set_chromaticities(Chromaticities::default());

        let chroma = header.chromaticities();
        assert!(chroma.is_some());
    }

    #[test]
    fn test_frame_rate() {
        let mut header = Header::new(100, 100);
        header.set_frames_per_second(Rational::new(24000, 1001));

        if let Some(AttributeValue::Rational(r)) = header.get_attribute("framesPerSecond") {
            assert!((r.to_f64() - 23.976).abs() < 0.01);
        }
    }
}
