//! TIFF tag definitions

/// Standard TIFF tags
pub mod tag {
    // Basic tags
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC_INTERPRETATION: u16 = 262;

    // Data organization
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;

    // Resolution
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const RESOLUTION_UNIT: u16 = 296;

    // Planar configuration
    pub const PLANAR_CONFIGURATION: u16 = 284;

    // Palette
    pub const COLOR_MAP: u16 = 320;

    // Tile organization
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;

    // Extra samples
    pub const EXTRA_SAMPLES: u16 = 338;
    pub const SAMPLE_FORMAT: u16 = 339;

    // JPEG tables
    pub const JPEG_TABLES: u16 = 347;

    // YCbCr
    pub const YCBCR_COEFFICIENTS: u16 = 529;
    pub const YCBCR_SUB_SAMPLING: u16 = 530;
    pub const YCBCR_POSITIONING: u16 = 531;

    // Reference
    pub const REFERENCE_BLACK_WHITE: u16 = 532;

    // ICC Profile
    pub const ICC_PROFILE: u16 = 34675;

    // Metadata
    pub const DATE_TIME: u16 = 306;
    pub const ARTIST: u16 = 315;
    pub const SOFTWARE: u16 = 305;
    pub const COPYRIGHT: u16 = 33432;
    pub const IMAGE_DESCRIPTION: u16 = 270;

    // Orientation
    pub const ORIENTATION: u16 = 274;

    // Predictor (for compression)
    pub const PREDICTOR: u16 = 317;
}

/// TIFF data types
pub mod data_type {
    pub const BYTE: u16 = 1;
    pub const ASCII: u16 = 2;
    pub const SHORT: u16 = 3;
    pub const LONG: u16 = 4;
    pub const RATIONAL: u16 = 5;
    pub const SBYTE: u16 = 6;
    pub const UNDEFINED: u16 = 7;
    pub const SSHORT: u16 = 8;
    pub const SLONG: u16 = 9;
    pub const SRATIONAL: u16 = 10;
    pub const FLOAT: u16 = 11;
    pub const DOUBLE: u16 = 12;

    // BigTIFF types
    pub const LONG8: u16 = 16;
    pub const SLONG8: u16 = 17;
    pub const IFD8: u16 = 18;

    /// Get byte size of data type
    pub fn size(type_id: u16) -> usize {
        match type_id {
            BYTE | ASCII | SBYTE | UNDEFINED => 1,
            SHORT | SSHORT => 2,
            LONG | SLONG | FLOAT => 4,
            RATIONAL | SRATIONAL | DOUBLE | LONG8 | SLONG8 | IFD8 => 8,
            _ => 0,
        }
    }

    /// Get name of data type
    pub fn name(type_id: u16) -> &'static str {
        match type_id {
            BYTE => "BYTE",
            ASCII => "ASCII",
            SHORT => "SHORT",
            LONG => "LONG",
            RATIONAL => "RATIONAL",
            SBYTE => "SBYTE",
            UNDEFINED => "UNDEFINED",
            SSHORT => "SSHORT",
            SLONG => "SLONG",
            SRATIONAL => "SRATIONAL",
            FLOAT => "FLOAT",
            DOUBLE => "DOUBLE",
            LONG8 => "LONG8",
            SLONG8 => "SLONG8",
            IFD8 => "IFD8",
            _ => "UNKNOWN",
        }
    }
}

/// Get tag name
pub fn tag_name(tag_id: u16) -> &'static str {
    match tag_id {
        tag::IMAGE_WIDTH => "ImageWidth",
        tag::IMAGE_LENGTH => "ImageLength",
        tag::BITS_PER_SAMPLE => "BitsPerSample",
        tag::COMPRESSION => "Compression",
        tag::PHOTOMETRIC_INTERPRETATION => "PhotometricInterpretation",
        tag::STRIP_OFFSETS => "StripOffsets",
        tag::SAMPLES_PER_PIXEL => "SamplesPerPixel",
        tag::ROWS_PER_STRIP => "RowsPerStrip",
        tag::STRIP_BYTE_COUNTS => "StripByteCounts",
        tag::X_RESOLUTION => "XResolution",
        tag::Y_RESOLUTION => "YResolution",
        tag::RESOLUTION_UNIT => "ResolutionUnit",
        tag::PLANAR_CONFIGURATION => "PlanarConfiguration",
        tag::COLOR_MAP => "ColorMap",
        tag::TILE_WIDTH => "TileWidth",
        tag::TILE_LENGTH => "TileLength",
        tag::TILE_OFFSETS => "TileOffsets",
        tag::TILE_BYTE_COUNTS => "TileByteCounts",
        tag::EXTRA_SAMPLES => "ExtraSamples",
        tag::SAMPLE_FORMAT => "SampleFormat",
        tag::DATE_TIME => "DateTime",
        tag::ARTIST => "Artist",
        tag::SOFTWARE => "Software",
        tag::COPYRIGHT => "Copyright",
        tag::IMAGE_DESCRIPTION => "ImageDescription",
        tag::ORIENTATION => "Orientation",
        tag::PREDICTOR => "Predictor",
        tag::ICC_PROFILE => "ICCProfile",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_size() {
        assert_eq!(data_type::size(data_type::BYTE), 1);
        assert_eq!(data_type::size(data_type::SHORT), 2);
        assert_eq!(data_type::size(data_type::LONG), 4);
        assert_eq!(data_type::size(data_type::RATIONAL), 8);
    }

    #[test]
    fn test_tag_name() {
        assert_eq!(tag_name(tag::IMAGE_WIDTH), "ImageWidth");
        assert_eq!(tag_name(tag::COMPRESSION), "Compression");
    }
}
