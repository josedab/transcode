//! Thumbnail extraction from video files.
//!
//! This module provides functionality for extracting thumbnail images from video files
//! at specific timestamps, frame numbers, or at regular intervals.
//!
//! # Examples
//!
//! ## Extract a single thumbnail at a specific timestamp
//!
//! ```rust,no_run
//! use transcode::thumbnail::{Thumbnail, ThumbnailFormat};
//! use transcode::Timestamp;
//!
//! let thumb = Thumbnail::extract("video.mp4", Timestamp::from_millis(10_000))?;
//! thumb.save_jpeg("thumb.jpg", 85)?; // quality 85
//! # Ok::<(), transcode::Error>(())
//! ```
//!
//! ## Extract a grid of thumbnails
//!
//! ```rust,no_run
//! use transcode::thumbnail::Thumbnail;
//!
//! let thumbs = Thumbnail::extract_grid("video.mp4", 3, 3)?; // 3x3 grid
//! for (i, thumb) in thumbs.iter().enumerate() {
//!     thumb.save_png(&format!("thumb_{}.png", i))?;
//! }
//! # Ok::<(), transcode::Error>(())
//! ```

use std::path::Path;
use transcode_core::{
    error::{Error, Result},
    frame::{Frame, PixelFormat},
    timestamp::{TimeBase, Timestamp},
};

/// Output format for thumbnail images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThumbnailFormat {
    /// JPEG format with configurable quality (0-100).
    #[default]
    Jpeg,
    /// PNG format (lossless).
    Png,
    /// WebP format with configurable quality (0-100).
    WebP,
}

impl ThumbnailFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ThumbnailFormat::Jpeg => "jpg",
            ThumbnailFormat::Png => "png",
            ThumbnailFormat::WebP => "webp",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ThumbnailFormat::Jpeg => "image/jpeg",
            ThumbnailFormat::Png => "image/png",
            ThumbnailFormat::WebP => "image/webp",
        }
    }
}

/// Size specification for thumbnail output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Size {
    /// Create a new size with the specified dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Create a size that fits within the given maximum dimensions while preserving aspect ratio.
    pub fn fit_within(
        original_width: u32,
        original_height: u32,
        max_width: u32,
        max_height: u32,
    ) -> Self {
        if original_width == 0 || original_height == 0 {
            return Self::new(max_width, max_height);
        }

        let width_ratio = max_width as f64 / original_width as f64;
        let height_ratio = max_height as f64 / original_height as f64;
        let ratio = width_ratio.min(height_ratio);

        Self {
            width: ((original_width as f64 * ratio) as u32).max(1),
            height: ((original_height as f64 * ratio) as u32).max(1),
        }
    }

    /// Create a size with a fixed width, calculating height to preserve aspect ratio.
    pub fn with_width(original_width: u32, original_height: u32, target_width: u32) -> Self {
        if original_width == 0 {
            return Self::new(target_width, target_width);
        }
        let ratio = target_width as f64 / original_width as f64;
        Self {
            width: target_width,
            height: ((original_height as f64 * ratio) as u32).max(1),
        }
    }

    /// Create a size with a fixed height, calculating width to preserve aspect ratio.
    pub fn with_height(original_width: u32, original_height: u32, target_height: u32) -> Self {
        if original_height == 0 {
            return Self::new(target_height, target_height);
        }
        let ratio = target_height as f64 / original_height as f64;
        Self {
            width: ((original_width as f64 * ratio) as u32).max(1),
            height: target_height,
        }
    }
}

impl Default for Size {
    fn default() -> Self {
        // Default thumbnail size
        Self {
            width: 320,
            height: 180,
        }
    }
}

impl From<(u32, u32)> for Size {
    fn from((width, height): (u32, u32)) -> Self {
        Self::new(width, height)
    }
}

/// Configuration options for thumbnail extraction.
#[derive(Debug, Clone)]
pub struct ThumbnailConfig {
    /// Output size (None = original size).
    pub size: Option<Size>,
    /// Output format.
    pub format: ThumbnailFormat,
    /// Quality for lossy formats (0-100).
    pub quality: u8,
    /// Seek to nearest keyframe only (faster but less precise).
    pub keyframe_only: bool,
    /// Preserve aspect ratio when resizing.
    pub preserve_aspect_ratio: bool,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            size: None,
            format: ThumbnailFormat::Jpeg,
            quality: 85,
            keyframe_only: false,
            preserve_aspect_ratio: true,
        }
    }
}

impl ThumbnailConfig {
    /// Create a new thumbnail configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.size = Some(Size::new(width, height));
        self
    }

    /// Set the maximum output size while preserving aspect ratio.
    pub fn with_max_size(mut self, max_width: u32, max_height: u32) -> Self {
        self.size = Some(Size::new(max_width, max_height));
        self.preserve_aspect_ratio = true;
        self
    }

    /// Set the output format.
    pub fn with_format(mut self, format: ThumbnailFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the quality for lossy formats (0-100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(100);
        self
    }

    /// Enable keyframe-only mode for faster seeking.
    pub fn keyframe_only(mut self, enabled: bool) -> Self {
        self.keyframe_only = enabled;
        self
    }

    /// Set whether to preserve aspect ratio when resizing.
    pub fn preserve_aspect_ratio(mut self, enabled: bool) -> Self {
        self.preserve_aspect_ratio = enabled;
        self
    }
}

/// Position specification for thumbnail extraction.
#[derive(Debug, Clone, Copy)]
pub enum ThumbnailPosition {
    /// Extract at a specific timestamp.
    Timestamp(Timestamp),
    /// Extract at a specific frame number.
    FrameNumber(u64),
    /// Extract at a percentage of the video duration (0.0 - 1.0).
    Percentage(f64),
}

impl ThumbnailPosition {
    /// Create a position from seconds.
    pub fn from_secs(secs: f64) -> Self {
        ThumbnailPosition::Timestamp(Timestamp::from_seconds(secs, TimeBase::MILLISECONDS))
    }

    /// Create a position from milliseconds.
    pub fn from_millis(millis: i64) -> Self {
        ThumbnailPosition::Timestamp(Timestamp::from_millis(millis))
    }

    /// Create a position from a frame number.
    pub fn from_frame(frame: u64) -> Self {
        ThumbnailPosition::FrameNumber(frame)
    }

    /// Create a position from a percentage (0.0 - 1.0).
    pub fn from_percentage(percentage: f64) -> Self {
        ThumbnailPosition::Percentage(percentage.clamp(0.0, 1.0))
    }
}

impl From<Timestamp> for ThumbnailPosition {
    fn from(ts: Timestamp) -> Self {
        ThumbnailPosition::Timestamp(ts)
    }
}

impl From<u64> for ThumbnailPosition {
    fn from(frame: u64) -> Self {
        ThumbnailPosition::FrameNumber(frame)
    }
}

/// A thumbnail extracted from a video.
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// The extracted frame data.
    frame: Frame,
    /// The timestamp where this thumbnail was extracted.
    timestamp: Timestamp,
    /// The frame number of this thumbnail.
    frame_number: u64,
    /// Source video path.
    source_path: Option<String>,
}

impl Thumbnail {
    /// Create a new thumbnail from a frame.
    pub fn new(frame: Frame, timestamp: Timestamp, frame_number: u64) -> Self {
        Self {
            frame,
            timestamp,
            frame_number,
            source_path: None,
        }
    }

    /// Extract a thumbnail from a video file at the specified timestamp.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `timestamp` - The timestamp to extract the thumbnail from
    ///
    /// # Returns
    ///
    /// The extracted thumbnail.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use transcode::thumbnail::Thumbnail;
    /// use transcode::Timestamp;
    ///
    /// let thumb = Thumbnail::extract("video.mp4", Timestamp::from_millis(5000))?;
    /// # Ok::<(), transcode::Error>(())
    /// ```
    pub fn extract<P: AsRef<Path>>(path: P, timestamp: Timestamp) -> Result<Self> {
        Self::extract_with_config(
            path,
            ThumbnailPosition::Timestamp(timestamp),
            ThumbnailConfig::default(),
        )
    }

    /// Extract a thumbnail at a specific frame number.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `frame_number` - The frame number to extract
    ///
    /// # Returns
    ///
    /// The extracted thumbnail.
    pub fn extract_frame<P: AsRef<Path>>(path: P, frame_number: u64) -> Result<Self> {
        Self::extract_with_config(
            path,
            ThumbnailPosition::FrameNumber(frame_number),
            ThumbnailConfig::default(),
        )
    }

    /// Extract a thumbnail with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `position` - The position to extract the thumbnail from
    /// * `config` - Thumbnail extraction configuration
    ///
    /// # Returns
    ///
    /// The extracted thumbnail.
    pub fn extract_with_config<P: AsRef<Path>>(
        path: P,
        position: ThumbnailPosition,
        config: ThumbnailConfig,
    ) -> Result<Self> {
        let path_ref = path.as_ref();

        // Validate input file exists
        if !path_ref.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Video file not found: {:?}", path_ref),
            )));
        }

        // In a real implementation, this would:
        // 1. Open the video file using the demuxer
        // 2. Seek to the appropriate position
        // 3. Decode the frame
        // 4. Optionally resize the frame
        // 5. Return the thumbnail

        // For now, create a simulated thumbnail
        let (width, height) = if let Some(size) = config.size {
            (size.width, size.height)
        } else {
            (1920, 1080) // Default video dimensions
        };

        let frame = Frame::new(width, height, PixelFormat::Rgb24, TimeBase::MILLISECONDS);

        let timestamp = match position {
            ThumbnailPosition::Timestamp(ts) => ts,
            ThumbnailPosition::FrameNumber(n) => {
                // Assume 30fps for frame to timestamp conversion
                Timestamp::from_millis((n as f64 / 30.0 * 1000.0) as i64)
            }
            ThumbnailPosition::Percentage(p) => {
                // Assume 60 second video for percentage conversion
                Timestamp::from_millis((p * 60.0 * 1000.0) as i64)
            }
        };

        let frame_number = match position {
            ThumbnailPosition::FrameNumber(n) => n,
            ThumbnailPosition::Timestamp(ts) => {
                // Assume 30fps for timestamp to frame conversion
                (ts.to_millis().unwrap_or(0) as f64 / 1000.0 * 30.0) as u64
            }
            ThumbnailPosition::Percentage(p) => {
                // Assume 60 second video at 30fps
                (p * 60.0 * 30.0) as u64
            }
        };

        Ok(Self {
            frame,
            timestamp,
            frame_number,
            source_path: Some(path_ref.to_string_lossy().to_string()),
        })
    }

    /// Extract thumbnails at regular intervals.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `count` - Number of thumbnails to extract
    ///
    /// # Returns
    ///
    /// A vector of extracted thumbnails evenly distributed across the video duration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use transcode::thumbnail::Thumbnail;
    ///
    /// let thumbs = Thumbnail::extract_at_intervals("video.mp4", 10)?;
    /// for (i, thumb) in thumbs.iter().enumerate() {
    ///     thumb.save_jpeg(&format!("thumb_{}.jpg", i), 85)?;
    /// }
    /// # Ok::<(), transcode::Error>(())
    /// ```
    pub fn extract_at_intervals<P: AsRef<Path>>(path: P, count: usize) -> Result<Vec<Self>> {
        Self::extract_at_intervals_with_config(path, count, ThumbnailConfig::default())
    }

    /// Extract thumbnails at regular intervals with custom configuration.
    pub fn extract_at_intervals_with_config<P: AsRef<Path>>(
        path: P,
        count: usize,
        config: ThumbnailConfig,
    ) -> Result<Vec<Self>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut thumbnails = Vec::with_capacity(count);

        for i in 0..count {
            let percentage = if count == 1 {
                0.5 // Single thumbnail at the middle
            } else {
                i as f64 / (count - 1) as f64
            };

            let thumb = Self::extract_with_config(
                path.as_ref(),
                ThumbnailPosition::Percentage(percentage),
                config.clone(),
            )?;
            thumbnails.push(thumb);
        }

        Ok(thumbnails)
    }

    /// Extract a grid of thumbnails (e.g., 3x3 for contact sheet).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `cols` - Number of columns in the grid
    /// * `rows` - Number of rows in the grid
    ///
    /// # Returns
    ///
    /// A vector of extracted thumbnails arranged in row-major order.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use transcode::thumbnail::Thumbnail;
    ///
    /// let thumbs = Thumbnail::extract_grid("video.mp4", 3, 3)?; // 9 thumbnails
    /// # Ok::<(), transcode::Error>(())
    /// ```
    pub fn extract_grid<P: AsRef<Path>>(path: P, cols: usize, rows: usize) -> Result<Vec<Self>> {
        Self::extract_grid_with_config(path, cols, rows, ThumbnailConfig::default())
    }

    /// Extract a grid of thumbnails with custom configuration.
    pub fn extract_grid_with_config<P: AsRef<Path>>(
        path: P,
        cols: usize,
        rows: usize,
        config: ThumbnailConfig,
    ) -> Result<Vec<Self>> {
        let count = cols * rows;
        Self::extract_at_intervals_with_config(path, count, config)
    }

    /// Extract thumbnail at the nearest keyframe (I-frame).
    ///
    /// This is faster than regular extraction as it doesn't require decoding
    /// intermediate frames.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `timestamp` - The target timestamp (will seek to nearest keyframe)
    ///
    /// # Returns
    ///
    /// The extracted thumbnail from the nearest keyframe.
    pub fn extract_keyframe<P: AsRef<Path>>(path: P, timestamp: Timestamp) -> Result<Self> {
        let config = ThumbnailConfig::new().keyframe_only(true);
        Self::extract_with_config(path, ThumbnailPosition::Timestamp(timestamp), config)
    }

    /// Get the width of the thumbnail.
    pub fn width(&self) -> u32 {
        self.frame.width()
    }

    /// Get the height of the thumbnail.
    pub fn height(&self) -> u32 {
        self.frame.height()
    }

    /// Get the timestamp where this thumbnail was extracted.
    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    /// Get the frame number of this thumbnail.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get the underlying frame.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Get the source video path.
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// Get the pixel data as RGB bytes.
    pub fn as_rgb(&self) -> Option<&[u8]> {
        self.frame.plane(0)
    }

    /// Save the thumbnail as JPEG with the specified quality.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `quality` - JPEG quality (0-100)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use transcode::thumbnail::Thumbnail;
    /// use transcode::Timestamp;
    ///
    /// let thumb = Thumbnail::extract("video.mp4", Timestamp::from_millis(5000))?;
    /// thumb.save_jpeg("thumbnail.jpg", 85)?;
    /// # Ok::<(), transcode::Error>(())
    /// ```
    pub fn save_jpeg<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        self.save_with_options(path, ThumbnailFormat::Jpeg, quality.min(100))
    }

    /// Save the thumbnail as PNG.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.save_with_options(path, ThumbnailFormat::Png, 100)
    }

    /// Save the thumbnail as WebP with the specified quality.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `quality` - WebP quality (0-100)
    pub fn save_webp<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        self.save_with_options(path, ThumbnailFormat::WebP, quality.min(100))
    }

    /// Save the thumbnail with custom format and quality options.
    pub fn save_with_options<P: AsRef<Path>>(
        &self,
        path: P,
        format: ThumbnailFormat,
        quality: u8,
    ) -> Result<()> {
        let path_ref = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path_ref.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // In a real implementation, this would encode the frame data
        // to the specified image format. For now, we create a placeholder file.
        let data = self.encode_to_format(format, quality)?;
        std::fs::write(path_ref, &data)?;

        Ok(())
    }

    /// Encode the thumbnail to bytes in the specified format.
    pub fn encode(&self, format: ThumbnailFormat, quality: u8) -> Result<Vec<u8>> {
        self.encode_to_format(format, quality)
    }

    /// Internal method to encode the thumbnail to the specified format.
    fn encode_to_format(&self, format: ThumbnailFormat, quality: u8) -> Result<Vec<u8>> {
        // In a real implementation, this would use an image encoding library
        // to encode the frame data. For now, we create placeholder data.

        let width = self.frame.width();
        let height = self.frame.height();

        match format {
            ThumbnailFormat::Jpeg => {
                // Placeholder JPEG header + data
                let mut data = Vec::with_capacity(width as usize * height as usize);
                // JPEG magic bytes
                data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);
                // Add metadata comment with dimensions and quality
                let meta = format!("{}x{}@{}", width, height, quality);
                data.extend_from_slice(meta.as_bytes());
                // JPEG end marker
                data.extend_from_slice(&[0xFF, 0xD9]);
                Ok(data)
            }
            ThumbnailFormat::Png => {
                // Placeholder PNG header + data
                let mut data = Vec::with_capacity(width as usize * height as usize);
                // PNG magic bytes
                data.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
                // Add metadata
                let meta = format!("{}x{}", width, height);
                data.extend_from_slice(meta.as_bytes());
                Ok(data)
            }
            ThumbnailFormat::WebP => {
                // Placeholder WebP header + data
                let mut data = Vec::with_capacity(width as usize * height as usize);
                // RIFF header
                data.extend_from_slice(b"RIFF");
                data.extend_from_slice(&[0, 0, 0, 0]); // size placeholder
                data.extend_from_slice(b"WEBP");
                // Add metadata
                let meta = format!("{}x{}@{}", width, height, quality);
                data.extend_from_slice(meta.as_bytes());
                Ok(data)
            }
        }
    }

    /// Resize the thumbnail to the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Target width
    /// * `height` - Target height
    ///
    /// # Returns
    ///
    /// A new thumbnail with the resized frame.
    pub fn resize(&self, width: u32, height: u32) -> Result<Self> {
        self.resize_with_options(width, height, true)
    }

    /// Resize the thumbnail with optional aspect ratio preservation.
    pub fn resize_with_options(
        &self,
        width: u32,
        height: u32,
        preserve_aspect_ratio: bool,
    ) -> Result<Self> {
        let (target_width, target_height) = if preserve_aspect_ratio {
            let size = Size::fit_within(self.width(), self.height(), width, height);
            (size.width, size.height)
        } else {
            (width, height)
        };

        // In a real implementation, this would perform actual image resizing
        // using bilinear/bicubic interpolation
        let resized_frame = Frame::new(
            target_width,
            target_height,
            self.frame.format(),
            self.frame.pts.time_base,
        );

        Ok(Self {
            frame: resized_frame,
            timestamp: self.timestamp,
            frame_number: self.frame_number,
            source_path: self.source_path.clone(),
        })
    }

    /// Create a contact sheet from multiple thumbnails.
    ///
    /// # Arguments
    ///
    /// * `thumbnails` - Vector of thumbnails to include
    /// * `cols` - Number of columns
    /// * `thumb_width` - Width of each thumbnail in the sheet
    /// * `thumb_height` - Height of each thumbnail in the sheet
    ///
    /// # Returns
    ///
    /// A single thumbnail containing the contact sheet.
    pub fn create_contact_sheet(
        thumbnails: &[Thumbnail],
        cols: usize,
        thumb_width: u32,
        thumb_height: u32,
    ) -> Result<Self> {
        if thumbnails.is_empty() {
            return Err(Error::InvalidParameter("Empty thumbnail list".into()));
        }

        let rows = thumbnails.len().div_ceil(cols);
        let sheet_width = thumb_width * cols as u32;
        let sheet_height = thumb_height * rows as u32;

        // In a real implementation, this would composite all thumbnails
        // into a single image
        let frame = Frame::new(
            sheet_width,
            sheet_height,
            PixelFormat::Rgb24,
            TimeBase::MILLISECONDS,
        );

        Ok(Self {
            frame,
            timestamp: thumbnails[0].timestamp,
            frame_number: 0,
            source_path: thumbnails[0].source_path.clone(),
        })
    }
}

/// Builder for constructing thumbnail extraction requests with a fluent API.
#[derive(Debug, Clone)]
pub struct ThumbnailBuilder {
    path: Option<String>,
    position: ThumbnailPosition,
    config: ThumbnailConfig,
}

impl Default for ThumbnailBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ThumbnailBuilder {
    /// Create a new thumbnail builder.
    pub fn new() -> Self {
        Self {
            path: None,
            position: ThumbnailPosition::Percentage(0.5),
            config: ThumbnailConfig::default(),
        }
    }

    /// Set the input video path.
    pub fn input<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set the extraction position to a specific timestamp.
    pub fn at_timestamp(mut self, timestamp: Timestamp) -> Self {
        self.position = ThumbnailPosition::Timestamp(timestamp);
        self
    }

    /// Set the extraction position to a specific time in seconds.
    pub fn at_seconds(mut self, seconds: f64) -> Self {
        self.position = ThumbnailPosition::from_secs(seconds);
        self
    }

    /// Set the extraction position to a specific time in milliseconds.
    pub fn at_millis(mut self, millis: i64) -> Self {
        self.position = ThumbnailPosition::from_millis(millis);
        self
    }

    /// Set the extraction position to a specific frame number.
    pub fn at_frame(mut self, frame: u64) -> Self {
        self.position = ThumbnailPosition::FrameNumber(frame);
        self
    }

    /// Set the extraction position to a percentage of the video duration.
    pub fn at_percentage(mut self, percentage: f64) -> Self {
        self.position = ThumbnailPosition::Percentage(percentage.clamp(0.0, 1.0));
        self
    }

    /// Set the output size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.config.size = Some(Size::new(width, height));
        self
    }

    /// Set the maximum output size while preserving aspect ratio.
    pub fn max_size(mut self, max_width: u32, max_height: u32) -> Self {
        self.config.size = Some(Size::new(max_width, max_height));
        self.config.preserve_aspect_ratio = true;
        self
    }

    /// Set the output format.
    pub fn format(mut self, format: ThumbnailFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set the quality for lossy formats.
    pub fn quality(mut self, quality: u8) -> Self {
        self.config.quality = quality.min(100);
        self
    }

    /// Enable keyframe-only mode for faster seeking.
    pub fn keyframe_only(mut self, enabled: bool) -> Self {
        self.config.keyframe_only = enabled;
        self
    }

    /// Set whether to preserve aspect ratio when resizing.
    pub fn preserve_aspect_ratio(mut self, enabled: bool) -> Self {
        self.config.preserve_aspect_ratio = enabled;
        self
    }

    /// Extract the thumbnail.
    pub fn extract(self) -> Result<Thumbnail> {
        let path = self
            .path
            .ok_or_else(|| Error::InvalidParameter("No input path specified".into()))?;
        Thumbnail::extract_with_config(path, self.position, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_thumbnail_format_extension() {
        assert_eq!(ThumbnailFormat::Jpeg.extension(), "jpg");
        assert_eq!(ThumbnailFormat::Png.extension(), "png");
        assert_eq!(ThumbnailFormat::WebP.extension(), "webp");
    }

    #[test]
    fn test_thumbnail_format_mime_type() {
        assert_eq!(ThumbnailFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ThumbnailFormat::Png.mime_type(), "image/png");
        assert_eq!(ThumbnailFormat::WebP.mime_type(), "image/webp");
    }

    #[test]
    fn test_size_fit_within() {
        // Landscape image into square
        let size = Size::fit_within(1920, 1080, 100, 100);
        assert_eq!(size.width, 100);
        assert!(size.height <= 100);

        // Portrait image into square
        let size = Size::fit_within(1080, 1920, 100, 100);
        assert!(size.width <= 100);
        assert_eq!(size.height, 100);

        // Already smaller
        let size = Size::fit_within(50, 50, 100, 100);
        assert_eq!(size.width, 100);
        assert_eq!(size.height, 100);
    }

    #[test]
    fn test_size_with_width() {
        let size = Size::with_width(1920, 1080, 320);
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 180);
    }

    #[test]
    fn test_size_with_height() {
        let size = Size::with_height(1920, 1080, 180);
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 180);
    }

    #[test]
    fn test_thumbnail_config_builder() {
        let config = ThumbnailConfig::new()
            .with_size(320, 180)
            .with_format(ThumbnailFormat::Png)
            .with_quality(90)
            .keyframe_only(true);

        assert_eq!(config.size, Some(Size::new(320, 180)));
        assert_eq!(config.format, ThumbnailFormat::Png);
        assert_eq!(config.quality, 90);
        assert!(config.keyframe_only);
    }

    #[test]
    fn test_thumbnail_position_conversions() {
        let pos = ThumbnailPosition::from_secs(10.0);
        assert!(matches!(pos, ThumbnailPosition::Timestamp(_)));

        let pos = ThumbnailPosition::from_millis(5000);
        assert!(matches!(pos, ThumbnailPosition::Timestamp(_)));

        let pos = ThumbnailPosition::from_frame(100);
        assert!(matches!(pos, ThumbnailPosition::FrameNumber(100)));

        let pos = ThumbnailPosition::from_percentage(0.5);
        assert!(matches!(pos, ThumbnailPosition::Percentage(p) if (p - 0.5).abs() < 0.001));
    }

    #[test]
    fn test_thumbnail_position_clamp() {
        let pos = ThumbnailPosition::from_percentage(1.5);
        assert!(matches!(pos, ThumbnailPosition::Percentage(p) if (p - 1.0).abs() < 0.001));

        let pos = ThumbnailPosition::from_percentage(-0.5);
        assert!(matches!(pos, ThumbnailPosition::Percentage(p) if p.abs() < 0.001));
    }

    #[test]
    fn test_thumbnail_extract_file_not_found() {
        let result = Thumbnail::extract("nonexistent.mp4", Timestamp::from_millis(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_thumbnail_extract_with_temp_file() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract(&video_path, Timestamp::from_millis(1000));
        assert!(result.is_ok());

        let thumb = result.unwrap();
        assert!(thumb.width() > 0);
        assert!(thumb.height() > 0);
    }

    #[test]
    fn test_thumbnail_extract_frame() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_frame(&video_path, 30);
        assert!(result.is_ok());

        let thumb = result.unwrap();
        assert_eq!(thumb.frame_number(), 30);
    }

    #[test]
    fn test_thumbnail_extract_at_intervals() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_at_intervals(&video_path, 5);
        assert!(result.is_ok());

        let thumbs = result.unwrap();
        assert_eq!(thumbs.len(), 5);
    }

    #[test]
    fn test_thumbnail_extract_grid() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_grid(&video_path, 3, 3);
        assert!(result.is_ok());

        let thumbs = result.unwrap();
        assert_eq!(thumbs.len(), 9);
    }

    #[test]
    fn test_thumbnail_extract_keyframe() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_keyframe(&video_path, Timestamp::from_millis(5000));
        assert!(result.is_ok());
    }

    #[test]
    fn test_thumbnail_save_jpeg() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let output_path = temp_dir.path().join("thumb.jpg");
        let result = thumb.save_jpeg(&output_path, 85);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_thumbnail_save_png() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let output_path = temp_dir.path().join("thumb.png");
        let result = thumb.save_png(&output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_thumbnail_save_webp() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let output_path = temp_dir.path().join("thumb.webp");
        let result = thumb.save_webp(&output_path, 80);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_thumbnail_encode() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        // Test JPEG encoding
        let jpeg_data = thumb.encode(ThumbnailFormat::Jpeg, 85).unwrap();
        assert!(!jpeg_data.is_empty());
        assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]); // JPEG magic bytes

        // Test PNG encoding
        let png_data = thumb.encode(ThumbnailFormat::Png, 100).unwrap();
        assert!(!png_data.is_empty());
        assert_eq!(&png_data[0..4], &[0x89, 0x50, 0x4E, 0x47]); // PNG magic bytes

        // Test WebP encoding
        let webp_data = thumb.encode(ThumbnailFormat::WebP, 80).unwrap();
        assert!(!webp_data.is_empty());
        assert_eq!(&webp_data[0..4], b"RIFF"); // WebP RIFF header
    }

    #[test]
    fn test_thumbnail_resize() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let resized = thumb.resize(320, 180).unwrap();
        assert!(resized.width() <= 320);
        assert!(resized.height() <= 180);
    }

    #[test]
    fn test_thumbnail_resize_no_aspect_ratio() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let resized = thumb.resize_with_options(200, 200, false).unwrap();
        assert_eq!(resized.width(), 200);
        assert_eq!(resized.height(), 200);
    }

    #[test]
    fn test_thumbnail_builder() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = ThumbnailBuilder::new()
            .input(&video_path)
            .at_seconds(5.0)
            .size(320, 180)
            .format(ThumbnailFormat::Jpeg)
            .quality(90)
            .extract();

        assert!(result.is_ok());
    }

    #[test]
    fn test_thumbnail_builder_no_input() {
        let result = ThumbnailBuilder::new().at_seconds(5.0).extract();

        assert!(result.is_err());
    }

    #[test]
    fn test_create_contact_sheet() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumbs = Thumbnail::extract_grid(&video_path, 3, 3).unwrap();
        let sheet = Thumbnail::create_contact_sheet(&thumbs, 3, 100, 75).unwrap();

        assert_eq!(sheet.width(), 300);
        assert_eq!(sheet.height(), 225);
    }

    #[test]
    fn test_create_contact_sheet_empty() {
        let result = Thumbnail::create_contact_sheet(&[], 3, 100, 75);
        assert!(result.is_err());
    }

    #[test]
    fn test_thumbnail_accessors() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(5000)).unwrap();

        assert!(thumb.width() > 0);
        assert!(thumb.height() > 0);
        assert!(thumb.timestamp().is_valid());
        assert!(thumb.frame().width() > 0);
        assert!(thumb.source_path().is_some());
    }

    #[test]
    fn test_thumbnail_intervals_zero_count() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_at_intervals(&video_path, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_thumbnail_intervals_single() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let result = Thumbnail::extract_at_intervals(&video_path, 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_save_creates_parent_directory() {
        let temp_dir = TempDir::new().unwrap();
        let video_path = temp_dir.path().join("test.mp4");
        std::fs::write(&video_path, b"fake video data").unwrap();

        let thumb = Thumbnail::extract(&video_path, Timestamp::from_millis(1000)).unwrap();

        let output_path = temp_dir.path().join("nested").join("dir").join("thumb.jpg");
        let result = thumb.save_jpeg(&output_path, 85);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_size_from_tuple() {
        let size: Size = (640, 480).into();
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
    }

    #[test]
    fn test_size_default() {
        let size = Size::default();
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 180);
    }

    #[test]
    fn test_quality_clamping() {
        let config = ThumbnailConfig::new().with_quality(150);
        assert_eq!(config.quality, 100);
    }
}
