//! OpenEXR channel definitions

use std::fmt;

/// Pixel data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    /// 32-bit unsigned integer
    Uint,
    /// 16-bit float (half)
    Half,
    /// 32-bit float
    Float,
}

impl Default for PixelType {
    fn default() -> Self {
        PixelType::Half
    }
}

impl PixelType {
    /// Create from u32 value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(PixelType::Uint),
            1 => Some(PixelType::Half),
            2 => Some(PixelType::Float),
            _ => None,
        }
    }

    /// Convert to u32
    pub fn to_u32(self) -> u32 {
        match self {
            PixelType::Uint => 0,
            PixelType::Half => 1,
            PixelType::Float => 2,
        }
    }

    /// Bytes per sample for this type
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PixelType::Uint => 4,
            PixelType::Half => 2,
            PixelType::Float => 4,
        }
    }

    /// Name of this pixel type
    pub fn name(self) -> &'static str {
        match self {
            PixelType::Uint => "UINT",
            PixelType::Half => "HALF",
            PixelType::Float => "FLOAT",
        }
    }
}

/// A single channel in an EXR image
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// Channel name (e.g., "R", "G", "B", "A")
    pub name: String,
    /// Pixel data type
    pub pixel_type: PixelType,
    /// Subsampling in X (1 = no subsampling)
    pub x_sampling: i32,
    /// Subsampling in Y (1 = no subsampling)
    pub y_sampling: i32,
    /// Perceptually linear (used for color management hints)
    pub p_linear: bool,
}

impl Channel {
    /// Create new channel
    pub fn new(name: &str, pixel_type: PixelType) -> Self {
        Channel {
            name: name.to_string(),
            pixel_type,
            x_sampling: 1,
            y_sampling: 1,
            p_linear: false,
        }
    }

    /// Create half-float channel
    pub fn half(name: &str) -> Self {
        Channel::new(name, PixelType::Half)
    }

    /// Create float channel
    pub fn float(name: &str) -> Self {
        Channel::new(name, PixelType::Float)
    }

    /// Create uint channel
    pub fn uint(name: &str) -> Self {
        Channel::new(name, PixelType::Uint)
    }

    /// Set subsampling
    pub fn with_sampling(mut self, x: i32, y: i32) -> Self {
        self.x_sampling = x;
        self.y_sampling = y;
        self
    }

    /// Set perceptually linear flag
    pub fn with_p_linear(mut self, p_linear: bool) -> Self {
        self.p_linear = p_linear;
        self
    }

    /// Check if this is a color channel
    pub fn is_color(&self) -> bool {
        matches!(self.name.as_str(), "R" | "G" | "B" | "r" | "g" | "b")
    }

    /// Check if this is an alpha channel
    pub fn is_alpha(&self) -> bool {
        matches!(self.name.as_str(), "A" | "a" | "Alpha" | "alpha")
    }

    /// Check if this is a luminance channel
    pub fn is_luminance(&self) -> bool {
        matches!(self.name.as_str(), "Y" | "L" | "y" | "l" | "luminance")
    }

    /// Check if this is a depth channel
    pub fn is_depth(&self) -> bool {
        matches!(self.name.as_str(), "Z" | "z" | "depth" | "ZBack")
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.pixel_type.name())
    }
}

/// List of channels in an EXR image
#[derive(Debug, Clone, Default)]
pub struct ChannelList {
    channels: Vec<Channel>,
}

impl ChannelList {
    /// Create empty channel list
    pub fn new() -> Self {
        ChannelList {
            channels: Vec::new(),
        }
    }

    /// Create RGB channel list
    pub fn rgb(pixel_type: PixelType) -> Self {
        let mut list = ChannelList::new();
        list.add(Channel::new("B", pixel_type));
        list.add(Channel::new("G", pixel_type));
        list.add(Channel::new("R", pixel_type));
        list
    }

    /// Create RGBA channel list
    pub fn rgba(pixel_type: PixelType) -> Self {
        let mut list = ChannelList::new();
        list.add(Channel::new("A", pixel_type));
        list.add(Channel::new("B", pixel_type));
        list.add(Channel::new("G", pixel_type));
        list.add(Channel::new("R", pixel_type));
        list
    }

    /// Create luminance channel list
    pub fn luminance(pixel_type: PixelType) -> Self {
        let mut list = ChannelList::new();
        list.add(Channel::new("Y", pixel_type));
        list
    }

    /// Create luminance + alpha channel list
    pub fn luminance_alpha(pixel_type: PixelType) -> Self {
        let mut list = ChannelList::new();
        list.add(Channel::new("A", pixel_type));
        list.add(Channel::new("Y", pixel_type));
        list
    }

    /// Add channel (maintains sorted order by name)
    pub fn add(&mut self, channel: Channel) {
        // Insert in sorted order
        let pos = self
            .channels
            .binary_search_by(|c| c.name.cmp(&channel.name))
            .unwrap_or_else(|e| e);
        self.channels.insert(pos, channel);
    }

    /// Get channel by name
    pub fn get(&self, name: &str) -> Option<&Channel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Get channel index by name
    pub fn index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.name == name)
    }

    /// Number of channels
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Iterate over channels
    pub fn iter(&self) -> impl Iterator<Item = &Channel> {
        self.channels.iter()
    }

    /// Get channels as slice
    pub fn as_slice(&self) -> &[Channel] {
        &self.channels
    }

    /// Calculate bytes per pixel
    pub fn bytes_per_pixel(&self) -> usize {
        self.channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum()
    }

    /// Check if has RGB channels
    pub fn has_rgb(&self) -> bool {
        self.get("R").is_some() && self.get("G").is_some() && self.get("B").is_some()
    }

    /// Check if has alpha channel
    pub fn has_alpha(&self) -> bool {
        self.get("A").is_some()
    }

    /// Check if has depth channel
    pub fn has_depth(&self) -> bool {
        self.get("Z").is_some()
    }

    /// Get color channels (R, G, B)
    pub fn color_channels(&self) -> Vec<&Channel> {
        self.channels.iter().filter(|c| c.is_color()).collect()
    }
}

impl IntoIterator for ChannelList {
    type Item = Channel;
    type IntoIter = std::vec::IntoIter<Channel>;

    fn into_iter(self) -> Self::IntoIter {
        self.channels.into_iter()
    }
}

impl<'a> IntoIterator for &'a ChannelList {
    type Item = &'a Channel;
    type IntoIter = std::slice::Iter<'a, Channel>;

    fn into_iter(self) -> Self::IntoIter {
        self.channels.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_type() {
        assert_eq!(PixelType::from_u32(0), Some(PixelType::Uint));
        assert_eq!(PixelType::from_u32(1), Some(PixelType::Half));
        assert_eq!(PixelType::from_u32(2), Some(PixelType::Float));
        assert_eq!(PixelType::from_u32(3), None);

        assert_eq!(PixelType::Half.bytes_per_sample(), 2);
        assert_eq!(PixelType::Float.bytes_per_sample(), 4);
        assert_eq!(PixelType::Uint.bytes_per_sample(), 4);
    }

    #[test]
    fn test_channel() {
        let c = Channel::half("R");
        assert_eq!(c.name, "R");
        assert_eq!(c.pixel_type, PixelType::Half);
        assert!(c.is_color());
        assert!(!c.is_alpha());

        let c = Channel::float("A");
        assert!(c.is_alpha());
    }

    #[test]
    fn test_channel_list() {
        let list = ChannelList::rgba(PixelType::Half);
        assert_eq!(list.len(), 4);
        assert!(list.has_rgb());
        assert!(list.has_alpha());

        // Channels should be sorted: A, B, G, R
        let names: Vec<_> = list.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["A", "B", "G", "R"]);
    }

    #[test]
    fn test_bytes_per_pixel() {
        let list = ChannelList::rgba(PixelType::Half);
        assert_eq!(list.bytes_per_pixel(), 8); // 4 channels * 2 bytes

        let list = ChannelList::rgba(PixelType::Float);
        assert_eq!(list.bytes_per_pixel(), 16); // 4 channels * 4 bytes
    }

    #[test]
    fn test_channel_lookup() {
        let mut list = ChannelList::new();
        list.add(Channel::half("R"));
        list.add(Channel::half("G"));
        list.add(Channel::half("B"));

        assert!(list.get("R").is_some());
        assert!(list.get("A").is_none());
        assert_eq!(list.index("G"), Some(1));
    }
}
