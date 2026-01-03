//! FFmpeg filter graph syntax parser.
//!
//! This module parses FFmpeg's filter graph syntax, supporting:
//! - Simple filter chains (scale=1920:1080,fps=30)
//! - Filter parameters with named or positional arguments
//! - Filter graph links ([in][out])
//! - Complex filter graphs with multiple inputs/outputs

use crate::error::{CompatError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A parsed filter with its name and parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// Filter name (e.g., "scale", "fps", "volume").
    pub name: String,
    /// Filter parameters as key-value pairs.
    /// For positional parameters, keys are "0", "1", etc.
    pub params: HashMap<String, String>,
    /// Input pad links.
    pub inputs: Vec<String>,
    /// Output pad links.
    pub outputs: Vec<String>,
}

impl Filter {
    /// Create a new filter with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a parameter.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Add an input link.
    pub fn input(mut self, link: impl Into<String>) -> Self {
        self.inputs.push(link.into());
        self
    }

    /// Add an output link.
    pub fn output(mut self, link: impl Into<String>) -> Self {
        self.outputs.push(link.into());
        self
    }

    /// Get a parameter by name.
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    /// Get a parameter by position.
    pub fn get_positional(&self, index: usize) -> Option<&str> {
        self.params.get(&index.to_string()).map(|s| s.as_str())
    }

    /// Get parameter as u32.
    pub fn get_param_u32(&self, key: &str) -> Option<u32> {
        self.get_param(key).and_then(|v| v.parse().ok())
    }

    /// Get parameter as f64.
    pub fn get_param_f64(&self, key: &str) -> Option<f64> {
        self.get_param(key).and_then(|v| v.parse().ok())
    }

    /// Parse a single filter string like "scale=1920:1080" or "fps=30".
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(CompatError::InvalidFilter("empty filter".to_string()));
        }

        // Find filter name and parameters
        let (name, params_str) = if let Some(eq_pos) = s.find('=') {
            (&s[..eq_pos], Some(&s[eq_pos + 1..]))
        } else {
            (s, None)
        };

        let mut filter = Self::new(name);

        if let Some(params_str) = params_str {
            filter.params = parse_filter_params(params_str)?;
        }

        Ok(filter)
    }
}

impl std::fmt::Display for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write input links
        for input in &self.inputs {
            write!(f, "[{}]", input)?;
        }

        // Write filter name
        write!(f, "{}", self.name)?;

        // Write parameters
        if !self.params.is_empty() {
            write!(f, "=")?;
            let mut params: Vec<_> = self.params.iter().collect();
            params.sort_by(|a, b| {
                // Sort numeric keys (positional) first, then alphabetic
                match (a.0.parse::<usize>(), b.0.parse::<usize>()) {
                    (Ok(n1), Ok(n2)) => n1.cmp(&n2),
                    (Ok(_), Err(_)) => std::cmp::Ordering::Less,
                    (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
                    (Err(_), Err(_)) => a.0.cmp(b.0),
                }
            });

            for (i, (key, value)) in params.iter().enumerate() {
                if i > 0 {
                    write!(f, ":")?;
                }
                // Use positional syntax if key is numeric
                if key.parse::<usize>().is_ok() {
                    write!(f, "{}", value)?;
                } else {
                    write!(f, "{}={}", key, value)?;
                }
            }
        }

        // Write output links
        for output in &self.outputs {
            write!(f, "[{}]", output)?;
        }

        Ok(())
    }
}

/// Parse filter parameters string like "1920:1080" or "w=1920:h=1080".
fn parse_filter_params(s: &str) -> Result<HashMap<String, String>> {
    let mut params = HashMap::new();
    let mut positional_idx = 0;

    // Split by colon, but handle escaped colons and quoted strings
    let parts = split_params(s)?;

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some(eq_pos) = part.find('=') {
            // Named parameter
            let key = &part[..eq_pos];
            let value = &part[eq_pos + 1..];
            params.insert(key.to_string(), value.to_string());
        } else {
            // Positional parameter
            params.insert(positional_idx.to_string(), part.to_string());
            positional_idx += 1;
        }
    }

    Ok(params)
}

/// Split parameters by colon, respecting quotes and escape sequences.
fn split_params(s: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut depth = 0; // For handling nested brackets

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // Escape sequence
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = c;
                current.push(c);
            }
            c if c == quote_char && in_quotes => {
                in_quotes = false;
                current.push(c);
            }
            '[' if !in_quotes => {
                depth += 1;
                current.push(c);
            }
            ']' if !in_quotes => {
                depth -= 1;
                current.push(c);
            }
            ':' if !in_quotes && depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    Ok(parts)
}

/// A chain of filters connected sequentially.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterChain {
    /// The filters in this chain.
    pub filters: Vec<Filter>,
}

impl FilterChain {
    /// Create a new empty filter chain.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the chain.
    pub fn push(mut self, filter: Filter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Parse a filter chain string like "scale=1920:1080,fps=30".
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        let mut chain = Self::new();
        let filter_strs = split_filter_chain(s)?;

        for filter_str in filter_strs {
            let (filter_str, inputs, outputs) = extract_links(&filter_str)?;
            let mut filter = Filter::parse(&filter_str)?;
            filter.inputs = inputs;
            filter.outputs = outputs;
            chain.filters.push(filter);
        }

        Ok(chain)
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Get the number of filters.
    pub fn len(&self) -> usize {
        self.filters.len()
    }
}

impl Default for FilterChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FilterChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, filter) in self.filters.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", filter)?;
        }
        Ok(())
    }
}

/// Split a filter chain by commas, respecting brackets and quotes.
fn split_filter_chain(s: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut bracket_depth = 0;

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(next) = chars.next() {
                    current.push('\\');
                    current.push(next);
                }
            }
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = c;
                current.push(c);
            }
            c if c == quote_char && in_quotes => {
                in_quotes = false;
                current.push(c);
            }
            '[' if !in_quotes => {
                bracket_depth += 1;
                current.push(c);
            }
            ']' if !in_quotes => {
                bracket_depth -= 1;
                current.push(c);
            }
            ',' if !in_quotes && bracket_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }

    Ok(parts)
}

/// Extract input/output links from a filter string.
fn extract_links(s: &str) -> Result<(String, Vec<String>, Vec<String>)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut remaining = s.trim();

    // Extract leading input links
    while remaining.starts_with('[') {
        if let Some(end) = remaining.find(']') {
            inputs.push(remaining[1..end].to_string());
            remaining = remaining[end + 1..].trim();
        } else {
            return Err(CompatError::InvalidFilter(format!(
                "unclosed bracket in: {}",
                s
            )));
        }
    }

    // Extract trailing output links
    let mut chars: Vec<char> = remaining.chars().collect();
    while !chars.is_empty() && *chars.last().unwrap() == ']' {
        // Find matching opening bracket
        let mut depth = 0;
        let mut start_idx = None;
        for (i, &c) in chars.iter().enumerate().rev() {
            if c == ']' {
                depth += 1;
            } else if c == '[' {
                depth -= 1;
                if depth == 0 {
                    start_idx = Some(i);
                    break;
                }
            }
        }

        if let Some(start) = start_idx {
            let link: String = chars[start + 1..chars.len() - 1].iter().collect();
            outputs.insert(0, link);
            chars.truncate(start);
        } else {
            break;
        }
    }

    let filter_str: String = chars.into_iter().collect();
    Ok((filter_str.trim().to_string(), inputs, outputs))
}

/// A complex filter graph with multiple chains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterGraph {
    /// The filter chains in this graph.
    pub chains: Vec<FilterChain>,
}

impl FilterGraph {
    /// Create a new empty filter graph.
    pub fn new() -> Self {
        Self { chains: Vec::new() }
    }

    /// Add a chain to the graph.
    pub fn add_chain(mut self, chain: FilterChain) -> Self {
        self.chains.push(chain);
        self
    }

    /// Parse a complex filter graph.
    ///
    /// Chains are separated by semicolons.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        let mut graph = Self::new();
        let chain_strs = split_chains(s)?;

        for chain_str in chain_strs {
            let chain = FilterChain::parse(&chain_str)?;
            if !chain.is_empty() {
                graph.chains.push(chain);
            }
        }

        Ok(graph)
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty() || self.chains.iter().all(|c| c.is_empty())
    }

    /// Get all filters in the graph.
    pub fn all_filters(&self) -> impl Iterator<Item = &Filter> {
        self.chains.iter().flat_map(|c| c.filters.iter())
    }

    /// Get a filter by name.
    pub fn find_filter(&self, name: &str) -> Option<&Filter> {
        self.all_filters().find(|f| f.name == name)
    }
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FilterGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, chain) in self.chains.iter().enumerate() {
            if i > 0 {
                write!(f, ";")?;
            }
            write!(f, "{}", chain)?;
        }
        Ok(())
    }
}

/// Split filter graph by semicolons.
fn split_chains(s: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut bracket_depth = 0;

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(next) = chars.next() {
                    current.push('\\');
                    current.push(next);
                }
            }
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = c;
                current.push(c);
            }
            c if c == quote_char && in_quotes => {
                in_quotes = false;
                current.push(c);
            }
            '[' if !in_quotes => {
                bracket_depth += 1;
                current.push(c);
            }
            ']' if !in_quotes => {
                bracket_depth -= 1;
                current.push(c);
            }
            ';' if !in_quotes && bracket_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }

    Ok(parts)
}

/// Common video filters as convenience constructors.
pub mod video {
    use super::Filter;

    /// Create a scale filter.
    pub fn scale(width: u32, height: u32) -> Filter {
        Filter::new("scale")
            .param("0", width.to_string())
            .param("1", height.to_string())
    }

    /// Create a scale filter with named parameters.
    pub fn scale_named(width: u32, height: u32) -> Filter {
        Filter::new("scale")
            .param("w", width.to_string())
            .param("h", height.to_string())
    }

    /// Create an fps filter.
    pub fn fps(rate: f64) -> Filter {
        Filter::new("fps").param("0", rate.to_string())
    }

    /// Create a crop filter.
    pub fn crop(width: u32, height: u32, x: u32, y: u32) -> Filter {
        Filter::new("crop")
            .param("w", width.to_string())
            .param("h", height.to_string())
            .param("x", x.to_string())
            .param("y", y.to_string())
    }

    /// Create a pad filter.
    pub fn pad(width: u32, height: u32, x: u32, y: u32) -> Filter {
        Filter::new("pad")
            .param("w", width.to_string())
            .param("h", height.to_string())
            .param("x", x.to_string())
            .param("y", y.to_string())
    }

    /// Create a setdar (display aspect ratio) filter.
    pub fn setdar(num: u32, den: u32) -> Filter {
        Filter::new("setdar").param("0", format!("{}/{}", num, den))
    }

    /// Create a setsar (sample aspect ratio) filter.
    pub fn setsar(num: u32, den: u32) -> Filter {
        Filter::new("setsar").param("0", format!("{}/{}", num, den))
    }

    /// Create a transpose filter.
    pub fn transpose(direction: u8) -> Filter {
        Filter::new("transpose").param("0", direction.to_string())
    }

    /// Create an hflip filter.
    pub fn hflip() -> Filter {
        Filter::new("hflip")
    }

    /// Create a vflip filter.
    pub fn vflip() -> Filter {
        Filter::new("vflip")
    }

    /// Create a null (passthrough) filter.
    pub fn null() -> Filter {
        Filter::new("null")
    }

    /// Create a format (pixel format) filter.
    pub fn format(pix_fmt: &str) -> Filter {
        Filter::new("format").param("0", pix_fmt.to_string())
    }

    /// Create a yadif (deinterlace) filter.
    pub fn yadif() -> Filter {
        Filter::new("yadif")
    }

    /// Create a delogo filter.
    pub fn delogo(x: u32, y: u32, width: u32, height: u32) -> Filter {
        Filter::new("delogo")
            .param("x", x.to_string())
            .param("y", y.to_string())
            .param("w", width.to_string())
            .param("h", height.to_string())
    }
}

/// Common audio filters as convenience constructors.
pub mod audio {
    use super::Filter;

    /// Create a volume filter.
    pub fn volume(level: f64) -> Filter {
        Filter::new("volume").param("0", level.to_string())
    }

    /// Create a volume filter with dB.
    pub fn volume_db(db: f64) -> Filter {
        Filter::new("volume").param("0", format!("{}dB", db))
    }

    /// Create an aresample filter.
    pub fn aresample(rate: u32) -> Filter {
        Filter::new("aresample").param("0", rate.to_string())
    }

    /// Create an atempo filter for speed adjustment.
    pub fn atempo(tempo: f64) -> Filter {
        Filter::new("atempo").param("0", tempo.to_string())
    }

    /// Create an anull (passthrough) filter.
    pub fn anull() -> Filter {
        Filter::new("anull")
    }

    /// Create an aformat filter.
    pub fn aformat(sample_fmt: &str, sample_rate: u32, channel_layout: &str) -> Filter {
        Filter::new("aformat")
            .param("sample_fmts", sample_fmt.to_string())
            .param("sample_rates", sample_rate.to_string())
            .param("channel_layouts", channel_layout.to_string())
    }

    /// Create a pan filter for channel remapping.
    pub fn pan(layout: &str, gains: &str) -> Filter {
        Filter::new("pan").param("0", format!("{}|{}", layout, gains))
    }

    /// Create a highpass filter.
    pub fn highpass(frequency: u32) -> Filter {
        Filter::new("highpass").param("f", frequency.to_string())
    }

    /// Create a lowpass filter.
    pub fn lowpass(frequency: u32) -> Filter {
        Filter::new("lowpass").param("f", frequency.to_string())
    }

    /// Create an equalizer filter.
    pub fn equalizer(frequency: u32, width: f64, gain: f64) -> Filter {
        Filter::new("equalizer")
            .param("f", frequency.to_string())
            .param("width_type", "h".to_string())
            .param("w", width.to_string())
            .param("g", gain.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_parse_simple() {
        let filter = Filter::parse("scale=1920:1080").unwrap();
        assert_eq!(filter.name, "scale");
        assert_eq!(filter.get_positional(0), Some("1920"));
        assert_eq!(filter.get_positional(1), Some("1080"));
    }

    #[test]
    fn test_filter_parse_named_params() {
        let filter = Filter::parse("scale=w=1920:h=1080").unwrap();
        assert_eq!(filter.name, "scale");
        assert_eq!(filter.get_param("w"), Some("1920"));
        assert_eq!(filter.get_param("h"), Some("1080"));
    }

    #[test]
    fn test_filter_parse_no_params() {
        let filter = Filter::parse("hflip").unwrap();
        assert_eq!(filter.name, "hflip");
        assert!(filter.params.is_empty());
    }

    #[test]
    fn test_filter_chain_parse() {
        let chain = FilterChain::parse("scale=1920:1080,fps=30").unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.filters[0].name, "scale");
        assert_eq!(chain.filters[1].name, "fps");
    }

    #[test]
    fn test_filter_graph_parse() {
        let graph =
            FilterGraph::parse("[0:v]scale=1920:1080[v];[0:a]volume=0.5[a]").unwrap();
        assert_eq!(graph.chains.len(), 2);

        let video_chain = &graph.chains[0];
        assert_eq!(video_chain.filters[0].inputs, vec!["0:v"]);
        assert_eq!(video_chain.filters[0].outputs, vec!["v"]);

        let audio_chain = &graph.chains[1];
        assert_eq!(audio_chain.filters[0].inputs, vec!["0:a"]);
        assert_eq!(audio_chain.filters[0].outputs, vec!["a"]);
    }

    #[test]
    fn test_filter_display() {
        let filter = video::scale(1920, 1080);
        let s = filter.to_string();
        assert!(s.contains("scale"));
        assert!(s.contains("1920"));
        assert!(s.contains("1080"));
    }

    #[test]
    fn test_filter_chain_display() {
        let chain = FilterChain::new()
            .push(video::scale(1920, 1080))
            .push(video::fps(30.0));
        let s = chain.to_string();
        assert!(s.contains("scale"));
        assert!(s.contains("fps"));
        assert!(s.contains(','));
    }

    #[test]
    fn test_extract_links() {
        let (filter_str, inputs, outputs) =
            extract_links("[in]scale=1920:1080[out]").unwrap();
        assert_eq!(filter_str, "scale=1920:1080");
        assert_eq!(inputs, vec!["in"]);
        assert_eq!(outputs, vec!["out"]);
    }

    #[test]
    fn test_video_filter_helpers() {
        let filter = video::scale(1280, 720);
        assert_eq!(filter.name, "scale");
        assert_eq!(filter.get_positional(0), Some("1280"));
        assert_eq!(filter.get_positional(1), Some("720"));

        let filter = video::fps(29.97);
        assert_eq!(filter.name, "fps");
        assert_eq!(filter.get_positional(0), Some("29.97"));
    }

    #[test]
    fn test_audio_filter_helpers() {
        let filter = audio::volume(0.5);
        assert_eq!(filter.name, "volume");
        assert_eq!(filter.get_positional(0), Some("0.5"));

        let filter = audio::volume_db(-3.0);
        assert_eq!(filter.name, "volume");
        assert_eq!(filter.get_positional(0), Some("-3dB"));
    }

    #[test]
    fn test_find_filter() {
        let graph = FilterGraph::parse("scale=1920:1080,fps=30;volume=0.5").unwrap();

        let scale = graph.find_filter("scale");
        assert!(scale.is_some());
        assert_eq!(scale.unwrap().get_positional(0), Some("1920"));

        let volume = graph.find_filter("volume");
        assert!(volume.is_some());

        assert!(graph.find_filter("nonexistent").is_none());
    }
}
