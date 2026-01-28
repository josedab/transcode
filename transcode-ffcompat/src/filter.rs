//! FFmpeg filter graph parsing and translation.

use serde::{Deserialize, Serialize};

/// A parsed FFmpeg filter with name and parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedFilter {
    pub name: String,
    pub params: Vec<FilterParam>,
}

/// A filter parameter (key=value or positional).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterParam {
    pub key: Option<String>,
    pub value: String,
}

/// A chain of filters connected with commas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterChain {
    pub filters: Vec<ParsedFilter>,
}

/// A complete filter graph (chains connected with semicolons).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterGraph {
    pub chains: Vec<FilterChain>,
}

/// Translated filter in Transcode-native format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeFilter {
    pub name: String,
    pub params: std::collections::HashMap<String, String>,
    pub supported: bool,
}

impl FilterGraph {
    /// Parse an FFmpeg filter graph string.
    /// Supports: `filter1=param1:param2,filter2=param1;chain2_filter1`
    pub fn parse(input: &str) -> Self {
        let chains = input
            .split(';')
            .map(|chain_str| {
                let filters = chain_str
                    .split(',')
                    .map(|filter_str| parse_single_filter(filter_str.trim()))
                    .collect();
                FilterChain { filters }
            })
            .collect();

        FilterGraph { chains }
    }

    /// Translate all filters to native Transcode equivalents.
    pub fn translate(&self) -> Vec<NativeFilter> {
        self.chains
            .iter()
            .flat_map(|chain| chain.filters.iter().map(translate_filter))
            .collect()
    }

    /// Calculate how many filters are supported.
    pub fn support_ratio(&self) -> f64 {
        let translated = self.translate();
        if translated.is_empty() {
            return 1.0;
        }
        let supported = translated.iter().filter(|f| f.supported).count();
        supported as f64 / translated.len() as f64
    }
}

fn parse_single_filter(s: &str) -> ParsedFilter {
    // Format: name=param1:param2:key=value
    let (name, params_str) = match s.split_once('=') {
        Some((n, p)) => (n.trim(), Some(p)),
        None => (s.trim(), None),
    };

    // Strip any stream specifiers like [0:v] from the name
    let name = name
        .trim_start_matches(|c: char| c == '[' || c.is_ascii_digit() || c == ':' || c == ']')
        .trim();

    let params = match params_str {
        Some(p) => p
            .split(':')
            .map(|param| {
                if let Some((k, v)) = param.split_once('=') {
                    FilterParam {
                        key: Some(k.to_string()),
                        value: v.to_string(),
                    }
                } else {
                    FilterParam {
                        key: None,
                        value: param.to_string(),
                    }
                }
            })
            .collect(),
        None => Vec::new(),
    };

    ParsedFilter {
        name: name.to_string(),
        params,
    }
}

fn translate_filter(filter: &ParsedFilter) -> NativeFilter {
    let mut params = std::collections::HashMap::new();

    match filter.name.as_str() {
        // Scaling
        "scale" => {
            for (i, p) in filter.params.iter().enumerate() {
                match (&p.key, i) {
                    (Some(k), _) => { params.insert(k.clone(), p.value.clone()); }
                    (None, 0) => { params.insert("width".into(), p.value.clone()); }
                    (None, 1) => { params.insert("height".into(), p.value.clone()); }
                    _ => {}
                }
            }
            NativeFilter { name: "scale".into(), params, supported: true }
        }
        // Frame rate
        "fps" => {
            if let Some(p) = filter.params.first() {
                params.insert("fps".into(), p.value.clone());
            }
            NativeFilter { name: "framerate".into(), params, supported: true }
        }
        // Crop
        "crop" => {
            for (i, p) in filter.params.iter().enumerate() {
                match (&p.key, i) {
                    (Some(k), _) => { params.insert(k.clone(), p.value.clone()); }
                    (None, 0) => { params.insert("width".into(), p.value.clone()); }
                    (None, 1) => { params.insert("height".into(), p.value.clone()); }
                    (None, 2) => { params.insert("x".into(), p.value.clone()); }
                    (None, 3) => { params.insert("y".into(), p.value.clone()); }
                    _ => {}
                }
            }
            NativeFilter { name: "crop".into(), params, supported: true }
        }
        // Deinterlace
        "yadif" | "bwdif" => {
            NativeFilter { name: "deinterlace".into(), params, supported: true }
        }
        // Transpose / rotation
        "transpose" => {
            if let Some(p) = filter.params.first() {
                params.insert("direction".into(), p.value.clone());
            }
            NativeFilter { name: "rotate".into(), params, supported: true }
        }
        // Color adjustment
        "eq" => {
            for p in &filter.params {
                if let Some(k) = &p.key {
                    params.insert(k.clone(), p.value.clone());
                }
            }
            NativeFilter { name: "color_adjust".into(), params, supported: true }
        }
        // Audio filters
        "loudnorm" => {
            for p in &filter.params {
                if let Some(k) = &p.key {
                    params.insert(k.clone(), p.value.clone());
                }
            }
            NativeFilter { name: "loudness_normalize".into(), params, supported: true }
        }
        "aresample" => {
            if let Some(p) = filter.params.first() {
                params.insert("sample_rate".into(), p.value.clone());
            }
            NativeFilter { name: "resample".into(), params, supported: true }
        }
        // HDR tone mapping
        "tonemap" => {
            for p in &filter.params {
                if let Some(k) = &p.key {
                    params.insert(k.clone(), p.value.clone());
                }
            }
            NativeFilter { name: "hdr_tonemap".into(), params, supported: true }
        }
        // Pad
        "pad" => {
            for (i, p) in filter.params.iter().enumerate() {
                match (&p.key, i) {
                    (Some(k), _) => { params.insert(k.clone(), p.value.clone()); }
                    (None, 0) => { params.insert("width".into(), p.value.clone()); }
                    (None, 1) => { params.insert("height".into(), p.value.clone()); }
                    _ => {}
                }
            }
            NativeFilter { name: "pad".into(), params, supported: true }
        }
        // Unsupported
        _ => NativeFilter {
            name: filter.name.clone(),
            params,
            supported: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_filter() {
        let graph = FilterGraph::parse("scale=1280:720");
        assert_eq!(graph.chains.len(), 1);
        assert_eq!(graph.chains[0].filters.len(), 1);
        assert_eq!(graph.chains[0].filters[0].name, "scale");
        assert_eq!(graph.chains[0].filters[0].params.len(), 2);
    }

    #[test]
    fn test_parse_filter_chain() {
        let graph = FilterGraph::parse("scale=1280:720,fps=30,yadif");
        assert_eq!(graph.chains[0].filters.len(), 3);
        assert_eq!(graph.chains[0].filters[2].name, "yadif");
    }

    #[test]
    fn test_parse_multi_chain() {
        let graph = FilterGraph::parse("scale=1280:720;loudnorm");
        assert_eq!(graph.chains.len(), 2);
    }

    #[test]
    fn test_translate_scale() {
        let graph = FilterGraph::parse("scale=1920:1080");
        let native = graph.translate();
        assert_eq!(native.len(), 1);
        assert_eq!(native[0].name, "scale");
        assert!(native[0].supported);
        assert_eq!(native[0].params.get("width"), Some(&"1920".to_string()));
    }

    #[test]
    fn test_translate_fps() {
        let graph = FilterGraph::parse("fps=30");
        let native = graph.translate();
        assert_eq!(native[0].name, "framerate");
        assert!(native[0].supported);
    }

    #[test]
    fn test_translate_unsupported() {
        let graph = FilterGraph::parse("overlay=10:10");
        let native = graph.translate();
        assert!(!native[0].supported);
    }

    #[test]
    fn test_support_ratio() {
        let graph = FilterGraph::parse("scale=1280:720,fps=30,overlay=10:10");
        let ratio = graph.support_ratio();
        assert!((ratio - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_complex_filter_chain() {
        let graph = FilterGraph::parse("scale=1280:720,yadif,eq=brightness=0.1:contrast=1.2");
        let native = graph.translate();
        assert_eq!(native.len(), 3);
        assert!(native.iter().all(|f| f.supported));
    }

    #[test]
    fn test_named_params() {
        let graph = FilterGraph::parse("eq=brightness=0.1:contrast=1.2");
        let native = graph.translate();
        assert_eq!(native[0].params.get("brightness"), Some(&"0.1".to_string()));
        assert_eq!(native[0].params.get("contrast"), Some(&"1.2".to_string()));
    }
}
