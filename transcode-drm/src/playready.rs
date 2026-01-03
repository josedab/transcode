//! PlayReady header and PSSH generation.
//!
//! This module implements Microsoft PlayReady-specific PSSH box and
//! PlayReady Header Object (PRO) generation for DRM packaging.

use crate::error::{PsshError, Result};
use crate::key::{system_ids, KeyId};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// PlayReady system ID.
pub const PLAYREADY_SYSTEM_ID: Uuid = system_ids::PLAYREADY;

/// PlayReady Header Object version.
pub const PRO_VERSION: u32 = 0x4_0_0_0; // Version 4.0.0.0

/// Maximum size for PlayReady header.
pub const MAX_HEADER_SIZE: usize = 64 * 1024;

/// PlayReady algorithm types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayReadyAlgorithm {
    /// AES-CTR encryption (CENC).
    #[default]
    AesCtr,
    /// AES-CBC encryption (CBCS/cocktail).
    AesCbc,
    /// Cocktail encryption (legacy).
    Cocktail,
}

impl PlayReadyAlgorithm {
    /// Get the algorithm ID string for the header.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AesCtr => "AESCTR",
            Self::AesCbc => "AESCBC",
            Self::Cocktail => "COCKTAIL",
        }
    }
}

/// PlayReady key record.
#[derive(Clone, Debug)]
pub struct KeyRecord {
    /// Key ID (GUID format).
    pub key_id: KeyId,
    /// Encryption type.
    pub algorithm: PlayReadyAlgorithm,
    /// Optional checksum (for verification).
    pub checksum: Option<Vec<u8>>,
}

impl KeyRecord {
    /// Create a new key record.
    pub fn new(key_id: KeyId, algorithm: PlayReadyAlgorithm) -> Self {
        Self {
            key_id,
            algorithm,
            checksum: None,
        }
    }

    /// Create with checksum.
    pub fn with_checksum(key_id: KeyId, algorithm: PlayReadyAlgorithm, checksum: Vec<u8>) -> Self {
        Self {
            key_id,
            algorithm,
            checksum: Some(checksum),
        }
    }
}

/// PlayReady Header Object (PRO).
///
/// Contains the WRMHEADER XML that describes the content protection.
#[derive(Clone, Debug)]
pub struct PlayReadyHeader {
    /// Version string.
    pub version: String,
    /// Key records.
    pub key_records: Vec<KeyRecord>,
    /// License acquisition URL.
    pub la_url: Option<String>,
    /// License user interface URL.
    pub lui_url: Option<String>,
    /// Domain service identifier.
    pub ds_id: Option<String>,
    /// Custom data.
    pub custom_data: Option<String>,
    /// Check digits for key verification.
    pub decryptor_setup: Option<String>,
}

impl PlayReadyHeader {
    /// Create a new PlayReady header with a single key.
    pub fn new(key_id: KeyId) -> Self {
        Self {
            version: "4.0.0.0".to_string(),
            key_records: vec![KeyRecord::new(key_id, PlayReadyAlgorithm::AesCtr)],
            la_url: None,
            lui_url: None,
            ds_id: None,
            custom_data: None,
            decryptor_setup: None,
        }
    }

    /// Create with multiple keys.
    pub fn with_keys(key_records: Vec<KeyRecord>) -> Self {
        Self {
            version: "4.0.0.0".to_string(),
            key_records,
            la_url: None,
            lui_url: None,
            ds_id: None,
            custom_data: None,
            decryptor_setup: None,
        }
    }

    /// Set the license acquisition URL.
    pub fn with_la_url(mut self, url: impl Into<String>) -> Self {
        self.la_url = Some(url.into());
        self
    }

    /// Set the license UI URL.
    pub fn with_lui_url(mut self, url: impl Into<String>) -> Self {
        self.lui_url = Some(url.into());
        self
    }

    /// Set custom data.
    pub fn with_custom_data(mut self, data: impl Into<String>) -> Self {
        self.custom_data = Some(data.into());
        self
    }

    /// Set decryptor setup.
    pub fn with_decryptor_setup(mut self, setup: impl Into<String>) -> Self {
        self.decryptor_setup = Some(setup.into());
        self
    }

    /// Add a key record.
    pub fn add_key(&mut self, key: KeyRecord) {
        self.key_records.push(key);
    }

    /// Generate the WRMHEADER XML.
    pub fn to_xml(&self) -> String {
        let mut xml = String::new();

        xml.push_str("<WRMHEADER xmlns=\"http://schemas.microsoft.com/DRM/2007/03/PlayReadyHeader\" version=\"");
        xml.push_str(&self.version);
        xml.push_str("\">");

        xml.push_str("<DATA>");

        // License acquisition URL
        if let Some(ref la_url) = self.la_url {
            xml.push_str("<LA_URL>");
            xml.push_str(&escape_xml(la_url));
            xml.push_str("</LA_URL>");
        }

        // License UI URL
        if let Some(ref lui_url) = self.lui_url {
            xml.push_str("<LUI_URL>");
            xml.push_str(&escape_xml(lui_url));
            xml.push_str("</LUI_URL>");
        }

        // Domain service ID
        if let Some(ref ds_id) = self.ds_id {
            xml.push_str("<DS_ID>");
            xml.push_str(ds_id);
            xml.push_str("</DS_ID>");
        }

        // Custom data
        if let Some(ref custom_data) = self.custom_data {
            xml.push_str("<CUSTOMATTRIBUTES><IIS_DRM_VERSION>8.0.1806.32</IIS_DRM_VERSION>");
            xml.push_str(&escape_xml(custom_data));
            xml.push_str("</CUSTOMATTRIBUTES>");
        }

        // Decryptor setup
        if let Some(ref decryptor_setup) = self.decryptor_setup {
            xml.push_str("<DECRYPTORSETUP>");
            xml.push_str(decryptor_setup);
            xml.push_str("</DECRYPTORSETUP>");
        }

        // Protection info with key records
        xml.push_str("<PROTECTINFO>");

        for key_record in &self.key_records {
            xml.push_str("<KID ALGID=\"");
            xml.push_str(key_record.algorithm.as_str());
            xml.push_str("\" VALUE=\"");
            // PlayReady uses base64 of little-endian GUID
            xml.push_str(&BASE64.encode(key_record.key_id.to_little_endian_bytes()));
            xml.push('"');

            if let Some(ref checksum) = key_record.checksum {
                xml.push_str(" CHECKSUM=\"");
                xml.push_str(&BASE64.encode(checksum));
                xml.push('"');
            }

            xml.push_str("/>");
        }

        xml.push_str("</PROTECTINFO>");
        xml.push_str("</DATA>");
        xml.push_str("</WRMHEADER>");

        xml
    }

    /// Generate the PlayReady Header Object (PRO) binary format.
    ///
    /// The PRO contains:
    /// - Length (4 bytes, little-endian)
    /// - Record count (2 bytes, little-endian)
    /// - Records (each with type, length, and value)
    pub fn to_pro(&self) -> Result<Vec<u8>> {
        let xml = self.to_xml();
        let xml_utf16: Vec<u16> = xml.encode_utf16().collect();
        let xml_bytes: Vec<u8> = xml_utf16
            .iter()
            .flat_map(|&c| c.to_le_bytes())
            .collect();

        if xml_bytes.len() > MAX_HEADER_SIZE {
            return Err(PsshError::DataTooLarge {
                size: xml_bytes.len(),
                max: MAX_HEADER_SIZE,
            }
            .into());
        }

        // Record: Type 1 (Rights Management Header)
        let record_type: u16 = 1;
        let record_length = xml_bytes.len() as u16;

        // PRO structure
        let pro_length = 4 + 2 + 2 + 2 + xml_bytes.len();
        let mut pro = Vec::with_capacity(pro_length);

        // Total length (little-endian)
        pro.extend_from_slice(&(pro_length as u32).to_le_bytes());

        // Number of records
        pro.extend_from_slice(&1u16.to_le_bytes());

        // Record type
        pro.extend_from_slice(&record_type.to_le_bytes());

        // Record length
        pro.extend_from_slice(&record_length.to_le_bytes());

        // Record value (UTF-16LE XML)
        pro.extend_from_slice(&xml_bytes);

        Ok(pro)
    }

    /// Parse from WRMHEADER XML.
    pub fn from_xml(xml: &str) -> Result<Self> {
        // Simplified XML parsing - in production, use a proper XML parser
        let mut header = Self {
            version: extract_attribute(xml, "version").unwrap_or_else(|| "4.0.0.0".to_string()),
            key_records: Vec::new(),
            la_url: extract_element(xml, "LA_URL"),
            lui_url: extract_element(xml, "LUI_URL"),
            ds_id: extract_element(xml, "DS_ID"),
            custom_data: None,
            decryptor_setup: extract_element(xml, "DECRYPTORSETUP"),
        };

        // Extract key records
        let mut pos = 0;
        while let Some(kid_start) = xml[pos..].find("<KID ") {
            let start = pos + kid_start;
            if let Some(end) = xml[start..].find("/>") {
                let kid_elem = &xml[start..start + end + 2];

                if let (Some(algid), Some(value)) = (
                    extract_attribute(kid_elem, "ALGID"),
                    extract_attribute(kid_elem, "VALUE"),
                ) {
                    let algorithm = match algid.as_str() {
                        "AESCTR" => PlayReadyAlgorithm::AesCtr,
                        "AESCBC" => PlayReadyAlgorithm::AesCbc,
                        "COCKTAIL" => PlayReadyAlgorithm::Cocktail,
                        _ => PlayReadyAlgorithm::AesCtr,
                    };

                    // Decode base64 key ID (little-endian GUID)
                    if let Ok(bytes) = BASE64.decode(&value) {
                        if bytes.len() == 16 {
                            // Convert from little-endian GUID to KeyId
                            let le_bytes: [u8; 16] = bytes.try_into().unwrap();
                            let be_bytes = [
                                le_bytes[3], le_bytes[2], le_bytes[1], le_bytes[0],
                                le_bytes[5], le_bytes[4],
                                le_bytes[7], le_bytes[6],
                                le_bytes[8], le_bytes[9], le_bytes[10], le_bytes[11],
                                le_bytes[12], le_bytes[13], le_bytes[14], le_bytes[15],
                            ];
                            let key_id = KeyId::from_bytes(be_bytes);

                            let checksum = extract_attribute(kid_elem, "CHECKSUM")
                                .and_then(|c| BASE64.decode(&c).ok());

                            header.key_records.push(KeyRecord {
                                key_id,
                                algorithm,
                                checksum,
                            });
                        }
                    }
                }

                pos = start + end + 2;
            } else {
                break;
            }
        }

        Ok(header)
    }

    /// Parse from PRO binary format.
    pub fn from_pro(data: &[u8]) -> Result<Self> {
        if data.len() < 10 {
            return Err(PsshError::InvalidFormat("PRO too small".into()).into());
        }

        let pro_length = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        if pro_length > data.len() {
            return Err(PsshError::InvalidFormat("PRO length mismatch".into()).into());
        }

        let record_count = u16::from_le_bytes(data[4..6].try_into().unwrap());

        let mut pos = 6;
        for _ in 0..record_count {
            if pos + 4 > data.len() {
                break;
            }

            let record_type = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            let record_length = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            if record_type == 1 && pos + record_length <= data.len() {
                // Rights Management Header (UTF-16LE XML)
                let xml_bytes = &data[pos..pos + record_length];
                let xml_utf16: Vec<u16> = xml_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let xml = String::from_utf16_lossy(&xml_utf16);
                return Self::from_xml(&xml);
            }

            pos += record_length;
        }

        Err(PsshError::InvalidFormat("No WRMHEADER record found".into()).into())
    }
}

/// PlayReady PSSH box.
#[derive(Clone, Debug)]
pub struct PlayReadyPssh {
    /// PSSH version.
    pub version: u8,
    /// PSSH flags.
    pub flags: u32,
    /// Key IDs (for version 1).
    pub key_ids: Vec<KeyId>,
    /// PlayReady Header Object data.
    pub header: PlayReadyHeader,
}

impl PlayReadyPssh {
    /// Create a new PlayReady PSSH box.
    pub fn new(header: PlayReadyHeader) -> Self {
        Self {
            version: 0,
            flags: 0,
            key_ids: Vec::new(),
            header,
        }
    }

    /// Create version 1 PSSH with key IDs in header.
    pub fn version1(header: PlayReadyHeader) -> Self {
        let key_ids: Vec<KeyId> = header.key_records.iter().map(|r| r.key_id).collect();
        Self {
            version: 1,
            flags: 0,
            key_ids,
            header,
        }
    }

    /// Create from a single key ID.
    pub fn from_key_id(key_id: KeyId) -> Self {
        Self::new(PlayReadyHeader::new(key_id))
    }

    /// Serialize to a complete PSSH box.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let pro_data = self.header.to_pro()?;
        let mut box_data = Vec::new();

        // Version and flags
        box_data.push(self.version);
        box_data.extend_from_slice(&self.flags.to_be_bytes()[1..]);

        // System ID
        box_data.extend_from_slice(PLAYREADY_SYSTEM_ID.as_bytes());

        // Version 1: Key IDs
        if self.version >= 1 {
            box_data.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for key_id in &self.key_ids {
                box_data.extend_from_slice(key_id.as_bytes());
            }
        }

        // Data size and data
        box_data.extend_from_slice(&(pro_data.len() as u32).to_be_bytes());
        box_data.extend_from_slice(&pro_data);

        // Create full box
        let box_size = 8 + box_data.len();
        let mut full_box = Vec::with_capacity(box_size);
        full_box.extend_from_slice(&(box_size as u32).to_be_bytes());
        full_box.extend_from_slice(b"pssh");
        full_box.extend_from_slice(&box_data);

        Ok(full_box)
    }

    /// Serialize to base64.
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(BASE64.encode(&bytes))
    }

    /// Parse from PSSH box bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(PsshError::InvalidFormat("PSSH box too small".into()).into());
        }

        let mut pos = 0;

        // Box size and type
        let _box_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        if &data[pos..pos + 4] != b"pssh" {
            return Err(PsshError::InvalidFormat("Not a PSSH box".into()).into());
        }
        pos += 4;

        // Version and flags
        let version = data[pos];
        let flags = u32::from_be_bytes([0, data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // System ID
        let system_id = Uuid::from_bytes(data[pos..pos + 16].try_into().unwrap());
        if system_id != PLAYREADY_SYSTEM_ID {
            return Err(PsshError::UnknownSystemId(system_id.to_string()).into());
        }
        pos += 16;

        // Version 1: Key IDs
        let mut key_ids = Vec::new();
        if version >= 1 {
            let kid_count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            for _ in 0..kid_count {
                if pos + 16 <= data.len() {
                    key_ids.push(KeyId::from_slice(&data[pos..pos + 16])?);
                    pos += 16;
                }
            }
        }

        // Data size and data
        let data_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + data_size > data.len() {
            return Err(PsshError::InvalidFormat("Data truncated".into()).into());
        }

        let header = PlayReadyHeader::from_pro(&data[pos..pos + data_size])?;

        Ok(Self {
            version,
            flags,
            key_ids,
            header,
        })
    }

    /// Parse from base64.
    pub fn from_base64(encoded: &str) -> Result<Self> {
        let bytes = BASE64
            .decode(encoded)
            .map_err(|e| PsshError::InvalidFormat(e.to_string()))?;
        Self::from_bytes(&bytes)
    }
}

/// Escape special XML characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Extract an attribute value from an XML element.
fn extract_attribute(xml: &str, name: &str) -> Option<String> {
    let pattern = format!("{}=\"", name);
    if let Some(start) = xml.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = xml[value_start..].find('"') {
            return Some(xml[value_start..value_start + end].to_string());
        }
    }
    None
}

/// Extract element content from XML.
fn extract_element(xml: &str, name: &str) -> Option<String> {
    let start_tag = format!("<{}>", name);
    let end_tag = format!("</{}>", name);

    if let Some(start) = xml.find(&start_tag) {
        let content_start = start + start_tag.len();
        if let Some(end) = xml[content_start..].find(&end_tag) {
            return Some(xml[content_start..content_start + end].to_string());
        }
    }
    None
}

/// Generate a PlayReady PSSH for content configuration.
pub fn generate_pssh(
    key_ids: &[KeyId],
    la_url: Option<&str>,
) -> Result<PlayReadyPssh> {
    if key_ids.is_empty() {
        return Err(PsshError::MissingField("key_ids".into()).into());
    }

    let key_records: Vec<KeyRecord> = key_ids
        .iter()
        .map(|&kid| KeyRecord::new(kid, PlayReadyAlgorithm::AesCtr))
        .collect();

    let mut header = PlayReadyHeader::with_keys(key_records);

    if let Some(url) = la_url {
        header = header.with_la_url(url);
    }

    Ok(PlayReadyPssh::new(header))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_id() -> KeyId {
        KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap()
    }

    #[test]
    fn test_playready_system_id() {
        assert_eq!(
            PLAYREADY_SYSTEM_ID.to_string(),
            "9a04f079-9840-4286-ab92-e65be0885f95"
        );
    }

    #[test]
    fn test_playready_header_xml() {
        let key_id = test_key_id();
        let header = PlayReadyHeader::new(key_id)
            .with_la_url("https://license.example.com/");

        let xml = header.to_xml();

        assert!(xml.contains("WRMHEADER"));
        assert!(xml.contains("version=\"4.0.0.0\""));
        assert!(xml.contains("LA_URL"));
        assert!(xml.contains("ALGID=\"AESCTR\""));
    }

    #[test]
    fn test_playready_header_pro_roundtrip() {
        let key_id = test_key_id();
        let header = PlayReadyHeader::new(key_id)
            .with_la_url("https://license.example.com/");

        let pro = header.to_pro().unwrap();
        let parsed = PlayReadyHeader::from_pro(&pro).unwrap();

        assert_eq!(parsed.key_records.len(), 1);
        assert_eq!(parsed.la_url, Some("https://license.example.com/".into()));
    }

    #[test]
    fn test_playready_pssh_roundtrip() {
        let key_id = test_key_id();
        let header = PlayReadyHeader::new(key_id);
        let pssh = PlayReadyPssh::new(header);

        let bytes = pssh.to_bytes().unwrap();
        let parsed = PlayReadyPssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.header.key_records.len(), 1);
    }

    #[test]
    fn test_playready_pssh_version1() {
        let key_id = test_key_id();
        let header = PlayReadyHeader::new(key_id);
        let pssh = PlayReadyPssh::version1(header);

        assert_eq!(pssh.version, 1);
        assert_eq!(pssh.key_ids.len(), 1);

        let bytes = pssh.to_bytes().unwrap();
        let parsed = PlayReadyPssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.key_ids.len(), 1);
    }

    #[test]
    fn test_playready_pssh_base64() {
        let key_id = test_key_id();
        let pssh = PlayReadyPssh::from_key_id(key_id);

        let base64 = pssh.to_base64().unwrap();
        let parsed = PlayReadyPssh::from_base64(&base64).unwrap();

        assert_eq!(parsed.header.key_records.len(), 1);
    }

    #[test]
    fn test_generate_pssh() {
        let key_id = test_key_id();
        let pssh = generate_pssh(&[key_id], Some("https://license.example.com/")).unwrap();

        assert_eq!(pssh.header.key_records.len(), 1);
        assert_eq!(
            pssh.header.la_url,
            Some("https://license.example.com/".into())
        );
    }

    #[test]
    fn test_generate_pssh_multiple_keys() {
        let key1 = KeyId::parse("11111111-1111-1111-1111-111111111111").unwrap();
        let key2 = KeyId::parse("22222222-2222-2222-2222-222222222222").unwrap();

        let pssh = generate_pssh(&[key1, key2], None).unwrap();

        assert_eq!(pssh.header.key_records.len(), 2);
    }

    #[test]
    fn test_key_id_little_endian() {
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();

        let le_bytes = key_id.to_little_endian_bytes();
        let be_bytes = key_id.to_big_endian_bytes();

        // First 4 bytes should be reversed
        assert_eq!(le_bytes[0], be_bytes[3]);
        assert_eq!(le_bytes[1], be_bytes[2]);
        assert_eq!(le_bytes[2], be_bytes[1]);
        assert_eq!(le_bytes[3], be_bytes[0]);
    }

    #[test]
    fn test_xml_escaping() {
        let escaped = escape_xml("Test <>&\"' data");
        assert_eq!(escaped, "Test &lt;&gt;&amp;&quot;&apos; data");
    }

    #[test]
    fn test_algorithm_str() {
        assert_eq!(PlayReadyAlgorithm::AesCtr.as_str(), "AESCTR");
        assert_eq!(PlayReadyAlgorithm::AesCbc.as_str(), "AESCBC");
        assert_eq!(PlayReadyAlgorithm::Cocktail.as_str(), "COCKTAIL");
    }

    #[test]
    fn test_pssh_box_structure() {
        let key_id = test_key_id();
        let pssh = PlayReadyPssh::from_key_id(key_id);
        let bytes = pssh.to_bytes().unwrap();

        // Check box header
        let size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        assert_eq!(size, bytes.len());
        assert_eq!(&bytes[4..8], b"pssh");

        // Check system ID
        assert_eq!(&bytes[12..28], PLAYREADY_SYSTEM_ID.as_bytes());
    }
}
