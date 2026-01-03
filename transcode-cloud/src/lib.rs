//! Cloud storage integration for transcode
//!
//! This crate provides S3, GCS, and Azure Blob storage integration.
//!
//! # Features
//!
//! - `s3` - Amazon S3 support
//! - `gcs` - Google Cloud Storage support
//! - `azure` - Azure Blob Storage support
//!
//! # Example
//!
//! ```ignore
//! use transcode_cloud::{CloudClient, CloudUrl};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = CloudClient::new().await?;
//!
//!     // Parse cloud URL (supports s3://, gs://, az://)
//!     let url = CloudUrl::parse("s3://bucket/path/video.mp4")?;
//!
//!     // Download file
//!     client.download_file(&url, "local.mp4".as_ref()).await?;
//!
//!     Ok(())
//! }
//! ```

use std::path::Path;

mod error;
mod s3;
mod gcs;
#[cfg(feature = "azure")]
mod azure;

pub use error::*;
pub use s3::*;
pub use gcs::*;
#[cfg(feature = "azure")]
pub use azure::*;

/// Result type for cloud operations
pub type Result<T> = std::result::Result<T, CloudError>;

/// Cloud storage provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudProvider {
    /// Amazon S3
    S3,
    /// Google Cloud Storage
    Gcs,
    /// Azure Blob Storage
    Azure,
}

/// Cloud storage URL
#[derive(Debug, Clone)]
pub struct CloudUrl {
    /// Provider
    pub provider: CloudProvider,
    /// Bucket name
    pub bucket: String,
    /// Object key/path
    pub key: String,
    /// Region (optional)
    pub region: Option<String>,
}

impl CloudUrl {
    /// Parse a cloud URL
    pub fn parse(url: &str) -> Result<Self> {
        if let Some(rest) = url.strip_prefix("s3://") {
            let (bucket, key) = rest.split_once('/').ok_or_else(|| {
                CloudError::InvalidUrl("missing object key".into())
            })?;
            Ok(Self {
                provider: CloudProvider::S3,
                bucket: bucket.to_string(),
                key: key.to_string(),
                region: None,
            })
        } else if let Some(rest) = url.strip_prefix("gs://") {
            let (bucket, key) = rest.split_once('/').ok_or_else(|| {
                CloudError::InvalidUrl("missing object key".into())
            })?;
            Ok(Self {
                provider: CloudProvider::Gcs,
                bucket: bucket.to_string(),
                key: key.to_string(),
                region: None,
            })
        } else if url.starts_with("az://") || url.starts_with("azure://") {
            let rest = url.strip_prefix("az://").or_else(|| url.strip_prefix("azure://")).unwrap();
            let (bucket, key) = rest.split_once('/').ok_or_else(|| {
                CloudError::InvalidUrl("missing object key".into())
            })?;
            Ok(Self {
                provider: CloudProvider::Azure,
                bucket: bucket.to_string(),
                key: key.to_string(),
                region: None,
            })
        } else {
            Err(CloudError::InvalidUrl(format!("unknown scheme: {}", url)))
        }
    }

    /// Convert to string URL
    pub fn to_url(&self) -> String {
        let scheme = match self.provider {
            CloudProvider::S3 => "s3",
            CloudProvider::Gcs => "gs",
            CloudProvider::Azure => "az",
        };
        format!("{}://{}/{}", scheme, self.bucket, self.key)
    }
}

/// Upload options
#[derive(Debug, Clone, Default)]
pub struct UploadOptions {
    /// Content type
    pub content_type: Option<String>,
    /// Storage class
    pub storage_class: Option<String>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Multipart threshold (bytes)
    pub multipart_threshold: Option<u64>,
    /// Part size for multipart (bytes)
    pub part_size: Option<u64>,
}

/// Download options
#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    /// Range start (bytes)
    pub range_start: Option<u64>,
    /// Range end (bytes)
    pub range_end: Option<u64>,
}

/// Object metadata
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object key
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Content type
    pub content_type: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<u64>,
    /// ETag
    pub etag: Option<String>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Cloud storage client trait
#[async_trait::async_trait]
pub trait CloudStorage: Send + Sync {
    /// Upload data to cloud
    async fn upload(&self, url: &CloudUrl, data: bytes::Bytes, options: UploadOptions) -> Result<()>;

    /// Download data from cloud
    async fn download(&self, url: &CloudUrl, options: DownloadOptions) -> Result<bytes::Bytes>;

    /// Get object metadata
    async fn head(&self, url: &CloudUrl) -> Result<ObjectMetadata>;

    /// Delete object
    async fn delete(&self, url: &CloudUrl) -> Result<()>;

    /// List objects with prefix
    async fn list(&self, url: &CloudUrl, prefix: Option<&str>) -> Result<Vec<ObjectMetadata>>;

    /// Check if object exists
    async fn exists(&self, url: &CloudUrl) -> Result<bool> {
        match self.head(url).await {
            Ok(_) => Ok(true),
            Err(CloudError::NotFound) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Unified cloud client
pub struct CloudClient {
    #[cfg(feature = "s3")]
    s3_client: Option<S3Client>,
    #[cfg(feature = "gcs")]
    gcs_client: Option<GcsClient>,
    #[cfg(feature = "azure")]
    azure_client: Option<AzureClient>,
}

impl CloudClient {
    /// Create a new cloud client
    pub async fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "s3")]
            s3_client: Some(S3Client::new().await?),
            #[cfg(feature = "gcs")]
            gcs_client: Some(GcsClient::new().await?),
            #[cfg(feature = "azure")]
            azure_client: Some(AzureClient::new().await?),
        })
    }

    /// Upload file to cloud
    pub async fn upload_file(&self, url: &CloudUrl, path: &Path) -> Result<()> {
        let data = tokio::fs::read(path).await?;
        self.upload(url, bytes::Bytes::from(data), UploadOptions::default()).await
    }

    /// Download file from cloud
    pub async fn download_file(&self, url: &CloudUrl, path: &Path) -> Result<()> {
        let data = self.download(url, DownloadOptions::default()).await?;
        tokio::fs::write(path, data).await?;
        Ok(())
    }

    /// Upload data
    #[allow(unused_variables)]
    pub async fn upload(&self, url: &CloudUrl, data: bytes::Bytes, options: UploadOptions) -> Result<()> {
        match url.provider {
            #[cfg(feature = "s3")]
            CloudProvider::S3 => {
                if let Some(ref client) = self.s3_client {
                    return client.upload(url, data, options).await;
                }
            }
            #[cfg(feature = "gcs")]
            CloudProvider::Gcs => {
                if let Some(ref client) = self.gcs_client {
                    return client.upload(url, data, options).await;
                }
            }
            #[cfg(feature = "azure")]
            CloudProvider::Azure => {
                if let Some(ref client) = self.azure_client {
                    return client.upload(url, data, options).await;
                }
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
        Err(CloudError::UnsupportedProvider(format!("{:?}", url.provider)))
    }

    /// Download data
    #[allow(unused_variables)]
    pub async fn download(&self, url: &CloudUrl, options: DownloadOptions) -> Result<bytes::Bytes> {
        match url.provider {
            #[cfg(feature = "s3")]
            CloudProvider::S3 => {
                if let Some(ref client) = self.s3_client {
                    return client.download(url, options).await;
                }
            }
            #[cfg(feature = "gcs")]
            CloudProvider::Gcs => {
                if let Some(ref client) = self.gcs_client {
                    return client.download(url, options).await;
                }
            }
            #[cfg(feature = "azure")]
            CloudProvider::Azure => {
                if let Some(ref client) = self.azure_client {
                    return client.download(url, options).await;
                }
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
        Err(CloudError::UnsupportedProvider(format!("{:?}", url.provider)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let url = CloudUrl::parse("s3://my-bucket/path/to/file.mp4").unwrap();
        assert_eq!(url.provider, CloudProvider::S3);
        assert_eq!(url.bucket, "my-bucket");
        assert_eq!(url.key, "path/to/file.mp4");
    }

    #[test]
    fn test_parse_gcs_url() {
        let url = CloudUrl::parse("gs://my-bucket/video.mp4").unwrap();
        assert_eq!(url.provider, CloudProvider::Gcs);
        assert_eq!(url.bucket, "my-bucket");
        assert_eq!(url.key, "video.mp4");
    }

    #[test]
    fn test_url_roundtrip() {
        let original = "s3://bucket/key/file.mp4";
        let url = CloudUrl::parse(original).unwrap();
        assert_eq!(url.to_url(), original);
    }
}
