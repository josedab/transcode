//! Google Cloud Storage integration

use crate::{CloudError, CloudUrl, DownloadOptions, ObjectMetadata, Result, UploadOptions};

/// GCS client configuration
#[derive(Debug, Clone, Default)]
pub struct GcsConfig {
    /// Project ID
    pub project_id: Option<String>,
    /// Custom endpoint
    pub endpoint: Option<String>,
}

/// GCS client
pub struct GcsClient {
    _config: GcsConfig,
}

impl GcsClient {
    /// Create a new GCS client
    pub async fn new() -> Result<Self> {
        Ok(Self {
            _config: GcsConfig::default(),
        })
    }

    /// Create with custom config
    pub async fn with_config(config: GcsConfig) -> Result<Self> {
        Ok(Self { _config: config })
    }

    /// Upload data to GCS
    #[cfg(feature = "gcs")]
    pub async fn upload(&self, url: &CloudUrl, data: bytes::Bytes, _options: UploadOptions) -> Result<()> {
        use cloud_storage::Client;

        let client = Client::default();
        client
            .object()
            .create(&url.bucket, data.to_vec(), &url.key, "application/octet-stream")
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(())
    }

    #[cfg(not(feature = "gcs"))]
    pub async fn upload(&self, _url: &CloudUrl, _data: bytes::Bytes, _options: UploadOptions) -> Result<()> {
        Err(CloudError::UnsupportedProvider("GCS feature not enabled".into()))
    }

    /// Download data from GCS
    #[cfg(feature = "gcs")]
    pub async fn download(&self, url: &CloudUrl, _options: DownloadOptions) -> Result<bytes::Bytes> {
        use cloud_storage::Client;

        let client = Client::default();
        let data = client
            .object()
            .download(&url.bucket, &url.key)
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(bytes::Bytes::from(data))
    }

    #[cfg(not(feature = "gcs"))]
    pub async fn download(&self, _url: &CloudUrl, _options: DownloadOptions) -> Result<bytes::Bytes> {
        Err(CloudError::UnsupportedProvider("GCS feature not enabled".into()))
    }

    /// Get object metadata
    #[cfg(feature = "gcs")]
    pub async fn head(&self, url: &CloudUrl) -> Result<ObjectMetadata> {
        use cloud_storage::Client;

        let client = Client::default();
        let obj = client
            .object()
            .read(&url.bucket, &url.key)
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;

        Ok(ObjectMetadata {
            key: url.key.clone(),
            size: obj.size,
            content_type: Some(obj.content_type),
            last_modified: None,
            etag: obj.etag,
            metadata: obj.metadata.unwrap_or_default(),
        })
    }

    #[cfg(not(feature = "gcs"))]
    pub async fn head(&self, _url: &CloudUrl) -> Result<ObjectMetadata> {
        Err(CloudError::UnsupportedProvider("GCS feature not enabled".into()))
    }

    /// Delete object
    #[cfg(feature = "gcs")]
    pub async fn delete(&self, url: &CloudUrl) -> Result<()> {
        use cloud_storage::Client;

        let client = Client::default();
        client
            .object()
            .delete(&url.bucket, &url.key)
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(())
    }

    #[cfg(not(feature = "gcs"))]
    pub async fn delete(&self, _url: &CloudUrl) -> Result<()> {
        Err(CloudError::UnsupportedProvider("GCS feature not enabled".into()))
    }

    /// List objects
    #[cfg(feature = "gcs")]
    pub async fn list(&self, url: &CloudUrl, prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        use cloud_storage::Client;

        let client = Client::default();
        let list_request = cloud_storage::ListRequest {
            prefix: prefix.map(String::from),
            ..Default::default()
        };

        let objects = client
            .object()
            .list(&url.bucket, list_request)
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;

        Ok(objects.into_iter().map(|obj| ObjectMetadata {
            key: obj.name,
            size: obj.size,
            content_type: Some(obj.content_type),
            last_modified: None,
            etag: obj.etag,
            metadata: obj.metadata.unwrap_or_default(),
        }).collect())
    }

    #[cfg(not(feature = "gcs"))]
    pub async fn list(&self, _url: &CloudUrl, _prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        Err(CloudError::UnsupportedProvider("GCS feature not enabled".into()))
    }
}
